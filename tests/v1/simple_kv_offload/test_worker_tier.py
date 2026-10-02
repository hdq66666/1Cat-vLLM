# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker side of the SimpleCPUOffload tier.

Copy backend selection, transfer ordering and accounting, failure handling
and verification, on tiny real CUDA tensors (no model, no engine).
"""

import threading
import time
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from unittest import mock

import msgspec
import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.simple_kv_offload import copy_backend as copy_backend_mod
from vllm.v1.simple_kv_offload import host_mem
from vllm.v1.simple_kv_offload import worker as worker_mod
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend, TransferEvent
from vllm.v1.simple_kv_offload.cuda_mem_ops import unpin_tensor
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.options import SimpleCPUOffloadOptions
from vllm.v1.simple_kv_offload.stats import (
    LOAD_BLOCKS,
    LOAD_BYTES,
    LOAD_DMA_MS_MAX,
    LOAD_DMA_MS_SUM,
    LOAD_EVENTS,
    LOAD_READY_MS_MAX,
    LOAD_READY_MS_SUM,
    PENDING_COMPUTE_AT_LOAD,
    STORE_BLOCKS,
    STORE_BYTES,
    STORE_DMA_MS_MAX,
    STORE_EVENTS,
    VERIFY_CHECKED,
    VERIFY_MISMATCH,
    SimpleCPUOffloadStats,
)
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker, _Launch

if not current_platform.is_cuda():
    pytest.skip("Requires CUDA", allow_module_level=True)

NUM_GPU_BLOCKS = 8
NUM_CPU_BLOCKS = 6
# attn.0 is K/V-outermost: two regions of 16 x 8 fp16 = 256 B per block;
# mamba.0 is one region of 64 fp32 = 256 B per block.
BYTES_PER_BLOCK = 3 * 256
# Several hundred ms of GPU spin: far longer than any test copy.
SLEEP_CYCLES = 500_000_000


@pytest.fixture
def make_worker() -> Iterator[Callable[..., SimpleCPUOffloadWorker]]:
    workers: list[SimpleCPUOffloadWorker] = []

    def factory(
        extra_kv_tensors: tuple[KVCacheTensor, ...] = (), **options
    ) -> SimpleCPUOffloadWorker:
        attn = torch.zeros(2, NUM_GPU_BLOCKS, 16, 8, dtype=torch.float16, device="cuda")
        mamba = torch.zeros(NUM_GPU_BLOCKS, 64, dtype=torch.float32, device="cuda")
        kv_cache_config = KVCacheConfig(
            num_blocks=NUM_GPU_BLOCKS,
            kv_cache_tensors=[
                KVCacheTensor(size=attn.nbytes, shared_by=["attn.0"]),
                KVCacheTensor(size=mamba.nbytes, shared_by=["mamba.0"]),
                *extra_kv_tensors,
            ],
            kv_cache_groups=[],
        )
        options.setdefault("min_host_mem_available_gib", 0.0)
        worker = SimpleCPUOffloadWorker(
            SimpleNamespace(),  # type: ignore[arg-type]
            kv_cache_config,
            cpu_capacity_bytes=NUM_CPU_BLOCKS * BYTES_PER_BLOCK,
            options=SimpleCPUOffloadOptions(**options),
        )
        workers.append(worker)
        worker.register_kv_caches({"attn.0": attn, "mamba.0": [mamba]})
        return worker

    yield factory

    for worker in workers:
        worker._backend.shutdown()
    torch.accelerator.synchronize()
    for worker in workers:
        # Freed host memory that is still registered cannot be registered
        # again when a later allocation reuses its address.
        if worker.cpu_kv_caches and is_pin_memory_available():
            for tensor in worker.cpu_kv_caches.values():
                unpin_tensor(tensor)


def _mock_worker(**options) -> SimpleCPUOffloadWorker:
    """A worker with mocked streams and backend (nothing registered)."""
    kv_cache_config = SimpleNamespace(num_blocks=NUM_GPU_BLOCKS)
    worker = SimpleCPUOffloadWorker(
        SimpleNamespace(),  # type: ignore[arg-type]
        kv_cache_config,  # type: ignore[arg-type]
        cpu_capacity_bytes=0,
        options=SimpleCPUOffloadOptions(**options),
    )
    worker.num_cpu_blocks = NUM_CPU_BLOCKS
    worker.load_stream = mock.Mock()
    worker.store_stream = mock.Mock()
    worker._backend = mock.Mock()
    return worker


def _step(
    worker: SimpleCPUOffloadWorker, metadata: SimpleCPUOffloadMetadata
) -> tuple[set[str], set[int]]:
    """One step: bind, get_finished, worker meta. Returns (recving, stored)."""
    worker.bind_connector_metadata(metadata)
    try:
        _, recving = worker.get_finished(set())
        meta = worker.build_connector_worker_meta()
    finally:
        worker.clear_connector_metadata()
    return recving or set(), set(meta.completed_store_events) if meta else set()


def _drive(
    worker: SimpleCPUOffloadWorker,
    metadata: SimpleCPUOffloadMetadata,
    recv: tuple[str, ...] = (),
    stores: tuple[int, ...] = (),
    timeout_s: float = 10.0,
) -> None:
    """Run steps (the first carries ``metadata``, later ones are empty) until
    the given requests finished loading and the given store events completed.
    """
    got_recv: set[str] = set()
    got_stores: set[int] = set()
    deadline = time.monotonic() + timeout_s
    while True:
        r, s = _step(worker, metadata)
        got_recv |= r
        got_stores |= s
        if set(recv) <= got_recv and set(stores) <= got_stores:
            return
        assert time.monotonic() < deadline, (got_recv, got_stores)
        metadata = SimpleCPUOffloadMetadata(
            load_event_to_reqs=metadata.load_event_to_reqs
        )
        time.sleep(0.001)


def _block(worker: SimpleCPUOffloadWorker, gpu: bool, block_id: int) -> list:
    caches = worker.gpu_kv_caches if gpu else worker.cpu_kv_caches
    assert caches is not None
    return [t[block_id].cpu() for t in caches.values()]


def _same(a: list, b: list) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def _fill_random(worker: SimpleCPUOffloadWorker, gpu: bool) -> None:
    caches = worker.gpu_kv_caches if gpu else worker.cpu_kv_caches
    assert caches is not None
    for t in caches.values():
        t.copy_(torch.randint(-128, 128, t.shape, dtype=torch.int8))


# --- Copy backend selection (P8) ---


@pytest.mark.parametrize("mode", ["batch", "memcpy"])
def test_self_test_passes(mode: str) -> None:
    load_stream, store_stream = torch.cuda.Stream(), torch.cuda.Stream()
    failure = DmaCopyBackend()._self_test(
        mode, torch.device("cuda"), load_stream, store_stream
    )
    assert failure is None


@pytest.mark.parametrize(
    "mode,copy_fn", [("batch", "copy_blocks"), ("memcpy", "swap_copy_blocks")]
)
def test_self_test_detects_misplaced_blocks(
    monkeypatch: pytest.MonkeyPatch, mode: str, copy_fn: str
) -> None:
    real_copy = getattr(copy_backend_mod, copy_fn)
    monkeypatch.setattr(
        copy_backend_mod,
        copy_fn,
        lambda src, dst, params: real_copy(src, dst[::-1], params),
    )
    load_stream, store_stream = torch.cuda.Stream(), torch.cuda.Stream()
    failure = DmaCopyBackend()._self_test(
        mode, torch.device("cuda"), load_stream, store_stream
    )
    assert failure is not None and "mismatch" in failure


@pytest.mark.parametrize(
    "requested,batch_available,results,expected",
    [
        ("auto", True, {"batch": None}, "batch"),
        ("auto", False, {"memcpy": None}, "memcpy"),
        ("auto", True, {"batch": "mismatch", "memcpy": None}, "memcpy"),
        ("auto", True, {"batch": "mismatch", "memcpy": "mismatch"}, None),
        ("batch", True, {"batch": None}, "batch"),
        ("batch", False, {}, None),
        ("batch", True, {"batch": "mismatch"}, None),
        ("memcpy", True, {"memcpy": None}, "memcpy"),
        ("memcpy", True, {"memcpy": "mismatch"}, None),
    ],
)
def test_copy_mode_selection(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    batch_available: bool,
    results: dict[str, str | None],
    expected: str | None,
) -> None:
    """Auto falls back to memcpy; a forced mode is never swapped silently."""
    monkeypatch.setattr(
        copy_backend_mod,
        "get_batch_memcpy_fn",
        lambda: object() if batch_available else None,
    )
    tested: list[str] = []

    def fake_self_test(self, mode, device, load_stream, store_stream):
        tested.append(mode)
        return results[mode]

    monkeypatch.setattr(DmaCopyBackend, "_self_test", fake_self_test)
    device = load_stream = store_stream = None
    backend = DmaCopyBackend()
    if expected is None:
        with pytest.raises(RuntimeError):
            backend._select_mode(requested, device, load_stream, store_stream)
    else:
        mode = backend._select_mode(requested, device, load_stream, store_stream)
        assert mode == expected
    # Only the modes that could be picked are tested, each at most once.
    assert tested == list(results)


# --- Real transfers ---


@pytest.mark.parametrize("copy_backend", ["batch", "memcpy"])
def test_round_trip(make_worker, copy_backend: str) -> None:
    worker = make_worker(copy_backend=copy_backend)
    assert worker._backend.mode == copy_backend
    assert worker.num_cpu_blocks == NUM_CPU_BLOCKS
    _fill_random(worker, gpu=True)
    expected = {g: _block(worker, True, g) for g in (1, 4, 7)}

    _drive(
        worker,
        SimpleCPUOffloadMetadata(
            store_event=0,
            store_gpu_blocks=[1, 4, 7],
            store_cpu_blocks=[5, 0, 2],
            num_cpu_blocks=NUM_CPU_BLOCKS,
        ),
        stores=(0,),
    )
    for g, c in ((1, 5), (4, 0), (7, 2)):
        assert _same(_block(worker, False, c), expected[g])

    for t in worker.gpu_kv_caches.values():
        t.zero_()
    _drive(
        worker,
        SimpleCPUOffloadMetadata(
            load_event=0,
            load_cpu_blocks=[5, 0, 2],
            load_gpu_blocks=[3, 6, 0],
            load_event_to_reqs={0: ["req"]},
            num_cpu_blocks=NUM_CPU_BLOCKS,
        ),
        recv=("req",),
    )
    for g_src, g_dst in ((1, 3), (4, 6), (7, 0)):
        assert _same(_block(worker, True, g_dst), expected[g_src])

    stats = worker.get_stats()
    assert stats is not None
    for prefix in ("load", "store"):
        assert stats.data[f"{prefix}_events"] == 1
        assert stats.data[f"{prefix}_blocks"] == 3
        assert stats.data[f"{prefix}_bytes"] == 3 * BYTES_PER_BLOCK
        assert stats.data[f"{prefix}_dma_ms_max"] > 0
        assert (
            stats.data[f"{prefix}_ready_ms_max"] >= stats.data[f"{prefix}_dma_ms_max"]
        )
    assert worker.get_stats() is None


@pytest.mark.parametrize("unordered", [False, True])
def test_load_lands_after_compute_stream_writes(
    monkeypatch: pytest.MonkeyPatch, make_worker, unordered: bool
) -> None:
    """P1: the step zeroes the load destination on the compute stream behind
    a backlog; the load must land after it. The unordered negative control
    lets the zero overwrite the loaded block, which VERIFY reports."""
    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_VERIFY", "all")
    if unordered:
        monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_MODE", "1")
        monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_UNORDERED_LOAD", "1")
    worker = make_worker()
    _fill_random(worker, gpu=False)

    torch.cuda._sleep(SLEEP_CYCLES)
    for t in worker.gpu_kv_caches.values():
        t[5].zero_()
    _drive(
        worker,
        SimpleCPUOffloadMetadata(
            load_event=0,
            load_cpu_blocks=[2],
            load_gpu_blocks=[5],
            load_event_to_reqs={0: ["req"]},
        ),
        recv=("req",),
    )
    torch.accelerator.synchronize()

    stats = worker.get_stats()
    assert stats is not None
    assert stats.data[PENDING_COMPUTE_AT_LOAD] == 1
    assert stats.data[VERIFY_CHECKED] == 1
    landed = _same(_block(worker, True, 5), _block(worker, False, 2))
    if unordered:
        assert not landed
        assert stats.data[VERIFY_MISMATCH] == 1
    else:
        assert landed
        assert VERIFY_MISMATCH not in stats.data


def test_debug_compute_delay_backs_up_the_compute_stream(
    monkeypatch: pytest.MonkeyPatch, make_worker
) -> None:
    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_MODE", "1")
    monkeypatch.setenv(
        "VLLM_SIMPLE_KV_OFFLOAD_DEBUG_COMPUTE_DELAY_CYCLES", str(SLEEP_CYCLES)
    )
    worker = make_worker()
    torch.accelerator.synchronize()

    _drive(
        worker,
        SimpleCPUOffloadMetadata(
            load_event=0,
            load_cpu_blocks=[0],
            load_gpu_blocks=[0],
            load_event_to_reqs={0: ["req"]},
        ),
        recv=("req",),
    )

    stats = worker.get_stats()
    assert stats is not None and stats.data[PENDING_COMPUTE_AT_LOAD] == 1


def _corrupting_copy(
    monkeypatch: pytest.MonkeyPatch, corrupt: Callable[[list[int]], None]
) -> None:
    """Make every copy issue correctly, then corrupt its destination."""
    real_copy = DmaCopyBackend._copy

    def copy(src_blocks, dst_blocks, params):
        real_copy(src_blocks, dst_blocks, params)
        torch.accelerator.synchronize()
        corrupt(dst_blocks)

    monkeypatch.setattr(DmaCopyBackend, "_copy", staticmethod(copy))


@pytest.mark.parametrize("verify,checked", [("sample", 3), ("all", 5)])
def test_verify_detects_a_corrupted_store(
    monkeypatch: pytest.MonkeyPatch, make_worker, verify: str, checked: int
) -> None:
    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_VERIFY", verify)
    worker = make_worker()
    _fill_random(worker, gpu=True)
    # Corrupt the middle pair's CPU copy in the mamba region.
    _corrupting_copy(
        monkeypatch,
        lambda dst: worker.cpu_kv_caches["mamba.0"][dst[2]].add_(1),
    )
    warning = mock.Mock()
    monkeypatch.setattr(worker_mod.logger, "warning", warning)

    _drive(
        worker,
        SimpleCPUOffloadMetadata(
            store_event=3,
            store_gpu_blocks=[0, 1, 2, 3, 4],
            store_cpu_blocks=[1, 2, 3, 4, 5],
        ),
        stores=(3,),
    )

    stats = worker.get_stats()
    assert stats is not None
    assert stats.data[VERIFY_CHECKED] == checked
    assert stats.data[VERIFY_MISMATCH] == 1
    warning.assert_called_once()
    args = warning.call_args.args
    assert args[1:] == ("store", 3, 1, checked, "mamba.0", 2, 3)


def test_verify_detects_a_corrupted_load(
    monkeypatch: pytest.MonkeyPatch, make_worker
) -> None:
    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_VERIFY", "sample")
    worker = make_worker()
    _fill_random(worker, gpu=False)
    _corrupting_copy(
        monkeypatch,
        lambda dst: worker.gpu_kv_caches["attn.0.1"][dst[0]].zero_(),
    )

    _drive(
        worker,
        SimpleCPUOffloadMetadata(
            load_event=0,
            load_cpu_blocks=[4, 5],
            load_gpu_blocks=[6, 7],
            load_event_to_reqs={0: ["req"]},
        ),
        recv=("req",),
    )

    stats = worker.get_stats()
    assert stats is not None
    assert stats.data[VERIFY_CHECKED] == 2
    assert stats.data[VERIFY_MISMATCH] == 1


# --- Failure handling (P9) ---


def test_injected_copy_failure_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, make_worker
) -> None:
    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_MODE", "1")
    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_FAIL_COPY", "1")
    worker = make_worker()

    # Launch without polling: the failure may surface at any later call.
    worker._launch([0], [0], is_store=True, event_idx=0)
    with pytest.raises(RuntimeError, match="copy thread failed") as exc_info:
        worker._backend.wait_drained(10.0)
    assert "TEST_FAIL_COPY" in str(exc_info.value.__cause__)

    # Every later entry point raises instead of waiting for the transfer.
    with pytest.raises(RuntimeError, match="copy thread failed"):
        _step(worker, SimpleCPUOffloadMetadata())
    with pytest.raises(RuntimeError, match="copy thread failed"):
        worker.handle_preemptions(SimpleCPUOffloadMetadata(need_flush=True))
    with pytest.raises(RuntimeError, match="copy thread failed"):
        worker._backend.launch_copy([0], [0], False, 1, worker._load_events)


def test_copy_exception_is_chained(
    monkeypatch: pytest.MonkeyPatch, make_worker
) -> None:
    worker = make_worker()
    boom = ValueError("driver said no")

    def failing_copy(src_blocks, dst_blocks, params):
        raise boom

    monkeypatch.setattr(DmaCopyBackend, "_copy", staticmethod(failing_copy))
    worker._launch([0], [0], is_store=False, event_idx=0)
    deadline = time.monotonic() + 10.0
    while worker._backend._error is None:
        assert time.monotonic() < deadline
        time.sleep(0.001)

    with pytest.raises(RuntimeError, match="copy thread failed") as exc_info:
        _step(worker, SimpleCPUOffloadMetadata())
    assert exc_info.value.__cause__ is boom


def test_watchdog_fails_a_transfer_that_never_completes() -> None:
    worker = _mock_worker(transfer_timeout_s=60.0)
    now = [100.0]
    worker._clock = lambda: now[0]
    load = SimpleCPUOffloadMetadata(
        load_event=0, load_cpu_blocks=[1], load_gpu_blocks=[2]
    )

    _step(worker, load)
    now[0] += 59.0
    _step(worker, SimpleCPUOffloadMetadata())
    now[0] += 2.0
    with pytest.raises(RuntimeError, match="load event 0 .* has not completed"):
        _step(worker, SimpleCPUOffloadMetadata())


def test_watchdog_ignores_an_idle_gap(make_worker) -> None:
    """A transfer that completed while no step ran is not overdue."""
    worker = make_worker(transfer_timeout_s=60.0)
    now = [100.0]
    worker._clock = lambda: now[0]
    store = SimpleCPUOffloadMetadata(
        store_event=0, store_gpu_blocks=[1], store_cpu_blocks=[1]
    )
    worker.bind_connector_metadata(store)
    # Launch only: complete the copy before anything polls it.
    worker._launch([1], [1], is_store=True, event_idx=0)
    worker.clear_connector_metadata()
    worker._backend.wait_drained(10.0)
    torch.accelerator.synchronize()

    now[0] += 3600.0
    assert _step(worker, SimpleCPUOffloadMetadata()) == (set(), {0})


# --- Flush (P9) ---


def _gate_copies(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """Hold every copy in the copy thread until the returned gate opens."""
    gate = threading.Event()
    real_copy = DmaCopyBackend._copy

    def copy(src_blocks, dst_blocks, params):
        assert gate.wait(30.0)
        real_copy(src_blocks, dst_blocks, params)

    monkeypatch.setattr(DmaCopyBackend, "_copy", staticmethod(copy))
    return gate


def test_flush_waits_for_queued_copies(
    monkeypatch: pytest.MonkeyPatch, make_worker
) -> None:
    worker = make_worker()
    gate = _gate_copies(monkeypatch)
    reqs = {0: ["r0"], 1: ["r1"], 2: ["r2"]}
    for i in range(3):
        _step(
            worker,
            SimpleCPUOffloadMetadata(
                load_event=i,
                load_cpu_blocks=[i],
                load_gpu_blocks=[i],
                load_event_to_reqs=reqs,
                store_event=i if i < 2 else -1,
                store_gpu_blocks=[4 + i] if i < 2 else [],
                store_cpu_blocks=[4 + i] if i < 2 else [],
            ),
        )
    # Everything is still queued behind the gate.
    assert not worker._load_events and not worker._store_events

    opener = threading.Timer(0.2, gate.set)
    opener.start()
    worker.handle_preemptions(SimpleCPUOffloadMetadata(need_flush=True))
    opener.join()

    assert gate.is_set()
    assert not worker._load_events and not worker._store_events
    assert not worker._load_launches and not worker._store_launches
    assert (worker._load_hwm, worker._store_hwm) == (2, 1)
    # Nothing was lost: the next step reports every transfer.
    recving, stored = _step(worker, SimpleCPUOffloadMetadata(load_event_to_reqs=reqs))
    assert recving == {"r0", "r1", "r2"}
    assert stored == {0, 1}


def test_flush_fails_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch, make_worker
) -> None:
    worker = make_worker(transfer_timeout_s=1.0)
    gate = _gate_copies(monkeypatch)
    _step(
        worker,
        SimpleCPUOffloadMetadata(
            load_event=0, load_cpu_blocks=[0], load_gpu_blocks=[0]
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="did not issue 1 queued copies"):
            worker.handle_preemptions(SimpleCPUOffloadMetadata(need_flush=True))
    finally:
        gate.set()


def test_flush_fails_when_the_copy_thread_dies(
    monkeypatch: pytest.MonkeyPatch, make_worker
) -> None:
    worker = make_worker()
    gate = threading.Event()

    def failing_copy(src_blocks, dst_blocks, params):
        assert gate.wait(30.0)
        raise ValueError("late failure")

    monkeypatch.setattr(DmaCopyBackend, "_copy", staticmethod(failing_copy))
    _step(
        worker,
        SimpleCPUOffloadMetadata(
            load_event=0, load_cpu_blocks=[0], load_gpu_blocks=[0]
        ),
    )
    opener = threading.Timer(0.2, gate.set)
    opener.start()
    with pytest.raises(RuntimeError, match="copy thread failed"):
        worker.handle_preemptions(SimpleCPUOffloadMetadata(need_flush=True))
    opener.join()


# --- CPU block count and bounds (P6), host memory (P7c) ---


def test_register_requires_every_kv_tensor(make_worker) -> None:
    unregistered = KVCacheTensor(size=4096, shared_by=["not_in_kv_caches"])
    with pytest.raises(RuntimeError, match="registered KV regions cover"):
        make_worker(extra_kv_tensors=(unregistered,))


def test_register_refuses_short_host_memory(
    monkeypatch: pytest.MonkeyPatch, make_worker
) -> None:
    monkeypatch.setattr(host_mem, "read_mem_available_bytes", lambda: 1024**3)
    with pytest.raises(ValueError, match="SimpleCPUOffloadWorker rank 0"):
        make_worker(min_host_mem_available_gib=15.0)


def test_bind_checks_the_scheduler_cpu_block_count() -> None:
    worker = _mock_worker()
    worker.bind_connector_metadata(SimpleCPUOffloadMetadata(num_cpu_blocks=0))
    worker.bind_connector_metadata(
        SimpleCPUOffloadMetadata(num_cpu_blocks=NUM_CPU_BLOCKS)
    )
    with pytest.raises(RuntimeError, match="scheduler=7, worker=6 .rank 0"):
        worker.bind_connector_metadata(SimpleCPUOffloadMetadata(num_cpu_blocks=7))


@pytest.mark.parametrize("is_store", [False, True])
@pytest.mark.parametrize(
    "cpu_blocks,gpu_blocks",
    [
        ([NUM_CPU_BLOCKS], [0]),
        ([-1], [0]),
        ([0], [NUM_GPU_BLOCKS]),
        ([0], [-1]),
        ([0, 1], [0]),
    ],
)
def test_launch_refuses_out_of_range_block_ids(
    is_store: bool, cpu_blocks: list[int], gpu_blocks: list[int]
) -> None:
    worker = _mock_worker()
    if is_store:
        metadata = SimpleCPUOffloadMetadata(
            store_event=0, store_cpu_blocks=cpu_blocks, store_gpu_blocks=gpu_blocks
        )
    else:
        metadata = SimpleCPUOffloadMetadata(
            load_event=0, load_cpu_blocks=cpu_blocks, load_gpu_blocks=gpu_blocks
        )
    with pytest.raises(RuntimeError, match="SimpleCPUOffload transfer"):
        _step(worker, metadata)
    worker._backend.launch_copy.assert_not_called()


# --- Stats (P11) ---


def test_stats_aggregate_over_ranks() -> None:
    """Four TP ranks: bytes and counts add up, latencies keep the maximum."""
    per_rank = []
    for rank in range(4):
        worker = _mock_worker()
        worker._bytes_per_block = BYTES_PER_BLOCK
        now = [10.0]
        worker._clock = lambda now=now: now[0]
        worker._load_launches[0] = _Launch(10.0, [0, 1, 2], [3, 4, 5])
        worker._store_launches[1] = _Launch(10.0, [0], [3])
        now[0] += 0.010 * (rank + 1)
        start = SimpleNamespace(elapsed_time=lambda done, ms=1.0 + rank: ms)
        worker._on_transfer_done(TransferEvent(0, start, None), is_store=False)
        worker._on_transfer_done(TransferEvent(1, start, None), is_store=True)
        stats = worker.get_stats()
        assert stats is not None and worker.get_stats() is None
        # Stats cross the process boundary as their msgpack-encoded data.
        wire = msgspec.msgpack.decode(msgspec.msgpack.encode(stats.data))
        per_rank.append(SimpleCPUOffloadStats(data=wire))

    total = per_rank[0]
    for stats in per_rank[1:]:
        total = total.aggregate(stats)
    assert isinstance(total, SimpleCPUOffloadStats)
    data = total.data
    assert data[LOAD_EVENTS] == 4 and data[STORE_EVENTS] == 4
    assert data[LOAD_BLOCKS] == 12 and data[STORE_BLOCKS] == 4
    assert data[LOAD_BYTES] == 12 * BYTES_PER_BLOCK
    assert data[STORE_BYTES] == 4 * BYTES_PER_BLOCK
    assert data[LOAD_DMA_MS_SUM] == pytest.approx(1 + 2 + 3 + 4)
    assert data[LOAD_DMA_MS_MAX] == data[STORE_DMA_MS_MAX] == 4.0
    assert data[LOAD_READY_MS_SUM] == pytest.approx(10 + 20 + 30 + 40)
    assert data[LOAD_READY_MS_MAX] == pytest.approx(40.0)
    reduced = total.reduce()
    assert reduced["load_events"] == 4
    assert reduced["load_dma_ms_max"] == 4.0
    assert reduced["load_ready_ms_avg"] == pytest.approx(25.0)
