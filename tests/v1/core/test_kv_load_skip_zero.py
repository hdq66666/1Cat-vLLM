# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async KV loads must not race the zeroing of their destination blocks.

Drives the real Scheduler with SimpleCPUOffloadConnector (lazy CPU tier) on a
hybrid layout (full attention + QSA compressor ring + Mamba ``align``), the
layout that needs KV cache zeroing: a request that hits the CPU tier is
admitted with an async load, and the blocks that load overwrites must not be
in that step's zero list (vllm-project/vllm#48481).
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.structured_output import StructuredOutputManager

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

FA_BLOCK = 16
RING_BLOCK = 4
MAMBA_BLOCK = 32
NUM_GPU_BLOCKS = 64
# Three Mamba boundary states plus a partial block: the CPU tier can serve
# 3 * MAMBA_BLOCK tokens of a repeated prompt.
PROMPT_LEN = 3 * MAMBA_BLOCK + 5
CPU_HIT_TOKENS = 3 * MAMBA_BLOCK
FA_GROUP, RING_GROUP = 0, 1

init_none_hash(sha256)


def _make_kv_cache_config() -> KVCacheConfig:
    specs = {
        "fa": FullAttentionSpec(
            block_size=FA_BLOCK, num_kv_heads=1, head_size=16, dtype=torch.float16
        ),
        "ring": CircularBufferSpec(
            block_size=RING_BLOCK,
            num_kv_heads=1,
            head_size=16,
            head_size_v=0,
            dtype=torch.float16,
        ),
        "mamba": MambaSpec(
            block_size=MAMBA_BLOCK,
            shapes=((1, 64),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
        ),
    }
    return KVCacheConfig(
        num_blocks=NUM_GPU_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=spec.page_size_bytes * NUM_GPU_BLOCKS, shared_by=[name])
            for name, spec in specs.items()
        ],
        kv_cache_groups=[
            KVCacheGroupSpec([name], spec) for name, spec in specs.items()
        ],
    )


def _make_scheduler() -> Scheduler:
    kv_cache_config = _make_kv_cache_config()
    gpu_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="facebook/opt-125m",
            trust_remote_code=True,
            dtype="float16",
            seed=42,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=256,
            max_model_len=1024,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(
            block_size=FA_BLOCK,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector",
            kv_role="kv_both",
            kv_load_failure_policy="recompute",
            kv_connector_extra_config={
                # Twice the GPU pool, so nothing is evicted from the CPU tier.
                "cpu_bytes_to_use": 2 * gpu_bytes,
                "lazy_offload": True,
                "store_reserve_free_blocks": 0,
                "max_store_blocks_per_step": NUM_GPU_BLOCKS,
                "min_host_mem_available_gib": 0,
            },
        ),
        device_config=DeviceConfig("cpu"),
    )
    vllm_config.cache_config.num_gpu_blocks = NUM_GPU_BLOCKS
    block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    assert (block_size, hash_block_size) == (MAMBA_BLOCK, RING_BLOCK)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=StructuredOutputManager(vllm_config),
        block_size=block_size,
        hash_block_size=hash_block_size,
    )


def _make_request(request_id: str, prompt_token_ids: list[int]) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(RING_BLOCK, sha256),
    )


def _step(
    scheduler: Scheduler,
    finished_recving: set[str] | None = None,
    invalid_block_ids: set[int] | None = None,
) -> tuple[SchedulerOutput, SimpleCPUOffloadMetadata]:
    """One engine step with an instant worker: this step's store completes
    and partial prefills sample nothing."""
    output = scheduler.schedule()
    meta = output.kv_connector_metadata
    assert isinstance(meta, SimpleCPUOffloadMetadata)
    req_ids = list(output.num_scheduled_tokens)
    sampled = []
    for req_id in req_ids:
        request = scheduler.requests[req_id]
        partial = request.num_computed_tokens < request.num_tokens
        sampled.append([] if partial else [0])
    worker_meta = (
        SimpleCPUOffloadWorkerMetadata(completed_store_events={meta.store_event: 1})
        if meta.store_event >= 0
        else None
    )
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=None,
            kv_connector_output=KVConnectorOutput(
                finished_sending=set(),
                finished_recving=finished_recving or set(),
                invalid_block_ids=invalid_block_ids or set(),
                kv_connector_worker_meta=worker_meta,
            ),
        ),
    )
    return output, meta


def _req_block_ids(scheduler: Scheduler, group: int, req_id: str) -> list[int]:
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[group]
    return [block.block_id for block in manager.req_to_blocks[req_id]]


def _admit_cpu_hit_with_fresh_request(
    scheduler: Scheduler,
) -> tuple[SchedulerOutput, SimpleCPUOffloadMetadata, Request, Request]:
    """Offload a finished prompt to the CPU tier, drop it from the GPU prefix
    cache, then admit a repeat of it (async CPU load) together with a fresh
    prompt in one step."""
    prompt = list(range(1000, 1000 + PROMPT_LEN))
    scheduler.add_request(_make_request("A", prompt))
    for _ in range(PROMPT_LEN):
        _step(scheduler)
        if "A" not in scheduler.requests:
            break
    assert "A" not in scheduler.requests
    # Idle steps: the lazy walk stores A's freed blocks and they complete.
    for _ in range(4):
        _step(scheduler)
    assert scheduler.kv_cache_manager.reset_prefix_cache()

    repeat = _make_request("B", prompt)
    fresh = _make_request("C", list(range(5000, 5000 + 40)))
    manager = scheduler.connector.scheduler_manager
    _, hit_tokens = manager.cpu_coordinator.find_longest_cache_hit(
        repeat.block_hashes, repeat.num_tokens - 1
    )
    assert hit_tokens == CPU_HIT_TOKENS, "the CPU tier did not get A's prefix"

    scheduler.add_request(repeat)
    scheduler.add_request(fresh)
    output, meta = _step(scheduler)
    assert repeat.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert repeat.num_computed_tokens == CPU_HIT_TOKENS
    assert fresh.status == RequestStatus.RUNNING
    return output, meta, repeat, fresh


def test_async_load_destinations_are_not_zeroed() -> None:
    scheduler = _make_scheduler()
    assert scheduler.needs_kv_cache_zeroing

    output, meta, repeat, _ = _admit_cpu_hit_with_fresh_request(scheduler)

    zeroed = set(output.new_block_ids_to_zero or ())
    loaded = set(meta.load_gpu_blocks)
    repeat_fa = _req_block_ids(scheduler, FA_GROUP, "B")
    loaded_fa = repeat_fa[: CPU_HIT_TOKENS // FA_BLOCK]
    # The load really targets B's full-attention blocks ...
    assert set(loaded_fa) <= loaded
    # ... and none of its destinations is zeroed in the same step.
    assert not loaded & zeroed
    # B's ring block is never loaded and still has to be zeroed.
    (repeat_ring,) = _req_block_ids(scheduler, RING_GROUP, "B")
    assert repeat_ring not in loaded
    assert repeat_ring in zeroed
    # A request admitted next to it keeps its new blocks zeroed.
    fresh_blocks = _req_block_ids(scheduler, FA_GROUP, "C") + _req_block_ids(
        scheduler, RING_GROUP, "C"
    )
    assert fresh_blocks
    assert set(fresh_blocks) <= zeroed
    # The skip set lives for one step only.
    assert not scheduler._skip_zero_block_ids

    # The load completes; B's loaded blocks are not zeroed later either and
    # B finishes from the loaded prefix.
    for finished_recving in ({"B"}, set()):
        next_output, _ = _step(scheduler, finished_recving=finished_recving)
        assert not scheduler._skip_zero_block_ids
        assert not set(next_output.new_block_ids_to_zero or ()) & set(loaded_fa)
    assert repeat.is_finished()


def test_no_skip_zero_switch_zeroes_load_destinations(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_MODE", "1")
    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_NO_SKIP_ZERO", "1")
    scheduler = _make_scheduler()

    output, meta, _, _ = _admit_cpu_hit_with_fresh_request(scheduler)

    zeroed = set(output.new_block_ids_to_zero or ())
    loaded_fa = _req_block_ids(scheduler, FA_GROUP, "B")[: CPU_HIT_TOKENS // FA_BLOCK]
    assert set(loaded_fa) <= set(meta.load_gpu_blocks)
    assert set(loaded_fa) <= zeroed


def test_no_skip_zero_switch_needs_test_mode(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_MODE", raising=False)
    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_NO_SKIP_ZERO", "1")
    # The connector refuses the switch outside test mode ...
    with pytest.raises(ValueError, match="TEST_MODE"):
        _make_scheduler()
    # ... and the scheduler ignores it on its own.
    scheduler = create_scheduler(num_blocks=16)
    assert not scheduler._zero_kv_load_blocks


def test_zeroing_helpers_cover_only_zeroed_token_groups() -> None:
    scheduler = _make_scheduler()
    _admit_cpu_hit_with_fresh_request(scheduler)
    kv_cache_manager = scheduler.kv_cache_manager
    repeat_fa = _req_block_ids(scheduler, FA_GROUP, "B")

    # Only full-attention blocks of B; the ring and Mamba groups never are.
    in_range = kv_cache_manager.get_zeroing_block_ids_in_range(
        "B", FA_BLOCK, 3 * FA_BLOCK + 1
    )
    assert in_range == repeat_fa[1:4]
    assert kv_cache_manager.get_zeroing_block_ids_in_range("unknown", 0, 64) == []

    assert kv_cache_manager.take_new_block_ids() == []
    kv_cache_manager.record_blocks_for_zeroing("B", 2 * FA_BLOCK)
    assert kv_cache_manager.take_new_block_ids() == repeat_fa[2:]
    with pytest.raises(AssertionError):
        kv_cache_manager.record_blocks_for_zeroing("B", FA_BLOCK + 1)


@pytest.mark.parametrize("needs_zeroing", [True, False])
def test_failed_async_load_rezeroes_unwritten_blocks(needs_zeroing: bool) -> None:
    scheduler = object.__new__(Scheduler)
    scheduler.connector = MagicMock()
    scheduler.needs_kv_cache_zeroing = needs_zeroing
    scheduler.kv_cache_manager = MagicMock()
    scheduler.failed_recving_kv_req_ids = {"req-1", "req-2"}
    scheduler.finished_recving_kv_req_ids = {"req-1", "req-2"}

    # Truncated at the first invalid block: the rest is recomputed locally.
    truncated = MagicMock(request_id="req-1", num_computed_tokens=48)
    scheduler._update_waiting_for_remote_kv(truncated)
    scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(truncated, 48)
    if needs_zeroing:
        scheduler.kv_cache_manager.record_blocks_for_zeroing.assert_called_once_with(
            "req-1", 48
        )
    else:
        scheduler.kv_cache_manager.record_blocks_for_zeroing.assert_not_called()

    # Nothing valid: the blocks are freed and re-recorded on reallocation.
    scheduler.kv_cache_manager.reset_mock()
    nothing = MagicMock(request_id="req-2", num_computed_tokens=0)
    scheduler._update_waiting_for_remote_kv(nothing)
    scheduler.kv_cache_manager.free.assert_called_once_with(nothing)
    scheduler.kv_cache_manager.record_blocks_for_zeroing.assert_not_called()
    assert not scheduler.failed_recving_kv_req_ids
    assert not scheduler.finished_recving_kv_req_ids


def test_new_block_ids_are_drained_without_zeroing() -> None:
    # Attention-only layouts do not zero, but the per-manager lists of new
    # block ids must still be drained every step.
    scheduler = create_scheduler(num_blocks=64)
    assert not scheduler.needs_kv_cache_zeroing
    for request in create_requests(num_requests=2, num_tokens=40):
        scheduler.add_request(request)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens
    assert output.new_block_ids_to_zero is None
    for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
        assert manager.new_block_ids == []


# --- Failed loads on the hybrid layout (fingerprint gate) ---

from vllm.v1.simple_kv_offload.fingerprint import BLOCK_STATE  # noqa: E402
from vllm.v1.simple_kv_offload.stats import (  # noqa: E402
    LOAD_FAILED_BLOCKS,
    LOAD_FAILED_REQUESTS,
)


@pytest.mark.parametrize("which", ["state_block", "every_block"])
def test_failed_async_load_on_hybrid_layout_recomputes_the_prompt(which: str) -> None:
    """The worker reports loaded blocks as failed (a bad fingerprint verdict
    on a gated load): the scheduler truncates the request to its GPU prefix
    across every KV cache group, the bad CPU copies leave the tier, and the
    request is prefilled again without the bad blocks."""
    scheduler = _make_scheduler()
    assert scheduler.recompute_kv_load_failures
    _, meta, repeat, _ = _admit_cpu_hit_with_fresh_request(scheduler)
    manager = scheduler.connector.scheduler_manager
    state = manager._reqs_to_load["B"]
    loaded_gpu = list(state.transfer_meta.gpu_block_ids)
    assert loaded_gpu == meta.load_gpu_blocks and len(state.kinds) == len(loaded_gpu)
    assert meta.load_spans == {meta.load_event: [(0, len(loaded_gpu))]}
    assert all(gid in manager._gpu_offloaded for gid in loaded_gpu)
    state_blocks = [g for g, k in zip(loaded_gpu, state.kinds) if k == BLOCK_STATE]
    assert state_blocks, "the load carries no Mamba state block"
    mamba_group = len(scheduler.kv_cache_config.kv_cache_groups) - 1
    mamba_blocks = _req_block_ids(scheduler, mamba_group, "B")
    if which == "every_block":
        invalid = set(loaded_gpu)
        expected = 0  # no GPU prefix: nothing is kept
    else:
        # Only the last state block is bad: the request resumes from the
        # last real state block before it (a null entry holds no state).
        invalid = {state_blocks[-1]}
        idx = mamba_blocks.index(state_blocks[-1])
        real = [j for j in range(idx) if mamba_blocks[j] != 0]
        expected = (real[-1] + 1) * MAMBA_BLOCK if real else 0
        assert expected < CPU_HIT_TOKENS

    # The load completes with a bad verdict.
    _step(scheduler, finished_recving={"B"}, invalid_block_ids=invalid)
    assert repeat.num_computed_tokens == expected
    assert not any(gid in manager._gpu_offloaded for gid in loaded_gpu)
    assert manager._stats.data[LOAD_FAILED_REQUESTS] == 1
    assert manager._stats.data[LOAD_FAILED_BLOCKS] == len(invalid)
    assert "B" not in manager._reqs_to_load
    # The bad CPU blocks are no longer served.
    _, hit_tokens = manager.cpu_coordinator.find_longest_cache_hit(
        repeat.block_hashes, repeat.num_tokens - 1
    )
    if which == "every_block":
        assert hit_tokens == 0, hit_tokens
    else:
        assert hit_tokens < CPU_HIT_TOKENS, hit_tokens

    # The request runs to the end; with nothing left on the CPU side there
    # is no second load.
    for _ in range(PROMPT_LEN + 8):
        _, meta = _step(scheduler)
        if which == "every_block":
            assert meta.load_event < 0
        if "B" not in scheduler.requests:
            break
    assert "B" not in scheduler.requests and repeat.is_finished()
    assert repeat.num_computed_tokens >= PROMPT_LEN
    # Its freed blocks are stored again (the offloaded marks were dropped).
    for _ in range(4):
        _step(scheduler)
    _, hit_tokens = manager.cpu_coordinator.find_longest_cache_hit(
        repeat.block_hashes, repeat.num_tokens - 1
    )
    assert hit_tokens == CPU_HIT_TOKENS


def test_invalid_blocks_truncate_to_the_shortest_valid_prefix_across_groups() -> None:
    """_update_requests_with_invalid_blocks on several KV cache groups: the
    computed prefix becomes the minimum over the groups of the first invalid
    block's start, and every group's downstream blocks are evicted."""
    scheduler = _make_scheduler()
    prompt = list(range(2000, 2000 + PROMPT_LEN))
    scheduler.add_request(_make_request("D", prompt))
    for _ in range(2):
        _step(scheduler)
    request = scheduler.requests["D"]
    computed = request.num_computed_tokens
    fa_blocks = _req_block_ids(scheduler, FA_GROUP, "D")
    mamba_group = len(scheduler.kv_cache_config.kv_cache_groups) - 1
    mamba_blocks = _req_block_ids(scheduler, mamba_group, "D")
    # A live (non-null) Mamba state block with a full-attention block after
    # its start and before the computed end.
    live = [
        i
        for i, b in enumerate(mamba_blocks)
        if b != 0 and i * MAMBA_BLOCK + FA_BLOCK < computed
    ]
    assert live, (mamba_blocks, computed)
    idx_m = live[-1]
    start = idx_m * MAMBA_BLOCK
    idx_f = start // FA_BLOCK + 1
    assert fa_blocks[idx_f] != 0

    invalid = {fa_blocks[idx_f], mamba_blocks[idx_m]}
    affected, tokens, evict = scheduler._update_requests_with_invalid_blocks(
        [request], invalid, {}, evict_blocks=True
    )
    assert affected == {"D"} and request.num_computed_tokens == start
    assert tokens == computed - start
    assert set(fa_blocks[idx_f:]) <= evict and set(mamba_blocks[idx_m:]) <= evict
    assert fa_blocks[idx_f - 1] not in evict

    # Only the full-attention block invalid: the cut lands after the state
    # block's start and the Mamba blocks stay.
    request.num_computed_tokens = computed
    affected, tokens, evict = scheduler._update_requests_with_invalid_blocks(
        [request], {fa_blocks[idx_f]}, {}, evict_blocks=True
    )
    assert affected == {"D"} and request.num_computed_tokens == start + FA_BLOCK
    assert tokens == computed - start - FA_BLOCK
    assert set(fa_blocks[idx_f:]) <= evict and not set(mamba_blocks) & evict
