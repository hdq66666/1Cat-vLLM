# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy CPU tier of SimpleCPUOffloadScheduler: paced cursor walk, idle drain,
no re-store, CPU hit refusal, reset and diagnostics (CPU only)."""

from __future__ import annotations

import pytest
import torch

import vllm.v1.simple_kv_offload.manager as manager_mod
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.request import Request
from vllm.v1.simple_kv_offload import stats as kv_stats
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.options import SimpleCPUOffloadOptions

from .test_scheduler import (
    BLOCK_SIZE,
    SchedulerFixture,
    _allocate_gpu_blocks,
    get_cpu_free_blocks,
    make_request,
    make_scheduler,
    make_scheduler_output,
    simulate_load_completion,
    simulate_store_completion,
)

GiB = 1024**3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lazy(
    num_gpu_blocks: int = 32, num_cpu_blocks: int = 32, **options
) -> SchedulerFixture:
    options.setdefault("store_reserve_free_blocks", 0)
    return make_scheduler(
        num_cpu_blocks=num_cpu_blocks,
        num_gpu_blocks=num_gpu_blocks,
        lazy=True,
        options=SimpleCPUOffloadOptions(lazy_offload=True, **options),
    )


def _idle():
    return make_scheduler_output({})


def _busy():
    # The lazy walk only looks at the token count of the step.
    return make_scheduler_output({"running-req": 1})


def _finish(gpu_pool: BlockPool, num_blocks: int) -> tuple[Request, list]:
    """A request computed num_blocks full blocks and finished: its hashed
    blocks are appended at the MRU tail of the free queue (in reverse order,
    as SingleTypeKVCacheManager.free() does)."""
    req = make_request(num_blocks=num_blocks)
    blocks = _allocate_gpu_blocks(gpu_pool, req, num_blocks)
    gpu_pool.free_blocks(reversed(blocks))
    return req, blocks


def _clone(req: Request, request_id: str) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=req.prompt_token_ids,
        sampling_params=req.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=req._block_hasher,
    )


def _free_queue(gpu_pool: BlockPool) -> list[KVCacheBlock]:
    queue = gpu_pool.free_block_queue
    out = []
    node = queue.fake_free_list_head.next_free_block
    while node is not queue.fake_free_list_tail:
        out.append(node)
        node = node.next_free_block
    return out


def _is_free(block: KVCacheBlock) -> bool:
    return (
        block.ref_cnt == 0
        and block.prev_free_block is not None
        and block.next_free_block is not None
    )


def _on_cpu(sched: SimpleCPUOffloadScheduler, block: KVCacheBlock) -> bool:
    assert block.block_hash is not None
    cached = sched.cpu_block_pool.cached_block_hash_to_block
    return cached.get_one_block(block.block_hash) is not None


def _drain(
    fix: SchedulerFixture, step_output_fn=_idle, max_steps: int = 100
) -> list[list[int]]:
    """Step until has_pending_work() turns False. A store event completes one
    step after it was emitted (update_from_output of the next step).
    Returns the GPU block ids stored per step."""
    sched = fix.scheduler
    stored: list[list[int]] = []
    inflight: list[int] = []
    for _ in range(max_steps):
        meta = sched.build_connector_meta(step_output_fn())
        for event in inflight:
            simulate_store_completion(sched, event)
        inflight = []
        if meta.store_event >= 0:
            stored.append(list(meta.store_gpu_blocks))
            inflight.append(meta.store_event)
        if not sched.has_pending_work():
            assert not inflight
            return stored
    raise AssertionError("has_pending_work() never turned False (livelock)")


def _evict_cpu_copy(sched: SimpleCPUOffloadScheduler, block: KVCacheBlock) -> None:
    assert block.block_hash is not None
    cached = sched.cpu_block_pool.cached_block_hash_to_block
    cpu_blk = cached.get_one_block(block.block_hash)
    assert cpu_blk is not None and cpu_blk.ref_cnt == 0
    sched.cpu_block_pool.evict_blocks({cpu_blk.block_id})


def _reuse(fix: SchedulerFixture, blocks: list) -> None:
    """A request hits the blocks in its prefix, runs for a step and finishes:
    they move back to the MRU tail of the free queue with their hashes.
    (Frees reach the pool before the touches of the next schedule(), so the
    walk always runs between a touch and the matching free.)"""
    fix.gpu_block_pool.touch(blocks)
    meta = fix.scheduler.build_connector_meta(_busy())
    assert meta.store_event < 0
    fix.gpu_block_pool.free_blocks(reversed(blocks))


# ---------------------------------------------------------------------------
# P3: lazy target counts only prefix-cacheable groups
# ---------------------------------------------------------------------------
def test_lazy_target_skips_non_cacheable_groups() -> None:
    fa = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=16, dtype=torch.float16
    )
    ring = CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=16, dtype=torch.float16
    )
    mamba = MambaSpec(block_size=32, shapes=((1, 64),), dtypes=(torch.float16,))
    config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[KVCacheTensor(size=1024 * 16, shared_by=["a", "r", "m"])],
        kv_cache_groups=[
            KVCacheGroupSpec(["a"], fa),
            KVCacheGroupSpec(["r"], ring),
            KVCacheGroupSpec(["m"], mamba),
        ],
    )
    # (cdiv(64, 16) + 2) * 2; the ring would add cdiv(64, 8) = 8 more.
    assert SimpleCPUOffloadScheduler._estimate_lazy_target_blocks(config, 64) == 12


# ---------------------------------------------------------------------------
# P4/P16: idle drain and has_pending_work()
# ---------------------------------------------------------------------------
def test_idle_drain_stores_every_hashed_block_then_stops() -> None:
    fix = _lazy(num_gpu_blocks=64, num_cpu_blocks=64, max_store_blocks_per_step=2)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    assert not sched.has_pending_work()

    _, blocks = _finish(gpu_pool, 20)
    stored = _drain(fix)

    flat = [bid for step in stored for bid in step]
    assert sorted(flat) == sorted(b.block_id for b in blocks), "each block once"
    # Idle steps select more than max_store_blocks_per_step, within the
    # in-flight pin cap (and visit budget) of 4 * max_store_blocks_per_step.
    assert 2 < max(len(step) for step in stored) <= 8
    assert all(_on_cpu(sched, b) and _is_free(b) for b in blocks)
    assert not sched._lazy_inflight_hashes

    # Stays idle: no further stores, has_pending_work() stays False.
    for _ in range(5):
        meta = sched.build_connector_meta(_idle())
        assert meta.store_event < 0
        assert not sched.has_pending_work()


def test_idle_drain_ends_when_reserve_blocks_every_store() -> None:
    # 15 usable GPU blocks, all free: the reserve never lets a store pin one.
    fix = _lazy(num_gpu_blocks=16, store_reserve_free_blocks=15)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    _finish(gpu_pool, 4)

    for _ in range(3):
        meta = sched.build_connector_meta(_idle())
        assert meta.store_event < 0
        assert not sched.has_pending_work(), "nothing can change: no livelock"
    stats = sched.get_stats()
    assert stats is not None and stats.data[kv_stats.RESERVE_STOPS] == 3


def test_idle_drain_paced_by_reserve() -> None:
    # 15 usable blocks, reserve 12: at most 3 stores in flight at a time.
    fix = _lazy(num_gpu_blocks=16, store_reserve_free_blocks=12)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    _, blocks = _finish(gpu_pool, 8)

    stored: list[int] = []
    inflight: list[int] = []
    for _ in range(50):
        meta = sched.build_connector_meta(_idle())
        assert gpu_pool.get_num_free_blocks() >= 12
        for event in inflight:
            simulate_store_completion(sched, event)
        inflight = []
        if meta.store_event >= 0:
            assert len(meta.store_gpu_blocks) <= 3
            stored.extend(meta.store_gpu_blocks)
            inflight.append(meta.store_event)
        if not sched.has_pending_work():
            break
    assert sorted(stored) == sorted(b.block_id for b in blocks)
    assert not sched.has_pending_work()


# ---------------------------------------------------------------------------
# T3: reserve, pacing, in-flight cap, idle multiplier, cursor
# ---------------------------------------------------------------------------
def test_busy_pacing_and_inflight_cap() -> None:
    fix = _lazy(num_gpu_blocks=34, num_cpu_blocks=64, max_store_blocks_per_step=4)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    _finish(gpu_pool, 30)

    events = []
    for _ in range(4):
        meta = sched.build_connector_meta(_busy())
        assert len(meta.store_gpu_blocks) == 4
        events.append(meta.store_event)
    # 16 GPU blocks pinned = 4 * max_store_blocks_per_step: nothing more.
    meta = sched.build_connector_meta(_busy())
    assert meta.store_event < 0
    assert sched.has_pending_work()
    stats = sched.get_stats()
    assert stats is not None
    assert stats.data[kv_stats.STORE_PINS_LAST] == 16
    assert stats.data[kv_stats.STORES_SELECTED] == 16

    simulate_store_completion(sched, events[0])
    meta = sched.build_connector_meta(_busy())
    assert len(meta.store_gpu_blocks) == 4
    for event in events[1:] + [meta.store_event]:
        simulate_store_completion(sched, event)
    # Idle: 8x per step, still capped at 16 pinned blocks.
    meta = sched.build_connector_meta(_idle())
    assert len(meta.store_gpu_blocks) == 10  # the rest of the 30 blocks
    simulate_store_completion(sched, meta.store_event)


def test_reserve_never_starves_running_requests() -> None:
    fix = _lazy(num_gpu_blocks=32, store_reserve_free_blocks=3)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    _, blocks = _finish(gpu_pool, 8)
    # Running requests hold everything else: 8 free blocks, all hashed.
    running = gpu_pool.get_new_blocks(gpu_pool.get_num_free_blocks() - 8)
    assert gpu_pool.get_num_free_blocks() == 8

    meta = sched.build_connector_meta(_busy())
    assert len(meta.store_gpu_blocks) == 5
    assert gpu_pool.get_num_free_blocks() == 3
    # A running request that needs one more block still gets it (it evicts
    # a block that was not offloaded yet: a later miss, never a stall).
    running += gpu_pool.get_new_blocks(1)

    simulate_store_completion(sched, meta.store_event)
    gpu_pool.free_blocks(running)
    stored = _drain(fix)
    flat = meta.store_gpu_blocks + [bid for step in stored for bid in step]
    assert len(flat) == 7
    assert sorted(flat) == sorted(b.block_id for b in blocks if b.block_hash)


def test_cursor_resumes_after_stored_run() -> None:
    """The cursor is the last visited block that was not selected, so the
    next walk continues right after the blocks stored by this one instead of
    restarting at the LRU head."""
    fix = _lazy(num_gpu_blocks=16, max_store_blocks_per_step=3)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    # Three unhashed blocks at the head, then the hashed blocks of two
    # requests.
    _finish(gpu_pool, 5)
    _finish(gpu_pool, 7)
    order = [b.block_id for b in _free_queue(gpu_pool) if b.block_hash is not None]

    stored: list[int] = []
    for _ in range(8):
        meta = sched.build_connector_meta(_busy())
        cursor = sched._cursor
        assert cursor is not None and _is_free(cursor)
        assert cursor.block_id not in meta.store_gpu_blocks
        if meta.store_event < 0:
            break
        stored.extend(meta.store_gpu_blocks)
        simulate_store_completion(sched, meta.store_event)
    assert stored == order


def test_cursor_restarts_at_head_when_allocated() -> None:
    fix = _lazy(num_gpu_blocks=8, max_store_blocks_per_step=2)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    _, blocks = _finish(gpu_pool, 6)
    meta = sched.build_connector_meta(_busy())
    simulate_store_completion(sched, meta.store_event)
    cursor = sched._cursor
    assert cursor is not None
    # A prefix hit takes the cursor block out of the free queue.
    gpu_pool.touch([cursor])
    meta = sched.build_connector_meta(_busy())
    assert len(meta.store_gpu_blocks) == 2
    simulate_store_completion(sched, meta.store_event)
    gpu_pool.free_blocks([cursor])
    stored = meta.store_gpu_blocks + [b for s in _drain(fix) for b in s]
    assert all(_on_cpu(sched, b) for b in blocks)
    assert len(stored) == len(set(stored))


# ---------------------------------------------------------------------------
# T4 / P5: no re-store, in-flight dedup, CPU LRU refresh
# ---------------------------------------------------------------------------
def test_no_restore_after_cpu_eviction() -> None:
    fix = _lazy(num_gpu_blocks=32)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    _, blocks = _finish(gpu_pool, 3)
    _drain(fix)
    cpu_free = get_cpu_free_blocks(sched)

    # The CPU copies are evicted while the GPU copies are reused and freed
    # again (back at the MRU tail, in front of the cursor).
    for block in blocks:
        _evict_cpu_copy(sched, block)
    _reuse(fix, blocks)
    queue_before = [b.block_id for b in _free_queue(gpu_pool)]

    for step_output in (_busy, _idle):
        meta = sched.build_connector_meta(step_output())
        assert meta.store_event < 0
        assert not sched.has_pending_work()
    assert [b.block_id for b in _free_queue(gpu_pool)] == queue_before
    assert get_cpu_free_blocks(sched) == cpu_free
    stats = sched.get_stats()
    assert stats is not None and stats.data[kv_stats.RESTORE_SKIPPED] == 3


def test_allow_restore_restores_on_busy_steps_only() -> None:
    fix = _lazy(num_gpu_blocks=32, allow_restore=True)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    _, blocks = _finish(gpu_pool, 3)
    _drain(fix)
    for block in blocks:
        _evict_cpu_copy(sched, block)
    _reuse(fix, blocks)

    # Idle steps never re-store, so the idle drain always ends.
    meta = sched.build_connector_meta(_idle())
    assert meta.store_event < 0
    assert not sched.has_pending_work()

    _reuse(fix, blocks)
    meta = sched.build_connector_meta(_busy())
    assert sorted(meta.store_gpu_blocks) == sorted(b.block_id for b in blocks)
    simulate_store_completion(sched, meta.store_event)
    assert all(_on_cpu(sched, b) for b in blocks)


def test_inflight_hash_dedup() -> None:
    """Two GPU blocks with the same content produce one store."""
    fix = _lazy(num_gpu_blocks=32)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req = make_request(num_blocks=2)
    first = _allocate_gpu_blocks(gpu_pool, req, 2)
    second = _allocate_gpu_blocks(gpu_pool, _clone(req, "req-dup"), 2)
    gpu_pool.free_blocks(first + second)

    meta = sched.build_connector_meta(_busy())
    assert meta.store_gpu_blocks == [b.block_id for b in first]
    stats = sched.get_stats()
    assert stats is not None and stats.data[kv_stats.INFLIGHT_SKIPPED] == 2

    # Not stored again once the first copy completed.
    simulate_store_completion(sched, meta.store_event)
    assert not sched._lazy_inflight_hashes
    assert _drain(fix) == []


def test_walk_refreshes_cpu_lru_of_reused_blocks() -> None:
    # 4 usable CPU blocks (the null block takes one).
    fix = _lazy(num_gpu_blocks=32, num_cpu_blocks=5)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    _, blocks_a = _finish(gpu_pool, 2)
    _drain(fix)
    _, blocks_b = _finish(gpu_pool, 2)
    _drain(fix)
    # A is reused on GPU; the walk passes it and refreshes its CPU copy.
    _reuse(fix, blocks_a)
    assert _drain(fix) == []
    _, blocks_c = _finish(gpu_pool, 2)
    _drain(fix)

    assert all(_on_cpu(sched, b) for b in blocks_a + blocks_c)
    assert not any(_on_cpu(sched, b) for b in blocks_b)


def test_load_destinations_are_not_stored_back() -> None:
    fix = _lazy(num_gpu_blocks=32, num_cpu_blocks=5)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, _ = _finish(gpu_pool, 2)
    _drain(fix)
    # Evict the GPU copies, then load them back for a resend.
    gpu_pool.evict_blocks({b.block_id for b in _free_queue(gpu_pool)})

    req2 = _clone(req, "req-load")
    hit, is_async = sched.get_num_new_matched_tokens(req2, 0)
    assert (hit, is_async) == (2 * BLOCK_SIZE, True)
    dst = gpu_pool.get_new_blocks(2)
    sched.update_state_after_alloc(req2, KVCacheBlocks(blocks=(dst,)), hit)
    meta = sched.build_connector_meta(_busy())
    assert meta.load_gpu_blocks == [b.block_id for b in dst]
    simulate_load_completion(sched, {req2.request_id})
    # The scheduler caches the loaded blocks, then the request finishes.
    gpu_pool.cache_full_blocks(req2, dst, 0, 2, BLOCK_SIZE, 0)
    for block in dst:
        assert sched._gpu_offloaded[block.block_id] == (
            block.block_hash,
            block.alloc_generation,
        )
        _evict_cpu_copy(sched, block)
    gpu_pool.free_blocks(reversed(dst))

    assert _drain(fix) == []
    stats = sched.get_stats()
    assert stats is not None and stats.data[kv_stats.RESTORE_SKIPPED] == 2


# ---------------------------------------------------------------------------
# P14: CPU hit validation before pinning
# ---------------------------------------------------------------------------
def test_misaligned_cpu_hit_is_refused_without_pins(monkeypatch) -> None:
    fix = _lazy(num_gpu_blocks=32)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, _ = _finish(gpu_pool, 3)
    _drain(fix)
    cpu_free = get_cpu_free_blocks(sched)
    errors: list[tuple] = []
    monkeypatch.setattr(manager_mod.logger, "error", lambda *a, **k: errors.append(a))
    now = [1000.0]
    sched._clock = lambda: now[0]

    # The local hit ends mid-block, so the CPU hit would end mid-block too.
    req2 = _clone(req, "req-misaligned")
    assert sched.get_num_new_matched_tokens(req2, BLOCK_SIZE // 2) == (0, False)
    assert get_cpu_free_blocks(sched) == cpu_free
    assert not sched._pending_cpu_hits
    assert len(errors) == 1

    # Rate limited: one ERROR line per minute, with the count since the last.
    sched.get_num_new_matched_tokens(req2, BLOCK_SIZE // 2)
    now[0] += 59
    sched.get_num_new_matched_tokens(req2, BLOCK_SIZE // 2)
    assert len(errors) == 1
    now[0] += 2
    sched.get_num_new_matched_tokens(req2, BLOCK_SIZE // 2)
    assert len(errors) == 2 and errors[1][-1] == 3
    stats = sched.get_stats()
    assert stats is not None and stats.data[kv_stats.CPU_HITS_REFUSED] == 4

    # A valid lookup still hits.
    assert sched.get_num_new_matched_tokens(req2, 0) == (3 * BLOCK_SIZE, True)


def test_misaligned_cpu_hit_raises_in_test_mode(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_MODE", "1")
    fix = _lazy(num_gpu_blocks=32)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, _ = _finish(gpu_pool, 3)
    _drain(fix)
    cpu_free = get_cpu_free_blocks(sched)
    with pytest.raises(AssertionError, match="invalid CPU hit"):
        sched.get_num_new_matched_tokens(_clone(req, "req-x"), BLOCK_SIZE // 2)
    assert get_cpu_free_blocks(sched) == cpu_free


def test_external_tokens_without_pending_hit_raise() -> None:
    fix = _lazy(num_gpu_blocks=32)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req = make_request(num_blocks=2)
    blocks = KVCacheBlocks(blocks=(gpu_pool.get_new_blocks(2),))
    with pytest.raises(RuntimeError, match="num_external_tokens"):
        sched.update_state_after_alloc(req, blocks, 2 * BLOCK_SIZE)


# ---------------------------------------------------------------------------
# P12: reset
# ---------------------------------------------------------------------------
def test_reset_refuses_while_transfers_are_in_flight() -> None:
    fix = _lazy(num_gpu_blocks=32)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 3)

    meta = sched.build_connector_meta(_busy())
    assert meta.store_event >= 0
    assert sched.reset() is False
    simulate_store_completion(sched, meta.store_event)

    # Pending load.
    gpu_pool.evict_blocks({b.block_id for b in blocks})
    req2 = _clone(req, "req-load")
    hit, _ = sched.get_num_new_matched_tokens(req2, 0)
    dst = gpu_pool.get_new_blocks(3)
    sched.update_state_after_alloc(req2, KVCacheBlocks(blocks=(dst,)), hit)
    assert sched.reset() is False
    meta = sched.build_connector_meta(_busy())
    assert sched.reset() is False
    simulate_load_completion(sched, {req2.request_id})
    gpu_pool.free_blocks(dst)

    # A pinned CPU hit that was never consumed is released by the reset.
    req3 = _clone(req, "req-pending")
    assert sched.get_num_new_matched_tokens(req3, 0)[0] == hit
    assert sched._pending_cpu_hits
    assert sched.reset() is True
    assert not sched._pending_cpu_hits
    assert len(sched.cpu_block_pool.cached_block_hash_to_block) == 0
    assert get_cpu_free_blocks(sched) == sched.num_cpu_blocks - 1
    assert sched._cursor is None
    assert not sched._gpu_offloaded and not sched._lazy_inflight_hashes
    assert not sched.has_pending_work()
    assert sched.get_num_new_matched_tokens(_clone(req, "req-after"), 0) == (0, False)


# ---------------------------------------------------------------------------
# P6b / P11 / P15 / P7d: metadata and diagnostics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lazy", [False, True])
def test_metadata_carries_num_cpu_blocks(lazy: bool) -> None:
    fix = make_scheduler(num_cpu_blocks=12, lazy=lazy)
    meta = fix.scheduler.build_connector_meta(make_scheduler_output({}))
    assert meta.num_cpu_blocks == fix.scheduler.num_cpu_blocks == 12


def test_scheduler_stats_round_trip(monkeypatch) -> None:
    monkeypatch.setattr(manager_mod, "read_mem_available_bytes", lambda: None)
    fix = _lazy(num_gpu_blocks=32, num_cpu_blocks=16)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 3)
    meta = sched.build_connector_meta(_busy())

    stats = sched.get_stats()
    assert stats is not None
    assert stats.data == {
        kv_stats.STORES_SELECTED: 3,
        kv_stats.STORE_PINS_LAST: 3,
        kv_stats.CPU_USED_BLOCKS_LAST: 4,  # the null block + 3 in flight
        kv_stats.CPU_CACHED_BLOCKS_LAST: 0,
    }
    simulate_store_completion(sched, meta.store_event)
    gpu_pool.evict_blocks({b.block_id for b in blocks})
    req2 = _clone(req, "req-load")
    hit, _ = sched.get_num_new_matched_tokens(req2, 0)
    sched.update_state_after_alloc(
        req2, KVCacheBlocks(blocks=(gpu_pool.get_new_blocks(3),)), hit
    )
    stats = sched.get_stats()
    assert stats is not None
    assert stats.data == {
        kv_stats.LOAD_REQUESTS: 1,
        kv_stats.STORE_PINS_LAST: 0,
        kv_stats.CPU_USED_BLOCKS_LAST: 4,  # the null block + 3 load sources
        kv_stats.CPU_CACHED_BLOCKS_LAST: 3,
    }
    assert set(stats.reduce()) >= {kv_stats.LOAD_REQUESTS, kv_stats.STORE_PINS_LAST}


def test_load_log_lines(monkeypatch) -> None:
    fix = _lazy(num_gpu_blocks=32)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 3)
    _drain(fix)
    gpu_pool.evict_blocks({b.block_id for b in blocks})
    lines: list[str] = []
    monkeypatch.setattr(
        manager_mod.logger, "info", lambda msg, *args: lines.append(msg % args)
    )
    now = [50.0]
    sched._clock = lambda: now[0]

    req2 = _clone(req, "req-log")
    hit, _ = sched.get_num_new_matched_tokens(req2, BLOCK_SIZE)
    sched.update_state_after_alloc(
        req2,
        KVCacheBlocks(blocks=(gpu_pool.get_new_blocks(3),)),
        hit,
    )
    sched.build_connector_meta(_busy())
    sched.build_connector_meta(_busy())
    now[0] += 0.25
    simulate_load_completion(sched, {req2.request_id})
    assert lines == [
        "SimpleCPU load req=req-log local_hit=16 cpu_hit=32 blocks=2 event=0",
        "SimpleCPU load done req=req-log event=0 steps=2 wall_ms=250.0",
    ]


def test_load_dst_recently_freed_counter() -> None:
    fix = _lazy(num_gpu_blocks=32)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 2)
    _drain(fix)
    gpu_pool.evict_blocks({b.block_id for b in blocks})

    def admit(request_id: str) -> list[int]:
        request = _clone(req, request_id)
        hit, _ = sched.get_num_new_matched_tokens(request, 0)
        dst = gpu_pool.get_new_blocks(2)
        sched.update_state_after_alloc(request, KVCacheBlocks(blocks=(dst,)), hit)
        return [b.block_id for b in dst]

    # F finishes: its unhashed pages are recycled first by the allocator.
    f_req = make_request(num_blocks=1)
    f_blocks = gpu_pool.get_new_blocks(2)
    sched.request_finished_all_groups(f_req, ([b.block_id for b in f_blocks],))
    gpu_pool.free_blocks(f_blocks)
    assert admit("req-b") == [b.block_id for b in f_blocks]
    stats = sched.get_stats()
    assert stats is not None and stats.data[kv_stats.LOAD_DST_RECENTLY_FREED] == 2

    # Two scheduler steps later F's pages no longer count.
    sched.build_connector_meta(_busy())
    assert sched._recently_freed
    sched.build_connector_meta(_busy())
    assert not sched._recently_freed


def test_host_mem_sampled_every_minute(monkeypatch) -> None:
    fix = _lazy(num_gpu_blocks=32)
    sched = fix.scheduler
    available: list[int | None] = [20 * GiB]
    monkeypatch.setattr(manager_mod, "read_mem_available_bytes", lambda: available[0])
    warnings: list[tuple] = []
    monkeypatch.setattr(
        manager_mod.logger, "warning", lambda *a, **k: warnings.append(a)
    )
    now = [0.0]
    sched._clock = lambda: now[0]

    sched.build_connector_meta(_idle())
    available[0] = 11 * GiB
    now[0] = 59.0
    sched.build_connector_meta(_idle())
    stats = sched.get_stats()
    assert stats is not None
    assert stats.data[kv_stats.HOST_MEM_AVAILABLE_GIB_MIN] == 20
    assert not warnings

    now[0] = 61.0
    sched.build_connector_meta(_idle())
    now[0] = 62.0
    available[0] = 5 * GiB
    sched.build_connector_meta(_idle())
    stats = sched.get_stats()
    assert stats is not None
    assert stats.data[kv_stats.HOST_MEM_AVAILABLE_GIB_MIN] == 11
    assert len(warnings) == 1

    available[0] = None
    now[0] = 200.0
    sched.build_connector_meta(_idle())
    stats = sched.get_stats()
    assert stats is not None
    assert kv_stats.HOST_MEM_AVAILABLE_GIB_MIN not in stats.data
