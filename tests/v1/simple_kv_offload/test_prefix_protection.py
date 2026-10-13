# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix protection of the lazy CPU tier: a prefix hit more than
protect_hits times within protect_window_s keeps its CPU copies for
protect_ttl_s, whatever turns the tier over in the meantime (CPU only)."""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock, get_group_id
from vllm.v1.request import Request
from vllm.v1.simple_kv_offload import stats as kv_stats
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.options import SimpleCPUOffloadOptions, parse_options

from .test_lazy_tier import _clone, _drain, _finish, _lazy, _on_cpu, _reuse
from .test_prod_shape import (
    FA,
    MAMBA_BLOCK,
    MAMBA_GROUPS,
    ProdHarness,
    prod_request,
)
from .test_scheduler import (
    BLOCK_SIZE,
    SchedulerFixture,
    _allocate_gpu_blocks,
    make_request,
    make_scheduler_output,
    simulate_load_completion,
)

_ids = itertools.count()


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _protected_tier(**options) -> tuple[SchedulerFixture, Clock]:
    """8 usable CPU blocks; more than 3 hits within 600 s protect for a day."""
    options.setdefault("protect_hits", 3)
    options.setdefault("protect_max_fraction", 0.5)
    fix = _lazy(num_gpu_blocks=48, num_cpu_blocks=9, **options)
    clock = Clock()
    fix.scheduler._clock = clock
    return fix, clock


def _hit(fix: SchedulerFixture, req: Request, blocks: list, times: int = 1) -> None:
    """`times` new requests find `blocks` in the GPU prefix cache and are
    scheduled (the two connector calls of Scheduler.schedule())."""
    sched = fix.scheduler
    for _ in range(times):
        again = _clone(req, f"{req.request_id}-hit-{next(_ids)}")
        num_local = len(blocks) * BLOCK_SIZE
        assert sched.get_num_new_matched_tokens(again, num_local) == (0, False)
        sched.update_state_after_alloc(again, KVCacheBlocks(blocks=(blocks,)), 0)


def _flood(fix: SchedulerFixture, rounds: int = 4) -> None:
    """Other traffic turns the whole CPU tier over `rounds` times."""
    for _ in range(rounds):
        _finish(fix.gpu_block_pool, 4)
        _drain(fix)
        _finish(fix.gpu_block_pool, 4)
        _drain(fix)


def _cpu_block(sched: SimpleCPUOffloadScheduler, block: KVCacheBlock) -> KVCacheBlock:
    assert block.block_hash is not None
    found = sched.cpu_block_pool.cached_block_hash_to_block.get_one_block(
        block.block_hash
    )
    assert found is not None
    return found


def _stats(sched: SimpleCPUOffloadScheduler) -> dict:
    stats = sched.get_stats()
    return dict(stats.data) if stats is not None else {}


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def test_protection_is_off_by_default() -> None:
    assert SimpleCPUOffloadOptions().protect_hits == 0
    fix = _lazy(num_gpu_blocks=48, num_cpu_blocks=9)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 2)
    _drain(fix)
    _hit(fix, req, blocks, times=50)
    assert not sched._protected and not sched._hit_times
    _flood(fix)
    assert not any(_on_cpu(sched, b) for b in blocks)
    assert kv_stats.PROTECTED_HASHES_LAST not in _stats(sched)


def test_parse_protect_options() -> None:
    opts = parse_options(
        {
            "lazy_offload": True,
            "protect_hits": 20,
            "protect_window_s": 600,
            "protect_ttl_s": 86400,
            "protect_max_fraction": 0.3,
            "protect_refresh_on_hit": False,
        }
    )
    assert (opts.protect_hits, opts.protect_window_s, opts.protect_ttl_s) == (
        20,
        600.0,
        86400.0,
    )
    assert opts.protect_max_fraction == 0.3 and not opts.protect_refresh_on_hit
    for bad in (
        {"protect_hits": -1},
        {"protect_window_s": 0},
        {"protect_ttl_s": 0},
        {"protect_max_fraction": 0.95},
        {"protect_max_fraction": -0.1},
    ):
        with pytest.raises(ValueError):
            parse_options(bad)


# ---------------------------------------------------------------------------
# The rule: more than protect_hits hits within the window
# ---------------------------------------------------------------------------
def test_prefix_at_the_threshold_is_evicted_like_any_other() -> None:
    fix, _ = _protected_tier()
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 2)
    _drain(fix)
    _hit(fix, req, blocks, times=3)  # exactly protect_hits: not more than
    assert not sched._protected
    _flood(fix)
    assert not any(_on_cpu(sched, b) for b in blocks)


def test_hot_prefix_survives_tier_turnover_and_still_loads() -> None:
    fix, _ = _protected_tier()
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 2)
    _drain(fix)
    _hit(fix, req, blocks, times=4)
    assert len(sched._protected) == 2
    held = [_cpu_block(sched, b) for b in blocks]
    assert [b.ref_cnt for b in held] == [1, 1]

    _flood(fix)
    assert all(_on_cpu(sched, b) for b in blocks)

    # The GPU copies are long gone: a cold start of the same prefix is
    # served from the protected CPU copies, which stay held afterwards.
    gpu_pool.evict_blocks({b.block_id for b in blocks})
    resend = _clone(req, "cold-start")
    assert sched.get_num_new_matched_tokens(resend, 0) == (2 * BLOCK_SIZE, True)
    dst = gpu_pool.get_new_blocks(2)
    sched.update_state_after_alloc(resend, KVCacheBlocks(blocks=(dst,)), 2 * BLOCK_SIZE)
    meta = sched.build_connector_meta(_busy_output())
    assert meta.load_cpu_blocks == [b.block_id for b in held]
    simulate_load_completion(sched, {resend.request_id})
    gpu_pool.cache_full_blocks(resend, dst, 0, 2, BLOCK_SIZE, 0)
    sched.request_finished_all_groups(resend, ([b.block_id for b in dst],))
    gpu_pool.free_blocks(reversed(dst))
    assert [b.ref_cnt for b in held] == [1, 1]
    data = _stats(sched)
    assert data[kv_stats.PROTECT_GRANTED] == 2
    assert data[kv_stats.LOAD_PROTECTED_BLOCKS] == 2
    assert data[kv_stats.PROTECTED_HASHES_LAST] == 2
    assert data[kv_stats.PROTECTED_CPU_BLOCKS_LAST] == 2


def _busy_output():
    return make_scheduler_output({"running-req": 1})


def test_hits_spread_wider_than_the_window_do_not_protect() -> None:
    fix, clock = _protected_tier()
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 2)
    _drain(fix)
    for _ in range(12):
        _hit(fix, req, blocks)
        clock.now += 250.0  # 4 hits span 750 s > 600 s
    assert not sched._protected
    # ... and four hits inside one window do.
    _hit(fix, req, blocks, times=4)
    assert len(sched._protected) == 2


def test_only_the_hit_part_of_a_prompt_is_protected() -> None:
    """Requests that share the first block and then differ protect the first
    block only: a hit on a block counts for the blocks before it, not after."""
    fix, _ = _protected_tier()
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 3)
    _drain(fix)
    for n in range(4):
        tokens = list(req.prompt_token_ids[:BLOCK_SIZE]) + [9_000_000 + n] * 40
        branch = Request(
            request_id=f"branch-{n}",
            prompt_token_ids=tokens,
            sampling_params=req.sampling_params,
            pooling_params=None,
            mm_features=None,
            block_hasher=req._block_hasher,
        )
        assert sched.get_num_new_matched_tokens(branch, BLOCK_SIZE) == (0, False)
        sched.update_state_after_alloc(branch, KVCacheBlocks(blocks=(blocks[:1],)), 0)
    assert list(sched._protect_pin_ids) == [_cpu_block(sched, blocks[0]).block_id]
    _flood(fix)
    assert _on_cpu(sched, blocks[0])
    assert not _on_cpu(sched, blocks[1]) and not _on_cpu(sched, blocks[2])


# ---------------------------------------------------------------------------
# Time to live
# ---------------------------------------------------------------------------
def test_protection_expires_after_ttl() -> None:
    fix, clock = _protected_tier(protect_ttl_s=3600.0)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 2)
    _drain(fix)
    _hit(fix, req, blocks, times=4)
    clock.now += 3599.0
    _flood(fix)
    assert all(_on_cpu(sched, b) for b in blocks)

    clock.now += 2.0
    sched.build_connector_meta(_busy_output())
    assert not sched._protected and not sched._protect_pin_ids
    assert all(_cpu_block(sched, b).ref_cnt == 0 for b in blocks)
    _flood(fix)
    assert not any(_on_cpu(sched, b) for b in blocks)
    assert _stats(sched)[kv_stats.PROTECT_EXPIRED] == 2


@pytest.mark.parametrize("refresh", [True, False])
def test_a_later_hit_restarts_the_ttl_only_with_refresh_on_hit(refresh: bool) -> None:
    fix, clock = _protected_tier(protect_ttl_s=3600.0, protect_refresh_on_hit=refresh)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 2)
    _drain(fix)
    _hit(fix, req, blocks, times=4)
    clock.now += 3000.0
    _hit(fix, req, blocks)  # one hit, far below the rate
    clock.now += 1000.0  # 4000 s after the protection was granted
    sched.build_connector_meta(_busy_output())
    assert (len(sched._protected) == 2) is refresh


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------
def test_budget_gives_up_the_oldest_protection_and_the_prefix_tail_first() -> None:
    # 9 CPU blocks, fraction 0.34 -> at most 3 held.
    fix, clock = _protected_tier(protect_max_fraction=0.34)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    assert sched._protect_max_pins == 3
    req_a, blocks_a = _finish(gpu_pool, 2)
    _drain(fix)
    _hit(fix, req_a, blocks_a, times=4)
    clock.now += 10.0
    req_b, blocks_b = _finish(gpu_pool, 2)
    _drain(fix)
    _hit(fix, req_b, blocks_b, times=4)

    # 4 blocks qualify, 3 may be held: A is older, and its tail goes first.
    held = set(sched._protect_pin_ids)
    assert held == {
        _cpu_block(sched, blocks_a[0]).block_id,
        _cpu_block(sched, blocks_b[0]).block_id,
        _cpu_block(sched, blocks_b[1]).block_id,
    }
    assert _stats(sched)[kv_stats.PROTECT_CAPPED] == 1
    # A given-up block is an ordinary cached block again.
    assert _cpu_block(sched, blocks_a[1]).ref_cnt == 0


def test_stores_never_run_out_of_cpu_blocks_with_a_full_budget() -> None:
    fix, _ = _protected_tier(protect_max_fraction=0.9)
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 6)
    _drain(fix)
    _hit(fix, req, blocks, times=4)
    assert len(sched._protect_pin_ids) == 6
    # Two CPU blocks are left for everything else; stores are paced by them.
    _flood(fix, rounds=3)
    assert all(_on_cpu(sched, b) for b in blocks)
    assert not sched.has_pending_work()


# ---------------------------------------------------------------------------
# Copies that reach the tier after the prefix became protected
# ---------------------------------------------------------------------------
def test_copy_stored_after_protection_is_held() -> None:
    fix, _ = _protected_tier()
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    # The prefix becomes hot while it only exists on the GPU.
    req = make_request(num_blocks=2)
    blocks = _allocate_gpu_blocks(gpu_pool, req, 2)
    _hit(fix, req, blocks, times=4)
    assert len(sched._protected) == 2 and not sched._protect_pin_ids
    # Its request finishes, the walk stores it: the new copies are held.
    gpu_pool.free_blocks(reversed(blocks))
    _drain(fix)
    assert len(sched._protect_pin_ids) == 2
    _flood(fix)
    assert all(_on_cpu(sched, b) for b in blocks)


def test_protected_prefix_is_stored_again_after_its_cpu_copy_was_evicted() -> None:
    """Without protection a GPU block whose CPU copy was evicted is not
    stored again during the same GPU residency (allow_restore is off)."""
    fix, _ = _protected_tier()
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 2)
    _drain(fix)
    for block in blocks:
        sched.cpu_block_pool.evict_blocks({_cpu_block(sched, block).block_id})
    _reuse(fix, blocks)
    assert _drain(fix) == [] and not any(_on_cpu(sched, b) for b in blocks)

    _hit(fix, req, blocks, times=4)
    assert len(sched._protected) == 2 and not sched._protect_pin_ids
    # Idle steps leave it alone (the idle drain must end) ...
    assert _drain(fix) == []
    # ... the next busy step after a request used it stores it again.
    _reuse(fix, blocks)
    stored = _drain(fix, step_output_fn=_busy_output)
    assert sorted(i for step in stored for i in step) == sorted(
        b.block_id for b in blocks
    )
    assert len(sched._protect_pin_ids) == 2
    _flood(fix)
    assert all(_on_cpu(sched, b) for b in blocks)


# ---------------------------------------------------------------------------
# Leaving the prefix cache
# ---------------------------------------------------------------------------
def test_reset_releases_protected_blocks() -> None:
    fix, _ = _protected_tier()
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 2)
    _drain(fix)
    _hit(fix, req, blocks, times=4)
    assert sched.reset() is True
    assert not sched._protected and not sched._protect_pin_ids
    assert not sched._hit_times
    assert all(b.ref_cnt == 0 for b in sched.cpu_block_pool.blocks if not b.is_null)


def test_quarantined_block_is_no_longer_counted_as_protected() -> None:
    fix, _ = _protected_tier()
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 2)
    _drain(fix)
    _hit(fix, req, blocks, times=4)
    bad = _cpu_block(sched, blocks[0])
    sched._quarantine([bad.block_id])
    assert bad.block_id not in sched._protect_pin_ids
    assert bad.ref_cnt == 1 and bad.block_hash is None  # the quarantine's own hold
    assert len(sched._protect_pin_ids) == 1
    # A later copy of the same prefix block is held again.
    _reuse(fix, blocks)
    _drain(fix, step_output_fn=_busy_output)
    assert len(sched._protect_pin_ids) == 2
    assert _cpu_block(sched, blocks[0]) is not bad


def test_preempted_request_and_repeated_lookup_count_once() -> None:
    fix, _ = _protected_tier()
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 2)
    _drain(fix)
    num_local = 2 * BLOCK_SIZE
    # The same request is looked up three times before it is scheduled, and
    # passes update_state_after_alloc twice (as after a load): one hit.
    again = _clone(req, "retry")
    for _ in range(3):
        sched.get_num_new_matched_tokens(again, num_local)
    for _ in range(2):
        sched.update_state_after_alloc(again, KVCacheBlocks(blocks=(blocks,)), 0)
    assert [len(t) for t in sched._hit_times.values()] == [1, 1]
    # A preempted request that resumes on its own blocks is not a hit.
    resumed = _clone(req, "resumed")
    resumed.num_preemptions = 1
    sched.get_num_new_matched_tokens(resumed, num_local)
    sched.update_state_after_alloc(resumed, KVCacheBlocks(blocks=(blocks,)), 0)
    assert [len(t) for t in sched._hit_times.values()] == [1, 1]
    # A request that is dropped before it is scheduled leaves nothing behind.
    dropped = _clone(req, "dropped")
    sched.get_num_new_matched_tokens(dropped, num_local)
    sched.request_finished_all_groups(dropped, ([],))
    assert not sched._local_hit_tokens


def test_hit_history_is_pruned() -> None:
    fix, clock = _protected_tier()
    sched, gpu_pool = fix.scheduler, fix.gpu_block_pool
    req, blocks = _finish(gpu_pool, 2)
    _drain(fix)
    _hit(fix, req, blocks, times=2)
    assert len(sched._hit_times) == 2
    clock.now += 700.0
    sched.build_connector_meta(_busy_output())
    assert not sched._hit_times


# ---------------------------------------------------------------------------
# Production shape: 816-token attention blocks, 1632-token state blocks,
# 8-token request hashes
# ---------------------------------------------------------------------------
def _prod(protect: bool) -> ProdHarness:
    options: dict[str, Any] = {"lazy_offload": True}
    if protect:
        options.update(protect_hits=3, protect_max_fraction=0.5)
    h = ProdHarness(
        num_gpu_blocks=48, retention=None, options=SimpleCPUOffloadOptions(**options)
    )
    clock = Clock()
    h.mgr._clock = clock
    return h


@pytest.mark.parametrize("protect", [True, False], ids=["protected", "control"])
def test_shared_prefix_cold_start_prod_shape(protect: bool) -> None:
    """Sessions share a prompt head and then differ; afterwards other traffic
    turns the CPU tier over. With protection a new session is served the
    same hit from CPU that the sessions got from the GPU, without it nothing.

    With MTP a hit stops one 1632-token block short of what is cached, so the
    block after the hit has to be protected along with it."""
    h = _prod(protect)
    hit = 2 * MAMBA_BLOCK
    base = prod_request(hit + MAMBA_BLOCK + 40)  # the shared head
    h.store_all(base)
    for _ in range(5):
        session = prod_request(hit + 4 * MAMBA_BLOCK + 11, prefix=base)
        # Served from the GPU prefix cache or, where the GPU no longer holds
        # the Mamba state of that position, from CPU: a hit either way.
        assert sum(_serve(h, session)) == hit

    if protect:
        assert h.mgr._protect_stride == 102  # 816-token blocks of 8-token hashes
        groups = [
            get_group_id(b.block_hash)
            for b in h.cpu_pool.blocks
            if b.block_id in h.mgr._protect_pin_ids
        ]
        # The 4 attention blocks of the hit and the 2 of the block after it,
        # with the state every Mamba group needs to resume at 4896 tokens.
        assert groups.count(FA) == 6
        assert all(groups.count(g) >= 1 for g in MAMBA_GROUPS)
    else:
        assert not h.mgr._protected

    # Unrelated traffic: twice as many new blocks as the CPU tier holds.
    stored_before = len(h.stored)
    while len(h.stored) - stored_before < 2 * h.mgr.num_cpu_blocks:
        h.store_all(prod_request(4 * MAMBA_BLOCK + 5))
    h.evict_gpu_copies()

    cold = prod_request(hit + 2 * MAMBA_BLOCK + 7, prefix=base)
    assert _serve(h, cold) == (0, hit if protect else 0)
    if protect:
        data = h.mgr.get_stats().data
        assert data[kv_stats.LOAD_PROTECTED_BLOCKS] > 0
        # Everything but the protected copies is released again.
        held = set(h.mgr._protect_pin_ids)
        assert all(
            b.ref_cnt == (1 if b.block_id in held else 0)
            for b in h.cpu_pool.blocks
            if not b.is_null
        )
    else:
        h.assert_no_leaks()


def _serve(h: ProdHarness, req: Request) -> tuple[int, int]:
    """Admit a request, complete its CPU load if it got one, run it to the
    end and finish it. Returns its (GPU, CPU) hit in tokens."""
    num_local, ext = h.admit(req)
    if ext:
        assert h.step(0).load_event >= 0
        h.complete_load(req)
    h.run(req)
    h.finish(req)
    h.drain()
    return num_local, ext
