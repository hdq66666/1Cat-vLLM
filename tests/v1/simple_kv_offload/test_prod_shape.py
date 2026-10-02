# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SimpleCPUOffloadScheduler at the production cache shape (CPU only).

Per TP rank: 13 QSA layers (816-token FullAttention main KV plus an MLA-style
compressed key with compress ratio 4, and an 8-token CircularBuffer ring),
39 GDN layers in 3 Mamba groups and one PLE Mamba group (1632 tokens, align
mode, 4 MTP state blocks), 355 GPU blocks of 11,541,504 bytes, scheduler
block 1632 and hash block 8. The GPU side is a real KVCacheManager with
use_eagle=True; the tests replay the scheduler's admission order
(Scheduler.schedule() waiting loop, _update_waiting_for_remote_kv,
_free_request) around it instead of driving a Scheduler.
"""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHashListWithBlockSize,
    KVCacheBlock,
    generate_scheduler_kv_cache_config,
    get_group_id,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.simple_kv_offload import stats as kv_stats
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.options import SimpleCPUOffloadOptions, parse_options

init_none_hash(sha256)

GiB = 1024**3
NUM_LAYERS = 52
FA_BLOCK = 816
MAMBA_BLOCK = 1632
HASH_BLOCK = 8
MAX_MODEL_LEN = 262144
MAX_NUM_BATCHED_TOKENS = 8192
MAX_NUM_SEQS = 5
NUM_SPEC_TOKENS = 4
WORLD_SIZE = 4
PROD_GPU_BLOCKS = 355
# 13 x (835,584 B main KV + 52,224 B compressed key)
SLOT_BYTES = 11_541_504
PROMPT_LEN = 6 * MAMBA_BLOCK + 37
FA, RING, PLE = 0, 1, 5
GDN = (2, 3, 4)
MAMBA_GROUPS = GDN + (PLE,)


# ---------------------------------------------------------------------------
# Production cache shape
# ---------------------------------------------------------------------------


def _prod_specs() -> dict:
    specs: dict = {}
    for layer in range(NUM_LAYERS):
        if layer % 4 != 3:
            specs[f"model.layers.{layer}.linear_attn"] = MambaSpec(
                block_size=MAMBA_BLOCK,
                shapes=((16, 1024),),
                dtypes=(torch.float16,),
                mamba_cache_mode="align",
                num_speculative_blocks=NUM_SPEC_TOKENS,
            )
            continue
        prefix = f"model.layers.{layer}.self_attn"
        specs[prefix] = FullAttentionSpec(
            block_size=FA_BLOCK,
            num_kv_heads=1,
            head_size=256,
            head_size_v=256,
            dtype=torch.float16,
        )
        specs[f"{prefix}.compressed"] = MLAAttentionSpec(
            block_size=FA_BLOCK,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.float16,
            compress_ratio=4,
        )
        specs[f"{prefix}.compressor_state"] = CircularBufferSpec(
            block_size=HASH_BLOCK,
            num_kv_heads=1,
            head_size=128,
            head_size_v=0,
            dtype=torch.float16,
        )
    specs["model.layers.0.ple"] = MambaSpec(
        block_size=MAMBA_BLOCK,
        shapes=((4, 1024),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
        num_speculative_blocks=NUM_SPEC_TOKENS,
        tp_replicated=True,
    )
    return specs


def _grouping_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=MAX_MODEL_LEN,
            get_num_kv_heads=lambda parallel_config: 1,
            get_total_num_hidden_layers=lambda: NUM_LAYERS,
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=MAX_NUM_SEQS,
        ),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=None, mamba_cache_mode="align"
        ),
    )


def prod_kv_cache_config(num_gpu_blocks: int) -> KVCacheConfig:
    """The scheduler-side KVCacheConfig, as EngineCore builds it."""
    vllm_config = _grouping_config()
    groups = get_kv_cache_groups(vllm_config, _prod_specs())
    worker_config = get_kv_cache_config_from_groups(
        vllm_config, groups, available_memory=num_gpu_blocks * SLOT_BYTES
    )
    return generate_scheduler_kv_cache_config([worker_config])


def _manager_config(use_eagle: bool) -> SimpleNamespace:
    return SimpleNamespace(
        kv_events_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            world_size=WORLD_SIZE,
        ),
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=MAX_NUM_SEQS,
        ),
        speculative_config=(
            SimpleNamespace(use_eagle_kv_cache=lambda: True) if use_eagle else None
        ),
        max_in_flight_tokens=2 * MAX_NUM_BATCHED_TOKENS,
    )


_request_ids = itertools.count()


def prod_request(
    num_tokens: int = PROMPT_LEN, prefix: Request | None = None
) -> Request:
    n = next(_request_ids)
    tokens = list(prefix.prompt_token_ids) if prefix is not None else []
    start = (n + 1) * 1_000_000
    tokens += range(start, start + num_tokens - len(tokens))
    return Request(
        request_id=f"prod-{n}",
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(HASH_BLOCK, sha256),
    )


def _sched_output(num_tokens: int, preempted: set[str] | None = None):
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"running": num_tokens} if num_tokens else {},
        total_num_scheduled_tokens=num_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        preempted_req_ids=preempted or set(),
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _is_free(block: KVCacheBlock) -> bool:
    return (
        block.ref_cnt == 0
        and block.prev_free_block is not None
        and block.next_free_block is not None
    )


def _group_hash(req: Request, group_id: int, block_size: int, index: int):
    hashes = BlockHashListWithBlockSize(req.block_hashes, HASH_BLOCK, block_size)
    return make_block_hash_with_group_id(hashes[index], group_id)


class ProdHarness:
    """KVCacheManager + SimpleCPUOffloadScheduler at the production shape."""

    def __init__(
        self,
        num_gpu_blocks: int = 128,
        options: SimpleCPUOffloadOptions | None = None,
        retention: int | None = 0,
        cpu_use_eagle: bool = True,
    ):
        self.config = prod_kv_cache_config(num_gpu_blocks)
        self.kvm = KVCacheManager(
            kv_cache_config=self.config,
            max_model_len=MAX_MODEL_LEN,
            hash_block_size=HASH_BLOCK,
            max_in_flight_tokens=2 * MAX_NUM_BATCHED_TOKENS,
            enable_caching=True,
            use_eagle=True,
            prefix_cache_retention_interval=retention,
        )
        self.pool = self.kvm.block_pool
        self.mgr = SimpleCPUOffloadScheduler(
            _manager_config(cpu_use_eagle),
            self.config,
            4 * num_gpu_blocks * SLOT_BYTES,
            scheduler_block_size=MAMBA_BLOCK,
            hash_block_size=HASH_BLOCK,
            lazy_offload=True,
            options=options or SimpleCPUOffloadOptions(lazy_offload=True),
        )
        self.mgr.bind_gpu_block_pool(self.pool)
        self.cpu_pool = self.mgr.cpu_block_pool
        self.inflight_stores: list[int] = []
        # (gpu id, cpu id, hash) of every store emitted so far.
        self.stored: list[tuple[int, int, object]] = []

    # -- one scheduler step ------------------------------------------------
    def step(
        self, num_tokens: int = 1, preempted: set[str] | None = None
    ) -> SimpleCPUOffloadMetadata:
        """build_connector_meta() at the end of schedule(); the stores of the
        previous step complete in this step's update_from_output()."""
        meta = self.mgr.build_connector_meta(_sched_output(num_tokens, preempted))
        for gpu_id, cpu_id in zip(meta.store_gpu_blocks, meta.store_cpu_blocks):
            gpu_hash = self.pool.blocks[gpu_id].block_hash
            assert gpu_hash is not None
            assert self.cpu_pool.blocks[cpu_id].block_hash == gpu_hash
            self.stored.append((gpu_id, cpu_id, gpu_hash))
        self.complete_stores()
        if meta.store_event >= 0:
            self.inflight_stores.append(meta.store_event)
        return meta

    def complete_stores(self) -> None:
        for event in self.inflight_stores:
            self.mgr.update_connector_output(
                KVConnectorOutput(
                    kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(
                        completed_store_events={event: WORLD_SIZE}
                    )
                )
            )
        self.inflight_stores = []

    def drain(self, num_tokens: int = 0, max_steps: int = 200) -> int:
        """Step until the connector has no pending work; returns the steps."""
        for n in range(1, max_steps + 1):
            self.step(num_tokens)
            if not self.mgr.has_pending_work():
                assert not self.inflight_stores
                return n
        raise AssertionError("has_pending_work() never turned False")

    # -- scheduler replay --------------------------------------------------
    def admit(self, req: Request) -> tuple[int, int]:
        """Admit a waiting request like Scheduler.schedule(): returns
        (local GPU hit, CPU hit) in tokens. A CPU hit parks it waiting for
        the load; otherwise its first chunk is scheduled."""
        self.kvm.new_step_starts()
        if req.num_computed_tokens == 0:
            new_computed, num_local = self.kvm.get_computed_blocks(req)
            ext, load_async = self.mgr.get_num_new_matched_tokens(req, num_local)
        else:
            new_computed, num_local = self.kvm.empty_kv_cache_blocks, 0
            ext, load_async = 0, False
        num_computed = req.num_computed_tokens + num_local + ext
        num_new = 0 if load_async else min(MAMBA_BLOCK, req.num_tokens - num_computed)
        blocks = self.kvm.allocate_slots(
            req,
            num_new,
            num_new_computed_tokens=num_local,
            new_computed_blocks=new_computed,
            num_lookahead_tokens=0 if load_async else NUM_SPEC_TOKENS,
            num_external_computed_tokens=ext,
            delay_cache_blocks=load_async,
        )
        assert blocks is not None
        self.mgr.update_state_after_alloc(req, self.kvm.get_blocks(req.request_id), ext)
        req.num_computed_tokens = num_computed + num_new
        return num_local, ext

    def run(self, req: Request) -> None:
        """Schedule the rest of the prompt in 1632-token chunks."""
        while req.num_computed_tokens < req.num_tokens:
            self.step(MAMBA_BLOCK)
            self.kvm.new_step_starts()
            num_new = min(MAMBA_BLOCK, req.num_tokens - req.num_computed_tokens)
            blocks = self.kvm.allocate_slots(
                req, num_new, num_lookahead_tokens=NUM_SPEC_TOKENS
            )
            assert blocks is not None
            self.mgr.update_state_after_alloc(
                req, self.kvm.get_blocks(req.request_id), 0
            )
            req.num_computed_tokens += num_new
        self.step(1)

    def prefill(self, req: Request) -> None:
        assert self.admit(req)[1] == 0
        self.run(req)

    def complete_load(self, req: Request) -> None:
        """Worker reports the load; Scheduler._update_waiting_for_remote_kv."""
        self.mgr.update_connector_output(
            KVConnectorOutput(finished_recving={req.request_id})
        )
        self.kvm.cache_blocks(req, req.num_computed_tokens)
        if req.num_computed_tokens == req.num_tokens:
            req.num_computed_tokens -= 1

    def finish(self, req: Request) -> None:
        """Scheduler._free_request: the connector sees the block ids first."""
        self.kvm.remove_skipped_blocks(
            req.request_id,
            req.num_computed_tokens,
            num_prompt_tokens=req.num_prompt_tokens,
        )
        block_ids = self.kvm.get_block_ids_for_computed_tokens(
            req.request_id, req.num_computed_tokens
        )
        self.mgr.request_finished_all_groups(req, block_ids)
        self.kvm.free(req)

    def store_all(self, req: Request) -> None:
        """Prefill and finish req, then let the idle drain offload it."""
        self.prefill(req)
        self.finish(req)
        self.drain()

    def evict_gpu_copies(self) -> None:
        self.pool.evict_blocks(
            {b.block_id for b in self.pool.blocks if _is_free(b) and b.block_hash}
        )

    def assert_no_leaks(self) -> None:
        assert not self.inflight_stores
        assert self.pool.get_num_free_blocks() == self.pool.num_gpu_blocks - 1
        assert all(b.ref_cnt == 0 for b in self.cpu_pool.blocks if not b.is_null)
        assert not self.mgr._reqs_to_load and not self.mgr._load_event_to_reqs
        assert not self.mgr._pending_cpu_hits
        assert not self.mgr._store_event_to_blocks
        assert self.mgr._num_store_pinned_blocks == 0
        assert not self.mgr._lazy_inflight_hashes


# ---------------------------------------------------------------------------
# T1: groups, lazy target and CPU block count
# ---------------------------------------------------------------------------
def test_prod_shape_groups_target_and_cpu_blocks() -> None:
    config = prod_kv_cache_config(PROD_GPU_BLOCKS)
    specs = [g.kv_cache_spec for g in config.kv_cache_groups]
    assert [type(s) for s in specs] == [
        FullAttentionSpec,
        CircularBufferSpec,
        MambaSpec,
        MambaSpec,
        MambaSpec,
        MambaSpec,
    ]
    assert [s.block_size for s in specs] == [816, 8, 1632, 1632, 1632, 1632]
    assert [s.tp_replicated for s in specs[2:]] == [False, False, False, True]
    assert not specs[RING].prefix_cacheable
    assert config.num_blocks == PROD_GPU_BLOCKS
    assert len(config.kv_cache_tensors) == 26
    assert sum(t.size for t in config.kv_cache_tensors) == PROD_GPU_BLOCKS * SLOT_BYTES
    block_size_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=FA_BLOCK, enable_prefix_caching=True, hash_block_size=None
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1
        ),
        kv_transfer_config=object(),
    )
    assert resolve_kv_cache_block_sizes(config, block_size_config) == (1632, 8)

    # (cdiv(8192, 816) + 4 Mamba groups x 2) x 2; the ring is not counted.
    assert (
        SimpleCPUOffloadScheduler._estimate_lazy_target_blocks(
            config, MAX_NUM_BATCHED_TOKENS
        )
        == 38
    )

    # 32 GiB server-wide = 8 GiB per rank. Scheduler formula
    # (_derive_cpu_config) == worker formula (bytes per block summed over the
    # unique KV storages, i.e. every KVCacheTensor).
    options = parse_options({"cpu_bytes_to_use": 32 * GiB, "lazy_offload": True})
    per_rank = options.cpu_bytes_per_rank(WORLD_SIZE)
    assert per_rank == 8 * GiB
    worker_bytes_per_block = sum(
        t.size // config.num_blocks for t in config.kv_cache_tensors
    )
    assert worker_bytes_per_block == SLOT_BYTES
    derived = SimpleCPUOffloadScheduler._derive_cpu_config(config, per_rank)
    assert derived.num_blocks == per_rank // worker_bytes_per_block == 744

    mgr = SimpleCPUOffloadScheduler(
        _manager_config(True),
        config,
        per_rank,
        scheduler_block_size=MAMBA_BLOCK,
        hash_block_size=HASH_BLOCK,
        lazy_offload=True,
        options=options,
    )
    assert mgr.num_cpu_blocks == 744
    assert mgr._target_free == 38
    assert mgr._store_reserve == MAX_NUM_SEQS * (1 + len(MAMBA_GROUPS)) == 25
    coordinator = mgr.cpu_coordinator
    assert sorted(gid for g in coordinator.attention_groups for gid in g.group_ids) == [
        FA,
        *MAMBA_GROUPS,
    ]
    managers = coordinator.single_type_managers
    assert all(managers[g].use_eagle for g in (FA, *MAMBA_GROUPS))


# ---------------------------------------------------------------------------
# T2: lazy store round trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("retention", [0, None], ids=["replay", "dense"])
def test_lazy_store_roundtrip_prod_shape(retention: int | None) -> None:
    max_per_step = 4
    h = ProdHarness(
        retention=retention,
        options=SimpleCPUOffloadOptions(
            lazy_offload=True, max_store_blocks_per_step=max_per_step
        ),
    )
    req = prod_request()
    h.prefill(req)
    ring_block = h.kvm.get_block_ids(req.request_id)[RING][0]
    h.finish(req)

    hashed = {b.block_id: b.block_hash for b in h.pool.blocks if b.block_hash}
    assert any(get_group_id(bh) in MAMBA_GROUPS for bh in hashed.values())
    stored_before = {gpu_id for gpu_id, _, _ in h.stored}
    remaining = len(hashed) - len(stored_before)
    num_steps = 0
    while True:
        h.step(1)
        num_steps += 1
        cursor = h.mgr._cursor
        assert cursor is not None and _is_free(cursor)
        if not h.mgr.has_pending_work():
            break
        assert num_steps <= math.ceil(remaining / max_per_step) + 2

    stored_ids = [gpu_id for gpu_id, _, _ in h.stored]
    assert len(stored_ids) == len(set(stored_ids)), "each block stored once"
    assert set(stored_ids) == set(hashed)
    assert ring_block not in stored_ids
    for gpu_id, cpu_id, block_hash in h.stored:
        gpu_block = h.pool.blocks[gpu_id]
        assert _is_free(gpu_block) and gpu_block.block_hash == block_hash
        assert get_group_id(block_hash) != RING
        cached = h.cpu_pool.cached_block_hash_to_block.get_one_block(block_hash)
        assert cached is not None and cached.block_id == cpu_id
    h.assert_no_leaks()


def test_idle_drain_prod_shape() -> None:
    h = ProdHarness(retention=None)
    req = prod_request()
    h.prefill(req)
    h.finish(req)
    # The step right after a request finished always runs (finished requests
    # are pending), and the idle drain keeps the engine stepping from there.
    h.step(0)
    num_steps = 1
    while h.mgr.has_pending_work():
        h.step(0)
        num_steps += 1
        assert num_steps < 50
    assert all(
        h.cpu_pool.cached_block_hash_to_block.get_one_block(b.block_hash)
        for b in h.pool.blocks
        if b.block_hash
    )
    for _ in range(3):
        meta = h.step(0)
        assert meta.store_event < 0 and meta.load_event < 0
        assert not h.mgr.has_pending_work()
    h.assert_no_leaks()


# ---------------------------------------------------------------------------
# T5 / P14: CPU hit alignment and the load layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["cpu-only", "suffix-only", "hole"])
def test_cpu_hit_alignment_prod_shape(case: str) -> None:
    h = ProdHarness(num_gpu_blocks=160, retention=None)
    original = prod_request()
    h.store_all(original)

    if case == "suffix-only":
        # Keep the head (FA pages 0-5, Mamba states at 1632 and 3264) on GPU;
        # the rest is only on CPU.
        keep = {_group_hash(original, FA, FA_BLOCK, i) for i in range(6)} | {
            _group_hash(original, g, MAMBA_BLOCK, i)
            for g in MAMBA_GROUPS
            for i in range(2)
        }
        h.pool.evict_blocks(
            {
                b.block_id
                for b in h.pool.blocks
                if _is_free(b) and b.block_hash and b.block_hash not in keep
            }
        )
    else:
        h.evict_gpu_copies()
    hole_tokens = None
    if case == "hole":
        hole_tokens = 6 * FA_BLOCK
        hole_hash = _group_hash(original, FA, FA_BLOCK, 6)
        hole = h.cpu_pool.cached_block_hash_to_block.get_one_block(hole_hash)
        assert hole is not None
        h.cpu_pool.evict_blocks({hole.block_id})

    resend = prod_request(PROMPT_LEN + 2000, prefix=original)
    local, ext = h.admit(resend)
    total = local + ext
    assert ext > 0 and ext % MAMBA_BLOCK == 0 and total % MAMBA_BLOCK == 0
    if case == "suffix-only":
        assert local > 0
    else:
        assert local == 0
    if hole_tokens is not None:
        assert total <= hole_tokens

    meta = h.step(0)
    assert meta.load_event >= 0
    pairs = dict(zip(meta.load_gpu_blocks, meta.load_cpu_blocks))
    ids = h.kvm.get_block_ids(resend.request_id)

    def cpu_hash(gpu_id: int):
        return h.cpu_pool.blocks[pairs[gpu_id]].block_hash

    fa_dst = [bid for bid in meta.load_gpu_blocks if bid in ids[FA]]
    assert fa_dst == ids[FA][local // FA_BLOCK : total // FA_BLOCK]
    for i, gpu_id in enumerate(fa_dst):
        assert cpu_hash(gpu_id) == _group_hash(
            resend, FA, FA_BLOCK, local // FA_BLOCK + i
        )
    assert not set(ids[RING]) & set(meta.load_gpu_blocks)
    for g in MAMBA_GROUPS:
        dst = [bid for bid in meta.load_gpu_blocks if bid in ids[g]]
        assert dst == [ids[g][total // MAMBA_BLOCK - 1]]
        assert cpu_hash(dst[0]) == _group_hash(
            resend, g, MAMBA_BLOCK, total // MAMBA_BLOCK - 1
        )
    assert len(pairs) == ext // FA_BLOCK + len(MAMBA_GROUPS)

    # Load done: the scheduler caches the blocks under the CPU hashes.
    h.complete_load(resend)
    for gpu_id in pairs:
        assert h.pool.blocks[gpu_id].block_hash == cpu_hash(gpu_id)
    h.run(resend)
    h.finish(resend)
    h.drain()
    h.assert_no_leaks()


def test_truncated_external_tokens_raise_prod_shape() -> None:
    h = ProdHarness(retention=None)
    original = prod_request()
    h.store_all(original)
    h.evict_gpu_copies()

    resend = prod_request(prefix=original)
    _, num_local = h.kvm.get_computed_blocks(resend)
    ext, _ = h.mgr.get_num_new_matched_tokens(resend, num_local)
    assert ext >= 2 * MAMBA_BLOCK
    cpu_free = h.cpu_pool.get_num_free_blocks()
    blocks = h.kvm.allocate_slots(
        resend,
        0,
        num_external_computed_tokens=ext - MAMBA_BLOCK,
        delay_cache_blocks=True,
    )
    assert blocks is not None
    with pytest.raises(RuntimeError, match="num_external_tokens"):
        h.mgr.update_state_after_alloc(
            resend, h.kvm.get_blocks(resend.request_id), ext - MAMBA_BLOCK
        )
    assert h.cpu_pool.get_num_free_blocks() > cpu_free
    assert not h.mgr._pending_cpu_hits and not h.mgr._reqs_to_load


@pytest.mark.parametrize("defect", ["mamba-without-state", "ring-blocks"])
def test_invalid_cpu_hit_refused_prod_shape(monkeypatch, defect: str) -> None:
    h = ProdHarness(retention=None)
    original = prod_request()
    h.store_all(original)
    h.evict_gpu_copies()

    resend = prod_request(prefix=original)
    coordinator = h.mgr.cpu_coordinator
    hit_blocks, hit_length = coordinator.find_longest_cache_hit(
        resend.block_hashes, resend.num_tokens - 1
    )
    assert hit_length > 0
    crafted = [list(blocks) for blocks in hit_blocks]
    if defect == "mamba-without-state":
        crafted[GDN[1]][-1] = h.cpu_pool.null_block
    else:
        crafted[RING] = [crafted[FA][0]]
    monkeypatch.setattr(
        coordinator,
        "find_longest_cache_hit",
        lambda block_hashes, max_length: (tuple(crafted), hit_length),
    )
    cpu_free = h.cpu_pool.get_num_free_blocks()
    assert h.mgr.get_num_new_matched_tokens(resend, 0) == (0, False)
    assert h.cpu_pool.get_num_free_blocks() == cpu_free
    assert not h.mgr._pending_cpu_hits
    stats = h.mgr.get_stats()
    assert stats is not None and stats.data[kv_stats.CPU_HITS_REFUSED] == 1

    monkeypatch.setattr(h.mgr, "_test_mode", True)
    with pytest.raises(AssertionError, match="invalid CPU hit"):
        h.mgr.get_num_new_matched_tokens(resend, 0)


# ---------------------------------------------------------------------------
# T6 / P10: MTP last-page drop parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cpu_use_eagle", [True, False])
def test_eagle_parity_cpu_vs_gpu(cpu_use_eagle: bool) -> None:
    h = ProdHarness(retention=None, cpu_use_eagle=cpu_use_eagle)
    original = prod_request()
    h.store_all(original)

    resend = prod_request(prefix=original)
    max_length = resend.num_tokens - 1
    _, gpu_hit = h.kvm.coordinator.find_longest_cache_hit(
        resend.block_hashes, max_length
    )
    _, cpu_hit = h.mgr.cpu_coordinator.find_longest_cache_hit(
        resend.block_hashes, max_length
    )
    assert gpu_hit == 5 * MAMBA_BLOCK
    if cpu_use_eagle:
        assert cpu_hit == gpu_hit
    else:
        # Without the MTP drop the CPU hit covers one more Mamba page than
        # the GPU would ever use.
        assert cpu_hit == gpu_hit + MAMBA_BLOCK


# ---------------------------------------------------------------------------
# T8 / T9 / T10: in-flight loads under preemption, abort and page reuse
# ---------------------------------------------------------------------------
def _start_load(h: ProdHarness, original: Request):
    resend = prod_request(prefix=original)
    local, ext = h.admit(resend)
    assert local == 0 and ext > 0
    meta = h.step(1)
    assert meta.load_event >= 0
    return resend, meta


def test_preemption_during_inflight_load() -> None:
    h = ProdHarness(retention=None)
    original = prod_request()
    h.store_all(original)
    h.evict_gpu_copies()

    running = prod_request(3 * MAMBA_BLOCK + 5)
    h.prefill(running)
    loading, meta = _start_load(h, original)
    dst = list(meta.load_gpu_blocks)
    for gpu_id in dst:
        block = h.pool.blocks[gpu_id]
        assert block.ref_cnt >= 2 and block.prev_free_block is None

    # KV pressure preempts the running request (Scheduler._preempt_request).
    running_ids = {b for ids in h.kvm.get_block_ids(running.request_id) for b in ids}
    h.kvm.free(running)
    running.num_computed_tokens = 0
    assert not running_ids & set(dst)
    meta = h.step(1, preempted={running.request_id})
    assert meta.need_flush
    for gpu_id in dst:
        assert h.pool.blocks[gpu_id].ref_cnt >= 2

    h.complete_load(loading)
    for gpu_id in dst:
        block = h.pool.blocks[gpu_id]
        assert block.ref_cnt == 1
        cached = h.cpu_pool.cached_block_hash_to_block.get_one_block(block.block_hash)
        assert cached is not None
    h.run(loading)

    # The preempted request resumes (possibly from its own CPU copy).
    local, ext = h.admit(running)
    if ext:
        h.step(1)
        h.complete_load(running)
    h.run(running)
    h.finish(running)
    h.finish(loading)
    h.drain()
    h.assert_no_leaks()


def test_abort_during_inflight_load() -> None:
    h = ProdHarness(retention=None)
    original = prod_request()
    h.store_all(original)
    h.evict_gpu_copies()
    loading, meta = _start_load(h, original)
    dst = list(meta.load_gpu_blocks)
    cpu_src = list(meta.load_cpu_blocks)

    # Aborted while WAITING_FOR_REMOTE_KVS: the scheduler tells the connector
    # but delays freeing the blocks until the load is reported.
    block_ids = h.kvm.get_block_ids_for_computed_tokens(
        loading.request_id, loading.num_computed_tokens
    )
    h.mgr.request_finished_all_groups(loading, block_ids)
    assert h.mgr._reqs_to_load[loading.request_id].finished
    h.step(1)
    assert all(h.pool.blocks[b].ref_cnt == 2 for b in dst)
    assert all(h.cpu_pool.blocks[b].ref_cnt == 1 for b in cpu_src)

    h.mgr.update_connector_output(
        KVConnectorOutput(finished_recving={loading.request_id})
    )
    assert all(h.pool.blocks[b].ref_cnt == 1 for b in dst)
    assert all(h.cpu_pool.blocks[b].ref_cnt == 0 for b in cpu_src)
    h.kvm.free(loading)
    h.drain()
    h.assert_no_leaks()


def test_finished_page_reuse_counter() -> None:
    """Async scheduling: F finishes in update_from_output() while the step
    after it is still on the GPU, and the next schedule() admits a CPU hit
    whose load destinations start with F's freed scratch pages (the exposure
    the load-after-compute barrier covers)."""
    h = ProdHarness(retention=None)
    original = prod_request()
    h.store_all(original)
    h.evict_gpu_copies()

    finished = prod_request(2 * MAMBA_BLOCK + 100)
    h.prefill(finished)
    finished_ids = {b for ids in h.kvm.get_block_ids(finished.request_id) for b in ids}
    unhashed = {b for b in finished_ids if b and h.pool.blocks[b].block_hash is None}
    h.step(1)  # S_k, still in flight when F's output is processed
    h.finish(finished)

    loading = prod_request(prefix=original)
    _, ext = h.admit(loading)
    assert ext > 0
    meta = h.step(1)
    fa_dst = [
        b
        for b in meta.load_gpu_blocks
        if b in h.kvm.get_block_ids(loading.request_id)[FA]
    ]
    assert fa_dst[0] in unhashed
    stats = h.mgr.get_stats()
    assert stats is not None
    assert stats.data[kv_stats.LOAD_DST_RECENTLY_FREED] > 0
    assert stats.data[kv_stats.LOAD_REQUESTS] == 1
