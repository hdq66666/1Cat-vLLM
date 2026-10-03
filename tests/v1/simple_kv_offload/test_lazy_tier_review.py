# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy CPU tier regressions found in review: the prefix-cache bypass also
bypasses the CPU tier, and the no-re-store memo is scoped to one GPU
residency of a block."""

from __future__ import annotations

import itertools

import torch

from tests.v1.simple_kv_offload.test_prod_shape import (
    HASH_BLOCK,
    ProdHarness,
    prod_request,
)
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
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadWorkerMetadata
from vllm.v1.simple_kv_offload.options import SimpleCPUOffloadOptions

init_none_hash(sha256)

_ids = itertools.count()


def _logprob_request(prefix: Request) -> Request:
    return Request(
        request_id=f"plp-{next(_ids)}",
        prompt_token_ids=list(prefix.prompt_token_ids),
        sampling_params=SamplingParams(max_tokens=1, prompt_logprobs=1),
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(HASH_BLOCK, sha256),
    )


def test_prompt_logprobs_bypasses_cpu_tier_prod_shape() -> None:
    h = ProdHarness(retention=None)
    original = prod_request()
    h.store_all(original)  # prefill, finish, idle drain; GPU copies kept

    # The GPU still holds the prefix: a plain resend would hit on GPU.
    plain = prod_request(prefix=original)
    _, gpu_hit_plain = h.kvm.get_computed_blocks(plain)
    assert gpu_hit_plain > 0

    plp = _logprob_request(original)
    assert plp.skip_reading_prefix_cache
    local, ext = h.admit(plp)
    # Let the load run and finish the request (no leak either way).
    if ext:
        meta = h.step(1)
        assert meta.load_event >= 0
        h.complete_load(plp)
    h.run(plp)
    h.finish(plp)
    h.drain()
    h.assert_no_leaks()

    # Correct behaviour: the prefix-cache bypass also bypasses the CPU tier.
    assert local == 0
    assert ext == 0, f"CPU tier served {ext} tokens to a prompt_logprobs request"


# The natural allocator only (no touch / evict_blocks shortcuts): the fork
# prepends freed unhashed blocks to the free-queue head, so a block that served
# as scratch is the next allocation's first block.
BLOCK = 16
_BYTES_PER_BLOCK = BLOCK * 1 * 16 * 2 * torch.float16.itemsize


def _vllm_config() -> VllmConfig:
    return VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=64,
            max_model_len=10000,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
        ),
        model_config=ModelConfig(
            model="facebook/opt-125m",
            trust_remote_code=True,
            dtype="float16",
            seed=42,
        ),
        cache_config=CacheConfig(
            block_size=BLOCK, gpu_memory_utilization=0.9, enable_prefix_caching=True
        ),
        kv_transfer_config=KVTransferConfig(
            kv_connector="SimpleCPUOffloadConnector", kv_role="kv_both"
        ),
        device_config=DeviceConfig("cpu"),
    )


def _lazy(num_gpu_blocks: int, num_cpu_blocks: int):
    kv_cache_config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=_BYTES_PER_BLOCK * num_gpu_blocks, shared_by=["l0"])
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["l0"],
                FullAttentionSpec(
                    block_size=BLOCK,
                    num_kv_heads=1,
                    head_size=16,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    sched = SimpleCPUOffloadScheduler(
        vllm_config=_vllm_config(),
        kv_cache_config=kv_cache_config,
        cpu_capacity_bytes=_BYTES_PER_BLOCK * num_cpu_blocks,
        scheduler_block_size=BLOCK,
        hash_block_size=BLOCK,
        lazy_offload=True,
        options=SimpleCPUOffloadOptions(lazy_offload=True, store_reserve_free_blocks=0),
    )
    gpu_pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks, enable_caching=True, hash_block_size=BLOCK
    )
    sched.bind_gpu_block_pool(gpu_pool)
    return sched, gpu_pool


def _req(tokens: list[int]) -> Request:
    return Request(
        request_id=f"memo-{next(_ids)}",
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _sched_out(num_tokens: int) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"running": num_tokens} if num_tokens else {},
        total_num_scheduled_tokens=num_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        preempted_req_ids=set(),
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _drain(sched, num_tokens: int = 1, max_steps: int = 50) -> list[int]:
    stored: list[int] = []
    inflight: list[int] = []
    for _ in range(max_steps):
        meta = sched.build_connector_meta(_sched_out(num_tokens))
        for ev in inflight:
            sched.update_connector_output(
                KVConnectorOutput(
                    kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(
                        completed_store_events={ev: sched._expected_worker_count}
                    )
                )
            )
        inflight = []
        if meta.store_event >= 0:
            stored.extend(meta.store_gpu_blocks)
            inflight.append(meta.store_event)
        if not sched.has_pending_work() and not inflight:
            return stored
    raise AssertionError("has_pending_work() never turned False")


def _run(gpu_pool: BlockPool, req: Request) -> list:
    """Allocate like allocate_slots (pop from the free-queue head), cache
    the full blocks, finish (free in reverse order like the scheduler)."""
    n_full = len(req.prompt_token_ids) // BLOCK
    n_total = -(-len(req.prompt_token_ids) // BLOCK)
    blocks = gpu_pool.get_new_blocks(n_total)
    if n_full:
        gpu_pool.cache_full_blocks(
            request=req,
            blocks=blocks,
            num_cached_blocks=0,
            num_full_blocks=n_full,
            block_size=BLOCK,
            kv_cache_group_id=0,
        )
    gpu_pool.free_blocks(reversed(blocks))
    return blocks


def _on_cpu(sched, bhash) -> bool:
    return (
        sched.cpu_block_pool.cached_block_hash_to_block.get_one_block(bhash) is not None
    )


def _on_gpu(gpu_pool, bhash) -> bool:
    return gpu_pool.cached_block_hash_to_block.get_one_block(bhash) is not None


def test_memo_is_scoped_to_one_residency() -> None:
    # 3 usable GPU blocks, 2 usable CPU slots (block 0 is null in both).
    sched, gpu = _lazy(num_gpu_blocks=4, num_cpu_blocks=3)
    assert sched.cpu_block_pool.get_num_free_blocks() == 2
    tok_a = list(range(1000, 1000 + BLOCK))

    # 1. A computes one full block and finishes; the walk offloads it.
    (g,) = _run(gpu, a := _req(tok_a))
    h_a = g.block_hash
    assert _drain(sched) == [g.block_id] and _on_cpu(sched, h_a)

    # 2. Two other single-block prompts are offloaded; the CPU LRU evicts
    #    A's copy (the walk's own cpu_pool.get_new_blocks does it).
    for base in (2000, 3000):
        _run(gpu, _req(list(range(base, base + BLOCK))))
        _drain(sched)
    assert not _on_cpu(sched, h_a), "CPU LRU should have evicted A"
    assert g.block_hash == h_a and _on_gpu(gpu, h_a)

    # 3. A short (partial-block, never hashed) request takes the LRU head,
    #    which is g: A's GPU copy is gone. g is freed unhashed and the fork
    #    prepends it to the free-queue head.
    (s,) = _run(gpu, _req([9] * 5))
    assert s is g and g.block_hash is None and not _on_gpu(gpu, h_a)
    assert gpu.free_block_queue.fake_free_list_head.next_free_block is g
    _drain(sched)

    # 4. A's prompt returns: misses both tiers, is recomputed, and the
    #    allocator hands out g (the head) for its first block.
    (g2,) = _run(gpu, _req(list(tok_a)))
    assert g2 is g and g.block_hash == h_a
    assert not _on_cpu(sched, h_a)

    # 5. Busy steps: the new residency is offloaded again.
    assert g.block_id in _drain(sched, num_tokens=1)
    assert _on_cpu(sched, h_a), "recomputed block never offloaded (stale memo)"
