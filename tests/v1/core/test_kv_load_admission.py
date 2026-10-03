# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A CPU-tier hit is admitted only when the request can go on after the load.

Drives the real Scheduler with SimpleCPUOffloadConnector (lazy CPU tier) on
the hybrid layout of test_kv_load_skip_zero (full attention + QSA compressor
ring + Mamba ``align``). The pool is sized so that the hit itself fits but the
request's first compute step after the load does not while other requests
decode. Without an admission guard the request is parked in WAITING holding
every loaded block and the decoders preempt each other (the production W8
signature); with it, the hit is deferred until the pool has room.
"""

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
from vllm.v1.simple_kv_offload import stats as kv_stats
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

FA_BLOCK = 16
RING_BLOCK = 4
MAMBA_BLOCK = 32
# Three Mamba boundary states plus a partial block: the CPU tier serves
# 3 * MAMBA_BLOCK tokens of a repeated prompt, 5 tokens stay to compute.
PROMPT_LEN = 3 * MAMBA_BLOCK + 5
CPU_HIT_TOKENS = 3 * MAMBA_BLOCK
DECODER_PROMPT_LEN = 20

init_none_hash(sha256)


def _make_kv_cache_config(num_gpu_blocks: int) -> KVCacheConfig:
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
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=spec.page_size_bytes * num_gpu_blocks, shared_by=[name])
            for name, spec in specs.items()
        ],
        kv_cache_groups=[
            KVCacheGroupSpec([name], spec) for name, spec in specs.items()
        ],
    )


def make_scheduler(num_gpu_blocks: int, **extra_config) -> Scheduler:
    kv_cache_config = _make_kv_cache_config(num_gpu_blocks)
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
            kv_connector_extra_config={
                # Twice the GPU pool, so nothing is evicted from the CPU tier.
                "cpu_bytes_to_use": 2 * gpu_bytes,
                "lazy_offload": True,
                "store_reserve_free_blocks": 0,
                "max_store_blocks_per_step": num_gpu_blocks,
                "min_host_mem_available_gib": 0,
                **extra_config,
            },
        ),
        device_config=DeviceConfig("cpu"),
    )
    vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
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


def make_request(
    request_id: str, prompt_token_ids: list[int], max_tokens: int = 1
) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(RING_BLOCK, sha256),
    )


def step(
    scheduler: Scheduler, finished_recving: set[str] | None = None
) -> tuple[SchedulerOutput, SimpleCPUOffloadMetadata]:
    """One engine step with an instant worker: this step's store completes,
    partial prefills sample nothing and everything else samples one token."""
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
                kv_connector_worker_meta=worker_meta,
            ),
        ),
    )
    return output, meta


def held_blocks(scheduler: Scheduler, req_id: str) -> int:
    """GPU blocks allocated to a request (null blocks excluded)."""
    coordinator = scheduler.kv_cache_manager.coordinator
    return sum(
        not block.is_null
        for manager in coordinator.single_type_managers
        for block in manager.req_to_blocks.get(req_id, ())
    )


def free_blocks(scheduler: Scheduler) -> int:
    return scheduler.kv_cache_manager.block_pool.get_num_free_blocks()


PROMPT = list(range(1000, 1000 + PROMPT_LEN))
PROMPT_2 = list(range(3000, 3000 + PROMPT_LEN))


def seed_cpu_tier(scheduler: Scheduler, prompt: list[int] = PROMPT) -> None:
    """Run the prompt once, let the lazy walk store it, drop the GPU copy."""
    req_id = f"seed-{prompt[0]}"
    scheduler.add_request(make_request(req_id, prompt))
    for _ in range(PROMPT_LEN):
        step(scheduler)
        if req_id not in scheduler.requests:
            break
    assert req_id not in scheduler.requests
    for _ in range(4):
        step(scheduler)
    assert scheduler.kv_cache_manager.reset_prefix_cache()
    manager = scheduler.connector.scheduler_manager
    repeat = make_request("probe", prompt)
    _, hit_tokens = manager.cpu_coordinator.find_longest_cache_hit(
        repeat.block_hashes, repeat.num_tokens - 1
    )
    assert hit_tokens == CPU_HIT_TOKENS, "the CPU tier did not get the prefix"


def guard_stat(scheduler: Scheduler, key: str) -> float:
    return scheduler.connector.scheduler_manager._stats.data.get(key, 0)


def fake_clock(scheduler: Scheduler) -> list[float]:
    clock = [1000.0]
    scheduler.connector.scheduler_manager._clock = lambda: clock[0]
    return clock


def start_decoders(scheduler: Scheduler, n: int, max_tokens: int) -> list[Request]:
    decoders = []
    for i in range(n):
        prompt = list(range(5000 + 100 * i, 5000 + 100 * i + DECODER_PROMPT_LEN))
        decoder = make_request(f"D{i}", prompt, max_tokens=max_tokens)
        scheduler.add_request(decoder)
        decoders.append(decoder)
    step(scheduler)
    assert all(d.status == RequestStatus.RUNNING for d in decoders)
    return decoders


def post_load_need(scheduler: Scheduler, req_id: str) -> int:
    """Blocks the first compute step after the load allocates, by the
    allocator's own estimator (no lookahead: the test has no spec decode)."""
    request = scheduler.requests[req_id]
    coordinator = scheduler.kv_cache_manager.coordinator
    return coordinator.get_num_blocks_to_allocate(
        request_id=req_id,
        num_tokens=request.num_tokens,
        new_computed_blocks=tuple([] for _ in coordinator.single_type_managers),
        num_encoder_tokens=0,
        total_computed_tokens=request.num_computed_tokens,
        num_tokens_main_model=request.num_tokens,
    )


def w8_pool_size(num_decoders: int) -> int:
    """Pool where the repeat's hit fits next to the decoders but its first
    compute step after the load does not (the production W8 shape).

    Measured on a roomy pool rather than derived by hand: the Mamba align
    manager's block accounting differs between the hit step and the step
    after it."""
    scheduler = make_scheduler(64)
    seed_cpu_tier(scheduler)
    decoders = start_decoders(scheduler, num_decoders, max_tokens=200)
    admit_repeat(scheduler)
    step(scheduler)
    held_by_decoders = sum(held_blocks(scheduler, d.request_id) for d in decoders)
    held_by_hit = held_blocks(scheduler, "B")
    need_next = post_load_need(scheduler, "B")
    assert need_next >= 2, need_next
    # Null block + what the decoders and the hit hold + one block short of
    # the step after the load.
    return 1 + held_by_decoders + held_by_hit + need_next - 1


def admit_repeat(
    scheduler: Scheduler, req_id: str = "B", prompt: list[int] = PROMPT
) -> Request:
    repeat = make_request(req_id, prompt)
    scheduler.add_request(repeat)
    return repeat


def test_w8_signature_without_guard(monkeypatch) -> None:
    """Baseline: a hit admitted with no room for the next step parks the
    request on its loaded blocks and the decoders preempt each other."""
    scheduler = make_scheduler(w8_pool_size(num_decoders=2))
    manager = scheduler.connector.scheduler_manager
    monkeypatch.setattr(manager, "_check_admission", lambda *a, **k: (None, True))
    seed_cpu_tier(scheduler)
    decoders = start_decoders(scheduler, 2, max_tokens=200)
    repeat = admit_repeat(scheduler)

    output, meta = step(scheduler)
    assert repeat.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert repeat.num_computed_tokens == CPU_HIT_TOKENS
    assert meta.load_gpu_blocks
    free_after_hit = free_blocks(scheduler)

    # The load completes; the next schedule() promotes the request to
    # WAITING but its 5 remaining tokens cannot be allocated: it keeps every
    # loaded block.
    step(scheduler, finished_recving={"B"})
    output, _ = step(scheduler)
    assert repeat.status == RequestStatus.WAITING
    assert "B" not in output.num_scheduled_tokens
    assert held_blocks(scheduler, "B") >= len(meta.load_gpu_blocks)

    assert free_after_hit < post_load_need(scheduler, "B")

    # The request only moves once a decoder is preempted for it: it waits on
    # its loaded blocks until a decoder needs a block of its own, and the
    # decoder then loses its place.
    first_scheduled = None
    for i in range(40):
        output, _ = step(scheduler)
        if "B" in output.num_scheduled_tokens:
            first_scheduled = i
            break
        assert held_blocks(scheduler, "B") >= len(meta.load_gpu_blocks)
    assert first_scheduled is not None and first_scheduled >= 5
    assert sum(d.num_preemptions for d in decoders) > 0


def test_hit_is_deferred_until_the_pool_has_room() -> None:
    scheduler = make_scheduler(w8_pool_size(num_decoders=2))
    manager = scheduler.connector.scheduler_manager
    clock = fake_clock(scheduler)
    seed_cpu_tier(scheduler)
    decoders = start_decoders(scheduler, 2, max_tokens=30)
    repeat = admit_repeat(scheduler)

    # Deferred: not admitted, nothing allocated, nothing pinned on CPU.
    cpu_free = manager.cpu_block_pool.get_num_free_blocks()
    output, meta = step(scheduler)
    assert repeat.status == RequestStatus.WAITING
    assert "B" not in output.num_scheduled_tokens
    assert held_blocks(scheduler, "B") == 0
    assert not meta.load_gpu_blocks
    assert manager.cpu_block_pool.get_num_free_blocks() == cpu_free
    assert not manager._pending_cpu_hits
    assert "B" in manager._deferred_hits
    assert guard_stat(scheduler, kv_stats.CPU_HITS_DEFERRED) == 1

    # The decoders run undisturbed; the hit is admitted once they are done.
    admitted_at = None
    load_done = False
    for i in range(120):
        clock[0] += 0.05
        finished_recving = set()
        if repeat.status == RequestStatus.WAITING_FOR_REMOTE_KVS and not load_done:
            finished_recving, load_done = {"B"}, True
        output, meta = step(scheduler, finished_recving=finished_recving)
        if meta.load_gpu_blocks and admitted_at is None:
            admitted_at = i
        if admitted_at is None:
            assert held_blocks(scheduler, "B") == 0
        if repeat.is_finished():
            break
    assert admitted_at is not None
    assert repeat.is_finished()
    assert sum(d.num_preemptions for d in decoders) == 0
    assert guard_stat(scheduler, kv_stats.LOAD_REQUESTS) == 1
    assert guard_stat(scheduler, kv_stats.CPU_HITS_DEFERRED) >= 2
    assert guard_stat(scheduler, kv_stats.CPU_HITS_COLD_FALLBACK) == 0
    assert "B" not in manager._deferred_hits


def test_hit_falls_back_to_cold_after_the_wait() -> None:
    scheduler = make_scheduler(
        w8_pool_size(num_decoders=2), cpu_hit_admission_wait_s=2.0
    )
    manager = scheduler.connector.scheduler_manager
    clock = fake_clock(scheduler)
    seed_cpu_tier(scheduler)
    decoders = start_decoders(scheduler, 2, max_tokens=30)
    repeat = admit_repeat(scheduler)

    step(scheduler)
    assert "B" in manager._deferred_hits
    clock[0] += 1.0
    step(scheduler)
    assert repeat.status == RequestStatus.WAITING
    assert held_blocks(scheduler, "B") == 0

    # The wait is over: the request is prefilled from scratch, in a cold
    # request's position (it fits now because it takes no load-time blocks).
    clock[0] += 1.0
    output, meta = step(scheduler)
    assert "B" in manager._cold_fallback_req_ids
    assert "B" not in manager._deferred_hits
    assert "B" in output.num_scheduled_tokens
    assert repeat.status == RequestStatus.RUNNING
    assert 0 < repeat.num_computed_tokens <= PROMPT_LEN
    assert not meta.load_gpu_blocks
    assert guard_stat(scheduler, kv_stats.CPU_HITS_COLD_FALLBACK) == 1
    assert guard_stat(scheduler, kv_stats.CPU_HITS_DEFERRED) == 3
    assert guard_stat(scheduler, kv_stats.CPU_HIT_WAIT_S_MAX) == 2.0
    assert guard_stat(scheduler, kv_stats.LOAD_REQUESTS) == 0

    # It goes on as a cold request would, without ever loading from CPU.
    for _ in range(2 * PROMPT_LEN):
        step(scheduler)
        if repeat.is_finished():
            break
    assert repeat.is_finished()
    assert guard_stat(scheduler, kv_stats.LOAD_REQUESTS) == 0
    assert "B" not in manager._cold_fallback_req_ids
    assert sum(d.num_preemptions for d in decoders) == 0


def test_hit_is_kept_while_a_cold_start_would_not_fit_either() -> None:
    """Giving the hit up only helps when a cold start would be admitted now;
    until then the request keeps waiting for its hit (up to the hard cap)."""
    pool = w8_pool_size(num_decoders=2)
    scheduler = make_scheduler(
        pool, cpu_hit_admission_wait_s=2.0, cpu_hit_admission_max_wait_s=6.0
    )
    manager = scheduler.connector.scheduler_manager
    clock = fake_clock(scheduler)
    seed_cpu_tier(scheduler)
    # Three decoders instead of two: the pool is short for a cold start too.
    decoders = start_decoders(scheduler, 3, max_tokens=200)
    repeat = admit_repeat(scheduler)
    step(scheduler)
    numbers, admitted = manager._check_admission(repeat, CPU_HIT_TOKENS, 0)
    assert not admitted and numbers is not None
    free, need_hit, need_next, margin, need_cold = numbers
    assert free < need_cold and free < need_hit + need_next + margin

    # Past the wait: still deferred, because a cold start would not fit.
    for _ in range(3):
        clock[0] += 1.0
        step(scheduler)
    assert "B" in manager._deferred_hits
    assert "B" not in manager._cold_fallback_req_ids
    assert repeat.status == RequestStatus.WAITING
    assert held_blocks(scheduler, "B") == 0

    # Past the hard cap: given up regardless.
    clock[0] += 3.0
    step(scheduler)
    assert "B" in manager._cold_fallback_req_ids
    assert guard_stat(scheduler, kv_stats.CPU_HITS_COLD_FALLBACK) == 1
    assert sum(d.num_preemptions for d in decoders) == 0


def test_hit_is_admitted_at_once_when_the_pool_has_room() -> None:
    # One more block than the step after the load needs, plus one per
    # running decoder.
    scheduler = make_scheduler(w8_pool_size(num_decoders=2) + 1 + 2)
    manager = scheduler.connector.scheduler_manager
    seed_cpu_tier(scheduler)
    start_decoders(scheduler, 2, max_tokens=30)
    repeat = admit_repeat(scheduler)

    output, meta = step(scheduler)
    assert repeat.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert meta.load_gpu_blocks
    assert guard_stat(scheduler, kv_stats.CPU_HITS_DEFERRED) == 0
    # The reserved estimate covers what the allocator will really take.
    assert manager._reqs_to_load["B"].need_next >= post_load_need(scheduler, "B")

    step(scheduler, finished_recving={"B"})
    output, _ = step(scheduler)
    assert "B" in output.num_scheduled_tokens


def test_in_flight_load_reserves_its_next_step() -> None:
    scheduler = make_scheduler(w8_pool_size(num_decoders=2) + 1 + 2)
    manager = scheduler.connector.scheduler_manager
    seed_cpu_tier(scheduler)
    seed_cpu_tier(scheduler, PROMPT_2)
    start_decoders(scheduler, 2, max_tokens=30)
    admit_repeat(scheduler)
    step(scheduler)
    reserved = manager._reqs_to_load["B"].need_next
    assert reserved > 0

    # A second hit sees the pool minus what B's first step after the load
    # will take, and is deferred for exactly that.
    other = make_request("C", PROMPT_2)
    free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
    numbers, admitted = manager._check_admission(other, CPU_HIT_TOKENS, 0)
    assert not admitted
    assert numbers is not None and numbers[0] == free - reserved


def test_abort_while_deferred_cleans_up() -> None:
    scheduler = make_scheduler(
        w8_pool_size(num_decoders=2), cpu_hit_admission_wait_s=1.0
    )
    manager = scheduler.connector.scheduler_manager
    clock = fake_clock(scheduler)
    seed_cpu_tier(scheduler)
    start_decoders(scheduler, 2, max_tokens=200)

    admit_repeat(scheduler, "B")
    step(scheduler)
    assert "B" in manager._deferred_hits
    scheduler.finish_requests("B", RequestStatus.FINISHED_ABORTED)
    step(scheduler)
    assert "B" not in manager._deferred_hits
    assert "B" not in scheduler.requests

    admit_repeat(scheduler, "B2")
    step(scheduler)
    clock[0] += 1.0
    step(scheduler)
    assert "B2" in manager._cold_fallback_req_ids
    scheduler.finish_requests("B2", RequestStatus.FINISHED_ABORTED)
    step(scheduler)
    assert "B2" not in manager._cold_fallback_req_ids
    assert not manager._deferred_hits


def test_without_a_bound_kv_cache_manager_hits_are_not_guarded() -> None:
    scheduler = make_scheduler(w8_pool_size(num_decoders=2))
    manager = scheduler.connector.scheduler_manager
    manager.bind_kv_cache_manager(None)
    seed_cpu_tier(scheduler)
    start_decoders(scheduler, 2, max_tokens=30)
    repeat = admit_repeat(scheduler)
    step(scheduler)
    assert repeat.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert guard_stat(scheduler, kv_stats.CPU_HITS_DEFERRED) == 0
