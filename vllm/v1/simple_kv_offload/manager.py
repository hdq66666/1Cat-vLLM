# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for SimpleCPUOffloadConnector."""

import contextlib
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.host_mem import GiB, read_mem_available_bytes
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.options import SimpleCPUOffloadOptions
from vllm.v1.simple_kv_offload.stats import (
    CPU_CACHED_BLOCKS_LAST,
    CPU_HITS_REFUSED,
    CPU_USED_BLOCKS_LAST,
    HOST_MEM_AVAILABLE_GIB_MIN,
    INFLIGHT_SKIPPED,
    LOAD_DST_RECENTLY_FREED,
    LOAD_REQUESTS,
    RESERVE_STOPS,
    RESTORE_SKIPPED,
    STORE_PINS_LAST,
    STORES_SELECTED,
    SimpleCPUOffloadStats,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId, KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Lazy mode: steps that schedule no tokens (idle drain) select this many
# times max_store_blocks_per_step.
IDLE_STORE_MULTIPLIER = 8
# Seconds between two host MemAvailable samples.
HOST_MEM_CHECK_INTERVAL_S = 60.0
# Seconds between two ERROR lines about refused CPU hits.
REFUSED_HIT_LOG_INTERVAL_S = 60.0
# A load destination freed by a request that finished within this many
# scheduler steps counts as LOAD_DST_RECENTLY_FREED.
RECENTLY_FREED_STEPS = 2


@dataclass
class TransferMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]


@dataclass
class LoadRequestState:
    request: "Request"
    transfer_meta: TransferMeta
    load_event: int | None = None
    finished: bool = False
    # For the admission / completion log lines.
    num_local_hit_tokens: int = 0
    num_cpu_hit_tokens: int = 0
    emit_step: int = 0
    emit_time: float = 0.0


@dataclass
class PendingCPUHit:
    """A CPU hit reported by get_num_new_matched_tokens(), pinned via touch()
    until update_state_after_alloc() consumes it."""

    blocks: tuple[list["KVCacheBlock"], ...]
    num_tokens: int
    # GPU-local prefix hit the CPU hit continues from.
    num_local_tokens: int


# NOTE: This per-request state is only used in eager mode.
@dataclass
class StoreRequestState:
    request: "Request"
    # Accumulated block IDs from scheduler_output via yield_req_data.
    block_ids: tuple[list[int], ...]
    # Per-group cursors tracking how many blocks have been stored/skipped.
    num_stored_blocks: list[int]
    store_events: set[int] = field(default_factory=set)
    finished: bool = False


class SimpleCPUOffloadScheduler:
    """Scheduler-side manager for CPU offloading."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        scheduler_block_size: int,
        hash_block_size: int,
        lazy_offload: bool = False,
        options: SimpleCPUOffloadOptions | None = None,
    ):
        self.vllm_config = vllm_config
        self.options = options or SimpleCPUOffloadOptions(lazy_offload=lazy_offload)
        self.kv_cache_config = kv_cache_config
        self.enable_kv_cache_events = (
            vllm_config.kv_events_config is not None
            and vllm_config.kv_events_config.enable_kv_cache_events
        )
        self.block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
        assert self.block_size % self.hash_block_size == 0
        # Derive a CPU KVCacheConfig from the GPU config and build a coordinator
        assert kv_cache_config is not None
        self.cpu_kv_cache_config = self._derive_cpu_config(
            kv_cache_config, cpu_capacity_bytes
        )
        self.num_cpu_blocks = self.cpu_kv_cache_config.num_blocks

        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d CPU blocks (%.2f GB, mode=%s)",
            self.num_cpu_blocks,
            cpu_capacity_bytes / (1024**3),
            "lazy" if lazy_offload else "eager",
        )

        # TODO (yifan): maybe need to enable kv_cache_events and metrics_collector here.
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        assert dcp_world_size == 1 and pcp_world_size == 1
        # Same EAGLE/MTP last-block drop as the GPU coordinator, so a CPU hit
        # never covers a page the GPU side would recompute.
        speculative_config = vllm_config.speculative_config
        use_eagle = (
            speculative_config is not None and speculative_config.use_eagle_kv_cache()
        )
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=self.cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            use_eagle=use_eagle,
            enable_caching=True,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=self.hash_block_size,
        )
        self.cpu_block_pool: BlockPool = self.cpu_coordinator.block_pool

        # GPU block pool reference - bound after scheduler builds kv_cache_manager
        self._gpu_block_pool: BlockPool | None = None

        # Load metadata
        self._reqs_to_load: dict[str, LoadRequestState] = {}
        # Inverse map: load_event_idx -> req_ids. Keyed by load_event_idx because
        # the worker reports completions by event index, not request id.
        self._load_event_to_reqs: dict[int, list[str]] = {}

        # CPU hits found by get_num_new_matched_tokens(), kept pinned via
        # touch() while awaiting update_state_after_alloc().
        self._pending_cpu_hits: dict[str, PendingCPUHit] = {}

        # Store metadata
        self._lazy_mode = lazy_offload
        # Lazy mode: use a cursor to track the last scanned block in the GPU free queue.
        self._cursor: KVCacheBlock | None = None
        if self._lazy_mode:
            self._target_free = self._estimate_lazy_target_blocks(
                kv_cache_config,
                vllm_config.scheduler_config.max_num_batched_tokens,
            )
        else:
            self._target_free = 0
        # Lazy mode: never let store pins take the GPU free pool down to
        # fewer blocks than running requests need to make progress.
        if self.options.store_reserve_free_blocks is not None:
            self._store_reserve = self.options.store_reserve_free_blocks
        else:
            num_mamba_groups = sum(
                isinstance(g.kv_cache_spec, MambaSpec)
                for g in kv_cache_config.kv_cache_groups
            )
            self._store_reserve = vllm_config.scheduler_config.max_num_seqs * (
                1 + num_mamba_groups
            )
        # Lazy mode: hashes selected for store whose copy has not completed.
        self._lazy_inflight_hashes: set[BlockHashWithGroupId] = set()
        # Lazy mode: the last walk stopped before the end of the free queue
        # with blocks left to store (keeps an idle engine stepping).
        self._lazy_backlog = False
        # GPU block id -> (hash, allocation generation) of the content it
        # holds that was already written to (or read from) CPU. An entry only
        # matches within the residency it was recorded in, even if a later
        # residency recomputes the same content (same hash).
        self._gpu_offloaded: dict[int, tuple[BlockHashWithGroupId, int]] = {}
        self._store_event_to_blocks: dict[int, TransferMeta] = {}
        # GPU blocks pinned by in-flight stores.
        self._num_store_pinned_blocks = 0
        # Eager mode only
        self._reqs_to_store: dict[str, StoreRequestState] = {}
        self._store_event_to_reqs: dict[int, list[str]] = {}
        self._in_flight_store_gpu_blocks: set[int] = set()

        # Event counters
        self._load_event_counter: int = 0
        self._store_event_counter: int = 0

        # For TP/PP: track partial store completions across steps.
        # Events must be reported by all world_size workers before considered complete.
        self._expected_worker_count = vllm_config.parallel_config.world_size
        self._store_event_pending_counts: dict[int, int] = {}

        # Number of build_connector_meta() calls, i.e. scheduler steps.
        self._num_steps = 0
        # Step -> GPU block ids of requests that finished in that step (only
        # the last RECENTLY_FREED_STEPS steps are kept).
        self._recently_freed: dict[int, set[int]] = {}

        # Diagnostics
        self._stats = SimpleCPUOffloadStats()
        self._test_mode = envs.VLLM_SIMPLE_KV_OFFLOAD_TEST_MODE
        self._clock = time.monotonic
        self._next_host_mem_check = 0.0
        self._next_refused_hit_log = 0.0
        self._num_refused_hits_unlogged = 0

    @staticmethod
    def _derive_cpu_config(
        gpu_config: "KVCacheConfig", cpu_capacity_bytes: int
    ) -> "KVCacheConfig":
        """Derive a CPU KVCacheConfig from the GPU config.
        Same kv_cache_groups, num_blocks scaled by CPU/GPU memory ratio."""
        # Import here to avoid potential circular imports
        from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfigCls
        from vllm.v1.kv_cache_interface import KVCacheTensor

        assert len(gpu_config.kv_cache_tensors) > 0

        gpu_total_bytes = sum(t.size for t in gpu_config.kv_cache_tensors)
        num_gpu_blocks = gpu_config.num_blocks
        num_cpu_blocks = max(1, num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes)
        # Create CPU kv_cache_tensors mirroring GPU by scaling size proportionally.
        cpu_tensors = [
            KVCacheTensor(
                size=t.size // num_gpu_blocks * num_cpu_blocks,
                shared_by=list(t.shared_by),
            )
            for t in gpu_config.kv_cache_tensors
        ]

        return KVCacheConfigCls(
            num_blocks=num_cpu_blocks,
            kv_cache_tensors=cpu_tensors,
            kv_cache_groups=gpu_config.kv_cache_groups,
        )

    @staticmethod
    def _estimate_lazy_target_blocks(
        kv_cache_config: "KVCacheConfig", max_num_batched_tokens: int
    ) -> int:
        """GPU blocks to keep available (free/offloaded) per step in lazy mode."""
        WATERMARK_RATIO = 1.0  # Reserve larger space to avoid running out of GPU blocks
        target = 0
        for g in kv_cache_config.kv_cache_groups:
            spec = g.kv_cache_spec
            if not spec.prefix_cacheable:
                # Per-request scratch (e.g. the QSA ring) is never offloaded.
                continue
            if isinstance(spec, MambaSpec):
                target += 2
            elif isinstance(spec, SlidingWindowSpec):
                target += cdiv(spec.sliding_window, spec.block_size) + 1
            else:
                target += cdiv(max_num_batched_tokens, spec.block_size)
        return int(target * (1 + WATERMARK_RATIO))

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Bind GPU block pool so that we can touch blocks during stores.
        Called by Scheduler after kv_cache_manager is ready."""
        self._gpu_block_pool = gpu_block_pool

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Return (num_new_tokens, is_async) from consecutive CPU cache hits."""

        # Pins found CPU blocks so they survive LRU eviction until
        # update_state_after_alloc() consumes them. Any pin from an earlier
        # call on the same request (e.g. retry after a failed allocate_slots)
        # is dropped first.
        if stale := self._pending_cpu_hits.pop(request.request_id, None):
            self._free_pending_cpu_hit(stale)

        # Requests that bypass the GPU prefix cache (e.g. prompt_logprobs)
        # must compute every prompt token: no CPU hit either.
        if request.skip_reading_prefix_cache:
            return 0, False

        num_skipped_hashes = num_computed_tokens // self.hash_block_size
        remaining_hashes = request.block_hashes[num_skipped_hashes:]

        if not remaining_hashes:
            return 0, False
        # Must recompute at least the last token, matching the logic in
        # kv_cache_manager.get_computed_blocks().
        max_hit_len = request.num_tokens - 1 - num_computed_tokens
        if max_hit_len <= 0:
            return 0, False
        cpu_hit_blocks, hit_length = self.cpu_coordinator.find_longest_cache_hit(
            remaining_hashes, max_hit_len
        )

        if hit_length > 0:
            reason = self._check_cpu_hit(
                cpu_hit_blocks, hit_length, num_computed_tokens
            )
            if reason is not None:
                self._refuse_cpu_hit(request.request_id, reason)
                return 0, False
            pin_blocks = [
                blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
            ]
            self.cpu_block_pool.touch(pin_blocks)
            self._pending_cpu_hits[request.request_id] = PendingCPUHit(
                blocks=cpu_hit_blocks,
                num_tokens=hit_length,
                num_local_tokens=num_computed_tokens,
            )
            return hit_length, True
        return 0, False

    def _check_cpu_hit(
        self,
        cpu_hit_blocks: tuple[list["KVCacheBlock"], ...],
        hit_length: int,
        num_computed_tokens: int,
    ) -> str | None:
        """Return why a CPU hit cannot be loaded as-is, or None if it can.

        Hits are scheduler-block aligned by construction; this guards the
        load path against a hit that would restore a partial page or a
        missing / stale Mamba state.
        """
        if hit_length % self.block_size != 0:
            return f"hit_length={hit_length} is not a multiple of {self.block_size}"
        if (num_computed_tokens + hit_length) % self.block_size != 0:
            return (
                f"num_computed_tokens={num_computed_tokens} + hit_length="
                f"{hit_length} is not a multiple of {self.block_size}"
            )
        for g, group in enumerate(self.cpu_kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            blocks = cpu_hit_blocks[g]
            if not spec.prefix_cacheable:
                if blocks:
                    return f"non-cacheable group {g} has {len(blocks)} hit blocks"
                continue
            if len(blocks) * spec.block_size != hit_length:
                return (
                    f"group {g} has {len(blocks)} hit blocks of "
                    f"{spec.block_size} tokens for hit_length={hit_length}"
                )
            if isinstance(spec, MambaSpec) and blocks[-1].is_null:
                return f"Mamba group {g} hit does not end with a state block"
        return None

    def _refuse_cpu_hit(self, req_id: str, reason: str) -> None:
        self._stats.add(CPU_HITS_REFUSED)
        if self._test_mode:
            raise AssertionError(
                f"SimpleCPUOffloadScheduler: invalid CPU hit for req_id={req_id}: "
                f"{reason}"
            )
        self._num_refused_hits_unlogged += 1
        now = self._clock()
        if now < self._next_refused_hit_log:
            return
        logger.error(
            "SimpleCPUOffloadScheduler: refused a CPU hit for req_id=%s (%s); "
            "%d refused since the last report. The tokens are recomputed.",
            req_id,
            reason,
            self._num_refused_hits_unlogged,
        )
        self._num_refused_hits_unlogged = 0
        self._next_refused_hit_log = now + REFUSED_HIT_LOG_INTERVAL_S

    # TODO(yifan): this API now only matches the suffix part of the prefix cache. A more
    # general API should scan blocks in both GPU and CPU block pool in a single pass.
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        block_ids_by_group = blocks.get_block_ids()
        num_groups = len(block_ids_by_group)

        # Store tracking (eager mode only). Register the request;
        # block IDs are accumulated from scheduler_output in
        # _prepare_eager_store_specs via yield_req_data.
        if not self._lazy_mode and req_id not in self._reqs_to_store:
            self._reqs_to_store[req_id] = StoreRequestState(
                request=request,
                block_ids=tuple([] for _ in range(num_groups)),
                num_stored_blocks=[0] * num_groups,
            )

        # Pop the CPU hit cached by get_num_new_matched_tokens(). The
        # found blocks were pinned there to survive LRU eviction in the window
        # between get_num_new_matched_tokens() and this matching call.
        pending = self._pending_cpu_hits.pop(req_id, None)

        if num_external_tokens == 0:
            if pending is not None:
                logger.warning(
                    "SimpleCPUOffloadScheduler: update_state_after_alloc "
                    "called for req_id=%s with no external tokens but "
                    "get_num_new_matched_tokens() unexpectedly recorded "
                    "a pending CPU hit; releasing the stale pin.",
                    req_id,
                )
                self._free_pending_cpu_hit(pending)
            return

        # The scheduler admits exactly the external tokens reported by
        # get_num_new_matched_tokens() (it never trims them) and parks the
        # request in WAITING_FOR_REMOTE_KVS until the load completes, so
        # anything else would either load a partial hit or never finish.
        if pending is None or num_external_tokens != pending.num_tokens:
            if pending is not None:
                self._free_pending_cpu_hit(pending)
            raise RuntimeError(
                f"SimpleCPUOffloadScheduler: update_state_after_alloc for "
                f"req_id={req_id} got num_external_tokens={num_external_tokens}"
                f" but get_num_new_matched_tokens() reported "
                f"{pending.num_tokens if pending is not None else 0}"
            )

        # Build transfer pairs across all groups. External blocks sit at the
        # tail of the computed range of every group.
        total_computed_tokens = pending.num_local_tokens + num_external_tokens
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups
        assert self._gpu_block_pool is not None
        gpu_blocks = self._gpu_block_pool.blocks

        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        cpu_blocks_to_touch: list[KVCacheBlock] = []

        for g in range(num_groups):
            spec = kv_cache_groups[g].kv_cache_spec
            if not spec.prefix_cacheable:
                # Per-request scratch; get_num_new_matched_tokens() refuses
                # hits with blocks in these groups.
                continue
            cpu_blocks_g = pending.blocks[g]
            # Number of blocks in the computed range for this group.
            n_computed_g = cdiv(total_computed_tokens, spec.block_size)
            gpu_ext_start = n_computed_g - len(cpu_blocks_g)
            group_gpu_ids = block_ids_by_group[g]

            for i, cpu_blk in enumerate(cpu_blocks_g):
                # Skip null blocks (e.g. sliding window or mamba padding).
                if cpu_blk.is_null:
                    continue
                gpu_block_id = group_gpu_ids[gpu_ext_start + i]
                if gpu_blocks[gpu_block_id].is_null:
                    raise RuntimeError(
                        f"SimpleCPUOffloadScheduler: req_id={req_id} group {g} "
                        f"would load CPU block {cpu_blk.block_id} into the "
                        "null GPU block"
                    )
                gpu_block_ids.append(gpu_block_id)
                cpu_block_ids.append(cpu_blk.block_id)
                cpu_blocks_to_touch.append(cpu_blk)

            if isinstance(spec, MambaSpec) and (
                not cpu_blocks_g
                or cpu_blocks_g[-1].is_null
                or gpu_block_ids[-1] != group_gpu_ids[n_computed_g - 1]
            ):
                # The request resumes from the state in its last computed
                # block; anything else restores a stale or missing state.
                raise RuntimeError(
                    f"SimpleCPUOffloadScheduler: req_id={req_id} Mamba group "
                    f"{g} load does not end in the state block "
                    f"{group_gpu_ids[n_computed_g - 1]} for "
                    f"{total_computed_tokens} computed tokens"
                )

        # Touch CPU blocks to prevent eviction during async load.
        self.cpu_block_pool.touch(cpu_blocks_to_touch)
        # Release the temporary pin held since get_num_new_matched_tokens().
        self._free_pending_cpu_hit(pending)

        # Touch GPU blocks to prevent freeing during async load
        self._gpu_block_pool.touch([gpu_blocks[bid] for bid in gpu_block_ids])

        # The loaded content is already on CPU: do not store it back.
        for gpu_block_id, cpu_blk in zip(gpu_block_ids, cpu_blocks_to_touch):
            assert cpu_blk.block_hash is not None
            self._gpu_offloaded[gpu_block_id] = (
                cpu_blk.block_hash,
                gpu_blocks[gpu_block_id].alloc_generation,
            )

        self._stats.add(LOAD_REQUESTS)
        if self._recently_freed:
            num_recently_freed = sum(
                any(bid in freed for freed in self._recently_freed.values())
                for bid in gpu_block_ids
            )
            if num_recently_freed:
                self._stats.add(LOAD_DST_RECENTLY_FREED, num_recently_freed)

        assert self._reqs_to_load.get(req_id) is None
        self._reqs_to_load[req_id] = LoadRequestState(
            request=request,
            transfer_meta=TransferMeta(gpu_block_ids, cpu_block_ids),
            num_local_hit_tokens=pending.num_local_tokens,
            num_cpu_hit_tokens=num_external_tokens,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:
        self._maybe_check_host_mem()

        # --- Stores ---
        store_event = -1
        store_gpu, store_cpu, store_req_ids = self.prepare_store_specs(scheduler_output)
        if store_gpu:
            store_event = self._store_event_counter
            self._store_event_counter += 1
            self._store_event_to_blocks[store_event] = TransferMeta(
                store_gpu, store_cpu
            )
            self._num_store_pinned_blocks += len(store_gpu)
            self._stats.add(STORES_SELECTED, len(store_gpu))
            if store_req_ids:  # For eager mode only, track req->blocks mapping
                self._store_event_to_reqs[store_event] = store_req_ids
                for req_id in store_req_ids:
                    store_state = self._reqs_to_store.get(req_id)
                    if store_state is not None:
                        store_state.store_events.add(store_event)

        # --- Loads ---
        load_event = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_req_ids: list[str] = []
        for req_id, load_state in self._reqs_to_load.items():
            if load_state.load_event is not None:
                continue
            assert load_state.transfer_meta is not None
            load_gpu.extend(load_state.transfer_meta.gpu_block_ids)
            load_cpu.extend(load_state.transfer_meta.cpu_block_ids)
            load_req_ids.append(req_id)
        if load_req_ids:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            now = self._clock()
            for req_id in load_req_ids:
                load_state = self._reqs_to_load[req_id]
                load_state.load_event = load_event
                load_state.emit_step = self._num_steps
                load_state.emit_time = now
                logger.info(
                    "SimpleCPU load req=%s local_hit=%d cpu_hit=%d blocks=%d event=%d",
                    req_id,
                    load_state.num_local_hit_tokens,
                    load_state.num_cpu_hit_tokens,
                    len(load_state.transfer_meta.gpu_block_ids),
                    load_event,
                )
            self._load_event_to_reqs[load_event] = load_req_ids

        result = SimpleCPUOffloadMetadata(
            load_event=load_event,
            load_gpu_blocks=load_gpu,
            load_cpu_blocks=load_cpu,
            load_event_to_reqs=self._load_event_to_reqs,
            store_event=store_event,
            store_gpu_blocks=store_gpu,
            store_cpu_blocks=store_cpu,
            need_flush=bool(scheduler_output.preempted_req_ids),
            num_cpu_blocks=self.num_cpu_blocks,
        )

        self._num_steps += 1
        oldest_step = self._num_steps - RECENTLY_FREED_STEPS + 1
        for step in [s for s in self._recently_freed if s < oldest_step]:
            del self._recently_freed[step]
        return result

    def _maybe_check_host_mem(self) -> None:
        """Sample host MemAvailable every HOST_MEM_CHECK_INTERVAL_S seconds."""
        now = self._clock()
        if now < self._next_host_mem_check:
            return
        self._next_host_mem_check = now + HOST_MEM_CHECK_INTERVAL_S
        available = read_mem_available_bytes()
        if available is None:
            return
        available_gib = available / GiB
        self._stats.add(HOST_MEM_AVAILABLE_GIB_MIN, available_gib)
        if available_gib < self.options.host_mem_warn_gib:
            logger.warning(
                "SimpleCPUOffloadScheduler: host MemAvailable is %.2f GiB, below "
                "host_mem_warn_gib=%.2f; the pinned CPU KV tier cannot be "
                "reclaimed.",
                available_gib,
                self.options.host_mem_warn_gib,
            )

    def prepare_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[int], list[int], list[str]]:
        """Prepare store specs for the store event."""
        if self._lazy_mode:
            return self._prepare_lazy_store_specs(scheduler_output)
        else:
            return self._prepare_eager_store_specs(scheduler_output)

    def _prepare_lazy_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[int], list[int], list[str]]:
        """Paced cursor walk over the GPU free queue.

        Freed hashed blocks are appended at the MRU tail of the GPU free queue.
        The cursor follows the tail, so blocks are copied to CPU a few steps
        after their request released them and stay valid on GPU until the
        allocator evicts them (inclusive tier).

        Each step visits at most max(target_free, 4 * max_store_blocks_per_step)
        nodes and selects at most max_store_blocks_per_step blocks
        (IDLE_STORE_MULTIPLIER times that on steps without scheduled tokens).
        Selection stops at the first candidate that would take the GPU free
        pool down to the reserve, find no allocatable CPU block, or pin more
        than 4 * max_store_blocks_per_step GPU blocks in in-flight stores; the
        cursor stays in front of that candidate.
        """
        gpu_pool = self._gpu_block_pool
        if gpu_pool is None or self._target_free <= 0:
            self._lazy_backlog = False
            return [], [], []

        free_queue = gpu_pool.free_block_queue
        tail = free_queue.fake_free_list_tail
        cpu_pool = self.cpu_block_pool
        cpu_cached = cpu_pool.cached_block_hash_to_block
        stats = self._stats

        max_per_step = self.options.max_store_blocks_per_step
        visit_budget = max(self._target_free, 4 * max_per_step)
        is_idle = scheduler_output.total_num_scheduled_tokens == 0
        max_select = max_per_step * (IDLE_STORE_MULTIPLIER if is_idle else 1)
        max_select = min(
            max_select,
            cpu_pool.get_num_free_blocks(),
            4 * max_per_step - self._num_store_pinned_blocks,
        )
        num_gpu_free = gpu_pool.get_num_free_blocks()
        # Re-storing a block whose CPU copy was evicted is opt-in, and never
        # done on idle steps so the idle drain always ends.
        skip_restore = not self.options.allow_restore or is_idle

        # Resume after the cursor while it is still a free block; a cursor
        # that was allocated (or touched by a prefix hit) restarts the walk
        # at the LRU head. Frees reach the pool before the next schedule()
        # allocates, so a cursor taken in this step is still taken here.
        cursor = self._cursor
        node: KVCacheBlock | None
        if (
            cursor is not None
            and cursor.ref_cnt == 0
            and cursor.prev_free_block is not None
            and cursor.next_free_block is not None
        ):
            node = cursor.next_free_block
        else:
            cursor = None
            node = free_queue.fake_free_list_head.next_free_block

        gpu_ids: list[int] = []
        block_hashes: list[BlockHashWithGroupId] = []
        num_visited = 0
        blocked = False
        while node is not None and node is not tail and num_visited < visit_budget:
            bhash = node.block_hash
            if bhash is None or node.is_null:
                pass
            elif (cached_blk := cpu_cached.get_one_block(bhash)) is not None:
                # Already on CPU: refresh its LRU position, the GPU copy is
                # being reused or kept.
                if cached_blk.ref_cnt == 0:
                    cpu_pool.touch([cached_blk])
                    cpu_pool.free_blocks([cached_blk])
            elif bhash in self._lazy_inflight_hashes:
                stats.add(INFLIGHT_SKIPPED)
            elif skip_restore and self._gpu_offloaded.get(node.block_id) == (
                bhash,
                node.alloc_generation,
            ):
                stats.add(RESTORE_SKIPPED)
            else:
                # Candidate. Stop in front of it if it cannot be selected.
                if len(gpu_ids) >= max_select:
                    blocked = True
                    break
                if num_gpu_free - len(gpu_ids) <= self._store_reserve:
                    stats.add(RESERVE_STOPS)
                    blocked = True
                    break
                gpu_ids.append(node.block_id)
                block_hashes.append(bhash)
                self._lazy_inflight_hashes.add(bhash)
                self._gpu_offloaded[node.block_id] = (bhash, node.alloc_generation)
                num_visited += 1
                node = node.next_free_block
                continue
            # Selected nodes leave the free queue when touched below, so the
            # cursor is the last visited node that stays.
            cursor = node
            num_visited += 1
            node = node.next_free_block

        self._cursor = cursor
        if node is None or node is tail:
            self._lazy_backlog = False
        elif blocked:
            # Progress next step needs this step's stores or earlier ones to
            # complete; with neither in flight nothing would change.
            self._lazy_backlog = bool(gpu_ids) or bool(self._store_event_to_blocks)
        else:
            # Visit budget exhausted.
            self._lazy_backlog = True

        # Batch-allocate CPU blocks and stamp hashes.
        if gpu_ids:
            cpu_blocks = cpu_pool.get_new_blocks(len(gpu_ids))
            cpu_ids = [blk.block_id for blk in cpu_blocks]
            for cpu_blk, bhash in zip(cpu_blocks, block_hashes):  # type: ignore[assignment]
                cpu_blk._block_hash = bhash  # type: ignore[assignment]
            # Touch GPU blocks to prevent eviction during async copy.
            gpu_pool.touch([gpu_pool.blocks[bid] for bid in gpu_ids])
        else:
            cpu_ids = []

        return gpu_ids, cpu_ids, []

    def _prepare_eager_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[int], list[int], list[str]]:
        """Identify newly computed blocks to offload from scheduler requests.

        Only considers blocks whose KV data has been **confirmed computed** by
        the GPU. This means blocks from the current step are NOT stored until the
        next step. If a request finishes in the same step as its last full block,
        that block may be missed. (TODO: flush on finish.)

        Returns:
            (gpu_block_ids, cpu_block_ids, req_ids) for the store event.
        """

        merged_gpu_block_ids: list[int] = []
        merged_cpu_block_ids: list[int] = []
        req_ids: list[str] = []

        gpu_block_pool = self._gpu_block_pool
        if gpu_block_pool is None:
            return [], [], []
        cpu_block_pool = self.cpu_block_pool
        num_free = cpu_block_pool.get_num_free_blocks()
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups
        num_groups = len(kv_cache_groups)
        # Dedup against blocks already scheduled.
        in_flight = self._in_flight_store_gpu_blocks

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            state = self._reqs_to_store.get(req_id)
            if state is None or state.finished:
                continue

            # Accumulate new block IDs.
            if preempted:
                state.block_ids = tuple([] for _ in range(num_groups))
                state.num_stored_blocks = [0] * num_groups
            if new_block_id_groups:
                for g in range(min(num_groups, len(new_block_id_groups))):
                    if new_block_id_groups[g] is not None:
                        state.block_ids[g].extend(new_block_id_groups[g])

            num_new_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_new_tokens == 0:
                continue

            block_ids_by_group = state.block_ids
            if not block_ids_by_group:
                continue

            # --- Phase 1: Scan blocks, classify as cached vs to-store ---
            gpu_block_ids: list[int] = []
            block_hashes_to_store: list[bytes] = []
            advanced_per_group: list[int] = [0] * num_groups
            out_of_space = False
            # Confirmed tokens: KV data written and visible to all streams.
            req = state.request
            confirmed_tokens = req.num_computed_tokens - req.num_output_placeholders
            # Cap to blocks with confirmed KV data.
            aligned_tokens = confirmed_tokens // self.block_size * self.block_size

            for g in range(num_groups):
                # FIXME (yifan): handle CPU cache eviction, where
                # num_stored_blocks can be stale and omit evicted blocks in
                # the middle of the request.
                already_stored_g = state.num_stored_blocks[g]
                group_gpu_ids = block_ids_by_group[g]

                g_block_size = kv_cache_groups[g].kv_cache_spec.block_size
                ready_blocks_g = aligned_tokens // g_block_size
                scannable = group_gpu_ids[already_stored_g:ready_blocks_g]

                for gpu_block_id in scannable:
                    gpu_block = gpu_block_pool.blocks[gpu_block_id]
                    if gpu_block.is_null:
                        advanced_per_group[g] += 1
                        continue

                    bhash_with_group = gpu_block.block_hash
                    if bhash_with_group is None:
                        # Masked-out SWA position the coordinator chose not to
                        # hash; it can never serve a prefix-cache hit, so skip.
                        advanced_per_group[g] += 1
                        continue

                    # Skip if already scheduled for store or already cached in CPU.
                    if (
                        gpu_block_id in in_flight
                        or cpu_block_pool.cached_block_hash_to_block.get_one_block(
                            bhash_with_group
                        )
                        is not None
                    ):
                        advanced_per_group[g] += 1
                        continue

                    if num_free <= 0:
                        out_of_space = True
                        break
                    num_free -= 1

                    gpu_block_ids.append(gpu_block_id)
                    block_hashes_to_store.append(bhash_with_group)
                    advanced_per_group[g] += 1

                if out_of_space:
                    break

            # --- Phase 2: Batch allocate CPU blocks and stamp hashes ---
            n_to_alloc = len(gpu_block_ids)
            if n_to_alloc > 0:
                cpu_blocks_alloc = cpu_block_pool.get_new_blocks(n_to_alloc)
                cpu_block_ids = [blk.block_id for blk in cpu_blocks_alloc]
                for cpu_blk, bhash in zip(cpu_blocks_alloc, block_hashes_to_store):
                    cpu_blk._block_hash = bhash  # type: ignore[assignment]
            else:
                cpu_block_ids = []

            if cpu_block_ids:
                req_ids.append(req_id)
                merged_gpu_block_ids.extend(gpu_block_ids)
                merged_cpu_block_ids.extend(cpu_block_ids)
                in_flight.update(gpu_block_ids)

                # Touch GPU blocks to prevent freeing during async copy
                gpu_block_pool.touch(
                    [gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
                )

                logger.debug(
                    "Request %s: Scheduling store of %d blocks to CPU (%d groups)",
                    req_id,
                    len(cpu_block_ids),
                    num_groups,
                )

            # Advance per-group cursors (includes cached hits + newly stored)
            for g in range(num_groups):
                state.num_stored_blocks[g] += advanced_per_group[g]

        return merged_gpu_block_ids, merged_cpu_block_ids, req_ids

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Handle async transfer completions from worker.

        Load completions arrive via finished_recving (real req_ids).
        Store completions arrive via kv_connector_worker_meta as
        per-event worker counts. We accumulate across steps and process
        a store event only when all workers have reported completion.
        """
        # --- Load completions ---
        for req_id in list(connector_output.finished_recving or []):
            load_state = self._reqs_to_load.get(req_id)
            if load_state is not None and load_state.load_event is not None:
                logger.info(
                    "SimpleCPU load done req=%s event=%d steps=%d wall_ms=%.1f",
                    req_id,
                    load_state.load_event,
                    self._num_steps - load_state.emit_step,
                    (self._clock() - load_state.emit_time) * 1000,
                )
            self._cleanup_load_request(req_id)

        # --- Store completions ---
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, SimpleCPUOffloadWorkerMetadata):
            return
        for event_idx, count in meta.completed_store_events.items():
            total = self._store_event_pending_counts.get(event_idx, 0) + count
            if total >= self._expected_worker_count:
                self._store_event_pending_counts.pop(event_idx, None)
                self._process_store_event(event_idx)
            else:
                self._store_event_pending_counts[event_idx] = total

    def _process_store_event(self, event_idx: int) -> None:
        """Process a fully-completed store event."""
        transfer = self._store_event_to_blocks.pop(event_idx)
        self._num_store_pinned_blocks -= len(transfer.gpu_block_ids)
        if not self._lazy_mode:
            self._in_flight_store_gpu_blocks.difference_update(transfer.gpu_block_ids)
        self._process_store_completion(transfer.gpu_block_ids, transfer.cpu_block_ids)
        logger.debug(
            "Store event %d completed: cached %d blocks to CPU",
            event_idx,
            len(transfer.cpu_block_ids),
        )

        # Eager only: update per-req state
        if not self._lazy_mode:
            for req_id in self._store_event_to_reqs.pop(event_idx, []):
                state = self._reqs_to_store.get(req_id)
                if state is None:
                    continue
                state.store_events.discard(event_idx)
                if state.finished and not state.store_events:
                    self._cleanup_store_request(req_id)

    def _process_store_completion(
        self, gpu_block_ids: list[int], cpu_block_ids: list[int]
    ) -> None:
        """Cache CPU blocks per-group and release GPU refs.

        Block hashes were stamped on CPU blocks at allocation time (in
        ``_prepare_*_store_specs``).  Here we just register them in the
        cache map so they become discoverable by the load path.
        """
        assert len(cpu_block_ids) == len(gpu_block_ids)

        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in cpu_block_ids]

        for cpu_block in cpu_blocks:
            bhash = cpu_block.block_hash
            assert bhash is not None
            self.cpu_block_pool.cached_block_hash_to_block.insert(bhash, cpu_block)
            self._lazy_inflight_hashes.discard(bhash)

        # Free CPU and GPU blocks' ref counts to turn them into prefix cache
        self.cpu_block_pool.free_blocks(cpu_blocks)
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.free_blocks(
            self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
        )

    def has_pending_stores(self) -> bool:
        """Return True if there are in-flight store transfers."""
        return bool(self._store_event_to_blocks)

    def has_pending_work(self) -> bool:
        """Return True while the engine must keep stepping without requests:
        stores are in flight, or the lazy walk has freed blocks left to store.
        """
        return bool(self._store_event_to_blocks) or (
            self._lazy_mode and self._lazy_backlog
        )

    def reset(self) -> bool:
        """Drop the CPU prefix cache. Returns False while transfers are in
        flight (the caller retries later)."""
        if (
            self._reqs_to_load
            or self._load_event_to_reqs
            or self._store_event_to_blocks
        ):
            return False
        for pending in self._pending_cpu_hits.values():
            self._free_pending_cpu_hit(pending)
        self._pending_cpu_hits.clear()
        self._cursor = None
        self._gpu_offloaded.clear()
        self._lazy_inflight_hashes.clear()
        self._lazy_backlog = False
        return self.cpu_block_pool.reset_prefix_cache()

    def get_stats(self) -> SimpleCPUOffloadStats | None:
        """Scheduler-side counters since the last call plus current gauges."""
        stats = self._stats
        stats.add(STORE_PINS_LAST, self._num_store_pinned_blocks)
        stats.add(
            CPU_USED_BLOCKS_LAST,
            self.num_cpu_blocks - self.cpu_block_pool.get_num_free_blocks(),
        )
        stats.add(
            CPU_CACHED_BLOCKS_LAST,
            len(self.cpu_block_pool.cached_block_hash_to_block),
        )
        if stats.is_empty():
            stats.reset()
            return None
        return stats.clone_and_reset()

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Always returns (False, None). GPU blocks are protected by ref_cnt,
        so the scheduler can free blocks immediately. ``block_ids`` only feed
        the LOAD_DST_RECENTLY_FREED diagnostic."""
        req_id = request.request_id

        # Diagnostics: count loads into pages of just-finished requests
        # (LOAD_DST_RECENTLY_FREED).
        if block_ids:
            self._recently_freed.setdefault(self._num_steps, set()).update(block_ids)

        # Release any temp CPU hit pin from get_num_new_matched_tokens()
        # if request is canceled or preempted before update_state_after_alloc()
        pending = self._pending_cpu_hits.pop(req_id, None)
        if pending is not None:
            self._free_pending_cpu_hit(pending)

        # Handle load: defer cleanup if load is in-flight
        load_state = self._reqs_to_load.get(req_id)
        if load_state is not None:
            if load_state.load_event is not None:
                load_state.finished = True  # Defer: load in-flight
            else:
                self._cleanup_load_request(req_id)

        # Handle store (eager mode only): defer cleanup if stores in-flight
        if not self._lazy_mode:
            store_state = self._reqs_to_store.get(req_id)
            if store_state is not None:
                if store_state.store_events:
                    store_state.finished = True  # Defer: stores in-flight
                else:
                    self._cleanup_store_request(req_id)

        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(
            request, block_ids=[bid for group in block_ids for bid in group]
        )

    def _free_pending_cpu_hit(self, pending: PendingCPUHit) -> None:
        """Release the temporary CPU block pin taken in get_num_new_matched_tokens()."""
        blocks_to_free = [
            blk for grp in pending.blocks for blk in grp if not blk.is_null
        ]
        if blocks_to_free:
            self.cpu_block_pool.free_blocks(blocks_to_free)

    def _cleanup_load_request(self, req_id: str) -> None:
        """Release all load resources for a request.

        Shared between request_finished() and update_connector_output() paths.
        Removes the request from _reqs_to_load, cleans up event mappings,
        and frees CPU/GPU touch refs.
        """
        state = self._reqs_to_load.pop(req_id, None)
        if state is None:
            return
        # Remove from load event mapping (only this req, not whole event)
        if state.load_event is not None:
            reqs = self._load_event_to_reqs.get(state.load_event)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._load_event_to_reqs.pop(state.load_event, None)

        if state.transfer_meta is not None:
            # Free CPU touch refs
            self.cpu_block_pool.free_blocks(
                self.cpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.cpu_block_ids
            )
            # Free GPU touch refs
            assert self._gpu_block_pool is not None
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.gpu_block_ids
            )

    def _cleanup_store_request(self, req_id: str) -> None:
        """Release store metadata for a request.

        Metadata-only cleanup but no block freeing. Job completion handles
        block caching and GPU ref freeing via _process_store_completion().
        """
        state = self._reqs_to_store.pop(req_id, None)
        if state is None:
            return
        for event_idx in list(state.store_events):
            if (reqs := self._store_event_to_reqs.get(event_idx)) is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._store_event_to_reqs.pop(event_idx, None)
        state.store_events.clear()

    def take_events(self) -> Iterable[KVCacheEvent]:
        return self.cpu_block_pool.take_events()
