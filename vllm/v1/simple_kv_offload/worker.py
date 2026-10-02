# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

import time
from typing import TYPE_CHECKING, NamedTuple

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend, TransferEvent
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor
from vllm.v1.simple_kv_offload.host_mem import (
    GiB,
    check_host_mem_before_pinning,
    read_mem_available_bytes,
)
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)
from vllm.v1.simple_kv_offload.options import SimpleCPUOffloadOptions, test_switch
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
    STORE_DMA_MS_SUM,
    STORE_EVENTS,
    STORE_READY_MS_MAX,
    STORE_READY_MS_SUM,
    VERIFY_CHECKED,
    VERIFY_MISMATCH,
    SimpleCPUOffloadStats,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Per-direction stats keys: events, blocks, bytes, DMA ms sum / max, ready ms
# sum / max.
_LOAD_STAT_KEYS = (
    LOAD_EVENTS,
    LOAD_BLOCKS,
    LOAD_BYTES,
    LOAD_DMA_MS_SUM,
    LOAD_DMA_MS_MAX,
    LOAD_READY_MS_SUM,
    LOAD_READY_MS_MAX,
)
_STORE_STAT_KEYS = (
    STORE_EVENTS,
    STORE_BLOCKS,
    STORE_BYTES,
    STORE_DMA_MS_SUM,
    STORE_DMA_MS_MAX,
    STORE_READY_MS_SUM,
    STORE_READY_MS_MAX,
)


class _Launch(NamedTuple):
    """A launched transfer, kept on the main thread until it completes."""

    launch_time: float  # worker clock (monotonic seconds)
    cpu_blocks: list[int]
    gpu_blocks: list[int]


def _global_rank() -> int:
    """This worker's global rank, for log and error messages."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class SimpleCPUOffloadWorker:
    """Worker-side handler for CPU offloading transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        options: SimpleCPUOffloadOptions | None = None,
    ):
        self.vllm_config = vllm_config
        self.options = options or SimpleCPUOffloadOptions()
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        # CUDA streams for the async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        self._backend = DmaCopyBackend()

        # Issued copies in event order, appended by the copy thread.
        self._load_events: list[TransferEvent] = []
        self._store_events: list[TransferEvent] = []
        # High-water marks: highest event_idx completed per stream.
        # When the event list is empty, the hwm covers all prior events.
        self._load_hwm: int = -1
        self._store_hwm: int = -1

        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

        # Pending event index sets, populated in bind_connector_metadata
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()
        # Completed store events to report via build_connector_worker_meta
        self._completed_store_events: dict[int, int] = {}

        # Launched transfers that have not completed yet, oldest first.
        self._load_launches: dict[int, _Launch] = {}
        self._store_launches: dict[int, _Launch] = {}
        # Monotonic clock of the transfer watchdog and the ready-time stats.
        self._clock = time.monotonic
        # Bytes of one block across every registered KV region.
        self._bytes_per_block: int = 0
        self._stats = SimpleCPUOffloadStats()
        # Recorded on the compute stream when loads are launched.
        self._compute_probe: torch.Event | None = None

        # Diagnostics (off in production).
        self._verify_mode: str = envs.VLLM_SIMPLE_KV_OFFLOAD_VERIFY
        self._unordered_load = bool(
            test_switch("VLLM_SIMPLE_KV_OFFLOAD_TEST_UNORDERED_LOAD")
        )
        self._compute_delay_cycles = int(
            test_switch("VLLM_SIMPLE_KV_OFFLOAD_DEBUG_COMPUTE_DELAY_CYCLES")
        )

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register GPU KV caches and allocate pinned CPU tensors.
        The worker will infer the underlying raw storage from the kv_caches.

        Args:
            kv_caches: Per-layer GPU KV caches. Values are either a single
                tensor (attention layers) or a list of tensors (Mamba layers
                in hybrid models). All values are included for offloading
                by resolving to their underlying raw storage.
        """
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        # Resolve each entry to a representative tensor for storage
        # deduplication. For attention layers the value is already a tensor;
        # for Mamba layers it is a list of tensors that all share the same
        # underlying raw storage, so we take the first one.
        def _repr_tensor(v: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
            assert isinstance(v, torch.Tensor | list)
            return v if isinstance(v, torch.Tensor) else v[0]

        any_tensor = _repr_tensor(next(iter(kv_caches.values())))
        self.device = any_tensor.device

        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks

        # Deduplicate: multiple layers may share the same backing storage.
        seen_ptrs: dict[int, tuple[str, torch.Tensor]] = {}
        for name, value in kv_caches.items():
            tensor = _repr_tensor(value)
            ptr = tensor.untyped_storage().data_ptr()
            if ptr not in seen_ptrs:
                seen_ptrs[ptr] = (name, tensor)

        # Build [num_blocks, block_bytes] int8 views from each unique
        # storage so that stride(0) gives block_bytes for the copy op.
        #
        # The physical layout varies across attention backends:
        #   FlashAttn/ROCm:  (2, num_blocks, ...) -> K/V outermost, 2 segments
        #   FlashInfer/MLA:  (num_blocks, ...)    -> blocks outermost, 1 segment
        # We derive page_size_bytes = storage.nbytes() // num_blocks, then
        # classify dims: any dim whose byte-stride exceeds page_size_bytes
        # must be an outer segment dim (e.g. the K/V dim of size 2). A less
        # hacky way is to update the interface with the layout.
        unique_gpu_caches: dict[str, torch.Tensor] = {}
        for name, tensor in seen_ptrs.values():
            storage = tensor.untyped_storage()
            raw = torch.empty(0, dtype=torch.int8, device=self.device).set_(
                storage, 0, (storage.nbytes(),)
            )
            el = tensor.element_size()
            page_size_bytes = storage.nbytes() // num_blocks
            outer_dims = [
                d for d in range(tensor.ndim) if tensor.stride(d) * el > page_size_bytes
            ]
            if not outer_dims:
                unique_gpu_caches[name] = raw.view(num_blocks, -1)
            else:
                seg_stride = tensor.stride(outer_dims[0]) * el
                for idx in range(tensor.shape[outer_dims[0]]):
                    offset = idx * seg_stride
                    chunk = raw[offset : offset + seg_stride]
                    unique_gpu_caches[f"{name}.{idx}"] = chunk.view(num_blocks, -1)

        # Compute per-tensor bytes_per_block. Tensors may have different
        # page_size_bytes (e.g., UniformTypeKVCacheSpecs with varying head_size).
        per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in unique_gpu_caches.values()
        ]
        total_bytes_per_block = sum(per_tensor_bpb)
        self._bytes_per_block = total_bytes_per_block

        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)

        # Every KV cache tensor must be registered (or restores would be
        # partial), and this worker must have exactly the CPU blocks the
        # scheduler hands out (SimpleCPUOffloadScheduler._derive_cpu_config),
        # or a copy would run past the end of the pinned tier.
        rank = _global_rank()
        registered_bytes = total_bytes_per_block * num_blocks
        kv_cache_bytes = sum(t.size for t in self.kv_cache_config.kv_cache_tensors)
        logger.info(
            "SimpleCPUOffloadWorker rank %d: %d unique GPU KV tensors, "
            "%d bytes/block x %d blocks = %d bytes registered of %d KV cache bytes",
            rank,
            len(unique_gpu_caches),
            total_bytes_per_block,
            num_blocks,
            registered_bytes,
            kv_cache_bytes,
        )
        if registered_bytes != kv_cache_bytes:
            raise RuntimeError(
                f"SimpleCPUOffloadWorker rank {rank}: the registered KV regions "
                f"cover {registered_bytes} bytes but the KV cache tensors hold "
                f"{kv_cache_bytes} bytes"
            )
        scheduler_num_cpu_blocks = max(
            1, num_blocks * self.cpu_capacity_bytes // kv_cache_bytes
        )
        logger.info(
            "SimpleCPUOffloadWorker rank %d: allocating %d CPU blocks (%.2f GB), "
            "scheduler formula: %d CPU blocks",
            rank,
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
            scheduler_num_cpu_blocks,
        )
        if self.num_cpu_blocks != scheduler_num_cpu_blocks:
            raise RuntimeError(
                f"SimpleCPUOffloadWorker rank {rank}: {self.num_cpu_blocks} CPU "
                f"blocks, but the scheduler derives {scheduler_num_cpu_blocks}"
            )

        pin_memory = is_pin_memory_available()
        if not pin_memory:
            logger.warning(
                "Pinned memory not available. CPU offload performance may be degraded."
            )

        # Per-rank check only: the ranks pin concurrently, EngineCore checks
        # the whole tier before any of them starts.
        what = f"SimpleCPUOffloadWorker rank {rank}"
        check_host_mem_before_pinning(
            self.num_cpu_blocks * total_bytes_per_block,
            self.options.min_host_mem_available_gib,
            what,
        )

        self.gpu_kv_caches = unique_gpu_caches
        self.cpu_kv_caches = {}
        for name, gpu_tensor in unique_gpu_caches.items():
            cpu_shape = (self.num_cpu_blocks,) + gpu_tensor.shape[1:]
            # Allocate non-pinned first, then pin via cudaHostRegister to
            # bypass PyTorch's CUDACachingHostAllocator which rounds up to
            # the next power of 2 (e.g. 100 GB -> 128 GB).
            tensor = torch.zeros(cpu_shape, dtype=gpu_tensor.dtype, device="cpu")
            if pin_memory:
                pin_tensor(tensor)
            self.cpu_kv_caches[name] = tensor

        available = read_mem_available_bytes()
        if available is not None:
            logger.info(
                "%s: MemAvailable %.2f GiB after pinning", what, available / GiB
            )

        # Use lowest priority so KV cache I/O yields to compute streams.
        low_pri, _ = torch.cuda.Stream.priority_range()
        self.load_stream = torch.cuda.Stream(priority=low_pri)
        self.store_stream = torch.cuda.Stream(priority=low_pri)

        # Initialize copy backend with caches and streams.
        self._backend.init(
            self.gpu_kv_caches,
            self.cpu_kv_caches,
            self.device,
            self.load_stream,
            self.store_stream,
            copy_backend=self.options.copy_backend,
        )

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        if metadata.num_cpu_blocks and metadata.num_cpu_blocks != self.num_cpu_blocks:
            raise RuntimeError(
                "SimpleCPUOffload CPU block count mismatch: scheduler="
                f"{metadata.num_cpu_blocks}, worker={self.num_cpu_blocks} "
                f"(rank {_global_rank()})"
            )
        self._connector_metadata = metadata
        if metadata.load_event >= 0:
            self._pending_load_event_indices.add(metadata.load_event)
        if metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        # NOTE: we defer launching both load and store to get_finished(),
        # which runs after model execution. This hides the CPU-side
        # block copy op overhead (~5ms) behind GPU compute.
        pass

    def wait_for_save(self) -> None:
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Submit transfers and report completed events to the scheduler.

        Called after model execution, with this step's work (and the zeroing
        of its new blocks) still queued on the compute stream. Both
        directions are ordered behind it on their transfer stream, without
        blocking the host: a store reads blocks the step may still write,
        and a load writes blocks the step may still zero or, when they are
        pages freed while an earlier step was in flight, still read.

        Returns:
            tuple of (finished_sending, finished_recving).
            - finished_sending: always None (stores use worker metadata).
            - finished_recving: req_ids whose loads have completed.
        """
        self._backend.check()

        # (1) Submit transfers
        metadata = self._connector_metadata
        if metadata is not None:
            # Launch loads (CPU->GPU).
            if metadata.load_cpu_blocks:
                self._order_load_behind_compute()
                self._launch(
                    metadata.load_cpu_blocks,
                    metadata.load_gpu_blocks,
                    is_store=False,
                    event_idx=metadata.load_event,
                )
            # Launch stores (GPU->CPU).
            if metadata.store_gpu_blocks:
                # The copy reads the live KV cache, which the compute stream is
                # still writing: order the transfer stream behind it first.
                assert self.store_stream is not None
                self.store_stream.wait_stream(torch.cuda.current_stream())
                self._launch(
                    metadata.store_cpu_blocks,
                    metadata.store_gpu_blocks,
                    is_store=True,
                    event_idx=metadata.store_event,
                )

        # (2) Track completed transfer events
        finished_recving: set[str] = set()

        if self._pending_load_event_indices or self._load_launches:
            load_wm = self._poll_stream_events(is_store=False)
            for j in [j for j in self._pending_load_event_indices if j <= load_wm]:
                self._pending_load_event_indices.discard(j)
                req_ids = (
                    metadata.load_event_to_reqs.get(j) if metadata is not None else None
                )
                if req_ids:
                    finished_recving.update(req_ids)

        if self._pending_store_event_indices or self._store_launches:
            store_wm = self._poll_stream_events(is_store=True)
            for j in [j for j in self._pending_store_event_indices if j <= store_wm]:
                self._pending_store_event_indices.discard(j)
                self._completed_store_events[j] = 1

        # (3) Fail loudly on a transfer that does not complete. Only what
        # the polls above left pending counts, so a long idle gap between
        # two steps cannot trip it.
        self._check_transfer_timeouts()

        return None, finished_recving or None

    def _order_load_behind_compute(self) -> None:
        """Order the next load behind everything queued on the compute stream."""
        assert self.load_stream is not None
        compute_stream = torch.cuda.current_stream()
        if self._compute_delay_cycles > 0:
            # Test only: emulate a compute backlog.
            torch.cuda._sleep(self._compute_delay_cycles)
        if not self._unordered_load:
            self.load_stream.wait_stream(compute_stream)
        if self._compute_probe is None:
            self._compute_probe = torch.Event()
        self._compute_probe.record(compute_stream)
        if not self._compute_probe.query():
            self._stats.add(PENDING_COMPUTE_AT_LOAD)

    def _launch(
        self,
        cpu_blocks: list[int],
        gpu_blocks: list[int],
        is_store: bool,
        event_idx: int,
    ) -> None:
        self._check_block_ids(cpu_blocks, gpu_blocks)
        launches = self._store_launches if is_store else self._load_launches
        launches[event_idx] = _Launch(self._clock(), cpu_blocks, gpu_blocks)
        if is_store:
            self._backend.launch_copy(
                gpu_blocks,
                cpu_blocks,
                is_store=True,
                event_idx=event_idx,
                events_list=self._store_events,
            )
        else:
            self._backend.launch_copy(
                cpu_blocks,
                gpu_blocks,
                is_store=False,
                event_idx=event_idx,
                events_list=self._load_events,
            )

    def _check_block_ids(self, cpu_blocks: list[int], gpu_blocks: list[int]) -> None:
        """The DMA is unchecked: refuse block ids outside either pool."""
        assert self.kv_cache_config is not None
        num_gpu_blocks = self.kv_cache_config.num_blocks
        if len(cpu_blocks) != len(gpu_blocks):
            raise RuntimeError(
                f"SimpleCPUOffload transfer pairs {len(cpu_blocks)} CPU blocks "
                f"with {len(gpu_blocks)} GPU blocks (rank {_global_rank()})"
            )
        if (
            min(cpu_blocks) < 0
            or max(cpu_blocks) >= self.num_cpu_blocks
            or min(gpu_blocks) < 0
            or max(gpu_blocks) >= num_gpu_blocks
        ):
            raise RuntimeError(
                "SimpleCPUOffload transfer out of bounds: CPU blocks "
                f"[{min(cpu_blocks)}, {max(cpu_blocks)}] of {self.num_cpu_blocks}, "
                f"GPU blocks [{min(gpu_blocks)}, {max(gpu_blocks)}] of "
                f"{num_gpu_blocks} (rank {_global_rank()})"
            )

    def _check_transfer_timeouts(self) -> None:
        now = self._clock()
        timeout_s = self.options.transfer_timeout_s
        for kind, launches in (
            ("load", self._load_launches),
            ("store", self._store_launches),
        ):
            if not launches:
                continue
            event_idx, launch = next(iter(launches.items()))
            age_s = now - launch.launch_time
            if age_s > timeout_s:
                raise RuntimeError(
                    f"SimpleCPUOffload {kind} event {event_idx} "
                    f"({len(launch.gpu_blocks)} blocks) has not completed "
                    f"{age_s:.1f}s after launch (transfer_timeout_s={timeout_s})"
                )

    def get_stats(self) -> SimpleCPUOffloadStats | None:
        """Transfer stats since the last call, or None if there are none."""
        if self._stats.is_empty():
            return None
        return self._stats.clone_and_reset()

    def build_connector_worker_meta(self) -> SimpleCPUOffloadWorkerMetadata | None:
        """Return completed store events since the last call."""
        if not self._completed_store_events:
            return None
        meta = SimpleCPUOffloadWorkerMetadata(
            completed_store_events=self._completed_store_events,
        )
        self._completed_store_events = {}
        return meta

    def handle_preemptions(
        self, kv_connector_metadata: SimpleCPUOffloadMetadata
    ) -> None:
        """Sync all in-flight transfers before preempted blocks are reused."""
        if not kv_connector_metadata.need_flush:
            return
        self._flush_and_sync_all()

    def _flush_and_sync_all(self) -> None:
        """Wait for every launched transfer to complete."""
        # Wait until the copy thread has issued (and listed the event of)
        # every launched copy; then nothing is appended while we drain.
        self._backend.wait_drained(self.options.transfer_timeout_s)

        for is_store, events in (
            (False, self._load_events),
            (True, self._store_events),
        ):
            while events:
                transfer = events[0]
                transfer.done.synchronize()
                events.pop(0)
                if is_store:
                    self._store_hwm = transfer.event_idx
                else:
                    self._load_hwm = transfer.event_idx
                self._on_transfer_done(transfer, is_store)

    def _poll_stream_events(self, is_store: bool) -> int:
        """Non-blocking poll for completed events and return the high-water mark."""
        self._backend.check()
        events = self._store_events if is_store else self._load_events
        hwm = self._store_hwm if is_store else self._load_hwm
        while events:
            transfer = events[0]
            if not transfer.done.query():
                break
            hwm = transfer.event_idx
            events.pop(0)
            self._on_transfer_done(transfer, is_store)
        if is_store:
            self._store_hwm = hwm
        else:
            self._load_hwm = hwm
        return hwm

    def _on_transfer_done(self, transfer: TransferEvent, is_store: bool) -> None:
        """Record stats of a completed transfer and verify it if enabled."""
        launches = self._store_launches if is_store else self._load_launches
        launch = launches.pop(transfer.event_idx, None)
        if launch is None:
            return
        ready_ms = (self._clock() - launch.launch_time) * 1000
        dma_ms = transfer.start.elapsed_time(transfer.done)
        num_blocks = len(launch.gpu_blocks)
        events, blocks, nbytes, dma_sum, dma_max, ready_sum, ready_max = (
            _STORE_STAT_KEYS if is_store else _LOAD_STAT_KEYS
        )
        self._stats.add(events)
        self._stats.add(blocks, num_blocks)
        self._stats.add(nbytes, num_blocks * self._bytes_per_block)
        self._stats.add(dma_sum, dma_ms)
        self._stats.add(dma_max, dma_ms)
        self._stats.add(ready_sum, ready_ms)
        self._stats.add(ready_max, ready_ms)
        if self._verify_mode != "off":
            self._verify_transfer(transfer.event_idx, launch, is_store)

    def _verify_transfer(self, event_idx: int, launch: _Launch, is_store: bool) -> None:
        """Compare the GPU and CPU copies of a completed transfer (blocking).

        Loads are checked before they are reported finished and stores while
        their GPU source is still pinned, so both sides must be identical.
        GPU blocks are read back one block and region at a time on the
        compute stream: no device memory is allocated, and a load is checked
        against anything the compute stream wrote to its blocks since.
        """
        num_pairs = len(launch.gpu_blocks)
        if self._verify_mode == "all":
            picks = list(range(num_pairs))
        else:
            picks = sorted({0, num_pairs // 2, num_pairs - 1})
        mismatches = 0
        first: tuple[str, int, int] | None = None
        for i in picks:
            cpu_id, gpu_id = launch.cpu_blocks[i], launch.gpu_blocks[i]
            region = self._first_differing_region(cpu_id, gpu_id)
            if region is not None:
                mismatches += 1
                if first is None:
                    first = (region, gpu_id, cpu_id)
        self._stats.add(VERIFY_CHECKED, len(picks))
        if first is not None:
            self._stats.add(VERIFY_MISMATCH, mismatches)
            logger.warning(
                "SimpleCPUOffload verify: %s event %d: %d of %d checked blocks "
                "differ between GPU and CPU; first in region %s, GPU block %d, "
                "CPU block %d",
                "store" if is_store else "load",
                event_idx,
                mismatches,
                len(picks),
                *first,
            )

    def _first_differing_region(self, cpu_id: int, gpu_id: int) -> str | None:
        assert self.gpu_kv_caches is not None and self.cpu_kv_caches is not None
        for name, gpu_region in self.gpu_kv_caches.items():
            cpu_region = self.cpu_kv_caches[name]
            if not torch.equal(gpu_region[gpu_id].cpu(), cpu_region[cpu_id]):
                return name
        return None
