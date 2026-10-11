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
from vllm.v1.simple_kv_offload.fingerprint import FingerprintChain
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


# Steps between "SimpleCPU control status" log lines while a control is on.
_CONTROL_STATUS_EVERY = 200


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
        # Diagnostic block probe (options.probe_block): per-region byte sums of
        # one block, taken on the compute stream at the end of every step and
        # read back one step later through a pinned ring, so a foreign write
        # into that block is reported without host synchronization.
        self._probe_step = 0
        self._probe_calls = 0
        self._probe_names: list[str] = []
        self._probe_pinned: list[torch.Tensor] = []
        self._probe_events: list[torch.cuda.Event | None] = []
        self._probe_steps: list[int] = []
        self._probe_last: torch.Tensor | None = None
        self._probe_disabled = False
        # Diagnostic fingerprint chain (options.fingerprint).
        self._fp: FingerprintChain | None = None
        # Gated loads (request directive "gate"): event -> (deadline on the
        # worker clock by which the chain's verdict must have arrived, GPU
        # blocks); a completed gated load is reported to the scheduler only
        # with its verdict.
        self._gated_loads: dict[int, tuple[float, list[int]]] = {}
        # GPU blocks of failed gated loads, reported once through
        # get_block_ids_with_load_errors().
        self._invalid_block_ids: set[int] = set()
        # Runtime controls applied through request directives, name -> value.
        self._controls: dict[str, str] = {}
        self._control_calls = 0

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
        if self.options.fingerprint:
            try:
                gpu_views, cpu_views, region_groups, state_groups = self._typed_views(
                    kv_caches, seen_ptrs
                )
                self._fp = FingerprintChain(
                    self.gpu_kv_caches,
                    self.cpu_kv_caches,
                    rank,
                    inject_gpu_load=self.options.fingerprint_inject_gpu_load,
                    inject_cpu_load=self.options.fingerprint_inject_cpu_load,
                    corrupt_plan=self.options.fingerprint_corrupt_plan,
                    gpu_views=gpu_views,
                    cpu_views=cpu_views,
                    region_groups=region_groups,
                    state_groups=state_groups,
                )
                self._backend.fp = self._fp
            except Exception:
                logger.exception("SimpleCPU fp could not be enabled")
                self._fp = None

    def _typed_views(
        self,
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
        seen_ptrs: dict[int, tuple[str, torch.Tensor]],
    ) -> tuple[
        dict[str, list[tuple[str, torch.Tensor]]],
        dict[str, list[tuple[str, torch.Tensor]]],
        dict[str, int],
        set[int],
    ]:
        """Typed [blocks, ...] views of every registered region and of its
        CPU mirror, tagged with what the region holds (fingerprint
        REGION_PARTS): K and V of an attention layer whose cache is laid out
        (blocks, 2, ...) or (2, blocks, ...), a whole attention region
        otherwise, or each state tensor of a state-space layer. Also the KV
        cache group of every region and the groups of the state-space
        layers, so the chain counts a block only in its owner's regions.
        Without KV cache groups the layout decides (a list of tensors is a
        state-space layer): attention regions are group 0, state regions
        group 1. Views never copy: a layout that cannot be viewed raises."""
        from vllm.v1.kv_cache_interface import MambaSpec

        assert self.gpu_kv_caches is not None and self.cpu_kv_caches is not None
        group_of: dict[str, int] = {}
        groups = self.kv_cache_config.kv_cache_groups if self.kv_cache_config else []
        for g, group in enumerate(groups):
            for layer_name in group.layer_names:
                group_of[layer_name] = g
        state_groups = {
            g
            for g, group in enumerate(groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
        }
        gpu_views: dict[str, list[tuple[str, torch.Tensor]]] = {}
        cpu_views: dict[str, list[tuple[str, torch.Tensor]]] = {}
        region_groups: dict[str, int] = {}
        num_blocks = self.gpu_kv_caches[next(iter(self.gpu_kv_caches))].shape[0]
        for name, tensor in seen_ptrs.values():
            value = kv_caches[name]
            gi = group_of.get(name)
            spec = groups[gi].kv_cache_spec if gi is not None else None
            is_state = (
                isinstance(spec, MambaSpec)
                if spec is not None
                else isinstance(value, list)
            )
            if gi is None:
                gi = 1 if is_state else 0
                if is_state:
                    state_groups.add(1)
            if is_state:
                state = value if isinstance(value, list) else [value]
                raw = self.gpu_kv_caches.get(name)
                if raw is None:
                    continue
                cpu_raw = self.cpu_kv_caches[name]
                parts: list[tuple[str, torch.Tensor]] = []
                cparts: list[tuple[str, torch.Tensor]] = []
                for t in state:
                    if t.shape[0] != num_blocks:
                        continue
                    parts.append(("state", t))
                    # The same strided layout over the CPU rows: state
                    # tensors are interleaved per block in the raw storage.
                    cparts.append(
                        (
                            "state",
                            torch.as_strided(
                                cpu_raw.view(t.dtype),
                                (cpu_raw.shape[0], *t.shape[1:]),
                                t.stride(),
                                t.storage_offset(),
                            ),
                        )
                    )
                gpu_views[name] = parts
                cpu_views[name] = cparts
                region_groups[name] = gi
                continue
            assert isinstance(value, torch.Tensor)
            if (
                value.dim() >= 2
                and value.shape[0] == 2
                and value.shape[1] == num_blocks
            ):
                # (2, blocks, ...): two registered regions, K and V.
                for idx, part in enumerate(("k", "v")):
                    region = f"{name}.{idx}"
                    if region in self.gpu_kv_caches:
                        gpu_views[region] = [(part, value[idx])]
                        cpu = self.cpu_kv_caches[region]
                        cpu_views[region] = [
                            (part, cpu.view(value.dtype).view(cpu.shape[0], -1))
                        ]
                        region_groups[region] = gi
                continue
            if name not in self.gpu_kv_caches:
                continue
            cpu = self.cpu_kv_caches[name]
            if (
                value.dim() >= 2
                and value.shape[0] == num_blocks
                and value.shape[1] == 2
            ):
                # (blocks, 2, ...): K and V halves of one region.
                flat = value.view(num_blocks, 2, -1)
                cflat = cpu.view(value.dtype).view(cpu.shape[0], 2, -1)
                gpu_views[name] = [("k", flat[:, 0]), ("v", flat[:, 1])]
                cpu_views[name] = [("k", cflat[:, 0]), ("v", cflat[:, 1])]
            else:
                gpu_views[name] = [("attn", value.view(num_blocks, -1))]
                cpu_views[name] = [
                    ("attn", cpu.view(value.dtype).view(cpu.shape[0], -1))
                ]
            region_groups[name] = gi
        return gpu_views, cpu_views, region_groups, state_groups

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
        if metadata.controls:
            self._apply_controls(metadata.controls)

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def _apply_controls(self, controls: dict[str, str]) -> None:
        """Runtime controls from request directives (experiments): flip the
        SM70 QSA grouped page4 route or its padding fix, or the NaN count
        of the logits, in this worker process without a restart."""
        import importlib
        import os

        for name, value in controls.items():
            if self._controls.get(name) == value:
                continue
            if name in ("qsa_grouped_page4", "qsa_grouped_pad_fix"):
                qsa = importlib.import_module("vllm.models.qwen4_exp.nvidia.ops.qsa")
                attr = (
                    "_SM70_QSA_GROUPED_PAGE4"
                    if name == "qsa_grouped_page4"
                    else "_SM70_QSA_GROUPED_PAD_FIX"
                )
                setattr(qsa, attr, value == "1")
            elif name == "compute_nans_in_logits":
                os.environ["VLLM_COMPUTE_NANS_IN_LOGITS"] = value
            else:
                logger.warning("SimpleCPU control %s=%s: unknown, ignored", name, value)
                continue
            self._controls[name] = value
            logger.warning(
                "SimpleCPU control applied rank=%d %s=%s %s",
                _global_rank(),
                name,
                value,
                self._control_status(),
            )

    @staticmethod
    def _control_status() -> str:
        """The SM70 QSA grouped page4 route's call count and switches: the
        count stops moving once the route is off."""
        import importlib

        try:
            qsa = importlib.import_module("vllm.models.qwen4_exp.nvidia.ops.qsa")
        except Exception:
            return "qsa=unavailable"
        return (
            f"grouped_page4_calls={getattr(qsa, '_SM70_QSA_GROUPED_PAGE4_CALLS', '?')}"
            f" grouped={getattr(qsa, '_SM70_QSA_GROUPED_PAGE4', '?')}"
            f" pad_fix={getattr(qsa, '_SM70_QSA_GROUPED_PAD_FIX', '?')}"
        )

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

        self._run_block_probe(metadata)
        if self._fp is not None:
            self._fp.tick(torch.cuda.current_stream())
        if self._controls:
            self._control_calls += 1
            if self._control_calls % _CONTROL_STATUS_EVERY == 0:
                logger.info(
                    "SimpleCPU control status rank=%d %s %s",
                    _global_rank(),
                    self._controls,
                    self._control_status(),
                )
        if self.options.debug_gpu_sleep_cycles > 0:
            torch.cuda._sleep(self.options.debug_gpu_sleep_cycles)
        if self.options.debug_host_sleep_ms > 0:
            time.sleep(self.options.debug_host_sleep_ms / 1000.0)

        # (2) Track completed transfer events
        finished_recving: set[str] = set()

        if self._pending_load_event_indices or self._load_launches:
            load_wm = self._poll_stream_events(is_store=False)
            for j in sorted(
                j for j in self._pending_load_event_indices if j <= load_wm
            ):
                if j in self._gated_loads and not self._gate_verdict(j, metadata):
                    # Verdict still pending: the request waits one more step.
                    continue
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

    def _run_block_probe(self, metadata: SimpleCPUOffloadMetadata | None) -> None:
        """See options.probe_block. Runs after this step's transfers were
        submitted; the fingerprint is taken on the compute stream, i.e. it
        reflects the block after this step's forward pass."""
        block_id = self.options.probe_block
        if block_id is None or self._probe_disabled or self.gpu_kv_caches is None:
            return
        self._probe_calls += 1
        if (self._probe_calls - 1) % max(1, self.options.probe_every) != 0:
            return
        try:
            self._probe_step += 1
            step = self._probe_step
            if not self._probe_names:
                self._probe_names = list(self.gpu_kv_caches)
                for _ in range(4):
                    self._probe_pinned.append(
                        torch.zeros(
                            len(self._probe_names), dtype=torch.int64
                        ).pin_memory()
                    )
                    self._probe_events.append(None)
                    self._probe_steps.append(0)
            slot = step % 4
            # Read back the fingerprint taken 4 steps ago (its event is done
            # by now; if not, skip it rather than wait).
            prev_event = self._probe_events[slot]
            if prev_event is not None and prev_event.query():
                current = self._probe_pinned[slot].clone()
                if self._probe_last is not None and not torch.equal(
                    current, self._probe_last
                ):
                    changed = [
                        f"{name}:{int(a)}->{int(b)}"
                        for name, a, b in zip(
                            self._probe_names, self._probe_last, current
                        )
                        if a != b
                    ]
                    logger.warning(
                        "SimpleCPUOffload block probe: GPU block %d changed at "
                        "probe step %d (rank %d): %s",
                        block_id,
                        self._probe_steps[slot],
                        _global_rank(),
                        ", ".join(changed[:8]),
                    )
                self._probe_last = current
            sums = torch.stack(
                [
                    region[block_id]
                    .reshape(-1)
                    .view(torch.uint8)
                    .sum(dtype=torch.int64)
                    for region in self.gpu_kv_caches.values()
                ]
            )
            self._probe_pinned[slot].copy_(sums, non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            self._probe_events[slot] = event
            self._probe_steps[slot] = step
        except Exception:
            logger.exception("SimpleCPUOffload block probe disabled after an error")
            self._probe_disabled = True

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
        metadata = self._connector_metadata
        if self._fp is not None:
            # Before the copy is queued, so the checker sees the launch first.
            if is_store:
                kinds = metadata.store_block_kinds if metadata is not None else []
                self._fp.store_launched(
                    event_idx,
                    gpu_blocks,
                    cpu_blocks,
                    torch.cuda.current_stream(),
                    kinds=list(kinds) if len(kinds) == len(gpu_blocks) else None,
                )
            else:
                kinds = metadata.load_block_kinds if metadata is not None else []
                corrupt = metadata.load_corrupt.get(event_idx) if metadata else None
                spans = (
                    [(start, end, mode) for start, end, mode in corrupt]
                    if corrupt
                    else None
                )
                self._fp.load_launched(
                    event_idx,
                    cpu_blocks,
                    gpu_blocks,
                    kinds=list(kinds) if len(kinds) == len(gpu_blocks) else None,
                    corrupt=spans,
                )
                if metadata is not None and metadata.load_gate.get(event_idx):
                    # The deadline starts when the copy completes.
                    self._gated_loads[event_idx] = (-1.0, list(gpu_blocks))
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

    def _gate_verdict(
        self, event_idx: int, metadata: SimpleCPUOffloadMetadata | None
    ) -> bool:
        """Whether a completed gated load may be reported to the scheduler.
        With a verdict naming bad blocks, the blocks of every request whose
        span holds one are reported as failed loads; without a verdict the
        load is held back until the gate timeout."""
        now = self._clock()
        deadline, gpu_blocks = self._gated_loads[event_idx]
        if deadline < 0:
            deadline = now + self.options.fingerprint_gate_timeout_s
            self._gated_loads[event_idx] = (deadline, gpu_blocks)
        verdict = self._fp.pop_load_verdict(event_idx) if self._fp else []
        if verdict is None:
            if now < deadline:
                return False
            logger.warning(
                "SimpleCPU gate rank=%d load event=%d: no fingerprint verdict "
                "within %.1fs, reported as loaded",
                _global_rank(),
                event_idx,
                self.options.fingerprint_gate_timeout_s,
            )
            verdict = []
        del self._gated_loads[event_idx]
        if not verdict:
            return True
        spans = metadata.load_spans.get(event_idx) if metadata is not None else None
        bad: set[int] = set()
        if spans and gpu_blocks:
            for start, end in spans:
                if any(start <= pos < end for pos in verdict):
                    bad.update(gpu_blocks[start:end])
        elif gpu_blocks:
            bad.update(gpu_blocks)
        self._invalid_block_ids.update(bad)
        logger.warning(
            "SimpleCPU gate rank=%d load event=%d FAILED: %d bad blocks at "
            "positions=%s; %d GPU blocks reported for recompute",
            _global_rank(),
            event_idx,
            len(verdict),
            verdict[:16],
            len(bad),
        )
        return True

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid, self._invalid_block_ids = self._invalid_block_ids, set()
        return invalid

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
        if self._fp is not None and not is_store:
            self._fp.load_seen_complete(transfer.event_idx)
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
