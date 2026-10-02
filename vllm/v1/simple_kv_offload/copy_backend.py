# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers."""

from __future__ import annotations

import queue
import threading
import time
from typing import NamedTuple

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    BatchMemcpyParams,
    SwapBlocksParams,
    build_params,
    build_swap_params,
    copy_blocks,
    driver_version,
    get_batch_memcpy_fn,
    pin_tensor,
    swap_copy_blocks,
    unpin_tensor,
)
from vllm.v1.simple_kv_offload.options import test_switch

logger = init_logger(__name__)

# Startup self-test scratch: regions x blocks x bytes per block.
_SELF_TEST_SHAPE = (2, 4, 64 * 1024)
_PAGE_BYTES = 4096


class TransferEvent(NamedTuple):
    """A copy the copy thread has issued on its transfer stream."""

    event_idx: int
    # Timing events recorded right before and right after the copy; the
    # start is behind any stream wait queued before the copy was launched.
    start: torch.Event
    done: torch.Event


class DmaCopyBackend:
    """Block copies issued from a background thread.

    ``mode`` is "batch" (one cuMemcpyBatchAsync per transfer) or "memcpy"
    (one cudaMemcpyAsync per block and region, via ``swap_blocks``).
    """

    def __init__(self) -> None:
        self.mode: str | None = None
        self._store_params: BatchMemcpyParams | SwapBlocksParams | None = None
        self._load_params: BatchMemcpyParams | SwapBlocksParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False
        # Test only: make the next copy raise inside the copy thread.
        self._fail_next_copy: bool = False
        # Copy-thread failure, re-raised on the main thread by check().
        self._error: BaseException | None = None
        # Copies handed to / issued by the copy thread, guarded by _cond.
        self._cond = threading.Condition()
        self._num_enqueued: int = 0
        self._num_issued: int = 0

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        copy_backend: str = "auto",
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream

        self.mode = self._select_mode(copy_backend, device, load_stream, store_stream)
        logger.info(
            "SimpleCPUOffload copy backend: mode=%s (requested %s), CUDA driver %s",
            self.mode,
            copy_backend,
            driver_version(),
        )
        self._store_params = self._make_params(
            self.mode, gpu_caches, cpu_caches, store_stream, is_store=True
        )
        self._load_params = self._make_params(
            self.mode, cpu_caches, gpu_caches, load_stream, is_store=False
        )
        self._fail_next_copy = bool(
            test_switch("VLLM_SIMPLE_KV_OFFLOAD_TEST_FAIL_COPY")
        )
        self._start_thread(device, load_stream, store_stream)

    def _start_thread(
        self,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(self._queue, device, load_stream, store_stream),
            daemon=True,
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[TransferEvent],
    ) -> None:
        self.check()
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        with self._cond:
            self._num_enqueued += 1
        self._queue.put(
            (src_blocks, dst_blocks, params, is_store, event_idx, events_list)
        )

    def check(self) -> None:
        """Re-raise a copy-thread failure on the calling thread."""
        if self._error is not None:
            raise RuntimeError("SimpleCPUOffload copy thread failed") from self._error

    def wait_drained(self, timeout_s: float) -> None:
        """Block until the copy thread has issued every launched copy.

        Afterwards every launched copy has its TransferEvent in its list.
        """
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._num_issued < self._num_enqueued and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        "SimpleCPUOffload copy thread did not issue "
                        f"{self._num_enqueued - self._num_issued} queued "
                        f"copies within {timeout_s:.0f}s"
                    )
                self._cond.wait(remaining)
        self.check()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _copy_loop(
        self,
        q: queue.SimpleQueue,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        # A failure stops the thread; the main thread then fails the step via
        # check() instead of waiting for transfers that will never complete.
        try:
            current_platform.set_device(device)
            while True:
                item = q.get()
                if item is None:
                    return
                (
                    src_blocks,
                    dst_blocks,
                    params,
                    is_store,
                    event_idx,
                    events_list,
                ) = item
                if self._fail_next_copy:
                    self._fail_next_copy = False
                    raise RuntimeError(
                        "injected copy failure (VLLM_SIMPLE_KV_OFFLOAD_TEST_FAIL_COPY)"
                    )
                stream = store_stream if is_store else load_stream
                start = torch.Event(enable_timing=True)
                start.record(stream)
                self._copy(src_blocks, dst_blocks, params)
                done = torch.Event(enable_timing=True)
                done.record(stream)
                events_list.append(TransferEvent(event_idx, start, done))
                # Count the copy only once its event is visible to the worker.
                with self._cond:
                    self._num_issued += 1
                    self._cond.notify_all()
        except BaseException as err:
            logger.exception("SimpleCPUOffload copy thread failed")
            with self._cond:
                self._error = err
                self._cond.notify_all()

    @staticmethod
    def _make_params(
        mode: str,
        src_caches: dict[str, torch.Tensor],
        dst_caches: dict[str, torch.Tensor],
        stream: torch.cuda.Stream,
        is_store: bool,
    ) -> BatchMemcpyParams | SwapBlocksParams:
        if mode == "memcpy":
            return build_swap_params(src_caches, dst_caches, stream)
        # Stores read the live GPU KV cache and must stay stream-ordered;
        # loads read host pinned memory and can use the relaxed order.
        return build_params(
            src_caches, dst_caches, stream, src_access_order_any=not is_store
        )

    @staticmethod
    def _copy(
        src_blocks: list[int],
        dst_blocks: list[int],
        params: BatchMemcpyParams | SwapBlocksParams,
    ) -> None:
        if isinstance(params, SwapBlocksParams):
            swap_copy_blocks(src_blocks, dst_blocks, params)
        else:
            copy_blocks(src_blocks, dst_blocks, params)

    def _select_mode(
        self,
        requested: str,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> str:
        """Pick the copy mode; "auto" uses batch when it passes the self-test.

        A forced mode is never swapped for the other one: it fails startup
        if it does not work.
        """
        failure: str | None
        if requested in ("auto", "batch"):
            if get_batch_memcpy_fn() is None:
                failure = "the batch memcpy entry point is unavailable"
            else:
                failure = self._self_test("batch", device, load_stream, store_stream)
            if failure is None:
                return "batch"
            if requested == "batch":
                raise RuntimeError(
                    f"SimpleCPUOffload copy_backend='batch' is unusable: {failure}"
                )
            logger.warning(
                "SimpleCPUOffload batch copy failed its self-test (%s); "
                "falling back to per-block cudaMemcpyAsync",
                failure,
            )
        failure = self._self_test("memcpy", device, load_stream, store_stream)
        if failure is not None:
            raise RuntimeError(
                f"SimpleCPUOffload memcpy copy self-test failed: {failure}"
            )
        return "memcpy"

    def _self_test(
        self,
        mode: str,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> str | None:
        """Round-trip scratch blocks GPU->CPU->GPU through ``mode``.

        Uses the transfer streams. Returns None on success, else what failed.
        """
        num_regions, num_blocks, block_bytes = _SELF_TEST_SHAPE
        expected = torch.randint(
            -128,
            128,
            _SELF_TEST_SHAPE,
            dtype=torch.int8,
            generator=torch.Generator().manual_seed(0),
        )
        gpu_src = {f"r{i}": expected[i].to(device) for i in range(num_regions)}
        gpu_dst = {name: torch.zeros_like(t) for name, t in gpu_src.items()}
        # Host memory like the tier's (pageable, pinned in place): a single
        # page-aligned registration, so that it shares no page with any
        # other registered range.
        nbytes = num_regions * num_blocks * block_bytes
        raw = torch.zeros(nbytes + _PAGE_BYTES, dtype=torch.int8)
        offset = -raw.data_ptr() % _PAGE_BYTES
        host = raw[offset : offset + nbytes]
        cpu = {
            name: region
            for name, region in zip(
                gpu_src, host.view(num_regions, num_blocks, block_bytes)
            )
        }
        pinned = False
        try:
            if is_pin_memory_available():
                pin_tensor(host)
                pinned = True
            torch.accelerator.synchronize(device)
            # Scrambled CPU slots catch per-block addressing errors.
            gpu_ids = list(range(num_blocks))
            cpu_ids = [2, 0, 3, 1]
            store = self._make_params(mode, gpu_src, cpu, store_stream, True)
            self._copy(gpu_ids, cpu_ids, store)
            store_stream.synchronize()
            for i, name in enumerate(cpu):
                if not torch.equal(cpu[name][cpu_ids], expected[i]):
                    return f"GPU->CPU copy mismatch in scratch region {name}"
            load = self._make_params(mode, cpu, gpu_dst, load_stream, False)
            self._copy(cpu_ids, gpu_ids, load)
            load_stream.synchronize()
            for i, name in enumerate(gpu_dst):
                if not torch.equal(gpu_dst[name].cpu(), expected[i]):
                    return f"CPU->GPU copy mismatch in scratch region {name}"
            return None
        except Exception as e:
            return f"{type(e).__name__}: {e}"
        finally:
            # A failed batch call may have issued part of its copies: never
            # unregister memory a copy can still be using.
            store_stream.synchronize()
            load_stream.synchronize()
            if pinned:
                unpin_tensor(host)
