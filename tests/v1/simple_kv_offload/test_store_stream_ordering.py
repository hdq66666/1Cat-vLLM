# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Both transfer directions race the compute stream unless ordered behind it.

A store reads the live GPU KV cache: it must be ordered behind the compute
stream and keep STREAM source-access ordering, or the driver is free to read
blocks the compute stream has not finished writing -- which silently corrupts
the offloaded copy while its block hash marks it valid. A load reads host
pinned memory (so ANY source order is fine) but writes GPU blocks the step may
still zero or, through pages freed while an earlier step was in flight, still
read: it must be ordered behind the compute stream too.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload import cuda_mem_ops
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

CU_MEMCPY_SRC_ACCESS_ORDER_STREAM = 1
CU_MEMCPY_SRC_ACCESS_ORDER_ANY = 3


def _caches() -> dict[str, torch.Tensor]:
    return {"layer.0": torch.zeros(4, 16, dtype=torch.int8)}


@pytest.mark.parametrize(
    "src_access_order_any,expected",
    [
        (False, CU_MEMCPY_SRC_ACCESS_ORDER_STREAM),
        (True, CU_MEMCPY_SRC_ACCESS_ORDER_ANY),
    ],
)
def test_build_params_source_access_order(
    src_access_order_any: bool, expected: int
) -> None:
    stream = SimpleNamespace(cuda_stream=0)

    params = cuda_mem_ops.build_params(
        _caches(),
        _caches(),
        stream,  # type: ignore[arg-type]
        src_access_order_any=src_access_order_any,
    )

    assert params.attrs.srcAccessOrder == expected


def _worker(recorder: mock.Mock) -> SimpleCPUOffloadWorker:
    worker = SimpleCPUOffloadWorker(
        SimpleNamespace(),  # type: ignore[arg-type]
        kv_cache_config=SimpleNamespace(num_blocks=16),  # type: ignore[arg-type]
        cpu_capacity_bytes=0,
    )
    worker.num_cpu_blocks = 8
    worker.load_stream = recorder.load_stream
    worker.store_stream = recorder.store_stream
    worker._backend = recorder.backend
    return worker


def _assert_waits_before_copy(recorder: mock.Mock, stream: str) -> None:
    ordered = [call[0] for call in recorder.mock_calls]
    assert f"{stream}.wait_stream" in ordered
    assert "backend.launch_copy" in ordered
    assert ordered.index(f"{stream}.wait_stream") < ordered.index(
        "backend.launch_copy"
    ), f"copy was queued before the barrier: {ordered}"


def test_store_is_ordered_behind_the_compute_stream() -> None:
    """The transfer stream waits on compute before the copy is queued."""
    recorder = mock.Mock()
    worker = _worker(recorder)
    worker.bind_connector_metadata(
        SimpleCPUOffloadMetadata(
            store_event=0, store_gpu_blocks=[7], store_cpu_blocks=[3]
        )
    )

    worker.get_finished(set())

    _assert_waits_before_copy(recorder, "store_stream")
    recorder.load_stream.wait_stream.assert_not_called()


def test_load_is_ordered_behind_the_compute_stream() -> None:
    """Loads overwrite blocks the compute stream may still zero or read."""
    recorder = mock.Mock()
    worker = _worker(recorder)
    worker.bind_connector_metadata(
        SimpleCPUOffloadMetadata(load_event=0, load_gpu_blocks=[7], load_cpu_blocks=[3])
    )

    worker.get_finished(set())

    _assert_waits_before_copy(recorder, "load_stream")
    recorder.store_stream.wait_stream.assert_not_called()


@pytest.mark.parametrize("test_mode", [True, False])
def test_unordered_load_switch_needs_test_mode(
    monkeypatch: pytest.MonkeyPatch, test_mode: bool
) -> None:
    """The negative control drops the load barrier, but only in test mode."""
    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_MODE", "1" if test_mode else "0")
    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_UNORDERED_LOAD", "1")
    recorder = mock.Mock()
    worker = _worker(recorder)
    worker.bind_connector_metadata(
        SimpleCPUOffloadMetadata(load_event=0, load_gpu_blocks=[7], load_cpu_blocks=[3])
    )

    worker.get_finished(set())

    recorder.backend.launch_copy.assert_called_once()
    if test_mode:
        recorder.load_stream.wait_stream.assert_not_called()
    else:
        _assert_waits_before_copy(recorder, "load_stream")
