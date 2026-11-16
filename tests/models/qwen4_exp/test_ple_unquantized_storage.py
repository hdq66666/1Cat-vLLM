# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unquantized (BF16/FP16) PLE tables on the device and pinned-host tiers.

Coverage: placement counts bytes of the storage dtype, the FP8-only disk tier
is rejected before any table is built, the SM70 Qwen3.8 runtime defaults keep
such checkpoints off the disk/hybrid PLE lane, and (on CUDA) the raw gather
kernel returns exactly the stored rows from device and pinned-host memory.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.models.runtime_defaults as model_defaults
from vllm.models.qwen4_exp.nvidia import ple_layer as ple

Embedding = ple.Qwen4ExpPinnedHostEmbedding
ROWS, DIM = 100, 4


def _raw_table(element_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        _meta_weight_shape=(ROWS, DIM),
        _meta_weight_dtype=torch.float16,
        embedding_dim=DIM,
        _row_bytes=DIM * element_size,
        _is_fp8_storage=False,
    )


def test_unquantized_placement_counts_storage_bytes(monkeypatch):
    # An FP16 row is twice the bytes of an FP8 one; counting one byte per
    # element would put twice the requested budget into pinned host memory.
    monkeypatch.setattr(ple, "ple_host_budget_bytes", lambda: 200)
    monkeypatch.setattr(ple, "ple_cascade_configured", lambda: False)
    monkeypatch.setattr(ple, "get_current_vllm_config_or_none", lambda: None)
    placement = Embedding._plan_placement(_raw_table(), torch.device("cuda:0"))
    assert placement.host_rows == 200 // (DIM * 2)
    assert placement.disk_rows == 0


def test_unquantized_storage_rejects_the_disk_cascade(monkeypatch):
    monkeypatch.setattr(ple, "ple_cascade_configured", lambda: True)
    monkeypatch.setattr(ple, "get_current_vllm_config_or_none", lambda: None)
    with pytest.raises(NotImplementedError, match="device and pinned-host"):
        Embedding._plan_placement(_raw_table(), torch.device("cuda:0"))


def test_unquantized_storage_rejects_disk_only_placement(monkeypatch):
    monkeypatch.setattr(ple, "ple_cascade_configured", lambda: False)
    monkeypatch.setattr(
        ple,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(kernel_config=SimpleNamespace(ple_disk_only=True)),
    )
    with pytest.raises(NotImplementedError, match="ple_disk_only"):
        Embedding._plan_placement(_raw_table(), torch.device("cuda:0"))


def _cfg(ple_layer_ids):
    text = SimpleNamespace(ple_layer_ids=ple_layer_ids)
    return SimpleNamespace(model_config=SimpleNamespace(hf_text_config=text))


def test_resident_ple_guard_ignores_models_without_ple(monkeypatch):
    def unexpected(_cfg):
        raise AssertionError("storage is only inspected for PLE models")

    monkeypatch.setattr(model_defaults, "ple_storage_format_rejection", unexpected)
    for lane in ("dual_compile", "disk_ple"):
        assert model_defaults.resident_ple_lane_rejection(_cfg(None), lane) is None
        assert model_defaults.resident_ple_lane_rejection(_cfg([]), lane) is None


@pytest.mark.parametrize("lane", ["dual_compile", "disk_ple"])
@pytest.mark.parametrize("reason", [None, "no raw E4M3 PLE storage"])
def test_resident_ple_guard_follows_the_storage_format(monkeypatch, lane, reason):
    logged: list[str] = []
    monkeypatch.setattr(
        model_defaults, "ple_storage_format_rejection", lambda _c: reason
    )
    monkeypatch.setattr(
        model_defaults.logger,
        "info_once",
        lambda msg, *args, **_kw: logged.append(msg % args),
    )
    got = model_defaults.resident_ple_lane_rejection(_cfg([3, 7]), lane)
    assert got == reason
    # A rejection is visible in the log with its reason; acceptance is silent.
    assert logged == ([] if reason is None else [logged[0]])
    if reason is not None:
        assert reason in logged[0]


def test_ple_storage_rejection_keeps_the_partition_check_first(monkeypatch):
    # The lane guard reads only the storage format; the cascade keeps the
    # pipeline-partition check ahead of it.
    import vllm.models.qwen4_exp.common.ple as common_ple

    cfg = _cfg([3, 7])
    cfg.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    monkeypatch.setattr(
        model_defaults, "ple_storage_format_rejection", lambda _c: "format"
    )

    def off_first_rank(_text, _pp):
        raise ValueError("PLE layers must sit on the first PP rank")

    monkeypatch.setattr(common_ple, "check_ple_layers_on_first_pp_rank", off_first_rank)
    assert model_defaults.ple_storage_rejection(cfg) == (
        "PLE layers must sit on the first PP rank"
    )
    monkeypatch.setattr(
        common_ple, "check_ple_layers_on_first_pp_rank", lambda _t, _p: None
    )
    assert model_defaults.ple_storage_rejection(cfg) == "format"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pinned_host", [False, True])
def test_raw_gather_kernel_returns_stored_rows(dtype, pinned_host):
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    rows, dim = 4096, 160
    table = torch.randn(rows, dim, dtype=torch.float32).to(dtype)
    if pinned_host:
        table = table.pin_memory()
        source = get_accelerator_view_from_cpu_tensor(table)
    else:
        source = table.cuda()
    ids = torch.randint(0, rows, (513,), device="cuda", dtype=torch.int64)
    out = torch.empty((ids.numel(), dim), dtype=dtype, device="cuda")
    ple._gather_ple_raw_from_pinned_kernel[(ids.numel(),)](
        source.data_ptr(),
        ids,
        out,
        embedding_dim=dim,
        BLOCK_D=256,
        STORAGE_DTYPE=ple._TRITON_STORAGE_DTYPES[dtype],
        num_warps=4,
    )
    torch.accelerator.synchronize()
    expected = table.cuda().index_select(0, ids)
    assert torch.equal(out, expected)
