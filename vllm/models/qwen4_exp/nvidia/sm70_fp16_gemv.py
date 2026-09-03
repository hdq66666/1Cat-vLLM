# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in checkpoint-FP16 Qwen3.8 decode GEMV kernels for SM70.

The route is deliberately narrow: exact Qwen3.8 Flash Next topology, TP4,
no speculative decoding, FP16 checkpoint weights, and one decode token. All
prefill and unsupported shapes retain the ordinary unquantized linear path.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


class _GemvPlan(NamedTuple):
    block_k: int
    num_warps: int
    load_policy: int
    # Largest M for which the tiled route still beats cuBLAS on SM70,
    # measured per role shape.  M above this falls back to F.linear.
    max_m: int = 1


_HC_DOWN_SUFFIX = ".input_mix_weight_down_block_inject"
_GDN_QKVZ_SUFFIX = ".linear_attn.in_proj_qkvz"
_GDN_BA_SUFFIX = ".linear_attn.in_proj_ba"
_GDN_OUT_SUFFIX = ".linear_attn.out_proj"
_QSA_QKV_SUFFIX = ".self_attn.qkv_proj"
_QSA_OUT_SUFFIX = ".self_attn.o_proj"
_QSA_INDEX_SUFFIX = ".self_attn.indexer.index_qk_proj"
_ROUTER_SUFFIX = ".mlp.gate"

# Plans are cold-cache CUDA Graph winners on real checkpoint weights. Keep the
# role in the key: GDN and QSA can share a physical shape while remaining
# independently auditable.
_ROLE_PLANS: tuple[tuple[str, tuple[int, int], _GemvPlan], ...] = (
    # max_m measured on 4xV100 (micro_m2d.py), tiled-vs-cuBLAS ratio in comment
    (_HC_DOWN_SUFFIX, (336, 10240), _GemvPlan(512, 4, 1, 2)),   # M3 0.95x
    (_GDN_QKVZ_SUFFIX, (4096, 2560), _GemvPlan(512, 2, 0, 4)),  # M4 1.12x
    (_GDN_BA_SUFFIX, (24, 2560), _GemvPlan(512, 4, 0, 16)),     # M16 1.63x
    (_GDN_OUT_SUFFIX, (2560, 1536), _GemvPlan(512, 4, 1, 8)),   # M8 1.14x
    (_QSA_QKV_SUFFIX, (3584, 2560), _GemvPlan(512, 2, 0, 8)),   # M8 1.09x
    (_QSA_OUT_SUFFIX, (2560, 1536), _GemvPlan(512, 4, 0, 8)),   # M8 1.08x
    (_QSA_INDEX_SUFFIX, (640, 2560), _GemvPlan(512, 2, 0, 16)), # M16 1.14x
    (_ROUTER_SUFFIX, (512, 2560), _GemvPlan(1024, 8, 0, 16)),   # M16 1.36x
)

# The fused GDN input op drives (4096, 2560) and (24, 2560) together, so it
# inherits the tighter of the two ceilings.
_GDN_FUSED_MAX_M = 4

_SHAPE_PLANS = {shape: plan for _, shape, plan in _ROLE_PLANS}


@triton.jit
def _qwen38_fp16_row_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOAD_POLICY: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        if LOAD_POLICY == 1:
            x = tl.load(
                x_ptr + indices,
                mask=mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            weight = tl.load(
                weight_ptr + row * K + indices,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            )
        else:
            x = tl.load(x_ptr + indices, mask=mask, other=0.0)
            weight = tl.load(
                weight_ptr + row * K + indices,
                mask=mask,
                other=0.0,
            )
        acc += x.to(tl.float32) * weight.to(tl.float32)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))


@triton.jit
def _qwen38_fp16_row_gemv_m_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOAD_POLICY: tl.constexpr,
):
    """Multi-token sibling of the M=1 row GEMV.

    Grid stays (N,): one program still owns one output row and streams the
    whole weight row exactly once, now reusing each loaded block across the
    BLOCK_M query rows.  That keeps HBM weight traffic identical to M=1.
    """
    row = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offs_k
        k_mask = indices < K
        x_mask = m_mask[:, None] & k_mask[None, :]
        if LOAD_POLICY == 1:
            weight = tl.load(
                weight_ptr + row * K + indices,
                mask=k_mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            x = tl.load(
                x_ptr + offs_m[:, None] * K + indices[None, :],
                mask=x_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
        else:
            weight = tl.load(weight_ptr + row * K + indices, mask=k_mask, other=0.0)
            x = tl.load(
                x_ptr + offs_m[:, None] * K + indices[None, :],
                mask=x_mask,
                other=0.0,
            )
        acc += x.to(tl.float32) * weight.to(tl.float32)[None, :]
    tl.store(out_ptr + offs_m * N + row, tl.sum(acc, axis=1), mask=m_mask)


@triton.jit
def _qwen38_fp16_gdn_input_kernel(
    x_ptr,
    qkvz_weight_ptr,
    ba_weight_ptr,
    qkv_out_ptr,
    z_out_ptr,
    b_out_ptr,
    a_out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    is_qkvz = row < 4096
    ba_row = row - 4096
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offsets
        mask = indices < K
        x = tl.load(x_ptr + indices, mask=mask, other=0.0)
        qkvz_weight = tl.load(
            qkvz_weight_ptr + row * K + indices,
            mask=is_qkvz & mask,
            other=0.0,
        )
        ba_weight = tl.load(
            ba_weight_ptr + ba_row * K + indices,
            mask=(~is_qkvz) & mask,
            other=0.0,
        )
        weight = tl.where(is_qkvz, qkvz_weight, ba_weight)
        acc += x.to(tl.float32) * weight.to(tl.float32)

    value = tl.sum(acc, axis=0)
    is_qkv = is_qkvz & (row < 2560)
    is_z = is_qkvz & (row >= 2560)
    is_b = (~is_qkvz) & (ba_row < 12)
    is_a = (~is_qkvz) & (ba_row >= 12)
    tl.store(qkv_out_ptr + row, value, mask=is_qkv)
    tl.store(z_out_ptr + row - 2560, value, mask=is_z)
    tl.store(b_out_ptr + ba_row, value, mask=is_b)
    tl.store(a_out_ptr + ba_row - 12, value, mask=is_a)


@triton.jit
def _qwen38_fp16_gdn_input_m_kernel(
    x_ptr,
    qkvz_weight_ptr,
    ba_weight_ptr,
    qkv_out_ptr,
    z_out_ptr,
    b_out_ptr,
    a_out_ptr,
    M,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Multi-token sibling of the fused GDN input projection."""
    row = tl.program_id(0)
    is_qkvz = row < 4096
    ba_row = row - 4096
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        indices = block_start + offs_k
        k_mask = indices < K
        x = tl.load(
            x_ptr + offs_m[:, None] * K + indices[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        qkvz_weight = tl.load(
            qkvz_weight_ptr + row * K + indices,
            mask=is_qkvz & k_mask,
            other=0.0,
        )
        ba_weight = tl.load(
            ba_weight_ptr + ba_row * K + indices,
            mask=(~is_qkvz) & k_mask,
            other=0.0,
        )
        weight = tl.where(is_qkvz, qkvz_weight, ba_weight)
        acc += x.to(tl.float32) * weight.to(tl.float32)[None, :]

    value = tl.sum(acc, axis=1)
    is_qkv = is_qkvz & (row < 2560)
    is_z = is_qkvz & (row >= 2560)
    is_b = (~is_qkvz) & (ba_row < 12)
    is_a = (~is_qkvz) & (ba_row >= 12)
    tl.store(qkv_out_ptr + offs_m * 2560 + row, value, mask=m_mask & is_qkv)
    tl.store(z_out_ptr + offs_m * 1536 + row - 2560, value, mask=m_mask & is_z)
    tl.store(b_out_ptr + offs_m * 12 + ba_row, value, mask=m_mask & is_b)
    tl.store(a_out_ptr + offs_m * 12 + ba_row - 12, value, mask=m_mask & is_a)


def _is_packed_row_major(tensor: torch.Tensor) -> bool:
    return tensor.ndim == 2 and tensor.stride() == (tensor.shape[1], 1)


def _runtime_ok(x: torch.Tensor, weight: torch.Tensor, max_m: int = 1) -> bool:
    return bool(
        not envs.VLLM_BATCH_INVARIANT
        and 1 <= x.shape[0] <= max_m
        and _is_packed_row_major(x)
        and _is_packed_row_major(weight)
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.shape[1] == weight.shape[1]
    )


def _qwen38_sm70_fp16_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    plan = _SHAPE_PLANS.get((weight.shape[0], weight.shape[1]))
    if plan is None or not _runtime_ok(x, weight, plan.max_m):
        return torch.nn.functional.linear(x, weight)

    rows = int(x.shape[0])
    n = int(weight.shape[0])
    if rows == 1:
        # Untouched single-token route: identical kernel, identical launch.
        out = torch.empty((1, n), dtype=x.dtype, device=x.device)
        _qwen38_fp16_row_gemv_kernel[(n,)](
            x,
            weight,
            out,
            K=weight.shape[1],
            BLOCK_K=plan.block_k,
            LOAD_POLICY=plan.load_policy,
            num_warps=plan.num_warps,
        )
        logger.info_once("SM70 Qwen3.8 checkpoint-FP16 M=1 GEMV route enabled.")
        return out

    out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    _qwen38_fp16_row_gemv_m_kernel[(n,)](
        x,
        weight,
        out,
        rows,
        N=n,
        K=weight.shape[1],
        BLOCK_M=triton.next_power_of_2(rows),
        BLOCK_K=plan.block_k,
        LOAD_POLICY=plan.load_policy,
        num_warps=plan.num_warps,
    )
    logger.info_once("SM70 Qwen3.8 checkpoint-FP16 M>1 tiled GEMV route enabled.")
    return out


def _qwen38_sm70_fp16_gemv_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_gemv",
    op_func=_qwen38_sm70_fp16_gemv,
    fake_impl=_qwen38_sm70_fp16_gemv_fake,
)


def _qwen38_sm70_fp16_gdn_input(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not (
        qkvz_weight.shape == (4096, 2560)
        and ba_weight.shape == (24, 2560)
        and _runtime_ok(x, qkvz_weight, _GDN_FUSED_MAX_M)
        and _runtime_ok(x, ba_weight, _GDN_FUSED_MAX_M)
    ):
        qkvz = torch.nn.functional.linear(x, qkvz_weight)
        ba = torch.nn.functional.linear(x, ba_weight)
        return (
            qkvz[..., :2560].contiguous(),
            qkvz[..., 2560:].contiguous(),
            ba[..., :12].contiguous(),
            ba[..., 12:].contiguous(),
        )

    rows = int(x.shape[0])
    qkv = x.new_empty((rows, 2560))
    z = x.new_empty((rows, 1536))
    b = x.new_empty((rows, 12))
    a = x.new_empty((rows, 12))
    if rows == 1:
        _qwen38_fp16_gdn_input_kernel[(4096 + 24,)](
            x,
            qkvz_weight,
            ba_weight,
            qkv,
            z,
            b,
            a,
            K=2560,
            BLOCK_K=512,
            num_warps=2,
        )
        logger.info_once(
            "SM70 Qwen3.8 checkpoint-FP16 fused GDN input route enabled."
        )
        return qkv, z, b, a

    _qwen38_fp16_gdn_input_m_kernel[(4096 + 24,)](
        x,
        qkvz_weight,
        ba_weight,
        qkv,
        z,
        b,
        a,
        rows,
        K=2560,
        BLOCK_M=triton.next_power_of_2(rows),
        BLOCK_K=512,
        num_warps=2,
    )
    logger.info_once(
        "SM70 Qwen3.8 checkpoint-FP16 fused GDN input M>1 tiled route enabled."
    )
    return qkv, z, b, a


def _qwen38_sm70_fp16_gdn_input_fake(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del qkvz_weight, ba_weight
    batch_shape = x.shape[:-1]
    return (
        x.new_empty((*batch_shape, 2560)),
        x.new_empty((*batch_shape, 1536)),
        x.new_empty((*batch_shape, 12)),
        x.new_empty((*batch_shape, 12)),
    )


direct_register_custom_op(
    op_name="qwen38_sm70_fp16_gdn_input",
    op_func=_qwen38_sm70_fp16_gdn_input,
    fake_impl=_qwen38_sm70_fp16_gdn_input_fake,
)


class Qwen38SM70FP16LinearMethod(UnquantizedLinearMethod):
    """Use the row-GEMV custom op for admitted single-token projections."""

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The first dynamic compile sees prefill (M > 1). Keep the M=1
        # decision inside the opaque op so decode does not inherit a baked-in
        # prefill branch.
        if bias is None:
            return torch.ops.vllm.qwen38_sm70_fp16_gemv(x, layer.weight)
        return super().apply(layer, x, bias)


def _plan_for(prefix: str, shape: tuple[int, int]) -> _GemvPlan | None:
    for suffix, expected_shape, plan in _ROLE_PLANS:
        if prefix.endswith(suffix) and shape == expected_shape:
            return plan
    return None


def _exact_runtime_contract(vllm_config=None) -> bool:
    try:
        config = vllm_config or get_current_vllm_config()
        text_config = config.model_config.hf_text_config
        tp_size = int(config.parallel_config.tensor_parallel_size)
    except (AssertionError, AttributeError, RuntimeError):
        return False

    return bool(
        tp_size == 4
        # The M=1 restriction that this clause used to stand in for now lives
        # in the kernels themselves (per-shape max_m), so speculative decoding
        # is admitted and simply routes through the tiled path.
        and int(getattr(text_config, "hidden_size", 0)) == 2560
        and int(getattr(text_config, "num_hidden_layers", 0)) == 48
        and int(getattr(text_config, "num_experts", 0)) == 512
        and int(getattr(text_config, "num_experts_per_tok", 0)) == 10
        and int(getattr(text_config, "moe_intermediate_size", 0)) == 640
        and int(getattr(text_config, "hc_count", 0)) == 4
        and int(getattr(text_config, "hc_lowrank", 0)) == 320
        and int(getattr(text_config, "num_attention_heads", 0)) == 24
        and int(getattr(text_config, "num_key_value_heads", 0)) == 2
        and int(getattr(text_config, "indexer_head_dim", 0)) == 128
        and int(getattr(text_config, "indexer_budget", 0)) == 2048
        and int(getattr(text_config, "indexer_compress_ratio", 0)) == 4
    )


def enable_qwen38_sm70_fp16_gemv(
    module: nn.Module, dtype: torch.dtype, vllm_config=None
) -> None:
    """Replace admitted unquantized methods before checkpoint loading."""
    if not envs.VLLM_SM70_QWEN38_FP16_GEMV:
        return
    capability_ok = current_platform.is_device_capability((7, 0))
    contract_ok = _exact_runtime_contract(vllm_config)
    if (
        envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8
        or dtype != torch.float16
        or not capability_ok
        or not contract_ok
    ):
        logger.warning_once(
            "Qwen3.8 checkpoint-FP16 GEMV opt-in rejected: "
            "online_qpn8=%s dtype=%s sm70=%s exact_contract=%s.",
            envs.VLLM_SM70_QWEN4_EXP_ONLINE_QPN8,
            dtype,
            capability_ok,
            contract_ok,
        )
        return

    replaced = 0
    for child in module.modules():
        if not (
            isinstance(child, LinearBase)
            and type(child.quant_method) is UnquantizedLinearMethod
        ):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.ndim != 2:
            continue
        shape = (int(weight.shape[0]), int(weight.shape[1]))
        if _plan_for(str(getattr(child, "prefix", "")), shape) is None:
            continue
        child.quant_method = Qwen38SM70FP16LinearMethod()
        replaced += 1

    fused_gdn_inputs = 0
    if envs.VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16:
        for child in module.modules():
            qkvz = getattr(child, "in_proj_qkvz", None)
            ba = getattr(child, "in_proj_ba", None)
            qkvz_weight = getattr(qkvz, "weight", None)
            ba_weight = getattr(ba, "weight", None)
            if not (
                isinstance(qkvz_weight, torch.Tensor)
                and qkvz_weight.shape == (4096, 2560)
                and isinstance(ba_weight, torch.Tensor)
                and ba_weight.shape == (24, 2560)
                and isinstance(
                    getattr(qkvz, "quant_method", None),
                    Qwen38SM70FP16LinearMethod,
                )
                and isinstance(
                    getattr(ba, "quant_method", None),
                    Qwen38SM70FP16LinearMethod,
                )
                and not bool(getattr(child, "gqa_interleaved_layout", True))
                and not bool(getattr(child, "disable_tp_for_ba_proj", True))
            ):
                continue
            child.sm70_qwen38_fp16_fused_input = True
            fused_gdn_inputs += 1

    if replaced:
        logger.info_once(
            "Prepared %d Qwen3.8 checkpoint-FP16 SM70 M=1 GEMV projections.",
            replaced,
        )
    else:
        logger.warning_once(
            "Qwen3.8 checkpoint-FP16 GEMV opt-in matched the runtime but no "
            "target projections were found."
        )
    if fused_gdn_inputs:
        logger.info_once(
            "Prepared %d Qwen3.8 fused checkpoint-FP16 GDN inputs.",
            fused_gdn_inputs,
        )
    elif envs.VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16:
        logger.warning_once(
            "Qwen3.8 fused checkpoint-FP16 GDN input opt-in found no targets."
        )


__all__ = [
    "Qwen38SM70FP16LinearMethod",
    "_qwen38_fp16_gdn_input_kernel",
    "enable_qwen38_sm70_fp16_gemv",
]
