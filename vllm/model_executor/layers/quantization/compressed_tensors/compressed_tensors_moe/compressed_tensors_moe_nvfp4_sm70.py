# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""compressed-tensors NVFP4 W4A16 experts on native SM70.

See patch_ctsm70moe.py for why this exists: the SM70 TurboMind NVFP4 MoE
kernels are reachable only from the ModelOpt config path, but they are a pure
W4A16 NVFP4 implementation and the compressed-tensors export carries the same
packed uint8 weights and FP8 E4M3 block scales. Only the parameter names and
the direction of the global scale differ, so this subclass creates the
parameters under the checkpoint's names, renames them, and lets the SM70
method do everything else.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.nvfp4_sm70_moe import (
    ModelOptNvFp4SM70MoEMethod,
)

from .compressed_tensors_moe_w4a4_nvfp4 import CompressedTensorsW4A4Nvfp4MoEMethod

logger = init_logger(__name__)


class CompressedTensorsNvFp4SM70MoEMethod(ModelOptNvFp4SM70MoEMethod):
    """NVFP4 W4A16 experts from a compressed-tensors checkpoint on SM70."""

    def __init__(self, moe: FusedMoEConfig) -> None:
        quant_config = ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4",
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
        super().__init__(quant_config, moe)
        # Needed by the borrowed create_weights below.
        self.group_size = 16
        self.use_global_sf = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # Register under the compressed-tensors names so the checkpoint loads.
        CompressedTensorsW4A4Nvfp4MoEMethod.create_weights(
            self,
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        for shard in ("w13", "w2"):
            packed = f"{shard}_weight_packed"
            if hasattr(layer, packed):
                setattr(
                    layer,
                    f"{shard}_weight",
                    torch.nn.Parameter(
                        getattr(layer, packed).data, requires_grad=False
                    ),
                )
                delattr(layer, packed)
            scale_2 = f"{shard}_weight_scale_2"
            gscale = f"{shard}_weight_global_scale"
            if not hasattr(layer, scale_2) and hasattr(layer, gscale):
                # compressed-tensors stores the quantization scale; the kernel
                # wants the dequantization multiplier.
                setattr(
                    layer,
                    scale_2,
                    torch.nn.Parameter(
                        (1.0 / getattr(layer, gscale).data.float()).contiguous(),
                        requires_grad=False,
                    ),
                )
        # W4A16 has no activation scales, but the SM70 method frees the full
        # ModelOpt parameter set by name; give it the two it expects.
        for shard in ("w13", "w2"):
            dst = f"{shard}_input_scale"
            if hasattr(layer, dst):
                continue
            src = f"{shard}_input_global_scale"
            data = (
                getattr(layer, src).data
                if hasattr(layer, src)
                else torch.empty(0, dtype=torch.float32)
            )
            setattr(
                layer, dst, torch.nn.Parameter(data, requires_grad=False)
            )
        super().process_weights_after_loading(layer)
        # Drop the compressed-tensors originals the SM70 repack no longer needs.
        for leftover in (
            "w13_weight_global_scale",
            "w2_weight_global_scale",
            "w13_input_global_scale",
            "w2_input_global_scale",
        ):
            if hasattr(layer, leftover):
                delattr(layer, leftover)
