"""
Polymorphic adapter assignment.

The adapter parameterisation follows the module type of each target:

* linear projections -> low-rank factors (LoRA, or DoRA if requested);
* convolutions       -> kernel-factorised adapter (``k x k`` A, ``1 x 1`` B);
* GRU / LSTM gates   -> gate-wise factors;
* normalisation layers inside adapted blocks -> affine (scale / shift) adapters.

With ``polymorphic=False`` only standard LoRA factors on linear /
convolutional targets are used and normalisation layers are not adapted.
"""

from typing import Dict, Mapping

import torch.nn as nn

from flowra.core.adapters.base import BaseAdapter
from flowra.core.adapters.lora import DoRALinear, LoRAConv, LoRALinear
from flowra.core.adapters.normalization import NormAffineAdapter
from flowra.core.adapters.rnn import GatedRNNAdapter
from flowra.core.analysis.handlers import TargetInfo

ADAPTER_VARIANTS = ("lora", "dora")


class PolymorphicAdapterFactory:
    def __init__(self, variant: str = "lora", dropout: float = 0.0, polymorphic: bool = True):
        if variant not in ADAPTER_VARIANTS:
            raise ValueError(f"Unknown adapter variant '{variant}', expected one of {ADAPTER_VARIANTS}")
        self.variant = variant
        self.dropout = dropout
        self.polymorphic = polymorphic

    def create(self, target: TargetInfo, rank: int, scaling: float) -> BaseAdapter:
        if target.kind == "linear":
            cls = DoRALinear if self.variant == "dora" else LoRALinear
            return cls(target.module, rank, scaling, self.dropout)
        if target.kind == "conv":
            return LoRAConv(target.module, rank, scaling, self.dropout)
        raise ValueError(f"Use create_rnn for target kind '{target.kind}'")

    def create_rnn(self, rnn: nn.Module, gate_ranks: Mapping[str, int], scaling: Mapping[str, float]) -> BaseAdapter:
        return GatedRNNAdapter(rnn, gate_ranks, scaling)

    def create_norm(self, norm: nn.Module) -> BaseAdapter:
        return NormAffineAdapter(norm, with_bias=getattr(norm, "bias", None) is not None)
