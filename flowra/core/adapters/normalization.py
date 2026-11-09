"""
Affine adapters for normalisation layers (LayerNorm, RMSNorm, BatchNorm).

``y' = norm(x) * (1 + delta_gamma) + delta_beta`` with zero-initialised
``delta_gamma`` / ``delta_beta``.  Used by the polymorphic adapter
assignment for normalisation layers inside adapted blocks.
"""

from typing import Dict

import torch
import torch.nn as nn

from flowra.core.adapters.base import BaseAdapter


def is_norm_layer(module: nn.Module) -> bool:
    if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
        return True
    return "norm" in type(module).__name__.lower() and isinstance(getattr(module, "weight", None), torch.Tensor)


def norm_features(module: nn.Module) -> int:
    return module.weight.shape[0]


class NormAffineAdapter(BaseAdapter):
    def __init__(self, base: nn.Module, with_bias: bool = True):
        super().__init__(base)
        d = norm_features(base)
        w = base.weight
        self.channel_dim = 1 if isinstance(base, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)) else -1
        self.delta_gamma = nn.Parameter(torch.zeros(d, dtype=w.dtype, device=w.device))
        self.delta_beta = nn.Parameter(torch.zeros(d, dtype=w.dtype, device=w.device)) if with_bias else None

    def _shape(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.channel_dim == -1:
            return v
        return v.reshape(1, -1, *([1] * (x.dim() - 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        out = out * (1 + self._shape(out, self.delta_gamma))
        if self.delta_beta is not None:
            out = out + self._shape(out, self.delta_beta)
        return out

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        bias = getattr(self.base, "bias", None)
        if self.delta_beta is not None and bias is None:
            raise RuntimeError("Cannot merge a shift into a normalisation layer without bias")
        self.base.weight.mul_(1 + self.delta_gamma)
        if bias is not None:
            bias.mul_(1 + self.delta_gamma)
            if self.delta_beta is not None:
                bias.add_(self.delta_beta)
        self.merged = True
