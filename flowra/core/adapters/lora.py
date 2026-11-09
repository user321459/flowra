"""
Module-level low-rank adapters (default configuration of the experiments).

* :class:`LoRALinear` - ``W_eff = W + scaling * B A`` for ``nn.Linear``;
* :class:`DoRALinear` - the same ranks realised with DoRA
  (magnitude / direction decomposition);
* :class:`LoRAConv` - convolutional adapter with a ``k x k`` down-projection
  ``A`` and a ``1 x 1`` up-projection ``B``; it is equivalent to the update
  ``Delta K = reshape(B A)`` of the flattened kernel
  ``C_out x (C_in k_h k_w)`` and costs ``r (C_out + C_in k_h k_w)``
  parameters.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowra.core.adapters.base import BaseAdapter, LowRankFactors


class LoRALinear(BaseAdapter):
    def __init__(self, base: nn.Linear, rank: int, scaling: float = 1.0, dropout: float = 0.0):
        super().__init__(base)
        w = base.weight
        self.lora = LowRankFactors(base.out_features, base.in_features, rank, scaling, w.dtype, w.device)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def factors(self) -> Dict[str, LowRankFactors]:
        return {"": self.lora}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged or self.lora.rank == 0:
            return out
        return out + self.lora.apply(self.dropout(x))

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight.add_(self.lora.delta().to(self.base.weight.dtype))
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight.sub_(self.lora.delta().to(self.base.weight.dtype))
            self.merged = False


class DoRALinear(LoRALinear):
    """
    ``W_eff = m * (W + scaling B A) / ||W + scaling B A||_c`` with a trainable
    per-output magnitude ``m`` initialised to the column norms of ``W``.
    """

    def __init__(self, base: nn.Linear, rank: int, scaling: float = 1.0, dropout: float = 0.0):
        super().__init__(base, rank, scaling, dropout)
        with torch.no_grad():
            self.magnitude = nn.Parameter(base.weight.norm(dim=1).clone())

    def _weight(self) -> torch.Tensor:
        w = self.base.weight + self.lora.delta()
        return self.magnitude.unsqueeze(1) * w / (w.norm(dim=1, keepdim=True) + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base(x)
        return F.linear(self.dropout(x), self._weight(), self.base.bias)

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self._original = self.base.weight.detach().clone()
            self.base.weight.copy_(self._weight())
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight.copy_(self._original)
            del self._original
            self.merged = False


class LoRAConv(BaseAdapter):
    def __init__(self, base: nn.Module, rank: int, scaling: float = 1.0, dropout: float = 0.0):
        if not isinstance(base, (nn.Conv1d, nn.Conv2d)) or base.groups != 1:
            raise TypeError("LoRAConv supports Conv1d / Conv2d with groups=1")
        super().__init__(base)
        w = base.weight
        self.kernel_shape = tuple(w.shape[1:])  # (C_in, k...) for A
        self.lora = LowRankFactors(w.shape[0], w[0].numel(), rank, scaling, w.dtype, w.device)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._conv = F.conv2d if isinstance(base, nn.Conv2d) else F.conv1d

    def factors(self) -> Dict[str, LowRankFactors]:
        return {"": self.lora}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged or self.lora.rank == 0:
            return out
        b = self.base
        r = self.lora.rank
        a_kernel = self.lora.A.reshape(r, *self.kernel_shape)
        b_kernel = self.lora.B.reshape(b.out_channels, r, *([1] * (len(self.kernel_shape) - 1)))
        if b.padding_mode != "zeros":
            raise NotImplementedError("Only zero padding is supported")
        h = self._conv(self.dropout(x), a_kernel, None, b.stride, b.padding, b.dilation)
        return out + self.lora.scaling * self._conv(h, b_kernel)

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight.add_(self.lora.delta().reshape_as(self.base.weight))
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight.sub_(self.lora.delta().reshape_as(self.base.weight))
            self.merged = False
