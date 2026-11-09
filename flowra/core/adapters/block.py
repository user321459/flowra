"""
Block-level adapters:

    W_eff^(i,j) = W^(i,j) + B^(i,j) A^(i,j),  B^(i,j) in R^{m/p x r}, A^(i,j) in R^{r x n/q}

Only blocks contained in the blueprint receive factors.  The update is
assembled into a dense ``Delta W`` before the matrix product, which keeps
the implementation simple; it is intended for the optional fine-grained
formulation, not for the default module-level experiments.
"""

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowra.algorithms.allocation.allocation import BlockAssignment
from flowra.core.adapters.base import BaseAdapter, LowRankFactors, ResizeRecord


class BlockLoRA(BaseAdapter):
    def __init__(self, base: nn.Module, blocks: Sequence[BlockAssignment], scaling: float = 1.0):
        if not isinstance(base, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            raise TypeError("BlockLoRA supports Linear / Conv1d / Conv2d")
        super().__init__(base)
        w = base.weight
        self.m, self.n = w.shape[0], w[0].numel()
        self.block_index = {}
        self.blocks = nn.ModuleDict()
        for b in blocks:
            key = f"{b.i}_{b.j}"
            self.blocks[key] = LowRankFactors(
                b.rows[1] - b.rows[0], b.cols[1] - b.cols[0], b.rank, scaling, w.dtype, w.device
            )
            self.block_index[key] = (b.rows, b.cols)

    def factors(self) -> Dict[str, LowRankFactors]:
        return dict(self.blocks.items())

    def set_blocks(self, blocks: Sequence[BlockAssignment], scaling: float) -> List[ResizeRecord]:
        """Apply a refined blueprint: resize retained blocks, add new ones, empty dropped ones."""
        w = self.base.weight
        records: List[ResizeRecord] = []
        wanted = set()
        for b in blocks:
            key = f"{b.i}_{b.j}"
            wanted.add(key)
            if key in self.blocks:
                rec = self.blocks[key].resize(b.rank)
                if rec:
                    records.append(rec)
            else:
                self.blocks[key] = LowRankFactors(
                    b.rows[1] - b.rows[0], b.cols[1] - b.cols[0], b.rank, scaling, w.dtype, w.device
                )
                self.block_index[key] = (b.rows, b.cols)
        for key, f in self.blocks.items():
            if key not in wanted and f.rank:
                records.append(f.resize(0))
        return records

    def delta(self) -> torch.Tensor:
        w = self.base.weight
        delta = torch.zeros(self.m, self.n, dtype=w.dtype, device=w.device)
        for key, f in self.blocks.items():
            (r0, r1), (c0, c1) = self.block_index[key]
            if f.rank:
                delta[r0:r1, c0:c1] = delta[r0:r1, c0:c1] + f.delta()
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base(x)
        b = self.base
        w = b.weight + self.delta().reshape_as(b.weight)
        if isinstance(b, nn.Linear):
            return F.linear(x, w, b.bias)
        conv = F.conv2d if isinstance(b, nn.Conv2d) else F.conv1d
        return conv(x, w, b.bias, b.stride, b.padding, b.dilation, b.groups)

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight.add_(self.delta().reshape_as(self.base.weight))
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight.sub_(self.delta().reshape_as(self.base.weight))
            self.merged = False
