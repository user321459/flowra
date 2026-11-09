"""
Low-rank building blocks shared by all adapters.

``LowRankFactors`` stores ``B in R^{m x r}`` and ``A in R^{r x n}`` with
``Delta W = scaling * B A``. Ranks can change during training:
:meth:`LowRankFactors.resize` keeps the existing rank-one components
``b_k a_k^T`` with the largest norm, and initialises new components with
``B = 0`` (Kaiming-uniform ``A``) so that the model function is unchanged
at the moment of the resize.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ResizeRecord:
    """Mapping between old and new parameters, used to carry optimizer state."""

    old_A: nn.Parameter
    old_B: nn.Parameter
    new_A: nn.Parameter
    new_B: nn.Parameter
    kept: List[int]  # indices of retained components (in the old factors)
    b_scale: float = 1.0
    old_rank: int = 0
    new_rank: int = 0


class LowRankFactors(nn.Module):
    def __init__(
        self,
        m: int,
        n: int,
        rank: int,
        scaling: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.m, self.n = int(m), int(n)
        self.scaling = float(scaling)
        A = torch.empty(rank, n, dtype=dtype, device=device)
        self._init_A(A)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(torch.zeros(m, rank, dtype=dtype, device=device))

    @staticmethod
    def _init_A(A: torch.Tensor) -> None:
        if A.numel():
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))

    @property
    def rank(self) -> int:
        return self.A.shape[0]

    @property
    def cost_per_rank(self) -> int:
        return self.m + self.n

    def num_parameters(self) -> int:
        return self.rank * (self.m + self.n)

    def delta(self) -> torch.Tensor:
        """``Delta W = scaling * B A`` of shape ``[m, n]``."""
        return self.scaling * (self.B @ self.A)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """``scaling * x A^T B^T`` for ``x[..., n]``."""
        return self.scaling * torch.nn.functional.linear(torch.nn.functional.linear(x, self.A), self.B)

    def component_scores(self) -> torch.Tensor:
        return self.B.detach().norm(dim=0) * self.A.detach().norm(dim=1)

    @torch.no_grad()
    def resize(self, new_rank: int, new_scaling: Optional[float] = None) -> Optional[ResizeRecord]:
        new_rank = max(0, min(int(new_rank), min(self.m, self.n)))
        new_scaling = self.scaling if new_scaling is None else float(new_scaling)
        if new_rank == self.rank and math.isclose(new_scaling, self.scaling):
            return None

        old_A, old_B = self.A, self.B
        scores = self.component_scores()
        order = torch.argsort(scores, descending=True, stable=True).tolist()
        kept = sorted(order[:new_rank])
        b_scale = self.scaling / new_scaling if new_scaling > 0 else 0.0

        A = torch.empty(new_rank, self.n, dtype=old_A.dtype, device=old_A.device)
        B = torch.zeros(self.m, new_rank, dtype=old_B.dtype, device=old_B.device)
        if kept:
            A[: len(kept)] = old_A[kept]
            B[:, : len(kept)] = old_B[:, kept] * b_scale
        if new_rank > len(kept):
            self._init_A(A[len(kept):])

        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.scaling = new_scaling
        return ResizeRecord(old_A, old_B, self.A, self.B, kept, b_scale, len(order), new_rank)

    def extra_repr(self) -> str:
        return f"m={self.m}, n={self.n}, rank={self.rank}, scaling={self.scaling:.4g}"


class BaseAdapter(nn.Module):
    """
    Wrapper around a frozen module.  Subclasses expose their low-rank factors
    through :meth:`factors` (``key -> LowRankFactors``; the key is ``""`` for
    single-matrix adapters and a gate name for recurrent adapters).
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.merged = False

    def factors(self) -> Dict[str, LowRankFactors]:
        return {}

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def merge(self) -> None:
        raise NotImplementedError

    def unmerge(self) -> None:
        raise NotImplementedError

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight
