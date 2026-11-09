"""
Rank allocation under a trainable-parameter budget.

* :func:`allocate_ranks` - the budget is first distributed proportionally to
  ``C_l^gamma`` and the remainder is assigned greedily to the layer with the
  highest marginal utility ``C_l / (B_used_l + m_l + n_l)``. Every increment
  respects the remaining budget, saturated layers (``r_l = min(m_l, n_l)``)
  are excluded, and the loop terminates when no feasible increment remains.
* :func:`generate_block_blueprint` - optional block ranks inside each layer.

The procedure is a greedy heuristic; no approximation guarantee is claimed.
"""

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

Shape = Tuple[int, int]


@dataclass
class AllocationResult:
    ranks: Dict[str, int]
    budget_total: int
    budget_used: Dict[str, int]
    proportional_ranks: Dict[str, int] = field(default_factory=dict)
    greedy_increments: int = 0

    @property
    def total_used(self) -> int:
        return sum(self.budget_used.values())

    @property
    def budget_remaining(self) -> int:
        return self.budget_total - self.total_used


def allocate_ranks(
    criticality: Mapping[str, float],
    shapes: Mapping[str, Shape],
    budget: int,
    gamma: float = 1.0,
    max_rank: Optional[Union[int, Mapping[str, int]]] = None,
) -> AllocationResult:
    """
    Proportional ranks followed by greedy increments that fit the remaining budget.

    Args:
        criticality: ``C_l`` per layer (non-negative).
        shapes: ``(m_l, n_l)`` per layer; the cost of one rank is ``m_l + n_l``.
        budget: ``B_total`` (number of trainable adapter parameters).
        gamma: allocation exponent (``1`` in the default configuration).
        max_rank: optional additional rank cap.
    """
    names = list(criticality)
    cost = {n: int(shapes[n][0] + shapes[n][1]) for n in names}
    cap = {n: int(min(shapes[n])) for n in names}
    if max_rank is not None:
        for n in names:
            limit = max_rank if isinstance(max_rank, int) else max_rank.get(n, cap[n])
            cap[n] = min(cap[n], int(limit))

    crit = {n: max(float(criticality[n]), 0.0) for n in names}
    powered = {n: crit[n] ** gamma if crit[n] > 0 else 0.0 for n in names}
    c_total = sum(powered.values())

    # Proportional initial ranks.
    ranks: Dict[str, int] = {}
    used: Dict[str, int] = {}
    for n in names:
        r = math.floor(budget * powered[n] / (c_total * cost[n])) if c_total > 0 else 0
        r = max(0, min(r, cap[n]))
        ranks[n] = r
        used[n] = r * cost[n]
    proportional = dict(ranks)

    remaining = budget - sum(used.values())

    # Greedy increments by marginal utility C_l / (B_used_l + m_l + n_l).
    # The utility of a layer only changes when that layer is incremented, so a
    # lazy max-heap is exact.  Ties are broken by layer order.
    order = {n: i for i, n in enumerate(names)}
    heap = [(-crit[n] / (used[n] + cost[n]), order[n], n) for n in names if ranks[n] < cap[n]]
    heapq.heapify(heap)
    increments = 0
    while heap:
        _, _, n = heapq.heappop(heap)
        if ranks[n] >= cap[n] or cost[n] > remaining:
            continue  # saturated or infeasible; the remaining budget never grows again
        ranks[n] += 1
        remaining -= cost[n]
        used[n] += cost[n]
        increments += 1
        if ranks[n] < cap[n] and cost[n] <= remaining:
            heapq.heappush(heap, (-crit[n] / (used[n] + cost[n]), order[n], n))

    return AllocationResult(
        ranks=ranks, budget_total=int(budget), budget_used=used,
        proportional_ranks=proportional, greedy_increments=increments,
    )


def uniform_ranks(shapes: Mapping[str, Shape], rank: int) -> Dict[str, int]:
    """Static uniform rank (reference configuration ``r = 16``)."""
    return {n: min(rank, min(s)) for n, s in shapes.items()}


def uniform_rank_for_budget(shapes: Mapping[str, Shape], budget: int) -> int:
    """Largest uniform rank whose cost fits into ``budget`` (matched-budget LoRA)."""
    per_rank = sum(m + n for m, n in shapes.values())
    return max(budget // max(per_rank, 1), 0)


# ---------------------------------------------------------------------------
# Block-level blueprint
# ---------------------------------------------------------------------------

BlockLayout = Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]


@dataclass
class BlockAssignment:
    layer: str
    i: int
    j: int
    rank: int
    rows: Tuple[int, int]
    cols: Tuple[int, int]

    @property
    def cost(self) -> int:
        return self.rank * ((self.rows[1] - self.rows[0]) + (self.cols[1] - self.cols[0]))


def generate_block_blueprint(
    layer_ranks: Mapping[str, int],
    shapes: Mapping[str, Shape],
    sensitivities: Mapping[str, Union[torch.Tensor, Sequence[Sequence[float]]]],
    layouts: Mapping[str, BlockLayout],
    beta: float = 1.0,
    tau: int = 1,
) -> List[BlockAssignment]:
    """
    For every layer with a non-zero rank, blocks are visited in decreasing
    order of ``S_l^(i,j)`` and receive rank proportional to
    ``(S_l^(i,j))^beta``, limited by the block dimensions and the free layer
    budget. Blocks with rank below ``tau`` are pruned; unused budget is left
    unallocated.
    """
    blueprint: List[BlockAssignment] = []
    for name, r_layer in layer_ranks.items():
        if r_layer <= 0:
            continue
        m, n = shapes[name]
        b_free = r_layer * (m + n)
        rows, cols = layouts[name]
        s = torch.as_tensor(sensitivities[name], dtype=torch.float64).clamp_min(0.0)
        s_pow = s.pow(beta)
        s_total = float(s_pow.sum())
        if s_total <= 0:
            s_pow = torch.ones_like(s_pow)
            s_total = float(s_pow.sum())
        flat = s_pow.flatten()
        order = sorted(range(flat.numel()), key=lambda k: (-float(flat[k]), k))
        q = len(cols)
        for k in order:
            i, j = divmod(k, q)
            r = max(1, math.floor(r_layer * float(flat[k]) / s_total))
            rows_ij, cols_ij = rows[i], cols[j]
            bm, bn = rows_ij[1] - rows_ij[0], cols_ij[1] - cols_ij[0]
            c = bm + bn
            r = min(r, min(bm, bn), math.floor(b_free / c))
            if r >= tau:
                blueprint.append(BlockAssignment(name, i, j, r, rows_ij, cols_ij))
                b_free -= r * c
    return blueprint


def blueprint_by_layer(blueprint: Sequence[BlockAssignment]) -> Dict[str, List[BlockAssignment]]:
    out: Dict[str, List[BlockAssignment]] = {}
    for b in blueprint:
        out.setdefault(b.layer, []).append(b)
    return out
