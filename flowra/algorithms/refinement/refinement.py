"""
Dynamic rank refinement during training.

The calibration descriptor ``D~_l`` stays fixed. At every step ``t in T_ref``
the gradient factor is updated,

    C_l(t) = D~_l * ||grad_{h_l} L_t||_2 * ||W_l||_F,

and the ranks are re-allocated under the same total budget ``B_total``.
Applying the new ranks to the adapters (retaining existing components,
initialising new ones) is done by
:meth:`flowra.core.adapters.base.LowRankFactors.resize`.

Default schedule: warm-up of 10% of the training steps ``T`` followed by a
refinement every 100 steps.
"""

import math
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from flowra.algorithms.allocation.allocation import AllocationResult, Shape, allocate_ranks

Adjuster = Callable[[AllocationResult, Mapping[str, float], Mapping[str, Shape], Mapping[str, float]], AllocationResult]


def refinement_schedule(total_steps: int, warmup_ratio: float = 0.1, interval: int = 100) -> List[int]:
    """``T_ref = {w, w + interval, w + 2 interval, ...} ∩ [0, T)`` with ``w = ceil(warmup_ratio * T)``."""
    if total_steps <= 0 or interval <= 0:
        return []
    start = max(1, math.ceil(warmup_ratio * total_steps))
    return list(range(start, total_steps, interval))


class RankRefiner:
    """
    Budget-preserving re-allocation from updated criticality scores.

    Args:
        descriptor_abs: fixed calibration descriptors ``D~_l``.
        weight_norms: ``||W_l||_F``.
        shapes: ``(m_l, n_l)``.
        budget: ``B_total``.
        gamma: allocation exponent.
        use_flow: if ``False`` the criticality omits ``D~_l`` (ablation).
        initial_grad_norms: calibration gradient norms, used for layers whose
            gradient is unavailable at step ``t``.
    """

    def __init__(
        self,
        descriptor_abs: Mapping[str, float],
        weight_norms: Mapping[str, float],
        shapes: Mapping[str, Tuple[int, int]],
        budget: int,
        gamma: float = 1.0,
        use_flow: bool = True,
        initial_grad_norms: Optional[Mapping[str, float]] = None,
        max_rank: Optional[int] = None,
        adjuster: Optional[Adjuster] = None,
    ):
        self.descriptor_abs = dict(descriptor_abs)
        self.weight_norms = dict(weight_norms)
        self.shapes = dict(shapes)
        self.budget = int(budget)
        self.gamma = gamma
        self.use_flow = use_flow
        self.max_rank = max_rank
        self.adjuster = adjuster
        self.last_grad_norms: Dict[str, float] = dict(initial_grad_norms or {})
        self.history: List[Dict] = []

    def criticality(self, grad_norms: Mapping[str, float]) -> Dict[str, float]:
        for name, g in grad_norms.items():
            if g is not None and math.isfinite(g):
                self.last_grad_norms[name] = float(g)
        crit = {}
        for name in self.shapes:
            g = self.last_grad_norms.get(name, 0.0)
            d = self.descriptor_abs.get(name, 0.0) if self.use_flow else 1.0
            crit[name] = d * g * self.weight_norms.get(name, 0.0)
        return crit

    def refine(self, step: int, grad_norms: Mapping[str, float]) -> AllocationResult:
        crit = self.criticality(grad_norms)
        result = allocate_ranks(crit, self.shapes, self.budget, gamma=self.gamma, max_rank=self.max_rank)
        if self.adjuster is not None:
            result = self.adjuster(result, crit, self.shapes, self.last_grad_norms)
        self.history.append({"step": step, "ranks": dict(result.ranks), "criticality": crit,
                             "budget_used": result.total_used})
        return result
