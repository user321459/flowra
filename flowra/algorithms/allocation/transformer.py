"""
Transformer-specific rank redistribution on top of the module-level allocator.

Inside each transformer layer the attention / FFN split is
``r_attn / r_ffn = (D_attn ||grad Attn L|| / (D_ffn ||grad FFN L||))^gamma``.
Head ranks are ``r_h ∝ C_h^gamma`` with ``C_h = D_h · E||grad Attn L||``.
A fraction ``alpha = 0.7`` of each head rank goes to the value and output
projections and the remainder to query and key.

The per-layer trainable-parameter budget is preserved; leftover capacity is
filled greedily. The same redistribution is applied when ranks are refined.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from flowra.algorithms.allocation.allocation import AllocationResult, Shape

Role = str  # "q" | "k" | "v" | "o"
LayerGroups = Dict[int, Dict[str, List[str]]]


def attn_role(local_name: str) -> Optional[Role]:
    """Map an attention projection name to {q, k, v, o}."""
    n = local_name.lower()
    leaf = n.split(".")[-1]
    if leaf in ("q", "q_proj", "query", "to_q", "wq") or "query" in n or n.endswith("q_proj") or n.endswith("to_q"):
        return "q"
    if leaf in ("k", "k_proj", "key", "to_k", "wk") or "key" in n or n.endswith("k_proj") or n.endswith("to_k"):
        return "k"
    if leaf in ("v", "v_proj", "value", "to_v", "wv") or "value" in n or n.endswith("v_proj") or n.endswith("to_v"):
        return "v"
    if leaf in ("o", "o_proj", "out_proj", "to_out", "wo", "c_proj") or n.endswith("o_proj") or n.endswith("out_proj"):
        return "o"
    if "attention" in n and leaf == "dense":
        return "o"
    return None


def transformer_groups(targets: Mapping[str, object]) -> LayerGroups:
    """Group adaptation targets of a transformer by layer and sub-layer."""
    groups: LayerGroups = {}
    for name, info in targets.items():
        meta = getattr(info, "meta", {})
        layer = meta.get("layer")
        if layer is None:
            continue
        g = groups.setdefault(layer, {"attn": [], "ffn": [], "q": [], "k": [], "v": [], "o": []})
        if meta.get("sublayer") == "attn":
            g["attn"].append(name)
            role = attn_role(meta.get("local_name", name))
            if role:
                g[role].append(name)
        else:
            g["ffn"].append(name)
    return groups


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cost(shapes: Mapping[str, Shape], name: str) -> int:
    m, n = shapes[name]
    return int(m + n)


def _split_rank(names: Sequence[str], total: int, criticality: Mapping[str, float]) -> Dict[str, int]:
    from flowra.core.analysis.handlers import proportional_integer_split

    if not names:
        return {}
    if len(names) == 1:
        return {names[0]: max(int(total), 0)}
    parts = proportional_integer_split([max(criticality.get(n, 0.0), 0.0) for n in names], max(int(total), 0))
    return dict(zip(names, parts))


def _attention_ranks(
    group: Mapping[str, List[str]],
    total_rank: int,
    criticality: Mapping[str, float],
    head_criticality: Optional[Sequence[float]],
    alpha: float,
) -> Dict[str, int]:
    """Distribute ``total_rank`` across Q/K/V/O using head scores and ``alpha``."""
    from flowra.core.analysis.handlers import allocate_head_ranks, split_attention_rank

    attn = group["attn"]
    if total_rank <= 0 or not attn:
        return {n: 0 for n in attn}

    q, k, v, o = group["q"], group["k"], group["v"], group["o"]
    if q and k and v and o:
        if head_criticality:
            totals = {"q": 0, "k": 0, "v": 0, "o": 0}
            for r_h in allocate_head_ranks(head_criticality, total_rank):
                for role, r in split_attention_rank(r_h, alpha).items():
                    totals[role] += r
        else:
            totals = split_attention_rank(total_rank, alpha)
        ranks: Dict[str, int] = {}
        for role, names in (("q", q), ("k", k), ("v", v), ("o", o)):
            ranks.update(_split_rank(names, totals[role], criticality))
        for name in attn:
            ranks.setdefault(name, 0)
        return ranks
    return _split_rank(attn, total_rank, criticality)


def _fit_layer_budget(
    ranks: Dict[str, int],
    names: Sequence[str],
    shapes: Mapping[str, Shape],
    criticality: Mapping[str, float],
    budget: int,
) -> None:
    """Increment / decrement ranks inside ``names`` so that the cost fits ``budget``."""
    cost = {n: _cost(shapes, n) for n in names}
    cap = {n: int(min(shapes[n])) for n in names}
    used = sum(ranks[n] * cost[n] for n in names)
    order = {n: i for i, n in enumerate(names)}

    while used > budget:
        candidates = [n for n in names if ranks[n] > 0]
        if not candidates:
            break
        n = min(candidates, key=lambda x: (criticality.get(x, 0.0) / max(cost[x], 1), -order[x]))
        ranks[n] -= 1
        used -= cost[n]

    remaining = budget - used
    heap = [(-criticality.get(n, 0.0) / (ranks[n] * cost[n] + cost[n]), order[n], n)
            for n in names if ranks[n] < cap[n] and cost[n] <= remaining]
    heapq.heapify(heap)
    while heap:
        _, _, n = heapq.heappop(heap)
        if ranks[n] >= cap[n] or cost[n] > remaining:
            continue
        ranks[n] += 1
        remaining -= cost[n]
        if ranks[n] < cap[n] and cost[n] <= remaining:
            heapq.heappush(heap, (-criticality.get(n, 0.0) / (ranks[n] * cost[n] + cost[n]), order[n], n))


def apply_transformer_formulas(
    allocation: AllocationResult,
    criticality: Mapping[str, float],
    shapes: Mapping[str, Shape],
    groups: LayerGroups,
    d_attn_ffn: Mapping[int, Tuple[float, float]],
    grad_norms: Mapping[str, float],
    head_descriptors: Optional[Mapping[int, Sequence[float]]] = None,
    gamma: float = 1.0,
    alpha: float = 0.7,
) -> AllocationResult:
    """
    Redistribute the ranks of ``allocation`` inside each transformer layer
    without exceeding the original per-layer parameter budget or the global
    ``B_total``.
    """
    ranks = dict(allocation.ranks)
    head_descriptors = head_descriptors or {}
    ratios: Dict[int, float] = {}

    for layer, group in groups.items():
        attn, ffn = group["attn"], group["ffn"]
        members = [n for n in attn + ffn if n in ranks]
        if not members:
            continue
        layer_budget = sum(ranks[n] * _cost(shapes, n) for n in members)
        total_rank = sum(ranks[n] for n in members)

        r_attn = sum(ranks[n] for n in attn if n in ranks)
        r_ffn = sum(ranks[n] for n in ffn if n in ranks)
        if attn and ffn and total_rank > 0:
            from flowra.core.analysis.handlers import attention_ffn_rank_ratio

            d_attn, d_ffn = d_attn_ffn.get(layer, (1.0, 1.0))
            g_attn = _mean([grad_norms.get(n, 0.0) for n in attn])
            g_ffn = _mean([grad_norms.get(n, 0.0) for n in ffn])
            rho = attention_ffn_rank_ratio(d_attn, g_attn, d_ffn, g_ffn, gamma)
            ratios[layer] = rho
            if rho == float("inf"):
                r_attn, r_ffn = total_rank, 0
            else:
                r_attn = int(round(total_rank * rho / (rho + 1.0)))
                r_attn = min(max(r_attn, 0), total_rank)
                r_ffn = total_rank - r_attn

        g_attn = _mean([grad_norms.get(n, 0.0) for n in attn])
        head_c = None
        if layer in head_descriptors:
            head_c = [max(float(d), 0.0) * g_attn for d in head_descriptors[layer]]

        new_ranks = dict(ranks)
        new_ranks.update(_attention_ranks(group, r_attn, criticality, head_c, alpha))
        new_ranks.update(_split_rank(ffn, r_ffn, criticality))
        for n in members:
            new_ranks[n] = max(0, min(int(new_ranks[n]), int(min(shapes[n]))))
        _fit_layer_budget(new_ranks, members, shapes, criticality, layer_budget)
        ranks.update({n: new_ranks[n] for n in members})

    used = {n: ranks[n] * _cost(shapes, n) for n in ranks}
    result = AllocationResult(
        ranks=ranks,
        budget_total=allocation.budget_total,
        budget_used=used,
        proportional_ranks=dict(allocation.proportional_ranks),
        greedy_increments=allocation.greedy_increments,
    )
    result.transformer_ratios = ratios  # type: ignore[attr-defined]
    return result


class TransformerRankAdjuster:
    """Callable used by :class:`FLOWRA` and :class:`RankRefiner`."""

    def __init__(
        self,
        groups: LayerGroups,
        d_attn_ffn: Mapping[int, Tuple[float, float]],
        head_descriptors: Optional[Mapping[int, Sequence[float]]] = None,
        gamma: float = 1.0,
        alpha: float = 0.7,
    ):
        self.groups = groups
        self.d_attn_ffn = dict(d_attn_ffn)
        self.head_descriptors = dict(head_descriptors or {})
        self.gamma = gamma
        self.alpha = alpha
        self.last_ratios: Dict[int, float] = {}

    def __call__(
        self,
        allocation: AllocationResult,
        criticality: Mapping[str, float],
        shapes: Mapping[str, Shape],
        grad_norms: Mapping[str, float],
    ) -> AllocationResult:
        result = apply_transformer_formulas(
            allocation, criticality, shapes, self.groups, self.d_attn_ffn,
            grad_norms, self.head_descriptors, self.gamma, self.alpha,
        )
        self.last_ratios = getattr(result, "transformer_ratios", {})
        return result


def signals_from_profile(
    groups: LayerGroups,
    point_descriptors: Mapping[str, float],
    modules: Mapping[str, object],
) -> Tuple[Dict[int, Tuple[float, float]], Dict[str, float]]:
    """``D_attn``, ``D_ffn`` per layer and per-module gradient norms from a calibration profile."""
    d_attn_ffn: Dict[int, Tuple[float, float]] = {}
    grad_norms: Dict[str, float] = {}
    for name, prof in modules.items():
        grad_norms[name] = float(getattr(prof, "grad_norm", 0.0))
    for layer, group in groups.items():
        d_attn = abs(float(point_descriptors.get(f"mid{layer}", 0.0)))
        if d_attn == 0.0 and group["attn"]:
            d_attn = _mean([abs(float(getattr(modules[n], "descriptor", 0.0))) for n in group["attn"] if n in modules])
        d_ffn = abs(float(point_descriptors.get(f"out{layer}", 0.0)))
        if d_ffn == 0.0 and group["ffn"]:
            d_ffn = _mean([abs(float(getattr(modules[n], "descriptor", 0.0))) for n in group["ffn"] if n in modules])
        d_attn_ffn[layer] = (d_attn, d_ffn)
    return d_attn_ffn, grad_norms
