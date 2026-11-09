"""
Multi-task adapter composition.

Each task contributes per-layer updates ``Delta W_t``.  The FLowRA composer
measures the subspace interference between tasks,

    I(i, j) = ||U_i^T U_j||_F^2 / min(k_i, k_j),

where ``U`` spans the dominant left singular subspace of ``Delta W``
(capturing ``energy`` of its spectrum).  When ``I(i, j) > eta`` (composition
threshold, default 0.3) the later update is projected onto the orthogonal
complement of the earlier ones before summation.  Baselines: naive
averaging, task arithmetic and TIES-Merging.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from flowra.core.adapters.base import BaseAdapter
from flowra.core.adapters.block import BlockLoRA
from flowra.core.adapters.lora import DoRALinear
from flowra.core.adapters.rnn import GatedRNNAdapter

Deltas = Dict[str, torch.Tensor]


@torch.no_grad()
def adapter_deltas(adapters: Dict[str, BaseAdapter]) -> Deltas:
    """Dense weight updates ``Delta W`` of all low-rank adapters (keyed by module name)."""
    out: Deltas = {}
    for name, adapter in adapters.items():
        if isinstance(adapter, BlockLoRA):
            out[name] = adapter.delta().reshape_as(adapter.base.weight).clone()
        elif isinstance(adapter, DoRALinear):
            out[name] = (adapter._weight() - adapter.base.weight).clone()
        elif isinstance(adapter, GatedRNNAdapter):
            for key, w in adapter._weights().items():
                out[f"{name}.{key}"] = (w - getattr(adapter.base, key)).clone()
        elif "" in adapter.factors():
            f = adapter.factors()[""]
            out[name] = f.delta().reshape_as(adapter.base.weight).clone() if f.rank else torch.zeros_like(adapter.base.weight)
    return out


@torch.no_grad()
def apply_deltas(model: nn.Module, deltas: Deltas) -> nn.Module:
    """Add merged updates to the weights of a (non-adapted) model."""
    for name, delta in deltas.items():
        module_name, _, param = name.rpartition(".")
        if param.startswith(("weight_ih", "weight_hh")):
            getattr(model.get_submodule(module_name), param).add_(delta)
        else:
            model.get_submodule(name).weight.add_(delta.reshape_as(model.get_submodule(name).weight))
    return model


def _principal_subspace(delta: torch.Tensor, energy: float) -> torch.Tensor:
    mat = delta.reshape(delta.shape[0], -1).float()
    if not torch.any(mat):
        return mat.new_zeros(mat.shape[0], 0)
    u, s, _ = torch.linalg.svd(mat, full_matrices=False)
    cum = (s ** 2).cumsum(0) / (s ** 2).sum()
    k = int((cum < energy).sum().item()) + 1
    return u[:, :k]


def subspace_interference(d1: torch.Tensor, d2: torch.Tensor, energy: float = 0.95) -> float:
    u1, u2 = _principal_subspace(d1, energy), _principal_subspace(d2, energy)
    k = min(u1.shape[1], u2.shape[1])
    if k == 0:
        return 0.0
    return float((u1.t() @ u2).pow(2).sum() / k)


class OrthogonalComposer:
    def __init__(self, eta: float = 0.3, energy: float = 0.95):
        self.eta = eta
        self.energy = energy
        self.report: Dict[str, List[Tuple[int, int, float]]] = {}

    @torch.no_grad()
    def compose(self, task_deltas: Sequence[Deltas], weights: Optional[Sequence[float]] = None) -> Deltas:
        weights = list(weights) if weights is not None else [1.0] * len(task_deltas)
        layers = sorted(set().union(*[d.keys() for d in task_deltas]))
        merged: Deltas = {}
        self.report = {}
        for layer in layers:
            items = [(t, d[layer]) for t, d in enumerate(task_deltas) if layer in d]
            accepted: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
            total = None
            for t, delta in items:
                original = delta.reshape(delta.shape[0], -1).float()
                mat = original
                for s, mat_s, u_s in accepted:
                    inter = subspace_interference(mat_s, original, self.energy)
                    self.report.setdefault(layer, []).append((s, t, inter))
                    if inter > self.eta and u_s.shape[1]:
                        mat = mat - u_s @ (u_s.t() @ mat)
                accepted.append((t, original, _principal_subspace(original, self.energy)))
                contrib = weights[t] * mat
                total = contrib if total is None else total + contrib
            merged[layer] = total.reshape_as(items[0][1]).to(items[0][1].dtype)
        return merged

    def max_interference(self) -> float:
        values = [v for entries in self.report.values() for _, _, v in entries]
        return max(values) if values else 0.0


class NaiveMergeComposer:
    """Average of task updates."""

    @torch.no_grad()
    def compose(self, task_deltas: Sequence[Deltas], weights=None) -> Deltas:
        layers = set().union(*[d.keys() for d in task_deltas])
        return {l: torch.stack([d[l] for d in task_deltas if l in d]).mean(0) for l in layers}


class TaskArithmeticComposer:
    """``sum_t lambda * Delta W_t``."""

    def __init__(self, scaling: float = 1.0):
        self.scaling = scaling

    @torch.no_grad()
    def compose(self, task_deltas: Sequence[Deltas], weights=None) -> Deltas:
        layers = set().union(*[d.keys() for d in task_deltas])
        return {l: self.scaling * torch.stack([d[l] for d in task_deltas if l in d]).sum(0) for l in layers}


class TIESComposer:
    """TIES-Merging: trim to the top-``density`` magnitudes, elect signs, disjoint mean."""

    def __init__(self, density: float = 0.2, scaling: float = 1.0):
        self.density = density
        self.scaling = scaling

    @torch.no_grad()
    def compose(self, task_deltas: Sequence[Deltas], weights=None) -> Deltas:
        layers = set().union(*[d.keys() for d in task_deltas])
        merged = {}
        for l in layers:
            stack = torch.stack([d[l] for d in task_deltas if l in d]).float()
            flat = stack.reshape(stack.shape[0], -1)
            k = max(1, int(self.density * flat.shape[1]))
            thresh = flat.abs().kthvalue(flat.shape[1] - k + 1, dim=1, keepdim=True).values
            trimmed = torch.where(flat.abs() >= thresh, flat, torch.zeros_like(flat))
            sign = torch.sign(trimmed.sum(0))
            agree = (torch.sign(trimmed) == sign) & (trimmed != 0)
            count = agree.sum(0).clamp_min(1)
            mean = (trimmed * agree).sum(0) / count
            merged[l] = (self.scaling * mean).reshape(stack.shape[1:]).to(task_deltas[0][l].dtype)
        return merged
