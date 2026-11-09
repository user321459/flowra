"""
Model profiling on the calibration set.

For every adaptation target ``l`` the profiler computes

* ``D_l`` - mean flow-change descriptor of ``(h_{l-1}, h_l, h_{l+1})``
* ``D~_l = |D_l|``
* ``G_l = E ||grad_{h_l} L||_2`` (task-loss gradient w.r.t. the output of l)
* ``||W_l||_F``
* ``C_l = D~_l * G_l * ||W_l||_F``

and, optionally, the block sensitivities ``S_l^(i,j)``.

The backbone is frozen during calibration: task-loss gradients are used only
to estimate sensitivity, no weight is updated.
"""

import contextlib
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from flowra.core.analysis.flow_descriptor import DEFAULT_EPS, FixedProjector
from flowra.core.analysis.handlers import (
    ArchitectureHandler,
    BlockLayout,
    CaptureStore,
    DescriptorSettings,
    TargetInfo,
    get_handler,
)
from flowra.utils.batching import batch_size, forward_loss, iter_samples, move_to_device, slice_batch


@dataclass
class ModuleProfile:
    """Calibration statistics of one adaptation target."""

    name: str
    kind: str
    group: str
    shape: Tuple[int, int]
    descriptor: float = 0.0  # signed D_l
    grad_norm: float = 0.0  # E ||grad_{h_l} L||_2
    weight_norm: float = 0.0  # ||W_l||_F
    num_samples: int = 0
    block_sensitivity: Optional[List[List[float]]] = None  # S_l^(i,j)
    block_layout: Optional[BlockLayout] = None

    @property
    def descriptor_abs(self) -> float:
        """Non-negative magnitude D~_l = |D_l| used for allocation."""
        return abs(self.descriptor)

    @property
    def criticality(self) -> float:
        """C_l = D~_l * ||grad_{h_l} L||_2 * ||W_l||_F."""
        return self.descriptor_abs * self.grad_norm * self.weight_norm

    @property
    def gradient_criticality(self) -> float:
        """C_l without the flow descriptor."""
        return self.grad_norm * self.weight_norm

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["descriptor_abs"] = self.descriptor_abs
        d["criticality"] = self.criticality
        d.pop("block_sensitivity")
        d.pop("block_layout")
        return d


@dataclass
class CalibrationProfile:
    """Result of Phase 1."""

    modules: Dict[str, ModuleProfile] = field(default_factory=dict)
    point_descriptors: Dict[str, float] = field(default_factory=dict)
    head_descriptors: Dict[int, List[float]] = field(default_factory=dict)  # mean |D_h|
    num_samples: int = 0
    handler: str = ""
    elapsed_sec: float = 0.0

    def criticalities(self, use_flow: bool = True) -> Dict[str, float]:
        if use_flow:
            return {n: p.criticality for n, p in self.modules.items()}
        return {n: p.gradient_criticality for n, p in self.modules.items()}

    def shapes(self) -> Dict[str, Tuple[int, int]]:
        return {n: p.shape for n, p in self.modules.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "handler": self.handler,
            "num_samples": self.num_samples,
            "elapsed_sec": self.elapsed_sec,
            "point_descriptors": self.point_descriptors,
            "head_descriptors": {str(k): v for k, v in self.head_descriptors.items()},
            "modules": [p.to_dict() for p in self.modules.values()],
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_csv(self, path: str) -> None:
        """Layer-level observations (one row per target)."""
        fields = ["name", "kind", "group", "m", "n", "descriptor", "descriptor_abs",
                  "grad_norm", "weight_norm", "criticality", "num_samples"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for p in self.modules.values():
                writer.writerow({
                    "name": p.name, "kind": p.kind, "group": p.group, "m": p.shape[0], "n": p.shape[1],
                    "descriptor": p.descriptor, "descriptor_abs": p.descriptor_abs, "grad_norm": p.grad_norm,
                    "weight_norm": p.weight_norm, "criticality": p.criticality, "num_samples": p.num_samples,
                })


def block_sum(matrix: torch.Tensor, layout: BlockLayout) -> torch.Tensor:
    """Sum of ``matrix`` over every block of ``layout`` -> ``[p, q]`` (2-D prefix sums)."""
    rows, cols = layout
    prefix = torch.nn.functional.pad(matrix.double().cumsum(0).cumsum(1), (1, 0, 1, 0))
    r = torch.tensor([b for b, _ in rows] + [rows[-1][1]], device=matrix.device)
    c = torch.tensor([b for b, _ in cols] + [cols[-1][1]], device=matrix.device)
    r0, r1 = r[:-1].unsqueeze(1), r[1:].unsqueeze(1)
    c0, c1 = c[:-1].unsqueeze(0), c[1:].unsqueeze(0)
    return (prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0]).float()


@contextlib.contextmanager
def _frozen(model: nn.Module, trainable: Iterable[nn.Parameter] = ()):
    flags = {id(p): p.requires_grad for p in model.parameters()}
    trainable_ids = {id(p) for p in trainable}
    was_training = model.training
    fastpath = None
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "get_fastpath_enabled"):
        fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
    try:
        for p in model.parameters():
            p.requires_grad_(id(p) in trainable_ids)
        model.eval()
        yield
    finally:
        for p in model.parameters():
            p.requires_grad_(flags[id(p)])
            p.grad = None
        model.train(was_training)
        if fastpath is not None:
            torch.backends.mha.set_fastpath_enabled(fastpath)


class FlowProfiler:
    """
    Calibration-phase profiler.

    Args:
        model: frozen pre-trained model.
        handler: architecture handler or its name (``"auto"``, ``"transformer"``,
            ``"feedforward"``, ``"residual"``, ``"rnn"``).
        target_modules / exclude_modules: optional regex patterns restricting
            the adaptation targets.
        alignment: dimension-alignment strategy for mismatched representations
            (``"zero_pad"`` by default, ``"linear"`` or ``"conv1x1"``).
        fp16_states: store calibration activations in FP16.
    """

    def __init__(
        self,
        model: nn.Module,
        handler: Union[str, ArchitectureHandler] = "auto",
        target_modules: Optional[Sequence[str]] = None,
        exclude_modules: Optional[Sequence[str]] = None,
        eps: float = DEFAULT_EPS,
        alignment: str = "zero_pad",
        fp16_states: bool = False,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ):
        self.model = model
        self.handler = get_handler(model, handler) if isinstance(handler, str) else handler
        self.targets: Dict[str, TargetInfo] = self.handler.discover_targets(model, target_modules, exclude_modules)
        if not self.targets:
            raise ValueError("No adaptation targets found")
        self.settings = DescriptorSettings(eps=eps, alignment=alignment, projector=FixedProjector(seed))
        self.fp16_states = fp16_states
        self.device = device or next(model.parameters()).device

    # ------------------------------------------------------------------
    def _target_params(self, info: TargetInfo) -> List[nn.Parameter]:
        if info.kind == "gate":
            l = info.meta["layer"]
            return [getattr(info.module, f"weight_ih_l{l}"), getattr(info.module, f"weight_hh_l{l}")]
        return [info.module.weight]

    def _weight_grad_2d(self, info: TargetInfo, grads: Dict[int, torch.Tensor]) -> Optional[torch.Tensor]:
        params = self._target_params(info)
        if any(grads.get(id(p)) is None for p in params):
            return None
        if info.kind == "gate":
            h, gi = info.module.hidden_size, info.meta["gate_idx"]
            return torch.cat([grads[id(p)][gi * h:(gi + 1) * h] for p in params], dim=1)
        g = grads[id(params[0])]
        return g.reshape(g.shape[0], -1)

    def _batches(self, loader: Iterable, max_samples: int):
        seen = 0
        for batch in loader:
            if seen >= max_samples:
                break
            n = batch_size(batch)
            if seen + n > max_samples:
                batch = slice_batch(batch, 0, max_samples - seen)
                n = max_samples - seen
            seen += n
            yield move_to_device(batch, self.device), n

    # ------------------------------------------------------------------
    def profile(
        self,
        loader: Iterable,
        loss_fn=None,
        max_samples: int = 512,
        block_level: bool = False,
        block_sensitivity: str = "auto",
        max_block_samples: Optional[int] = None,
    ) -> CalibrationProfile:
        """
        Run profiling on up to ``max_samples`` calibration examples.

        ``block_sensitivity`` selects the block score: ``"eq3"`` (element-wise
        product of the descriptor Jacobian and the task gradient), ``"eq4"``
        (product of Jacobian and gradient Frobenius norms, for fully connected
        layers) or ``"auto"`` (``eq4`` for linear layers, ``eq3`` otherwise).
        """
        start = time.time()
        d_sum = {n: 0.0 for n in self.targets}
        g_sum = {n: 0.0 for n in self.targets}
        counts = {n: 0 for n in self.targets}
        g_counts = {n: 0 for n in self.targets}
        point_sum: Dict[str, float] = {}
        point_count: Dict[str, int] = {}
        head_sum: Dict[int, torch.Tensor] = {}
        head_count: Dict[int, int] = {}
        total = 0

        with _frozen(self.model):
            for batch, n in self._batches(loader, max_samples):
                store = CaptureStore(batch_dim=self.handler.batch_dim, fp16=self.fp16_states)
                handles = self.handler.attach(self.model, self.targets, store)
                try:
                    with torch.enable_grad():
                        loss, _ = forward_loss(self.model, batch, loss_fn)
                        if loss.requires_grad:
                            loss.backward()
                finally:
                    self.handler.detach(handles)

                with torch.no_grad():
                    divergences = self.handler.compute_divergence(store, self.targets, self.settings)
                    for name, d in divergences.items():
                        d_sum[name] += float(d.sum())
                        counts[name] += d.shape[0]
                    for name in self.targets:
                        g = store.per_sample_grad_norm(name)
                        if g is not None:
                            g_sum[name] += float(g.sum())
                            g_counts[name] += g.shape[0]
                    for key, d in self.handler.point_descriptors(store, self.settings).items():
                        point_sum[key] = point_sum.get(key, 0.0) + float(d.sum())
                        point_count[key] = point_count.get(key, 0) + d.shape[0]
                    compute_heads = getattr(self.handler, "compute_head_descriptors", None)
                    if compute_heads is not None:
                        for layer, d in compute_heads(store, self.settings).items():
                            head_sum[layer] = d.abs().sum(0) if layer not in head_sum else head_sum[layer] + d.abs().sum(0)
                            head_count[layer] = head_count.get(layer, 0) + d.shape[0]
                total += n
                del store

        profile = CalibrationProfile(num_samples=total, handler=self.handler.name)
        for name, info in self.targets.items():
            with torch.no_grad():
                w_norm = float(self.handler.weight_matrix(info).float().norm())
            profile.modules[name] = ModuleProfile(
                name=name, kind=info.kind, group=info.group, shape=tuple(info.shape),
                descriptor=d_sum[name] / max(counts[name], 1),
                grad_norm=g_sum[name] / max(g_counts[name], 1),
                weight_norm=w_norm,
                num_samples=counts[name],
            )
        profile.point_descriptors = {k: point_sum[k] / point_count[k] for k in point_sum}
        profile.head_descriptors = {
            l: (head_sum[l] / max(head_count[l], 1)).detach().cpu().tolist() for l in head_sum
        }

        if block_level:
            self.compute_block_sensitivities(
                loader, profile, loss_fn,
                max_samples=max_block_samples or max_samples, mode=block_sensitivity,
            )
        profile.elapsed_sec = time.time() - start
        return profile

    # ------------------------------------------------------------------
    def compute_block_sensitivities(
        self,
        loader: Iterable,
        profile: CalibrationProfile,
        loss_fn=None,
        max_samples: int = 512,
        mode: str = "auto",
    ) -> None:
        """
        Block sensitivities, computed per calibration sample:

        * ``eq3``: ``S^(i,j) = 1/N sum_k || dD_l/dW^(i,j) ⊙ grad_{W^(i,j)} L ||_1``
        * ``eq4``: ``S^(i,j) = E[ ||J^(i,j)||_F ||G^(i,j)||_F ]`` with
          ``J^(i,j) = d h_out / d W^(i,j)``; for ``h_out = W x`` this is
          ``||J^(i,j)||_F^2 = |rows of block| * sum_tokens ||x_cols||^2``.
        """
        layouts = {n: self.handler.block_layout(info) for n, info in self.targets.items()}
        sums = {n: None for n in self.targets}
        counted = 0
        all_params = [p for info in self.targets.values() for p in self._target_params(info)]
        uniq_params = list({id(p): p for p in all_params}.values())

        def use_eq4(info: TargetInfo) -> bool:
            return info.kind == "linear" and mode in ("auto", "eq4")

        with _frozen(self.model, trainable=uniq_params):
            for batch, _ in self._batches(loader, max_samples):
                for sample in iter_samples(batch):
                    store = CaptureStore(batch_dim=self.handler.batch_dim, keep_graph=True)
                    inputs: Dict[str, torch.Tensor] = {}
                    pre_handles = []
                    for name, info in self.targets.items():
                        if use_eq4(info):
                            def pre(module, args, name=name):
                                inputs[name] = args[0].detach()
                            pre_handles.append(info.module.register_forward_pre_hook(pre))
                    handles = self.handler.attach(self.model, self.targets, store)
                    try:
                        with torch.enable_grad():
                            loss, _ = forward_loss(self.model, sample, loss_fn)
                            divergences = self.handler.compute_divergence(store, self.targets, self.settings)
                    finally:
                        self.handler.detach(handles)
                        for h in pre_handles:
                            h.remove()

                    loss_grads = torch.autograd.grad(loss, uniq_params, retain_graph=True, allow_unused=True)
                    loss_grads = {id(p): g for p, g in zip(uniq_params, loss_grads)}

                    d_grads: Dict[int, Dict[int, torch.Tensor]] = {}
                    groups: Dict[int, List[str]] = {}
                    for name, d in divergences.items():
                        groups.setdefault(id(d), []).append(name)
                    if mode != "eq4":
                        for key, names in groups.items():
                            needed = [n for n in names if not use_eq4(self.targets[n])]
                            if not needed:
                                continue
                            d = divergences[needed[0]]
                            params = list({id(p): p for n in needed for p in self._target_params(self.targets[n])}.values())
                            if not d.requires_grad:
                                continue
                            grads = torch.autograd.grad(d.mean(), params, retain_graph=True, allow_unused=True)
                            d_grads[key] = {id(p): g for p, g in zip(params, grads)}

                    for name, info in self.targets.items():
                        g_l = self._weight_grad_2d(info, loss_grads)
                        if g_l is None:
                            continue
                        if use_eq4(info):
                            x = inputs.get(name)
                            if x is None:
                                continue
                            x_sq = x.float().pow(2).reshape(-1, x.shape[-1]).sum(0)  # [n]
                            rows, cols = layouts[name]
                            col_sq = torch.stack([x_sq[c0:c1].sum() for c0, c1 in cols])
                            row_sz = torch.tensor([r1 - r0 for r0, r1 in rows], dtype=torch.float32, device=x_sq.device)
                            j_norm = (row_sz.unsqueeze(1) * col_sq.unsqueeze(0)).sqrt()
                            g_norm = block_sum(g_l.float().pow(2), layouts[name]).sqrt()
                            s = j_norm * g_norm
                        else:
                            key = id(divergences.get(name))
                            g_d = self._weight_grad_2d(info, d_grads.get(key, {}))
                            if g_d is None:
                                continue
                            s = block_sum((g_d * g_l).abs().float(), layouts[name])
                        sums[name] = s if sums[name] is None else sums[name] + s
                    counted += 1
                    del store, loss, divergences

        for name, s in sums.items():
            mod = profile.modules[name]
            mod.block_layout = layouts[name]
            rows, cols = layouts[name]
            if s is None:
                s = torch.zeros(len(rows), len(cols))
            mod.block_sensitivity = (s / max(counted, 1)).cpu().tolist()
