"""
FLowRA framework.

    framework = FLOWRA(model, FlowConfig(budget=0.0087))
    framework.adapt(calibration_loader)
    framework.train(train_loader, TrainingConfig(), eval_fn=...)

Profiling estimates ``D_l``, ``|D_l|`` and ``C_l`` (and, for
``granularity="block"``, the block sensitivities ``S_l^(i,j)``). Ranks are
then allocated under the budget and adapters are injected. During training
the ranks are refined at the steps in ``T_ref``.

The default (``granularity="module"``) assigns ranks directly to target modules.
"""

import math
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from flowra.algorithms.allocation.allocation import (
    AllocationResult,
    BlockAssignment,
    allocate_ranks,
    blueprint_by_layer,
    generate_block_blueprint,
)
from flowra.algorithms.allocation.transformer import (
    TransformerRankAdjuster,
    signals_from_profile,
    transformer_groups,
)
from flowra.algorithms.refinement.refinement import RankRefiner, refinement_schedule
from flowra.configs.flow_config import FlowConfig
from flowra.configs.training_config import TrainingConfig
from flowra.core.adapters.base import BaseAdapter, LowRankFactors, ResizeRecord
from flowra.core.adapters.block import BlockLoRA
from flowra.core.adapters.factory import PolymorphicAdapterFactory
from flowra.core.adapters.normalization import is_norm_layer
from flowra.core.analysis.handlers import (
    FeedForwardHandler,
    ResidualCNNHandler,
    TransformerHandler,
    first_tensor,
    name_matches,
)
from flowra.core.analysis.profiler import CalibrationProfile, FlowProfiler
from flowra.core.training.trainer import FlowraTrainer
from flowra.utils.logging_utils import ExperimentLogger


def set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, module)


class GradNormTracker:
    """Tracks ``||grad_{h_l} L_t||_2`` of adapted module outputs during training."""

    def __init__(self, modules: Dict[str, nn.Module], batch_dim: int = 0, estimator: str = "ema",
                 ema_beta: float = 0.9):
        if estimator not in ("step", "window", "ema"):
            raise ValueError(f"Unknown gradient estimator '{estimator}'")
        self.batch_dim = batch_dim
        self.estimator = estimator
        self.ema_beta = ema_beta
        self.ema: Dict[str, float] = {}
        self.latest: Dict[str, float] = {}
        self.window_sum: Dict[str, float] = {}
        self.window_n: Dict[str, int] = {}
        self.handles = []
        for name, module in modules.items():

            def hook(mod, args, output, name=name):
                t = first_tensor(output)
                if mod.training and t is not None and t.requires_grad:
                    t.register_hook(lambda g, name=name: self._record(name, g))

            self.handles.append(module.register_forward_hook(hook))

    def _record(self, name: str, grad: torch.Tensor) -> None:
        g = grad.movedim(self.batch_dim, 0) if self.batch_dim else grad
        n = g.shape[0]
        value = float(g.detach().float().reshape(n, -1).norm(dim=1).mean()) * n
        self.latest[name] = value
        prev = self.ema.get(name)
        self.ema[name] = value if prev is None else self.ema_beta * prev + (1 - self.ema_beta) * value
        self.window_sum[name] = self.window_sum.get(name, 0.0) + value
        self.window_n[name] = self.window_n.get(name, 0) + 1

    def values(self) -> Dict[str, float]:
        if self.estimator == "window":
            return {k: self.window_sum[k] / self.window_n[k] for k in self.window_sum}
        if self.estimator == "ema":
            return dict(self.ema)
        return dict(self.latest)

    def reset_window(self) -> None:
        self.window_sum.clear()
        self.window_n.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


class FLOWRA:
    def __init__(self, model: nn.Module, config: Optional[FlowConfig] = None, logger: Optional[ExperimentLogger] = None):
        self.config = config or FlowConfig()
        cfg = self.config
        self.logger = logger or ExperimentLogger(console=cfg.verbose)
        self.device = torch.device(cfg.device) if cfg.device else next(model.parameters()).device
        self.model = model.to(self.device)
        torch.manual_seed(cfg.seed)

        exclude = list(cfg.exclude_modules or []) + list(cfg.modules_to_save or [])
        self.profiler = FlowProfiler(
            model, handler=cfg.handler, target_modules=cfg.target_modules, exclude_modules=exclude or None,
            eps=cfg.eps, alignment=cfg.alignment, fp16_states=cfg.fp16_states, device=self.device, seed=cfg.seed,
        )
        self.handler = self.profiler.handler
        self.targets = self.profiler.targets
        self.factory = PolymorphicAdapterFactory(cfg.adapter_variant, cfg.dropout, cfg.polymorphic)

        self.base_params = sum(p.numel() for p in model.parameters())
        self.budget_total = int(round(cfg.budget * self.base_params))
        self.norm_layers = self._find_norm_layers() if cfg.polymorphic else {}
        self.norm_params = sum(self._norm_cost(m) for m in self.norm_layers.values())
        if self.norm_params > self.budget_total // 2:
            self.norm_layers, self.norm_params = {}, 0
        self.lowrank_budget = self.budget_total - self.norm_params

        self.profile_result: Optional[CalibrationProfile] = None
        self.allocation: Optional[AllocationResult] = None
        self.blueprint: Optional[List[BlockAssignment]] = None
        self.adapters: Dict[str, BaseAdapter] = {}
        self.factor_map: Dict[str, LowRankFactors] = {}
        self.block_adapters: Dict[str, BlockLoRA] = {}
        self.flow_scale: Dict[str, float] = {}
        self.rank_history: List[Dict[str, Any]] = []
        self.refiner: Optional[RankRefiner] = None
        self._tracker: Optional[GradNormTracker] = None
        self._transformer_adjuster: Optional[TransformerRankAdjuster] = None

        self.logger.info(
            f"FLowRA: {self.base_params:,} base parameters, {len(self.targets)} targets "
            f"(handler={self.handler.name}), budget B={cfg.budget:.4%} -> {self.budget_total:,} parameters"
            + (f" ({self.norm_params:,} for {len(self.norm_layers)} normalisation adapters)" if self.norm_layers else "")
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _find_norm_layers(self) -> Dict[str, nn.Module]:
        if isinstance(self.handler, TransformerHandler):
            roots = [self.handler.blocks_name]
        elif isinstance(self.handler, ResidualCNNHandler):
            roots = [name for name, _ in self.handler.blocks]
        elif isinstance(self.handler, FeedForwardHandler):
            roots = [""]
        else:
            return {}
        saved = self.config.modules_to_save
        found = {}
        for name, mod in self.model.named_modules():
            if not name or not is_norm_layer(mod) or (saved and name_matches(name, saved)):
                continue
            if any(r == "" or name.startswith(r + ".") for r in roots):
                found[name] = mod
        return found

    @staticmethod
    def _norm_cost(mod: nn.Module) -> int:
        d = mod.weight.shape[0]
        return 2 * d if getattr(mod, "bias", None) is not None else d

    def criticality(self) -> Dict[str, float]:
        if self.profile_result is None:
            raise RuntimeError("Run profile() first")
        return self.profile_result.criticalities(use_flow=self.config.use_flow)

    def _scaling(self, name: str, rank: int) -> float:
        """Adapter scaling.  ``flow``: ``alpha_l = sqrt(psi_l / r_l)`` with ``psi_l = C_l / mean C``."""
        cfg = self.config
        if rank <= 0:
            return 1.0
        if cfg.scaling == "lora":
            return cfg.lora_alpha / rank
        if cfg.scaling == "rslora":
            return cfg.lora_alpha / math.sqrt(rank)
        return math.sqrt(self.flow_scale.get(name, 1.0) / rank)

    def _apply_transformer_formulas(self, result: AllocationResult, crit: Dict[str, float]) -> AllocationResult:
        """Redistribute ranks inside each transformer layer (attention/FFN and V/O split)."""
        cfg = self.config
        groups = transformer_groups(self.targets)
        if not groups:
            return result
        d_attn_ffn, grad_norms = signals_from_profile(
            groups, self.profile_result.point_descriptors, self.profile_result.modules,
        )
        self._transformer_adjuster = TransformerRankAdjuster(
            groups, d_attn_ffn, self.profile_result.head_descriptors,
            gamma=cfg.gamma, alpha=cfg.attention_alpha,
        )
        return self._transformer_adjuster(result, crit, {n: t.shape for n, t in self.targets.items()}, grad_norms)

    def _compute_flow_scale(self) -> None:
        crit = self.criticality()
        active = [c for c in crit.values() if c > 0]
        mean = sum(active) / len(active) if active else 1.0
        self.flow_scale = {n: max(c / mean, self.config.min_flow_scale) for n, c in crit.items()}

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------
    def profile(self, calibration_loader: Iterable, loss_fn=None) -> CalibrationProfile:
        cfg = self.config
        self.logger.info("=" * 70)
        self.logger.info(f"Phase 1: model profiling (N <= {cfg.calibration_samples} calibration samples)")
        prof = self.profiler.profile(
            calibration_loader, loss_fn, max_samples=cfg.calibration_samples,
            block_level=cfg.granularity == "block", block_sensitivity=cfg.block_sensitivity,
        )
        self.profile_result = prof
        extra = f", {len(prof.head_descriptors)} layers with head descriptors" if prof.head_descriptors else ""
        self.logger.info(f"profiled {len(prof.modules)} targets on {prof.num_samples} samples in {prof.elapsed_sec:.1f}s{extra}")
        self.logger.info(f"{'target':<48} {'D_l':>10} {'|D_l|':>9} {'G_l':>10} {'||W||_F':>9} {'C_l':>10}")
        for p in prof.modules.values():
            self.logger.info(f"{p.name:<48} {p.descriptor:>10.4e} {p.descriptor_abs:>9.3e} "
                             f"{p.grad_norm:>10.3e} {p.weight_norm:>9.3f} {p.criticality:>10.3e}")
        self.logger.log("profile", prof.to_dict())
        return prof

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------
    def allocate(self) -> AllocationResult:
        cfg = self.config
        crit = self.criticality()
        shapes = {n: t.shape for n, t in self.targets.items()}
        self.logger.info("=" * 70)
        self.logger.info(f"Phase 2: blueprint generation (gamma={cfg.gamma}, use_flow={cfg.use_flow}, "
                         f"granularity={cfg.granularity})")
        result = allocate_ranks(crit, shapes, self.lowrank_budget, gamma=cfg.gamma)
        if cfg.transformer_formulas and isinstance(self.handler, TransformerHandler):
            result = self._apply_transformer_formulas(result, crit)
        self.allocation = result
        self._compute_flow_scale()
        extra = ""
        if self._transformer_adjuster and self._transformer_adjuster.last_ratios:
            ratios = ", ".join(f"L{l}={r:.2f}" for l, r in sorted(self._transformer_adjuster.last_ratios.items())
                               if r != float("inf"))
            extra = f"; r_attn/r_ffn [{ratios}], alpha={cfg.attention_alpha} (V/O)"
        self.logger.info(f"proportional stage: {sum(result.proportional_ranks.values())} total rank, "
                         f"greedy stage: +{result.greedy_increments} increments, "
                         f"used {result.total_used:,}/{result.budget_total:,} parameters "
                         f"(remaining {result.budget_remaining:,}){extra}")
        for n, r in result.ranks.items():
            self.logger.info(f"  rank[{n}] = {r}")

        if cfg.granularity == "block":
            sens = {n: self.profile_result.modules[n].block_sensitivity for n in self.targets}
            layouts = {n: self.profile_result.modules[n].block_layout for n in self.targets}
            block_ranks = {n: r for n, r in result.ranks.items() if self.targets[n].kind != "gate"}
            self.blueprint = generate_block_blueprint(block_ranks, shapes, sens, layouts, cfg.beta, cfg.tau)
            used = sum(b.cost for b in self.blueprint)
            self.logger.info(f"block blueprint: {len(self.blueprint)} blocks, {used:,} parameters "
                             f"(beta={cfg.beta}, tau={cfg.tau})")
        self.logger.log("allocation", {
            "ranks": result.ranks, "proportional_ranks": result.proportional_ranks,
            "budget_total": result.budget_total, "budget_used": result.total_used,
            "criticality": crit, "granularity": cfg.granularity,
            "blocks": [b.__dict__ for b in self.blueprint] if self.blueprint else None,
        })
        self.rank_history.append({"step": 0, "ranks": dict(result.ranks)})
        return result

    # ------------------------------------------------------------------
    # Phase 3: injection
    # ------------------------------------------------------------------
    def inject(self) -> nn.Module:
        if self.allocation is None:
            raise RuntimeError("Run allocate() first")
        cfg = self.config
        ranks = self.allocation.ranks
        by_layer = blueprint_by_layer(self.blueprint) if self.blueprint is not None else None

        for p in self.model.parameters():
            p.requires_grad_(False)

        rnn_groups: Dict[str, Dict[str, Any]] = {}
        for name, target in self.targets.items():
            r = ranks.get(name, 0)
            if target.kind == "gate":
                group = rnn_groups.setdefault(target.meta["rnn"], {"module": target.module, "ranks": {}, "scaling": {}})
                key = f"l{target.meta['layer']}.{target.meta['gate']}"
                group["ranks"][key] = r
                group["scaling"][key] = self._scaling(name, r)
                continue
            if by_layer is not None:
                adapter = BlockLoRA(target.module, by_layer.get(name, []), self._scaling(name, r))
                self.block_adapters[name] = adapter
            else:
                adapter = self.factory.create(target, r, self._scaling(name, r))
                self.factor_map[name] = adapter.factors()[""]
            set_submodule(self.model, name, adapter)
            self.adapters[name] = adapter

        for rnn_name, group in rnn_groups.items():
            adapter = self.factory.create_rnn(group["module"], group["ranks"], group["scaling"])
            set_submodule(self.model, rnn_name, adapter)
            self.adapters[rnn_name] = adapter
            for key, f in adapter.factors().items():
                self.factor_map[f"{rnn_name}.{key}"] = f

        for name, norm in self.norm_layers.items():
            adapter = self.factory.create_norm(norm)
            set_submodule(self.model, name, adapter)
            self.adapters[name] = adapter

        for name, mod in self.model.named_modules():
            if cfg.modules_to_save and name and name_matches(name, cfg.modules_to_save):
                for p in mod.parameters():
                    p.requires_grad_(True)

        counts = self.parameter_counts()
        self.logger.info(
            f"injected {len(self.adapters)} adapters: {counts['adapter']:,} adapter parameters "
            f"({counts['adapter'] / self.base_params:.4%} of the model, budget {self.budget_total:,}), "
            f"{counts['modules_to_save']:,} parameters in modules_to_save"
        )
        self.logger.log("injection", counts)
        return self.model

    def adapt(self, calibration_loader: Iterable, loss_fn=None) -> nn.Module:
        self.profile(calibration_loader, loss_fn)
        self.allocate()
        return self.inject()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def adapter_parameters(self) -> List[nn.Parameter]:
        params = {}
        for adapter in self.adapters.values():
            for p in adapter.parameters():
                if p.requires_grad:
                    params[id(p)] = p
        return list(params.values())

    def trainable_parameters(self) -> List[nn.Parameter]:
        params = {id(p): p for p in self.adapter_parameters()}
        for p in self.model.parameters():
            if p.requires_grad:
                params[id(p)] = p
        return list(params.values())

    def parameter_counts(self) -> Dict[str, int]:
        adapter_ids = {id(p) for p in self.adapter_parameters()}
        adapter = sum(p.numel() for p in self.adapter_parameters())
        other = sum(p.numel() for p in self.model.parameters() if p.requires_grad and id(p) not in adapter_ids)
        return {"adapter": adapter, "modules_to_save": other, "budget": self.budget_total, "base": self.base_params}

    def current_ranks(self) -> Dict[str, int]:
        if self.block_adapters:
            return {n: sum(f.rank for f in a.factors().values()) for n, a in self.block_adapters.items()}
        return {n: f.rank for n, f in self.factor_map.items()}

    def train(
        self,
        train_loader: Iterable,
        training_config: Optional[TrainingConfig] = None,
        loss_fn=None,
        eval_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.adapters:
            raise RuntimeError("Run adapt() first")
        cfg, tcfg = self.config, training_config or TrainingConfig()
        total = tcfg.max_steps or tcfg.epochs * len(train_loader)
        schedule = refinement_schedule(total, cfg.warmup_ratio, cfg.refine_interval) if cfg.refinement else []
        self.logger.info("=" * 70)
        self.logger.info(f"Phase 3: adapter training, T={total} steps, lr={tcfg.lr}, "
                         f"refinement={'on' if schedule else 'off'}"
                         + (f" (T_ref: {len(schedule)} steps, first={schedule[0]}, every {cfg.refine_interval})"
                            if schedule else ""))
        if schedule:
            prof = self.profile_result
            tracked = {n: self.adapters[n] for n in self.targets if n in self.adapters}
            self._tracker = GradNormTracker(tracked, self.handler.batch_dim, tcfg.refine_grad_estimator)
            self.refiner = RankRefiner(
                descriptor_abs={n: p.descriptor_abs for n, p in prof.modules.items()},
                weight_norms={n: p.weight_norm for n, p in prof.modules.items()},
                shapes={n: t.shape for n, t in self.targets.items()},
                budget=self.lowrank_budget, gamma=cfg.gamma, use_flow=cfg.use_flow,
                initial_grad_norms={n: p.grad_norm for n, p in prof.modules.items()},
                adjuster=self._transformer_adjuster,
            )
        trainer = FlowraTrainer(
            self.model, self.trainable_parameters, train_loader, tcfg, loss_fn=loss_fn, eval_fn=eval_fn,
            refine_fn=self._refine if schedule else None, refine_steps=schedule, logger=self.logger,
            device=self.device,
        )
        try:
            history = trainer.train()
        finally:
            if self._tracker is not None:
                self._tracker.remove()
                self._tracker = None
        self.logger.log("rank_history", {"history": self.rank_history})
        return history

    def _refine(self, step: int) -> Optional[List[ResizeRecord]]:
        grads = self._tracker.values() if self._tracker else {}
        result = self.refiner.refine(step, grads)
        old = self.current_ranks()
        records: List[ResizeRecord] = []
        if self.block_adapters:
            prof = self.profile_result
            shapes = {n: t.shape for n, t in self.targets.items()}
            blocks = blueprint_by_layer(generate_block_blueprint(
                {n: r for n, r in result.ranks.items() if n in self.block_adapters}, shapes,
                {n: prof.modules[n].block_sensitivity for n in self.block_adapters},
                {n: prof.modules[n].block_layout for n in self.block_adapters},
                self.config.beta, self.config.tau,
            ))
            for n, adapter in self.block_adapters.items():
                records.extend(adapter.set_blocks(blocks.get(n, []), self._scaling(n, result.ranks.get(n, 0))))
        for name, r_new in result.ranks.items():
            f = self.factor_map.get(name)
            if f is None or f.rank == r_new:
                continue
            rec = f.resize(r_new, self._scaling(name, r_new))
            if rec:
                records.append(rec)
        new = self.current_ranks()
        changed = {n: (old[n], new[n]) for n in new if old.get(n) != new[n]}
        if self._tracker:
            self._tracker.reset_window()
        self.rank_history.append({"step": step, "ranks": dict(new)})
        moved = sum(abs(b - a) for a, b in changed.values())
        self.logger.log(
            "refine",
            {"step": step, "changed": changed, "ranks": new, "budget_used": result.total_used,
             "criticality": self.refiner.history[-1]["criticality"]},
            f"rank refinement @ step {step}: {len(changed)} targets changed (sum |dr| = {moved}), "
            f"budget {result.total_used:,}/{self.lowrank_budget:,}",
        )
        if not changed and not self.block_adapters:
            return None
        return records

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def merge(self) -> nn.Module:
        """Merge all adapters into the frozen weights (inference without overhead)."""
        for adapter in self.adapters.values():
            adapter.merge()
        return self.model

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        state = {}
        for name, adapter in self.adapters.items():
            for k, v in adapter.state_dict().items():
                if not k.startswith("base."):
                    state[f"{name}.{k}"] = v
        return state

    def save(self, path: str) -> None:
        torch.save({
            "config": self.config.to_dict(),
            "ranks": self.current_ranks(),
            "adapters": self.adapter_state_dict(),
            "profile": self.profile_result.to_dict() if self.profile_result else None,
        }, path)

    def load(self, path: str) -> None:
        """Load adapters saved by :meth:`save` into an adapted model (ranks are restored first)."""
        if self.block_adapters:
            raise NotImplementedError("Loading block-level adapters is not supported")
        ckpt = torch.load(path, map_location=self.device)
        for name, r in ckpt["ranks"].items():
            f = self.factor_map.get(name)
            if f is not None and f.rank != r:
                f.resize(r, self._scaling(name, r))
        for name, adapter in self.adapters.items():
            prefix = name + "."
            state = {k[len(prefix):]: v for k, v in ckpt["adapters"].items() if k.startswith(prefix)}
            missing, unexpected = adapter.load_state_dict(state, strict=False)
            if unexpected or any(not k.startswith("base.") for k in missing):
                raise RuntimeError(f"Adapter state mismatch for {name}: missing={missing}, unexpected={unexpected}")

    def rank_table(self) -> str:
        lines = [f"{'target':<48} {'rank':>5} {'C_l':>10}"]
        crit = self.criticality() if self.profile_result else {}
        for n, r in self.current_ranks().items():
            lines.append(f"{n:<48} {r:>5d} {crit.get(n, float('nan')):>10.3e}")
        return "\n".join(lines)
