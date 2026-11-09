"""
Adapter training loop.

Only the adapter parameters ``Phi`` are updated.  At every step
``t in T_ref`` the trainer calls ``refine_fn(t)``; when ranks change, the
optimizer is rebuilt and the AdamW moments of retained components are
carried over.
"""

import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from flowra.configs.training_config import TrainingConfig
from flowra.core.adapters.base import ResizeRecord
from flowra.utils.batching import forward_loss, move_to_device
from flowra.utils.logging_utils import ExperimentLogger


def lr_multiplier(step: int, total: int, warmup: int, schedule: str) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    if schedule == "poly":
        return (1.0 - min(progress, 1.0)) ** 0.9  # SETR-Naive ADE20K
    return max(0.0, 1.0 - progress)


class FlowraTrainer:
    def __init__(
        self,
        model: nn.Module,
        params_fn: Callable[[], Iterable[nn.Parameter]],
        train_loader: Iterable,
        config: TrainingConfig,
        loss_fn=None,
        eval_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
        refine_fn: Optional[Callable[[int], Optional[List[ResizeRecord]]]] = None,
        refine_steps: Sequence[int] = (),
        logger: Optional[ExperimentLogger] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.params_fn = params_fn
        self.train_loader = train_loader
        self.config = config
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.refine_fn = refine_fn
        self.refine_steps = set(refine_steps)
        self.logger = logger or ExperimentLogger(console=True)
        self.device = device or next(model.parameters()).device
        self.global_step = 0
        self.history: List[Dict[str, Any]] = []
        self._last_eval: Optional[Dict[str, float]] = None
        self.optimizer = self._new_optimizer()

    @property
    def total_steps(self) -> int:
        if self.config.max_steps:
            return self.config.max_steps
        return self.config.epochs * len(self.train_loader)

    def _new_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.params_fn() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.config.lr, weight_decay=self.config.weight_decay)

    def _rebuild_optimizer(self, records: List[ResizeRecord]) -> None:
        old = self.optimizer
        new = self._new_optimizer()
        for group_new, group_old in zip(new.param_groups, old.param_groups):
            group_new["lr"] = group_old["lr"]
        replaced = {}
        for rec in records:
            replaced[rec.new_A] = (old.state.get(rec.old_A), rec, "A")
            replaced[rec.new_B] = (old.state.get(rec.old_B), rec, "B")
        for group in new.param_groups:
            for p in group["params"]:
                if p in replaced:
                    state, rec, which = replaced[p]
                    if state:
                        new.state[p] = _transfer_state(state, p, rec, which)
                elif p in old.state:
                    new.state[p] = old.state[p]
        self.optimizer = new

    def _set_lr(self) -> float:
        total = self.total_steps
        warmup = int(self.config.lr_warmup_ratio * total)
        lr = self.config.lr * lr_multiplier(self.global_step, total, warmup, self.config.lr_schedule)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def evaluate(self) -> Dict[str, float]:
        if self.eval_fn is None:
            return {}
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            metrics = self.eval_fn(self.model)
        self.model.train(was_training)
        return metrics

    def train(self) -> List[Dict[str, Any]]:
        cfg = self.config
        total = self.total_steps
        self.model.train()
        start = time.time()
        running, running_n = 0.0, 0
        done = False
        epoch = 0
        while not done:
            for batch in self.train_loader:
                if self.global_step >= total:
                    done = True
                    break
                lr = self._set_lr()
                batch = move_to_device(batch, self.device)
                loss, _ = forward_loss(self.model, batch, self.loss_fn)
                loss.backward()
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        [p for g in self.optimizer.param_groups for p in g["params"]], cfg.grad_clip
                    )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                running += float(loss.detach())
                running_n += 1
                step = self.global_step

                if self.refine_fn is not None and step in self.refine_steps:
                    records = self.refine_fn(step)
                    if records is not None:
                        self._rebuild_optimizer(records)

                self.global_step += 1
                if cfg.log_every and self.global_step % cfg.log_every == 0:
                    rec = {"step": self.global_step, "epoch": epoch, "loss": running / running_n, "lr": lr}
                    self.history.append(rec)
                    self.logger.log("train", rec, f"step {self.global_step:>6d}/{total} | epoch {epoch} | "
                                                  f"loss {rec['loss']:.4f} | lr {lr:.2e}")
                    running, running_n = 0.0, 0
                if cfg.eval_every and self.global_step % cfg.eval_every == 0:
                    self._log_eval(epoch)
            epoch += 1
            if not cfg.max_steps and epoch >= cfg.epochs:
                done = True
        if cfg.eval_every and self.global_step % cfg.eval_every == 0 and self._last_eval is not None:
            metrics = self._last_eval
        else:
            metrics = self._log_eval(epoch, final=True)
        self.logger.log("train_end", {"steps": self.global_step, "seconds": time.time() - start, **metrics},
                        f"training finished: {self.global_step} steps in {time.time() - start:.1f}s")
        return self.history

    def _log_eval(self, epoch: int, final: bool = False) -> Dict[str, float]:
        metrics = self.evaluate()
        self._last_eval = metrics
        if metrics:
            rec = {"step": self.global_step, "epoch": epoch, "final": final, **metrics}
            self.history.append(rec)
            self.logger.log("eval", rec, f"eval @ step {self.global_step}: " +
                            ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return metrics


def _transfer_state(state: Dict[str, Any], new_param: torch.Tensor, rec: ResizeRecord, which: str) -> Dict[str, Any]:
    kept = rec.kept
    k = len(kept)
    out = {"step": state["step"].clone() if torch.is_tensor(state["step"]) else state["step"]}
    for key in ("exp_avg", "exp_avg_sq"):
        old = state.get(key)
        if old is None:
            continue
        new = torch.zeros_like(new_param)
        if which == "A":
            if k:
                new[:k] = old[kept]
                if key == "exp_avg_sq" and new.shape[0] > k:
                    new[k:] = old[kept].mean(dim=0, keepdim=True)
        else:
            scale = rec.b_scale if key == "exp_avg" else rec.b_scale ** 2
            if k:
                new[:, :k] = old[:, kept] * scale
                if key == "exp_avg_sq" and new.shape[1] > k:
                    new[:, k:] = new[:, :k].mean(dim=1, keepdim=True)
        out[key] = new
    return out
