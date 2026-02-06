"""
Training Callbacks

Callbacks for monitoring and controlling training.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import os
import json


class TrainingCallback(ABC):
    """Base class for training callbacks."""
    
    def on_train_begin(self, trainer: Any) -> None:
        pass
    
    def on_train_end(self, trainer: Any) -> None:
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        pass
    
    def on_step_begin(self, trainer: Any, step: int) -> None:
        pass
    
    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        pass
    
    def on_evaluate(self, trainer: Any, metrics: Dict[str, float]) -> None:
        pass


class EarlyStoppingCallback(TrainingCallback):
    """Stop training when metric stops improving."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0, metric: str = "eval_loss", mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
    
    def on_evaluate(self, trainer: Any, metrics: Dict[str, float]) -> None:
        if self.metric not in metrics:
            return
        current = metrics[self.metric]
        improved = (current < self.best_value - self.min_delta) if self.mode == 'min' else (current > self.best_value + self.min_delta)
        if improved:
            self.best_value = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class CheckpointCallback(TrainingCallback):
    """Save model checkpoints."""
    
    def __init__(self, output_dir: str, save_strategy: str = "epoch", save_steps: int = 500, save_total_limit: int = 3):
        self.output_dir = output_dir
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.saved_checkpoints: List[str] = []
        os.makedirs(output_dir, exist_ok=True)
    
    def _save(self, trainer: Any, name: str):
        path = os.path.join(self.output_dir, name)
        state = {'adapters': {k: v.state_dict() for k, v in trainer.adapters.items()}, 'step': trainer.global_step}
        torch.save(state, path)
        self.saved_checkpoints.append(path)
        if self.save_total_limit and len(self.saved_checkpoints) > self.save_total_limit:
            old = self.saved_checkpoints.pop(0)
            if os.path.exists(old):
                os.remove(old)
    
    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        if self.save_strategy == "steps" and step > 0 and step % self.save_steps == 0:
            self._save(trainer, f"checkpoint-{step}.pt")
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        if self.save_strategy == "epoch":
            self._save(trainer, f"checkpoint-epoch-{epoch}.pt")


class LoggingCallback(TrainingCallback):
    """Log training metrics."""
    
    def __init__(self, log_steps: int = 10):
        self.log_steps = log_steps
        self.history: List[Dict] = []
    
    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        if step > 0 and step % self.log_steps == 0:
            lr = trainer.scheduler.get_last_lr()[0] if hasattr(trainer, 'scheduler') else 0
            self.history.append({"step": step, "loss": loss, "lr": lr})
            print(f"Step {step}: loss={loss:.4f}, lr={lr:.2e}")


class RefinementCallback(TrainingCallback):
    """Progressive subspace refinement callback."""
    
    def __init__(self, refiner: Any, interval: int = 100, warmup: int = 100):
        self.refiner = refiner
        self.interval = interval
        self.warmup = warmup
    
    def on_step_end(self, trainer: Any, step: int, loss: float) -> None:
        if step >= self.warmup and step % self.interval == 0:
            changes = self.refiner.refine(trainer.adapters, step)
            if changes:
                print(f"Refined {len(changes)} adapters at step {step}")
