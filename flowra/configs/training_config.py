"""Adapter-training configuration shared by FLowRA and the baselines."""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 3
    max_steps: Optional[int] = None
    lr_schedule: str = "linear"  # "linear" | "cosine" | "constant" | "poly"
    lr_warmup_ratio: float = 0.06
    grad_clip: Optional[float] = 1.0
    log_every: int = 10
    eval_every: Optional[int] = None
    # Estimate of ||grad_{h_l} L_t||_2 used during rank refinement:
    # "step" (current minibatch), "ema" (beta = 0.9) or "window" (mean since last refinement).
    refine_grad_estimator: str = "step"
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
