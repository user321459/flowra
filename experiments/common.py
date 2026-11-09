"""Shared paths, YAML loading and training helpers for experiment scripts."""

import os
import sys
from typing import Any, Dict, Optional

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flowra.configs.training_config import TrainingConfig  # noqa: E402


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_config(path: Optional[str], default_rel: str) -> str:
    if path:
        return path
    return os.path.join(ROOT, "experiments", "configs", default_rel)


def device_str() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_config(cfg: Dict[str, Any], seed: int, epochs: Optional[int] = None,
                    max_steps: Optional[int] = None, lr: Optional[float] = None,
                    eval_every: Optional[int] = None) -> TrainingConfig:
    raw = dict(cfg.get("training") or {})
    raw["seed"] = seed
    if epochs is not None:
        raw["epochs"] = epochs
    if max_steps is not None:
        raw["max_steps"] = max_steps
    if lr is not None:
        raw["lr"] = lr
    if eval_every is not None:
        raw["eval_every"] = eval_every
    return TrainingConfig.from_dict(raw)
