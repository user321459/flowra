"""
FLowRA configuration.

User-configurable parameters:

* ``budget`` - trainable-parameter budget ``B`` as a fraction of the model
  parameters (e.g. 0.0005-0.02);
* ``eta`` - composition threshold used when combining task-specific adapters
  (default 0.3).

Remaining settings are internal and fixed by default: ``gamma = 1``,
``beta`` / ``tau`` (block-level only), transformer attention/FFN and V/O
splits (``alpha = 0.7``), a refinement warm-up of 10% of ``T`` then every
100 steps, and up to 512 calibration examples from the training split.
"""

import json
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, List, Optional

GRANULARITIES = ("module", "block")
SCALING_MODES = ("flow", "lora", "rslora")


@dataclass
class FlowConfig:
    # User-configurable parameters
    budget: float = 0.0087
    eta: float = 0.3

    # Internal settings (not tuned by the user)
    gamma: float = 1.0
    beta: float = 1.0
    tau: int = 1
    granularity: str = "module"
    calibration_samples: int = 512
    eps: float = 1e-8
    alignment: str = "zero_pad"
    handler: str = "auto"
    fp16_states: bool = False
    block_sensitivity: str = "auto"
    refinement: bool = True
    warmup_ratio: float = 0.1
    refine_interval: int = 100
    scaling: str = "flow"
    min_flow_scale: float = 0.05
    lora_alpha: float = 16.0
    transformer_formulas: bool = True
    attention_alpha: float = 0.7       # fraction of a head rank assigned to V/O

    # Adapter realisation
    adapter_variant: str = "lora"
    polymorphic: bool = True
    dropout: float = 0.0

    # Ablation switches
    use_flow: bool = True
    orthogonal_composition: bool = True

    # Target selection (regex patterns; None = handler default).  Modules in
    # ``modules_to_save`` (e.g. a new classification head) are trained fully,
    # excluded from the targets and reported separately from the budget.
    target_modules: Optional[List[str]] = None
    exclude_modules: Optional[List[str]] = None
    modules_to_save: Optional[List[str]] = None

    device: Optional[str] = None
    seed: int = 0
    verbose: bool = True

    def __post_init__(self):
        if not 0 < self.budget <= 1:
            raise ValueError(f"budget must be in (0, 1], got {self.budget}")
        if not 0 <= self.eta <= 1:
            raise ValueError(f"eta must be in [0, 1], got {self.eta}")
        if self.granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {GRANULARITIES}")
        if self.scaling not in SCALING_MODES:
            raise ValueError(f"scaling must be one of {SCALING_MODES}")
        if self.gamma <= 0 or self.beta <= 0:
            raise ValueError("gamma and beta must be positive")
        if not 0.0 <= self.attention_alpha <= 1.0:
            raise ValueError(f"attention_alpha must be in [0, 1], got {self.attention_alpha}")

    @property
    def user_parameters(self) -> Dict[str, float]:
        return {"budget": self.budget, "eta": self.eta}

    def variant(self, name: str) -> "FlowConfig":
        """Named ablation: ``full``, ``no_flow``, ``no_refinement``,
        ``no_polymorphic``, ``no_composition``."""
        changes = {
            "full": {},
            "no_flow": {"use_flow": False},
            "no_refinement": {"refinement": False},
            "no_polymorphic": {"polymorphic": False},
            "no_composition": {"orthogonal_composition": False},
        }
        if name not in changes:
            raise ValueError(f"Unknown variant '{name}', expected one of {sorted(changes)}")
        return replace(self, **changes[name])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlowConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FlowConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
