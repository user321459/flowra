"""
FLowRA: Flow-Guided Rank Allocation for Parameter-Efficient Fine-Tuning.

Calibration computes a flow-change descriptor ``D_l`` of hidden-state
trajectories and a layer criticality ``C_l = |D_l| ||grad_{h_l} L||_2 ||W_l||_F``.
Ranks are allocated under a trainable-parameter budget ``B`` and refined during
training from updated gradient statistics. User-configurable parameters: the
budget ``B`` and the composition threshold ``eta``.
"""

__version__ = "1.0.0"

from flowra.algorithms import (
    NaiveMergeComposer,
    OrthogonalComposer,
    RankRefiner,
    TaskArithmeticComposer,
    TIESComposer,
    adapter_deltas,
    allocate_ranks,
    apply_deltas,
    generate_block_blueprint,
    refinement_schedule,
)
from flowra.configs import FlowConfig, TrainingConfig
from flowra.core.adapters import (
    BlockLoRA,
    DoRALinear,
    GatedRNNAdapter,
    LoRAConv,
    LoRALinear,
    LowRankFactors,
    NormAffineAdapter,
    PolymorphicAdapterFactory,
)
from flowra.core.analysis import (
    CalibrationProfile,
    FlowProfiler,
    ModuleProfile,
    align_states,
    flow_change,
    get_handler,
)
from flowra.core.framework import FLOWRA
from flowra.core.training import FlowraTrainer
from flowra.utils.logging_utils import ExperimentLogger

__all__ = [
    "__version__",
    "FLOWRA",
    "FlowConfig",
    "TrainingConfig",
    "FlowProfiler",
    "CalibrationProfile",
    "ModuleProfile",
    "flow_change",
    "align_states",
    "get_handler",
    "allocate_ranks",
    "generate_block_blueprint",
    "RankRefiner",
    "refinement_schedule",
    "OrthogonalComposer",
    "NaiveMergeComposer",
    "TaskArithmeticComposer",
    "TIESComposer",
    "adapter_deltas",
    "apply_deltas",
    "LowRankFactors",
    "LoRALinear",
    "DoRALinear",
    "LoRAConv",
    "BlockLoRA",
    "GatedRNNAdapter",
    "NormAffineAdapter",
    "PolymorphicAdapterFactory",
    "FlowraTrainer",
    "ExperimentLogger",
]
