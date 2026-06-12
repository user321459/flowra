"""
FLOWRA: Flow-based Low-Rank Adaptation Framework
=================================================

A unified information flow analysis framework for optimizing low-rank adaptations
in neural networks. Implements spectral-aware rank allocation, polymorphic adapters,
progressive subspace refinement, orthogonal composition, and flow-aware initialization.

Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Team"
__license__ = "MIT"

# Core framework
from flowra.core.framework import FLOWRA
from flowra.core.analysis import FlowAnalyzer, FlowProfile, LayerFlowInfo
from flowra.core.adapters import (
    BaseAdapter,
    LoRAAdapter,
    LoRALinear,
    LoRAConv2d,
    AttentionAdapter,
    MultiHeadAttentionAdapter,
    NormalizationAdapter,
    LayerNormAdapter,
    BatchNormAdapter,
    AdapterFactory,
    PolymorphicAdapterFactory,
)
from flowra.core.training import (
    FlowraTrainer,
    TrainingConfig,
    TrainingCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    LoggingCallback,
)

# Algorithms
from flowra.algorithms.allocation import (
    RankAllocator,
    SpectralRankAllocator,
    UniformRankAllocator,
    AdaptiveRankAllocator,
)
from flowra.algorithms.refinement import (
    SubspaceRefiner,
    ProgressiveSubspaceRefiner,
    GradientSubspaceTracker,
)
from flowra.algorithms.composition import (
    AdapterComposer,
    OrthogonalComposer,
    LinearComposer,
    TaskArithmeticComposer,
)
from flowra.algorithms.initialization import (
    AdapterInitializer,
    FlowAwareInitializer,
    KaimingInitializer,
    SVDInitializer,
)

# Configs
from flowra.configs import (
    FlowConfig,
    AdapterConfig,
    TrainingConfig,
    AnalysisConfig,
    CompositionConfig,
)

# Utils
from flowra.utils import (
    save_adapter,
    load_adapter,
    count_parameters,
    get_layer_info,
    merge_adapters_into_model,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "FLOWRA",
    "FlowAnalyzer",
    "FlowProfile",
    "LayerFlowInfo",
    # Adapters
    "BaseAdapter",
    "LoRAAdapter",
    "LoRALinear",
    "LoRAConv2d",
    "AttentionAdapter",
    "MultiHeadAttentionAdapter",
    "NormalizationAdapter",
    "LayerNormAdapter",
    "BatchNormAdapter",
    "AdapterFactory",
    "PolymorphicAdapterFactory",
    # Training
    "FlowraTrainer",
    "TrainingConfig",
    "TrainingCallback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "LoggingCallback",
    # Algorithms
    "RankAllocator",
    "SpectralRankAllocator",
    "UniformRankAllocator",
    "AdaptiveRankAllocator",
    "SubspaceRefiner",
    "ProgressiveSubspaceRefiner",
    "GradientSubspaceTracker",
    "AdapterComposer",
    "OrthogonalComposer",
    "LinearComposer",
    "TaskArithmeticComposer",
    "AdapterInitializer",
    "FlowAwareInitializer",
    "KaimingInitializer",
    "SVDInitializer",
    # Configs
    "FlowConfig",
    "AdapterConfig",
    "AnalysisConfig",
    "CompositionConfig",
    # Utils
    "save_adapter",
    "load_adapter",
    "count_parameters",
    "get_layer_info",
    "merge_adapters_into_model",
]


def get_version():
    """Return the current version of FLOWRA."""
    return __version__


def get_device():
    """Get the best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
