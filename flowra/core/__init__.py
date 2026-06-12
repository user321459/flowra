"""
Core FLOWRA Components

This package contains the main framework implementation:
- analysis: Flow analysis and Fisher Information computation
- adapters: Low-rank adapter implementations
- training: Training loops and optimization
- framework: Main FLOWRA class
"""

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
    TrainingCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    LoggingCallback,
)

__all__ = [
    "FLOWRA",
    "FlowAnalyzer",
    "FlowProfile",
    "LayerFlowInfo",
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
    "FlowraTrainer",
    "TrainingCallback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "LoggingCallback",
]
