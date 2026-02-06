"""
Adapter Implementations

Low-rank adapter modules for different layer types:
- LoRA adapters for linear and convolutional layers
- Attention-specific adapters with Q/K/V split
- Normalization adapters for LayerNorm/BatchNorm
"""

from flowra.core.adapters.base import BaseAdapter, AdapterFactory
from flowra.core.adapters.lora import LoRAAdapter, LoRALinear, LoRAConv2d
from flowra.core.adapters.attention import AttentionAdapter, MultiHeadAttentionAdapter
from flowra.core.adapters.normalization import (
    NormalizationAdapter, 
    LayerNormAdapter, 
    BatchNormAdapter
)
from flowra.core.adapters.factory import PolymorphicAdapterFactory

__all__ = [
    "BaseAdapter",
    "AdapterFactory",
    "LoRAAdapter",
    "LoRALinear",
    "LoRAConv2d",
    "AttentionAdapter",
    "MultiHeadAttentionAdapter",
    "NormalizationAdapter",
    "LayerNormAdapter",
    "BatchNormAdapter",
    "PolymorphicAdapterFactory",
]
