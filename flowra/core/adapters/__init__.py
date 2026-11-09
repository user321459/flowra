"""Adapter modules used to realise the allocated ranks."""

from flowra.core.adapters.base import BaseAdapter, LowRankFactors, ResizeRecord
from flowra.core.adapters.block import BlockLoRA
from flowra.core.adapters.factory import PolymorphicAdapterFactory
from flowra.core.adapters.lora import DoRALinear, LoRAConv, LoRALinear
from flowra.core.adapters.normalization import NormAffineAdapter, is_norm_layer
from flowra.core.adapters.rnn import GatedRNNAdapter

__all__ = [
    "BaseAdapter",
    "LowRankFactors",
    "ResizeRecord",
    "BlockLoRA",
    "PolymorphicAdapterFactory",
    "DoRALinear",
    "LoRAConv",
    "LoRALinear",
    "NormAffineAdapter",
    "is_norm_layer",
    "GatedRNNAdapter",
]
