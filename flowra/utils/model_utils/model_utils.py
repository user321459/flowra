"""Model Utilities"""
import torch
import torch.nn as nn
from typing import Dict, Any
import copy

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def get_layer_info(model: nn.Module) -> Dict[str, Dict]:
    info = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
            params = list(module.parameters())
            if params:
                info[name] = {
                    'type': type(module).__name__,
                    'shape': tuple(params[0].shape),
                    'params': sum(p.numel() for p in params),
                    'trainable': any(p.requires_grad for p in params)
                }
    return info

def merge_adapters_into_model(model: nn.Module, adapters: Dict[str, nn.Module]) -> nn.Module:
    merged = copy.deepcopy(model)
    for name, adapter in adapters.items():
        parts = name.split('.')
        parent = merged
        for part in parts[:-1]:
            parent = getattr(parent, part)
        layer = getattr(parent, parts[-1])
        if hasattr(adapter, 'get_adaptation'):
            with torch.no_grad():
                layer.weight.add_(adapter.scaling * adapter.get_adaptation())
    return merged
