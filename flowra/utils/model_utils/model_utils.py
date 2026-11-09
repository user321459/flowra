"""Model Utilities"""
import torch.nn as nn
from typing import Dict


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
