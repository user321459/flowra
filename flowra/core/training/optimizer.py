"""
Optimizer Utilities

Helper functions for creating optimizers and schedulers.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Iterator
import math


def get_adapter_param_groups(
    adapters: Dict[str, nn.Module],
    base_lr: float = 1e-4,
    weight_decay: float = 0.01,
    no_decay_keywords: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Create parameter groups for adapter training.
    
    Separates parameters that should and shouldn't have weight decay.
    """
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "LayerNorm", "layer_norm", "norm"]
    
    decay_params = []
    no_decay_params = []
    
    for name, adapter in adapters.items():
        for param_name, param in adapter.named_parameters():
            if not param.requires_grad:
                continue
            
            if any(kw in param_name for kw in no_decay_keywords):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    return [
        {"params": decay_params, "weight_decay": weight_decay, "lr": base_lr},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": base_lr},
    ]


def create_optimizer(
    params: Iterator[nn.Parameter],
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer by type."""
    if optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_training_steps: int = 1000,
    num_warmup_steps: int = 0,
    **kwargs
) -> Any:
    """Create learning rate scheduler."""
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps, **kwargs)
    elif scheduler_type == "linear":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            return max(0.0, (num_training_steps - step) / max(1, num_training_steps - num_warmup_steps))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "cosine_warmup":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
