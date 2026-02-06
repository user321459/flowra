"""
Normalization Layer Adapters

Affine transformation adapters for LayerNorm and BatchNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from flowra.core.adapters.base import BaseAdapter


class NormalizationAdapter(BaseAdapter):
    """
    Affine Transformation Adapter for normalization layers.
    
    Implements: γ' = γ + δγ, β' = β + δβ
    
    For LayerNorm and BatchNorm, we learn additive corrections
    to the scale (γ) and shift (β) parameters.
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        rank: int = 0,  # Ignored for norm adapters
        scaling: float = 1.0,
        adapt_weight: bool = True,
        adapt_bias: bool = True
    ):
        super().__init__(original_layer, rank=0, scaling=scaling, dropout=0.0)
        
        # Get dimension
        if isinstance(original_layer, nn.LayerNorm):
            dim = original_layer.normalized_shape[0]
        elif isinstance(original_layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            dim = original_layer.num_features
        else:
            dim = original_layer.weight.shape[0]
        
        self.dim = dim
        self.adapt_weight = adapt_weight
        self.adapt_bias = adapt_bias
        
        # Affine corrections (initialized to zero)
        if adapt_weight:
            self.delta_gamma = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer('delta_gamma', torch.zeros(dim))
        
        if adapt_bias:
            self.delta_beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer('delta_beta', torch.zeros(dim))
    
    def get_adaptation(self) -> torch.Tensor:
        """Return concatenated delta parameters."""
        return torch.cat([self.delta_gamma, self.delta_beta])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.original_layer(x)
        
        # Get original parameters
        orig_weight = self.original_layer.weight
        orig_bias = (
            self.original_layer.bias 
            if self.original_layer.bias is not None 
            else torch.zeros_like(orig_weight)
        )
        
        # Apply corrections
        new_weight = orig_weight + self.scaling * self.delta_gamma
        new_bias = orig_bias + self.scaling * self.delta_beta
        
        # Forward with adapted parameters
        if isinstance(self.original_layer, nn.LayerNorm):
            return F.layer_norm(
                x,
                self.original_layer.normalized_shape,
                new_weight,
                new_bias,
                self.original_layer.eps
            )
        elif isinstance(self.original_layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
            return F.batch_norm(
                x,
                self.original_layer.running_mean,
                self.original_layer.running_var,
                new_weight,
                new_bias,
                self.original_layer.training,
                self.original_layer.momentum,
                self.original_layer.eps
            )
        else:
            # Fallback
            out = self.original_layer(x)
            return out * (1 + self.scaling * self.delta_gamma) + self.scaling * self.delta_beta
    
    def merge_weights(self):
        if self.merged:
            return
        
        with torch.no_grad():
            self.original_layer.weight.add_(self.scaling * self.delta_gamma)
            if self.original_layer.bias is not None:
                self.original_layer.bias.add_(self.scaling * self.delta_beta)
        
        self.merged = True
    
    def unmerge_weights(self):
        if not self.merged:
            return
        
        with torch.no_grad():
            self.original_layer.weight.sub_(self.scaling * self.delta_gamma)
            if self.original_layer.bias is not None:
                self.original_layer.bias.sub_(self.scaling * self.delta_beta)
        
        self.merged = False
    
    def get_trainable_params(self) -> int:
        count = 0
        if self.adapt_weight:
            count += self.dim
        if self.adapt_bias:
            count += self.dim
        return count


class LayerNormAdapter(NormalizationAdapter):
    """Adapter specifically for nn.LayerNorm."""
    
    def __init__(
        self,
        original_layer: nn.LayerNorm,
        scaling: float = 1.0,
        adapt_weight: bool = True,
        adapt_bias: bool = True
    ):
        if not isinstance(original_layer, nn.LayerNorm):
            raise TypeError("LayerNormAdapter requires nn.LayerNorm")
        
        super().__init__(
            original_layer,
            scaling=scaling,
            adapt_weight=adapt_weight,
            adapt_bias=adapt_bias
        )


class BatchNormAdapter(NormalizationAdapter):
    """Adapter specifically for BatchNorm layers."""
    
    def __init__(
        self,
        original_layer: nn.Module,
        scaling: float = 1.0,
        adapt_weight: bool = True,
        adapt_bias: bool = True,
        adapt_running_stats: bool = False
    ):
        if not isinstance(original_layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            raise TypeError("BatchNormAdapter requires BatchNorm layer")
        
        super().__init__(
            original_layer,
            scaling=scaling,
            adapt_weight=adapt_weight,
            adapt_bias=adapt_bias
        )
        
        self.adapt_running_stats = adapt_running_stats
        
        if adapt_running_stats:
            # Also adapt running mean and var
            self.delta_mean = nn.Parameter(
                torch.zeros_like(original_layer.running_mean)
            )
            self.delta_var = nn.Parameter(
                torch.zeros_like(original_layer.running_var)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.original_layer(x)
        
        orig_weight = self.original_layer.weight
        orig_bias = self.original_layer.bias or torch.zeros_like(orig_weight)
        
        new_weight = orig_weight + self.scaling * self.delta_gamma
        new_bias = orig_bias + self.scaling * self.delta_beta
        
        if self.adapt_running_stats:
            running_mean = self.original_layer.running_mean + self.delta_mean
            running_var = self.original_layer.running_var + self.delta_var.abs()
        else:
            running_mean = self.original_layer.running_mean
            running_var = self.original_layer.running_var
        
        return F.batch_norm(
            x,
            running_mean,
            running_var,
            new_weight,
            new_bias,
            self.original_layer.training,
            self.original_layer.momentum,
            self.original_layer.eps
        )


class RMSNormAdapter(NormalizationAdapter):
    """
    Adapter for RMSNorm (used in LLaMA, etc.).
    
    RMSNorm: y = x / RMS(x) * γ
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        scaling: float = 1.0
    ):
        # RMSNorm only has weight, no bias
        super().__init__(
            original_layer,
            scaling=scaling,
            adapt_weight=True,
            adapt_bias=False
        )
        
        # Override delta_beta to be a buffer
        del self.delta_beta
        self.register_buffer('delta_beta', torch.zeros(self.dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.original_layer(x)
        
        # Apply RMSNorm manually with adapted weight
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.original_layer.eps)
        
        new_weight = self.original_layer.weight + self.scaling * self.delta_gamma
        
        return x_normed * new_weight
