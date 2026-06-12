"""
LoRA Adapter Implementations

Standard Low-Rank Adaptation for linear and convolutional layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from flowra.core.adapters.base import BaseAdapter


class LoRAAdapter(BaseAdapter):
    """
    Standard Low-Rank Adaptation.
    
    Implements: W' = W + scaling * BA
    
    where:
    - W: Original frozen weight [out_features, in_features]
    - B: Learnable down-projection [out_features, rank]
    - A: Learnable up-projection [rank, in_features]
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        rank: int,
        scaling: float = 1.0,
        dropout: float = 0.0,
        use_rslora: bool = True,
        init_B: Optional[torch.Tensor] = None,
        init_A: Optional[torch.Tensor] = None
    ):
        super().__init__(original_layer, rank, scaling, dropout)
        
        # Get dimensions
        if isinstance(original_layer, nn.Linear):
            self.out_features = original_layer.out_features
            self.in_features = original_layer.in_features
            self.layer_type = 'linear'
        elif isinstance(original_layer, (nn.Conv1d, nn.Conv2d)):
            self.out_features = original_layer.out_channels
            self.in_features = (
                original_layer.in_channels * 
                original_layer.kernel_size[0] * 
                (original_layer.kernel_size[1] if len(original_layer.kernel_size) > 1 else 1)
            )
            self.layer_type = 'conv'
            self.kernel_size = original_layer.kernel_size
        else:
            raise ValueError(f"Unsupported layer type: {type(original_layer)}")
        
        # Rank-stabilized scaling
        if use_rslora:
            self.scaling = scaling / math.sqrt(rank)
        
        # Create LoRA matrices
        if init_B is not None:
            self.B = nn.Parameter(init_B)
        else:
            self.B = nn.Parameter(torch.zeros(self.out_features, rank))
        
        if init_A is not None:
            self.A = nn.Parameter(init_A)
        else:
            self.A = nn.Parameter(torch.zeros(rank, self.in_features))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    
    def get_adaptation(self) -> torch.Tensor:
        """Compute BA adaptation matrix."""
        return self.B @ self.A
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: y = Wx + scaling * B(Ax)"""
        original_out = self.original_layer(x)
        
        if self.merged:
            return original_out
        
        # LoRA path
        x_dropped = self.dropout(x)
        
        if self.layer_type == 'linear':
            lora_out = F.linear(F.linear(x_dropped, self.A), self.B)
        else:
            # For conv: reshape and apply
            x_flat = x_dropped.view(x_dropped.size(0), -1)
            lora_flat = F.linear(F.linear(x_flat, self.A), self.B)
            lora_out = lora_flat.view(original_out.shape)
        
        return original_out + self.scaling * lora_out


class LoRALinear(LoRAAdapter):
    """LoRA specifically for nn.Linear layers."""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_rslora: bool = True,
        fan_in_fan_out: bool = False
    ):
        if not isinstance(original_layer, nn.Linear):
            raise TypeError("LoRALinear requires nn.Linear layer")
        
        scaling = alpha / rank if not use_rslora else alpha
        super().__init__(original_layer, rank, scaling, dropout, use_rslora)
        
        self.alpha = alpha
        self.fan_in_fan_out = fan_in_fan_out
        
        if fan_in_fan_out:
            # Transpose A and B for GPT-2 style Conv1D
            self.A = nn.Parameter(self.A.t().contiguous())
            self.B = nn.Parameter(self.B.t().contiguous())
    
    def get_adaptation(self) -> torch.Tensor:
        if self.fan_in_fan_out:
            return (self.B.t() @ self.A.t()).t()
        return self.B @ self.A
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_layer(x)
        
        if self.merged:
            return original_out
        
        x_dropped = self.dropout(x)
        
        if self.fan_in_fan_out:
            lora_out = F.linear(F.linear(x_dropped, self.A.t()), self.B.t())
        else:
            lora_out = F.linear(F.linear(x_dropped, self.A), self.B)
        
        return original_out + self.scaling * lora_out


class LoRAConv2d(BaseAdapter):
    """LoRA for nn.Conv2d layers."""
    
    def __init__(
        self,
        original_layer: nn.Conv2d,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
        use_rslora: bool = True
    ):
        if not isinstance(original_layer, nn.Conv2d):
            raise TypeError("LoRAConv2d requires nn.Conv2d layer")
        
        scaling = alpha / rank if not use_rslora else alpha / math.sqrt(rank)
        super().__init__(original_layer, rank, scaling, dropout)
        
        self.in_channels = original_layer.in_channels
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups
        
        # LoRA as 1x1 convolutions
        self.lora_A = nn.Conv2d(
            self.in_channels, rank, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.lora_B = nn.Conv2d(
            rank, self.out_channels, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def get_adaptation(self) -> torch.Tensor:
        """Get equivalent weight adaptation."""
        # This is approximate - full conv adaptation is more complex
        A_weight = self.lora_A.weight.view(self.rank, self.in_channels)
        B_weight = self.lora_B.weight.view(self.out_channels, self.rank)
        return B_weight @ A_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_layer(x)
        
        if self.merged:
            return original_out
        
        x_dropped = self.dropout(x)
        lora_out = self.lora_B(self.lora_A(x_dropped))
        
        # Handle stride mismatch
        if lora_out.shape != original_out.shape:
            lora_out = F.interpolate(
                lora_out, 
                size=original_out.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        return original_out + self.scaling * lora_out


class DoRAAdapter(LoRAAdapter):
    """
    Weight-Decomposed Low-Rank Adaptation (DoRA).
    
    Decomposes the weight update into magnitude and direction:
    W' = m * (W + ΔW) / ||W + ΔW||
    
    where m is a learned magnitude vector.
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        rank: int,
        scaling: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__(original_layer, rank, scaling, dropout, use_rslora=False)
        
        # Magnitude vector
        with torch.no_grad():
            weight_norm = self.original_layer.weight.norm(dim=1, keepdim=True)
        self.magnitude = nn.Parameter(weight_norm.squeeze())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.original_layer(x)
        
        # Compute adapted weight
        weight = self.original_layer.weight + self.scaling * (self.B @ self.A)
        
        # Normalize and apply magnitude
        weight_norm = weight.norm(dim=1, keepdim=True)
        weight_normalized = weight / (weight_norm + 1e-8)
        weight_final = self.magnitude.unsqueeze(1) * weight_normalized
        
        # Apply
        x_dropped = self.dropout(x)
        return F.linear(x_dropped, weight_final, self.original_layer.bias)
