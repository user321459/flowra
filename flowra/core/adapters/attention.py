"""
Attention-Specific Adapters

Structured adaptation for attention mechanisms with separate
handling of Q, K, V projections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

from flowra.core.adapters.base import BaseAdapter


class AttentionAdapter(BaseAdapter):
    """
    Structured Attention Adapter with Q/K/V rank split.
    
    Implements: W' = W + scaling * (B₁A₁ + B₂A₂)
    
    The rank is split between query/key (B₁A₁) and value (B₂A₂)
    based on attention-specific flow analysis.
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        total_rank: int,
        qk_ratio: float = 0.5,
        scaling: float = 1.0,
        dropout: float = 0.0
    ):
        rank_qk = max(1, int(total_rank * qk_ratio))
        rank_v = max(1, total_rank - rank_qk)
        
        super().__init__(original_layer, total_rank, scaling, dropout)
        
        self.rank_qk = rank_qk
        self.rank_v = rank_v
        self.qk_ratio = qk_ratio
        
        if not isinstance(original_layer, nn.Linear):
            raise ValueError("AttentionAdapter requires Linear layer")
        
        self.out_features = original_layer.out_features
        self.in_features = original_layer.in_features
        
        # QK adaptation path
        self.B1 = nn.Parameter(torch.zeros(self.out_features, rank_qk))
        self.A1 = nn.Parameter(torch.zeros(rank_qk, self.in_features))
        nn.init.kaiming_uniform_(self.A1, a=math.sqrt(5))
        
        # V adaptation path
        self.B2 = nn.Parameter(torch.zeros(self.out_features, rank_v))
        self.A2 = nn.Parameter(torch.zeros(rank_v, self.in_features))
        nn.init.kaiming_uniform_(self.A2, a=math.sqrt(5))
    
    def get_adaptation(self) -> torch.Tensor:
        """Compute combined adaptation B₁A₁ + B₂A₂."""
        return self.B1 @ self.A1 + self.B2 @ self.A2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_layer(x)
        
        if self.merged:
            return original_out
        
        x_dropped = self.dropout(x)
        
        # Dual adaptation paths
        lora_qk = F.linear(F.linear(x_dropped, self.A1), self.B1)
        lora_v = F.linear(F.linear(x_dropped, self.A2), self.B2)
        
        return original_out + self.scaling * (lora_qk + lora_v)
    
    def get_qk_adaptation(self) -> torch.Tensor:
        """Get Q/K specific adaptation."""
        return self.B1 @ self.A1
    
    def get_v_adaptation(self) -> torch.Tensor:
        """Get V specific adaptation."""
        return self.B2 @ self.A2


class MultiHeadAttentionAdapter(nn.Module):
    """
    Adapter for complete multi-head attention modules.
    
    Wraps Q, K, V, O projections with separate adapters
    and provides unified interface.
    """
    
    def __init__(
        self,
        attention_module: nn.Module,
        rank: int = 8,
        scaling: float = 1.0,
        adapt_query: bool = True,
        adapt_key: bool = True,
        adapt_value: bool = True,
        adapt_output: bool = False,
        qk_ratio: float = 0.5
    ):
        super().__init__()
        
        self.attention = attention_module
        self.rank = rank
        self.scaling = scaling
        
        self.adapters: Dict[str, AttentionAdapter] = nn.ModuleDict()
        
        # Find and wrap projection layers
        for name, module in attention_module.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            
            name_lower = name.lower()
            
            if adapt_query and any(p in name_lower for p in ['query', 'q_proj', 'q']):
                self.adapters[name] = AttentionAdapter(
                    module, rank, qk_ratio=1.0, scaling=scaling
                )
            elif adapt_key and any(p in name_lower for p in ['key', 'k_proj', 'k']):
                self.adapters[name] = AttentionAdapter(
                    module, rank, qk_ratio=1.0, scaling=scaling
                )
            elif adapt_value and any(p in name_lower for p in ['value', 'v_proj', 'v']):
                self.adapters[name] = AttentionAdapter(
                    module, rank, qk_ratio=0.0, scaling=scaling  # All V
                )
            elif adapt_output and any(p in name_lower for p in ['out', 'o_proj', 'output']):
                self.adapters[name] = AttentionAdapter(
                    module, rank, qk_ratio=0.5, scaling=scaling
                )
    
    def forward(self, *args, **kwargs):
        """Forward through attention with adapters."""
        # Replace modules with adapted versions temporarily
        original_modules = {}
        
        for name, adapter in self.adapters.items():
            parts = name.split('.')
            parent = self.attention
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            attr = parts[-1]
            original_modules[name] = getattr(parent, attr)
            setattr(parent, attr, adapter)
        
        # Forward
        output = self.attention(*args, **kwargs)
        
        # Restore original modules
        for name, original in original_modules.items():
            parts = name.split('.')
            parent = self.attention
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], original)
        
        return output
    
    def get_trainable_params(self) -> int:
        return sum(a.get_trainable_params() for a in self.adapters.values())
    
    def merge_weights(self):
        for adapter in self.adapters.values():
            adapter.merge_weights()
    
    def unmerge_weights(self):
        for adapter in self.adapters.values():
            adapter.unmerge_weights()


class QKVAdapter(BaseAdapter):
    """
    Combined Q/K/V adapter for fused QKV projections.
    
    Some models use a single linear layer for all of Q, K, V.
    This adapter handles that case with separate sub-adaptations.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int,
        num_heads: int,
        head_dim: int,
        scaling: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__(original_layer, rank, scaling, dropout)
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim
        
        # Separate ranks for Q, K, V
        rank_per = max(1, rank // 3)
        
        # Q adapter
        self.B_q = nn.Parameter(torch.zeros(self.hidden_size, rank_per))
        self.A_q = nn.Parameter(torch.zeros(rank_per, original_layer.in_features))
        nn.init.kaiming_uniform_(self.A_q, a=math.sqrt(5))
        
        # K adapter
        self.B_k = nn.Parameter(torch.zeros(self.hidden_size, rank_per))
        self.A_k = nn.Parameter(torch.zeros(rank_per, original_layer.in_features))
        nn.init.kaiming_uniform_(self.A_k, a=math.sqrt(5))
        
        # V adapter
        self.B_v = nn.Parameter(torch.zeros(self.hidden_size, rank_per))
        self.A_v = nn.Parameter(torch.zeros(rank_per, original_layer.in_features))
        nn.init.kaiming_uniform_(self.A_v, a=math.sqrt(5))
    
    def get_adaptation(self) -> torch.Tensor:
        """Get combined QKV adaptation."""
        adapt_q = self.B_q @ self.A_q
        adapt_k = self.B_k @ self.A_k
        adapt_v = self.B_v @ self.A_v
        
        return torch.cat([adapt_q, adapt_k, adapt_v], dim=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_layer(x)
        
        if self.merged:
            return original_out
        
        x_dropped = self.dropout(x)
        
        # Separate QKV adaptations
        q_adapt = F.linear(F.linear(x_dropped, self.A_q), self.B_q)
        k_adapt = F.linear(F.linear(x_dropped, self.A_k), self.B_k)
        v_adapt = F.linear(F.linear(x_dropped, self.A_v), self.B_v)
        
        # Combine
        lora_out = torch.cat([q_adapt, k_adapt, v_adapt], dim=-1)
        
        return original_out + self.scaling * lora_out
