"""
Polymorphic Adapter Factory

Creates appropriate adapters based on layer type and flow profile.
Implements Polymorphic Adapter Assignment.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List

from flowra.core.adapters.base import BaseAdapter, AdapterFactory
from flowra.core.adapters.lora import LoRAAdapter, LoRALinear, LoRAConv2d, DoRAAdapter
from flowra.core.adapters.attention import AttentionAdapter, QKVAdapter
from flowra.core.adapters.normalization import (
    NormalizationAdapter,
    LayerNormAdapter,
    BatchNormAdapter
)
from flowra.core.analysis.flow_metrics import FlowProfile


class PolymorphicAdapterFactory(AdapterFactory):
    """
    Factory that creates adapters based on flow profiles.
    
    Implements:
    1. High ρ + linear/conv → Standard LoRA with full rank
    2. Attention mechanism → Split-rank attention adapter
    3. Low ψ + normalization → Affine adapter only
    4. Otherwise → Reduced-rank LoRA
    
    Args:
        tau_high: Redundancy threshold for full-rank LoRA (default: 0.5)
        tau_low: Sensitivity threshold for minimal adaptation (default: 0.1)
        use_dora: Whether to use DoRA instead of LoRA
        attention_patterns: Patterns to identify attention layers
    """
    
    def __init__(
        self,
        tau_high: float = 0.5,
        tau_low: float = 0.1,
        use_dora: bool = False,
        attention_patterns: List[str] = None,
        norm_patterns: List[str] = None
    ):
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.use_dora = use_dora
        
        self.attention_patterns = attention_patterns or [
            "query", "key", "value", "q_proj", "k_proj", "v_proj",
            "qkv", "attention", "self_attn", "attn"
        ]
        
        self.norm_patterns = norm_patterns or [
            "layernorm", "layer_norm", "ln_", "norm", "rmsnorm"
        ]
    
    def _detect_layer_type(self, layer: nn.Module, layer_name: str) -> str:
        """Detect adapter type based on layer and name."""
        name_lower = layer_name.lower()
        
        # Check normalization
        if isinstance(layer, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            return "norm"
        if any(p in name_lower for p in self.norm_patterns):
            return "norm"
        
        # Check attention
        if any(p in name_lower for p in self.attention_patterns):
            return "attention"
        
        # Check convolution
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return "conv"
        
        # Default to linear
        return "linear"
    
    def create_adapter(
        self,
        layer: nn.Module,
        layer_name: str,
        rank: int,
        flow_profile: Optional[FlowProfile] = None,
        scaling: float = 1.0,
        dropout: float = 0.0,
        init_B: Optional[torch.Tensor] = None,
        init_A: Optional[torch.Tensor] = None,
        **kwargs
    ) -> BaseAdapter:
        """
        Create the appropriate adapter based on layer type and flow profile.
        
        Args:
            layer: Original layer to adapt
            layer_name: Name of the layer
            rank: Allocated rank
            flow_profile: Flow metrics (optional, affects adapter type)
            scaling: Scaling factor
            dropout: Dropout probability
            init_B, init_A: Optional initialization tensors
            **kwargs: Additional configuration
        
        Returns:
            Configured adapter instance
        """
        layer_type = self._detect_layer_type(layer, layer_name)
        
        # Get flow metrics if available
        if flow_profile is not None:
            psi = flow_profile.psi
            gamma = flow_profile.gamma
            rho = flow_profile.rho
        else:
            psi = 0.5
            gamma = 0.5
            rho = 0.5
        
        # Algorithm 2 decision tree
        if layer_type == "attention":
            # Type 2: Split-rank attention adapter
            qk_ratio = 0.5 + 0.3 * (gamma - 0.5)
            qk_ratio = max(0.2, min(0.8, qk_ratio))
            
            return AttentionAdapter(
                layer,
                total_rank=rank,
                qk_ratio=qk_ratio,
                scaling=scaling,
                dropout=dropout
            )
        
        elif layer_type == "norm":
            if psi < self.tau_low:
                # Type 3: Minimal affine adaptation
                if isinstance(layer, nn.LayerNorm):
                    return LayerNormAdapter(layer, scaling=scaling)
                elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    return BatchNormAdapter(layer, scaling=scaling)
                else:
                    return NormalizationAdapter(layer, scaling=scaling)
            else:
                # Still norm but with higher sensitivity - adapt both
                return NormalizationAdapter(
                    layer, 
                    scaling=scaling,
                    adapt_weight=True,
                    adapt_bias=True
                )
        
        elif layer_type == "conv":
            if isinstance(layer, nn.Conv2d):
                return LoRAConv2d(
                    layer,
                    rank=rank,
                    alpha=scaling * rank,
                    dropout=dropout
                )
            else:
                # Fallback for Conv1d/Conv3d
                return LoRAAdapter(
                    layer,
                    rank=rank,
                    scaling=scaling,
                    dropout=dropout
                )
        
        else:  # linear
            if rho > self.tau_high:
                # Type 1: Full-rank LoRA
                actual_rank = rank
            else:
                # Reduced rank
                actual_rank = max(1, rank // 2)
            
            if self.use_dora:
                return DoRAAdapter(
                    layer,
                    rank=actual_rank,
                    scaling=scaling,
                    dropout=dropout
                )
            else:
                return LoRALinear(
                    layer,
                    rank=actual_rank,
                    alpha=scaling * actual_rank,
                    dropout=dropout
                )
    
    def get_adapter_type(
        self,
        layer: nn.Module,
        layer_name: str
    ) -> str:
        """Determine adapter type for a layer."""
        return self._detect_layer_type(layer, layer_name)
    
    def get_adapter_info(
        self,
        layer: nn.Module,
        layer_name: str,
        flow_profile: Optional[FlowProfile] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get adapter type and configuration without creating it.
        
        Useful for planning and logging.
        
        Returns:
            Tuple of (adapter_class_name, config_dict)
        """
        layer_type = self._detect_layer_type(layer, layer_name)
        
        if flow_profile is not None:
            psi, gamma, rho = flow_profile.psi, flow_profile.gamma, flow_profile.rho
        else:
            psi, gamma, rho = 0.5, 0.5, 0.5
        
        if layer_type == "attention":
            qk_ratio = 0.5 + 0.3 * (gamma - 0.5)
            return ("AttentionAdapter", {"qk_ratio": qk_ratio})
        
        elif layer_type == "norm":
            if psi < self.tau_low:
                return ("NormalizationAdapter", {"minimal": True})
            return ("NormalizationAdapter", {"minimal": False})
        
        elif layer_type == "conv":
            return ("LoRAConv2d", {})
        
        else:
            if rho > self.tau_high:
                return ("LoRALinear", {"rank_factor": 1.0})
            return ("LoRALinear", {"rank_factor": 0.5})


def create_adapter_for_model(
    model: nn.Module,
    rank_allocation: Dict[str, int],
    flow_profiles: Dict[str, FlowProfile] = None,
    factory: PolymorphicAdapterFactory = None,
    **kwargs
) -> Dict[str, BaseAdapter]:
    """
    Create adapters for all specified layers in a model.
    
    Args:
        model: Model to adapt
        rank_allocation: Dict mapping layer names to ranks
        flow_profiles: Optional flow profiles for each layer
        factory: Adapter factory (default: PolymorphicAdapterFactory)
        **kwargs: Additional configuration for adapters
    
    Returns:
        Dict mapping layer names to adapter instances
    """
    if factory is None:
        factory = PolymorphicAdapterFactory()
    
    if flow_profiles is None:
        flow_profiles = {}
    
    adapters = {}
    
    for name, rank in rank_allocation.items():
        # Get the layer
        parts = name.split('.')
        layer = model
        for part in parts:
            layer = getattr(layer, part)
        
        # Get flow profile if available
        profile = flow_profiles.get(name)
        
        # Create adapter
        adapter = factory.create_adapter(
            layer=layer,
            layer_name=name,
            rank=rank,
            flow_profile=profile,
            **kwargs
        )
        
        adapters[name] = adapter
    
    return adapters
