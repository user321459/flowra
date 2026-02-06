"""
Base Adapter Classes

Abstract base class and factory interface for all adapter types.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple


class BaseAdapter(ABC, nn.Module):
    """
    Abstract base class for all FLOWRA adapters.
    
    Adapters wrap existing layers and add trainable low-rank parameters
    while keeping the original weights frozen.
    
    Attributes:
        original_layer: The wrapped layer
        rank: Rank of the low-rank decomposition
        scaling: Scaling factor for the adaptation
        merged: Whether adaptation is merged into weights
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        rank: int,
        scaling: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.scaling = scaling
        self.merged = False
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    @abstractmethod
    def get_adaptation(self) -> torch.Tensor:
        """Return the low-rank adaptation ΔW."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptation applied."""
        pass
    
    def merge_weights(self) -> None:
        """Merge adaptation into original weights for inference."""
        if self.merged:
            return
        
        with torch.no_grad():
            adaptation = self.get_adaptation()
            self.original_layer.weight.add_(self.scaling * adaptation)
        
        self.merged = True
    
    def unmerge_weights(self) -> None:
        """Unmerge adaptation from original weights."""
        if not self.merged:
            return
        
        with torch.no_grad():
            adaptation = self.get_adaptation()
            self.original_layer.weight.sub_(self.scaling * adaptation)
        
        self.merged = False
    
    def get_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_adapter_state(self) -> Dict[str, torch.Tensor]:
        """Get adapter-specific state dict."""
        return {
            k: v for k, v in self.state_dict().items()
            if 'original_layer' not in k
        }
    
    def load_adapter_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Load adapter-specific state dict."""
        current_state = self.state_dict()
        for k, v in state.items():
            if k in current_state:
                current_state[k] = v
        self.load_state_dict(current_state, strict=False)


class AdapterFactory(ABC):
    """
    Abstract factory for creating adapters.
    """
    
    @abstractmethod
    def create_adapter(
        self,
        layer: nn.Module,
        layer_name: str,
        rank: int,
        **kwargs
    ) -> BaseAdapter:
        """
        Create an adapter for the given layer.
        
        Args:
            layer: Original layer to adapt
            layer_name: Name of the layer
            rank: Rank for the adaptation
            **kwargs: Additional configuration
        
        Returns:
            Configured adapter instance
        """
        pass
    
    @abstractmethod
    def get_adapter_type(
        self,
        layer: nn.Module,
        layer_name: str
    ) -> str:
        """
        Determine the adapter type for a layer.
        
        Returns:
            Adapter type string: 'lora', 'attention', 'norm'
        """
        pass
