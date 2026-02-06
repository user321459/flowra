"""
Flow Metrics Data Structures

Defines the data structures for storing and manipulating flow analysis results.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import numpy as np


@dataclass
class FlowProfile:
    """
    Complete flow profile for a layer.
    
    Contains all metrics needed for adaptation decisions based on
    the unified information flow analysis framework.
    
    Attributes
    ----------
    psi : float
        Flow Sensitivity Index (Eq. 2) - captures geometric and 
        information-theoretic relevance of the layer.
        Higher ψ indicates more task-relevant information flows through the layer.
    
    gamma : float
        Flow Conductance - measures directional heterogeneity.
        Higher γ indicates the layer needs higher rank to capture
        its directional diversity.
    
    rho : float
        Redundancy Coefficient - from singular value analysis.
        Higher ρ indicates the layer is more amenable to low-rank approximation.
    
    Examples
    --------
    >>> profile = FlowProfile(psi=0.15, gamma=0.3, rho=0.8)
    >>> profile.importance_score
    0.1575
    """
    psi: float = 0.0      # Flow sensitivity index
    gamma: float = 0.0    # Flow conductance  
    rho: float = 0.0      # Redundancy coefficient
    
    # Optional detailed metrics
    fisher_trace: Optional[float] = None
    jacobian_norm: Optional[float] = None
    effective_rank: Optional[float] = None
    
    # Spectral information
    top_eigenvalues: Optional[torch.Tensor] = None
    eigenvalue_ratio: Optional[float] = None  # λ_max / λ_min
    
    @property
    def importance_score(self) -> float:
        """
        Combined importance score for ranking layers.
        
        Combines sensitivity and conductance with redundancy as a modifier.
        Layers with high importance should get more adaptation capacity.
        """
        base_score = self.psi * (1 + self.gamma)
        redundancy_modifier = 0.5 + 0.5 * self.rho  # Range: [0.5, 1.0]
        return base_score * redundancy_modifier
    
    def to_tensor(self) -> torch.Tensor:
        """Convert primary metrics to tensor [ψ, γ, ρ]."""
        return torch.tensor([self.psi, self.gamma, self.rho])
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        result = {
            "psi": self.psi,
            "gamma": self.gamma,
            "rho": self.rho,
        }
        if self.fisher_trace is not None:
            result["fisher_trace"] = self.fisher_trace
        if self.jacobian_norm is not None:
            result["jacobian_norm"] = self.jacobian_norm
        if self.effective_rank is not None:
            result["effective_rank"] = self.effective_rank
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlowProfile":
        """Create from dictionary."""
        return cls(
            psi=d.get("psi", 0.0),
            gamma=d.get("gamma", 0.0),
            rho=d.get("rho", 0.0),
            fisher_trace=d.get("fisher_trace"),
            jacobian_norm=d.get("jacobian_norm"),
            effective_rank=d.get("effective_rank"),
        )
    
    def __repr__(self) -> str:
        return f"FlowProfile(ψ={self.psi:.4f}, γ={self.gamma:.4f}, ρ={self.rho:.4f})"


@dataclass
class LayerFlowInfo:
    """
    Extended information about a layer's flow characteristics.
    
    Contains both the flow profile and additional statistics needed
    for rank allocation and adapter assignment.
    
    Attributes
    ----------
    name : str
        Full name of the layer in the model.
    
    flow_profile : FlowProfile
        Flow metrics for this layer.
    
    layer_type : str
        Type of layer: 'linear', 'conv', 'attention', 'norm', 'embedding'.
    
    weight_shape : Tuple[int, ...]
        Shape of the weight tensor.
    
    fisher_block : Optional[torch.Tensor]
        Block-diagonal Fisher Information component.
    
    Examples
    --------
    >>> info = LayerFlowInfo(
    ...     name="model.layer.0.attention.query",
    ...     flow_profile=FlowProfile(psi=0.15, gamma=0.3, rho=0.8),
    ...     layer_type="attention",
    ...     weight_shape=(768, 768),
    ... )
    """
    name: str
    flow_profile: FlowProfile
    layer_type: str
    weight_shape: Tuple[int, ...]
    
    # Statistics
    fisher_block: Optional[torch.Tensor] = None
    jacobian_contribution: Optional[torch.Tensor] = None
    gradient_covariance: Optional[torch.Tensor] = None
    
    # Singular value information
    singular_values: Optional[torch.Tensor] = None
    left_singular_vectors: Optional[torch.Tensor] = None
    right_singular_vectors: Optional[torch.Tensor] = None
    
    # Metadata
    num_parameters: int = 0
    is_attention_component: bool = False
    attention_type: Optional[str] = None  # 'query', 'key', 'value', 'output'
    
    def __post_init__(self):
        """Compute derived attributes."""
        if self.num_parameters == 0:
            self.num_parameters = int(np.prod(self.weight_shape))
    
    @property
    def out_features(self) -> int:
        """Get output dimension."""
        return self.weight_shape[0]
    
    @property
    def in_features(self) -> int:
        """Get input dimension (flattened for conv)."""
        return int(np.prod(self.weight_shape[1:]))
    
    @property
    def max_rank(self) -> int:
        """Maximum possible rank for this layer."""
        return min(self.out_features, self.in_features)
    
    def get_rank_params(self, rank: int) -> int:
        """Calculate parameters needed for given rank."""
        return rank * (self.out_features + self.in_features)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without large tensors)."""
        return {
            "name": self.name,
            "flow_profile": self.flow_profile.to_dict(),
            "layer_type": self.layer_type,
            "weight_shape": list(self.weight_shape),
            "num_parameters": self.num_parameters,
            "is_attention_component": self.is_attention_component,
            "attention_type": self.attention_type,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LayerFlowInfo":
        """Create from dictionary."""
        return cls(
            name=d["name"],
            flow_profile=FlowProfile.from_dict(d["flow_profile"]),
            layer_type=d["layer_type"],
            weight_shape=tuple(d["weight_shape"]),
            num_parameters=d.get("num_parameters", 0),
            is_attention_component=d.get("is_attention_component", False),
            attention_type=d.get("attention_type"),
        )
    
    def __repr__(self) -> str:
        return (
            f"LayerFlowInfo({self.name}, type={self.layer_type}, "
            f"shape={self.weight_shape}, {self.flow_profile})"
        )


@dataclass
class ModelFlowAnalysis:
    """
    Complete flow analysis results for an entire model.
    """
    # Per-layer information
    layer_info: Dict[str, LayerFlowInfo] = field(default_factory=dict)
    
    # Global statistics
    total_parameters: int = 0
    total_fisher_trace: float = 0.0
    total_jacobian_norm: float = 0.0
    
    # Sorted indices for various criteria
    sensitivity_ranking: List[str] = field(default_factory=list)
    conductance_ranking: List[str] = field(default_factory=list)
    importance_ranking: List[str] = field(default_factory=list)
    
    # Model metadata
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    
    def __post_init__(self):
        """Compute derived attributes."""
        if self.layer_info and not self.sensitivity_ranking:
            self._compute_rankings()
    
    def _compute_rankings(self):
        """Compute sorted rankings for different criteria."""
        layers = list(self.layer_info.items())
        
        # Sort by sensitivity (ψ)
        self.sensitivity_ranking = [
            name for name, _ in sorted(
                layers, 
                key=lambda x: x[1].flow_profile.psi, 
                reverse=True
            )
        ]
        
        # Sort by conductance (γ)
        self.conductance_ranking = [
            name for name, _ in sorted(
                layers,
                key=lambda x: x[1].flow_profile.gamma,
                reverse=True
            )
        ]
        
        # Sort by combined importance
        self.importance_ranking = [
            name for name, _ in sorted(
                layers,
                key=lambda x: x[1].flow_profile.importance_score,
                reverse=True
            )
        ]
    
    def get_top_layers(self, n: int, criterion: str = "importance") -> List[str]:
        """Get top N layers by specified criterion."""
        if criterion == "sensitivity":
            return self.sensitivity_ranking[:n]
        elif criterion == "conductance":
            return self.conductance_ranking[:n]
        else:
            return self.importance_ranking[:n]
    
    def get_layers_by_type(self, layer_type: str) -> Dict[str, LayerFlowInfo]:
        """Get all layers of a specific type."""
        return {
            name: info for name, info in self.layer_info.items()
            if info.layer_type == layer_type
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Model Flow Analysis Summary",
            "=" * 50,
            f"Total layers analyzed: {len(self.layer_info)}",
            f"Total parameters: {self.total_parameters:,}",
            "",
            "Top 5 layers by importance:",
        ]
        
        for name in self.importance_ranking[:5]:
            info = self.layer_info[name]
            lines.append(
                f"  {name}: ψ={info.flow_profile.psi:.4f}, "
                f"γ={info.flow_profile.gamma:.4f}"
            )
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "layer_info": {
                name: info.to_dict() 
                for name, info in self.layer_info.items()
            },
            "total_parameters": self.total_parameters,
            "sensitivity_ranking": self.sensitivity_ranking,
            "conductance_ranking": self.conductance_ranking,
            "importance_ranking": self.importance_ranking,
            "model_name": self.model_name,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelFlowAnalysis":
        """Create from dictionary."""
        analysis = cls(
            total_parameters=d.get("total_parameters", 0),
            sensitivity_ranking=d.get("sensitivity_ranking", []),
            conductance_ranking=d.get("conductance_ranking", []),
            importance_ranking=d.get("importance_ranking", []),
            model_name=d.get("model_name"),
        )
        
        if "layer_info" in d:
            analysis.layer_info = {
                name: LayerFlowInfo.from_dict(info)
                for name, info in d["layer_info"].items()
            }
        
        return analysis
