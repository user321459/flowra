"""
Adapter Configuration

Configuration for individual adapter modules including per-layer settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class MergeStrategy(Enum):
    """How to merge adapter weights with base weights."""
    ADDITIVE = "additive"     # W + αBA
    MULTIPLICATIVE = "mult"   # W * (1 + αBA)
    GATE = "gate"             # W + g * αBA where g is learned


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA (Low-Rank Adaptation) modules.
    
    Attributes
    ----------
    rank : int
        Rank of the low-rank decomposition.
    
    alpha : float
        Scaling factor (actual scaling is alpha/rank).
    
    dropout : float
        Dropout probability applied to adapter output.
    
    fan_in_fan_out : bool
        Set True for Conv1D layers.
    
    merge_weights : bool
        Whether to merge weights during inference.
    
    init_lora_weights : str
        Initialization method: "default", "gaussian", "eva", "olora", "pissa"
    """
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    fan_in_fan_out: bool = False
    merge_weights: bool = False
    init_lora_weights: str = "default"
    use_rslora: bool = True  # Rank-stabilized scaling
    
    @property
    def scaling(self) -> float:
        """Compute effective scaling factor."""
        if self.use_rslora:
            return self.alpha / (self.rank ** 0.5)
        return self.alpha / self.rank


@dataclass
class DoRAConfig(LoRAConfig):
    """
    Configuration for DoRA (Weight-Decomposed Low-Rank Adaptation).
    
    DoRA decomposes weight updates into magnitude and direction components.
    """
    use_dora: bool = True
    magnitude_init: str = "ones"  # "ones", "norm", "learned"


@dataclass  
class AttentionAdapterConfig:
    """
    Configuration for attention-specific adapters.
    
    Supports separate configurations for Q, K, V projections.
    """
    total_rank: int = 16
    qk_ratio: float = 0.5  # Fraction of rank for Q/K
    
    # Per-projection settings
    adapt_query: bool = True
    adapt_key: bool = True
    adapt_value: bool = True
    adapt_output: bool = False
    
    # Scaling
    alpha: float = 32.0
    dropout: float = 0.0
    
    @property
    def qk_rank(self) -> int:
        return max(1, int(self.total_rank * self.qk_ratio))
    
    @property
    def v_rank(self) -> int:
        return max(1, self.total_rank - self.qk_rank)


@dataclass
class NormAdapterConfig:
    """
    Configuration for normalization layer adapters.
    """
    adapt_weight: bool = True
    adapt_bias: bool = True
    init_scale: float = 1.0
    init_bias: float = 0.0


@dataclass
class AdapterConfig:
    """
    Complete adapter configuration for a model.
    
    This class holds configurations for all adapter types and
    provides methods to generate per-layer configurations.
    
    Attributes
    ----------
    lora : LoRAConfig
        Default configuration for LoRA adapters.
    
    attention : AttentionAdapterConfig
        Configuration for attention-specific adapters.
    
    norm : NormAdapterConfig
        Configuration for normalization adapters.
    
    per_layer_config : Dict[str, Dict[str, Any]]
        Override configurations for specific layers.
    
    merge_strategy : MergeStrategy
        How to merge adapter with base weights.
    """
    
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dora: DoRAConfig = field(default_factory=DoRAConfig)
    attention: AttentionAdapterConfig = field(default_factory=AttentionAdapterConfig)
    norm: NormAdapterConfig = field(default_factory=NormAdapterConfig)
    
    per_layer_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    merge_strategy: MergeStrategy = MergeStrategy.ADDITIVE
    
    # Auto-detection settings
    attention_patterns: List[str] = field(default_factory=lambda: [
        "query", "key", "value", "q_proj", "k_proj", "v_proj",
        "qkv", "attention", "self_attn", "attn"
    ])
    norm_patterns: List[str] = field(default_factory=lambda: [
        "layernorm", "layer_norm", "ln_", "norm", "rmsnorm"
    ])
    
    def get_layer_config(
        self, 
        layer_name: str, 
        layer_type: str
    ) -> Dict[str, Any]:
        """
        Get configuration for a specific layer.
        
        Args:
            layer_name: Full name of the layer
            layer_type: Type of layer ("linear", "attention", "norm")
        
        Returns:
            Configuration dictionary for the layer
        """
        # Check for per-layer override
        if layer_name in self.per_layer_config:
            return self.per_layer_config[layer_name]
        
        # Return appropriate default config
        if layer_type == "attention":
            return {
                "total_rank": self.attention.total_rank,
                "qk_ratio": self.attention.qk_ratio,
                "alpha": self.attention.alpha,
                "dropout": self.attention.dropout,
            }
        elif layer_type == "norm":
            return {
                "adapt_weight": self.norm.adapt_weight,
                "adapt_bias": self.norm.adapt_bias,
            }
        else:  # Default to LoRA
            return {
                "rank": self.lora.rank,
                "alpha": self.lora.alpha,
                "dropout": self.lora.dropout,
                "use_rslora": self.lora.use_rslora,
            }
    
    def set_layer_config(self, layer_name: str, config: Dict[str, Any]):
        """Set configuration for a specific layer."""
        self.per_layer_config[layer_name] = config
    
    def is_attention_layer(self, layer_name: str) -> bool:
        """Check if layer name matches attention patterns."""
        name_lower = layer_name.lower()
        return any(pattern in name_lower for pattern in self.attention_patterns)
    
    def is_norm_layer(self, layer_name: str) -> bool:
        """Check if layer name matches normalization patterns."""
        name_lower = layer_name.lower()
        return any(pattern in name_lower for pattern in self.norm_patterns)
    
    def detect_layer_type(self, layer_name: str, module_type: str) -> str:
        """
        Detect the adapter type to use for a layer.
        
        Args:
            layer_name: Name of the layer
            module_type: Type of the module (e.g., "Linear", "LayerNorm")
        
        Returns:
            Adapter type: "lora", "attention", or "norm"
        """
        if self.is_norm_layer(layer_name) or "Norm" in module_type:
            return "norm"
        elif self.is_attention_layer(layer_name):
            return "attention"
        else:
            return "lora"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lora": self.lora.__dict__,
            "attention": self.attention.__dict__,
            "norm": self.norm.__dict__,
            "per_layer_config": self.per_layer_config,
            "merge_strategy": self.merge_strategy.value,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdapterConfig":
        """Create from dictionary."""
        config = cls()
        if "lora" in d:
            config.lora = LoRAConfig(**d["lora"])
        if "attention" in d:
            config.attention = AttentionAdapterConfig(**d["attention"])
        if "norm" in d:
            config.norm = NormAdapterConfig(**d["norm"])
        if "per_layer_config" in d:
            config.per_layer_config = d["per_layer_config"]
        if "merge_strategy" in d:
            config.merge_strategy = MergeStrategy(d["merge_strategy"])
        return config


@dataclass
class LayerAdapterSpec:
    """
    Specification for adapting a single layer.
    
    This is the output of the allocation algorithm, specifying
    exactly how each layer should be adapted.
    """
    layer_name: str
    adapter_type: str  # "lora", "attention", "norm"
    rank: int
    scaling: float
    
    # Flow metrics (for reference)
    flow_sensitivity: float = 0.0
    flow_conductance: float = 0.0
    redundancy: float = 0.0
    
    # Layer info
    weight_shape: Tuple[int, ...] = ()
    param_count: int = 0
    
    def __repr__(self) -> str:
        return (
            f"LayerAdapterSpec({self.layer_name}, "
            f"type={self.adapter_type}, rank={self.rank}, "
            f"ψ={self.flow_sensitivity:.4f})"
        )
