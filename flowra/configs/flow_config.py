"""
Main FLOWRA Configuration

The FlowConfig class is the primary configuration object that combines
all sub-configurations and provides the main interface for users.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import json


class AdapterType(Enum):
    """Types of adapters available in FLOWRA."""
    LORA = "lora"
    ATTENTION = "attention"
    NORMALIZATION = "normalization"
    AUTO = "auto"  # Automatically determined by flow analysis


class AllocationStrategy(Enum):
    """Rank allocation strategies."""
    SPECTRAL = "spectral"  # Flow-based spectral allocation
    UNIFORM = "uniform"    # Equal rank for all layers
    ADAPTIVE = "adaptive"  # Adaptive based on training dynamics


class RefinementStrategy(Enum):
    """Subspace refinement strategies during training."""
    NONE = "none"         # No refinement
    PROGRESSIVE = "progressive"  # Progressive pruning
    ELASTIC = "elastic"   # Elastic rank adjustment


class InitializationStrategy(Enum):
    """Adapter initialization strategies."""
    FLOW_AWARE = "flow_aware"  # Based on gradient covariance
    KAIMING = "kaiming"        # Standard Kaiming initialization
    SVD = "svd"                # SVD of pretrained weights
    ZERO = "zero"              # Zero initialization for B


@dataclass
class FlowConfig:
    """
    Main configuration class for FLOWRA framework.
    
    This is the primary interface for configuring all aspects of
    the FLOWRA adaptation process.
    
    Attributes
    ----------
    total_budget : float
        Fraction of base model parameters to use for adaptation.
        Default is 0.01 (1% of base parameters).
    
    interference_threshold : float
        Threshold η for multi-task composition safety.
        Pairs with interference > η will be orthogonalized.
        Default is 0.3.
    
    adapter_type : AdapterType
        Type of adapter to use. AUTO will determine based on flow analysis.
    
    allocation_strategy : AllocationStrategy
        How to allocate ranks across layers.
    
    refinement_strategy : RefinementStrategy
        How to refine subspaces during training.
    
    initialization_strategy : InitializationStrategy
        How to initialize adapter parameters.
    
    min_rank : int
        Minimum rank for any layer. Default is 1.
    
    max_rank : int
        Maximum rank for any layer. Default is 64.
    
    dropout : float
        Dropout probability for adapters. Default is 0.0.
    
    target_modules : Optional[List[str]]
        List of module name patterns to adapt. None means auto-detect.
    
    exclude_modules : Optional[List[str]]
        List of module name patterns to exclude from adaptation.
    
    use_rslora : bool
        Whether to use rank-stabilized LoRA scaling. Default is True.
    
    device : Optional[str]
        Device to use. None means auto-detect.
    
    seed : Optional[int]
        Random seed for reproducibility.
    
    verbose : bool
        Whether to print progress information.
    
    Examples
    --------
    >>> config = FlowConfig(total_budget=0.01, verbose=True)
    >>> framework = FLOWRA(model, config)
    
    >>> # Custom configuration
    >>> config = FlowConfig(
    ...     total_budget=0.02,
    ...     allocation_strategy=AllocationStrategy.SPECTRAL,
    ...     refinement_strategy=RefinementStrategy.PROGRESSIVE,
    ...     target_modules=["q_proj", "v_proj", "k_proj"],
    ...     min_rank=4,
    ...     max_rank=32,
    ... )
    """
    
    # Main parameters (only 2 required as per paper)
    total_budget: float = 0.01
    interference_threshold: float = 0.3
    
    # Strategy selections
    adapter_type: AdapterType = AdapterType.AUTO
    allocation_strategy: AllocationStrategy = AllocationStrategy.SPECTRAL
    refinement_strategy: RefinementStrategy = RefinementStrategy.NONE
    initialization_strategy: InitializationStrategy = InitializationStrategy.FLOW_AWARE
    
    # Rank constraints
    min_rank: int = 1
    max_rank: int = 64
    max_rank_ratio: float = 0.5  # Max fraction of min(m,n) for rank
    
    # Adapter settings
    dropout: float = 0.0
    use_rslora: bool = True  # Rank-stabilized scaling
    use_dora: bool = False   # Weight-decomposed adaptation
    
    # Module selection
    target_modules: Optional[List[str]] = None
    exclude_modules: Optional[List[str]] = None
    
    # Flow analysis settings
    num_fisher_samples: int = 100
    fisher_subsample_size: Optional[int] = None
    use_empirical_fisher: bool = True
    
    # Refinement settings
    refinement_interval: int = 100
    refinement_warmup: int = 100
    refinement_threshold: float = 0.1
    min_rank_fraction: float = 0.25
    
    # Composition settings
    composition_method: str = "orthogonal"
    energy_threshold: float = 0.95
    
    # System settings
    device: Optional[str] = None
    seed: Optional[int] = None
    verbose: bool = True
    
    # Advanced settings
    gradient_checkpointing: bool = False
    mixed_precision: bool = False
    compile_model: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        if not 0 < self.total_budget <= 1:
            raise ValueError(f"total_budget must be in (0, 1], got {self.total_budget}")
        
        if not 0 <= self.interference_threshold <= 1:
            raise ValueError(f"interference_threshold must be in [0, 1], got {self.interference_threshold}")
        
        if self.min_rank < 1:
            raise ValueError(f"min_rank must be >= 1, got {self.min_rank}")
        
        if self.max_rank < self.min_rank:
            raise ValueError(f"max_rank ({self.max_rank}) must be >= min_rank ({self.min_rank})")
        
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FlowConfig":
        """Create configuration from dictionary."""
        # Convert enum strings back to enums
        enum_fields = {
            "adapter_type": AdapterType,
            "allocation_strategy": AllocationStrategy,
            "refinement_strategy": RefinementStrategy,
            "initialization_strategy": InitializationStrategy,
        }
        
        processed = config_dict.copy()
        for field_name, enum_class in enum_fields.items():
            if field_name in processed and isinstance(processed[field_name], str):
                processed[field_name] = enum_class(processed[field_name])
        
        return cls(**processed)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "FlowConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def get_effective_budget(self, base_params: int) -> int:
        """Calculate effective parameter budget given base model size."""
        return int(base_params * self.total_budget)
    
    def __repr__(self) -> str:
        return (
            f"FlowConfig(\n"
            f"  total_budget={self.total_budget},\n"
            f"  interference_threshold={self.interference_threshold},\n"
            f"  allocation_strategy={self.allocation_strategy.value},\n"
            f"  refinement_strategy={self.refinement_strategy.value},\n"
            f"  min_rank={self.min_rank}, max_rank={self.max_rank}\n"
            f")"
        )


# Preset configurations
def get_default_config() -> FlowConfig:
    """Get default FLOWRA configuration."""
    return FlowConfig()


def get_efficient_config() -> FlowConfig:
    """Get configuration optimized for efficiency (smaller adapters)."""
    return FlowConfig(
        total_budget=0.005,
        max_rank=16,
        refinement_strategy=RefinementStrategy.PROGRESSIVE,
    )


def get_quality_config() -> FlowConfig:
    """Get configuration optimized for quality (larger adapters)."""
    return FlowConfig(
        total_budget=0.05,
        max_rank=128,
        initialization_strategy=InitializationStrategy.SVD,
    )


def get_multi_task_config() -> FlowConfig:
    """Get configuration optimized for multi-task composition."""
    return FlowConfig(
        total_budget=0.02,
        interference_threshold=0.2,
        composition_method="orthogonal",
    )
