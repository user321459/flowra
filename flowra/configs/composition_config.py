"""
Multi-Task Composition Configuration

Configuration for merging and composing multiple task-specific adapters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import numpy as np


class CompositionMethod(Enum):
    """Methods for composing multiple adapters."""
    LINEAR = "linear"                # Simple weighted average
    ORTHOGONAL = "orthogonal"        # FLOWRA orthogonal composition
    TASK_ARITHMETIC = "task_arithmetic"  # Task arithmetic
    TIES = "ties"                    # TIES merging
    DARE = "dare"                    # DARE (Drop And Rescale)
    SLERP = "slerp"                  # Spherical linear interpolation
    CUSTOM = "custom"                # User-defined


class ConflictResolution(Enum):
    """How to resolve conflicts between adapters."""
    ORTHOGONALIZE = "orthogonalize"  # Gram-Schmidt orthogonalization
    DROP = "drop"                    # Drop conflicting components
    RESCALE = "rescale"              # Rescale to reduce interference
    KEEP_FIRST = "keep_first"        # Keep first adapter, adjust others
    AVERAGE = "average"              # Average conflicting directions


class RankReduction(Enum):
    """How to reduce rank of merged adapter."""
    SVD = "svd"                      # SVD truncation
    ENERGY = "energy"                # Energy-based thresholding
    MAGNITUDE = "magnitude"          # Magnitude pruning
    NONE = "none"                    # No reduction


@dataclass
class InterferenceConfig:
    """
    Configuration for interference detection and handling.
    """
    # Threshold for detecting conflicts
    threshold: float = 0.3  # η from paper
    
    # Measurement
    metric: str = "cosine"  # "cosine", "euclidean", "correlation"
    per_layer: bool = True  # Compute per-layer vs global
    
    # Handling
    resolution: ConflictResolution = ConflictResolution.ORTHOGONALIZE
    
    # Orthogonalization
    angle_threshold: float = np.pi / 4  # Angles below this get orthogonalized
    gram_schmidt_iterations: int = 1
    
    # Reporting
    report_conflicts: bool = True
    visualize_matrix: bool = False


@dataclass
class MergingConfig:
    """
    Configuration for the actual merging process.
    """
    # Method
    method: CompositionMethod = CompositionMethod.ORTHOGONAL
    
    # Weights
    task_weights: Optional[List[float]] = None  # None = uniform
    normalize_weights: bool = True
    
    # TIES specific
    ties_density: float = 0.5
    ties_majority_sign: bool = True
    
    # DARE specific
    dare_rate: float = 0.9
    dare_rescale: bool = True
    
    # SLERP specific
    slerp_t: float = 0.5
    
    # Post-processing
    rank_reduction: RankReduction = RankReduction.SVD
    energy_threshold: float = 0.95
    max_output_rank: Optional[int] = None


@dataclass
class CompositionConfig:
    """
    Complete configuration for multi-task adapter composition.
    
    Attributes
    ----------
    interference : InterferenceConfig
        Configuration for interference detection and handling.
    
    merging : MergingConfig
        Configuration for the merging process.
    
    validate_composition : bool
        Whether to validate the composed adapter.
    
    compute_metrics : bool
        Whether to compute quality metrics after composition.
    
    Examples
    --------
    >>> config = CompositionConfig(
    ...     interference=InterferenceConfig(threshold=0.2),
    ...     merging=MergingConfig(
    ...         method=CompositionMethod.ORTHOGONAL,
    ...         task_weights=[0.6, 0.4],
    ...     ),
    ... )
    """
    
    # Sub-configurations
    interference: InterferenceConfig = field(default_factory=InterferenceConfig)
    merging: MergingConfig = field(default_factory=MergingConfig)
    
    # Validation
    validate_composition: bool = True
    max_interference_after: float = 0.5  # Fail if exceeded
    
    # Metrics
    compute_metrics: bool = True
    metrics_to_compute: List[str] = field(default_factory=lambda: [
        "interference", "effective_rank", "parameter_count"
    ])
    
    # Output
    output_format: str = "adapter"  # "adapter", "state_dict", "merged_model"
    save_intermediate: bool = False
    
    # Logging
    verbose: bool = True
    
    def set_task_weights(self, weights: List[float]):
        """Set task weights for composition."""
        self.merging.task_weights = weights
        if self.merging.normalize_weights:
            total = sum(weights)
            self.merging.task_weights = [w / total for w in weights]
    
    def get_task_weights(self, num_tasks: int) -> List[float]:
        """Get task weights, defaulting to uniform if not set."""
        if self.merging.task_weights is not None:
            return self.merging.task_weights
        return [1.0 / num_tasks] * num_tasks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interference": {
                "threshold": self.interference.threshold,
                "resolution": self.interference.resolution.value,
            },
            "merging": {
                "method": self.merging.method.value,
                "task_weights": self.merging.task_weights,
                "rank_reduction": self.merging.rank_reduction.value,
            },
            "validate_composition": self.validate_composition,
        }


@dataclass
class TaskInfo:
    """
    Information about a single task for composition.
    """
    name: str
    adapter_path: Optional[str] = None
    adapter_state: Optional[Dict[str, Any]] = None
    weight: float = 1.0
    
    # Task metadata
    task_type: Optional[str] = None  # "classification", "generation", etc.
    domain: Optional[str] = None
    language: Optional[str] = None
    
    # Performance metrics (for informed weighting)
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    
    def __repr__(self) -> str:
        return f"TaskInfo({self.name}, weight={self.weight})"


@dataclass
class CompositionResult:
    """
    Result of adapter composition.
    """
    # Composed adapter
    merged_state: Dict[str, Any]
    
    # Metrics
    interference_before: Optional[np.ndarray] = None
    interference_after: Optional[np.ndarray] = None
    effective_rank: Optional[float] = None
    total_parameters: int = 0
    
    # Task info
    task_names: List[str] = field(default_factory=list)
    task_weights: List[float] = field(default_factory=list)
    
    # Conflicts
    conflicting_pairs: List[Tuple[int, int]] = field(default_factory=list)
    num_conflicts_resolved: int = 0
    
    # Quality
    composition_quality: float = 1.0  # 1.0 = perfect, lower = more interference
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Composition Result",
            "=" * 40,
            f"Tasks: {', '.join(self.task_names)}",
            f"Weights: {self.task_weights}",
            f"Total parameters: {self.total_parameters:,}",
            f"Effective rank: {self.effective_rank:.2f}" if self.effective_rank else "",
            f"Conflicts resolved: {self.num_conflicts_resolved}",
            f"Quality score: {self.composition_quality:.3f}",
        ]
        return "\n".join(filter(None, lines))


# Preset configurations
def get_default_composition_config() -> CompositionConfig:
    """Get default composition configuration."""
    return CompositionConfig()


def get_strict_composition_config() -> CompositionConfig:
    """Get configuration with strict interference handling."""
    return CompositionConfig(
        interference=InterferenceConfig(
            threshold=0.2,
            resolution=ConflictResolution.ORTHOGONALIZE,
        ),
        merging=MergingConfig(
            rank_reduction=RankReduction.SVD,
            energy_threshold=0.99,
        ),
        max_interference_after=0.3,
    )


def get_fast_composition_config() -> CompositionConfig:
    """Get configuration for fast composition (less validation)."""
    return CompositionConfig(
        interference=InterferenceConfig(
            threshold=0.5,
            resolution=ConflictResolution.AVERAGE,
        ),
        merging=MergingConfig(
            rank_reduction=RankReduction.NONE,
        ),
        validate_composition=False,
        compute_metrics=False,
    )
