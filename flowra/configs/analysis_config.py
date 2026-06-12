"""
Flow Analysis Configuration

Configuration for Fisher Information estimation and flow metric computation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum


class FisherEstimationType(Enum):
    """Methods for estimating Fisher Information."""
    EMPIRICAL = "empirical"      # Standard empirical Fisher
    EXACT = "exact"              # Exact Fisher (requires 2nd order)
    KRONECKER = "kronecker"      # KFAC-style Kronecker approximation
    DIAGONAL = "diagonal"        # Diagonal approximation
    BLOCKWISE = "blockwise"      # Block-diagonal approximation


class JacobianEstimationType(Enum):
    """Methods for estimating layer Jacobians."""
    POWER_ITERATION = "power_iteration"  # Spectral norm via power iteration
    SVD = "svd"                          # Full SVD (expensive)
    HUTCHINSON = "hutchinson"            # Hutchinson trace estimator
    ANALYTICAL = "analytical"            # Layer-specific analytical bounds


class ImportanceMetric(Enum):
    """Metrics for layer importance."""
    FLOW_SENSITIVITY = "flow_sensitivity"  # ψ from paper (Eq. 2)
    FLOW_CONDUCTANCE = "flow_conductance"  # γ from paper
    FISHER_TRACE = "fisher_trace"          # Tr(F)
    GRADIENT_NORM = "gradient_norm"        # ||∇L||
    ACTIVATION_NORM = "activation_norm"    # ||h||
    WANDA = "wanda"                        # Weights and activations
    CUSTOM = "custom"                      # User-defined


@dataclass
class FisherConfig:
    """
    Configuration for Fisher Information estimation.
    """
    estimation_type: FisherEstimationType = FisherEstimationType.DIAGONAL
    
    # Sampling
    num_samples: int = 100
    batch_size: int = 32
    
    # Approximation settings
    use_empirical: bool = True  # Use gradient outer products
    damping: float = 1e-4       # Regularization for inversion
    
    # KFAC specific
    kfac_update_freq: int = 10
    kfac_cov_ema_decay: float = 0.95
    
    # Memory optimization
    use_gradient_checkpointing: bool = False
    chunk_size: Optional[int] = None  # Process layers in chunks
    
    # Numerical stability
    eps: float = 1e-10
    max_eigenvalue: float = 1e6


@dataclass
class JacobianConfig:
    """
    Configuration for Jacobian norm estimation.
    """
    estimation_type: JacobianEstimationType = JacobianEstimationType.POWER_ITERATION
    
    # Power iteration
    num_iterations: int = 10
    tolerance: float = 1e-4
    
    # Hutchinson
    num_hutchinson_samples: int = 50
    
    # Output
    compute_full_jacobian: bool = False
    store_jacobians: bool = False


@dataclass
class MetricConfig:
    """
    Configuration for flow metric computation.
    """
    primary_metric: ImportanceMetric = ImportanceMetric.FLOW_SENSITIVITY
    
    # Metric weights (for combined scoring)
    sensitivity_weight: float = 1.0
    conductance_weight: float = 1.0
    redundancy_weight: float = 0.5
    
    # Thresholds
    min_sensitivity: float = 1e-6
    max_conductance: float = 1.0
    
    # Normalization
    normalize_per_layer: bool = True
    use_softmax_normalization: bool = False
    temperature: float = 1.0


@dataclass
class AnalysisConfig:
    """
    Complete configuration for flow analysis.
    
    This configuration controls how the flow analyzer computes
    Fisher Information, Jacobian norms, and flow metrics.
    
    Attributes
    ----------
    fisher : FisherConfig
        Configuration for Fisher Information estimation.
    
    jacobian : JacobianConfig
        Configuration for Jacobian norm estimation.
    
    metrics : MetricConfig
        Configuration for flow metric computation.
    
    calibration_samples : int
        Number of samples to use for calibration.
    
    use_cache : bool
        Whether to cache computed statistics.
    
    analyze_attention_separately : bool
        Whether to compute separate metrics for Q/K/V.
    
    analyze_norms : bool
        Whether to analyze normalization layers.
    
    verbose : bool
        Whether to print progress information.
    
    Examples
    --------
    >>> config = AnalysisConfig(
    ...     calibration_samples=200,
    ...     fisher=FisherConfig(num_samples=150),
    ...     analyze_attention_separately=True,
    ... )
    """
    
    # Sub-configurations
    fisher: FisherConfig = field(default_factory=FisherConfig)
    jacobian: JacobianConfig = field(default_factory=JacobianConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    
    # Data settings
    calibration_samples: int = 100
    calibration_batch_size: int = 32
    
    # Analysis scope
    analyze_linear: bool = True
    analyze_conv: bool = True
    analyze_attention_separately: bool = True
    analyze_norms: bool = True
    analyze_embeddings: bool = False
    
    # Filtering
    min_layer_params: int = 1000  # Skip tiny layers
    max_layer_params: Optional[int] = None
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    
    # Caching
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    # Output
    compute_redundancy: bool = True
    compute_effective_rank: bool = True
    store_singular_values: bool = False
    
    # Logging
    verbose: bool = True
    log_interval: int = 10
    
    def should_analyze_layer(self, layer_name: str, layer_type: str, num_params: int) -> bool:
        """
        Determine if a layer should be analyzed.
        
        Args:
            layer_name: Name of the layer
            layer_type: Type of layer (e.g., "Linear", "Conv2d")
            num_params: Number of parameters
        
        Returns:
            True if the layer should be analyzed
        """
        # Check parameter count
        if num_params < self.min_layer_params:
            return False
        if self.max_layer_params and num_params > self.max_layer_params:
            return False
        
        # Check include patterns
        if self.include_patterns:
            if not any(p in layer_name for p in self.include_patterns):
                return False
        
        # Check exclude patterns
        if self.exclude_patterns:
            if any(p in layer_name for p in self.exclude_patterns):
                return False
        
        # Check layer type
        if "Linear" in layer_type and not self.analyze_linear:
            return False
        if "Conv" in layer_type and not self.analyze_conv:
            return False
        if "Norm" in layer_type and not self.analyze_norms:
            return False
        if "Embedding" in layer_type and not self.analyze_embeddings:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fisher": {
                "estimation_type": self.fisher.estimation_type.value,
                "num_samples": self.fisher.num_samples,
                "damping": self.fisher.damping,
            },
            "jacobian": {
                "estimation_type": self.jacobian.estimation_type.value,
                "num_iterations": self.jacobian.num_iterations,
            },
            "metrics": {
                "primary_metric": self.metrics.primary_metric.value,
            },
            "calibration_samples": self.calibration_samples,
            "analyze_attention_separately": self.analyze_attention_separately,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisConfig":
        """Create from dictionary."""
        config = cls()
        
        if "fisher" in d:
            fisher_dict = d["fisher"]
            if "estimation_type" in fisher_dict:
                config.fisher.estimation_type = FisherEstimationType(fisher_dict["estimation_type"])
            if "num_samples" in fisher_dict:
                config.fisher.num_samples = fisher_dict["num_samples"]
        
        if "calibration_samples" in d:
            config.calibration_samples = d["calibration_samples"]
        
        return config


# Preset configurations
def get_fast_analysis_config() -> AnalysisConfig:
    """Get configuration for fast analysis (fewer samples)."""
    return AnalysisConfig(
        calibration_samples=50,
        fisher=FisherConfig(num_samples=50),
        jacobian=JacobianConfig(num_iterations=5),
        compute_redundancy=False,
        store_singular_values=False,
    )


def get_thorough_analysis_config() -> AnalysisConfig:
    """Get configuration for thorough analysis."""
    return AnalysisConfig(
        calibration_samples=500,
        fisher=FisherConfig(
            num_samples=300,
            estimation_type=FisherEstimationType.BLOCKWISE,
        ),
        jacobian=JacobianConfig(
            num_iterations=20,
            compute_full_jacobian=True,
        ),
        compute_redundancy=True,
        store_singular_values=True,
    )


def get_memory_efficient_config() -> AnalysisConfig:
    """Get configuration optimized for low memory usage."""
    return AnalysisConfig(
        calibration_samples=100,
        calibration_batch_size=8,
        fisher=FisherConfig(
            num_samples=50,
            use_gradient_checkpointing=True,
            chunk_size=10,
        ),
        use_cache=False,
        store_singular_values=False,
    )
