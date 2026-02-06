"""
Flow Analysis Module

This subpackage implements the theoretical foundation of FLOWRA:
- Fisher Information Matrix estimation
- Flow Sensitivity Index computation
- Flow Conductance metrics
- Redundancy analysis
"""

from flowra.core.analysis.flow_analyzer import FlowAnalyzer
from flowra.core.analysis.flow_metrics import FlowProfile, LayerFlowInfo
from flowra.core.analysis.fisher import (
    FisherEstimator,
    EmpiricalFisherEstimator,
    DiagonalFisherEstimator,
    KFACEstimator,
)
from flowra.core.analysis.jacobian import (
    JacobianEstimator,
    PowerIterationEstimator,
    HutchinsonEstimator,
)
from flowra.core.analysis.redundancy import (
    RedundancyAnalyzer,
    compute_effective_rank,
    compute_redundancy_coefficient,
)

__all__ = [
    "FlowAnalyzer",
    "FlowProfile",
    "LayerFlowInfo",
    "FisherEstimator",
    "EmpiricalFisherEstimator",
    "DiagonalFisherEstimator",
    "KFACEstimator",
    "JacobianEstimator",
    "PowerIterationEstimator",
    "HutchinsonEstimator",
    "RedundancyAnalyzer",
    "compute_effective_rank",
    "compute_redundancy_coefficient",
]
