"""
Calibration-phase analysis: flow-change descriptors, architecture handlers
and layer / block criticality scores.
"""

from flowra.core.analysis.flow_descriptor import (
    FixedProjector,
    align_states,
    flow_change,
    flow_change_from_sequence,
)
from flowra.core.analysis.handlers import (
    ArchitectureHandler,
    FeedForwardHandler,
    GatedRNNHandler,
    ResidualCNNHandler,
    TargetInfo,
    TransformerHandler,
    allocate_head_ranks,
    attention_ffn_rank_ratio,
    extract_block,
    get_handler,
    head_divergence,
    head_flow_descriptors,
    split_attention_rank,
)
from flowra.core.analysis.profiler import CalibrationProfile, FlowProfiler, ModuleProfile

__all__ = [
    "FixedProjector",
    "align_states",
    "flow_change",
    "flow_change_from_sequence",
    "ArchitectureHandler",
    "FeedForwardHandler",
    "GatedRNNHandler",
    "ResidualCNNHandler",
    "TargetInfo",
    "TransformerHandler",
    "allocate_head_ranks",
    "attention_ffn_rank_ratio",
    "extract_block",
    "get_handler",
    "head_divergence",
    "head_flow_descriptors",
    "split_attention_rank",
    "CalibrationProfile",
    "FlowProfiler",
    "ModuleProfile",
]
