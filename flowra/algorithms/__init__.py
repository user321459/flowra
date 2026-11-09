"""Rank allocation, dynamic refinement and multi-task composition."""

from flowra.algorithms.allocation import (
    AllocationResult,
    BlockAssignment,
    allocate_ranks,
    generate_block_blueprint,
    uniform_rank_for_budget,
    uniform_ranks,
)
from flowra.algorithms.composition import (
    NaiveMergeComposer,
    OrthogonalComposer,
    TaskArithmeticComposer,
    TIESComposer,
    adapter_deltas,
    apply_deltas,
)
from flowra.algorithms.refinement import RankRefiner, refinement_schedule

__all__ = [
    "AllocationResult",
    "BlockAssignment",
    "allocate_ranks",
    "generate_block_blueprint",
    "uniform_rank_for_budget",
    "uniform_ranks",
    "NaiveMergeComposer",
    "OrthogonalComposer",
    "TaskArithmeticComposer",
    "TIESComposer",
    "adapter_deltas",
    "apply_deltas",
    "RankRefiner",
    "refinement_schedule",
]
