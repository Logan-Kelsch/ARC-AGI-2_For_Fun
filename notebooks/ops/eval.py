"""Compatibility module for the canonical GP loss evaluator.

The active loss-tree implementation lives in notebooks.ops.loss. This module
re-exports that API so older notebook imports continue to use the same
transition orientation and multi-sample behavior.
"""

from .loss import (
    ARC_COLOR_COUNT,
    ARC_MAX_GRID_DIM,
    DIMENSION_LABELS,
    DIMENSION_STATE_COUNT,
    INVALID_STATE,
    TRANSITION_LABELS,
    TRANSITION_STATE_COUNT,
    LossNode,
    TransitionStateNode,
    LossTree,
    color_transition_matrix,
    dimension_transition_matrix,
    inspect_loss,
    loss_resolution,
    resolve_loss_tree,
)

__all__ = [
    "ARC_COLOR_COUNT",
    "ARC_MAX_GRID_DIM",
    "DIMENSION_LABELS",
    "DIMENSION_STATE_COUNT",
    "INVALID_STATE",
    "TRANSITION_LABELS",
    "TRANSITION_STATE_COUNT",
    "LossNode",
    "TransitionStateNode",
    "LossTree",
    "color_transition_matrix",
    "dimension_transition_matrix",
    "inspect_loss",
    "loss_resolution",
    "resolve_loss_tree",
]
