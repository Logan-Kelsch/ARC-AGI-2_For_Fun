"""ARC-AGI-2 program-synthesis research scaffold."""

from .data import ArcTask, Demonstration, Grid
from .registry import available_solvers, create_solver
from .scoring import grid_exact
from .tensors import (
    NUM_COLORS,
    TensorEvaluation,
    evaluate_tensor,
    get_task_grid,
    grid_to_tensor,
    tensor_to_grid,
)

__all__ = [
    "ArcTask",
    "Demonstration",
    "Grid",
    "NUM_COLORS",
    "TensorEvaluation",
    "available_solvers",
    "create_solver",
    "evaluate_tensor",
    "get_task_grid",
    "grid_exact",
    "grid_to_tensor",
    "tensor_to_grid",
]
