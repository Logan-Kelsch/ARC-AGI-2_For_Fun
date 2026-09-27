"""ARC-AGI-2 program-synthesis research scaffold."""

from .data import ArcTask, Demonstration, Grid
from .registry import available_solvers, create_solver
from .scoring import grid_exact

__all__ = [
    "ArcTask",
    "Demonstration",
    "Grid",
    "available_solvers",
    "create_solver",
    "grid_exact",
]
