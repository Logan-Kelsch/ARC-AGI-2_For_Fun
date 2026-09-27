from __future__ import annotations

from typing import Protocol

from .data import ArcTask, Grid
from .scoring import AttemptPair
from .search import rank_programs


class Solver(Protocol):
    name: str

    def solve_task(self, task: ArcTask) -> list[AttemptPair]:
        ...


def _zeros_like(grid: Grid) -> Grid:
    return [[0 for _ in row] for row in grid]


class NullSolver:
    """Pipeline sanity baseline: identity first, all-zero grid second."""

    name = "null"

    def solve_task(self, task: ArcTask) -> list[AttemptPair]:
        return [
            AttemptPair(
                attempt_1=[row[:] for row in test_input],
                attempt_2=_zeros_like(test_input),
            )
            for test_input in task.test_inputs
        ]


class PrimitiveSearchSolver:
    """Tiny deterministic program search used only to prove the search boundary."""

    name = "primitive"

    def solve_task(self, task: ArcTask) -> list[AttemptPair]:
        ranked = rank_programs(task)
        if not ranked:
            return NullSolver().solve_task(task)

        first = ranked[0].program
        second = ranked[1].program if len(ranked) > 1 else first

        return [
            AttemptPair(
                attempt_1=first.apply(test_input),
                attempt_2=second.apply(test_input),
            )
            for test_input in task.test_inputs
        ]
