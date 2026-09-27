from __future__ import annotations

from dataclasses import dataclass

from .data import ArcTask
from .scoring import TaskScore, score_task
from .solver import Solver


@dataclass(frozen=True)
class EvaluationReport:
    solved_outputs: int
    total_outputs: int
    task_scores: tuple[TaskScore, ...]

    @property
    def accuracy(self) -> float:
        return self.solved_outputs / self.total_outputs if self.total_outputs else 0.0


def evaluate_tasks(
    tasks: dict[str, ArcTask],
    solver: Solver,
    *,
    limit: int | None = None,
) -> EvaluationReport:
    selected = list(tasks.values())
    if limit is not None:
        selected = selected[:limit]

    scores: list[TaskScore] = []
    for task in selected:
        predictions = solver.solve_task(task)
        scores.append(score_task(task, predictions))

    return EvaluationReport(
        solved_outputs=sum(score.solved_outputs for score in scores),
        total_outputs=sum(score.total_outputs for score in scores),
        task_scores=tuple(scores),
    )
