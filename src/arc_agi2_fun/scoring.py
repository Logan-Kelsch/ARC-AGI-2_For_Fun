from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import ArcTask, Grid


def grid_exact(predicted: Grid, target: Grid) -> bool:
    return predicted == target


def cell_accuracy(predicted: Grid, target: Grid) -> float:
    p = np.asarray(predicted, dtype=np.int8)
    t = np.asarray(target, dtype=np.int8)
    if p.shape != t.shape or p.size == 0:
        return 0.0
    return float(np.mean(p == t))


@dataclass(frozen=True)
class AttemptPair:
    attempt_1: Grid
    attempt_2: Grid


@dataclass(frozen=True)
class TaskScore:
    task_id: str
    solved_outputs: int
    total_outputs: int

    @property
    def accuracy(self) -> float:
        return self.solved_outputs / self.total_outputs if self.total_outputs else 0.0


def score_task(task: ArcTask, predictions: list[AttemptPair]) -> TaskScore:
    if task.test_outputs is None:
        raise ValueError(f"Task {task.task_id} has no known test outputs.")
    if len(predictions) != len(task.test_outputs):
        raise ValueError("Prediction count does not match test-output count.")

    solved = 0
    for attempts, target in zip(predictions, task.test_outputs):
        solved += int(
            grid_exact(attempts.attempt_1, target)
            or grid_exact(attempts.attempt_2, target)
        )
    return TaskScore(task.task_id, solved, len(task.test_outputs))
