from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import ArcTask, Grid
from .solver import Solver


def _valid_grid(grid: Any) -> bool:
    if not isinstance(grid, list) or not grid:
        return False
    width = len(grid[0]) if isinstance(grid[0], list) else 0
    if width == 0:
        return False
    for row in grid:
        if not isinstance(row, list) or len(row) != width:
            return False
        if any(not isinstance(v, int) or v < 0 or v > 9 for v in row):
            return False
    return True


def build_submission(
    tasks: dict[str, ArcTask],
    solver: Solver,
) -> dict[str, list[dict[str, Grid]]]:
    submission: dict[str, list[dict[str, Grid]]] = {}

    for task_id, task in tasks.items():
        predictions = solver.solve_task(task)
        if len(predictions) != len(task.test_inputs):
            raise ValueError(
                f"{task_id}: solver returned {len(predictions)} predictions for "
                f"{len(task.test_inputs)} test inputs."
            )

        rows: list[dict[str, Grid]] = []
        for prediction in predictions:
            if not _valid_grid(prediction.attempt_1):
                raise ValueError(f"{task_id}: invalid attempt_1 grid.")
            if not _valid_grid(prediction.attempt_2):
                raise ValueError(f"{task_id}: invalid attempt_2 grid.")
            rows.append(
                {
                    "attempt_1": prediction.attempt_1,
                    "attempt_2": prediction.attempt_2,
                }
            )
        submission[task_id] = rows

    return submission


def write_submission(
    submission: dict[str, list[dict[str, Grid]]],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(submission, separators=(",", ":")))
    return path
