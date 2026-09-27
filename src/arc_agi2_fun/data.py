from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

Grid = list[list[int]]


@dataclass(frozen=True)
class Demonstration:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class ArcTask:
    task_id: str
    train: tuple[Demonstration, ...]
    test_inputs: tuple[Grid, ...]
    test_outputs: tuple[Grid, ...] | None = None


def _validate_grid(grid: Any) -> Grid:
    if not isinstance(grid, list) or not grid:
        raise ValueError("Grid must be a non-empty list of rows.")
    width = None
    out: Grid = []
    for row in grid:
        if not isinstance(row, list) or not row:
            raise ValueError("Grid rows must be non-empty lists.")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError("Grid must be rectangular.")
        converted = [int(value) for value in row]
        if any(value < 0 or value > 9 for value in converted):
            raise ValueError("ARC cell values must be integers from 0 to 9.")
        out.append(converted)
    if len(out) > 30 or (width or 0) > 30:
        raise ValueError("ARC grids may not exceed 30x30.")
    return out


def _task_from_payload(task_id: str, payload: dict[str, Any]) -> ArcTask:
    train = tuple(
        Demonstration(
            input=_validate_grid(pair["input"]),
            output=_validate_grid(pair["output"]),
        )
        for pair in payload["train"]
    )
    test_inputs = tuple(_validate_grid(pair["input"]) for pair in payload["test"])
    test_outputs: tuple[Grid, ...] | None = None
    if payload["test"] and all("output" in pair for pair in payload["test"]):
        test_outputs = tuple(_validate_grid(pair["output"]) for pair in payload["test"])
    return ArcTask(task_id, train, test_inputs, test_outputs)


def load_task_file(path: str | Path) -> ArcTask:
    path = Path(path)
    return _task_from_payload(path.stem, json.loads(path.read_text()))


def load_official_split(data_root: str | Path, split: str) -> dict[str, ArcTask]:
    root = Path(data_root) / "data" / split
    if not root.exists():
        raise FileNotFoundError(
            f"Missing {root}. Run `make data` to fetch the official ARC-AGI-2 repo."
        )
    tasks = {path.stem: load_task_file(path) for path in sorted(root.glob("*.json"))}
    if not tasks:
        raise RuntimeError(f"No ARC task JSON files found in {root}.")
    return tasks


def load_kaggle_challenges(
    challenges_path: str | Path,
    solutions_path: str | Path | None = None,
) -> dict[str, ArcTask]:
    challenges = json.loads(Path(challenges_path).read_text())
    solutions = (
        json.loads(Path(solutions_path).read_text())
        if solutions_path is not None
        else None
    )

    tasks: dict[str, ArcTask] = {}
    for task_id, payload in challenges.items():
        task = _task_from_payload(task_id, payload)
        if solutions is not None:
            outputs = tuple(_validate_grid(grid) for grid in solutions[task_id])
            task = ArcTask(task.task_id, task.train, task.test_inputs, outputs)
        tasks[task_id] = task
    return tasks


def find_kaggle_competition_dir(root: str | Path = "/kaggle/input") -> Path:
    root = Path(root)
    direct = root / "arc-prize-2026-arc-agi-2"
    if (direct / "arc-agi_test_challenges.json").exists():
        return direct

    matches = list(root.rglob("arc-agi_test_challenges.json"))
    if not matches:
        raise FileNotFoundError(
            "Could not find arc-agi_test_challenges.json under /kaggle/input."
        )
    return matches[0].parent
