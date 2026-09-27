from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .data import ArcTask, Grid

NUM_COLORS = 10


def grid_to_tensor(
    grid: Grid,
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Convert an ARC HxW integer grid to an HxWx10 one-hot tensor."""
    array = np.asarray(grid, dtype=np.int8)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D ARC grid, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError("ARC grid may not be empty.")
    if np.any((array < 0) | (array >= NUM_COLORS)):
        raise ValueError("ARC cell values must be in [0, 9].")
    return np.eye(NUM_COLORS, dtype=dtype)[array]


def tensor_to_grid(tensor: np.ndarray) -> Grid:
    """Decode an HxWx10 tensor to an ARC grid using channel argmax.

    The tensor may contain one-hot values, probabilities, logits, or arbitrary
    real-valued channel scores.
    """
    array = np.asarray(tensor)
    if array.ndim != 3:
        raise ValueError(
            f"Expected an HxWx{NUM_COLORS} tensor, got shape {array.shape}."
        )
    if array.shape[-1] != NUM_COLORS:
        raise ValueError(
            f"Expected {NUM_COLORS} color channels, got {array.shape[-1]}."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Spatial tensor dimensions may not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Tensor contains NaN or infinite values.")
    return np.argmax(array, axis=-1).astype(int).tolist()


def get_task_grid(
    task: ArcTask,
    *,
    section: Literal["train", "test"],
    index: int,
    side: Literal["input", "output"],
) -> Grid:
    """Select any known input/output grid from a public ARC task."""
    if section == "train":
        if index < 0 or index >= len(task.train):
            raise IndexError(f"Training example index {index} is out of range.")
        pair = task.train[index]
        return pair.input if side == "input" else pair.output

    if index < 0 or index >= len(task.test_inputs):
        raise IndexError(f"Test example index {index} is out of range.")
    if side == "input":
        return task.test_inputs[index]

    if task.test_outputs is None:
        raise ValueError(
            f"Task {task.task_id} does not expose test outputs in this dataset."
        )
    return task.test_outputs[index]


@dataclass(frozen=True)
class TensorEvaluation:
    prediction_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    shape_match: bool
    exact_match: bool
    cell_accuracy: float
    correct_cells: int
    total_cells: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_tensor(
    prediction: np.ndarray,
    target: np.ndarray,
) -> TensorEvaluation:
    """Evaluate a submitted 3D tensor against a target 3D ARC tensor.

    Both tensors are decoded with argmax across their 10 color channels.
    Competition success is exact equality of the decoded output grid.
    """
    prediction_array = np.asarray(prediction)
    target_array = np.asarray(target)

    predicted_grid = np.asarray(tensor_to_grid(prediction_array), dtype=np.int8)
    target_grid = np.asarray(tensor_to_grid(target_array), dtype=np.int8)

    shape_match = predicted_grid.shape == target_grid.shape
    if not shape_match:
        return TensorEvaluation(
            prediction_shape=tuple(prediction_array.shape),
            target_shape=tuple(target_array.shape),
            shape_match=False,
            exact_match=False,
            cell_accuracy=0.0,
            correct_cells=0,
            total_cells=int(target_grid.size),
        )

    correct = predicted_grid == target_grid
    correct_cells = int(np.sum(correct))
    total_cells = int(target_grid.size)

    return TensorEvaluation(
        prediction_shape=tuple(prediction_array.shape),
        target_shape=tuple(target_array.shape),
        shape_match=True,
        exact_match=bool(np.all(correct)),
        cell_accuracy=correct_cells / total_cells if total_cells else 0.0,
        correct_cells=correct_cells,
        total_cells=total_cells,
    )
