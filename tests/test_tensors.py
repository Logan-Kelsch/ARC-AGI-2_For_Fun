import numpy as np
import pytest

from arc_agi2_fun.data import ArcTask, Demonstration
from arc_agi2_fun.tensors import (
    evaluate_tensor,
    get_task_grid,
    grid_to_tensor,
    tensor_to_grid,
)


def test_grid_tensor_round_trip():
    grid = [[0, 3, 9], [2, 2, 1]]
    tensor = grid_to_tensor(grid)

    assert tensor.shape == (2, 3, 10)
    assert tensor.dtype == np.float32
    assert np.allclose(tensor.sum(axis=-1), 1.0)
    assert tensor_to_grid(tensor) == grid


def test_tensor_decoder_accepts_logits():
    logits = np.zeros((1, 2, 10), dtype=float)
    logits[0, 0, 4] = 3.5
    logits[0, 1, 7] = 8.0

    assert tensor_to_grid(logits) == [[4, 7]]


def test_evaluate_tensor_reports_exact_and_partial_accuracy():
    target = grid_to_tensor([[1, 2], [3, 4]])
    exact = target.copy()
    partial = grid_to_tensor([[1, 2], [0, 4]])

    exact_result = evaluate_tensor(exact, target)
    partial_result = evaluate_tensor(partial, target)

    assert exact_result.exact_match is True
    assert exact_result.cell_accuracy == 1.0
    assert partial_result.exact_match is False
    assert partial_result.correct_cells == 3
    assert partial_result.total_cells == 4
    assert partial_result.cell_accuracy == 0.75


def test_evaluate_tensor_rejects_spatial_shape_mismatch_as_failure():
    target = grid_to_tensor([[1, 2], [3, 4]])
    prediction = grid_to_tensor([[1, 2]])

    result = evaluate_tensor(prediction, target)

    assert result.shape_match is False
    assert result.exact_match is False
    assert result.cell_accuracy == 0.0


def test_get_task_grid_selects_train_and_public_test_cells():
    task = ArcTask(
        "toy",
        (Demonstration([[1]], [[2]]),),
        ([[3]],),
        ([[4]],),
    )

    assert get_task_grid(task, section="train", index=0, side="input") == [[1]]
    assert get_task_grid(task, section="train", index=0, side="output") == [[2]]
    assert get_task_grid(task, section="test", index=0, side="input") == [[3]]
    assert get_task_grid(task, section="test", index=0, side="output") == [[4]]


def test_get_task_grid_rejects_hidden_test_output():
    task = ArcTask(
        "hidden",
        (Demonstration([[1]], [[2]]),),
        ([[3]],),
        None,
    )

    with pytest.raises(ValueError):
        get_task_grid(task, section="test", index=0, side="output")
