import numpy as np
import pytest

from notebooks.ops.loss import (
    INVALID_STATE,
    TRANSITION_LABELS,
    color_transition_matrix,
    loss_resolution,
)


def test_loss_tree_has_fixed_hierarchy_and_default_solution_indices():
    tree = loss_resolution(
        [[0, 1], [2, 2]],
        [[0, 1], [2, 2]],
    )

    assert tree.root.name == "root"
    assert [child.name for child in tree.root.children] == ["shape", "composite"]
    assert [child.name for child in tree.shape.children] == ["h", "w"]

    assert tree.h.is_leaf
    assert tree.w.is_leaf
    assert tree.composite.is_leaf

    assert all(node.solution_idx == -1 for node in tree.root.walk())


def test_exact_prediction_is_fully_resolved():
    grid = np.array(
        [
            [0, 1],
            [2, 2],
        ]
    )

    tree = loss_resolution(grid, grid)

    assert tree.h.loss == 0
    assert tree.w.loss == 0
    assert tree.shape.resolved
    assert tree.composite.resolved
    assert tree.resolved

    matrix = tree.composite.loss
    assert matrix.shape == (11, 11)
    assert matrix[0, 0] == 1
    assert matrix[1, 1] == 1
    assert matrix[2, 2] == 2
    assert matrix.sum() == 4


def test_same_shape_color_errors_are_off_diagonal_transitions():
    target = np.array(
        [
            [1, 2],
            [2, 0],
        ]
    )
    predicted = np.array(
        [
            [1, 3],
            [0, 0],
        ]
    )

    tree = loss_resolution(predicted, target)
    matrix = tree.composite.loss

    assert tree.shape.resolved
    assert not tree.composite.resolved
    assert not tree.resolved

    assert matrix[1, 1] == 1
    assert matrix[2, 3] == 1
    assert matrix[2, 0] == 1
    assert matrix[0, 0] == 1
    assert matrix.sum() == 4


def test_prediction_smaller_than_target_uses_target_to_invalid_transitions():
    target = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    predicted = np.array(
        [
            [1, 2],
            [4, 5],
        ]
    )

    tree = loss_resolution(predicted, target)
    matrix = tree.composite.loss

    assert tree.h.loss == 0
    assert tree.w.loss == 1
    assert not tree.shape.resolved
    assert matrix[3, INVALID_STATE] == 1
    assert matrix[6, INVALID_STATE] == 1
    assert matrix.sum() == 6
    assert not tree.composite.resolved


def test_prediction_larger_than_target_uses_invalid_to_prediction_transitions():
    target = np.array([[1, 2]])
    predicted = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    tree = loss_resolution(predicted, target)
    matrix = tree.composite.loss

    assert tree.h.loss == 1
    assert tree.w.loss == 1
    assert matrix[INVALID_STATE, 3] == 1
    assert matrix[INVALID_STATE, 4] == 1
    assert matrix[INVALID_STATE, 5] == 1
    assert matrix[INVALID_STATE, 6] == 1
    assert matrix.sum() == 6


def test_crossed_shape_mismatch_ignores_coordinates_outside_both_grids():
    target = np.array(
        [
            [1, 1, 1],
            [2, 2, 2],
        ]
    )
    predicted = np.array(
        [
            [1, 1],
            [2, 2],
            [3, 3],
        ]
    )

    matrix = color_transition_matrix(predicted, target)

    # 6 target cells + 6 predicted cells - 4 overlapping coordinates.
    assert matrix.sum() == 8


def test_transition_matrix_orientation_is_target_rows_prediction_columns():
    target = np.array([[8]])
    predicted = np.array([[2]])

    matrix = color_transition_matrix(predicted, target)

    assert matrix[8, 2] == 1
    assert matrix[2, 8] == 0


def test_solution_indices_can_be_assigned_independently_per_node():
    tree = loss_resolution([[1]], [[1]])

    tree.root.solution_idx = 4
    tree.shape.solution_idx = 7
    tree.h.solution_idx = 9
    tree.composite.solution_idx = 12

    assert tree.root.solution_idx == 4
    assert tree.shape.solution_idx == 7
    assert tree.h.solution_idx == 9
    assert tree.w.solution_idx == -1
    assert tree.composite.solution_idx == 12


def test_summary_exposes_tree_state_without_scalar_total_loss():
    tree = loss_resolution([[1, 2]], [[1, 3]])

    summary = tree.summary()

    assert summary["resolved"] is False
    assert summary["shape"]["h"]["loss"] == 0
    assert summary["shape"]["w"]["loss"] == 0
    assert summary["composite"]["transition_matrix"][3, 2] == 1
    assert summary["composite"]["labels"] == TRANSITION_LABELS


def test_non_arc_values_are_rejected():
    with pytest.raises(ValueError):
        loss_resolution([[10]], [[0]])

    with pytest.raises(ValueError):
        loss_resolution([[1.5]], [[1]])
