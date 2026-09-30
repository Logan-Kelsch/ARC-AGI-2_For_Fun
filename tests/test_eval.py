import numpy as np
import pytest

from notebooks.ops.eval import (
    INVALID_STATE,
    DimensionLoss,
    color_transition_matrix,
    loss_resolution,
)


def test_loss_tree_has_requested_structure_and_default_solution_indices():
    tree = loss_resolution(
        [[0, 1], [2, 3]],
        [[0, 1], [2, 3]],
    )

    assert tree.root.name == "root"
    assert set(tree.root.children) == {"shape", "composite"}
    assert set(tree["shape"].children) == {"h", "w"}
    assert tree["shape"]["h"].children == {}
    assert tree["shape"]["w"].children == {}
    assert tree["composite"].children == {}

    assert all(node.solution_idx == -1 for node in tree.walk())


def test_shape_leaves_store_predicted_target_and_signed_delta():
    tree = loss_resolution(
        np.zeros((2, 4), dtype=int),
        np.zeros((3, 2), dtype=int),
    )

    h = tree["shape"]["h"].value
    w = tree["shape"]["w"].value

    assert isinstance(h, DimensionLoss)
    assert h.predicted == 2
    assert h.target == 3
    assert h.delta == -1
    assert not h.resolved

    assert w.predicted == 4
    assert w.target == 2
    assert w.delta == 2
    assert not w.resolved


def test_perfect_prediction_has_only_diagonal_transitions():
    y = np.array(
        [
            [0, 1],
            [2, 2],
        ]
    )

    tree = loss_resolution(y, y)
    matrix = tree.transition_matrix

    assert matrix.shape == (11, 11)
    assert matrix[0, 0] == 1
    assert matrix[1, 1] == 1
    assert matrix[2, 2] == 2
    assert matrix.sum() == 4
    assert np.array_equal(matrix, np.diag(np.diag(matrix)))
    assert tree.shape_resolved
    assert tree.composite_resolved
    assert tree.resolved


def test_color_errors_are_predicted_to_target_transitions():
    yhat = np.array(
        [
            [0, 1],
            [2, 3],
        ]
    )
    y = np.array(
        [
            [0, 2],
            [2, 1],
        ]
    )

    matrix = color_transition_matrix(yhat, y)

    assert matrix[0, 0] == 1
    assert matrix[1, 2] == 1
    assert matrix[2, 2] == 1
    assert matrix[3, 1] == 1
    assert matrix.sum() == 4


def test_target_pixels_missing_from_prediction_use_invalid_row():
    yhat = np.array([[1, 2]])
    y = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    tree = loss_resolution(yhat, y)
    matrix = tree.transition_matrix

    assert matrix[1, 1] == 1
    assert matrix[2, 2] == 1
    assert matrix[INVALID_STATE, 3] == 1
    assert matrix[INVALID_STATE, 4] == 1
    assert matrix.sum() == 4
    assert not tree.resolved


def test_prediction_pixels_outside_target_use_invalid_column():
    yhat = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )
    y = np.array([[1, 2]])

    matrix = color_transition_matrix(yhat, y)

    assert matrix[1, 1] == 1
    assert matrix[2, 2] == 1
    assert matrix[3, INVALID_STATE] == 1
    assert matrix[4, INVALID_STATE] == 1
    assert matrix.sum() == 4


def test_cross_dimension_mismatch_counts_only_union_of_real_pixels():
    yhat = np.array(
        [
            [1, 1, 9],
            [2, 2, 9],
        ]
    )
    y = np.array(
        [
            [1, 1],
            [2, 2],
            [3, 3],
        ]
    )

    matrix = color_transition_matrix(yhat, y)

    assert matrix[1, 1] == 2
    assert matrix[2, 2] == 2
    assert matrix[9, INVALID_STATE] == 2
    assert matrix[INVALID_STATE, 3] == 2
    assert matrix.sum() == 8


def test_solution_idx_can_be_claimed_by_later_gp_search():
    tree = loss_resolution([[1]], [[1]])

    tree["shape"]["h"].solution_idx = 7
    tree["composite"].solution_idx = 12

    assert tree["shape"]["h"].solution_idx == 7
    assert tree["composite"].solution_idx == 12
    assert tree["shape"]["w"].solution_idx == -1


def test_invalid_arc_colors_are_rejected():
    with pytest.raises(ValueError):
        loss_resolution([[10]], [[0]])
