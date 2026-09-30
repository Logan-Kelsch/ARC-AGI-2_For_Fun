import numpy as np
import pytest

from notebooks.ops.loss import (
    INVALID_STATE,
    color_transition_matrix,
    inspect_loss,
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
    assert tree.sample_count == 1

    assert all(node.solution_idx == -1 for node in tree.root.walk())


def test_exact_single_prediction_is_fully_resolved():
    grid = np.array(
        [
            [0, 1],
            [2, 2],
        ]
    )

    tree = loss_resolution(grid, grid)

    assert tree.h.resolved
    assert tree.w.resolved
    assert tree.shape.resolved
    assert tree.composite.resolved
    assert tree.resolved

    matrix = tree.composite.matrix
    assert matrix.shape == (11, 11)
    assert matrix[0, 0] == 1
    assert matrix[1, 1] == 1
    assert matrix[2, 2] == 2
    assert matrix.sum() == 4


def test_transition_orientation_is_predicted_rows_target_columns():
    predicted = np.array([[2]])
    target = np.array([[8]])

    matrix = color_transition_matrix(predicted, target)

    assert matrix[2, 8] == 1
    assert matrix[8, 2] == 0


def test_multisample_shape_has_aggregate_and_sample_specific_matrices():
    predicted = [
        np.zeros((2, 3), dtype=int),
        np.zeros((3, 2), dtype=int),
        np.zeros((4, 5), dtype=int),
    ]
    target = [
        np.zeros((2, 3), dtype=int),
        np.zeros((4, 2), dtype=int),
        np.zeros((4, 3), dtype=int),
    ]

    tree = loss_resolution(predicted, target)

    assert tree.sample_count == 3

    assert tree.h.matrix[2, 2] == 1
    assert tree.h.matrix[3, 4] == 1
    assert tree.h.matrix[4, 4] == 1
    assert tree.h.matrix.sum() == 3

    assert tree.w.matrix[3, 3] == 1
    assert tree.w.matrix[2, 2] == 1
    assert tree.w.matrix[5, 3] == 1
    assert tree.w.matrix.sum() == 3

    assert tree.h.sample_matrices[0][2, 2] == 1
    assert tree.h.sample_matrices[1][3, 4] == 1
    assert tree.h.sample_matrices[2][4, 4] == 1

    assert tree.w.sample_matrices[0][3, 3] == 1
    assert tree.w.sample_matrices[1][2, 2] == 1
    assert tree.w.sample_matrices[2][5, 3] == 1

    assert tree.h.sample_resolved == (True, False, True)
    assert tree.w.sample_resolved == (True, True, False)
    assert not tree.shape.resolved


def test_multisample_composite_aggregates_without_losing_sample_identity():
    predicted = [
        np.array([[3, 0]]),
        np.array([[3, 1]]),
    ]
    target = [
        np.array([[3, 0]]),
        np.array([[4, 1]]),
    ]

    tree = loss_resolution(predicted, target)
    aggregate = tree.composite.matrix

    assert aggregate[3, 3] == 1
    assert aggregate[3, 4] == 1
    assert aggregate[0, 0] == 1
    assert aggregate[1, 1] == 1
    assert aggregate.sum() == 4

    assert tree.composite.sample_matrices[0][3, 3] == 1
    assert tree.composite.sample_matrices[1][3, 4] == 1

    assert tree.composite.sample_resolved == (True, False)
    assert tree.composite.state_status(0) == "resolved"
    assert tree.composite.state_status(1) == "resolved"
    assert tree.composite.state_status(3) == "degenerate"
    assert tree.composite.state_status(4) == "degenerate"
    assert tree.composite.state_status(5) == "unused"

    assert tree.composite.degeneracies(3) == [(3, 4, 1)]
    assert tree.composite.degeneracies(4) == [(3, 4, 1)]
    assert tree.composite.degeneracies(3, sample_idx=0) == []
    assert tree.composite.degeneracies(3, sample_idx=1) == [(3, 4, 1)]


def test_shape_mismatch_uses_invalid_state_in_composite():
    predicted = np.array([[1, 2]])
    target = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    tree = loss_resolution(predicted, target)

    assert tree.h.matrix[1, 2] == 1
    assert tree.w.matrix[2, 2] == 1

    assert tree.composite.matrix[1, 1] == 1
    assert tree.composite.matrix[2, 2] == 1
    assert tree.composite.matrix[INVALID_STATE, 3] == 1
    assert tree.composite.matrix[INVALID_STATE, 4] == 1

    assert tree.composite.state_status(INVALID_STATE) == "degenerate"
    assert not tree.resolved


def test_gpmat_style_object_arrays_can_be_passed_directly():
    predicted = np.empty(2, dtype=object)
    target = np.empty(2, dtype=object)

    predicted[0] = np.array([[1, 1]])
    predicted[1] = np.array([[2], [2]])

    target[0] = np.array([[1, 1]])
    target[1] = np.array([[2], [3]])

    tree = loss_resolution(predicted, target)

    assert tree.sample_count == 2
    assert tree.h.resolved
    assert tree.w.resolved
    assert not tree.composite.resolved
    assert tree.composite.matrix[2, 3] == 1


def test_color_is_only_resolved_when_all_transitions_involving_it_are_diagonal():
    predicted = [
        np.array([[3, 7]]),
        np.array([[2, 3]]),
    ]
    target = [
        np.array([[3, 7]]),
        np.array([[3, 3]]),
    ]

    tree = loss_resolution(predicted, target)

    # 3 -> 3 is clean, but 2 -> 3 means target color 3 is still involved
    # in an unresolved transition. State 3 must therefore receive an X.
    assert tree.composite.state_status(3) == "degenerate"
    assert tree.composite.degeneracies(3) == [(2, 3, 1)]


def test_inspect_loss_marks_every_degeneracy_with_x(capsys):
    predicted = [
        np.array([[3, 0]]),
        np.array([[3, 1, 1]]),
    ]
    target = [
        np.array([[3, 0]]),
        np.array([[4, 1]]),
    ]

    tree = loss_resolution(predicted, target)
    report = inspect_loss(tree)

    captured = capsys.readouterr().out

    assert report in captured
    assert "X root" in report
    assert "X shape" in report
    assert "X w" in report
    assert "X composite" in report
    assert "X 3" in report
    assert "X 4" in report
    assert "sample 1" in report
    assert "3 -> 4 x1" in report


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


def test_summary_exposes_aggregate_and_sample_specific_state():
    predicted = [
        np.array([[1, 2]]),
        np.array([[1, 2]]),
    ]
    target = [
        np.array([[1, 2]]),
        np.array([[1, 3]]),
    ]

    tree = loss_resolution(predicted, target)
    summary = tree.summary()

    assert summary["sample_count"] == 2
    assert summary["resolved"] is False

    assert summary["shape"]["h"]["transition_matrix"][1, 1] == 2
    assert len(summary["shape"]["h"]["sample_transition_matrices"]) == 2

    assert summary["composite"]["transition_matrix"][2, 3] == 1
    assert len(summary["composite"]["sample_transition_matrices"]) == 2


def test_sample_count_must_match():
    with pytest.raises(ValueError, match="same number of samples"):
        loss_resolution(
            [np.array([[1]]), np.array([[2]])],
            [np.array([[1]])],
        )


def test_non_arc_values_are_rejected():
    with pytest.raises(ValueError):
        loss_resolution([[10]], [[0]])

    with pytest.raises(ValueError):
        loss_resolution([[1.5]], [[1]])
