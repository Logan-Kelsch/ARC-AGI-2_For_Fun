import numpy as np
import pytest

from notebooks.ops.loss import (
    INVALID_STATE,
    TransitionStateNode,
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
    assert tree.sample_count == 1
    assert all(node.solution_idx == -1 for node in tree.root.walk())


def test_exact_prediction_has_no_nonzero_delta_source_nodes():
    grid = np.array(
        [
            [0, 1],
            [2, 2],
        ]
    )

    tree = loss_resolution(grid, grid)

    assert tree.resolved
    assert tree.h.state_nodes == ()
    assert tree.w.state_nodes == ()
    assert tree.composite.state_nodes == ()
    assert tree.h.uniform
    assert tree.w.uniform
    assert tree.composite.uniform


def test_transition_orientation_is_predicted_rows_target_columns():
    predicted = np.array([[2]])
    target = np.array([[8]])

    matrix = color_transition_matrix(predicted, target)

    assert matrix[2, 8] == 1
    assert matrix[8, 2] == 0


def test_uniform_off_diagonal_mapping_is_one_checked_source_state():
    predicted = [
        np.ones((1, 6), dtype=int),
        np.ones((1, 6), dtype=int),
        np.ones((1, 5), dtype=int),
        np.ones((1, 6), dtype=int),
        np.ones((1, 5), dtype=int),
    ]
    target = [np.zeros_like(grid) for grid in predicted]

    tree = loss_resolution(predicted, target)
    one = tree.composite.source_state(1)

    assert isinstance(one, TransitionStateNode)
    assert one.uniform
    assert one.source == 1
    assert one.target == 0
    assert one.target_counts == {0: 28}
    assert one.sample_indices == (0, 1, 2, 3, 4)
    assert one.solution_idx == -1

    assert tree.composite.state_nodes == (one,)

    with pytest.raises(KeyError):
        tree.composite.source_state(0)


def test_branching_source_mapping_is_nonuniform():
    predicted = [
        np.array([[1, 1, 1]]),
        np.array([[1, 1, 1]]),
    ]
    target = [
        np.array([[0, 0, 0]]),
        np.array([[0, 3, 0]]),
    ]

    tree = loss_resolution(predicted, target)
    one = tree.composite.source_state(1)

    assert not one.uniform
    assert one.target is None
    assert one.target_counts == {0: 5, 3: 1}
    assert tree.composite.state_nodes == (one,)
    assert tree.composite.state_status(1) == "nonuniform"


def test_identity_plus_off_diagonal_for_same_source_is_nonuniform():
    predicted = [
        np.array([[1, 1]]),
        np.array([[1, 1]]),
    ]
    target = [
        np.array([[1, 0]]),
        np.array([[1, 0]]),
    ]

    tree = loss_resolution(predicted, target)
    one = tree.composite.source_state(1)

    assert one.target_counts == {0: 2, 1: 2}
    assert not one.uniform
    assert tree.composite.state_status(1) == "nonuniform"


def test_source_status_uses_rows_only_not_incoming_target_transitions():
    predicted = [
        np.array([[1, 0]]),
        np.array([[1, 0]]),
    ]
    target = [
        np.array([[0, 0]]),
        np.array([[0, 0]]),
    ]

    tree = loss_resolution(predicted, target)

    assert tree.composite.state_status(1) == "uniform"
    assert tree.composite.state_status(0) == "identity"

    assert tree.composite.degeneracies(1) == [(1, 0, 2)]
    assert tree.composite.degeneracies(0) == []


def test_shape_source_states_are_directly_represented_for_h_and_w():
    predicted = [
        np.zeros((2, 3), dtype=int),
        np.zeros((2, 4), dtype=int),
        np.zeros((3, 4), dtype=int),
    ]
    target = [
        np.zeros((5, 3), dtype=int),
        np.zeros((5, 6), dtype=int),
        np.zeros((7, 6), dtype=int),
    ]

    tree = loss_resolution(predicted, target)

    assert tree.h.source_state(2).uniform
    assert tree.h.source_state(2).target == 5
    assert tree.h.source_state(3).uniform
    assert tree.h.source_state(3).target == 7

    assert tree.w.source_state(4).uniform
    assert tree.w.source_state(4).target == 6

    with pytest.raises(KeyError):
        tree.w.source_state(3)


def test_shape_same_source_branching_to_multiple_targets_is_nonuniform():
    predicted = [
        np.zeros((2, 2), dtype=int),
        np.zeros((2, 2), dtype=int),
    ]
    target = [
        np.zeros((3, 2), dtype=int),
        np.zeros((4, 2), dtype=int),
    ]

    tree = loss_resolution(predicted, target)

    state = tree.h.source_state(2)
    assert not state.uniform
    assert state.target_counts == {3: 1, 4: 1}


def test_invalid_state_can_be_a_source_state():
    predicted = np.array([[1, 2]])
    target = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )

    tree = loss_resolution(predicted, target)
    invalid = tree.composite.source_state(INVALID_STATE)

    assert not invalid.uniform
    assert invalid.target_counts == {3: 1, 4: 1}


def test_gpmat_style_object_arrays_can_be_passed_directly():
    predicted = np.empty(2, dtype=object)
    target = np.empty(2, dtype=object)

    predicted[0] = np.array([[1, 1]])
    predicted[1] = np.array([[2], [2]])

    target[0] = np.array([[0, 0]])
    target[1] = np.array([[3], [3]])

    tree = loss_resolution(predicted, target)

    assert tree.sample_count == 2
    assert tree.composite.source_state(1).uniform
    assert tree.composite.source_state(1).target == 0
    assert tree.composite.source_state(2).uniform
    assert tree.composite.source_state(2).target == 3


def test_source_state_solution_idx_remains_available_for_later_gp_evaluation():
    tree = loss_resolution(
        [np.array([[1]])],
        [np.array([[0]])],
    )

    state = tree.composite.source_state(1)
    state.solution_idx = 17

    assert state.solution_idx == 17
    assert tree.composite.solution_idx == -1


def test_summary_contains_direct_state_nodes_and_uniformity_only():
    predicted = [
        np.array([[1, 2]]),
        np.array([[1, 2]]),
    ]
    target = [
        np.array([[0, 3]]),
        np.array([[0, 4]]),
    ]

    tree = loss_resolution(predicted, target)
    summary = tree.summary()

    states = summary["composite"]["state_nodes"]

    assert states[0]["source"] == 1
    assert states[0]["uniform"] is True
    assert states[0]["target"] == 0

    assert states[1]["source"] == 2
    assert states[1]["uniform"] is False
    assert states[1]["target_counts"] == {3: 1, 4: 1}

    assert "solved_states" not in summary["composite"]
    assert "unsolved_states" not in summary["composite"]


def test_inspect_loss_shows_direct_checks_and_x_without_partition_buckets(capsys):
    predicted = [
        np.array([[1, 2, 2]]),
        np.array([[1, 2, 2]]),
    ]
    target = [
        np.array([[0, 3, 3]]),
        np.array([[0, 3, 4]]),
    ]

    tree = loss_resolution(predicted, target)
    report = inspect_loss(tree)

    captured = capsys.readouterr().out

    assert report in captured
    assert "X root" in report
    assert "X composite" in report
    assert "✓ 1 -> 0 x2" in report
    assert "X 2 -> {3 x3, 4 x1}" in report

    assert "solved" not in report.lower()
    assert "unsolved" not in report.lower()

    assert "X 0 ->" not in report
    assert "X 3 ->" not in report
    assert "X 4 ->" not in report


def test_all_uniform_nonzero_delta_mappings_give_checked_structure(capsys):
    predicted = [
        np.array([[1, 2]]),
        np.array([[1, 2]]),
    ]
    target = [
        np.array([[0, 3]]),
        np.array([[0, 3]]),
    ]

    tree = loss_resolution(predicted, target)
    report = inspect_loss(tree)

    capsys.readouterr()

    assert tree.composite.uniform
    assert "✓ root" in report
    assert "✓ composite" in report
    assert "✓ 1 -> 0 x2" in report
    assert "✓ 2 -> 3 x2" in report


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
