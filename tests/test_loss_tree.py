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
    assert tree.composite.solved_states == ()
    assert tree.composite.unsolved_states == ()


def test_transition_orientation_is_predicted_rows_target_columns():
    predicted = np.array([[2]])
    target = np.array([[8]])

    matrix = color_transition_matrix(predicted, target)

    assert matrix[2, 8] == 1
    assert matrix[8, 2] == 0


def test_uniform_off_diagonal_color_mapping_is_solved_source_state():
    predicted = [
        np.array([[1, 0, 1, 1, 0, 1]]),
        np.array([[1, 1, 0, 1, 0, 1]]),
        np.array([[1, 0, 1, 0, 1]]),
        np.array([[1, 1, 1, 0, 0, 1]]),
        np.array([[1, 0, 1, 1, 0]]),
    ]
    target = [
        np.zeros_like(grid)
        for grid in predicted
    ]

    tree = loss_resolution(predicted, target)

    one = tree.composite.source_state(1)

    assert isinstance(one, TransitionStateNode)
    assert one.solved
    assert one.source == 1
    assert one.target == 0
    assert one.target_counts == {0: 20}
    assert one.sample_indices == (0, 1, 2, 3, 4)
    assert one.solution_idx == -1

    assert tuple(node.source for node in tree.composite.solved_states) == (1,)
    assert tree.composite.unsolved_states == ()

    # Target color 0 is not duplicated as a separate source-state node.
    with pytest.raises(KeyError):
        tree.composite.source_state(0)


def test_source_mapping_with_multiple_targets_is_unsolved():
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

    assert not one.solved
    assert one.target is None
    assert one.target_counts == {0: 5, 3: 1}
    assert tree.composite.solved_states == ()
    assert tree.composite.unsolved_states == (one,)
    assert tree.composite.state_status(1) == "unsolved"


def test_identity_plus_off_diagonal_for_same_source_is_unsolved():
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
    assert not one.solved
    assert tree.composite.state_status(1) == "unsolved"


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

    assert tree.composite.state_status(1) == "solved"
    assert tree.composite.state_status(0) == "identity"

    # 1 -> 0 is represented once, under source state 1 only.
    assert tree.composite.degeneracies(1) == [(1, 0, 2)]
    assert tree.composite.degeneracies(0) == []


def test_shape_source_states_are_partitioned_independently_for_h_and_w():
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

    # H source 2 consistently maps to 5. H source 3 maps to 7.
    assert tree.h.source_state(2).solved
    assert tree.h.source_state(2).target == 5
    assert tree.h.source_state(3).solved
    assert tree.h.source_state(3).target == 7

    # W source 4 consistently maps to 6; W source 3 is identity and omitted.
    assert tree.w.source_state(4).solved
    assert tree.w.source_state(4).target == 6
    with pytest.raises(KeyError):
        tree.w.source_state(3)


def test_shape_same_source_branching_to_multiple_targets_is_unsolved():
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
    assert not state.solved
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

    assert not invalid.solved
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
    assert tree.composite.source_state(1).solved
    assert tree.composite.source_state(1).target == 0
    assert tree.composite.source_state(2).solved
    assert tree.composite.source_state(2).target == 3


def test_source_state_solution_idx_can_be_assigned():
    tree = loss_resolution(
        [np.array([[1]])],
        [np.array([[0]])],
    )

    state = tree.composite.source_state(1)
    state.solution_idx = 17

    assert state.solution_idx == 17
    assert tree.composite.solution_idx == -1


def test_summary_contains_solved_and_unsolved_source_partitions():
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

    solved = summary["composite"]["solved_states"]
    unsolved = summary["composite"]["unsolved_states"]

    assert solved[0]["source"] == 1
    assert solved[0]["target"] == 0
    assert unsolved[0]["source"] == 2
    assert unsolved[0]["target_counts"] == {3: 1, 4: 1}


def test_inspect_loss_partitions_source_states_into_solved_and_unsolved(capsys):
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
    assert "solved (1)" in report
    assert "✓ 1 -> 0 x2" in report
    assert "unsolved (1)" in report
    assert "X 2 -> {3 x3, 4 x1}" in report

    # Target states are not duplicated as their own loss nodes.
    assert "X 0 ->" not in report
    assert "X 3 ->" not in report
    assert "X 4 ->" not in report


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
