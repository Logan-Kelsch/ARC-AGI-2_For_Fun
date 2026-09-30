import numpy as np
import pytest

from types import SimpleNamespace

from notebooks.ops.GP import (
    GP_EvalNode,
    GP_EvalTree,
    GP_Set,
    gp_fill_status,
    init_gp_eval_tree,
    init_gp_mat,
)


def _gp_set(rows, statuses):
    data = np.empty(len(rows), dtype=object)

    for sample_index, row in enumerate(rows):
        genes = np.empty(len(row), dtype=object)
        for gene_index, value in enumerate(row):
            genes[gene_index] = value
        data[sample_index] = genes

    return GP_Set(
        input=np.empty(len(rows), dtype=object),
        output=np.empty(len(rows), dtype=object),
        data=data,
        op=[],
        source=[],
        status=list(statuses),
    )


def test_gp_fill_status_classifies_uniform_unique_and_partitioned():
    gp_set = _gp_set(
        [
            [5, "a", 1],
            [5, "b", 1],
            [5, "c", 2],
        ],
        ["NULL", "NULL", "NULL"],
    )

    returned = gp_fill_status(gp_set)

    assert returned is gp_set
    assert gp_set.status == ["unif", "uniq", "part"]


def test_gp_fill_status_accepts_none_as_unresolved():
    gp_set = _gp_set(
        [
            [3],
            [3],
        ],
        [None],
    )

    gp_fill_status(gp_set)

    assert gp_set.status == ["unif"]


def test_list_and_ndarray_with_same_structure_are_equivalent():
    gp_set = _gp_set(
        [
            [[[1, 2], [3, 4]]],
            [np.array([[1, 2], [3, 4]])],
            [(np.array([1, 2]), np.array([3, 4]))],
        ],
        ["NULL"],
    )

    gp_fill_status(gp_set)

    assert gp_set.status == ["unif"]


def test_nested_object_arrays_compare_recursively():
    nested_a = np.empty(2, dtype=object)
    nested_a[0] = np.array([1, 2, 3])

    nested_inner_a = np.empty(2, dtype=object)
    nested_inner_a[0] = np.array([[True, False], [False, True]])
    nested_inner_a[1] = np.array([[8, 0], [0, 8]])
    nested_a[1] = nested_inner_a

    nested_b = [
        [1, 2, 3],
        [
            [[True, False], [False, True]],
            [[8, 0], [0, 8]],
        ],
    ]

    gp_set = _gp_set(
        [
            [nested_a],
            [nested_b],
        ],
        ["null"],
    )

    gp_fill_status(gp_set)

    assert gp_set.status == ["unif"]


def test_partition_means_some_but_not_all_values_are_equivalent():
    gp_set = _gp_set(
        [
            [np.array([1, 2])],
            [[1, 2]],
            [np.array([9, 9])],
            [[9, 9]],
        ],
        ["NULL"],
    )

    gp_fill_status(gp_set)

    assert gp_set.status == ["part"]


def test_existing_non_null_status_is_not_recomputed():
    gp_set = _gp_set(
        [
            [1],
            [2],
            [3],
        ],
        ["unif"],
    )

    gp_fill_status(gp_set)

    assert gp_set.status == ["unif"]


def test_missing_gene_for_any_sample_raises():
    gp_set = _gp_set(
        [
            [1, 2],
            [1],
        ],
        ["NULL", "NULL"],
    )

    with pytest.raises(ValueError, match="does not contain gene index 1"):
        gp_fill_status(gp_set)



def test_init_gp_mat_attaches_output_evaluation_tree():
    train = [
        SimpleNamespace(
            input=np.array(
                [
                    [0, 1, 1],
                    [0, 0, 1],
                ]
            ),
            output=np.array(
                [
                    [0, 1, 1],
                    [0, 0, 1],
                ]
            ),
        ),
        SimpleNamespace(
            input=np.array(
                [
                    [2, 0],
                    [2, 2],
                    [0, 0],
                ]
            ),
            output=np.array(
                [
                    [2, 0],
                    [2, 2],
                    [0, 0],
                ]
            ),
        ),
    ]

    gp_set = init_gp_mat(train)
    tree = gp_set.eval_tree

    assert isinstance(tree, GP_EvalTree)
    assert tree.sample_count == 2

    assert set(tree.root.children) == {"shape", "composite"}
    assert set(tree.shape.children) == {"h", "w"}
    assert set(tree.composite.children) == {
        "color_id",
        "color_presence",
    }


def test_gp_eval_tree_stores_exact_output_targets_for_every_sample():
    outputs = [
        np.array(
            [
                [0, 1, 1],
                [0, 0, 1],
            ]
        ),
        np.array(
            [
                [2, 0],
                [2, 2],
                [0, 0],
            ]
        ),
    ]

    tree = init_gp_eval_tree(outputs)

    assert np.array_equal(tree.root.target[0], outputs[0])
    assert np.array_equal(tree.root.target[1], outputs[1])

    assert np.array_equal(tree.shape.target[0], np.array([2, 3]))
    assert np.array_equal(tree.shape.target[1], np.array([3, 2]))

    assert tree.h.target.tolist() == [2, 3]
    assert tree.w.target.tolist() == [3, 2]

    assert np.array_equal(tree.color_id.target[0], np.array([0, 1]))
    assert np.array_equal(tree.color_id.target[1], np.array([0, 2]))

    assert tree.color_presence.target[0].shape == (2, 2, 3)
    assert tree.color_presence.target[1].shape == (2, 3, 2)

    composite_0 = tree.composite.target[0]
    assert np.array_equal(composite_0[0], tree.color_id.target[0])
    assert np.array_equal(
        composite_0[1],
        tree.color_presence.target[0],
    )


def test_eval_tree_nodes_start_unanswered_and_unnumbered():
    tree = init_gp_eval_tree(
        [
            np.array([[0, 1]]),
            np.array([[2], [2]]),
        ]
    )

    for node in tree.walk():
        assert isinstance(node, GP_EvalNode)
        assert node.answer_present is False
        assert node.gene_idx == -1
        assert node.solution_number == -1
        assert node.terminal is False


def test_shape_identity_gene_can_be_recorded_as_first_terminal_solution():
    train = [
        SimpleNamespace(
            input=np.array(
                [
                    [0, 1, 1],
                    [0, 0, 1],
                ]
            ),
            output=np.array(
                [
                    [0, 1, 1],
                    [0, 0, 1],
                ]
            ),
        ),
        SimpleNamespace(
            input=np.array(
                [
                    [2, 0],
                    [2, 2],
                    [0, 0],
                ]
            ),
            output=np.array(
                [
                    [2, 0],
                    [2, 2],
                    [0, 0],
                ]
            ),
        ),
    ]

    gp_set = init_gp_mat(train)
    tree = gp_set.eval_tree
    shape_node = tree.shape

    for sample_idx in range(len(train)):
        assert np.array_equal(
            gp_set.data[sample_idx][1],
            shape_node.target[sample_idx],
        )

    tree.mark_terminal_solution("shape", gene_idx=1)

    assert shape_node.answer_present is True
    assert shape_node.gene_idx == 1
    assert shape_node.solution_number == 1
    assert shape_node.terminal is True
    assert tree.next_solution_number == 2
    assert tree.terminal_nodes == (shape_node,)


def test_terminal_branch_is_not_searchable_but_parent_remains_available():
    tree = init_gp_eval_tree([np.array([[0, 1], [1, 0]])])

    tree.mark_terminal_solution("shape", gene_idx=4)

    searchable = set(node.name for node in tree.searchable_nodes())

    assert "root" in searchable
    assert "composite" in searchable

    assert "shape" not in searchable
    assert "h" not in searchable
    assert "w" not in searchable


def test_solution_numbers_are_global_and_chronological():
    tree = init_gp_eval_tree(
        [
            np.array([[0, 1], [0, 1]]),
            np.array([[0, 2], [0, 2]]),
        ]
    )

    first = tree.mark_terminal_solution("shape", gene_idx=1)
    second = tree.extract_color_terminal(0, gene_idx=8)

    assert first.solution_number == 1
    assert second.solution_number == 2
    assert [node.solution_number for node in tree.terminal_nodes] == [1, 2]
    assert tree.next_solution_number == 3


def test_color_subset_is_pulled_from_both_pools_and_retained_as_terminal():
    outputs = [
        np.array(
            [
                [0, 1, 1],
                [0, 0, 1],
            ]
        ),
        np.array(
            [
                [0, 2],
                [2, 0],
            ]
        ),
    ]

    tree = init_gp_eval_tree(outputs)

    original_ids = [
        np.asarray(value).copy()
        for value in tree.color_id.target
    ]
    original_presence = [
        np.asarray(value).copy()
        for value in tree.color_presence.target
    ]

    terminal = tree.extract_color_terminal(0, gene_idx=6)

    assert terminal.name == "color_0"
    assert terminal.answer_present is True
    assert terminal.terminal is True
    assert terminal.solution_number == 1
    assert terminal.gene_idx == 6
    assert terminal.source_nodes == ("color_id", "color_presence")

    for sample_idx in range(tree.sample_count):
        subset = terminal.target[sample_idx]

        assert np.array_equal(subset[0], np.array([0]))
        assert subset[1].shape[0] == 1

        remaining_ids = np.asarray(
            tree.color_id.remaining_target[sample_idx]
        )
        assert 0 not in remaining_ids

        remaining_presence = np.asarray(
            tree.color_presence.remaining_target[sample_idx]
        )
        assert len(remaining_ids) == remaining_presence.shape[0]

        # Full targets remain unchanged so composite/full-pool tests still work.
        assert np.array_equal(
            tree.color_id.target[sample_idx],
            original_ids[sample_idx],
        )
        assert np.array_equal(
            tree.color_presence.target[sample_idx],
            original_presence[sample_idx],
        )

    assert tree.composite.terminal is False
    assert "composite" in {
        node.name for node in tree.searchable_nodes()
    }


def test_extracted_color_cannot_be_solved_again():
    tree = init_gp_eval_tree(
        [
            np.array([[0, 1]]),
            np.array([[0, 2]]),
        ]
    )

    tree.extract_color_terminal(0, gene_idx=5)

    with pytest.raises(
        ValueError,
        match="already contains child 'color_0'|not available",
    ):
        tree.extract_color_terminal(0, gene_idx=7)


def test_color_subset_preserves_absence_across_samples():
    outputs = [
        np.array([[0, 1]]),
        np.array([[2, 2]]),
    ]

    tree = init_gp_eval_tree(outputs)
    terminal = tree.extract_color_terminal(0, gene_idx=9)

    sample_0 = terminal.target[0]
    sample_1 = terminal.target[1]

    assert np.array_equal(sample_0[0], np.array([0]))
    assert sample_0[1].shape == (1, 1, 2)

    assert sample_1[0].size == 0
    assert sample_1[1].shape == (0, 1, 2)


def test_generic_subset_can_be_retained_without_closing_parent():
    tree = init_gp_eval_tree(
        [
            np.array([[0, 1]]),
            np.array([[0, 2]]),
        ]
    )

    target = np.empty(2, dtype=object)
    target[0] = "subset-a"
    target[1] = "subset-b"

    terminal = tree.add_terminal_subset(
        "composite",
        name="custom_subset",
        target=target,
        gene_idx=11,
        source_nodes=("future_pool",),
    )

    assert terminal.solution_number == 1
    assert terminal.terminal
    assert terminal.gene_idx == 11
    assert terminal.source_nodes == ("future_pool",)

    assert tree.composite.terminal is False
    assert "composite" in {
        node.name for node in tree.searchable_nodes()
    }
    assert "custom_subset" not in {
        node.name for node in tree.searchable_nodes()
    }


def test_marking_same_terminal_twice_is_rejected():
    tree = init_gp_eval_tree([np.array([[0]])])

    tree.mark_terminal_solution("shape", gene_idx=1)

    with pytest.raises(ValueError, match="already terminal"):
        tree.mark_terminal_solution("shape", gene_idx=2)


def test_parent_can_be_solved_after_child_subset():
    tree = init_gp_eval_tree(
        [
            np.array([[0, 1]]),
            np.array([[0, 2]]),
        ]
    )

    color_terminal = tree.extract_color_terminal(0, gene_idx=4)
    composite = tree.mark_terminal_solution("composite", gene_idx=12)

    assert color_terminal.solution_number == 1
    assert composite.solution_number == 2
    assert composite.terminal

    searchable = set(node.name for node in tree.searchable_nodes())

    assert "root" in searchable
    assert "composite" not in searchable
    assert "color_id" not in searchable
    assert "color_presence" not in searchable


def test_clear_answers_restores_pools_history_and_solution_counter():
    outputs = [
        np.array([[0, 1]]),
        np.array([[0, 2]]),
    ]

    tree = init_gp_eval_tree(outputs)

    tree.mark_terminal_solution("shape", gene_idx=1)
    tree.extract_color_terminal(0, gene_idx=7)

    assert tree.next_solution_number == 3
    assert "color_0" in tree.composite.children

    tree.clear_answers()

    assert tree.next_solution_number == 1
    assert "color_0" not in tree.composite.children
    assert tree.terminal_nodes == ()

    for node in tree.walk():
        assert node.answer_present is False
        assert node.gene_idx == -1
        assert node.solution_number == -1
        assert node.terminal is False

    for sample_idx in range(tree.sample_count):
        assert np.array_equal(
            tree.color_id.remaining_target[sample_idx],
            tree.color_id.target[sample_idx],
        )
        assert np.array_equal(
            tree.color_presence.remaining_target[sample_idx],
            tree.color_presence.target[sample_idx],
        )


def test_pool_targets_are_initialized_as_independent_copies():
    tree = init_gp_eval_tree([np.array([[0, 1]])])

    tree.color_id.remaining_target[0][0] = 9

    assert np.array_equal(
        tree.color_id.target[0],
        np.array([0, 1]),
    )
    assert np.array_equal(
        tree.color_id.remaining_target[0],
        np.array([9, 1]),
    )
