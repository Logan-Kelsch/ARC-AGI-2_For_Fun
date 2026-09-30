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


def test_eval_tree_nodes_start_unanswered_with_no_gene_index():
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


def test_shape_identity_gene_can_later_be_associated_exactly_across_samples():
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
    shape_node = gp_set.eval_tree.shape

    # Gene index 1 is currently the input-shape extraction gene.
    # The future evaluator should be able to establish this exact equality
    # across all samples. This test only proves the representations align.
    for sample_idx in range(len(train)):
        assert np.array_equal(
            gp_set.data[sample_idx][1],
            shape_node.target[sample_idx],
        )

    assert shape_node.answer_present is False
    assert shape_node.gene_idx == -1

    shape_node.mark_answer(1)

    assert shape_node.answer_present is True
    assert shape_node.gene_idx == 1


def test_mark_and_clear_answer_only_manage_retained_bookkeeping():
    node = GP_EvalNode(
        name="example",
        target=np.array([1, 2], dtype=object),
    )

    with pytest.raises(ValueError):
        node.mark_answer(-1)

    node.mark_answer(4)

    assert node.answer_present is True
    assert node.gene_idx == 4

    node.clear_answer()

    assert node.answer_present is False
    assert node.gene_idx == -1


def test_clear_answers_resets_entire_eval_tree():
    tree = init_gp_eval_tree([np.array([[0]])])

    tree.root.mark_answer(0)
    tree.shape.mark_answer(1)
    tree.color_presence.mark_answer(3)

    tree.clear_answers()

    assert all(
        not node.answer_present and node.gene_idx == -1
        for node in tree.walk()
    )
