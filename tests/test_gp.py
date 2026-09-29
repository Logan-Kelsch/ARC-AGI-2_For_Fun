import numpy as np
import pytest

from notebooks.ops.GP import GP_Set, gp_fill_status


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
