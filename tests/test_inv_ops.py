import numpy as np
import pytest

from notebooks.ops.inv_ops import (
    INV_OP_REGISTRY,
    inv_bool2_intersect,
    inv_bool2_union,
    inv_bool_cavity,
    inv_bool_complement,
    inv_bool_sum,
    inv_dim0_flip,
    inv_dim1_flip,
    inv_dim2_flip,
    inv_mat2_cwrotate,
    inv_partition_bool_trim,
    inv_partition_composite,
    inv_partition_shape,
)
from notebooks.ops.ops import OP_REGISTRY


def test_every_builtin_operation_has_registered_inverse():
    for name, info in OP_REGISTRY.items():
        assert info.inverse_op is not None, name
        assert info.inverse_op in INV_OP_REGISTRY, name


def test_inverse_registry_distinguishes_reconstructive_and_null_relations():
    assert INV_OP_REGISTRY["inv_partition_shape"].reconstructive
    assert INV_OP_REGISTRY["inv_partition_composite"].reconstructive
    assert INV_OP_REGISTRY["inv_bool_complement"].reconstructive
    assert INV_OP_REGISTRY["inv_mat2_cwrotate"].reconstructive

    assert not INV_OP_REGISTRY["inv_bool_sum"].reconstructive
    assert not INV_OP_REGISTRY["inv_bool_cavity"].reconstructive
    assert not INV_OP_REGISTRY["inv_bool2_union"].reconstructive
    assert not INV_OP_REGISTRY["inv_bool2_intersect"].reconstructive


def test_inv_partition_composite_reassembles_color_pairs():
    color0 = np.int64(0)
    mask0 = np.array(
        [
            [True, False],
            [False, True],
        ],
        dtype=bool,
    )

    color3 = np.int64(3)
    mask3 = np.array(
        [
            [False, True],
            [True, False],
        ],
        dtype=bool,
    )

    result = inv_partition_composite(
        color0,
        mask0,
        color3,
        mask3,
    )

    assert result.dtype == np.int64
    assert np.array_equal(
        result,
        np.array(
            [
                [0, 3],
                [3, 0],
            ],
            dtype=np.int64,
        ),
    )


def test_inv_partition_composite_requires_exact_cover():
    with pytest.raises(ValueError, match="exactly once"):
        inv_partition_composite(
            np.int64(0),
            np.array([[True, False]], dtype=bool),
            np.int64(1),
            np.array([[False, False]], dtype=bool),
        )


def test_inv_partition_shape_validates_reconstructed_composite():
    composite = np.array(
        [
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=np.int64,
    )

    result = inv_partition_shape(
        np.array([2, 3], dtype=np.int64),
        composite,
    )

    assert np.array_equal(result, composite)

    with pytest.raises(ValueError, match="does not match"):
        inv_partition_shape(
            np.array([3, 2], dtype=np.int64),
            composite,
        )


def test_reversible_unary_inverses_restore_sources():
    bool_source = np.array(
        [
            [True, False],
            [False, True],
        ],
        dtype=bool,
    )
    assert np.array_equal(
        inv_bool_complement(~bool_source),
        bool_source,
    )

    matrix = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.int64,
    )
    clockwise = np.rot90(matrix, k=-1)
    assert np.array_equal(
        inv_mat2_cwrotate(clockwise),
        matrix,
    )

    cube = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    assert np.array_equal(
        inv_dim0_flip(np.flip(cube, axis=0)),
        cube,
    )
    assert np.array_equal(
        inv_dim1_flip(np.flip(cube, axis=1)),
        cube,
    )
    assert np.array_equal(
        inv_dim2_flip(np.flip(cube, axis=2)),
        cube,
    )


def test_inv_partition_bool_trim_reconstructs_when_shape_is_known():
    trimmed = np.array(
        [
            [True, False],
            [True, True],
        ],
        dtype=bool,
    )

    result = inv_partition_bool_trim(
        np.array([1, 2], dtype=np.int64),
        trimmed,
        np.array([4, 5], dtype=np.int64),
    )

    expected = np.zeros((4, 5), dtype=bool)
    expected[1:3, 2:4] = trimmed

    assert np.array_equal(result, expected)


def test_null_inverse_helpers_verify_relations_but_do_not_create_st_proofs():
    source = np.array([True, False, True], dtype=bool)

    assert np.array_equal(
        inv_bool_sum(np.int64(2), source),
        source,
    )

    with pytest.raises(ValueError, match="bool_sum"):
        inv_bool_sum(np.int64(1), source)

    left = np.array([True, False], dtype=bool)
    right = np.array([False, True], dtype=bool)

    union = np.logical_or(left, right)
    intersect = np.logical_and(left, right)

    recovered_left, recovered_right = inv_bool2_union(
        union,
        left,
        right,
    )
    assert np.array_equal(recovered_left, left)
    assert np.array_equal(recovered_right, right)

    recovered_left, recovered_right = inv_bool2_intersect(
        intersect,
        left,
        right,
    )
    assert np.array_equal(recovered_left, left)
    assert np.array_equal(recovered_right, right)


def test_inv_bool_cavity_relation_checker():
    source = np.array(
        [
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ],
        dtype=bool,
    )
    cavity = np.array(
        [
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ],
        dtype=bool,
    )

    recovered = inv_bool_cavity(cavity, source)

    assert np.array_equal(recovered, source)
