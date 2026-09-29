import numpy as np
import pytest

from notebooks.ops.grid_ops import grid_dissection


def test_grid_dissection_returns_expected_structure():
    grid = [
        [0, 1, 1],
        [2, 0, 2],
    ]

    result = grid_dissection(grid)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)

    np.testing.assert_array_equal(result[0], np.array([2, 3]))
    np.testing.assert_array_equal(result[1][0], np.array([0, 1, 2]))

    presence = result[1][1]
    assert presence.shape == (3, 2, 3)
    assert presence.dtype == bool

    np.testing.assert_array_equal(
        presence[0],
        np.array(
            [
                [True, False, False],
                [False, True, False],
            ]
        ),
    )
    np.testing.assert_array_equal(
        presence[1],
        np.array(
            [
                [False, True, True],
                [False, False, False],
            ]
        ),
    )
    np.testing.assert_array_equal(
        presence[2],
        np.array(
            [
                [False, False, False],
                [True, False, True],
            ]
        ),
    )


def test_grid_dissection_sorts_noncontiguous_color_indices():
    result = grid_dissection([[8, 2], [8, 2]])

    np.testing.assert_array_equal(result[1][0], np.array([2, 8]))


def test_grid_dissection_accepts_numpy_input():
    grid = np.array([[3, 3], [0, 3]], dtype=np.int8)

    result = grid_dissection(grid)

    np.testing.assert_array_equal(result[0], np.array([2, 2]))
    np.testing.assert_array_equal(result[1][0], np.array([0, 3]))


def test_grid_dissection_rejects_non_2d_input():
    with pytest.raises(ValueError):
        grid_dissection([0, 1, 2])
