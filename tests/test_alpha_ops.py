import numpy as np
import pytest

from notebooks.ops.alpha_ops import DEFAULT_ABS_KERNELS, survey_abs_kernels


def test_default_kernel_bank_contains_requested_shapes():
    assert list(DEFAULT_ABS_KERNELS) == [
        "self",
        "left",
        "right",
        "up",
        "down",
        "lower_left",
        "upper_left",
        "upper_right",
        "lower_right",
        "star_3",
        "full_3x3",
    ]


def test_self_kernel_returns_absolute_scalar_magnitude():
    matrix = np.array([[-2.0, 3.0], [0.0, -4.0]])

    response = survey_abs_kernels(matrix, kernel="self")

    np.testing.assert_allclose(
        response,
        np.array([[2.0, 3.0], [0.0, 4.0]]),
    )


def test_left_and_right_kernels_measure_directional_absolute_difference():
    matrix = np.array([[1.0, 3.0, 6.0]])

    left = survey_abs_kernels(matrix, kernel="left")
    right = survey_abs_kernels(matrix, kernel="right")

    np.testing.assert_allclose(left, np.array([[0.0, 2.0, 3.0]]))
    np.testing.assert_allclose(right, np.array([[2.0, 3.0, 0.0]]))


def test_custom_kernel_matches_equivalent_named_kernel():
    matrix = np.array([[1.0, 3.0, 6.0]])

    named = survey_abs_kernels(matrix, kernel="left")
    custom = survey_abs_kernels(matrix, kernel=[1, 1, 0])

    np.testing.assert_allclose(custom, named)


def test_all_returns_one_response_per_default_kernel():
    matrix = np.arange(9, dtype=float).reshape(3, 3)

    responses = survey_abs_kernels(matrix, kernel="all")

    assert list(responses) == list(DEFAULT_ABS_KERNELS)
    assert all(response.shape == (3, 3) for response in responses.values())


def test_one_hot_tensor_is_reduced_to_scalar_response_map():
    tensor = np.eye(10, dtype=float)[np.array([[1, 1, 2]])]

    response = survey_abs_kernels(tensor, kernel="left")

    assert response.shape == (1, 3)
    np.testing.assert_allclose(response, np.array([[0.0, 0.0, 0.2]]))


def test_even_kernel_dimensions_are_rejected():
    with pytest.raises(ValueError):
        survey_abs_kernels(np.ones((3, 3)), kernel=[[1, 1], [1, 1]])
