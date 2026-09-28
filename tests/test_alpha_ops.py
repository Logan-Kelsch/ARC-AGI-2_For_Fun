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


def test_grid_values_are_categorical_color_channels_not_magnitudes():
    grid = np.array([[8, 8, 2]])

    response = survey_abs_kernels(grid, kernel="self")

    assert set(response) == {2, 8}
    np.testing.assert_allclose(response[8], np.array([[1.0, 1.0, 0.0]]))
    np.testing.assert_allclose(response[2], np.array([[0.0, 0.0, 1.0]]))


def test_directional_kernel_operates_separately_per_color():
    grid = np.array([[8, 8, 2]])

    response = survey_abs_kernels(grid, kernel="right")

    # Color 8: x0 sees self + right = 2 -> clipped to 1.
    # x1 sees self 8 + right color 2 = 1.
    np.testing.assert_allclose(response[8], np.array([[1.0, 1.0, 0.0]]))

    # Color 2 exists only at x2. At x1 the right lookup sees it, and at x2
    # the center itself contributes it.
    np.testing.assert_allclose(response[2], np.array([[0.0, 1.0, 1.0]]))


def test_kernel_contributions_are_added_not_averaged():
    tensor = np.zeros((1, 2, 10), dtype=float)
    tensor[0, 0, 4] = 0.4
    tensor[0, 1, 4] = 0.4

    response = survey_abs_kernels(tensor, kernel="right")

    # At x0, center and right both contribute 0.4:
    # 0.4 + 0.4 = 0.8. An averaging implementation would incorrectly give 0.4.
    assert response[4][0, 0] == pytest.approx(0.8)


def test_kernel_sum_saturates_at_one():
    tensor = np.zeros((1, 2, 10), dtype=float)
    tensor[0, 0, 4] = 0.7
    tensor[0, 1, 4] = 0.7

    response = survey_abs_kernels(tensor, kernel="right")

    assert response[4][0, 0] == pytest.approx(1.0)


def test_custom_kernel_matches_equivalent_named_kernel():
    grid = np.array([[1, 1, 2]])

    named = survey_abs_kernels(grid, kernel="left")
    custom = survey_abs_kernels(grid, kernel=[1, 1, 0])

    assert set(custom) == set(named)
    for color in named:
        np.testing.assert_allclose(custom[color], named[color])


def test_all_returns_kernel_then_color_then_intensity_map():
    grid = np.array([[0, 1], [1, 2]])

    responses = survey_abs_kernels(grid, kernel="all")

    assert list(responses) == list(DEFAULT_ABS_KERNELS)
    for kernel_responses in responses.values():
        assert set(kernel_responses) == {0, 1, 2}
        assert all(response.shape == (2, 2) for response in kernel_responses.values())
        assert all(
            np.all((response >= 0.0) & (response <= 1.0))
            for response in kernel_responses.values()
        )


def test_even_kernel_dimensions_are_rejected():
    with pytest.raises(ValueError):
        survey_abs_kernels(np.ones((3, 3)), kernel=[[1, 1], [1, 1]])


def test_continuous_2d_grid_values_are_rejected_as_invalid_categories():
    with pytest.raises(ValueError):
        survey_abs_kernels(np.array([[0.0, 1.5]]), kernel="self")
