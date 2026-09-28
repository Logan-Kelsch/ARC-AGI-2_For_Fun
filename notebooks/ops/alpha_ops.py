from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

from arc_agi2_fun.visualize import plot_grid

KernelLike: TypeAlias = Sequence[float] | Sequence[Sequence[float]] | np.ndarray

# Every default is centered on the pixel being evaluated.
#
# For the directional masks, the center is the middle entry. The non-center
# 1s select the neighbors whose absolute difference from the center is measured.
#
# The quadrant masks are represented on 3x3 grids so that the anchor remains
# unambiguous and odd-sized.
DEFAULT_ABS_KERNELS: dict[str, np.ndarray] = {
    "self": np.array([1], dtype=float),
    "left": np.array([1, 1, 0], dtype=float),
    "right": np.array([0, 1, 1], dtype=float),
    "up": np.array([[1], [1], [0]], dtype=float),
    "down": np.array([[0], [1], [1]], dtype=float),
    "lower_left": np.array(
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
        ],
        dtype=float,
    ),
    "upper_left": np.array(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=float,
    ),
    "upper_right": np.array(
        [
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
        ],
        dtype=float,
    ),
    "lower_right": np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    ),
    "star_3": np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=float,
    ),
    "full_3x3": np.ones((3, 3), dtype=float),
}


def _normalize_kernel(kernel: KernelLike) -> np.ndarray:
    """Convert a 1D/2D kernel to an odd-sized 2D floating-point array."""
    resolved = np.asarray(kernel, dtype=float)

    if resolved.ndim == 1:
        resolved = resolved.reshape(1, -1)
    elif resolved.ndim != 2:
        raise ValueError(
            f"kernel must be a 1D or 2D list/array, got shape {resolved.shape}."
        )

    if resolved.size == 0:
        raise ValueError("kernel may not be empty.")
    if resolved.shape[0] % 2 == 0 or resolved.shape[1] % 2 == 0:
        raise ValueError(
            "kernel dimensions must be odd so the center pixel is unambiguous."
        )
    if not np.all(np.isfinite(resolved)):
        raise ValueError("kernel contains NaN or infinite values.")
    if not np.any(resolved):
        raise ValueError("kernel must contain at least one non-zero value.")

    return resolved


def _validate_matrix(matrix: np.ndarray | Sequence) -> np.ndarray:
    """Accept either HxW scalar data or HxWxC feature/tensor data."""
    array = np.asarray(matrix, dtype=float)

    if array.ndim not in (2, 3):
        raise ValueError(
            "matrix must be HxW or HxWxC; "
            f"received shape {array.shape}."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("matrix spatial dimensions may not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix contains NaN or infinite values.")

    return array


def _reduce_feature_difference(diff: np.ndarray) -> np.ndarray:
    """Reduce vector-valued pixels to one scalar absolute response per pixel."""
    if diff.ndim == 2:
        return diff
    return np.mean(diff, axis=-1)


def _self_magnitude(matrix: np.ndarray) -> np.ndarray:
    absolute = np.abs(matrix)
    if absolute.ndim == 2:
        return absolute
    return np.mean(absolute, axis=-1)


def _abs_kernel_response(matrix: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Evaluate one center-anchored absolute-difference neighborhood kernel."""
    height, width = matrix.shape[:2]
    kernel_height, kernel_width = kernel.shape
    center_y = kernel_height // 2
    center_x = kernel_width // 2

    non_center_terms: list[tuple[int, int, float]] = []
    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            weight = float(kernel[kernel_y, kernel_x])
            if weight == 0:
                continue

            dy = kernel_y - center_y
            dx = kernel_x - center_x
            if dy == 0 and dx == 0:
                continue

            non_center_terms.append((dy, dx, abs(weight)))

    # A self-only kernel is useful as the zeroth-order survey: the absolute
    # magnitude of the current scalar/vector pixel.
    if not non_center_terms:
        return _self_magnitude(matrix)

    pad_y = center_y
    pad_x = center_x
    if matrix.ndim == 2:
        pad_width = ((pad_y, pad_y), (pad_x, pad_x))
    else:
        pad_width = ((pad_y, pad_y), (pad_x, pad_x), (0, 0))

    # Edge padding avoids manufacturing artificial high responses solely
    # because a pixel sits on the image boundary.
    padded = np.pad(matrix, pad_width, mode="edge")

    response = np.zeros((height, width), dtype=float)
    total_weight = 0.0

    for dy, dx, weight in non_center_terms:
        neighbor = padded[
            pad_y + dy : pad_y + dy + height,
            pad_x + dx : pad_x + dx + width,
            ...,
        ]
        difference = _reduce_feature_difference(np.abs(matrix - neighbor))
        response += weight * difference
        total_weight += weight

    if total_weight == 0:
        return response

    return response / total_weight


def _response_to_plot_grid(
    response: np.ndarray,
    *,
    scale_max: float,
) -> list[list[int]]:
    """Map a floating response map to ARC colors 0..9 for visualization only."""
    if scale_max <= 0:
        return np.zeros(response.shape, dtype=int).tolist()

    visual = np.rint(9.0 * np.clip(response / scale_max, 0.0, 1.0))
    return visual.astype(int).tolist()


def _plot_responses(responses: dict[str, np.ndarray]) -> None:
    """Render raw response maps through the repository's ARC plot_grid helper."""
    import matplotlib.pyplot as plt

    names = list(responses)
    columns = 3
    rows = int(np.ceil(len(names) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 4 * rows),
        squeeze=False,
    )

    global_max = max(
        (float(np.max(response)) for response in responses.values()),
        default=0.0,
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, name in zip(axes.flat, names):
        ax.axis("on")
        response = responses[name]
        plot_grid(
            _response_to_plot_grid(response, scale_max=global_max),
            title=f"{name} | raw max={float(np.max(response)):.4g}",
            ax=ax,
        )

    figure.suptitle(
        "survey_abs_kernels responses "
        "(shared 0-9 visualization scale; returned arrays remain raw)"
    )
    plt.tight_layout()
    plt.show()


def survey_abs_kernels(
    matrix: np.ndarray | Sequence,
    kernel: str | KernelLike = "all",
    *,
    plot: bool = False,
) -> dict[str, np.ndarray] | np.ndarray:
    """Survey local absolute differences with one or many center-anchored kernels.

    Parameters
    ----------
    matrix:
        Either an HxW scalar matrix or an HxWxC feature tensor. This works
        directly with the HxWx10 one-hot ARC tensors from grid_to_tensor().

    kernel:
        - "all": evaluate the complete DEFAULT_ABS_KERNELS bank and return a
          dict mapping kernel name -> HxW response.
        - a default kernel name such as "left" or "full_3x3": evaluate that
          single kernel and return its HxW response.
        - any custom 1D/2D list or NumPy array with odd dimensions.

        Non-zero non-center entries select neighbors. Their absolute numeric
        values act as weights. The center of the kernel is the anchor pixel.

    plot:
        When True, visualize the response(s) with plot_grid. Raw responses are
        scaled together to ARC values 0..9 only for display. Returned arrays
        are never quantized or modified.

    Returns
    -------
    dict[str, np.ndarray] | np.ndarray
        "all" returns every named HxW response in a dict.
        A single named/custom kernel returns one HxW floating response map.

    Notes
    -----
    For every selected neighboring offset j, the scalar response is:

        mean_j( |x_center - x_neighbor_j| )

    weighted by the absolute kernel values.

    For HxWxC tensors, each vector difference is additionally averaged over C.

    The self-only kernel has no neighbor to compare against, so it returns the
    absolute magnitude of the current pixel (mean absolute magnitude for HxWxC).
    """
    array = _validate_matrix(matrix)

    if isinstance(kernel, str):
        if kernel == "all":
            responses = {
                name: _abs_kernel_response(array, _normalize_kernel(mask))
                for name, mask in DEFAULT_ABS_KERNELS.items()
            }
            if plot:
                _plot_responses(responses)
            return responses

        try:
            selected_kernel = DEFAULT_ABS_KERNELS[kernel]
        except KeyError as exc:
            choices = ", ".join(["all", *DEFAULT_ABS_KERNELS.keys()])
            raise ValueError(
                f"Unknown kernel name {kernel!r}. Available: {choices}."
            ) from exc

        response = _abs_kernel_response(
            array,
            _normalize_kernel(selected_kernel),
        )
        if plot:
            _plot_responses({kernel: response})
        return response

    selected_kernel = _normalize_kernel(kernel)
    response = _abs_kernel_response(array, selected_kernel)
    if plot:
        _plot_responses({"custom": response})
    return response
