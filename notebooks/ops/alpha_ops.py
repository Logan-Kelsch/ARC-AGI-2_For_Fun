from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from arc_agi2_fun.visualize import ARC_COLORS

KernelLike: TypeAlias = Sequence[float] | Sequence[Sequence[float]] | np.ndarray

DEFAULT_ABS_KERNELS: dict[str, np.ndarray] = {
    "self": np.array([1], dtype=float),
    "left": np.array([1, 1, 0], dtype=float),
    "right": np.array([0, 1, 1], dtype=float),
    "up": np.array([[1], [1], [0]], dtype=float),
    "down": np.array([[0], [1], [1]], dtype=float),
    "lower_left": np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=float),
    "upper_left": np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=float),
    "upper_right": np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=float),
    "lower_right": np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=float),
    "star_3": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=float),
    "full_3x3": np.ones((3, 3), dtype=float),
}


def _normalize_kernel(kernel: KernelLike) -> np.ndarray:
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


def _categorical_grid_to_channels(
    grid: np.ndarray | Sequence,
) -> dict[int, np.ndarray]:
    """Split an ARC HxW grid into binary channels without treating colors numerically."""
    array = np.asarray(grid)
    if array.ndim != 2:
        raise ValueError(f"categorical ARC input must be HxW, got {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("grid dimensions may not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError("grid contains NaN or infinite values.")
    if not np.all(array == np.floor(array)):
        raise ValueError("ARC grid values are categorical integers, not continuous.")
    if np.any((array < 0) | (array > 9)):
        raise ValueError("ARC colors must be integers from 0 to 9.")

    categorical = array.astype(np.int8)
    return {
        int(color): (categorical == color).astype(float)
        for color in np.unique(categorical)
    }


def _tensor_to_channels(tensor: np.ndarray) -> dict[int, np.ndarray]:
    """Interpret HxWx10 as already-separated color intensity channels."""
    array = np.asarray(tensor, dtype=float)
    if array.ndim != 3 or array.shape[-1] != 10:
        raise ValueError("3D input must be HxWx10, one channel per ARC color.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("tensor spatial dimensions may not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError("tensor contains NaN or infinite values.")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("color-channel intensities must be in [0, 1].")

    return {
        color: array[..., color].copy()
        for color in range(10)
        if np.any(array[..., color] > 0)
    }


def _resolve_color_channels(
    matrix: np.ndarray | Sequence,
) -> dict[int, np.ndarray]:
    array = np.asarray(matrix)
    if array.ndim == 2:
        return _categorical_grid_to_channels(array)
    if array.ndim == 3:
        return _tensor_to_channels(array)
    raise ValueError(
        "matrix must be a categorical HxW ARC grid or HxWx10 color-intensity tensor."
    )


def _kernel_intensity_response(
    channel: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """Sum literal kernel intensity contributions and saturate at 1."""
    height, width = channel.shape
    kh, kw = kernel.shape
    cy, cx = kh // 2, kw // 2

    padded = np.pad(
        channel,
        ((cy, cy), (cx, cx)),
        mode="constant",
        constant_values=0.0,
    )

    response = np.zeros((height, width), dtype=float)
    for ky in range(kh):
        for kx in range(kw):
            weight = abs(float(kernel[ky, kx]))
            if weight == 0:
                continue
            sampled = padded[ky : ky + height, kx : kx + width]
            response += weight * sampled

    return np.clip(response, 0.0, 1.0)


def _evaluate_kernel(
    channels: dict[int, np.ndarray],
    kernel: np.ndarray,
) -> dict[int, np.ndarray]:
    return {
        color: _kernel_intensity_response(channel, kernel)
        for color, channel in channels.items()
    }


def _color_intensity_cmap(color: int) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"arc_color_{color}_intensity",
        ["#000000", ARC_COLORS[color]],
    )


def _plot_color_responses(
    responses: dict[int, np.ndarray],
    *,
    kernel_name: str,
) -> None:
    colors = list(responses)
    columns = min(5, max(1, len(colors)))
    rows = int(np.ceil(len(colors) / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.5 * columns, 3.5 * rows),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, color in zip(axes.flat, colors):
        ax.axis("on")
        ax.imshow(
            responses[color],
            cmap=_color_intensity_cmap(color),
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        ax.set_title(f"color {color} | {kernel_name}")
        ax.set_xticks([])
        ax.set_yticks([])

    figure.suptitle(
        f"survey_abs_kernels: {kernel_name} "
        "(0 = no intensity, 1 = full intensity)"
    )
    plt.tight_layout()
    plt.show()


def _plot_all_responses(
    responses: dict[str, dict[int, np.ndarray]],
) -> None:
    for kernel_name, color_responses in responses.items():
        _plot_color_responses(color_responses, kernel_name=kernel_name)


def survey_abs_kernels(
    matrix: np.ndarray | Sequence,
    kernel: str | KernelLike = "all",
    *,
    plot: bool = False,
) -> dict[str, dict[int, np.ndarray]] | dict[int, np.ndarray]:
    """Survey categorical ARC color channels with additive intensity kernels.

    ARC grid values are categorical labels. A value of 8 means color 8, not a
    larger magnitude than color 2.

    For a normal HxW grid, each detected color becomes a binary intensity
    channel. Kernel values then contribute literally to that channel. There is
    no normalization or averaging by kernel size/weight.

    Example for kernel [0, 1, 1]:
        two full-intensity selected pixels -> 1 + 1 = 2 -> clipped to 1.

    Returns
    -------
    kernel="all":
        dict[kernel_name][color] -> HxW intensity map

    one kernel:
        dict[color] -> HxW intensity map
    """
    channels = _resolve_color_channels(matrix)

    if isinstance(kernel, str):
        if kernel == "all":
            responses = {
                name: _evaluate_kernel(channels, _normalize_kernel(mask))
                for name, mask in DEFAULT_ABS_KERNELS.items()
            }
            if plot:
                _plot_all_responses(responses)
            return responses

        try:
            selected_kernel = DEFAULT_ABS_KERNELS[kernel]
        except KeyError as exc:
            choices = ", ".join(["all", *DEFAULT_ABS_KERNELS.keys()])
            raise ValueError(
                f"Unknown kernel name {kernel!r}. Available: {choices}."
            ) from exc

        responses = _evaluate_kernel(channels, _normalize_kernel(selected_kernel))
        if plot:
            _plot_color_responses(responses, kernel_name=kernel)
        return responses

    responses = _evaluate_kernel(channels, _normalize_kernel(kernel))
    if plot:
        _plot_color_responses(responses, kernel_name="custom")
    return responses
