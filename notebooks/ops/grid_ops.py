from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def grid_dissection(grid: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    """Dissect a 2D categorical grid into shape, colors, and per-color presence.

    Returns
    -------
    np.ndarray
        Object array with exactly two top-level entries:

        result[0]
            1D ndarray: [height, width]

        result[1]
            Object array with two entries:
              result[1][0] -> sorted 1D array of used color indices
              result[1][1] -> boolean ndarray of shape
                              (num_colors, height, width)

        Presence map i corresponds to color result[1][0][i].
    """
    array = np.asarray(grid)

    if array.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("grid dimensions may not be empty.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("grid values must be numeric color indices.")
    if not np.all(np.isfinite(array)):
        raise ValueError("grid contains NaN or infinite values.")
    if not np.all(array == np.floor(array)):
        raise ValueError("grid color indices must be integers.")

    array = array.astype(int, copy=False)

    shape = np.asarray(array.shape, dtype=int)
    colors = np.unique(array)
    presence = np.stack(
        [(array == color) for color in colors],
        axis=0,
    ).astype(bool, copy=False)

    color_info = np.empty(2, dtype=object)
    color_info[0] = colors
    color_info[1] = presence

    result = np.empty(2, dtype=object)
    result[0] = shape
    result[1] = color_info

    return result
