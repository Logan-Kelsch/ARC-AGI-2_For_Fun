from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def grid_dissection(grid: np.ndarray | Sequence[Sequence[int]], as_seperate:bool=False) -> np.ndarray:
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

    if(as_seperate):
        return result[0], color_info[0], color_info[1]
    else:
        return result

import numpy as np

def reconstruct_grid(dissection_or_colors, presence=None):
    """
    Reconstruct a categorical ARC grid from any of:

    1. Full grid_dissection output:
        reconstruct_grid(dissection)

    2. The second item from grid_dissection:
        reconstruct_grid(dissection[1])

    3. Colors + respective boolean presence matrices:
        reconstruct_grid(colors, presence)
    """

    def is_color_info(x):
        try:
            return (
                len(x) == 2
                and np.asarray(x[0]).ndim == 1
                and np.asarray(x[1]).ndim == 3
            )
        except (TypeError, IndexError):
            return False

    # Case 3: colors and presence supplied separately
    if presence is not None:
        colors = np.asarray(dissection_or_colors)
        presence = np.asarray(presence, dtype=bool)

    # Case 2: dissection[1]
    elif is_color_info(dissection_or_colors):
        colors = np.asarray(dissection_or_colors[0])
        presence = np.asarray(dissection_or_colors[1], dtype=bool)

    # Case 1: full dissection
    elif (
        len(dissection_or_colors) == 2
        and is_color_info(dissection_or_colors[1])
    ):
        colors = np.asarray(dissection_or_colors[1][0])
        presence = np.asarray(dissection_or_colors[1][1], dtype=bool)

    else:
        raise ValueError("Could not interpret reconstruction input.")

    if len(colors) != presence.shape[0]:
        raise ValueError("Each color must have one corresponding presence matrix.")

    # Every cell should belong to exactly one categorical color.
    if not np.all(presence.sum(axis=0) == 1):
        raise ValueError("Every pixel must belong to exactly one color.")

    return (
        presence * colors[:, None, None]
    ).sum(axis=0).astype(int)