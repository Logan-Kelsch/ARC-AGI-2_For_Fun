from __future__ import annotations

from typing import Any

import numpy as np


def first_grid(frame: Any) -> np.ndarray | None:
    """Return the first ARC grid as a NumPy array when one is present."""
    raw = getattr(frame, "frame", None)
    if raw is None:
        return None

    arr = np.asarray(raw)
    if arr.size == 0:
        return None
    if arr.ndim == 3:
        return arr[0]
    if arr.ndim == 2:
        return arr
    return None


def center_coordinate(frame: Any) -> dict[str, int]:
    """Choose a valid-looking center coordinate for a complex action."""
    grid = first_grid(frame)
    if grid is None:
        return {"x": 32, "y": 32}

    height, width = grid.shape[-2:]
    return {
        "x": int(max(0, min(63, (width - 1) // 2))),
        "y": int(max(0, min(63, (height - 1) // 2))),
    }


def summarize_frame(frame: Any) -> dict[str, Any]:
    """Small stable summary used for logs and future evidence records."""
    grid = first_grid(frame)
    state = getattr(frame, "state", None)
    out: dict[str, Any] = {
        "state": getattr(state, "name", str(state)),
        "levels_completed": int(getattr(frame, "levels_completed", 0) or 0),
        "height": None,
        "width": None,
        "values": None,
    }
    if grid is not None:
        out["height"], out["width"] = map(int, grid.shape[-2:])
        out["values"] = [int(v) for v in np.unique(grid)]
    return out
