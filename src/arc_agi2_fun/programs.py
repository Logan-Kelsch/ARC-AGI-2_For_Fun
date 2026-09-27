from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .data import Grid


@dataclass(frozen=True)
class GridProgram:
    name: str
    transform: Callable[[np.ndarray], np.ndarray]
    complexity: int = 1

    def apply(self, grid: Grid) -> Grid:
        array = np.asarray(grid, dtype=np.int8)
        out = self.transform(array)
        return np.asarray(out, dtype=np.int8).tolist()


def _identity(x: np.ndarray) -> np.ndarray:
    return x.copy()


PROGRAM_LIBRARY: tuple[GridProgram, ...] = (
    GridProgram("identity", _identity, 0),
    GridProgram("flip_horizontal", np.fliplr, 1),
    GridProgram("flip_vertical", np.flipud, 1),
    GridProgram("rotate_90", lambda x: np.rot90(x, 1), 1),
    GridProgram("rotate_180", lambda x: np.rot90(x, 2), 1),
    GridProgram("rotate_270", lambda x: np.rot90(x, 3), 1),
    GridProgram("transpose", np.transpose, 1),
)
