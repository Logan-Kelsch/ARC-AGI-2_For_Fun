from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .data import ArcTask, Grid

ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
]
ARC_CMAP = ListedColormap(ARC_COLORS)


def plot_grid(grid: Grid, title: str = "ARC grid", ax=None):
    """Render one ARC grid and return its matplotlib axis."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    _draw(ax, grid, title)
    plt.tight_layout()
    return ax


def _draw(ax, grid: Grid, title: str) -> None:
    data = np.asarray(grid)
    ax.imshow(data, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", linewidth=0.5)


def plot_task(task: ArcTask) -> None:
    columns = max(len(task.train) + len(task.test_inputs), 1)
    fig, axes = plt.subplots(2, columns, figsize=(3 * columns, 6), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    for index, pair in enumerate(task.train):
        axes[0, index].axis("on")
        _draw(axes[0, index], pair.input, f"train {index + 1} input")
        axes[1, index].axis("on")
        _draw(axes[1, index], pair.output, f"train {index + 1} output")

    offset = len(task.train)
    for index, grid in enumerate(task.test_inputs):
        col = offset + index
        axes[0, col].axis("on")
        _draw(axes[0, col], grid, f"test {index + 1} input")
        if task.test_outputs is not None:
            axes[1, col].axis("on")
            _draw(axes[1, col], task.test_outputs[index], f"test {index + 1} output")

    fig.suptitle(task.task_id)
    plt.tight_layout()
