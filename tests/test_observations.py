from types import SimpleNamespace

import numpy as np

from arc_fun.observations import center_coordinate, first_grid, summarize_frame


def test_first_grid_and_center():
    frame = SimpleNamespace(
        frame=[np.zeros((5, 7), dtype=int)],
        state=SimpleNamespace(name="NOT_FINISHED"),
        levels_completed=0,
    )
    grid = first_grid(frame)
    assert grid.shape == (5, 7)
    assert center_coordinate(frame) == {"x": 3, "y": 2}


def test_summary():
    frame = SimpleNamespace(
        frame=[np.array([[0, 1], [1, 2]])],
        state=SimpleNamespace(name="NOT_FINISHED"),
        levels_completed=2,
    )
    assert summarize_frame(frame) == {
        "state": "NOT_FINISHED",
        "levels_completed": 2,
        "height": 2,
        "width": 2,
        "values": [0, 1, 2],
    }
