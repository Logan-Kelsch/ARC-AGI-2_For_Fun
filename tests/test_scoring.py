from arc_agi2_fun.data import ArcTask, Demonstration
from arc_agi2_fun.scoring import AttemptPair, cell_accuracy, grid_exact, score_task


def test_grid_exact_and_cell_accuracy():
    assert grid_exact([[1, 2]], [[1, 2]])
    assert not grid_exact([[1, 2]], [[1, 3]])
    assert cell_accuracy([[1, 2]], [[1, 3]]) == 0.5
    assert cell_accuracy([[1]], [[1, 2]]) == 0.0


def test_pass_at_two_scoring():
    task = ArcTask(
        "toy",
        (Demonstration([[1]], [[2]]),),
        ([[3]],),
        ([[4]],),
    )
    result = score_task(task, [AttemptPair([[0]], [[4]])])
    assert result.solved_outputs == 1
    assert result.total_outputs == 1
