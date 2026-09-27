from arc_agi2_fun.data import ArcTask, Demonstration


def test_task_dataclasses_hold_grids():
    task = ArcTask(
        task_id="toy",
        train=(Demonstration([[1]], [[2]]),),
        test_inputs=([[3]],),
        test_outputs=([[4]],),
    )
    assert task.train[0].input == [[1]]
    assert task.test_outputs == ([[4]],)
