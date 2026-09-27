from arc_agi2_fun.data import ArcTask, Demonstration
from arc_agi2_fun.search import rank_programs


def test_identity_ranks_first_for_identity_task():
    task = ArcTask(
        "identity",
        (
            Demonstration([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
            Demonstration([[5]], [[5]]),
        ),
        ([[7]],),
    )
    ranked = rank_programs(task)
    assert ranked[0].program.name == "identity"
    assert ranked[0].exact_pairs == 2
