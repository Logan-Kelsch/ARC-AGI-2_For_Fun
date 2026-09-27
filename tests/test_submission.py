from arc_agi2_fun.data import ArcTask, Demonstration
from arc_agi2_fun.solver import NullSolver
from arc_agi2_fun.submission import build_submission


def test_submission_has_two_attempts_per_test_input():
    task = ArcTask(
        "toy",
        (Demonstration([[1]], [[1]]),),
        ([[2]], [[3]]),
    )
    submission = build_submission({"toy": task}, NullSolver())
    assert len(submission["toy"]) == 2
    assert set(submission["toy"][0]) == {"attempt_1", "attempt_2"}
