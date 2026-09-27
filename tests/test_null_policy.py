import pytest

arcengine = pytest.importorskip("arcengine")
from arcengine import FrameData, GameAction, GameState

from arc_fun.policy import NullPolicy


def frame(state, actions=None):
    return FrameData(
        frame=[[[0 for _ in range(5)] for _ in range(5)]],
        state=state,
        levels_completed=0,
        available_actions=actions or [],
    )


def test_null_policy_resets_unplayed_game():
    policy = NullPolicy()
    decision = policy.choose_action([], frame(GameState.NOT_PLAYED))
    assert decision.action is GameAction.RESET


def test_null_policy_is_deterministic():
    policy = NullPolicy()
    latest = frame(GameState.NOT_FINISHED)
    a = policy.choose_action([], latest).action
    b = policy.choose_action([], latest).action
    assert a is b
