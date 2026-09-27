from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from arcengine import FrameData, GameAction, GameState

from .observations import center_coordinate


@dataclass(frozen=True)
class ActionDecision:
    action: GameAction
    reasoning: dict[str, Any]


class Policy(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> ActionDecision:
        ...


def _available_actions(latest_frame: FrameData) -> list[GameAction]:
    """Prefer environment-advertised actions, falling back to the enum."""
    raw = getattr(latest_frame, "available_actions", None)
    if raw:
        actions: list[GameAction] = []
        for value in raw:
            if isinstance(value, GameAction):
                actions.append(value)
                continue
            try:
                actions.append(GameAction.from_id(int(value)))
            except (TypeError, ValueError):
                continue
        if actions:
            return actions
    return list(GameAction)


class NullPolicy:
    """Deterministic legal-action baseline with no game-specific knowledge."""

    name = "null_policy_v0"

    def reset(self) -> None:
        return None

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> ActionDecision:
        available = _available_actions(latest_frame)

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            action = GameAction.RESET
            return ActionDecision(action, {"policy": self.name, "reason": "reset"})

        candidates = [a for a in available if a is not GameAction.RESET]
        if not candidates:
            candidates = [a for a in GameAction if a is not GameAction.RESET]
        if not candidates:
            action = GameAction.RESET
        else:
            action = sorted(candidates, key=lambda a: a.name)[0]

        if action.is_complex():
            action.set_data(center_coordinate(latest_frame))

        return ActionDecision(
            action=action,
            reasoning={
                "policy": self.name,
                "reason": "deterministic_first_available_action",
            },
        )
