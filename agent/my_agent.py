"""Thin ARC-AGI-3 framework adapter around the shared policy implementation."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

_repo_src = Path(__file__).resolve().parents[1] / "src"
if _repo_src.exists():
    sys.path.insert(0, str(_repo_src))

from arc_fun.policy import NullPolicy


class MyAgent(Agent):
    """Competition-facing adapter. Replace policy internals, not this loop."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.policy = NullPolicy()
        self.policy.reset()

    @property
    def name(self) -> str:
        return f"{super().name}.{self.policy.name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> GameAction:
        decision = self.policy.choose_action(frames, latest_frame)
        decision.action.reasoning = decision.reasoning
        return decision.action
