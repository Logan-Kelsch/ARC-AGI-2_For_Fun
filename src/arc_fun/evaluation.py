from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EpisodeSummary:
    game_id: str
    final_state: str
    levels_completed: int
    actions_taken: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_agent(game_id: str, agent: Any) -> EpisodeSummary:
    latest = agent.frames[-1]
    state = getattr(latest, "state", None)
    return EpisodeSummary(
        game_id=game_id,
        final_state=getattr(state, "name", str(state)),
        levels_completed=int(getattr(latest, "levels_completed", 0) or 0),
        actions_taken=int(getattr(agent, "action_counter", 0) or 0),
    )
