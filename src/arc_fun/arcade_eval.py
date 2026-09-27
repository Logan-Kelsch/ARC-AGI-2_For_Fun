from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import arc_agi
from arc_agi import Arcade, OperationMode
from arcengine import FrameData, FrameDataRaw, GameState

from .policy_registry import create_policy


def short_game_id(game_id: str) -> str:
    """Normalize versioned ids such as ls20-<hash> to ls20."""
    return game_id.split("-")[0]


def _operation_mode(mode: str) -> OperationMode:
    normalized = mode.strip().lower()
    mapping = {
        "normal": OperationMode.NORMAL,
        "offline": OperationMode.OFFLINE,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError("mode must be 'normal' or 'offline'") from exc


def make_arcade(mode: str = "offline") -> Arcade:
    """Create the official ARC arcade in normal or fully cached/offline mode."""
    return arc_agi.Arcade(operation_mode=_operation_mode(mode))


def list_games(mode: str = "offline") -> list[dict[str, Any]]:
    """List environments visible to the official arcade."""
    arcade = make_arcade(mode)
    games = []
    for info in arcade.get_environments():
        games.append(
            {
                "game_id": short_game_id(info.game_id),
                "versioned_game_id": info.game_id,
                "title": getattr(info, "title", ""),
            }
        )
    return sorted(games, key=lambda row: row["game_id"])


def cache_all_public_games() -> list[dict[str, Any]]:
    """Download/cache every currently advertised public game."""
    arcade = make_arcade("normal")
    rows: list[dict[str, Any]] = []

    for info in arcade.get_environments():
        game_id = short_game_id(info.game_id)
        try:
            env = arcade.make(game_id, seed=0)
            rows.append(
                {
                    "game_id": game_id,
                    "title": getattr(info, "title", ""),
                    "cached": env is not None,
                    "error": None if env is not None else "arcade.make returned None",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "game_id": game_id,
                    "title": getattr(info, "title", ""),
                    "cached": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return sorted(rows, key=lambda row: row["game_id"])


def _frame_from_raw(raw: FrameDataRaw | None) -> FrameData:
    if raw is None:
        raise RuntimeError("ARC environment returned no frame data.")

    return FrameData(
        game_id=raw.game_id,
        frame=[arr.tolist() for arr in raw.frame],
        state=raw.state,
        levels_completed=raw.levels_completed,
        win_levels=raw.win_levels,
        guid=raw.guid,
        full_reset=raw.full_reset,
        available_actions=raw.available_actions,
    )


@dataclass
class GameEvaluation:
    game_id: str
    policy: str
    seed: int
    mode: str
    final_state: str
    levels_completed: int
    actions_taken: int
    policy_seconds: float
    official_local_score: float | None
    scorecard: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_game(
    game_id: str,
    *,
    policy_name: str = "null",
    seed: int = 0,
    max_actions: int = 80,
    mode: str = "offline",
    render_mode: str | None = None,
    save_recording: bool = False,
) -> GameEvaluation:
    """Evaluate one registered policy on one real ARC environment."""
    arcade = make_arcade(mode)
    env = arcade.make(
        game_id,
        seed=seed,
        render_mode=render_mode,
        save_recording=save_recording,
    )
    if env is None:
        hint = " Run make arcade-cache first." if mode == "offline" else ""
        raise RuntimeError(f"Could not create game {game_id!r}.{hint}")

    policy = create_policy(policy_name)
    policy.reset()

    # Match the official Agent history convention.
    frames: list[FrameData] = [FrameData(levels_completed=0)]
    latest = _frame_from_raw(env.observation_space)

    action_counter = 0
    policy_seconds = 0.0

    # The official Agent loop uses <= MAX_ACTIONS; preserve that behavior.
    while latest.state is not GameState.WIN and action_counter <= max_actions:
        started = perf_counter()
        decision = policy.choose_action(frames, latest)
        policy_seconds += perf_counter() - started

        action = decision.action
        reasoning = decision.reasoning
        action.reasoning = reasoning

        raw_next = env.step(
            action,
            data=action.action_data.model_dump(),
            reasoning=reasoning,
        )
        latest = _frame_from_raw(raw_next)
        frames.append(latest)
        action_counter += 1

    scorecard_obj = arcade.get_scorecard()
    score_value = getattr(scorecard_obj, "score", None)

    scorecard: dict[str, Any] | None = None
    if hasattr(scorecard_obj, "model_dump"):
        scorecard = scorecard_obj.model_dump()

    state_name = getattr(latest.state, "name", str(latest.state))

    return GameEvaluation(
        game_id=short_game_id(game_id),
        policy=policy_name,
        seed=seed,
        mode=mode,
        final_state=state_name,
        levels_completed=int(latest.levels_completed or 0),
        actions_taken=action_counter,
        policy_seconds=policy_seconds,
        official_local_score=float(score_value) if score_value is not None else None,
        scorecard=scorecard,
    )
