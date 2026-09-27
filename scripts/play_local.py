from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

if not VENDOR.exists():
    raise SystemExit(f"Missing {VENDOR}. Run `make setup` first.")
sys.path.insert(0, str(VENDOR))

import arc_agi
from arc_agi import OperationMode

from arc_fun.evaluation import summarize_agent


def load_agent_class():
    path = ROOT / "agent" / "my_agent.py"
    spec = importlib.util.spec_from_file_location("user_agent_module", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


def short_game_id(game_id: str) -> str:
    return game_id.split("-")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=None, help="One id or comma-separated ids.")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--render", default=None, choices=["terminal"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env_info = arcade.get_environments()

    if args.list:
        for info in env_info:
            print(f"{short_game_id(info.game_id)}\t{getattr(info, 'title', '')}")
        return

    available = {short_game_id(info.game_id): info for info in env_info}
    if args.game:
        requested = [short_game_id(x.strip()) for x in args.game.split(",") if x.strip()]
        missing = [g for g in requested if g not in available]
        if missing:
            raise SystemExit(f"Unknown game(s): {missing}")
        game_ids = requested
    else:
        game_ids = sorted(available)

    AgentClass = load_agent_class()
    AgentClass.MAX_ACTIONS = min(AgentClass.MAX_ACTIONS, args.max_steps)

    rows = []
    for game_id in game_ids:
        print(f"\n=== {game_id} ===")
        env = arcade.make(game_id, render_mode=args.render)
        if env is None:
            print("Could not create environment; skipping.")
            continue

        agent = AgentClass(
            card_id="local-dev",
            game_id=game_id,
            agent_name=f"MyAgent.local.{game_id}",
            ROOT_URL="http://localhost",
            record=False,
            arc_env=env,
            tags=["local-dev"],
        )
        agent.main()
        summary = summarize_agent(game_id, agent)
        rows.append(summary.to_dict())
        print(summary.to_dict())

    scorecard = arcade.get_scorecard()
    score = getattr(scorecard, "score", scorecard)

    print("\n=== aggregate local scorecard ===")
    print(score)
    print("\n=== episodes ===")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
