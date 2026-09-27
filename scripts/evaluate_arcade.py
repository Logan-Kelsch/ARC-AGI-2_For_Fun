from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc_fun.arcade_eval import evaluate_game, list_games
from arc_fun.policy_registry import available_policies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a replaceable policy on one official ARC-AGI-3 game."
    )
    parser.add_argument("--game", help="Game id, for example ls20.")
    parser.add_argument("--policy", default="null")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--mode", choices=["offline", "normal"], default="offline")
    parser.add_argument(
        "--render",
        choices=["terminal", "terminal-fast"],
        default=None,
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--list-games", action="store_true")
    parser.add_argument("--list-policies", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.list_policies:
        for name in available_policies():
            print(name)
        return

    if args.list_games:
        games = list_games(args.mode)
        if not games:
            print("No games visible. If using offline mode, run: make arcade-cache")
        for game in games:
            print(f"{game['game_id']:8} {game['title']}")
        return

    if not args.game:
        parser.error("--game is required unless --list-games/--list-policies is used")

    result = evaluate_game(
        args.game,
        policy_name=args.policy,
        seed=args.seed,
        max_actions=args.max_actions,
        mode=args.mode,
        render_mode=args.render,
        save_recording=args.record,
    )

    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return

    print("\n=== ARC policy evaluation ===")
    print(f"game:                 {result.game_id}")
    print(f"policy:               {result.policy}")
    print(f"seed:                 {result.seed}")
    print(f"mode:                 {result.mode}")
    print(f"final state:          {result.final_state}")
    print(f"levels completed:     {result.levels_completed}")
    print(f"real actions taken:   {result.actions_taken}")
    print(f"policy compute sec:   {result.policy_seconds:.6f}")
    print(f"official local score: {result.official_local_score}")

    if result.scorecard is not None:
        print("\n=== official scorecard ===")
        print(json.dumps(result.scorecard, indent=2, default=str))


if __name__ == "__main__":
    main()
