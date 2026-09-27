from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc_fun.arcade_eval import cache_all_public_games, list_games


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover and cache ARC-AGI-3 public game environments."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List cached games in offline mode instead of downloading.",
    )
    args = parser.parse_args()

    if args.list:
        rows = list_games("offline")
        if not rows:
            print("No cached games found. Run: make arcade-cache")
            return
        print(f"{len(rows)} cached games")
        for row in rows:
            print(f"{row['game_id']:8} {row['title']}")
        return

    rows = cache_all_public_games()
    success = sum(bool(row["cached"]) for row in rows)
    print(f"Cached/available: {success}/{len(rows)}")
    for row in rows:
        status = "OK" if row["cached"] else "FAIL"
        suffix = f" — {row['error']}" if row["error"] else ""
        print(f"[{status}] {row['game_id']:8} {row['title']}{suffix}")

    if success != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
