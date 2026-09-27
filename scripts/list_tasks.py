from __future__ import annotations

import argparse

from arc_agi2_fun.data import load_official_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ARC-AGI-2")
    parser.add_argument("--split", choices=["training", "evaluation"], default="evaluation")
    args = parser.parse_args()

    tasks = load_official_split(args.data_dir, args.split)
    print(f"{len(tasks)} tasks in {args.split}")
    for task_id in tasks:
        print(task_id)


if __name__ == "__main__":
    main()
