from __future__ import annotations

import argparse

from arc_agi2_fun.data import load_official_split
from arc_agi2_fun.registry import create_solver
from arc_agi2_fun.submission import build_submission, write_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ARC-AGI-2")
    parser.add_argument("--split", choices=["training", "evaluation"], default="evaluation")
    parser.add_argument("--solver", default="primitive")
    parser.add_argument("--output", default="results/submission.json")
    args = parser.parse_args()

    tasks = load_official_split(args.data_dir, args.split)
    solver = create_solver(args.solver)
    path = write_submission(build_submission(tasks, solver), args.output)
    print(f"Wrote {len(tasks)} tasks to {path}")


if __name__ == "__main__":
    main()
