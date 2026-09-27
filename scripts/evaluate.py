from __future__ import annotations

import argparse

from arc_agi2_fun.data import load_official_split
from arc_agi2_fun.evaluation import evaluate_tasks
from arc_agi2_fun.registry import create_solver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ARC-AGI-2")
    parser.add_argument("--split", choices=["training", "evaluation"], default="evaluation")
    parser.add_argument("--solver", default="primitive")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tasks = load_official_split(args.data_dir, args.split)
    solver = create_solver(args.solver)
    report = evaluate_tasks(tasks, solver, limit=args.limit)

    print(f"solver: {solver.name}")
    print(f"split: {args.split}")
    print(f"solved outputs: {report.solved_outputs}/{report.total_outputs}")
    print(f"exact pass@2 accuracy: {100 * report.accuracy:.2f}%")

    solved_tasks = [s.task_id for s in report.task_scores if s.solved_outputs]
    if solved_tasks:
        print("tasks with >=1 solved output:")
        for task_id in solved_tasks:
            print(f"  {task_id}")


if __name__ == "__main__":
    main()
