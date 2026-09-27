from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from arc_agi2_fun.data import load_official_split
from arc_agi2_fun.visualize import plot_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ARC-AGI-2")
    parser.add_argument("--split", choices=["training", "evaluation"], default="training")
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    tasks = load_official_split(args.data_dir, args.split)
    try:
        task = tasks[args.task]
    except KeyError as exc:
        raise SystemExit(f"Unknown task id {args.task!r}") from exc

    plot_task(task)
    plt.show()


if __name__ == "__main__":
    main()
