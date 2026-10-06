from __future__ import annotations

import argparse

try:
    from notebooks.ops import crawl_synth_v1
except ImportError:
    from ops import crawl_synth_v1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the persistent Grammar-UCT ARC training crawl."
    )
    parser.add_argument(
        "--first-task",
        action="append",
        default=[],
        dest="first_tasks",
        help="Training task ID to try before random unsolved-task sampling.",
    )
    parser.add_argument("--max-gp", type=int, default=1000)
    parser.add_argument("--max-sp", type=int, default=100)
    parser.add_argument("--prune-size", type=int, default=5)
    parser.add_argument(
        "--max-total-generations",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--max-task-generations",
        type=int,
        default=None,
    )
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument(
        "--exploration-start",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--exploration-end",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--exploration-horizon",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--depth-exploration-power",
        type=float,
        default=1.0,
        help=(
            "Exponent applied to the logarithmic depth exploration discount. "
            "1.0 gives 1/log2(depth+2)."
        ),
    )
    parser.add_argument(
        "--depth-focus",
        type=float,
        default=0.0,
        help="Deprecated compatibility parameter; no longer adds deep bias.",
    )
    parser.add_argument(
        "--operation-selection",
        choices=("l1", "softmax", "argmax"),
        default="l1",
        help=(
            "Operation-arm selection policy. l1 samples proportional to "
            "exploit+explore score; softmax samples from exp(score); "
            "argmax reproduces deterministic highest-score selection."
        ),
    )
    parser.add_argument(
        "--source-probe-attempts",
        type=int,
        default=32,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--verbosity", type=int, default=2)
    parser.add_argument("--status-every", type=int, default=100)
    parser.add_argument(
        "--plot-every-generations",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--no-success-plots",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    result = crawl_synth_v1(
        first_tasks=args.first_tasks,
        max_GP=args.max_gp,
        max_SP=args.max_sp,
        prune_size_GP=args.prune_size,
        max_total_generations=args.max_total_generations,
        max_task_generations=args.max_task_generations,
        gamma=args.gamma,
        exploration_start=args.exploration_start,
        exploration_end=args.exploration_end,
        exploration_horizon=args.exploration_horizon,
        depth_exploration_power=args.depth_exploration_power,
        depth_focus=args.depth_focus,
        operation_selection=args.operation_selection,
        source_probe_attempts=args.source_probe_attempts,
        rng=args.seed,
        data_root=args.data_root,
        verbosity=args.verbosity,
        status_every=args.status_every,
        plot_every_generations=args.plot_every_generations,
        state_path=args.state_path,
        show_success_plots=not args.no_success_plots,
    )

    print(
        "crawl complete | "
        f"solved={len(result.solved_task_ids)} | "
        f"GP generations={result.policy.total_gene_generations:,} | "
        f"task attempts={result.policy.task_attempts}"
    )


if __name__ == "__main__":
    main()
