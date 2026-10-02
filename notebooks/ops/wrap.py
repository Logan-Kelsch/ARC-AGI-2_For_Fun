from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from arc_agi2_fun.data import ArcTask, load_task_file

from .environment import (
    ProgramMeta,
    ProgramX,
    STNodeRef,
    STSet,
    SolutionTree,
    get_GP_pool,
    get_ST_unsovled_frontier,
    init_env,
)
from .kelschinator import Kelschinator
from .ops import GP_generate, GP_prune, SP_generate
from .solve import (
    solve_0dim_1gene_basic,
    solve_2dim_1gene_basic,
)


def _repo_data_root() -> Path:
    """Return the default local ARC-AGI-2 data checkout."""
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "ARC-AGI-2"
    )


def _load_task_by_id(
    task_id: str,
    *,
    data_root: str | Path | None = None,
    split: str | None = None,
) -> ArcTask:
    """Load one official public task without materializing the whole split."""
    task_id = str(task_id).strip()

    if not task_id:
        raise ValueError("task_id must be a non-empty string.")

    root = (
        _repo_data_root()
        if data_root is None
        else Path(data_root)
    )

    splits = (
        (str(split),)
        if split is not None
        else ("training", "evaluation")
    )

    matches: list[Path] = []

    for split_name in splits:
        path = root / "data" / split_name / f"{task_id}.json"

        if path.exists():
            matches.append(path)

    if not matches:
        searched = ", ".join(
            str(root / "data" / name / f"{task_id}.json")
            for name in splits
        )
        raise FileNotFoundError(
            f"Could not find ARC task {task_id!r}. Searched: {searched}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Task ID {task_id!r} exists in multiple splits; "
            "pass split='training' or split='evaluation'."
        )

    return load_task_file(matches[0])


def _freeze(value: Any) -> Any:
    """Convert metadata values into stable hashable operation signatures."""
    if isinstance(value, np.generic):
        return ("numpy_scalar", str(value.dtype), value.item())

    if isinstance(value, np.ndarray):
        return (
            "array",
            str(value.dtype),
            tuple(value.shape),
            tuple(_freeze(item) for item in value.flat),
        )

    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (
                        str(key),
                        _freeze(item),
                    )
                    for key, item in value.items()
                )
            ),
        )

    if isinstance(value, (list, tuple)):
        return (
            type(value).__name__,
            tuple(_freeze(item) for item in value),
        )

    return value


def _operation_application_count(meta: ProgramMeta) -> int:
    """Count distinct transformation applications represented by metadata.

    Multiple output genes emitted by one operation call count once.  Raw input
    or output genes are not operations.

    partition_select_residual is an initialization-wide partition whose many
    subset-support genes carry distinct semantic params; it therefore counts
    once per source rather than once per materialized subset.
    """
    signatures: set[Any] = set()

    for gidx, op_name in enumerate(meta.op):
        if op_name.startswith("raw_"):
            continue

        if op_name == "partition_select_residual":
            signature = (
                op_name,
                _freeze(meta.source[gidx]),
            )
        else:
            signature = (
                op_name,
                _freeze(meta.source[gidx]),
                _freeze(meta.params[gidx]),
            )

        signatures.add(signature)

    return len(signatures)


def _solve_frontier(
    GP_meta: ProgramMeta,
    GP_X: ProgramX,
    SP_X: ProgramX,
    ST: SolutionTree,
) -> None:
    """Run every currently implemented exact one-gene matcher."""

    if ST.solved:
        return

    gp_0d = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=0,
        max_dim=0,
    )
    sp_0d = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=0,
        max_dim=0,
    )

    solve_0dim_1gene_basic(
        gp_0d,
        sp_0d,
        GP_X=GP_X,
        SP_X=SP_X,
        ST=ST,
    )

    if ST.solved:
        return

    gp_2d = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=2,
        max_dim=2,
    )
    sp_2d = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=2,
        max_dim=2,
    )

    solve_2dim_1gene_basic(
        gp_2d,
        sp_2d,
        GP_X=GP_X,
        SP_X=SP_X,
        ST=ST,
    )


def _exact_test_match(
    task: ArcTask,
    kelschinator: Kelschinator,
) -> bool:
    """Require exact equality on every known test output."""
    if task.test_outputs is None:
        return False

    if len(task.test_inputs) != len(task.test_outputs):
        return False

    for X_test, Y_test in zip(
        task.test_inputs,
        task.test_outputs,
    ):
        y_hat = kelschinator.transform(
            np.asarray(X_test, dtype=np.int64)
        )
        target = np.asarray(
            Y_test,
            dtype=np.int64,
        )

        if not np.array_equal(y_hat, target):
            return False

    return True



def _st_progress(ST: SolutionTree) -> tuple[int, int, float]:
    """Return solved/total concrete nodes reachable from the ST root."""
    if not ST.roots:
        return 0, 0, 1.0

    reachable: set[Any] = set()

    def visit_requirement(requirement: Any) -> None:
        if isinstance(requirement, STNodeRef):
            visit_node(requirement.node_id)
            return

        if isinstance(requirement, STSet):
            for member in requirement.members:
                visit_requirement(member)
            return

        # Inverse references are innate grammar knowledge and do not contribute
        # concrete SP targets to the progress denominator.

    def visit_node(node_id: Any) -> None:
        if node_id in reachable:
            return

        reachable.add(node_id)
        node = ST[node_id]

        if node.derivation is not None:
            visit_requirement(node.derivation)

    for root in ST.roots:
        visit_node(root)

    concrete = [
        node_id
        for node_id in reachable
        if ST[node_id].sp_gidx is not None
    ]

    total = len(concrete)

    if total == 0:
        return 0, 0, 1.0

    solved = sum(
        1
        for node_id in concrete
        if ST.is_solved(node_id)
    )

    return solved, total, solved / total


def _print_iteration_status(
    *,
    verbosity: int,
    iteration: int,
    elapsed_seconds: float,
    ST: SolutionTree,
    GP_X: ProgramX,
    SP_X: ProgramX,
) -> None:
    """Print one concise search-loop status line."""
    if verbosity <= 0:
        return

    base = (
        f"[iter {iteration}] "
        f"{elapsed_seconds:.3f}s | "
        f"solved={ST.solved}"
    )

    if verbosity == 1:
        print(base)
        return

    solved_nodes, total_nodes, proportion = _st_progress(ST)

    print(
        f"{base} | "
        f"ST={solved_nodes}/{total_nodes} "
        f"({proportion:.1%}) | "
        f"GP={len(GP_X)} | "
        f"SP={len(SP_X)}"
    )

def synth(
    task_id: str,
    max_GP: int = 1000,
    max_SP: int = 1000,
    gen_size_GP: int = 10,
    *,
    rng: np.random.Generator | int | None = None,
    data_root: str | Path | None = None,
    split: str | None = None,
    select_residual_max_destinations: int = 5,
    verbosity: int = 0,
) -> tuple[
    bool,
    Kelschinator,
    int,
    int,
    int,
    int,
    int,
]:
    """Run the current bidirectional synthesis loop for one ARC task.

    Parameters
    ----------
    task_id:
        Official ARC task ID.

    max_GP / max_SP:
        Loose gene-count ceilings.  Generation is allowed whenever the current
        side is strictly below its ceiling.  A final generation call may push
        the resulting gene count above the ceiling; no additional generation
        is then allowed on that side.

    gen_size_GP:
        Passed directly to GP_generate(..., n_new_genes=gen_size_GP) on every
        GP growth iteration.

    rng:
        Optional deterministic seed or NumPy Generator shared by GP/SP growth.

    data_root:
        Optional ARC-AGI-2 checkout root.  Defaults to
        <repo>/data/ARC-AGI-2.

    split:
        Optional "training" or "evaluation".  When omitted, both are searched
        and an ambiguous duplicate task ID is rejected.

    verbosity:
        Search-loop reporting level:
          0 = no wrapper status output
          1 = iteration number, iteration time, and ST solved state
          2 = level 1 plus reachable concrete-ST progress and GP/SP sizes

    Returns
    -------
    (
        solved_exactly,
        kelschinator,
        final_GP_length,
        final_SP_length,
        final_GP_operation_count,
        final_SP_operation_count,
        while_loop_iterations,
    )

    solved_exactly is True only when ST is solved, Kelschinator successfully
    distills the proof, and every known test output is matched exactly.

    If no exact solution is found, the returned Kelschinator remains unfitted
    and its last_error_ explains the final fit state when fitting was attempted.
    """
    for name, value in (
        ("max_GP", max_GP),
        ("max_SP", max_SP),
        ("gen_size_GP", gen_size_GP),
    ):
        if isinstance(value, bool) or int(value) < 0:
            raise ValueError(
                f"{name} must be a non-negative integer."
            )

    max_GP = int(max_GP)
    max_SP = int(max_SP)
    gen_size_GP = int(gen_size_GP)

    if isinstance(verbosity, bool) or verbosity not in {0, 1, 2}:
        raise ValueError("verbosity must be one of 0, 1, or 2.")

    if gen_size_GP == 0 and max_GP > 0:
        # Zero-size GP growth is legal, but it cannot ever advance the GP side.
        gp_exhausted = True
    else:
        gp_exhausted = False

    sp_exhausted = False

    task = _load_task_by_id(
        task_id,
        data_root=data_root,
        split=split,
    )

    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(
        task.train,
        select_residual_max_destinations=(
            select_residual_max_destinations
        ),
    )

    rng = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(rng)
    )

    iterations = 0

    # Initial semantics may already meet exactly before any random growth.
    _solve_frontier(
        GP_meta,
        GP_X,
        SP_X,
        ST,
    )

    while not ST.solved:
        can_grow_gp = (
            not gp_exhausted
            and len(GP_X) < max_GP
        )
        can_grow_sp = (
            not sp_exhausted
            and len(SP_X) < max_SP
        )

        if not can_grow_gp and not can_grow_sp:
            break

        iterations += 1
        iteration_started = perf_counter()

        if can_grow_gp:
            generated_gp = GP_generate(
                GP_meta,
                GP_X,
                n_new_genes=gen_size_GP,
                rng=rng,
            )

            if not generated_gp:
                gp_exhausted = True

        if can_grow_sp and not ST.solved:
            generated_sp = SP_generate(
                SP_meta,
                SP_X,
                ST,
                rng=rng,
            )

            if not generated_sp:
                sp_exhausted = True

        _solve_frontier(
            GP_meta,
            GP_X,
            SP_X,
            ST,
        )

        _print_iteration_status(
            verbosity=verbosity,
            iteration=iterations,
            elapsed_seconds=(
                perf_counter() - iteration_started
            ),
            ST=ST,
            GP_X=GP_X,
            SP_X=SP_X,
        )

    kelschinator = Kelschinator()
    solved_exactly = False

    if ST.solved and kelschinator.fit(ST):
        solved_exactly = _exact_test_match(
            task,
            kelschinator,
        )

    return (
        solved_exactly,
        kelschinator,
        len(GP_X),
        len(SP_X),
        _operation_application_count(GP_meta),
        _operation_application_count(SP_meta),
        iterations,
    )

def synth_v2(
    task_id: str,
    max_GP: int = 1000,
    max_SP: int = 1000,
    gen_size_GP: int = 10,
    prune_size_GP: int = 5,
    *,
    rng: np.random.Generator | int | None = None,
    data_root: str | Path | None = None,
    split: str | None = None,
    select_residual_max_destinations: int = 5,
    verbosity: int = 0,
) -> tuple[
    bool,
    Kelschinator,
    int,
    int,
    int,
    int,
    int,
]:
    """Run synthesis with probabilistic GP generation and pruning.

    This follows synth()'s search loop and return contract, with one additional
    per-GP-growth control:

    prune_size_GP:
        Number of GP genes requested from GP_prune after each successful GP
        generation iteration. Default is 5.

    Iteration order is:

        GP generation
        SP generation
        solve current ST frontier
        GP pruning

    Solving occurs before pruning so any newly useful GP gene is first recorded
    in ST. GP_prune then protects that solver and its complete GP dependency
    closure before selecting expendable leaves.

    Setting prune_size_GP=0 disables pruning and recovers synth()'s loop
    behavior while retaining the v2 entry point.

    Note that max_GP remains a live GP-size ceiling. If pruning keeps the live
    GP pool below that ceiling, max_GP alone may no longer bound the number of
    iterations; generation exhaustion or SP exhaustion/limits still apply.
    """
    for name, value in (
        ("max_GP", max_GP),
        ("max_SP", max_SP),
        ("gen_size_GP", gen_size_GP),
        ("prune_size_GP", prune_size_GP),
    ):
        if isinstance(value, bool) or int(value) < 0:
            raise ValueError(
                f"{name} must be a non-negative integer."
            )

    max_GP = int(max_GP)
    max_SP = int(max_SP)
    gen_size_GP = int(gen_size_GP)
    prune_size_GP = int(prune_size_GP)

    if isinstance(verbosity, bool) or verbosity not in {0, 1, 2}:
        raise ValueError("verbosity must be one of 0, 1, or 2.")

    if gen_size_GP == 0 and max_GP > 0:
        gp_exhausted = True
    else:
        gp_exhausted = False

    sp_exhausted = False

    task = _load_task_by_id(
        task_id,
        data_root=data_root,
        split=split,
    )

    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(
        task.train,
        select_residual_max_destinations=(
            select_residual_max_destinations
        ),
    )

    rng = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(rng)
    )

    iterations = 0

    _solve_frontier(
        GP_meta,
        GP_X,
        SP_X,
        ST,
    )

    while not ST.solved:
        can_grow_gp = (
            not gp_exhausted
            and len(GP_X) < max_GP
        )
        can_grow_sp = (
            not sp_exhausted
            and len(SP_X) < max_SP
        )

        if not can_grow_gp and not can_grow_sp:
            break

        iterations += 1
        iteration_started = perf_counter()
        generated_gp: list[int] = []

        if can_grow_gp:
            generated_gp = GP_generate(
                GP_meta,
                GP_X,
                n_new_genes=gen_size_GP,
                rng=rng,
            )

            if not generated_gp:
                gp_exhausted = True

        if can_grow_sp and not ST.solved:
            generated_sp = SP_generate(
                SP_meta,
                SP_X,
                ST,
                rng=rng,
            )

            if not generated_sp:
                sp_exhausted = True

        _solve_frontier(
            GP_meta,
            GP_X,
            SP_X,
            ST,
        )

        if (
            generated_gp
            and prune_size_GP > 0
            and not ST.solved
        ):
            GP_prune(
                GP_meta,
                GP_X,
                ST,
                prune=prune_size_GP,
                rng=rng,
            )

        _print_iteration_status(
            verbosity=verbosity,
            iteration=iterations,
            elapsed_seconds=(
                perf_counter() - iteration_started
            ),
            ST=ST,
            GP_X=GP_X,
            SP_X=SP_X,
        )

    kelschinator = Kelschinator()
    solved_exactly = False

    if ST.solved and kelschinator.fit(ST):
        solved_exactly = _exact_test_match(
            task,
            kelschinator,
        )

    return (
        solved_exactly,
        kelschinator,
        len(GP_X),
        len(SP_X),
        _operation_application_count(GP_meta),
        _operation_application_count(SP_meta),
        iterations,
    )

