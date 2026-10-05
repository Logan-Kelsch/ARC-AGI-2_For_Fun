from __future__ import annotations

import json

import numpy as np
import pytest

from notebooks.ops import synth, synth_v2
from notebooks.ops.environment import ProgramMeta, init_env
import notebooks.ops.solve as solve_module
import notebooks.ops.wrap as wrap_module
from notebooks.ops.wrap import (
    _operation_application_count,
    _print_iteration_status,
    _st_progress,
)


def _write_task(
    root,
    task_id,
    *,
    train,
    test,
    split="training",
):
    path = root / "data" / split / f"{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "train": train,
                "test": test,
            }
        )
    )
    return path


def test_synth_solves_exact_identity_before_generation(tmp_path):
    task_id = "identity_task"

    _write_task(
        tmp_path,
        task_id,
        train=[
            {
                "input": [[0, 1], [1, 0]],
                "output": [[0, 1], [1, 0]],
            },
            {
                "input": [[2, 0, 2]],
                "output": [[2, 0, 2]],
            },
        ],
        test=[
            {
                "input": [[3, 0], [0, 3]],
                "output": [[3, 0], [0, 3]],
            }
        ],
    )

    (
        solved_exactly,
        kelschinator,
        final_gp_len,
        final_sp_len,
        final_gp_ops,
        final_sp_ops,
        iterations,
    ) = synth(
        task_id,
        max_GP=100,
        max_SP=100,
        gen_size_GP=5,
        data_root=tmp_path,
        split="training",
        rng=0,
    )

    assert solved_exactly
    assert kelschinator.is_fitted_
    assert iterations == 0
    assert final_gp_len > 0
    assert final_sp_len > 0
    assert final_gp_ops >= 2
    assert final_sp_ops >= 2

    prediction = kelschinator.transform(
        np.array([[3, 0], [0, 3]], dtype=np.int64)
    )
    assert np.array_equal(
        prediction,
        np.array([[3, 0], [0, 3]], dtype=np.int64),
    )


def test_synth_stops_unsolved_when_both_gene_budgets_are_closed(tmp_path):
    task_id = "rotation_task"

    _write_task(
        tmp_path,
        task_id,
        train=[
            {
                "input": [[1, 2, 3], [4, 5, 6]],
                "output": [[4, 1], [5, 2], [6, 3]],
            },
            {
                "input": [[7, 8], [9, 0], [1, 2]],
                "output": [[1, 9, 7], [2, 0, 8]],
            },
        ],
        test=[
            {
                "input": [[1, 2], [3, 4]],
                "output": [[3, 1], [4, 2]],
            }
        ],
    )

    (
        solved_exactly,
        kelschinator,
        final_gp_len,
        final_sp_len,
        final_gp_ops,
        final_sp_ops,
        iterations,
    ) = synth(
        task_id,
        max_GP=0,
        max_SP=0,
        gen_size_GP=10,
        data_root=tmp_path,
        split="training",
        rng=0,
    )

    assert not solved_exactly
    assert not kelschinator.is_fitted_
    assert iterations == 0
    assert final_gp_len > 0
    assert final_sp_len > 0
    assert final_gp_ops >= 2
    assert final_sp_ops >= 2


def test_operation_application_count_collapses_multi_output_calls():
    meta = ProgramMeta(side="GP")

    meta.append(
        source=-1,
        op="raw_input",
        dims=2,
    )

    # One partition_shape call -> two genes.
    meta.append(
        source=0,
        op="partition_shape",
        dims=0,
    )
    meta.append(
        source=0,
        op="partition_shape",
        dims=0,
    )

    # One partition_composite call -> multiple genes.
    for dims in (0, 2, 0, 2):
        meta.append(
            source=0,
            op="partition_composite",
            dims=dims,
        )

    # Two genuinely distinct transform applications.
    meta.append(
        source=3,
        op="mat2_cwrotate",
        dims=2,
    )
    meta.append(
        source=5,
        op="mat2_cwrotate",
        dims=2,
    )

    assert _operation_application_count(meta) == 4



def _status_env():
    train = [
        {
            "input": np.array(
                [[0, 1], [1, 0]],
                dtype=np.int64,
            ),
            "output": np.array(
                [[0, 2], [2, 0]],
                dtype=np.int64,
            ),
        },
        {
            "input": np.array(
                [[1, 0], [0, 1]],
                dtype=np.int64,
            ),
            "output": np.array(
                [[2, 0], [0, 2]],
                dtype=np.int64,
            ),
        },
    ]
    return init_env(train)


def test_st_progress_counts_root_reachable_concrete_nodes():
    GP_meta, GP_X, SP_meta, SP_X, ST = _status_env()

    solved, total, proportion = _st_progress(ST)

    assert total > 0
    assert solved >= 0
    assert solved <= total
    assert proportion == solved / total

    ST.mark_solution(0, 0, rule="identity")

    solved_after, total_after, proportion_after = _st_progress(ST)

    assert total_after == total
    assert solved_after >= solved
    assert proportion_after >= proportion


def test_iteration_status_verbosity_zero_prints_nothing(capsys):
    GP_meta, GP_X, SP_meta, SP_X, ST = _status_env()

    _print_iteration_status(
        verbosity=0,
        iteration=4,
        elapsed_seconds=0.125,
        ST=ST,
        GP_X=GP_X,
        SP_X=SP_X,
    )

    assert capsys.readouterr().out == ""


def test_iteration_status_verbosity_one_is_compact(capsys):
    GP_meta, GP_X, SP_meta, SP_X, ST = _status_env()

    _print_iteration_status(
        verbosity=1,
        iteration=4,
        elapsed_seconds=0.125,
        ST=ST,
        GP_X=GP_X,
        SP_X=SP_X,
    )

    output = capsys.readouterr().out.strip()

    assert output == "[iter 4] 0.125s | solved=False"


def test_iteration_status_verbosity_two_adds_progress_and_sizes(capsys):
    GP_meta, GP_X, SP_meta, SP_X, ST = _status_env()

    solved, total, proportion = _st_progress(ST)

    _print_iteration_status(
        verbosity=2,
        iteration=7,
        elapsed_seconds=1.5,
        ST=ST,
        GP_X=GP_X,
        SP_X=SP_X,
    )

    output = capsys.readouterr().out.strip()

    assert output.startswith(
        "[iter 7] 1.500s | solved=False | "
    )
    assert f"ST={solved}/{total} ({proportion:.1%})" in output
    assert f"GP={len(GP_X)}" in output
    assert f"SP={len(SP_X)}" in output


def test_synth_rejects_invalid_verbosity_before_task_loading():
    with pytest.raises(
        ValueError,
        match="verbosity must be one of 0, 1, or 2",
    ):
        synth("does-not-matter", verbosity=3)

def test_solve_frontier_never_uses_equality_identity_recovery(monkeypatch):
    GP_meta, GP_X, SP_meta, SP_X, ST = _status_env()

    monkeypatch.setattr(
        solve_module,
        "_map_pool_to_gp_gidx",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy GP equality mapping was called")
        ),
    )
    monkeypatch.setattr(
        solve_module,
        "_map_frontier_to_sp_gidx",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy 0D SP equality mapping was called")
        ),
    )
    monkeypatch.setattr(
        solve_module,
        "_map_2d_frontier_to_sp_gidx",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy 2D SP equality mapping was called")
        ),
    )

    wrap_module._solve_frontier(
        GP_meta,
        GP_X,
        SP_meta,
        SP_X,
        ST,
        solve_module.SolveEvaluationCache(),
    )


def test_synth_v2_identity_matches_synth_without_entering_loop(tmp_path):
    task_id = "identity_task_v2"

    _write_task(
        tmp_path,
        task_id,
        train=[
            {
                "input": [[0, 1], [1, 0]],
                "output": [[0, 1], [1, 0]],
            },
            {
                "input": [[2, 0, 2]],
                "output": [[2, 0, 2]],
            },
        ],
        test=[
            {
                "input": [[3, 0], [0, 3]],
                "output": [[3, 0], [0, 3]],
            }
        ],
    )

    result = synth_v2(
        task_id,
        max_GP=100,
        max_SP=100,
        gen_size_GP=5,
        prune_size_GP=5,
        data_root=tmp_path,
        split="training",
        rng=0,
    )

    solved_exactly, kelschinator, *_, iterations = result

    assert solved_exactly
    assert kelschinator.is_fitted_
    assert iterations == 0


def test_synth_v2_solves_before_pruning_each_gp_growth_iteration(
    tmp_path,
    monkeypatch,
):
    task_id = "synth_v2_order_task"

    _write_task(
        tmp_path,
        task_id,
        train=[
            {
                "input": [[1, 2], [3, 4]],
                "output": [[4, 3], [2, 1]],
            },
            {
                "input": [[5, 6], [7, 8]],
                "output": [[8, 7], [6, 5]],
            },
        ],
        test=[
            {
                "input": [[9, 0], [1, 2]],
                "output": [[2, 1], [0, 9]],
            }
        ],
    )

    events = []

    def fake_solve(*args, **kwargs):
        events.append("solve")

    def fake_gp_generate(GP_meta, GP_X, n_new_genes, *, rng=None, **kwargs):
        events.append("generate")
        gidx = GP_X.append_gene(
            [np.int64(123)] * GP_X.sample_count
        )
        GP_meta.append(
            source=1,
            op="test_synth_v2_generated",
            dims=0,
        )
        return [gidx]

    def fake_gp_prune(GP_meta, GP_X, ST, prune=5, *, rng=None):
        events.append(("prune", prune))
        return []

    monkeypatch.setattr(
        wrap_module,
        "_solve_frontier",
        fake_solve,
    )
    monkeypatch.setattr(
        wrap_module,
        "GP_generate",
        fake_gp_generate,
    )
    monkeypatch.setattr(
        wrap_module,
        "GP_prune",
        fake_gp_prune,
    )

    # Initial GP has fewer than 100 genes. The fake generation runs once per
    # iteration, so close the GP side immediately after the first append.
    original_init_env = wrap_module.init_env
    initial_sizes = {}

    def recording_init_env(*args, **kwargs):
        env = original_init_env(*args, **kwargs)
        initial_sizes["gp"] = len(env[1])
        return env

    monkeypatch.setattr(
        wrap_module,
        "init_env",
        recording_init_env,
    )

    # Use an SP ceiling of zero so only the GP side can advance. Setting max_GP
    # to initial+1 requires resolving the initial size first, so use a small
    # preflight environment built from the written task.
    task = wrap_module._load_task_by_id(
        task_id,
        data_root=tmp_path,
        split="training",
    )
    preflight = original_init_env(task.train)
    max_gp = len(preflight[1]) + 1

    result = synth_v2(
        task_id,
        max_GP=max_gp,
        max_SP=0,
        gen_size_GP=1,
        prune_size_GP=3,
        data_root=tmp_path,
        split="training",
        rng=0,
    )

    assert result[-1] == 1
    assert events == [
        "solve",
        "generate",
        "solve",
        ("prune", 3),
    ]


def test_synth_v2_zero_prune_size_skips_gp_prune(
    tmp_path,
    monkeypatch,
):
    task_id = "synth_v2_no_prune_task"

    _write_task(
        tmp_path,
        task_id,
        train=[
            {
                "input": [[1, 2], [3, 4]],
                "output": [[4, 3], [2, 1]],
            }
        ],
        test=[
            {
                "input": [[5, 6], [7, 8]],
                "output": [[8, 7], [6, 5]],
            }
        ],
    )

    calls = []

    monkeypatch.setattr(
        wrap_module,
        "_solve_frontier",
        lambda *args, **kwargs: None,
    )

    def fake_gp_generate(GP_meta, GP_X, n_new_genes, *, rng=None, **kwargs):
        gidx = GP_X.append_gene(
            [np.int64(99)] * GP_X.sample_count
        )
        GP_meta.append(
            source=1,
            op="test_v2_no_prune",
            dims=0,
        )
        return [gidx]

    monkeypatch.setattr(
        wrap_module,
        "GP_generate",
        fake_gp_generate,
    )
    monkeypatch.setattr(
        wrap_module,
        "GP_prune",
        lambda *args, **kwargs: calls.append(True),
    )

    task = wrap_module._load_task_by_id(
        task_id,
        data_root=tmp_path,
        split="training",
    )
    initial_gp = len(wrap_module.init_env(task.train)[1])

    synth_v2(
        task_id,
        max_GP=initial_gp + 1,
        max_SP=0,
        gen_size_GP=1,
        prune_size_GP=0,
        data_root=tmp_path,
        split="training",
        rng=0,
    )

    assert calls == []


def test_synth_v2_rejects_invalid_prune_size_before_task_loading():
    with pytest.raises(
        ValueError,
        match="prune_size_GP must be a non-negative integer",
    ):
        synth_v2(
            "does-not-matter",
            prune_size_GP=-1,
        )

