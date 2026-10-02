from __future__ import annotations

import json

import numpy as np

from notebooks.ops import synth
from notebooks.ops.environment import ProgramMeta
from notebooks.ops.wrap import _operation_application_count


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
