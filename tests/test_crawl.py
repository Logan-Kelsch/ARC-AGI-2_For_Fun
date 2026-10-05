from __future__ import annotations

import json
import math

import numpy as np

from notebooks.ops.crawl import (
    GrammarSourceKey,
    GrammarUCTPolicy,
    UCTStat,
    _DecisionRecord,
    _TaskUCTState,
    _build_depth0_coverage,
    crawl_synth_v1,
)
from notebooks.ops.environment import ProgramMeta, ProgramX
from notebooks.ops.ops import OP_REGISTRY, operation


def test_uct_exploitation_is_sqrt_solve_proportion():
    stat = UCTStat(
        visits=4,
        solve_credit=1.0,
    )

    assert stat.exploitation == 0.5


def test_uct_exploration_decays_to_configured_floor_at_horizon():
    policy = GrammarUCTPolicy(
        exploration_start=8.0,
        exploration_end=0.05,
        exploration_horizon=1_000_000,
    )

    assert policy.exploration_coefficient() == 8.0

    policy.total_gene_generations = 500_000
    midpoint = policy.exploration_coefficient()

    assert 0.05 < midpoint < 8.0

    policy.total_gene_generations = 1_000_000
    assert math.isclose(
        policy.exploration_coefficient(),
        0.05,
    )

    policy.total_gene_generations = 2_000_000
    assert math.isclose(
        policy.exploration_coefficient(),
        0.05,
    )


def test_gamma_backpropagates_solver_credit_through_gp_decisions():
    policy = GrammarUCTPolicy(gamma=0.85)

    parent_key = GrammarSourceKey(
        op_name="parent_op",
        source_depth=0,
        source_dims=(2,),
        source_dtypes=("bool",),
        source_shapes=("square:fixed:small",),
    )
    child_key = GrammarSourceKey(
        op_name="child_op",
        source_depth=1,
        source_dims=(2,),
        source_dtypes=("bool",),
        source_shapes=("square:fixed:small",),
    )

    parent_decision = policy.record_attempt(
        "parent_op",
        parent_key,
    )
    child_decision = policy.record_attempt(
        "child_op",
        child_key,
    )

    state = _TaskUCTState(
        run_id=1,
        seed_gene_ids=(0,),
        gene_depth={0: 0, 10: 1, 11: 2},
        phenotype_cache={},
        coverage_queue=None,
    )
    state.coverage_queue = __import__("collections").deque()
    state.gene_to_decision = {
        10: parent_decision,
        11: child_decision,
    }
    state.decisions = {
        parent_decision: _DecisionRecord(
            decision_id=parent_decision,
            op_name="parent_op",
            source_key=parent_key,
            parent_gene_ids=(0,),
            output_gene_ids=(10,),
        ),
        child_decision: _DecisionRecord(
            decision_id=child_decision,
            op_name="child_op",
            source_key=child_key,
            parent_gene_ids=(10,),
            output_gene_ids=(11,),
        ),
    }

    state.reward_solver_genes(
        policy,
        [11],
    )

    assert math.isclose(
        policy.source_stats[child_key].solve_credit,
        1.0,
    )
    assert math.isclose(
        policy.source_stats[parent_key].solve_credit,
        0.85,
    )
    assert math.isclose(
        policy.source_stats[parent_key].exploitation,
        math.sqrt(0.85),
    )


def test_depth0_coverage_orders_one_source_operation_by_low_gidx():
    @operation(
        partition="null",
        output_count=1,
    )
    def test_depth0_identity(meta, X, source_idx):
        values = [
            np.asarray(value).copy()
            for value in X[source_idx]
        ]
        gidx = X.append_gene(values)
        meta.append(
            source=source_idx,
            op="test_depth0_identity",
            dims=meta.dims[source_idx],
        )
        return gidx

    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=2)

    for index in range(3):
        gidx = X.append_gene(
            [
                np.int64(index),
                np.int64(index + 1),
            ]
        )
        meta.append(
            source=-1,
            op=f"seed_{index}",
            dims=0,
        )
        assert gidx == index

    seed_ids = tuple(
        meta.stable_id(gidx)
        for gidx in range(len(meta))
    )
    coverage = _build_depth0_coverage(
        meta,
        X,
        seed_ids,
    )
    selected = [
        candidate.source_gene_ids
        for candidate in coverage
        if candidate.op_name == "test_depth0_identity"
    ]

    assert selected == [(0,), (1,), (2,)]

    OP_REGISTRY.pop("test_depth0_identity", None)


def _write_identity_task(root, task_id):
    path = root / "data" / "training" / f"{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "train": [
                    {
                        "input": [[0, 1], [1, 0]],
                        "output": [[0, 1], [1, 0]],
                    }
                ],
                "test": [
                    {
                        "input": [[2, 0], [0, 2]],
                        "output": [[2, 0], [0, 2]],
                    }
                ],
            }
        )
    )


def test_crawl_honors_first_task_order_and_removes_exact_solutions(tmp_path):
    _write_identity_task(tmp_path, "first")
    _write_identity_task(tmp_path, "second")

    result = crawl_synth_v1(
        first_tasks=["second", "first"],
        max_GP=50,
        max_SP=50,
        prune_size_GP=0,
        max_total_generations=10,
        max_task_generations=5,
        data_root=tmp_path,
        rng=0,
        verbosity=0,
        show_success_plots=False,
    )

    assert [
        attempt.task_id
        for attempt in result.attempts
    ] == ["second", "first"]
    assert result.solved_task_ids == {"first", "second"}
    assert result.policy.total_gene_generations == 0
