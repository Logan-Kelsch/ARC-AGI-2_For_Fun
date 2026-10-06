from __future__ import annotations

from collections import deque
import json
import math

import numpy as np

from notebooks.ops.crawl import (
    GrammarSourceKey,
    GrammarUCTPolicy,
    UCTStat,
    _DecisionRecord,
    _TaskUCTState,
    _aggregate_depth_distribution,
    _arm_selection_probabilities,
    _select_arm_index,
    _build_depth0_coverage,
    _build_exploration_heatmap,
    crawl_synth_v1,
)
from notebooks.ops.environment import ProgramMeta, ProgramX
from notebooks.ops.ops import OP_REGISTRY, operation


def test_l1_arm_selection_normalizes_raw_scores():
    probabilities = _arm_selection_probabilities(
        [1.0, 2.0, 3.0],
        mode="l1",
    )

    assert np.allclose(
        probabilities,
        np.array([1.0, 2.0, 3.0]) / 6.0,
    )


def test_l1_arm_selection_falls_back_to_uniform_for_zero_scores():
    probabilities = _arm_selection_probabilities(
        [0.0, 0.0, 0.0],
        mode="l1",
    )

    assert np.allclose(
        probabilities,
        np.full(3, 1.0 / 3.0),
    )


def test_softmax_arm_selection_uses_stable_exponential_normalization():
    probabilities = _arm_selection_probabilities(
        [1.0, 2.0, 3.0],
        mode="softmax",
    )
    expected = np.exp(np.array([-2.0, -1.0, 0.0]))
    expected /= expected.sum()

    assert np.allclose(probabilities, expected)


def test_argmax_arm_selection_remains_available_and_deterministic():
    index = _select_arm_index(
        [1.0, 4.0, 2.0],
        mode="argmax",
        rng=np.random.default_rng(0),
    )

    assert index == 1


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


def test_source_exploration_is_log_scaled_down_by_depth():
    policy = GrammarUCTPolicy(
        exploration_start=8.0,
        exploration_end=0.05,
        depth_exploration_power=1.0,
    )
    policy.total_decisions = 100

    def key(depth):
        return GrammarSourceKey(
            op_name="op",
            source_depth=depth,
            source_dims=(2,),
            source_dtypes=("bool",),
            source_shapes=("square:fixed:small",),
        )

    depth0 = policy.source_score(key(0))[2]
    depth1 = policy.source_score(key(1))[2]
    depth2 = policy.source_score(key(2))[2]
    depth6 = policy.source_score(key(6))[2]

    assert depth0 > depth1 > depth2 > depth6
    assert math.isclose(
        depth2 / depth0,
        0.5,
        rel_tol=1e-12,
    )
    assert math.isclose(
        depth6 / depth0,
        1.0 / 3.0,
        rel_tol=1e-12,
    )


def test_depth_plot_distribution_uses_logarithmic_bins():
    distribution = {
        depth: {
            "exploitation": [float(depth)],
            "exploration": [float(depth) + 0.5],
        }
        for depth in range(18)
    }

    aggregated = _aggregate_depth_distribution(distribution)

    assert [label for label, _ in aggregated] == [
        "0",
        "1",
        "2-3",
        "4-7",
        "8-15",
        "16-31",
    ]
    assert aggregated[2][1]["exploitation"] == [2.0, 3.0]
    assert aggregated[4][1]["exploration"] == [
        float(depth) + 0.5
        for depth in range(8, 16)
    ]


def test_exploration_heatmap_aligns_depth_bins_by_iteration():
    history = [
        (0, {0: 8.0, 1: 5.0}),
        (1, {0: 7.0, 1: 4.0, 2: 3.0}),
        (2, {0: 6.0, 2: 2.0, 3: 4.0}),
    ]

    matrix, labels, iterations = _build_exploration_heatmap(
        history
    )

    assert labels == ["0", "1", "2-3"]
    assert iterations == [0, 1, 2]
    assert np.allclose(
        matrix[0],
        [8.0, 7.0, 6.0],
        equal_nan=True,
    )
    assert np.allclose(
        matrix[1],
        [5.0, 4.0, np.nan],
        equal_nan=True,
    )
    assert np.allclose(
        matrix[2],
        [np.nan, 3.0, 3.0],
        equal_nan=True,
    )


def test_loaded_crawl_prints_grammar_summary_and_operation_ranks(
    tmp_path,
    capsys,
):
    _write_identity_task(tmp_path, "first")

    policy = GrammarUCTPolicy(
        total_gene_generations=10,
        total_decisions=8,
        task_attempts=3,
    )
    policy.operation_stats = {
        "strong_op": UCTStat(
            visits=4,
            solve_credit=4.0,
        ),
        "weak_op": UCTStat(
            visits=4,
            solve_credit=1.0,
        ),
    }
    policy.source_stats = {
        GrammarSourceKey(
            op_name="strong_op",
            source_depth=0,
            source_dims=(2,),
            source_dtypes=("bool",),
            source_shapes=("square:fixed:small",),
        ): UCTStat(
            visits=4,
            solve_credit=4.0,
        ),
        GrammarSourceKey(
            op_name="weak_op",
            source_depth=2,
            source_dims=(2,),
            source_dtypes=("bool",),
            source_shapes=("square:fixed:small",),
        ): UCTStat(
            visits=4,
            solve_credit=1.0,
        ),
    }

    state_path = tmp_path / "grammar.json"
    policy.save(state_path)

    crawl_synth_v1(
        max_GP=50,
        max_SP=50,
        prune_size_GP=0,
        max_total_generations=10,
        data_root=tmp_path,
        state_path=state_path,
        verbosity=1,
        show_success_plots=False,
    )

    output = capsys.readouterr().out
    assert "[loaded Grammar-UCT]" in output
    assert "depth grammar:" in output
    assert "strongest operations:" in output
    assert "strong_op:" in output
    assert "weakest operations:" in output
    assert "weak_op:" in output


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
        coverage_queue=deque(),
    )
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


def test_seen_task_ids_persist_in_policy_state(tmp_path):
    policy = GrammarUCTPolicy(
        seen_task_ids={"a", "b"},
    )
    path = tmp_path / "grammar.json"
    policy.save(path)

    loaded = GrammarUCTPolicy.load(path)

    assert loaded.seen_task_ids == {"a", "b"}


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
    assert result.policy.seen_task_ids == {"first", "second"}
    assert result.policy.total_gene_generations == 0
