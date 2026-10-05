from __future__ import annotations

from fractions import Fraction

import numpy as np

from notebooks.ops.environment import (
    ProgramMeta,
    ProgramX,
    SolutionTree,
    get_GP_pool,
    get_ST_unsovled_frontier,
)
import notebooks.ops.solve as solve_module
from notebooks.ops.solve import (
    PairEvaluationMatrix,
    solve_0dim_1gene_basic,
)


def _scalar_gp(genes):
    sample_count = len(genes[0])
    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=sample_count)

    for index, values in enumerate(genes):
        gidx = X.append_gene(values)
        meta.append(
            source=-1,
            op=f"gp_{index}",
            dims=0,
        )
        assert gidx == index

    return meta, X


def _scalar_sp(targets):
    sample_count = len(targets[0])
    meta = ProgramMeta(side="SP")
    X = ProgramX(side="SP", sample_count=sample_count)

    for index, values in enumerate(targets):
        gidx = X.append_gene(values)
        meta.append(
            source=-1,
            op=f"sp_{index}",
            dims=0,
        )
        assert gidx == index

    ST = SolutionTree.from_sp_meta(meta)
    return meta, X, ST


def _solve(gp_genes, sp_targets):
    GP_meta, GP_X = _scalar_gp(gp_genes)
    SP_meta, SP_X, ST = _scalar_sp(sp_targets)

    GP_pool = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=0,
        max_dim=0,
    )
    ST_frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=0,
        max_dim=0,
    )

    solved = solve_0dim_1gene_basic(
        GP_pool,
        ST_frontier,
        GP_X=GP_X,
        SP_X=SP_X,
        ST=ST,
    )

    return solved, GP_meta, GP_X, SP_meta, SP_X, ST


def test_identity_has_highest_priority_even_when_later_gp_gene():
    solved, _, _, _, _, ST = _solve(
        [
            [
                np.int64(0),
                np.int64(1),
                np.int64(2),
            ],
            [
                np.int64(10),
                np.int64(11),
                np.int64(12),
            ],
        ],
        [
            [
                np.int64(10),
                np.int64(11),
                np.int64(12),
            ]
        ],
    )

    assert len(solved) == 1
    solution = solved[0]

    assert solution.rule == "identity"
    assert solution.gp_gidx == 1
    assert solution.complexity_level == 0
    assert ST[0].gp_gidx == 1
    assert ST[0].solution_rule == "identity"
    assert ST[0].solution_params == {}


def test_square_rule_is_available_and_preserves_int_dtype():
    solved, _, _, _, _, ST = _solve(
        [
            [
                np.int64(2),
                np.int64(3),
                np.int64(4),
            ]
        ],
        [
            [
                np.int64(4),
                np.int64(9),
                np.int64(16),
            ]
        ],
    )

    assert len(solved) == 1
    solution = solved[0]

    assert solution.rule == "square"
    assert solution.target_dtype == np.dtype("int64")
    assert solution.complexity_level == 3
    assert ST[0].solution_rule == "square"


def test_abs_and_negate_rules_are_exact_parameter_free_matches():
    abs_solved, _, _, _, _, _ = _solve(
        [
            [
                np.int64(-1),
                np.int64(-2),
                np.int64(3),
            ]
        ],
        [
            [
                np.int64(1),
                np.int64(2),
                np.int64(3),
            ]
        ],
    )
    assert abs_solved[0].rule == "abs"

    negate_solved, _, _, _, _, _ = _solve(
        [
            [
                np.int64(1),
                np.int64(-2),
                np.int64(3),
            ]
        ],
        [
            [
                np.int64(-1),
                np.int64(2),
                np.int64(-3),
            ]
        ],
    )
    assert negate_solved[0].rule == "negate"


def test_add_constant_fits_exactly_and_records_parameter():
    solved, _, _, _, _, ST = _solve(
        [
            [
                np.int64(1),
                np.int64(4),
                np.int64(8),
            ]
        ],
        [
            [
                np.int64(3),
                np.int64(6),
                np.int64(10),
            ]
        ],
    )

    assert solved[0].rule == "add_constant"
    assert solved[0].params == {"c": Fraction(2, 1)}
    assert ST[0].solution_params == {"c": Fraction(2, 1)}


def test_constant_minus_x_and_scale_are_available():
    reflected, _, _, _, _, _ = _solve(
        [
            [
                np.int64(1),
                np.int64(2),
                np.int64(4),
            ]
        ],
        [
            [
                np.int64(9),
                np.int64(8),
                np.int64(6),
            ]
        ],
    )

    assert reflected[0].rule == "constant_minus_x"
    assert reflected[0].params == {"c": Fraction(10, 1)}

    scaled, _, _, _, _, _ = _solve(
        [
            [
                np.int64(1),
                np.int64(3),
                np.int64(5),
            ]
        ],
        [
            [
                np.int64(2),
                np.int64(6),
                np.int64(10),
            ]
        ],
    )

    assert scaled[0].rule == "scale"
    assert scaled[0].params == {"c": Fraction(2, 1)}


def test_affine_requires_three_samples_and_solves_exact_line():
    solved, _, _, _, _, ST = _solve(
        [
            [
                np.int64(1),
                np.int64(2),
                np.int64(4),
            ]
        ],
        [
            [
                np.int64(5),
                np.int64(8),
                np.int64(14),
            ]
        ],
    )

    assert solved[0].rule == "affine"
    assert solved[0].params == {
        "a": Fraction(3, 1),
        "b": Fraction(2, 1),
    }
    assert ST[0].solution_rule == "affine"

    unsolved, _, _, _, _, ST2 = _solve(
        [
            [
                np.int64(1),
                np.int64(2),
            ]
        ],
        [
            [
                np.int64(5),
                np.int64(8),
            ]
        ],
    )

    assert unsolved == []
    assert not ST2.is_solved(0)


def test_float_frontier_dtype_is_preserved():
    solved, _, _, _, _, ST = _solve(
        [
            [
                np.int64(1),
                np.int64(2),
                np.int64(3),
            ]
        ],
        [
            [
                np.float32(1.5),
                np.float32(2.5),
                np.float32(3.5),
            ]
        ],
    )

    assert len(solved) == 1
    solution = solved[0]

    assert solution.rule == "add_constant"
    assert solution.target_dtype == np.dtype("float32")
    assert ST[0].solution_rule == "add_constant"


def test_non_numeric_scalar_only_uses_identity():
    solved, _, _, _, _, ST = _solve(
        [
            [
                np.str_("a"),
                np.str_("b"),
                np.str_("c"),
            ]
        ],
        [
            [
                np.str_("a"),
                np.str_("b"),
                np.str_("c"),
            ]
        ],
    )

    assert solved[0].rule == "identity"
    assert ST.is_solved(0)


def test_solver_can_solve_multiple_frontier_targets():
    GP_meta, GP_X = _scalar_gp(
        [
            [
                np.int64(2),
                np.int64(3),
                np.int64(4),
            ],
            [
                np.int64(10),
                np.int64(20),
                np.int64(30),
            ],
        ]
    )
    SP_meta, SP_X, ST = _scalar_sp(
        [
            [
                np.int64(4),
                np.int64(9),
                np.int64(16),
            ],
            [
                np.int64(10),
                np.int64(20),
                np.int64(30),
            ],
        ]
    )

    # Make both concrete SP nodes part of one AND proof under root 0.
    # Root 0 is still directly solvable, but its derived branch references node 1.
    from notebooks.ops.environment import STNodeRef, STSet

    ST[0].derivation = STSet(
        mode="OR",
        members=[
            STSet(
                mode="AND",
                members=[STNodeRef(1)],
                partition="and",
            )
        ],
    )

    GP_pool = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=0,
        max_dim=0,
    )
    ST_frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=0,
        max_dim=0,
    )

    solved = solve_0dim_1gene_basic(
        GP_pool,
        ST_frontier,
        GP_X=GP_X,
        SP_X=SP_X,
        ST=ST,
    )

    assert len(solved) == 2

    by_sp = {
        solution.sp_gidx: solution
        for solution in solved
    }

    assert by_sp[0].rule == "square"
    assert by_sp[1].rule == "identity"
    assert ST.is_solved(0)


def test_solver_rejects_non_0dim_pool_data():
    GP_meta = ProgramMeta(side="GP")
    GP_X = ProgramX(side="GP", sample_count=2)
    GP_X.append_gene(
        [
            np.array([1, 2], dtype=np.int64),
            np.array([3, 4], dtype=np.int64),
        ]
    )
    GP_meta.append(
        source=-1,
        op="vector",
        dims=1,
    )

    SP_meta, SP_X, ST = _scalar_sp(
        [
            [
                np.int64(1),
                np.int64(2),
            ]
        ]
    )

    GP_pool = np.empty(1, dtype=object)
    GP_pool[0] = GP_X[0]

    ST_frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=0,
        max_dim=0,
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "0D genes",
    ):
        solve_0dim_1gene_basic(
            GP_pool,
            ST_frontier,
            GP_X=GP_X,
            SP_X=SP_X,
            ST=ST,
        )


def test_pair_evaluation_matrix_sync_preserves_only_stable_id_overlap():
    matrix = PairEvaluationMatrix(0)

    matrix.sync([10, 11], [20])
    matrix.mark_evaluated(10, 20)

    matrix.sync([10, 11, 12], [20, 21])

    assert matrix.gp_gene_ids == (10, 11, 12)
    assert matrix.sp_gene_ids == (20, 21)
    assert matrix.matrix.shape == (3, 2)
    assert matrix.was_evaluated(10, 20)
    assert not matrix.was_evaluated(11, 20)
    assert not matrix.was_evaluated(12, 20)
    assert not matrix.was_evaluated(10, 21)

    matrix.sync([11, 12], [20, 21])

    assert matrix.gp_gene_ids == (11, 12)
    assert matrix.matrix.shape == (2, 2)
    assert not np.any(matrix.matrix)


def test_0d_evaluation_cache_skips_completed_pairs_and_only_evaluates_new_gp(
    monkeypatch,
):
    GP_meta, GP_X = _scalar_gp(
        [
            [np.int64(1), np.int64(2), np.int64(3)],
            [np.int64(4), np.int64(5), np.int64(6)],
        ]
    )
    SP_meta, SP_X, ST = _scalar_sp(
        [
            [np.int64(100), np.int64(101), np.int64(105)],
        ]
    )

    evaluation_matrix = PairEvaluationMatrix(0)
    calls = 0
    original = solve_module._candidate_rule_solutions

    def counted(x_gene, y_gene):
        nonlocal calls
        calls += 1
        return original(x_gene, y_gene)

    monkeypatch.setattr(
        solve_module,
        "_candidate_rule_solutions",
        counted,
    )

    def run():
        return solve_0dim_1gene_basic(
            get_GP_pool(
                GP_meta,
                GP_X,
                min_dim=0,
                max_dim=0,
            ),
            get_ST_unsovled_frontier(
                ST,
                SP_X,
                min_dim=0,
                max_dim=0,
            ),
            GP_X=GP_X,
            SP_X=SP_X,
            ST=ST,
            GP_meta=GP_meta,
            SP_meta=SP_meta,
            evaluation_matrix=evaluation_matrix,
        )

    assert run() == []
    assert calls == 2
    assert evaluation_matrix.matrix.shape == (2, 1)
    assert np.all(evaluation_matrix.matrix)

    assert run() == []
    assert calls == 2

    GP_X.append_gene(
        [np.int64(7), np.int64(8), np.int64(10)]
    )
    GP_meta.append(
        source=0,
        op="new_gp",
        dims=0,
    )

    assert run() == []
    assert calls == 3
    assert evaluation_matrix.matrix.shape == (3, 1)
    assert np.all(evaluation_matrix.matrix)


def test_0d_cache_preserves_distinct_ids_for_equal_valued_gp_genes():
    GP_meta, GP_X = _scalar_gp(
        [
            [np.int64(2), np.int64(3), np.int64(4)],
            [np.int64(2), np.int64(3), np.int64(4)],
        ]
    )
    SP_meta, SP_X, ST = _scalar_sp(
        [
            [np.int64(100), np.int64(101), np.int64(105)],
        ]
    )
    evaluation_matrix = PairEvaluationMatrix(0)

    solved = solve_0dim_1gene_basic(
        get_GP_pool(
            GP_meta,
            GP_X,
            min_dim=0,
            max_dim=0,
        ),
        get_ST_unsovled_frontier(
            ST,
            SP_X,
            min_dim=0,
            max_dim=0,
        ),
        GP_X=GP_X,
        SP_X=SP_X,
        ST=ST,
        GP_meta=GP_meta,
        SP_meta=SP_meta,
        evaluation_matrix=evaluation_matrix,
    )

    assert solved == []
    assert evaluation_matrix.gp_gene_ids == (
        GP_meta.stable_id(0),
        GP_meta.stable_id(1),
    )
    assert evaluation_matrix.matrix.shape == (2, 1)
    assert np.all(evaluation_matrix.matrix)


def test_0d_cache_preserves_distinct_ids_for_equal_valued_sp_frontier_genes():
    GP_meta, GP_X = _scalar_gp(
        [
            [np.int64(1), np.int64(2), np.int64(3)],
        ]
    )
    SP_meta, SP_X, ST = _scalar_sp(
        [
            [np.int64(50), np.int64(60), np.int64(70)],
            [np.int64(50), np.int64(60), np.int64(70)],
        ]
    )

    from notebooks.ops.environment import STNode, STNodeRef, STSet

    ST.roots = ("root_equal_targets",)
    ST.nodes["root_equal_targets"] = STNode(
        node_id="root_equal_targets",
        label="root equal targets",
        sp_gidx=None,
        op="logical_test",
        dims=0,
        derivation=STSet(
            mode="AND",
            members=[
                STNodeRef(0),
                STNodeRef(1),
            ],
            partition="and",
        ),
    )

    evaluation_matrix = PairEvaluationMatrix(0)
    frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=0,
        max_dim=0,
    )

    assert len(frontier) == 2

    solve_0dim_1gene_basic(
        get_GP_pool(
            GP_meta,
            GP_X,
            min_dim=0,
            max_dim=0,
        ),
        frontier,
        GP_X=GP_X,
        SP_X=SP_X,
        ST=ST,
        GP_meta=GP_meta,
        SP_meta=SP_meta,
        evaluation_matrix=evaluation_matrix,
    )

    assert evaluation_matrix.sp_gene_ids == (
        SP_meta.stable_id(0),
        SP_meta.stable_id(1),
    )

