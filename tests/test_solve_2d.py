from __future__ import annotations

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
    solve_2dim_1gene_basic,
)


def _matrix_gp(genes):
    sample_count = len(genes[0])
    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=sample_count)

    for index, values in enumerate(genes):
        gidx = X.append_gene(values)
        meta.append(
            source=-1,
            op=f"gp_{index}",
            dims=2,
        )
        assert gidx == index

    return meta, X


def _matrix_sp(targets):
    sample_count = len(targets[0])
    meta = ProgramMeta(side="SP")
    X = ProgramX(side="SP", sample_count=sample_count)

    for index, values in enumerate(targets):
        gidx = X.append_gene(values)
        meta.append(
            source=-1,
            op=f"sp_{index}",
            dims=2,
        )
        assert gidx == index

    ST = SolutionTree.from_sp_meta(meta)
    return meta, X, ST


def _solve(gp_genes, sp_targets):
    GP_meta, GP_X = _matrix_gp(gp_genes)
    SP_meta, SP_X, ST = _matrix_sp(sp_targets)

    GP_pool = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=2,
        max_dim=2,
    )
    ST_frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=2,
        max_dim=2,
    )

    solved = solve_2dim_1gene_basic(
        GP_pool,
        ST_frontier,
        GP_X=GP_X,
        SP_X=SP_X,
        ST=ST,
    )

    return solved, GP_meta, GP_X, SP_meta, SP_X, ST


def _embed(source, shape, row, col):
    result = np.zeros(shape, dtype=source.dtype)
    h, w = source.shape
    result[row:row + h, col:col + w] = source
    return result


def test_2d_identity_has_priority_over_earlier_embed_candidate():
    small = [
        np.array([[True]], dtype=bool),
        np.array([[False, True]], dtype=bool),
    ]
    target = [
        _embed(small[0], (3, 3), 0, 0),
        _embed(small[1], (2, 4), 0, 0),
    ]

    solved, _, _, _, _, ST = _solve(
        [
            small,
            [value.copy() for value in target],
        ],
        [target],
    )

    assert len(solved) == 1
    solution = solved[0]

    assert solution.rule == "identity"
    assert solution.gp_gidx == 1
    assert solution.complexity_level == 0
    assert ST[0].solution_rule == "identity"


def test_2d_structural_top_left_boolean_embed():
    gp = [
        np.array(
            [
                [True, False],
                [True, True],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [False, True],
                [True, False],
            ],
            dtype=bool,
        ),
    ]
    target = [
        _embed(gp[0], (4, 5), 0, 0),
        _embed(gp[1], (3, 4), 0, 0),
    ]

    solved, _, _, _, _, ST = _solve(
        [gp],
        [target],
    )

    assert len(solved) == 1
    solution = solved[0]

    assert solution.rule == "embed_zero_structural"
    assert solution.params == {"position": "top_left"}
    assert solution.complexity_level == 1
    assert solution.target_dtype == np.dtype("bool")
    assert ST[0].solution_rule == "embed_zero_structural"


def test_2d_structural_center_embed():
    gp = [
        np.array([[True]], dtype=bool),
        np.array([[True, True]], dtype=bool),
    ]
    target = [
        _embed(gp[0], (3, 3), 1, 1),
        _embed(gp[1], (3, 4), 1, 1),
    ]

    solved, _, _, _, _, _ = _solve(
        [gp],
        [target],
    )

    assert solved[0].rule == "embed_zero_structural"
    assert solved[0].params == {"position": "center"}


def test_2d_fixed_embed_requires_same_coordinates_across_samples():
    gp = [
        np.array([[1, 2]], dtype=np.int64),
        np.array([[3], [4]], dtype=np.int64),
    ]
    target = [
        _embed(gp[0], (4, 6), 1, 2),
        _embed(gp[1], (5, 5), 1, 2),
    ]

    solved, _, _, _, _, _ = _solve(
        [gp],
        [target],
    )

    assert solved[0].rule == "embed_zero_fixed"
    assert solved[0].params == {"row": 1, "col": 2}
    assert solved[0].complexity_level == 2


def test_2d_fixed_embed_is_not_fit_from_single_sample():
    gp = [
        np.array([[7]], dtype=np.int64),
    ]
    target = [
        _embed(gp[0], (4, 5), 1, 2),
    ]

    solved, _, _, _, _, ST = _solve(
        [gp],
        [target],
    )

    assert solved == []
    assert not ST.is_solved(0)


def test_2d_partial_containment_does_not_count_as_solution():
    gp = [
        np.array(
            [
                [True, True],
                [False, True],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [True, False],
                [True, True],
            ],
            dtype=bool,
        ),
    ]
    target = [
        _embed(gp[0], (4, 4), 1, 1),
        _embed(gp[1], (4, 4), 1, 1),
    ]

    # Add unexplained material outside the embedded GP object.
    target[0][0, 3] = True
    target[1][3, 0] = True

    solved, _, _, _, _, ST = _solve(
        [gp],
        [target],
    )

    assert solved == []
    assert not ST.is_solved(0)


def test_2d_structural_bottom_right_crop():
    gp = [
        np.arange(16, dtype=np.int64).reshape(4, 4),
        np.arange(25, dtype=np.int64).reshape(5, 5),
    ]
    target = [
        gp[0][-2:, -2:].copy(),
        gp[1][-2:, -2:].copy(),
    ]

    solved, _, _, _, _, _ = _solve(
        [gp],
        [target],
    )

    assert solved[0].rule == "crop_structural"
    assert solved[0].params == {"position": "bottom_right"}
    assert solved[0].complexity_level == 3


def test_2d_fixed_crop_uses_same_origin_across_samples():
    gp = [
        np.arange(30, dtype=np.int64).reshape(5, 6),
        np.arange(42, dtype=np.int64).reshape(6, 7),
    ]
    target = [
        gp[0][1:3, 2:5].copy(),
        gp[1][1:3, 2:5].copy(),
    ]

    solved, _, _, _, _, _ = _solve(
        [gp],
        [target],
    )

    assert solved[0].rule == "crop_fixed"
    assert solved[0].params == {"row": 1, "col": 2}
    assert solved[0].complexity_level == 4


def test_2d_zero_fill_shift():
    gp = [
        np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, False],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [False, True, False],
                [False, False, False],
                [False, False, False],
            ],
            dtype=bool,
        ),
    ]
    target = [
        np.array(
            [
                [False, False, False],
                [False, True, False],
                [False, False, True],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [False, False, False],
                [False, False, True],
                [False, False, False],
            ],
            dtype=bool,
        ),
    ]

    solved, _, _, _, _, _ = _solve(
        [gp],
        [target],
    )

    assert solved[0].rule == "shift_zero"
    assert solved[0].params == {"dr": 1, "dc": 1}
    assert solved[0].complexity_level == 5


def test_2d_integer_tiling():
    gp = [
        np.array([[1, 2]], dtype=np.int64),
        np.array([[3, 4]], dtype=np.int64),
    ]
    target = [
        np.tile(gp[0], (2, 3)),
        np.tile(gp[1], (2, 3)),
    ]

    solved, _, _, _, _, ST = _solve(
        [gp],
        [target],
    )

    assert solved[0].rule == "tile"
    assert solved[0].params == {"rows": 2, "cols": 3}
    assert solved[0].complexity_level == 6
    assert ST[0].solution_rule == "tile"


def test_2d_target_dtype_is_retained_for_lossless_cast():
    gp = [
        np.array([[1, 2], [3, 4]], dtype=np.int64),
        np.array([[5, 6], [7, 8]], dtype=np.int64),
    ]
    target = [
        gp[0].astype(np.float32),
        gp[1].astype(np.float32),
    ]

    solved, _, _, _, _, _ = _solve(
        [gp],
        [target],
    )

    assert solved[0].rule == "identity"
    assert solved[0].target_dtype == np.dtype("float32")


def test_2d_solver_rejects_non_2d_pool_data():
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

    SP_meta, SP_X, ST = _matrix_sp(
        [
            [
                np.array([[1]], dtype=np.int64),
                np.array([[2]], dtype=np.int64),
            ]
        ]
    )

    GP_pool = np.empty(1, dtype=object)
    GP_pool[0] = GP_X[0]

    ST_frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=2,
        max_dim=2,
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "2D genes",
    ):
        solve_2dim_1gene_basic(
            GP_pool,
            ST_frontier,
            GP_X=GP_X,
            SP_X=SP_X,
            ST=ST,
        )


def test_2d_evaluation_cache_skips_completed_pair(monkeypatch):
    GP_meta, GP_X = _matrix_gp(
        [
            [
                np.zeros((2, 2), dtype=bool),
                np.zeros((2, 2), dtype=bool),
            ]
        ]
    )
    SP_meta, SP_X, ST = _matrix_sp(
        [
            [
                np.array(
                    [[True, False], [False, False]],
                    dtype=bool,
                ),
                np.array(
                    [[False, True], [False, False]],
                    dtype=bool,
                ),
            ]
        ]
    )

    evaluation_matrix = PairEvaluationMatrix(2)
    calls = 0
    original = solve_module._candidate_2d_rule_solutions

    def counted(x_gene, y_gene):
        nonlocal calls
        calls += 1
        return original(x_gene, y_gene)

    monkeypatch.setattr(
        solve_module,
        "_candidate_2d_rule_solutions",
        counted,
    )

    def run():
        return solve_2dim_1gene_basic(
            get_GP_pool(
                GP_meta,
                GP_X,
                min_dim=2,
                max_dim=2,
            ),
            get_ST_unsovled_frontier(
                ST,
                SP_X,
                min_dim=2,
                max_dim=2,
            ),
            GP_X=GP_X,
            SP_X=SP_X,
            ST=ST,
            GP_meta=GP_meta,
            SP_meta=SP_meta,
            evaluation_matrix=evaluation_matrix,
        )

    assert run() == []
    assert calls == 1
    assert evaluation_matrix.matrix.shape == (1, 1)
    assert evaluation_matrix.matrix[0, 0]

    assert run() == []
    assert calls == 1


def test_2d_cache_preserves_distinct_ids_for_equal_valued_gp_genes():
    duplicate = [
        np.array([[True, False]], dtype=bool),
        np.array([[False, True]], dtype=bool),
    ]
    GP_meta, GP_X = _matrix_gp(
        [
            duplicate,
            [value.copy() for value in duplicate],
        ]
    )
    SP_meta, SP_X, ST = _matrix_sp(
        [
            [
                np.array([[True, True]], dtype=bool),
                np.array([[True, True]], dtype=bool),
            ]
        ]
    )
    evaluation_matrix = PairEvaluationMatrix(2)

    solved = solve_2dim_1gene_basic(
        get_GP_pool(
            GP_meta,
            GP_X,
            min_dim=2,
            max_dim=2,
        ),
        get_ST_unsovled_frontier(
            ST,
            SP_X,
            min_dim=2,
            max_dim=2,
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


def test_2d_solver_direct_ids_bypass_equality_identity_recovery(monkeypatch):
    duplicate = [
        np.array([[True, False]], dtype=bool),
        np.array([[False, True]], dtype=bool),
    ]
    GP_meta, GP_X = _matrix_gp(
        [
            duplicate,
            [value.copy() for value in duplicate],
        ]
    )
    SP_meta, SP_X, ST = _matrix_sp(
        [
            [
                np.array([[True, True]], dtype=bool),
                np.array([[True, True]], dtype=bool),
            ]
        ]
    )
    GP_pool = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=2,
        max_dim=2,
    )
    ST_frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=2,
        max_dim=2,
    )

    monkeypatch.setattr(
        solve_module,
        "_map_pool_to_gp_gidx",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy GP equality mapping was called")
        ),
    )
    monkeypatch.setattr(
        solve_module,
        "_map_2d_frontier_to_sp_gidx",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy SP equality mapping was called")
        ),
    )

    solved = solve_2dim_1gene_basic(
        GP_pool,
        ST_frontier,
        GP_X=GP_X,
        SP_X=SP_X,
        ST=ST,
        GP_meta=GP_meta,
        SP_meta=SP_meta,
        evaluation_matrix=PairEvaluationMatrix(2),
        GP_gidxs=(0, 1),
        SP_gidxs=(0,),
    )

    assert solved == []

