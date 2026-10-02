from __future__ import annotations

import numpy as np
import pytest

from notebooks.ops import (
    Kelschinator,
    get_GP_pool,
    get_ST_unsovled_frontier,
    init_env,
    mat2_cwrotate,
    solve_0dim_1gene_basic,
    solve_2dim_1gene_basic,
)


def _pair(input_grid, output_grid):
    return {
        "input": np.asarray(input_grid, dtype=np.int64),
        "output": np.asarray(output_grid, dtype=np.int64),
    }


def test_kelschinator_rejects_unsolved_st():
    train = [
        _pair(
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
        ),
        _pair(
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
        ),
    ]

    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(train)

    model = Kelschinator()

    assert model.fit(ST) is False
    assert model.is_fitted_ is False
    assert "not fully solved" in model.last_error_


def test_kelschinator_distills_direct_rotated_gp_program():
    inputs = [
        np.array(
            [[1, 2, 3], [4, 5, 6]],
            dtype=np.int64,
        ),
        np.array(
            [[7, 8], [9, 10], [11, 12]],
            dtype=np.int64,
        ),
    ]
    outputs = [
        np.rot90(value, k=-1).copy()
        for value in inputs
    ]
    train = [
        _pair(x, y)
        for x, y in zip(inputs, outputs)
    ]

    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(train)

    rotated_gidx = mat2_cwrotate(
        GP_meta,
        GP_X,
        0,
    )

    GP_pool = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=2,
        max_dim=2,
        dtype=np.int64,
    )
    frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=2,
        max_dim=2,
        dtype=np.int64,
    )

    solutions = solve_2dim_1gene_basic(
        GP_pool,
        frontier,
        GP_X=GP_X,
        SP_X=SP_X,
        ST=ST,
    )

    assert ST.solved
    assert solutions[0].gp_gidx == rotated_gidx
    assert solutions[0].rule == "identity"

    model = Kelschinator()

    assert model.fit(ST) is True
    assert model.is_fitted_
    assert model.pipeline_

    test_input = np.array(
        [
            [13, 14, 15],
            [16, 17, 18],
        ],
        dtype=np.int64,
    )

    expected = np.rot90(
        test_input,
        k=-1,
    ).copy()

    assert np.array_equal(
        model.transform(test_input),
        expected,
    )

    # The fitted model is independent of later ST solution-state mutation.
    ST.clear_solution(0)
    assert np.array_equal(
        model.transform(test_input),
        expected,
    )


def test_kelschinator_distills_boolean_reconstruction_and_scalar_rule():
    train = [
        _pair(
            [[0, 1], [1, 0]],
            [[0, 2], [2, 0]],
        ),
        _pair(
            [[1, 0, 1], [0, 0, 1]],
            [[2, 0, 2], [0, 0, 2]],
        ),
        _pair(
            [[0, 0, 1], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 2], [2, 2, 0], [0, 2, 0]],
        ),
    ]

    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(train)

    scalar_gp = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=0,
        max_dim=0,
        dtype=np.int64,
    )
    scalar_frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=0,
        max_dim=0,
        dtype=np.int64,
    )

    scalar_solutions = solve_0dim_1gene_basic(
        scalar_gp,
        scalar_frontier,
        GP_X=GP_X,
        SP_X=SP_X,
        ST=ST,
    )

    bool_gp = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=2,
        max_dim=2,
        dtype=bool,
    )
    bool_frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=2,
        max_dim=2,
        dtype=bool,
    )

    matrix_solutions = solve_2dim_1gene_basic(
        bool_gp,
        bool_frontier,
        GP_X=GP_X,
        SP_X=SP_X,
        ST=ST,
    )

    assert scalar_solutions
    assert matrix_solutions
    assert ST.solved

    # At least one output color-ID relationship should be 1 -> 2.
    assert any(
        solution.rule in {"add_constant", "scale"}
        for solution in scalar_solutions
    )

    model = Kelschinator()

    assert model.fit(ST) is True, model.last_error_

    test_input = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.int64,
    )
    expected = np.where(
        test_input == 1,
        2,
        test_input,
    ).astype(np.int64)

    prediction = model.transform(test_input)

    assert prediction.dtype == np.int64
    assert np.array_equal(
        prediction,
        expected,
    )


def test_kelschinator_distills_direct_structural_embedding():
    inputs = [
        np.array([[1]], dtype=np.int64),
        np.array([[2, 3]], dtype=np.int64),
    ]
    outputs = [
        np.pad(
            inputs[0],
            ((1, 1), (1, 1)),
            constant_values=0,
        ),
        np.pad(
            inputs[1],
            ((1, 1), (1, 1)),
            constant_values=0,
        ),
    ]
    train = [
        _pair(x, y)
        for x, y in zip(inputs, outputs)
    ]

    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(train)

    GP_pool = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=2,
        max_dim=2,
        dtype=np.int64,
    )
    frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=2,
        max_dim=2,
        dtype=np.int64,
    )

    solutions = solve_2dim_1gene_basic(
        GP_pool,
        frontier,
        GP_X=GP_X,
        SP_X=SP_X,
        ST=ST,
    )

    assert solutions
    assert solutions[0].rule == "embed_zero_structural"
    assert solutions[0].params == {"position": "center"}
    assert ST.solved

    model = Kelschinator()
    assert model.fit(ST) is True, model.last_error_

    test_input = np.array(
        [
            [4, 5],
            [6, 7],
        ],
        dtype=np.int64,
    )
    expected = np.pad(
        test_input,
        ((1, 1), (1, 1)),
        constant_values=0,
    )

    assert np.array_equal(
        model.transform(test_input),
        expected,
    )


def test_kelschinator_transform_requires_fit():
    model = Kelschinator()

    with pytest.raises(RuntimeError, match="successfully fit"):
        model.transform(
            np.zeros((2, 2), dtype=np.int64)
        )
