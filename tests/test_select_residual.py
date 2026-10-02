from __future__ import annotations

import numpy as np
import pytest

from notebooks.ops import (
    Kelschinator,
    ProgramMeta,
    ProgramX,
    STInverseRef,
    STNodeRef,
    STSet,
    init_env,
    inv_partition_select_residual,
    inv_select_residual_leaf,
    inv_select_residual_step,
    partition_select_residual,
)


def _pair(input_grid, output_grid):
    return {
        "input": np.asarray(input_grid, dtype=np.int64),
        "output": np.asarray(output_grid, dtype=np.int64),
    }


def _color_gene_map(meta, X):
    result = {}

    for gidx, op_name in enumerate(meta.op):
        if op_name != "partition_composite":
            continue
        if meta.dims[gidx] != 0:
            continue

        values = [
            int(np.asarray(value).item())
            for value in X[gidx]
        ]

        if values and len(set(values)) == 1:
            result[values[0]] = gidx

    return result


def _color_mask_map(meta, X):
    scalar = _color_gene_map(meta, X)
    result = {}

    for color, color_gidx in scalar.items():
        mask_gidx = color_gidx + 1

        if (
            mask_gidx < len(meta)
            and meta.op[mask_gidx] == "partition_composite"
            and meta.dims[mask_gidx] == 2
        ):
            result[color] = mask_gidx

    return result


def test_partition_select_residual_materializes_shared_subset_supports():
    inputs = [
        np.zeros((2, 3), dtype=np.int64),
        np.zeros((2, 3), dtype=np.int64),
    ]
    outputs = [
        np.array(
            [
                [0, 1, 2],
                [0, 1, 2],
            ],
            dtype=np.int64,
        ),
        np.array(
            [
                [2, 1, 0],
                [2, 1, 0],
            ],
            dtype=np.int64,
        ),
    ]

    meta = ProgramMeta(side="SP")
    X = ProgramX(side="SP", sample_count=2)
    X.append_gene(outputs)
    meta.append(source=-1, op="raw_output", dims=2)

    partition = partition_select_residual(
        meta,
        X,
        inputs,
    )

    assert partition is not None
    assert partition.source_destinations == {
        0: (0, 1, 2),
    }

    # Every non-empty subset of three destinations appears once.
    assert len(partition.support_gidxs) == 7

    full = partition.support_gidxs[
        (0, (0, 1, 2))
    ]
    residual_12 = partition.support_gidxs[
        (0, (1, 2))
    ]
    residual_2 = partition.support_gidxs[
        (0, (2,))
    ]

    assert np.all(X[full, 0])
    assert np.all(X[full, 1])

    assert np.array_equal(
        X[residual_12, 0],
        np.array(
            [
                [False, True, True],
                [False, True, True],
            ],
            dtype=bool,
        ),
    )

    assert np.array_equal(
        X[residual_2, 0],
        np.array(
            [
                [False, False, True],
                [False, False, True],
            ],
            dtype=bool,
        ),
    )

    assert meta.params[residual_12] == {
        "source_color": 0,
        "destinations": (1, 2),
    }


def test_partition_select_residual_skips_unaligned_shapes_without_mutation():
    inputs = [
        np.zeros((2, 2), dtype=np.int64),
    ]
    outputs = [
        np.zeros((3, 2), dtype=np.int64),
    ]

    meta = ProgramMeta(side="SP")
    X = ProgramX(side="SP", sample_count=1)
    X.append_gene(outputs)
    meta.append(source=-1, op="raw_output", dims=2)

    before = len(X)

    partition = partition_select_residual(
        meta,
        X,
        inputs,
    )

    assert partition is None
    assert len(X) == before
    assert len(meta) == before


def test_init_env_builds_recursive_select_residual_dag():
    train = [
        _pair(
            [
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 1, 2],
                [0, 1, 2],
            ],
        ),
        _pair(
            [
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [2, 1, 0],
                [2, 1, 0],
            ],
        ),
    ]

    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(train)

    assert isinstance(ST[0].derivation, STSet)
    assert ST[0].derivation.mode == "OR"
    assert len(ST[0].derivation.members) == 2

    select_branch = ST[0].derivation.members[1]

    assert isinstance(select_branch, STSet)
    assert select_branch.mode == "AND"
    assert select_branch.members[0] == STInverseRef(
        "inv_partition_select_residual"
    )

    top_id = "select_residual:x=0:A=0,1,2"
    residual_id = "select_residual:x=0:A=1,2"
    leaf_id = "select_residual:x=0:A=2"

    assert select_branch.members[1:] == [
        STNodeRef(top_id),
    ]

    top = ST[top_id].derivation
    assert isinstance(top, STSet)
    assert top.mode == "OR"
    assert len(top.members) == 3

    # Select destination 0 as the base and carry {1,2}.
    select_zero = top.members[0]
    assert isinstance(select_zero, STSet)
    assert select_zero.members[0] == STInverseRef(
        "inv_select_residual_step"
    )
    assert select_zero.members[-1] == STNodeRef(
        residual_id
    )

    residual = ST[residual_id].derivation
    assert isinstance(residual, STSet)
    assert residual.mode == "OR"
    assert len(residual.members) == 2

    # Its first branch selects 1 and carries singleton {2}.
    select_one = residual.members[0]
    assert select_one.members[0] == STInverseRef(
        "inv_select_residual_step"
    )
    assert select_one.members[-1] == STNodeRef(
        leaf_id
    )

    leaf = ST[leaf_id].derivation
    assert isinstance(leaf, STSet)
    assert len(leaf.members) == 1
    assert leaf.members[0].members[0] == STInverseRef(
        "inv_select_residual_leaf"
    )


def test_select_residual_boolean_path_solves_root_with_nested_supports():
    train = [
        _pair(
            [[0, 0, 0]],
            [[0, 1, 2]],
        ),
        _pair(
            [[0, 0, 0]],
            [[2, 1, 0]],
        ),
    ]

    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(train)

    output_colors = _color_gene_map(SP_meta, SP_X)

    support_by_subset = {}

    for gidx, op_name in enumerate(SP_meta.op):
        if op_name != "partition_select_residual":
            continue

        params = SP_meta.params[gidx]

        if params["source_color"] == 0:
            support_by_subset[
                tuple(params["destinations"])
            ] = gidx

    # Choose the path:
    #   base 0 on {0,1,2}
    #   base 1 on {1,2}
    #   base 2 on {2}
    #
    # This requires only the carried supports for those nested subsets,
    # rather than all three direct transition masks.
    for color in (0, 1, 2):
        ST.mark_solution(
            output_colors[color],
            100 + color,
        )

    for subset in (
        (0, 1, 2),
        (1, 2),
        (2,),
    ):
        ST.mark_solution(
            support_by_subset[subset],
            200 + len(subset),
        )

    assert ST.is_solved(
        "select_residual:x=0:A=2"
    )
    assert ST.is_solved(
        "select_residual:x=0:A=1,2"
    )
    assert ST.is_solved(
        "select_residual:x=0:A=0,1,2"
    )
    assert ST.solved

    # The ordinary output presence masks were never solved.
    output_masks = _color_mask_map(
        SP_meta,
        SP_X,
    )
    assert all(
        ST[gidx].gp_gidx == -1
        for gidx in output_masks.values()
    )


def test_select_residual_inverses_reconstruct_nested_paint_stack():
    support_all = np.ones((2, 3), dtype=bool)
    support_12 = np.array(
        [
            [False, True, True],
            [False, True, True],
        ],
        dtype=bool,
    )
    support_2 = np.array(
        [
            [False, False, True],
            [False, False, True],
        ],
        dtype=bool,
    )

    layer_2 = inv_select_residual_leaf(
        np.int64(2),
        support_2,
    )
    layer_12 = inv_select_residual_step(
        np.int64(1),
        support_12,
        layer_2,
    )
    layer_012 = inv_select_residual_step(
        np.int64(0),
        support_all,
        layer_12,
    )

    result = inv_partition_select_residual(
        layer_012,
    )

    assert np.array_equal(
        result,
        np.array(
            [
                [0, 1, 2],
                [0, 1, 2],
            ],
            dtype=np.int64,
        ),
    )


def test_select_residual_step_rejects_residual_outside_parent_support():
    support = np.array(
        [[True, False]],
        dtype=bool,
    )
    residual = np.array(
        [[-1, 2]],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="contained",
    ):
        inv_select_residual_step(
            np.int64(1),
            support,
            residual,
        )


def test_kelschinator_can_compile_select_residual_root_alternative():
    train = [
        _pair(
            [
                [0, 1, 0],
                [1, 0, 1],
            ],
            [
                [0, 2, 0],
                [2, 0, 2],
            ],
        ),
        _pair(
            [
                [1, 0],
                [0, 1],
                [1, 0],
            ],
            [
                [2, 0],
                [0, 2],
                [2, 0],
            ],
        ),
    ]

    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(train)

    gp_colors = _color_gene_map(
        GP_meta,
        GP_X,
    )
    gp_masks = _color_mask_map(
        GP_meta,
        GP_X,
    )
    sp_colors = _color_gene_map(
        SP_meta,
        SP_X,
    )

    supports = {}

    for gidx, op_name in enumerate(SP_meta.op):
        if op_name != "partition_select_residual":
            continue

        params = SP_meta.params[gidx]
        key = (
            params["source_color"],
            tuple(params["destinations"]),
        )
        supports[key] = gidx

    # Solve destination colors from their corresponding input colors.
    ST.mark_solution(
        sp_colors[0],
        gp_colors[0],
        rule="identity",
    )
    ST.mark_solution(
        sp_colors[2],
        gp_colors[1],
        rule="add_constant",
        params={"c": 1},
    )

    # Each source has one destination, so its full support is the only
    # select-residual mask required.  These are exactly the input color masks.
    ST.mark_solution(
        supports[(0, (0,))],
        gp_masks[0],
        rule="identity",
    )
    ST.mark_solution(
        supports[(1, (2,))],
        gp_masks[1],
        rule="identity",
    )

    # Leave the absolute output composite masks unresolved so the only complete
    # root proof is the select-residual alternative.
    output_masks = _color_mask_map(
        SP_meta,
        SP_X,
    )
    assert all(
        ST[gidx].gp_gidx == -1
        for gidx in output_masks.values()
    )
    assert ST.solved

    model = Kelschinator()

    assert model.fit(ST), model.last_error_

    test_input = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    expected = np.where(
        test_input == 1,
        2,
        0,
    ).astype(np.int64)

    assert np.array_equal(
        model.transform(test_input),
        expected,
    )
