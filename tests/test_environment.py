from types import SimpleNamespace

import numpy as np
import pytest

from notebooks.ops.environment import (
    ProgramMeta,
    ProgramX,
    STInverseRef,
    STNodeRef,
    STSet,
    SolutionTree,
    get_GP_pool,
    get_GP_pool_gidxs,
    get_ST_unsolved_frontier,
    get_ST_unsolved_frontier_gidxs,
    get_ST_unsovled_frontier,
    get_ST_unsovled_frontier_gidxs,
    init_env,
)
from notebooks.ops.inv_ops import (
    INV_OP_REGISTRY,
    inverse_operation,
)
from notebooks.ops.ops import (
    GP_generate,
    GP_prune,
    OP_REGISTRY,
    SP_generate,
    bool2_intersect,
    bool2_union,
    bool_cavity,
    bool_complement,
    bool_mat_ident,
    bool_sum,
    dim0_flip,
    dim1_flip,
    dim2_flip,
    equivalent_gene_idx,
    generation_exists,
    gene_atomic_dtypes,
    genes_exactly_equal,
    indiv_1dim,
    mat2_cwrotate,
    operation,
    operation_output_count,
    partition_bool_subjects,
    partition_bool_trim,
    partition_composite,
    partition_shape,
    valid_generation,
    _sample_uniform_legal_candidate,
    _source_selection_probabilities,
    _gp_prune_probabilities,
)


def _train_pairs():
    return [
        SimpleNamespace(
            input=np.array(
                [
                    [0, 1, 1],
                    [0, 0, 1],
                ]
            ),
            output=np.array(
                [
                    [0, 2],
                    [2, 2],
                    [0, 0],
                ]
            ),
        ),
        SimpleNamespace(
            input=np.array(
                [
                    [3, 0],
                    [3, 3],
                    [0, 0],
                ]
            ),
            output=np.array(
                [
                    [4, 4, 0],
                    [0, 4, 0],
                ]
            ),
        ),
    ]


def test_init_env_returns_five_components():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    assert isinstance(GP_meta, ProgramMeta)
    assert isinstance(GP_X, ProgramX)
    assert isinstance(SP_meta, ProgramMeta)
    assert isinstance(SP_X, ProgramX)
    assert isinstance(ST, SolutionTree)

    assert GP_meta.side == "GP"
    assert GP_X.side == "GP"
    assert SP_meta.side == "SP"
    assert SP_X.side == "SP"


def test_program_x_is_gene_major_then_sample_major():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    assert GP_X.shape == (9, 2)
    assert SP_X.shape == (9, 2)

    assert np.array_equal(
        GP_X[0, 0],
        _train_pairs()[0].input,
    )
    assert np.array_equal(
        GP_X[0, 1],
        _train_pairs()[1].input,
    )

    assert np.array_equal(
        SP_X[0, 0],
        _train_pairs()[0].output,
    )
    assert np.array_equal(
        SP_X[0, 1],
        _train_pairs()[1].output,
    )


def test_initial_metadata_is_parallel_to_gene_indices():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    assert GP_meta.op == [
        "raw_input",
        "partition_shape",
        "partition_shape",
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
    ]
    assert GP_meta.source == [-1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert GP_meta.dims == [2, 0, 0, 0, 2, 0, 2, 0, 2]
    assert GP_meta.params == [{}, {}, {}, {}, {}, {}, {}, {}, {}]

    assert SP_meta.op == [
        "raw_output",
        "partition_shape",
        "partition_shape",
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
    ]
    assert SP_meta.source == [-1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert SP_meta.dims == [2, 0, 0, 0, 2, 0, 2, 0, 2]
    assert SP_meta.params == [{}, {}, {}, {}, {}, {}, {}, {}, {}]

    assert len(GP_meta) == len(GP_X) == 9
    assert len(SP_meta) == len(SP_X) == 9


def test_partition_shape_generates_h_and_w_scalar_genes():
    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=2)

    X.append_gene(
        [
            np.zeros((2, 3), dtype=np.int64),
            np.zeros((5, 4), dtype=np.int64),
        ]
    )
    meta.append(source=-1, op="raw_input", dims=2)

    h_gidx, w_gidx = partition_shape(meta, X, 0)

    assert (h_gidx, w_gidx) == (1, 2)

    assert X[h_gidx, 0] == np.int64(2)
    assert X[h_gidx, 1] == np.int64(5)
    assert X[w_gidx, 0] == np.int64(3)
    assert X[w_gidx, 1] == np.int64(4)

    assert meta.source == [-1, 0, 0]
    assert meta.op == [
        "raw_input",
        "partition_shape",
        "partition_shape",
    ]
    assert meta.dims == [2, 0, 0]


def test_indiv_1dim_generates_one_scalar_gene_per_position():
    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=2)

    X.append_gene(
        [
            np.array([10, 20, 30], dtype=np.int64),
            np.array([40, 50, 60], dtype=np.int64),
        ]
    )
    meta.append(source=-1, op="raw_input", dims=1)

    assert valid_generation(meta, X, indiv_1dim, 0)
    assert operation_output_count(indiv_1dim, meta, X, 0) == 3

    generated = indiv_1dim(meta, X, 0)

    assert generated == (1, 2, 3)
    assert meta.source == [-1, 0, 0, 0]
    assert meta.op == [
        "raw_input",
        "indiv_1dim",
        "indiv_1dim",
        "indiv_1dim",
    ]
    assert meta.dims == [1, 0, 0, 0]

    assert X[1, 0] == np.int64(10)
    assert X[1, 1] == np.int64(40)
    assert X[2, 0] == np.int64(20)
    assert X[2, 1] == np.int64(50)
    assert X[3, 0] == np.int64(30)
    assert X[3, 1] == np.int64(60)


def test_indiv_1dim_requires_exactly_1d_and_consistent_length():
    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=2)

    X.append_gene(
        [
            np.array([1, 2], dtype=np.int64),
            np.array([3, 4, 5], dtype=np.int64),
        ]
    )
    meta.append(source=-1, op="raw_input", dims=1)

    assert not valid_generation(meta, X, indiv_1dim, 0)

    matrix_meta, matrix_X = _raw_program(
        "GP",
        [np.array([[1, 2]], dtype=np.int64)],
        raw_op="raw_input",
    )

    assert not valid_generation(
        matrix_meta,
        matrix_X,
        indiv_1dim,
        0,
    )


def test_indiv_1dim_sp_generation_creates_and_branch_inside_source_or():
    meta = ProgramMeta(side="SP")
    X = ProgramX(side="SP", sample_count=2)

    X.append_gene(
        [
            np.array([1, 2, 3], dtype=np.int64),
            np.array([4, 5, 6], dtype=np.int64),
        ]
    )
    meta.append(source=-1, op="raw_output", dims=1)

    ST = SolutionTree.from_sp_meta(meta)

    generated = SP_generate(
        meta,
        X,
        ST,
        rng=0,
        operation_names=("indiv_1dim",),
    )

    assert generated == [1, 2, 3]

    source_derivation = ST[0].derivation
    assert isinstance(source_derivation, STSet)
    assert source_derivation.mode == "OR"

    branch = source_derivation.members[0]
    assert isinstance(branch, STSet)
    assert branch.mode == "AND"
    assert branch.partition == "and"
    assert branch.members == [
        STInverseRef("inv_indiv_1dim"),
        STNodeRef(1),
        STNodeRef(2),
        STNodeRef(3),
    ]

    assert not ST.is_solved(0)

    ST.mark_solution(1, 101)
    ST.mark_solution(2, 102)
    assert not ST.is_solved(0)

    ST.mark_solution(3, 103)
    assert ST.is_solved(0)


def test_partition_composite_generates_one_id_mask_pair_per_color():
    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=2)

    X.append_gene(
        [
            np.array([[0, 1], [1, 0]], dtype=np.int64),
            np.array([[2, 2, 0]], dtype=np.int64),
        ]
    )
    meta.append(source=-1, op="raw_input", dims=2)

    generated = partition_composite(meta, X, 0)

    assert generated == (1, 2, 3, 4, 5, 6)

    color0, mask0, color1, mask1, color2, mask2 = generated

    assert X[color0, 0] == np.int64(0)
    assert X[color0, 1] == np.int64(0)
    assert X[color1, 0] == np.int64(1)
    assert X[color1, 1] == np.int64(1)
    assert X[color2, 0] == np.int64(2)
    assert X[color2, 1] == np.int64(2)

    assert isinstance(X[color0, 0], np.int64)
    assert isinstance(X[color1, 0], np.int64)
    assert isinstance(X[color2, 0], np.int64)

    assert np.array_equal(
        X[mask0, 0],
        np.array([[True, False], [False, True]], dtype=bool),
    )
    assert np.array_equal(
        X[mask0, 1],
        np.array([[False, False, True]], dtype=bool),
    )

    assert np.array_equal(
        X[mask1, 0],
        np.array([[False, True], [True, False]], dtype=bool),
    )
    assert np.array_equal(
        X[mask1, 1],
        np.array([[False, False, False]], dtype=bool),
    )

    assert np.array_equal(
        X[mask2, 0],
        np.array([[False, False], [False, False]], dtype=bool),
    )
    assert np.array_equal(
        X[mask2, 1],
        np.array([[True, True, False]], dtype=bool),
    )

    assert meta.source == [-1, 0, 0, 0, 0, 0, 0]
    assert meta.op == [
        "raw_input",
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
    ]
    assert meta.dims == [2, 0, 2, 0, 2, 0, 2]


def test_partition_composite_uses_union_of_colors_across_samples():
    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=2)

    X.append_gene(
        [
            np.array([[1, 1]], dtype=np.int64),
            np.array([[2, 2]], dtype=np.int64),
        ]
    )
    meta.append(source=-1, op="raw_input", dims=2)

    generated = partition_composite(meta, X, 0)

    assert generated == (1, 2, 3, 4)

    assert X[1, 0] == X[1, 1] == np.int64(1)
    assert np.array_equal(X[2, 0], np.array([[True, True]]))
    assert np.array_equal(X[2, 1], np.array([[False, False]]))

    assert X[3, 0] == X[3, 1] == np.int64(2)
    assert np.array_equal(X[4, 0], np.array([[False, False]]))
    assert np.array_equal(X[4, 1], np.array([[True, True]]))


def test_partition_composite_dynamic_output_count_matches_color_union():
    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=2)

    X.append_gene(
        [
            np.array([[0, 1]], dtype=np.int64),
            np.array([[0, 2]], dtype=np.int64),
        ]
    )
    meta.append(source=-1, op="raw_input", dims=2)

    assert OP_REGISTRY["partition_composite"].output_count is None
    assert operation_output_count(
        partition_composite,
        meta,
        X,
        0,
    ) == 6


def test_default_partition_ops_are_and_partitions_with_inverses():
    assert partition_shape.partition == "and"
    assert partition_composite.partition == "and"

    assert OP_REGISTRY["partition_shape"].partition == "and"
    assert OP_REGISTRY["partition_shape"].inverse_op == "inv_partition_shape"

    assert OP_REGISTRY["partition_composite"].partition == "and"
    assert (
        OP_REGISTRY["partition_composite"].inverse_op
        == "inv_partition_composite"
    )


def test_null_partition_operation_is_allowed_on_gp_but_rejected_on_sp():
    @operation(partition="null", output_count=1)
    def test_nonpartition(meta, X, source_idx):
        values = [
            np.asarray(value).copy()
            for value in X[source_idx]
        ]
        gidx = X.append_gene(values)
        meta.append(
            source=source_idx,
            op="test_nonpartition",
            dims=meta.dims[source_idx],
        )
        return gidx

    gp_meta = ProgramMeta(side="GP")
    gp_x = ProgramX(side="GP", sample_count=1)
    gp_x.append_gene([np.array([[1]])])
    gp_meta.append(source=-1, op="raw_input", dims=2)

    sp_meta = ProgramMeta(side="SP")
    sp_x = ProgramX(side="SP", sample_count=1)
    sp_x.append_gene([np.array([[1]])])
    sp_meta.append(source=-1, op="raw_output", dims=2)

    assert test_nonpartition(gp_meta, gp_x, 0) == 1

    with pytest.raises(PermissionError, match="NULL partition"):
        test_nonpartition(sp_meta, sp_x, 0)

    assert len(sp_meta) == 1
    assert len(sp_x) == 1

    OP_REGISTRY.pop("test_nonpartition", None)


def test_solution_tree_initializes_shape_as_h_and_w_boolean_proof():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    assert ST.roots == (0,)

    # Nine concrete SP genes plus logical shape and composite nodes.
    assert len(ST) == 11
    assert "shape" in ST.nodes
    assert "composite" in ST.nodes

    assert ST[0].label == "root"
    assert ST[1].label == "h"
    assert ST[2].label == "w"
    assert ST["shape"].sp_gidx is None
    assert ST["composite"].sp_gidx is None

    shape_or = ST["shape"].derivation
    assert isinstance(shape_or, STSet)
    assert shape_or.mode == "OR"

    shape_branch = shape_or.members[0]
    assert isinstance(shape_branch, STSet)
    assert shape_branch.mode == "AND"
    assert shape_branch.partition == "and"
    assert shape_branch.members == [
        STNodeRef(1),
        STNodeRef(2),
    ]

    assert isinstance(ST[0].derivation, STSet)
    assert ST[0].derivation.mode == "OR"

    root_branch = ST[0].derivation.members[0]
    assert isinstance(root_branch, STSet)
    assert root_branch.mode == "AND"
    assert root_branch.partition == "and"

    assert root_branch.members == [
        STInverseRef("inv_partition_shape"),
        STNodeRef("shape"),
        STNodeRef("composite"),
    ]

    composite_or = ST["composite"].derivation
    assert isinstance(composite_or, STSet)
    assert composite_or.mode == "OR"

    composite_branch = composite_or.members[0]
    assert isinstance(composite_branch, STSet)
    assert composite_branch.mode == "AND"
    assert composite_branch.partition == "and"
    assert composite_branch.members[0] == STInverseRef(
        "inv_partition_composite"
    )

    assert composite_branch.members[1:] == [
        STNodeRef(gidx)
        for gidx in range(3, 9)
    ]

    assert not ST.solved
    assert ST.unresolved_leaf_nodes() == tuple(range(1, 9))


def test_solution_tree_boolean_and_requires_h_w_and_every_composite_leaf():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    # Solve h, w, and all but the last composite leaf.
    for gidx in range(1, 8):
        ST.mark_solution(gidx, 100 + gidx)

    assert ST.is_solved("shape")
    assert not ST.is_solved("composite")
    assert not ST.is_solved(0)

    ST.mark_solution(8, 108)

    assert ST.is_solved("shape")
    assert ST.is_solved("composite")
    assert ST.is_solved(0)
    assert ST.solved


def test_solution_tree_can_record_and_clear_direct_gp_solution():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    ST.mark_solution(sp_gidx=1, gp_gidx=7)

    assert ST[1].gp_gidx == 7
    assert ST.is_solved(1)
    assert ST.needs_gp_solution(1) is False

    ST.clear_solution(1)

    assert ST[1].gp_gidx == -1
    assert not ST.is_solved(1)
    assert ST.needs_gp_solution(1)


def test_solution_tree_sync_preserves_boolean_structure_and_matches():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    original_root_derivation = ST[0].derivation
    ST.mark_solution(1, 5)

    new_gidx = mat2_cwrotate(SP_meta, SP_X, 4)
    assert new_gidx == 9

    ST.sync(SP_meta)

    assert len(ST) == 12
    assert ST[1].gp_gidx == 5
    assert ST[9].source == 4
    assert ST[9].gp_gidx == -1
    assert ST[0].derivation is original_root_derivation


def test_or_partition_adds_inverse_and_transformed_alternative():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    source_gidx = 4
    transformed_gidx = mat2_cwrotate(SP_meta, SP_X, source_gidx)

    ST.register_generation(
        SP_meta,
        source_gidx=source_gidx,
        generated_gidxs=(transformed_gidx,),
        partition="or",
        inverse_op="inv_mat2_cwrotate",
        op_name="mat2_cwrotate",
    )

    assert not ST.is_solved(source_gidx)

    derivation = ST[source_gidx].derivation
    assert isinstance(derivation, STSet)
    assert derivation.mode == "OR"

    branch = derivation.members[-1]
    assert isinstance(branch, STSet)
    assert branch.mode == "AND"
    assert branch.partition == "or"
    assert branch.members == [
        STInverseRef("inv_mat2_cwrotate"),
        STNodeRef(transformed_gidx),
    ]

    ST.mark_solution(transformed_gidx, 55)

    assert ST[source_gidx].gp_gidx == -1
    assert ST.is_solved(source_gidx)


def test_user_example_shape_and_rotated_composite_path_solves_root():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    # Shape requires both h and w.
    ST.mark_solution(1, 10)
    ST.mark_solution(2, 11)
    assert ST.is_solved("shape")

    # Solve all composite leaves except one presence mask.
    for gidx in (3, 5, 6, 7, 8):
        ST.mark_solution(gidx, 100 + gidx)

    unresolved_mask = 4
    rotated = mat2_cwrotate(
        SP_meta,
        SP_X,
        unresolved_mask,
    )
    ST.register_generation(
        SP_meta,
        source_gidx=unresolved_mask,
        generated_gidxs=(rotated,),
        partition="or",
        inverse_op="inv_mat2_cwrotate",
        op_name="mat2_cwrotate",
    )
    ST.mark_solution(rotated, 99)

    assert ST[unresolved_mask].gp_gidx == -1
    assert ST.is_solved(unresolved_mask)
    assert ST.is_solved("composite")
    assert ST.is_solved(0)
    assert ST.solved


def test_get_ST_unsovled_frontier_returns_unresolved_concrete_sp_data():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    frontier = get_ST_unsovled_frontier(ST, SP_X)

    assert frontier.shape == (9,)
    assert frontier.dtype == object

    # Traversal order follows the unresolved concrete proof graph:
    # raw output, h, w, then the per-color ID/mask genes.
    for result_idx, sp_gidx in enumerate(range(9)):
        assert genes_exactly_equal(
            frontier[result_idx],
            SP_X[sp_gidx],
        )


def test_get_ST_unsovled_frontier_filters_scalar_nodes_by_dim():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=0,
        max_dim=0,
    )

    # h, w, and three scalar color-ID genes.
    expected_gidx = (1, 2, 3, 5, 7)

    assert frontier.shape == (5,)

    for result_idx, sp_gidx in enumerate(expected_gidx):
        assert genes_exactly_equal(
            frontier[result_idx],
            SP_X[sp_gidx],
        )


def test_get_ST_unsovled_frontier_filters_matrix_nodes_by_dim():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=2,
        max_dim=2,
    )

    # Raw output plus the three color-presence masks.
    expected_gidx = (0, 4, 6, 8)

    assert frontier.shape == (4,)

    for result_idx, sp_gidx in enumerate(expected_gidx):
        assert genes_exactly_equal(
            frontier[result_idx],
            SP_X[sp_gidx],
        )


def test_get_ST_unsovled_frontier_filters_by_dtype():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    bool_matrices = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=2,
        max_dim=2,
        dtype=bool,
    )

    # Only the three color-presence masks. Raw output is int64.
    expected_gidx = (4, 6, 8)

    assert bool_matrices.shape == (3,)

    for result_idx, sp_gidx in enumerate(expected_gidx):
        assert genes_exactly_equal(
            bool_matrices[result_idx],
            SP_X[sp_gidx],
        )

    int_scalars = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=0,
        max_dim=0,
        dtype=np.int64,
    )

    assert int_scalars.shape == (5,)


def test_get_ST_unsovled_frontier_dtype_none_keeps_all_datatypes():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    scalar_frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=0,
        max_dim=0,
        dtype=None,
    )

    assert scalar_frontier.shape == (5,)


def test_get_GP_pool_returns_complete_pool_without_filters():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    pool = get_GP_pool(GP_meta, GP_X)

    assert pool.shape == (len(GP_X),)
    assert pool.dtype == object

    for gidx in range(len(GP_X)):
        assert genes_exactly_equal(
            pool[gidx],
            GP_X[gidx],
        )


def test_get_GP_pool_filters_dim_and_dtype_symmetrically():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    scalar_pool = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=0,
        max_dim=0,
    )

    # h, w, and three scalar color-ID genes.
    assert scalar_pool.shape == (5,)

    bool_matrix_pool = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=2,
        max_dim=2,
        dtype=np.bool_,
    )

    # Three color-presence masks.
    assert bool_matrix_pool.shape == (3,)

    int_matrix_pool = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=2,
        max_dim=2,
        dtype="int64",
    )

    # Raw input matrix only.
    assert int_matrix_pool.shape == (1,)
    assert genes_exactly_equal(
        int_matrix_pool[0],
        GP_X[0],
    )


def test_identity_pool_helpers_return_exact_live_gene_indices():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    gp_gidxs = get_GP_pool_gidxs(
        GP_meta,
        GP_X,
        min_dim=0,
        max_dim=0,
    )
    gp_pool = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=0,
        max_dim=0,
    )

    assert len(gp_gidxs) == len(gp_pool)

    for position, gidx in enumerate(gp_gidxs):
        assert genes_exactly_equal(
            gp_pool[position],
            GP_X[gidx],
        )

    sp_gidxs = get_ST_unsovled_frontier_gidxs(
        ST,
        SP_X,
        min_dim=2,
        max_dim=2,
    )
    sp_frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=2,
        max_dim=2,
    )

    assert sp_gidxs == get_ST_unsolved_frontier_gidxs(
        ST,
        SP_X,
        min_dim=2,
        max_dim=2,
    )
    assert len(sp_gidxs) == len(sp_frontier)

    for position, gidx in enumerate(sp_gidxs):
        assert genes_exactly_equal(
            sp_frontier[position],
            SP_X[gidx],
        )


def test_identity_pool_helpers_preserve_equal_valued_initial_genes():
    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=2)

    for op_name in ("semantic_a", "semantic_b"):
        gidx = X.append_gene(
            [np.int64(4), np.int64(7)]
        )
        meta.append(
            source=-1,
            op=op_name,
            dims=0,
        )
        assert gidx == len(meta) - 1

    assert get_GP_pool_gidxs(
        meta,
        X,
        min_dim=0,
        max_dim=0,
    ) == (0, 1)


def test_get_GP_pool_updates_as_GP_grows():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    before = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=2,
        max_dim=2,
        dtype=bool,
    )

    new_gidx = bool_complement(
        GP_meta,
        GP_X,
        4,
    )

    after = get_GP_pool(
        GP_meta,
        GP_X,
        min_dim=2,
        max_dim=2,
        dtype=bool,
    )

    assert after.shape == (before.shape[0] + 1,)
    assert genes_exactly_equal(
        after[-1],
        GP_X[new_gidx],
    )


def test_pool_helpers_validate_dtype_and_gp_side():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    with pytest.raises(TypeError, match="NumPy dtype"):
        get_ST_unsovled_frontier(
            ST,
            SP_X,
            dtype=object(),
        )

    with pytest.raises(TypeError, match="NumPy dtype"):
        get_GP_pool(
            GP_meta,
            GP_X,
            dtype=object(),
        )

    with pytest.raises(ValueError, match="GP_meta.side"):
        get_GP_pool(
            SP_meta,
            GP_X,
        )

    with pytest.raises(ValueError, match="GP_X.side"):
        get_GP_pool(
            GP_meta,
            SP_X,
        )


def test_get_ST_unsovled_frontier_prunes_solved_nodes_and_subtrees():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    ST.mark_solution(1, 100)

    scalar_frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=0,
        max_dim=0,
    )

    # h is solved, so only w and the three color IDs remain.
    assert scalar_frontier.shape == (4,)

    ST.mark_solution(0, 999)

    # Solving the root directly means no remaining frontier is required.
    frontier = get_ST_unsovled_frontier(ST, SP_X)

    assert frontier.shape == (0,)
    assert frontier.dtype == object


def test_get_ST_unsovled_frontier_keeps_direct_and_derived_or_candidates():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    source_gidx = 4
    transformed_gidx = mat2_cwrotate(
        SP_meta,
        SP_X,
        source_gidx,
    )
    ST.register_generation(
        SP_meta,
        source_gidx=source_gidx,
        generated_gidxs=(transformed_gidx,),
        partition="or",
        inverse_op="inv_mat2_cwrotate",
        op_name="mat2_cwrotate",
    )

    matrix_frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=2,
        max_dim=2,
    )

    # The original source is still directly solvable, while the rotated gene is
    # an alternate OR path, so both remain in the frontier.
    assert any(
        genes_exactly_equal(gene, SP_X[source_gidx])
        for gene in matrix_frontier
    )
    assert any(
        genes_exactly_equal(gene, SP_X[transformed_gidx])
        for gene in matrix_frontier
    )

    ST.mark_solution(source_gidx, 200)

    matrix_frontier = get_ST_unsovled_frontier(
        ST,
        SP_X,
        min_dim=2,
        max_dim=2,
    )

    # Directly solving the source prunes its alternate transformed subtree.
    assert not any(
        genes_exactly_equal(gene, SP_X[source_gidx])
        for gene in matrix_frontier
    )
    assert not any(
        genes_exactly_equal(gene, SP_X[transformed_gidx])
        for gene in matrix_frontier
    )


def test_get_ST_unsovled_frontier_validates_dim_bounds_and_side():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    with pytest.raises(ValueError, match="min_dim"):
        get_ST_unsovled_frontier(
            ST,
            SP_X,
            min_dim=2,
            max_dim=1,
        )

    with pytest.raises(ValueError, match=">= 0"):
        get_ST_unsovled_frontier(
            ST,
            SP_X,
            min_dim=-1,
        )

    with pytest.raises(TypeError, match="non-negative integer"):
        get_ST_unsovled_frontier(
            ST,
            SP_X,
            max_dim=1.5,
        )

    with pytest.raises(ValueError, match="SP_X.side"):
        get_ST_unsovled_frontier(ST, GP_X)


def test_correctly_spelled_ST_frontier_alias_matches_requested_name():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    requested_name = get_ST_unsovled_frontier(ST, SP_X)
    corrected_alias = get_ST_unsolved_frontier(ST, SP_X)

    assert requested_name.shape == corrected_alias.shape

    for left, right in zip(requested_name, corrected_alias):
        assert genes_exactly_equal(left, right)


def test_program_meta_rows_and_program_x_object_matrix_are_easy_to_inspect():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    rows = GP_meta.rows()
    matrix = GP_X.as_object_array()

    assert rows[0] == {
        "gidx": 0,
        "gene_id": 0,
        "source": -1,
        "op": "raw_input",
        "dims": 2,
        "params": {},
    }
    assert matrix.shape == (9, 2)
    assert matrix[1, 0] == np.int64(2)
    assert matrix[2, 0] == np.int64(3)



def _raw_program(side, values, *, raw_op):
    meta = ProgramMeta(side=side)
    X = ProgramX(side=side, sample_count=len(values))
    X.append_gene(values)
    meta.append(
        source=-1,
        op=raw_op,
        dims=np.asarray(values[0]).ndim,
    )
    return meta, X


def test_valid_generation_partition_shape_requires_exactly_2d():
    meta, X = _raw_program(
        "GP",
        [np.array([[1, 2], [3, 4]], dtype=np.int64)],
        raw_op="raw_input",
    )

    assert valid_generation(meta, X, partition_shape, 0)

    h_gidx, w_gidx = partition_shape(meta, X, 0)

    assert not valid_generation(meta, X, partition_shape, 0)
    assert generation_exists(meta, partition_shape, 0)

    assert meta.dims[h_gidx] == 0
    assert meta.dims[w_gidx] == 0

    vector_meta, vector_X = _raw_program(
        "GP",
        [np.array([1, 2], dtype=np.int64)],
        raw_op="raw_input",
    )
    assert not valid_generation(
        vector_meta,
        vector_X,
        partition_shape,
        0,
    )

    cube_meta, cube_X = _raw_program(
        "GP",
        [np.zeros((2, 2, 2), dtype=np.int64)],
        raw_op="raw_input",
    )
    assert not valid_generation(
        cube_meta,
        cube_X,
        partition_shape,
        0,
    )


def test_valid_generation_partition_composite_requires_dims_gt_one_and_int64():
    int_meta, int_X = _raw_program(
        "GP",
        [np.array([[0, 1], [1, 0]], dtype=np.int64)],
        raw_op="raw_input",
    )

    assert gene_atomic_dtypes(int_X, 0) == (np.dtype("int64"),)
    assert valid_generation(
        int_meta,
        int_X,
        partition_composite,
        0,
    )

    float_meta, float_X = _raw_program(
        "GP",
        [np.array([[0.0, 1.0]], dtype=np.float64)],
        raw_op="raw_input",
    )

    assert not valid_generation(
        float_meta,
        float_X,
        partition_composite,
        0,
    )

    bool_meta, bool_X = _raw_program(
        "GP",
        [
            np.ones(
                (2, 2, 2),
                dtype=bool,
            )
        ],
        raw_op="raw_input",
    )

    assert bool_meta.dims[0] == 3
    assert not valid_generation(
        bool_meta,
        bool_X,
        partition_composite,
        0,
    )

    vector_meta, vector_X = _raw_program(
        "GP",
        [np.array([0, 1], dtype=np.int64)],
        raw_op="raw_input",
    )

    assert not valid_generation(
        vector_meta,
        vector_X,
        partition_composite,
        0,
    )


def test_partition_composite_skips_invalid_empty_source_before_counting_outputs():
    meta, X = _raw_program(
        "GP",
        [np.empty((0, 0), dtype=bool)],
        raw_op="raw_input",
    )

    candidate = _sample_uniform_legal_candidate(
        meta,
        X,
        rng=np.random.default_rng(0),
        max_output_count=1,
        operation_names=("partition_composite",),
    )

    assert candidate is None


def test_partition_bool_trim_empty_output_is_rejected_transactionally(capsys):
    meta, X = _raw_program(
        "GP",
        [np.zeros((2, 2), dtype=bool)],
        raw_op="raw_input",
    )

    created = GP_generate(
        meta,
        X,
        3,
        rng=0,
        max_attempts_per_generation=3,
        operation_names=("partition_bool_trim",),
    )

    output = capsys.readouterr().out

    assert created == []
    assert len(meta) == len(X) == 1
    assert "3 consecutive rejected stochastic attempts" in output


def test_exact_transition_duplicate_includes_operation_source_and_params():
    @operation(
        partition="null",
        output_count=1,
        min_dims_exclusive=0,
    )
    def test_take_n(meta, X, source_idx, n=1):
        values = [
            np.asarray(value)[:n].copy()
            for value in X[source_idx]
        ]
        gidx = X.append_gene(values)
        meta.append(
            source=source_idx,
            op="test_take_n",
            dims=meta.dims[source_idx],
        )
        return gidx

    meta, X = _raw_program(
        "GP",
        [np.array([1, 2, 3], dtype=np.int64)],
        raw_op="raw_input",
    )

    assert valid_generation(
        meta,
        X,
        test_take_n,
        0,
        params={"n": 1},
    )

    first = test_take_n(meta, X, 0, n=1)

    assert meta.params[first] == {"n": 1}
    assert not valid_generation(
        meta,
        X,
        test_take_n,
        0,
        params={"n": 1},
    )
    assert valid_generation(
        meta,
        X,
        test_take_n,
        0,
        params={"n": 2},
    )

    second = test_take_n(meta, X, 0, n=2)

    assert meta.params[second] == {"n": 2}

    OP_REGISTRY.pop("test_take_n", None)



def test_uniform_operation_sampling_is_not_weighted_by_source_count():
    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=1)

    for value in range(5):
        X.append_gene([np.int64(value)])
        meta.append(
            source=-1,
            op=f"raw_{value}",
            dims=0,
        )

    def only_first_source(meta, X, source_idx, params):
        return int(source_idx) == 0

    @operation(
        partition="null",
        output_count=1,
        allowed_dims=(0,),
        validator=only_first_source,
    )
    def test_one_source_op(meta, X, source_idx):
        raise AssertionError("sampling test should not execute operations")

    @operation(
        partition="null",
        output_count=1,
        allowed_dims=(0,),
    )
    def test_five_source_op(meta, X, source_idx):
        raise AssertionError("sampling test should not execute operations")

    rng = np.random.default_rng(12345)
    op_counts = {
        "test_one_source_op": 0,
        "test_five_source_op": 0,
    }
    five_source_counts = {
        source_idx: 0
        for source_idx in range(5)
    }

    for _ in range(4000):
        candidate = _sample_uniform_legal_candidate(
            meta,
            X,
            rng=rng,
            operation_names=(
                "test_one_source_op",
                "test_five_source_op",
            ),
        )

        assert candidate is not None

        info, source_idx, params = candidate

        assert params == {}
        op_counts[info.name] += 1

        if info.name == "test_one_source_op":
            assert source_idx == 0
        else:
            five_source_counts[int(source_idx)] += 1

    one_share = (
        op_counts["test_one_source_op"] / 4000
    )

    # The old combined (operation, source) lottery would put this operation
    # near 1/6 because it has one source versus five. The hierarchical sampler
    # gives each legal operation one equal first-stage slot.
    assert 0.45 <= one_share <= 0.55

    # Operation selection remains uniform. Source selection is intentionally
    # no longer uniform and is tested separately below.
    assert sum(five_source_counts.values()) == op_counts["test_five_source_op"]

    OP_REGISTRY.pop("test_one_source_op", None)
    OP_REGISTRY.pop("test_five_source_op", None)


def test_source_selection_softmax_prefers_earlier_parent_sources():
    meta = ProgramMeta(side="GP")

    meta.append(source=-1, op="raw", dims=0)   # candidate 0, x=-1
    meta.append(source=0, op="child_a", dims=0) # candidate 1, x=0
    meta.append(source=0, op="child_b", dims=0) # candidate 2, x=0
    meta.append(source=2, op="later", dims=0)   # candidate 3, x=2

    valid_sources = [0, 1, 2, 3]
    probabilities = _source_selection_probabilities(
        meta,
        valid_sources,
    )

    assert np.isclose(np.sum(probabilities), 1.0)
    assert probabilities[0] > probabilities[1]
    assert np.isclose(probabilities[1], probabilities[2])
    assert probabilities[2] > probabilities[3]


def test_source_selection_softmax_matches_requested_log_score_formula():
    meta = ProgramMeta(side="GP")
    meta.append(source=-1, op="raw", dims=0)
    meta.append(source=0, op="child", dims=0)
    meta.append(source=3, op="later", dims=0)

    valid_sources = [0, 1, 2]
    probabilities = _source_selection_probabilities(
        meta,
        valid_sources,
    )

    k = len(OP_REGISTRY)
    x = np.asarray([-1.0, 0.0, 3.0])
    scores = -np.log(x + 2.0) / np.log(k + 1.0)
    expected = np.exp(scores - np.max(scores))
    expected /= np.sum(expected)

    assert np.allclose(probabilities, expected)


def test_source_selection_softmax_uses_candidate_gene_parent_not_candidate_gidx():
    meta = ProgramMeta(side="GP")

    # These are far apart in candidate gidx but share the same provenance.
    meta.append(source=-1, op="raw", dims=0)
    meta.append(source=0, op="a", dims=0)
    meta.append(source=0, op="b", dims=0)
    meta.append(source=0, op="c", dims=0)
    meta.append(source=0, op="d", dims=0)

    probabilities = _source_selection_probabilities(
        meta,
        [1, 4],
    )

    assert np.allclose(probabilities, [0.5, 0.5])


def test_source_selection_softmax_handles_multi_source_candidate_symmetrically():
    meta = ProgramMeta(side="GP")
    meta.append(source=-1, op="raw", dims=0)
    meta.append(source=0, op="a", dims=0)
    meta.append(source=2, op="b", dims=0)

    forward = _source_selection_probabilities(
        meta,
        [(1, 2), (0, 2)],
    )
    reverse = _source_selection_probabilities(
        meta,
        [(2, 1), (2, 0)],
    )

    assert np.allclose(forward, reverse)


def _append_test_gp_gene(meta, X, *, source, op, value):
    gidx = X.append_gene([np.int64(value)] * X.sample_count)
    meta.append(
        source=source,
        op=op,
        dims=0,
    )
    return gidx


def test_gp_prune_protects_st_solutions_and_recursive_dependencies():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    g_parent = _append_test_gp_gene(
        GP_meta, GP_X, source=1, op="test_parent", value=10
    )
    g_solver = _append_test_gp_gene(
        GP_meta, GP_X, source=g_parent, op="test_solver", value=11
    )
    g_other = _append_test_gp_gene(
        GP_meta, GP_X, source=1, op="test_other", value=12
    )

    ST.mark_solution(1, g_solver)

    removed = GP_prune(
        GP_meta,
        GP_X,
        ST,
        prune=5,
        rng=0,
    )

    assert g_solver not in removed
    assert g_parent not in removed
    assert g_other in removed

    solver_node = ST[1]
    assert solver_node.gp_gidx >= 0
    assert GP_meta.op[solver_node.gp_gidx] == "test_solver"

    solver_source = int(
        GP_meta.source[solver_node.gp_gidx]
    )
    assert GP_meta.op[solver_source] == "test_parent"


def test_gp_prune_remaps_surviving_sources_and_st_gp_indices():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    removable = _append_test_gp_gene(
        GP_meta, GP_X, source=1, op="test_removable", value=20
    )
    parent = _append_test_gp_gene(
        GP_meta, GP_X, source=1, op="test_parent_keep", value=21
    )
    solver = _append_test_gp_gene(
        GP_meta, GP_X, source=parent, op="test_solver_keep", value=22
    )

    ST.mark_solution(1, solver)

    removed = GP_prune(
        GP_meta,
        GP_X,
        ST,
        prune=1,
        rng=0,
    )

    assert removed == [removable]

    remapped_solver = ST[1].gp_gidx
    assert GP_meta.op[remapped_solver] == "test_solver_keep"

    remapped_parent = int(
        GP_meta.source[remapped_solver]
    )
    assert GP_meta.op[remapped_parent] == "test_parent_keep"


def test_gp_prune_recalculates_leaves_and_can_remove_chain_exactly():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    parent = _append_test_gp_gene(
        GP_meta, GP_X, source=1, op="test_chain_parent", value=30
    )
    child = _append_test_gp_gene(
        GP_meta, GP_X, source=parent, op="test_chain_child", value=31
    )

    removed = GP_prune(
        GP_meta,
        GP_X,
        ST,
        prune=2,
        rng=0,
    )

    assert removed == [child, parent]
    assert "test_chain_child" not in GP_meta.op
    assert "test_chain_parent" not in GP_meta.op


def test_gp_prune_preserves_remaining_multioutput_sibling_slots():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    first = _append_test_gp_gene(
        GP_meta, GP_X, source=1, op="test_bundle", value=35
    )
    second = _append_test_gp_gene(
        GP_meta, GP_X, source=1, op="test_bundle", value=36
    )

    removed = GP_prune(
        GP_meta,
        GP_X,
        ST,
        prune=1,
        rng=0,
    )

    assert removed == [second]
    assert "test_bundle" in GP_meta.op
    remaining = [
        gidx
        for gidx, op_name in enumerate(GP_meta.op)
        if op_name == "test_bundle"
    ]
    assert len(remaining) == 1
    assert GP_X[remaining[0], 0] == np.int64(35)


def test_gp_prune_float_is_proportion_of_preprune_gp_size():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    for index in range(5):
        _append_test_gp_gene(
            GP_meta,
            GP_X,
            source=1,
            op=f"test_float_{index}",
            value=40 + index,
        )

    before = len(GP_X)
    removed = GP_prune(
        GP_meta,
        GP_X,
        ST,
        prune=0.2,
        rng=1,
    )

    assert len(removed) == int(np.floor(0.2 * before))
    assert len(GP_X) == before - len(removed)


def test_gp_prune_never_selects_source_zero_or_negative():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    source_zero = _append_test_gp_gene(
        GP_meta, GP_X, source=0, op="test_source_zero", value=50
    )
    eligible = _append_test_gp_gene(
        GP_meta, GP_X, source=1, op="test_source_positive", value=51
    )

    removed = GP_prune(
        GP_meta,
        GP_X,
        ST,
        prune=5,
        rng=0,
    )

    assert eligible in removed
    assert source_zero not in removed
    assert "test_source_zero" in GP_meta.op


def test_gp_prune_probabilities_follow_positive_log_source_score():
    GP_meta = ProgramMeta(side="GP")
    GP_X = ProgramX(side="GP", sample_count=1)

    _append_test_gp_gene(
        GP_meta, GP_X, source=-1, op="raw", value=0
    )
    _append_test_gp_gene(
        GP_meta, GP_X, source=0, op="source_zero", value=1
    )
    g_early = _append_test_gp_gene(
        GP_meta, GP_X, source=1, op="early", value=2
    )
    g_late = _append_test_gp_gene(
        GP_meta, GP_X, source=2, op="late", value=3
    )

    probabilities = _gp_prune_probabilities(
        GP_meta,
        [g_early, g_late],
    )

    k = len(OP_REGISTRY)
    x = np.asarray([1.0, 2.0])
    scores = np.log(x + 1.0) / np.log(k + 1.0)
    expected = np.exp(scores - np.max(scores))
    expected /= np.sum(expected)

    assert np.allclose(probabilities, expected)
    assert probabilities[1] > probabilities[0]


def test_gp_prune_validates_count_and_proportion():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    with pytest.raises(TypeError):
        GP_prune(GP_meta, GP_X, ST, prune=True)

    with pytest.raises(ValueError, match=">= 0"):
        GP_prune(GP_meta, GP_X, ST, prune=-1)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        GP_prune(GP_meta, GP_X, ST, prune=1.1)


def test_gp_stable_gene_ids_survive_pruning_and_are_not_renumbered():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    first = _append_test_gp_gene(
        GP_meta,
        GP_X,
        source=1,
        op="stable_id_first",
        value=70,
    )
    second = _append_test_gp_gene(
        GP_meta,
        GP_X,
        source=1,
        op="stable_id_second",
        value=71,
    )

    before = {
        GP_meta.op[gidx]: GP_meta.stable_id(gidx)
        for gidx in range(len(GP_meta))
    }
    max_before_id = max(GP_meta.gene_id)

    removed = GP_prune(
        GP_meta,
        GP_X,
        ST,
        prune=1,
        rng=0,
    )

    assert len(removed) == 1

    after = {
        GP_meta.op[gidx]: GP_meta.stable_id(gidx)
        for gidx in range(len(GP_meta))
    }

    for op_name in set(before).intersection(after):
        assert after[op_name] == before[op_name]

    new_gidx = _append_test_gp_gene(
        GP_meta,
        GP_X,
        source=1,
        op="stable_id_after_prune",
        value=72,
    )
    assert GP_meta.stable_id(new_gidx) > max_before_id
    assert len(GP_meta.gene_id) == len(GP_meta) == len(GP_X)


def test_gp_duplicate_retry_keeps_same_prior_until_100_failure_switch(capsys):
    attempts = [0]

    @operation(
        partition="null",
        output_count=1,
        min_dims_exclusive=-1,
    )
    def test_always_duplicate(meta, X, source_idx):
        attempts[0] += 1
        values = [
            np.asarray(value).copy()
            for value in X[source_idx]
        ]
        gidx = X.append_gene(values)
        meta.append(
            source=source_idx,
            op="test_always_duplicate",
            dims=meta.dims[source_idx],
        )
        return gidx

    meta, X = _raw_program(
        "GP",
        [np.array([[1, 2]], dtype=np.int64)],
        raw_op="raw_input",
    )

    created = GP_generate(
        meta,
        X,
        1,
        rng=0,
        operation_names=("test_always_duplicate",),
    )

    output = capsys.readouterr().out

    assert created == []
    assert attempts[0] == 100
    assert len(meta) == len(X) == 1
    assert (
        "100 consecutive rejected stochastic attempts"
        in output
    )

    OP_REGISTRY.pop("test_always_duplicate", None)


def test_gp_generate_never_retains_equivalent_gene_data():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    before = len(GP_X)

    new_gidx = GP_generate(
        GP_meta,
        GP_X,
        12,
        rng=7,
        operation_names=(
            "mat2_cwrotate",
            "dim0_flip",
            "dim1_flip",
            "bool_complement",
        ),
    )

    assert len(new_gidx) > 0
    assert len(GP_X) == before + len(new_gidx)
    assert len(GP_meta) == len(GP_X)

    for i in range(len(GP_X)):
        for j in range(i):
            assert not genes_exactly_equal(GP_X[i], GP_X[j])


def test_gp_generate_target_allows_atomic_multioutput_overshoot():
    @operation(
        partition="null",
        output_count=3,
        min_dims_exclusive=-1,
    )
    def test_three_outputs(meta, X, source_idx):
        generated = []

        for delta in (10, 20, 30):
            values = [
                np.asarray(value) + delta
                for value in X[source_idx]
            ]
            gidx = X.append_gene(values)
            meta.append(
                source=source_idx,
                op="test_three_outputs",
                dims=meta.dims[source_idx],
            )
            generated.append(gidx)

        return tuple(generated)

    meta, X = _raw_program(
        "GP",
        [
            np.int64(1),
            np.int64(2),
        ],
        raw_op="raw_input",
    )

    created = GP_generate(
        meta,
        X,
        1,
        rng=0,
        operation_names=("test_three_outputs",),
    )

    assert len(created) == 3
    assert len(meta) == len(X) == 4
    assert [
        int(np.asarray(X[gidx, 0]).item())
        for gidx in created
    ] == [11, 21, 31]

    OP_REGISTRY.pop("test_three_outputs", None)


def test_gp_generate_zero_is_noop():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    assert GP_generate(GP_meta, GP_X, 0, rng=0) == []
    assert len(GP_X) == 9


def test_sp_generate_runs_one_reversible_step_and_updates_st():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    before = len(SP_X)

    new_gidx = SP_generate(
        SP_meta,
        SP_X,
        ST,
        rng=11,
        operation_names=("mat2_cwrotate",),
    )

    assert len(new_gidx) == 1
    assert len(SP_X) == before + 1
    assert len(SP_meta) == len(SP_X)

    # ST also contains logical shape and composite nodes.
    assert len(ST) == len(SP_X) + 2

    gidx = new_gidx[0]
    assert OP_REGISTRY[SP_meta.op[gidx]].partition == "or"
    assert ST[gidx].gp_gidx == -1

    source = SP_meta.source[gidx]
    assert ST[source].derivation is not None
    assert ST[source].derivation.mode == "OR"


def test_sp_generate_returns_empty_when_no_reversible_partition_is_valid():
    meta = ProgramMeta(side="SP")
    X = ProgramX(side="SP", sample_count=1)

    X.append_gene([5])
    meta.append(
        source=-1,
        op="raw_output",
        dims=0,
    )

    ST = SolutionTree.from_sp_meta(meta)

    assert SP_generate(meta, X, ST, rng=3) == []
    assert len(meta) == len(X) == len(ST) == 1


def test_parameter_sampler_supports_future_integer_parameter_ops():
    def sample_k(rng):
        return {"k": int(rng.integers(1, 1000))}

    @operation(
        partition="null",
        output_count=1,
        min_dims_exclusive=0,
        parameter_sampler=sample_k,
    )
    def test_param_op(meta, X, source_idx, k):
        values = [
            np.asarray(value) + k
            for value in X[source_idx]
        ]
        gidx = X.append_gene(values)
        meta.append(
            source=source_idx,
            op="test_param_op",
            dims=meta.dims[source_idx],
        )
        return gidx

    meta, X = _raw_program(
        "GP",
        [np.array([1, 2], dtype=np.int64)],
        raw_op="raw_input",
    )

    created = GP_generate(
        meta,
        X,
        3,
        rng=123,
        operation_names=("test_param_op",),
    )

    assert len(created) == 3
    assert all(
        isinstance(meta.params[gidx]["k"], int)
        for gidx in created
    )
    assert len({
        meta.params[gidx]["k"]
        for gidx in created
    }) == 3

    OP_REGISTRY.pop("test_param_op", None)



def test_equivalent_gene_idx_requires_exact_shape_dtype_and_contents():
    meta, X = _raw_program(
        "GP",
        [
            np.array([[1, 2]], dtype=np.int64),
            np.array([[3, 4]], dtype=np.int64),
        ],
        raw_op="raw_input",
    )

    exact = np.empty(2, dtype=object)
    exact[0] = np.array([[1, 2]], dtype=np.int64)
    exact[1] = np.array([[3, 4]], dtype=np.int64)

    different_shape = np.empty(2, dtype=object)
    different_shape[0] = np.array([1, 2], dtype=np.int64)
    different_shape[1] = np.array([3, 4], dtype=np.int64)

    different_dtype = np.empty(2, dtype=object)
    different_dtype[0] = np.array([[1, 2]], dtype=np.float64)
    different_dtype[1] = np.array([[3, 4]], dtype=np.float64)

    different_contents = np.empty(2, dtype=object)
    different_contents[0] = np.array([[1, 9]], dtype=np.int64)
    different_contents[1] = np.array([[3, 4]], dtype=np.int64)

    assert equivalent_gene_idx(X, exact) == 0
    assert equivalent_gene_idx(X, different_shape) == -1
    assert equivalent_gene_idx(X, different_dtype) == -1
    assert equivalent_gene_idx(X, different_contents) == -1


def test_duplicate_output_operation_is_rolled_back_and_space_exhausts(capsys):
    @operation(
        partition="null",
        output_count=1,
        min_dims_exclusive=-1,
    )
    def test_identity_copy(meta, X, source_idx):
        values = [
            np.asarray(value).copy()
            for value in X[source_idx]
        ]
        gidx = X.append_gene(values)
        meta.append(
            source=source_idx,
            op="test_identity_copy",
            dims=meta.dims[source_idx],
        )
        return gidx

    meta, X = _raw_program(
        "GP",
        [np.array([[1, 2]], dtype=np.int64)],
        raw_op="raw_input",
    )

    created = GP_generate(
        meta,
        X,
        1,
        rng=0,
        operation_names=("test_identity_copy",),
    )

    output = capsys.readouterr().out

    assert created == []
    assert len(meta) == 1
    assert len(X) == 1
    assert (
        "100 consecutive rejected stochastic attempts"
        in output
    )

    OP_REGISTRY.pop("test_identity_copy", None)


def test_multi_output_operation_rolls_back_if_any_output_duplicates(capsys):
    @operation(
        partition="null",
        output_count=2,
        min_dims_exclusive=-1,
    )
    def test_mixed_outputs(meta, X, source_idx):
        duplicate_values = [
            np.asarray(value).copy()
            for value in X[source_idx]
        ]
        novel_values = [
            np.asarray(value) + 100
            for value in X[source_idx]
        ]

        first = X.append_gene(duplicate_values)
        meta.append(
            source=source_idx,
            op="test_mixed_outputs",
            dims=meta.dims[source_idx],
        )

        second = X.append_gene(novel_values)
        meta.append(
            source=source_idx,
            op="test_mixed_outputs",
            dims=meta.dims[source_idx],
        )

        return first, second

    meta, X = _raw_program(
        "GP",
        [np.array([1, 2], dtype=np.int64)],
        raw_op="raw_input",
    )

    created = GP_generate(
        meta,
        X,
        2,
        rng=0,
        operation_names=("test_mixed_outputs",),
    )

    capsys.readouterr()

    assert created == []
    assert len(meta) == 1
    assert len(X) == 1

    OP_REGISTRY.pop("test_mixed_outputs", None)


def test_sp_generation_rolls_back_duplicate_data_and_reports_exhaustion(capsys):
    @inverse_operation("test_sp_identity")
    def inv_test_sp_identity(value):
        return np.asarray(value).copy()

    @operation(
        partition="or",
        inverse_op="inv_test_sp_identity",
        output_count=1,
        min_dims_exclusive=-1,
    )
    def test_sp_identity(meta, X, source_idx):
        values = [
            np.asarray(value).copy()
            for value in X[source_idx]
        ]
        gidx = X.append_gene(values)
        meta.append(
            source=source_idx,
            op="test_sp_identity",
            dims=meta.dims[source_idx],
        )
        return gidx

    meta, X = _raw_program(
        "SP",
        [np.array([[1]], dtype=np.int64)],
        raw_op="raw_output",
    )
    ST = SolutionTree.from_sp_meta(meta)

    created = SP_generate(
        meta,
        X,
        ST,
        rng=0,
        operation_names=("test_sp_identity",),
    )

    output = capsys.readouterr().out

    assert created == []
    assert len(meta) == len(X) == len(ST) == 1
    assert (
        "100 consecutive rejected stochastic attempts"
        in output
    )

    OP_REGISTRY.pop("test_sp_identity", None)
    INV_OP_REGISTRY.pop("inv_test_sp_identity", None)



def test_bool_complement_accepts_boolean_any_dim_and_complements_all_values():
    meta, X = _raw_program(
        "GP",
        [
            np.array([True, False, True], dtype=bool),
            np.array([False, False], dtype=bool),
        ],
        raw_op="raw_input",
    )

    assert valid_generation(meta, X, bool_complement, 0)

    gidx = bool_complement(meta, X, 0)

    assert meta.source[gidx] == 0
    assert meta.dims[gidx] == 1
    assert np.array_equal(
        X[gidx, 0],
        np.array([False, True, False], dtype=bool),
    )
    assert np.array_equal(
        X[gidx, 1],
        np.array([True, True], dtype=bool),
    )


def test_bool_complement_rejects_non_boolean_gene():
    meta, X = _raw_program(
        "GP",
        [np.array([0, 1], dtype=np.int64)],
        raw_op="raw_input",
    )

    assert not valid_generation(meta, X, bool_complement, 0)


def _two_bool_gene_program(first_values, second_values):
    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=len(first_values))

    first = X.append_gene(first_values)
    meta.append(source=-1, op="raw_a", dims=np.asarray(first_values[0]).ndim)

    second = X.append_gene(second_values)
    meta.append(source=-1, op="raw_b", dims=np.asarray(second_values[0]).ndim)

    return meta, X, first, second


def test_bool2_union_and_intersection_use_two_same_shaped_boolean_sources():
    meta, X, first, second = _two_bool_gene_program(
        [
            np.array([[True, False], [False, True]], dtype=bool),
            np.array([[True, False]], dtype=bool),
        ],
        [
            np.array([[False, True], [False, True]], dtype=bool),
            np.array([[False, True]], dtype=bool),
        ],
    )

    assert valid_generation(meta, X, bool2_union, (first, second))
    assert valid_generation(meta, X, bool2_intersect, (first, second))

    union_gidx = bool2_union(meta, X, (first, second))
    intersect_gidx = bool2_intersect(meta, X, (first, second))

    assert meta.source[union_gidx] == (first, second)
    assert meta.source[intersect_gidx] == (first, second)

    assert np.array_equal(
        X[union_gidx, 0],
        np.array([[True, True], [False, True]], dtype=bool),
    )
    assert np.array_equal(
        X[intersect_gidx, 0],
        np.array([[False, False], [False, True]], dtype=bool),
    )

    assert np.array_equal(
        X[union_gidx, 1],
        np.array([[True, True]], dtype=bool),
    )
    assert np.array_equal(
        X[intersect_gidx, 1],
        np.array([[False, False]], dtype=bool),
    )


def test_bool2_ops_reject_mismatched_shape_or_nonboolean_source():
    meta, X, first, second = _two_bool_gene_program(
        [np.array([True, False], dtype=bool)],
        [np.array([[True, False]], dtype=bool)],
    )

    assert not valid_generation(meta, X, bool2_union, (first, second))
    assert not valid_generation(meta, X, bool2_intersect, (first, second))

    third = X.append_gene([np.array([1, 0], dtype=np.int64)])
    meta.append(source=-1, op="raw_int", dims=1)

    assert not valid_generation(meta, X, bool2_union, (first, third))


def test_bool2_commutative_source_order_counts_as_same_transition():
    meta, X, first, second = _two_bool_gene_program(
        [np.array([True, False], dtype=bool)],
        [np.array([False, True], dtype=bool)],
    )

    bool2_union(meta, X, (first, second))

    assert generation_exists(meta, bool2_union, (second, first))
    assert not valid_generation(meta, X, bool2_union, (second, first))


def test_mat2_cwrotate_rotates_any_dtype_clockwise():
    meta, X = _raw_program(
        "GP",
        [
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
                dtype=np.int64,
            )
        ],
        raw_op="raw_input",
    )

    assert valid_generation(meta, X, mat2_cwrotate, 0)

    gidx = mat2_cwrotate(meta, X, 0)

    assert np.array_equal(
        X[gidx, 0],
        np.array(
            [
                [4, 1],
                [5, 2],
                [6, 3],
            ],
            dtype=np.int64,
        ),
    )


def test_mat2_cwrotate_rejects_non_2d_gene():
    meta, X = _raw_program(
        "GP",
        [np.ones((2, 2, 2), dtype=bool)],
        raw_op="raw_input",
    )

    assert not valid_generation(meta, X, mat2_cwrotate, 0)


def test_dim_flips_apply_to_requested_axis_and_enforce_minimum_dims():
    array = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    meta, X = _raw_program(
        "GP",
        [array],
        raw_op="raw_input",
    )

    g0 = dim0_flip(meta, X, 0)
    g1 = dim1_flip(meta, X, 0)
    g2 = dim2_flip(meta, X, 0)

    assert np.array_equal(X[g0, 0], np.flip(array, axis=0))
    assert np.array_equal(X[g1, 0], np.flip(array, axis=1))
    assert np.array_equal(X[g2, 0], np.flip(array, axis=2))

    vector_meta, vector_X = _raw_program(
        "GP",
        [np.array([1, 2], dtype=np.int64)],
        raw_op="raw_input",
    )

    assert valid_generation(vector_meta, vector_X, dim0_flip, 0)
    assert not valid_generation(vector_meta, vector_X, dim1_flip, 0)
    assert not valid_generation(vector_meta, vector_X, dim2_flip, 0)


def test_partition_bool_trim_emits_y_x_and_mask_genes():
    meta, X = _raw_program(
        "GP",
        [
            np.array(
                [
                    [False, False, False, False],
                    [False, False, True, True],
                    [False, False, True, False],
                    [False, False, False, False],
                ],
                dtype=bool,
            )
        ],
        raw_op="raw_input",
    )

    assert valid_generation(meta, X, partition_bool_trim, 0)
    assert operation_output_count(
        partition_bool_trim,
        meta,
        X,
        0,
    ) == 3

    y_gidx, x_gidx, data_gidx = partition_bool_trim(meta, X, 0)

    assert X[y_gidx, 0] == np.int64(1)
    assert X[x_gidx, 0] == np.int64(2)
    assert np.array_equal(
        X[data_gidx, 0],
        np.array(
            [
                [True, True],
                [True, False],
            ],
            dtype=bool,
        ),
    )
    assert meta.dims[y_gidx] == 0
    assert meta.dims[x_gidx] == 0
    assert meta.dims[data_gidx] == 2


def test_partition_bool_trim_is_now_exactly_2d_boolean():
    vector_meta, vector_X = _raw_program(
        "GP",
        [np.array([False, True, True], dtype=bool)],
        raw_op="raw_input",
    )
    assert not valid_generation(
        vector_meta,
        vector_X,
        partition_bool_trim,
        0,
    )

    cube_meta, cube_X = _raw_program(
        "GP",
        [np.ones((2, 2, 2), dtype=bool)],
        raw_op="raw_input",
    )
    assert not valid_generation(
        cube_meta,
        cube_X,
        partition_bool_trim,
        0,
    )

    int_meta, int_X = _raw_program(
        "GP",
        [np.array([[0, 1]], dtype=np.int64)],
        raw_op="raw_input",
    )
    assert not valid_generation(
        int_meta,
        int_X,
        partition_bool_trim,
        0,
    )

    tight_meta, tight_X = _raw_program(
        "GP",
        [np.array([[True, True], [True, True]], dtype=bool)],
        raw_op="raw_input",
    )
    assert not valid_generation(
        tight_meta,
        tight_X,
        partition_bool_trim,
        0,
    )


def test_partition_bool_trim_all_false_direct_call_has_empty_mask():
    source = np.zeros((2, 3), dtype=bool)
    meta, X = _raw_program(
        "GP",
        [source],
        raw_op="raw_input",
    )

    y_gidx, x_gidx, data_gidx = partition_bool_trim(meta, X, 0)

    assert X[y_gidx, 0] == np.int64(0)
    assert X[x_gidx, 0] == np.int64(0)
    assert X[data_gidx, 0].shape == (0, 0)
    assert X[data_gidx, 0].dtype == bool


def test_partition_bool_subjects_uses_8_connected_components():
    source = np.array(
        [
            [True, False, False, False, False, False],
            [False, True, False, True, True, False],
            [False, False, True, True, False, False],
            [False, False, False, False, False, True],
        ],
        dtype=bool,
    )
    meta, X = _raw_program(
        "GP",
        [source],
        raw_op="raw_input",
    )

    assert valid_generation(meta, X, partition_bool_subjects, 0)
    assert operation_output_count(
        partition_bool_subjects,
        meta,
        X,
        0,
    ) == 6

    generated = partition_bool_subjects(meta, X, 0)
    assert len(generated) == 6

    y0, x0, mask0, y1, x1, mask1 = generated

    # (0,0)->(1,1)->(2,2) is one diagonal-connected body, and it also
    # reaches the cluster at (1,3)/(2,3) through the 3x3 neighborhood.
    assert X[y0, 0] == np.int64(0)
    assert X[x0, 0] == np.int64(0)
    assert np.array_equal(
        X[mask0, 0],
        np.array(
            [
                [True, False, False, False],
                [False, True, False, True],
                [False, False, True, True],
            ],
            dtype=bool,
        ),
    )

    assert X[y1, 0] == np.int64(3)
    assert X[x1, 0] == np.int64(5)
    assert np.array_equal(
        X[mask1, 0],
        np.array([[True]], dtype=bool),
    )


def test_partition_bool_subjects_o_with_center_dot_is_two_subjects():
    source = np.array(
        [
            [True, True, True, True, True],
            [True, False, False, False, True],
            [True, False, True, False, True],
            [True, False, False, False, True],
            [True, True, True, True, True],
        ],
        dtype=bool,
    )
    meta, X = _raw_program(
        "GP",
        [source],
        raw_op="raw_input",
    )

    generated = partition_bool_subjects(meta, X, 0)
    assert len(generated) == 6

    outer_y, outer_x, outer_mask, dot_y, dot_x, dot_mask = generated

    assert X[outer_y, 0] == np.int64(0)
    assert X[outer_x, 0] == np.int64(0)

    expected_outer = source.copy()
    expected_outer[2, 2] = False
    assert np.array_equal(X[outer_mask, 0], expected_outer)

    assert X[dot_y, 0] == np.int64(2)
    assert X[dot_x, 0] == np.int64(2)
    assert np.array_equal(
        X[dot_mask, 0],
        np.array([[True]], dtype=bool),
    )


def test_partition_bool_subjects_aligns_subject_slots_across_samples():
    sample_a = np.array(
        [
            [True, True, False, False],
            [False, False, False, True],
        ],
        dtype=bool,
    )
    sample_b = np.array(
        [
            [False, True, False, False],
            [False, True, False, False],
            [False, False, False, True],
        ],
        dtype=bool,
    )
    meta, X = _raw_program(
        "GP",
        [sample_a, sample_b],
        raw_op="raw_input",
    )

    generated = partition_bool_subjects(meta, X, 0)
    assert len(generated) == 6

    y0, x0, mask0, y1, x1, mask1 = generated

    assert [int(value) for value in X[y0]] == [0, 0]
    assert [int(value) for value in X[x0]] == [0, 1]
    assert [int(value) for value in X[y1]] == [1, 2]
    assert [int(value) for value in X[x1]] == [3, 3]
    assert all(np.asarray(value).dtype == bool for value in X[mask0])
    assert all(np.asarray(value).dtype == bool for value in X[mask1])


def test_partition_bool_subjects_rejects_mismatched_or_zero_subject_counts():
    mismatched_meta, mismatched_X = _raw_program(
        "GP",
        [
            np.array(
                [
                    [True, False, False],
                    [False, False, True],
                ],
                dtype=bool,
            ),
            np.array(
                [
                    [True, False, False],
                    [False, True, False],
                ],
                dtype=bool,
            ),
        ],
        raw_op="raw_input",
    )
    assert not valid_generation(
        mismatched_meta,
        mismatched_X,
        partition_bool_subjects,
        0,
    )

    empty_meta, empty_X = _raw_program(
        "GP",
        [np.zeros((3, 3), dtype=bool)],
        raw_op="raw_input",
    )
    assert not valid_generation(
        empty_meta,
        empty_X,
        partition_bool_subjects,
        0,
    )


def test_partition_bool_subjects_requires_2d_bool_and_is_gp_only():
    int_meta, int_X = _raw_program(
        "GP",
        [np.array([[0, 1]], dtype=np.int64)],
        raw_op="raw_input",
    )
    assert not valid_generation(
        int_meta,
        int_X,
        partition_bool_subjects,
        0,
    )

    vector_meta, vector_X = _raw_program(
        "GP",
        [np.array([True, False], dtype=bool)],
        raw_op="raw_input",
    )
    assert not valid_generation(
        vector_meta,
        vector_X,
        partition_bool_subjects,
        0,
    )

    sp_meta, sp_X = _raw_program(
        "SP",
        [np.array([[True]], dtype=bool)],
        raw_op="raw_output",
    )
    assert not valid_generation(
        sp_meta,
        sp_X,
        partition_bool_subjects,
        0,
    )


def test_operation_partition_categories():
    assert OP_REGISTRY["bool_complement"].partition == "or"
    assert OP_REGISTRY["mat2_cwrotate"].partition == "or"
    assert OP_REGISTRY["dim0_flip"].partition == "or"
    assert OP_REGISTRY["dim1_flip"].partition == "or"
    assert OP_REGISTRY["dim2_flip"].partition == "or"

    assert OP_REGISTRY["partition_bool_trim"].partition == "and"
    assert OP_REGISTRY["indiv_1dim"].partition == "and"

    assert OP_REGISTRY["bool2_union"].partition == "null"
    assert OP_REGISTRY["bool2_intersect"].partition == "null"
    assert OP_REGISTRY["bool_sum"].partition == "null"
    assert OP_REGISTRY["bool_mat_ident"].partition == "null"
    assert OP_REGISTRY["partition_bool_subjects"].partition == "null"
    assert OP_REGISTRY["bool_cavity"].partition == "null"




def test_gp_generate_can_discover_binary_boolean_operation():
    meta, X, first, second = _two_bool_gene_program(
        [
            np.array([True, False, False], dtype=bool),
            np.array([False, True, False], dtype=bool),
        ],
        [
            np.array([False, True, False], dtype=bool),
            np.array([True, False, False], dtype=bool),
        ],
    )

    created = GP_generate(
        meta,
        X,
        1,
        rng=0,
        operation_names=("bool2_union",),
    )

    assert len(created) == 1
    assert meta.op[created[0]] == "bool2_union"
    assert meta.source[created[0]] == (first, second)



def test_bool_mat_ident_groups_matching_2d_boolean_patterns():
    pattern_a = np.array(
        [
            [True, False],
            [False, True],
        ],
        dtype=bool,
    )
    pattern_b = np.array(
        [
            [True, True],
            [False, False],
        ],
        dtype=bool,
    )
    pattern_c = np.array(
        [
            [False, True],
            [True, False],
        ],
        dtype=bool,
    )

    meta, X = _raw_program(
        "GP",
        [
            pattern_a,
            pattern_a.copy(),
            pattern_b,
            pattern_c,
            pattern_b.copy(),
        ],
        raw_op="raw_input",
    )

    assert valid_generation(meta, X, bool_mat_ident, 0)

    gidx = bool_mat_ident(meta, X, 0)

    assert meta.source[gidx] == 0
    assert meta.op[gidx] == "bool_mat_ident"
    assert meta.dims[gidx] == 0
    assert [int(value) for value in X[gidx]] == [0, 0, 1, 2, 1]
    assert all(isinstance(value, np.int64) for value in X[gidx])


def test_bool_mat_ident_rejects_uniform_and_all_unique_identity_sets():
    pattern_a = np.array([[True, False]], dtype=bool)
    pattern_b = np.array([[False, True]], dtype=bool)
    pattern_c = np.array([[True, True]], dtype=bool)

    uniform_meta, uniform_X = _raw_program(
        "GP",
        [
            pattern_a,
            pattern_a.copy(),
            pattern_a.copy(),
        ],
        raw_op="raw_input",
    )
    assert not valid_generation(
        uniform_meta,
        uniform_X,
        bool_mat_ident,
        0,
    )

    unique_meta, unique_X = _raw_program(
        "GP",
        [pattern_a, pattern_b, pattern_c],
        raw_op="raw_input",
    )
    assert not valid_generation(
        unique_meta,
        unique_X,
        bool_mat_ident,
        0,
    )


def test_bool_mat_ident_requires_2d_boolean_source_and_is_gp_only():
    int_meta, int_X = _raw_program(
        "GP",
        [
            np.array([[0, 1]], dtype=np.int64),
            np.array([[0, 1]], dtype=np.int64),
            np.array([[1, 0]], dtype=np.int64),
        ],
        raw_op="raw_input",
    )
    assert not valid_generation(int_meta, int_X, bool_mat_ident, 0)

    one_dim_meta, one_dim_X = _raw_program(
        "GP",
        [
            np.array([True, False], dtype=bool),
            np.array([True, False], dtype=bool),
            np.array([False, True], dtype=bool),
        ],
        raw_op="raw_input",
    )
    assert not valid_generation(
        one_dim_meta,
        one_dim_X,
        bool_mat_ident,
        0,
    )

    sp_meta, sp_X = _raw_program(
        "SP",
        [
            np.array([[True, False]], dtype=bool),
            np.array([[True, False]], dtype=bool),
            np.array([[False, True]], dtype=bool),
        ],
        raw_op="raw_output",
    )
    assert not valid_generation(sp_meta, sp_X, bool_mat_ident, 0)


def test_bool_sum_counts_true_values_and_returns_scalar_int64():
    meta, X = _raw_program(
        "GP",
        [
            np.array([[True, False, True]], dtype=bool),
            np.array([[True, True, False, True]], dtype=bool),
        ],
        raw_op="raw_input",
    )

    assert valid_generation(meta, X, bool_sum, 0)

    gidx = bool_sum(meta, X, 0)

    assert meta.source[gidx] == 0
    assert meta.dims[gidx] == 0
    assert isinstance(X[gidx, 0], np.int64)
    assert isinstance(X[gidx, 1], np.int64)
    assert X[gidx, 0] == 2
    assert X[gidx, 1] == 3


def test_bool_sum_accepts_scalar_bool_and_rejects_non_bool():
    scalar_meta = ProgramMeta(side="GP")
    scalar_X = ProgramX(side="GP", sample_count=1)
    scalar_X.append_gene([np.bool_(True)])
    scalar_meta.append(source=-1, op="raw_bool", dims=0)

    assert valid_generation(scalar_meta, scalar_X, bool_sum, 0)

    gidx = bool_sum(scalar_meta, scalar_X, 0)

    assert scalar_meta.dims[gidx] == 0
    assert isinstance(scalar_X[gidx, 0], np.int64)
    assert scalar_X[gidx, 0] == 1

    int_meta, int_X = _raw_program(
        "GP",
        [np.array([0, 1], dtype=np.int64)],
        raw_op="raw_input",
    )

    assert not valid_generation(int_meta, int_X, bool_sum, 0)


def test_bool_cavity_matches_requested_1d_examples():
    meta, X = _raw_program(
        "GP",
        [
            np.array([False, True, False, True, False], dtype=bool),
            np.array([False, True, False, False, True], dtype=bool),
        ],
        raw_op="raw_input",
    )

    assert valid_generation(meta, X, bool_cavity, 0)

    gidx = bool_cavity(meta, X, 0)

    assert np.array_equal(
        X[gidx, 0],
        np.array([False, False, True, False, False], dtype=bool),
    )
    assert np.array_equal(
        X[gidx, 1],
        np.array([False, False, True, True, False], dtype=bool),
    )


def test_bool_cavity_matches_requested_2d_example():
    source = np.array(
        [
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ],
        dtype=bool,
    )

    meta, X = _raw_program(
        "GP",
        [source],
        raw_op="raw_input",
    )

    gidx = bool_cavity(meta, X, 0)

    assert np.array_equal(
        X[gidx, 0],
        np.array(
            [
                [False, False, False],
                [False, True, False],
                [False, False, False],
            ],
            dtype=bool,
        ),
    )


def test_bool_cavity_uses_axis_adjacent_connectivity_not_diagonals():
    source = np.array(
        [
            [False, True, False],
            [True, False, True],
            [False, True, False],
        ],
        dtype=bool,
    )

    meta, X = _raw_program(
        "GP",
        [source],
        raw_op="raw_input",
    )

    gidx = bool_cavity(meta, X, 0)

    # The center is diagonally adjacent to boundary False cells, but diagonal
    # contact does not open the cavity.
    assert X[gidx, 0][1, 1]


def test_bool_cavity_marks_multiple_enclosed_regions():
    source = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, False, True, True, True, False, True],
            [True, False, True, False, True, False, True],
            [True, True, True, True, True, True, True],
        ],
        dtype=bool,
    )

    meta, X = _raw_program(
        "GP",
        [source],
        raw_op="raw_input",
    )

    gidx = bool_cavity(meta, X, 0)

    expected = np.logical_not(source)

    assert np.array_equal(X[gidx, 0], expected)


def test_bool_cavity_does_not_mark_false_region_connected_to_boundary():
    source = np.array(
        [
            [False, True, True, True],
            [False, False, True, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        dtype=bool,
    )

    meta, X = _raw_program(
        "GP",
        [source],
        raw_op="raw_input",
    )

    gidx = bool_cavity(meta, X, 0)

    assert not np.any(X[gidx, 0])


def test_bool_cavity_requires_boolean_source_with_true_in_every_sample():
    no_true_meta, no_true_X = _raw_program(
        "GP",
        [np.zeros((2, 2), dtype=bool)],
        raw_op="raw_input",
    )

    assert not valid_generation(
        no_true_meta,
        no_true_X,
        bool_cavity,
        0,
    )

    int_meta, int_X = _raw_program(
        "GP",
        [np.array([[0, 1]], dtype=np.int64)],
        raw_op="raw_input",
    )

    assert not valid_generation(
        int_meta,
        int_X,
        bool_cavity,
        0,
    )


def test_bool_cavity_accepts_scalar_true_and_returns_scalar_false():
    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=1)
    X.append_gene([np.bool_(True)])
    meta.append(source=-1, op="raw_bool", dims=0)

    assert valid_generation(meta, X, bool_cavity, 0)

    gidx = bool_cavity(meta, X, 0)

    assert meta.dims[gidx] == 0
    assert bool(X[gidx, 0]) is False


def test_bool_sum_and_cavity_are_null_partition_gp_only():
    assert OP_REGISTRY["bool_sum"].partition == "null"
    assert OP_REGISTRY["bool_cavity"].partition == "null"
