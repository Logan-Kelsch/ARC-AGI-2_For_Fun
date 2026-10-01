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
    init_env,
)
from notebooks.ops.inv_ops import (
    INV_OP_REGISTRY,
    inverse_operation,
)
from notebooks.ops.ops import (
    GP_generate,
    OP_REGISTRY,
    SP_generate,
    bool2_intersect,
    bool2_union,
    bool_cavity,
    bool_complement,
    bool_sum,
    dim0_flip,
    dim1_flip,
    dim2_flip,
    equivalent_gene_idx,
    generation_exists,
    gene_atomic_dtypes,
    genes_exactly_equal,
    mat2_cwrotate,
    operation,
    operation_output_count,
    partition_bool_trim,
    partition_composite,
    partition_shape,
    valid_generation,
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

    assert GP_X.shape == (8, 2)
    assert SP_X.shape == (8, 2)

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
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
    ]
    assert GP_meta.source == [-1, 0, 0, 0, 0, 0, 0, 0]
    assert GP_meta.dims == [2, 1, 0, 2, 0, 2, 0, 2]
    assert GP_meta.params == [{}, {}, {}, {}, {}, {}, {}, {}]

    assert SP_meta.op == [
        "raw_output",
        "partition_shape",
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
        "partition_composite",
    ]
    assert SP_meta.source == [-1, 0, 0, 0, 0, 0, 0, 0]
    assert SP_meta.dims == [2, 1, 0, 2, 0, 2, 0, 2]
    assert SP_meta.params == [{}, {}, {}, {}, {}, {}, {}, {}]

    assert len(GP_meta) == len(GP_X) == 8
    assert len(SP_meta) == len(SP_X) == 8


def test_partition_shape_generates_shape_gene():
    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=2)

    X.append_gene(
        [
            np.zeros((2, 3), dtype=int),
            np.zeros((5, 4), dtype=int),
        ]
    )
    meta.append(source=-1, op="raw_input", dims=2)

    gidx = partition_shape(meta, X, 0)

    assert gidx == 1
    assert np.array_equal(X[1, 0], np.array([2, 3]))
    assert np.array_equal(X[1, 1], np.array([5, 4]))

    assert meta.source[1] == 0
    assert meta.op[1] == "partition_shape"
    assert meta.dims[1] == 1


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


def test_solution_tree_initializes_as_root_shape_and_composite_boolean_proof():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    assert ST.roots == (0,)

    # Eight concrete SP genes plus one logical composite node.
    assert len(ST) == 9
    assert "composite" in ST.nodes

    assert ST[0].label == "root"
    assert ST[1].label == "shape"
    assert ST["composite"].sp_gidx is None

    assert isinstance(ST[0].derivation, STSet)
    assert ST[0].derivation.mode == "OR"

    root_branch = ST[0].derivation.members[0]
    assert isinstance(root_branch, STSet)
    assert root_branch.mode == "AND"
    assert root_branch.partition == "and"

    assert isinstance(root_branch.members[0], STInverseRef)
    assert root_branch.members[0].inverse_op == "inv_partition_shape"
    assert root_branch.members[1] == STNodeRef(1)
    assert root_branch.members[2] == STNodeRef("composite")

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
        for gidx in range(2, 8)
    ]

    assert not ST.solved
    assert ST.unresolved_leaf_nodes() == tuple(range(1, 8))


def test_solution_tree_boolean_and_requires_every_initial_leaf():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    for gidx in range(1, 7):
        ST.mark_solution(gidx, 100 + gidx)

    assert not ST.is_solved("composite")
    assert not ST.is_solved(0)

    ST.mark_solution(7, 107)

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

    new_gidx = partition_shape(SP_meta, SP_X, 1)
    assert new_gidx == 8

    ST.sync(SP_meta)

    assert len(ST) == 10
    assert ST[1].gp_gidx == 5
    assert ST[8].source == 1
    assert ST[8].gp_gidx == -1
    assert ST[0].derivation is original_root_derivation


def test_or_partition_adds_inverse_and_transformed_alternative():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    source_gidx = 3
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

    # The source itself remains unsolved, but solving the transformed
    # representation is enough because the inverse operation is innate.
    ST.mark_solution(transformed_gidx, 55)

    assert ST[source_gidx].gp_gidx == -1
    assert ST.is_solved(source_gidx)


def test_user_example_shape_and_rotated_composite_path_solves_root():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    # Shape is solved directly.
    ST.mark_solution(1, 10)

    # Solve all composite leaves except one mask.
    for gidx in (2, 4, 5, 6, 7):
        ST.mark_solution(gidx, 100 + gidx)

    unresolved_mask = 3
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



def test_program_meta_rows_and_program_x_object_matrix_are_easy_to_inspect():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    rows = GP_meta.rows()
    matrix = GP_X.as_object_array()

    assert rows[0] == {
        "gidx": 0,
        "source": -1,
        "op": "raw_input",
        "dims": 2,
        "params": {},
    }
    assert matrix.shape == (8, 2)
    assert np.array_equal(matrix[1, 0], np.array([2, 3]))



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


def test_valid_generation_partition_shape_requires_dims_gt_zero():
    meta, X = _raw_program(
        "GP",
        [np.array([[1, 2], [3, 4]], dtype=np.int64)],
        raw_op="raw_input",
    )

    assert valid_generation(meta, X, partition_shape, 0)

    shape_gidx = partition_shape(meta, X, 0)

    assert not valid_generation(meta, X, partition_shape, 0)
    assert generation_exists(meta, partition_shape, 0)

    # Shape is 1D, so partition_shape can still be applied to the new gene.
    assert meta.dims[shape_gidx] == 1
    assert valid_generation(meta, X, partition_shape, shape_gidx)

    scalar_gidx = X.append_gene([5])
    meta.append(source=0, op="manual_scalar", dims=0)

    assert not valid_generation(
        meta,
        X,
        partition_shape,
        scalar_gidx,
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


def test_gp_generate_never_retains_equivalent_gene_data():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    before = len(GP_X)

    new_gidx = GP_generate(
        GP_meta,
        GP_X,
        20,
        rng=7,
        operation_names=("partition_shape", "partition_composite"),
    )

    assert len(new_gidx) > 0
    assert len(GP_X) == before + len(new_gidx)
    assert len(GP_meta) == len(GP_X)

    for i in range(len(GP_X)):
        for j in range(i):
            assert not genes_exactly_equal(GP_X[i], GP_X[j])


def test_gp_generate_zero_is_noop():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    assert GP_generate(GP_meta, GP_X, 0, rng=0) == []
    assert len(GP_X) == 8


def test_sp_generate_runs_one_full_partition_step_and_syncs_st():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    before = len(SP_X)

    new_gidx = SP_generate(
        SP_meta,
        SP_X,
        ST,
        rng=11,
        operation_names=("partition_shape", "partition_composite"),
    )

    assert len(new_gidx) >= 1
    assert len(SP_X) == before + len(new_gidx)
    assert len(SP_meta) == len(SP_X)
    assert len(ST) == len(SP_X)

    for gidx in new_gidx:
        assert OP_REGISTRY[SP_meta.op[gidx]].full_partition
        assert ST[gidx].gp_gidx == -1


def test_sp_generate_returns_empty_when_no_registered_full_partition_is_valid():
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
        "entire legal generation space was explored"
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
        "entire legal generation space was explored"
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


def test_partition_bool_trim_1d_returns_offset_and_trimmed_structure():
    meta, X = _raw_program(
        "GP",
        [np.array([False, True, True], dtype=bool)],
        raw_op="raw_input",
    )

    assert valid_generation(meta, X, partition_bool_trim, 0)

    offset_gidx, data_gidx = partition_bool_trim(meta, X, 0)

    assert np.array_equal(
        X[offset_gidx, 0],
        np.array([1], dtype=np.int64),
    )
    assert np.array_equal(
        X[data_gidx, 0],
        np.array([True, True], dtype=bool),
    )
    assert meta.dims[offset_gidx] == 1
    assert meta.dims[data_gidx] == 1


def test_partition_bool_trim_2d_matches_requested_example():
    meta, X = _raw_program(
        "GP",
        [
            np.array(
                [
                    [False, False],
                    [False, True],
                ],
                dtype=bool,
            )
        ],
        raw_op="raw_input",
    )

    offset_gidx, data_gidx = partition_bool_trim(meta, X, 0)

    assert np.array_equal(
        X[offset_gidx, 0],
        np.array([1, 1], dtype=np.int64),
    )
    assert np.array_equal(
        X[data_gidx, 0],
        np.array([[True]], dtype=bool),
    )


def test_partition_bool_trim_finds_nd_boolean_bounding_box():
    source = np.zeros((4, 5, 6), dtype=bool)
    source[1:3, 2:4, 1:5] = True

    meta, X = _raw_program(
        "GP",
        [source],
        raw_op="raw_input",
    )

    offset_gidx, data_gidx = partition_bool_trim(meta, X, 0)

    assert np.array_equal(
        X[offset_gidx, 0],
        np.array([1, 2, 1], dtype=np.int64),
    )
    assert X[data_gidx, 0].shape == (2, 2, 4)
    assert np.all(X[data_gidx, 0])


def test_partition_bool_trim_requires_bool_dim_gt_zero_and_trim_boundary():
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

    scalar_meta = ProgramMeta(side="GP")
    scalar_X = ProgramX(side="GP", sample_count=1)
    scalar_X.append_gene([np.bool_(True)])
    scalar_meta.append(source=-1, op="raw_bool", dims=0)

    assert not valid_generation(
        scalar_meta,
        scalar_X,
        partition_bool_trim,
        0,
    )


def test_partition_bool_trim_all_false_returns_empty_nd_structure():
    source = np.zeros((2, 3), dtype=bool)
    meta, X = _raw_program(
        "GP",
        [source],
        raw_op="raw_input",
    )

    offset_gidx, data_gidx = partition_bool_trim(meta, X, 0)

    assert np.array_equal(
        X[offset_gidx, 0],
        np.array([0, 0], dtype=np.int64),
    )
    assert X[data_gidx, 0].shape == (0, 0)
    assert X[data_gidx, 0].dtype == bool


def test_operation_partition_categories():
    assert OP_REGISTRY["bool_complement"].partition == "or"
    assert OP_REGISTRY["mat2_cwrotate"].partition == "or"
    assert OP_REGISTRY["dim0_flip"].partition == "or"
    assert OP_REGISTRY["dim1_flip"].partition == "or"
    assert OP_REGISTRY["dim2_flip"].partition == "or"

    assert OP_REGISTRY["partition_bool_trim"].partition == "and"

    assert OP_REGISTRY["bool2_union"].partition == "null"
    assert OP_REGISTRY["bool2_intersect"].partition == "null"
    assert OP_REGISTRY["bool_sum"].partition == "null"
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
