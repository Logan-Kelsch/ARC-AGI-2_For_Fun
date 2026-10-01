from types import SimpleNamespace

import numpy as np
import pytest

from notebooks.ops.environment import (
    ProgramMeta,
    ProgramX,
    SolutionTree,
    init_env,
)
from notebooks.ops.ops import (
    GP_generate,
    OP_REGISTRY,
    SP_generate,
    bool2_intersect,
    bool2_union,
    bool_complement,
    dim0_flip,
    dim1_flip,
    dim2_flip,
    equivalent_gene_idx,
    generation_exists,
    gene_atomic_dtypes,
    genes_exactly_equal,
    mat2_cwrotate,
    operation,
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

    assert GP_X.shape == (4, 2)
    assert SP_X.shape == (4, 2)

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
    ]
    assert GP_meta.source == [-1, 0, 0, 0]
    assert GP_meta.dims == [2, 1, 1, 3]
    assert GP_meta.params == [{}, {}, {}, {}]

    assert SP_meta.op == [
        "raw_output",
        "partition_shape",
        "partition_composite",
        "partition_composite",
    ]
    assert SP_meta.source == [-1, 0, 0, 0]
    assert SP_meta.dims == [2, 1, 1, 3]
    assert SP_meta.params == [{}, {}, {}, {}]

    assert len(GP_meta) == len(GP_X) == 4
    assert len(SP_meta) == len(SP_X) == 4


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


def test_partition_composite_generates_color_and_presence_genes():
    meta = ProgramMeta(side="GP")
    X = ProgramX(side="GP", sample_count=2)

    X.append_gene(
        [
            np.array([[0, 1], [1, 0]]),
            np.array([[2, 2, 0]]),
        ]
    )
    meta.append(source=-1, op="raw_input", dims=2)

    color_gidx, presence_gidx = partition_composite(meta, X, 0)

    assert (color_gidx, presence_gidx) == (1, 2)

    assert np.array_equal(X[color_gidx, 0], np.array([0, 1]))
    assert np.array_equal(X[color_gidx, 1], np.array([0, 2]))

    assert X[presence_gidx, 0].shape == (2, 2, 2)
    assert X[presence_gidx, 1].shape == (2, 1, 3)
    assert X[presence_gidx, 0].dtype == bool

    assert np.array_equal(
        X[presence_gidx, 0][0],
        np.array([[True, False], [False, True]]),
    )
    assert np.array_equal(
        X[presence_gidx, 0][1],
        np.array([[False, True], [True, False]]),
    )

    assert meta.source == [-1, 0, 0]
    assert meta.op == [
        "raw_input",
        "partition_composite",
        "partition_composite",
    ]
    assert meta.dims == [2, 1, 3]


def test_default_partition_ops_are_marked_full_partition():
    assert partition_shape.full_partition is True
    assert partition_composite.full_partition is True

    assert OP_REGISTRY["partition_shape"].full_partition is True
    assert OP_REGISTRY["partition_composite"].full_partition is True


def test_non_full_partition_operation_is_allowed_on_gp_but_rejected_on_sp():
    @operation(full_partition=False, output_count=1)
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

    with pytest.raises(PermissionError, match="not full_partition"):
        test_nonpartition(sp_meta, sp_x, 0)

    assert len(sp_meta) == 1
    assert len(sp_x) == 1

    OP_REGISTRY.pop("test_nonpartition", None)


def test_solution_tree_is_derived_from_sp_structure_and_starts_unsolved():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    assert ST.roots == (0,)
    assert len(ST) == 4

    assert ST[0].parents == ()
    assert ST[0].children == [1, 2, 3]

    assert ST[1].parents == (0,)
    assert ST[2].parents == (0,)
    assert ST[3].parents == (0,)

    assert ST[0].op == "raw_output"
    assert ST[1].op == "partition_shape"
    assert ST[2].op == "partition_composite"
    assert ST[3].op == "partition_composite"

    assert [ST[i].dims for i in range(4)] == [2, 1, 1, 3]
    assert all(ST[i].gp_gidx == -1 for i in range(4))
    assert all(not ST[i].solved for i in range(4))


def test_solution_tree_can_record_gp_gene_that_solves_sp_gene():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    ST.mark_solution(sp_gidx=1, gp_gidx=7)

    assert ST[1].gp_gidx == 7
    assert ST[1].solved
    assert ST[0].gp_gidx == -1

    ST.clear_solution(1)

    assert ST[1].gp_gidx == -1
    assert not ST[1].solved


def test_solution_tree_sync_preserves_existing_matches():
    GP_meta, GP_X, SP_meta, SP_X, ST = init_env(_train_pairs())

    ST.mark_solution(1, 5)

    new_gidx = partition_shape(SP_meta, SP_X, 1)
    assert new_gidx == 4

    ST.sync(SP_meta)

    assert len(ST) == 5
    assert ST[1].gp_gidx == 5
    assert ST[4].parents == (1,)
    assert ST[4].gp_gidx == -1


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
    assert matrix.shape == (4, 2)
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
        full_partition=False,
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
    assert len(GP_X) == 4


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
        full_partition=False,
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
        full_partition=False,
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
        full_partition=False,
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
    @operation(
        full_partition=True,
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


def test_only_partition_bool_trim_is_sp_legal_among_new_ops():
    full_partition_names = {
        name
        for name in (
            "bool_complement",
            "bool2_union",
            "bool2_intersect",
            "mat2_cwrotate",
            "dim0_flip",
            "dim1_flip",
            "dim2_flip",
            "partition_bool_trim",
        )
        if OP_REGISTRY[name].full_partition
    }

    assert full_partition_names == {"partition_bool_trim"}


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
