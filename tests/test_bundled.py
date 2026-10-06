from __future__ import annotations

import numpy as np

from notebooks.ops.bundled import (
    BUNDLED_INV_OP_REGISTRY,
    BUNDLED_OP_REGISTRY,
    ComponentPairEvaluationCache,
    GeneComponentRef,
    GP_generate_bundled,
    apply_bundled_inverse,
    apply_bundled_inverse_gene,
    apply_bundled_operation,
    component_gene,
    component_ref,
    component_refs_for_gene,
    init_env_bundled,
    solve_bundled_frontier,
)
from notebooks.ops.environment import ProgramMeta, ProgramX
from notebooks.ops.inv_ops import INV_OP_REGISTRY
from notebooks.ops.ops import OP_REGISTRY


def _program(side: str, values):
    meta = ProgramMeta(side=side)
    X = ProgramX(side=side, sample_count=len(values))
    gidx = X.append_gene(values)
    meta.append(
        source=-1,
        op=f"raw_{side.lower()}",
        dims=int(np.asarray(values[0]).ndim),
    )
    assert gidx == 0
    return meta, X


def test_bundled_registry_covers_every_legacy_operation():
    assert set(BUNDLED_OP_REGISTRY) == set(OP_REGISTRY)

    for name, info in BUNDLED_OP_REGISTRY.items():
        assert info.name == name
        assert info.output_count == 1
        assert info.legacy is OP_REGISTRY[name]


def test_bundled_inverse_registry_covers_every_legacy_inverse():
    assert set(BUNDLED_INV_OP_REGISTRY) == set(INV_OP_REGISTRY)

    for name, info in BUNDLED_INV_OP_REGISTRY.items():
        assert info.legacy is INV_OP_REGISTRY[name]


def test_shape_partition_is_one_gene_with_h_w_subarray():
    grids = [
        np.zeros((2, 3), dtype=np.int64),
        np.zeros((4, 5), dtype=np.int64),
    ]
    meta, X = _program("GP", grids)

    gidx = apply_bundled_operation(
        meta,
        X,
        "partition_shape",
        0,
    )

    assert gidx == 1
    assert len(X) == 2

    first = X[gidx, 0]
    second = X[gidx, 1]

    assert first.shape == (1, 2)
    assert second.shape == (1, 2)
    assert int(first[0, 0]) == 2
    assert int(first[0, 1]) == 3
    assert int(second[0, 0]) == 4
    assert int(second[0, 1]) == 5

    refs = component_refs_for_gene(meta, X, gidx)
    assert [ref.path for ref in refs] == [
        (0, 0),
        (0, 1),
    ]


def test_composite_partition_is_one_gene_of_color_mask_rows():
    grids = [
        np.array(
            [
                [0, 2],
                [2, 0],
            ],
            dtype=np.int64,
        ),
        np.array(
            [
                [2, 2],
                [0, 0],
            ],
            dtype=np.int64,
        ),
    ]
    meta, X = _program("GP", grids)

    gidx = apply_bundled_operation(
        meta,
        X,
        "partition_composite",
        0,
    )

    assert gidx == 1
    assert len(X) == 2

    value = X[gidx, 0]
    assert value.shape == (2, 2)
    assert int(value[0, 0]) == 0
    assert int(value[1, 0]) == 2
    assert np.array_equal(
        value[0, 1],
        grids[0] == 0,
    )
    assert np.array_equal(
        value[1, 1],
        grids[0] == 2,
    )

    reconstructed = apply_bundled_inverse(
        "inv_partition_composite",
        value,
    )
    assert np.array_equal(
        reconstructed,
        grids[0],
    )


def test_manifest_aware_inverse_reconstructs_composite_application():
    grids = [
        np.array(
            [
                [0, 2],
                [2, 0],
            ],
            dtype=np.int64,
        ),
        np.array(
            [
                [2, 2],
                [0, 0],
            ],
            dtype=np.int64,
        ),
    ]
    meta, X = _program("GP", grids)

    gidx = apply_bundled_operation(
        meta,
        X,
        "partition_composite",
        0,
    )

    reconstructed = apply_bundled_inverse_gene(
        meta,
        X,
        gidx,
        0,
    )

    assert np.array_equal(reconstructed, grids[0])


def test_manifest_aware_inverse_preserves_multiple_component_applications():
    grids = [
        np.array(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=np.int64,
        ),
        np.array(
            [
                [1, 1],
                [0, 0],
            ],
            dtype=np.int64,
        ),
    ]
    meta, X = _program("GP", grids)

    composite_gidx = apply_bundled_operation(
        meta,
        X,
        "partition_composite",
        0,
    )
    complement_gidx = apply_bundled_operation(
        meta,
        X,
        "bool_complement",
        composite_gidx,
    )

    reconstructed = apply_bundled_inverse_gene(
        meta,
        X,
        complement_gidx,
        0,
    )

    assert reconstructed.shape == (2, 1)
    assert np.array_equal(
        reconstructed[0, 0],
        grids[0] == 0,
    )
    assert np.array_equal(
        reconstructed[1, 0],
        grids[0] == 1,
    )


def test_bool_operation_maps_over_compatible_components_in_bundle():
    grids = [
        np.array(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=np.int64,
        ),
        np.array(
            [
                [1, 1],
                [0, 0],
            ],
            dtype=np.int64,
        ),
    ]
    meta, X = _program("GP", grids)

    composite_gidx = apply_bundled_operation(
        meta,
        X,
        "partition_composite",
        0,
    )
    complement_gidx = apply_bundled_operation(
        meta,
        X,
        "bool_complement",
        composite_gidx,
    )

    # The scalar color-ID components are ignored by bool_complement; the two
    # boolean mask components each produce one bundled output row.
    value = X[complement_gidx, 0]
    assert value.shape == (2, 1)

    assert np.array_equal(
        value[0, 0],
        np.logical_not(grids[0] == 0),
    )
    assert np.array_equal(
        value[1, 0],
        np.logical_not(grids[0] == 1),
    )


def test_binary_operation_can_draw_two_leaves_from_same_bundle():
    grids = [
        np.array(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=np.int64,
        ),
        np.array(
            [
                [1, 0],
                [0, 1],
            ],
            dtype=np.int64,
        ),
    ]
    meta, X = _program("GP", grids)

    composite_gidx = apply_bundled_operation(
        meta,
        X,
        "partition_composite",
        0,
    )
    union_gidx = apply_bundled_operation(
        meta,
        X,
        "bool2_union",
        (composite_gidx, composite_gidx),
    )

    # The two distinct Boolean color-mask leaves are valid operands even
    # though both live inside the same physical source gene.
    value = X[union_gidx, 0]
    assert value.shape == (1, 1)
    assert np.array_equal(
        value[0, 0],
        np.ones_like(grids[0], dtype=bool),
    )


def test_subject_partition_keeps_all_subject_triples_in_one_gene():
    samples = [
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ]
    meta, X = _program("GP", samples)

    gidx = apply_bundled_operation(
        meta,
        X,
        "partition_bool_subjects",
        0,
    )

    value = X[gidx, 0]
    assert value.shape == (2, 3)
    assert np.asarray(value[0, 0]).ndim == 0
    assert np.asarray(value[0, 1]).ndim == 0
    assert np.asarray(value[0, 2]).ndim == 2
    assert np.asarray(value[1, 2]).ndim == 2


def test_component_cache_distinguishes_paths_inside_same_gene():
    cache = ComponentPairEvaluationCache()
    gp_a = GeneComponentRef(7, (0, 0))
    gp_b = GeneComponentRef(7, (0, 1))
    sp = GeneComponentRef(3, (0, 0))

    cache.mark_evaluated(0, gp_a, sp)

    assert cache.was_evaluated(0, gp_a, sp)
    assert not cache.was_evaluated(0, gp_b, sp)


def test_bundled_init_stores_raw_shape_composite_only():
    training = [
        {
            "input": [
                [0, 1],
                [1, 0],
            ],
            "output": [
                [0, 1],
                [1, 0],
            ],
        },
        {
            "input": [
                [1, 1],
                [0, 0],
            ],
            "output": [
                [1, 1],
                [0, 0],
            ],
        },
    ]

    GP_meta, GP_X, SP_meta, SP_X, ST = init_env_bundled(
        training
    )

    assert len(GP_X) == 3
    assert len(SP_X) == 3
    assert GP_meta.op == [
        "raw_input",
        "partition_shape",
        "partition_composite",
    ]
    assert SP_meta.op == [
        "raw_output",
        "partition_shape",
        "partition_composite",
    ]

    # The raw input/output match exactly, so one component comparison can solve
    # the root directly while all generated bundle data remains in three slots.
    cache = ComponentPairEvaluationCache()
    solved = solve_bundled_frontier(
        GP_meta,
        GP_X,
        SP_meta,
        SP_X,
        ST,
        cache,
    )

    assert solved
    assert ST.solved


def test_component_gene_extracts_virtual_leaf_without_materializing_new_gene():
    grids = [
        np.zeros((2, 3), dtype=np.int64),
        np.zeros((4, 5), dtype=np.int64),
    ]
    meta, X = _program("GP", grids)
    gidx = apply_bundled_operation(
        meta,
        X,
        "partition_shape",
        0,
    )

    before = len(X)
    h_ref = component_ref(meta, gidx, (0, 0))
    h_gene = component_gene(meta, X, h_ref)

    assert len(X) == before
    assert [int(value) for value in h_gene] == [2, 4]


def test_bundled_gp_generation_adds_one_slot_per_operation():
    grids = [
        np.zeros((2, 3), dtype=np.int64),
        np.zeros((4, 5), dtype=np.int64),
    ]
    meta, X = _program("GP", grids)

    generated = GP_generate_bundled(
        meta,
        X,
        1,
        rng=0,
        operation_names=["partition_shape"],
    )

    assert generated == [1]
    assert len(X) == 2
