from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Any, Iterable

import numpy as np

from .environment import (
    ProgramMeta,
    ProgramX,
    STInverseRef,
    STNode,
    STNodeRef,
    STSet,
    SolutionTree,
    source_indices,
)
from .inv_ops import INV_OP_REGISTRY, InverseOperationInfo
from .ops import (
    OP_REGISTRY,
    OperationInfo,
    equivalent_gene_idx,
    sample_operation_params,
    values_exactly_equal,
)
from .solve import (
    _candidate_2d_rule_solutions,
    _candidate_rule_solutions,
)


BUNDLE_META_KEY = "__bundle_v1__"


@dataclass(frozen=True, order=True)
class GeneComponentRef:
    """Stable identity of one semantic component inside one stored gene."""

    gene_id: int
    path: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "gene_id", int(self.gene_id))
        object.__setattr__(
            self,
            "path",
            tuple(int(value) for value in self.path),
        )


@dataclass(frozen=True)
class BundledOperationInfo:
    """Parallel registry metadata for one-slot bundled operations."""

    name: str
    partition: str
    inverse_op: str | None
    source_count: int
    ordered_sources: bool
    legacy: OperationInfo

    @property
    def output_count(self) -> int:
        return 1


@dataclass(frozen=True)
class BundledInverseOperationInfo:
    """Bundled inverse adapter around one legacy inverse."""

    name: str
    forward_op: str
    reconstructive: bool
    legacy: InverseOperationInfo


@dataclass
class ComponentPairEvaluationCache:
    """Stable component-pair cache keyed by gene ID + nested path.

    The legacy cache is a dense matrix over whole-gene identities. Bundled
    genes expose a dynamic number of semantic leaves, so a sparse stable-key
    cache is a better fit and survives pruning / live-index compaction.
    """

    evaluated: dict[
        int,
        set[tuple[GeneComponentRef, GeneComponentRef]],
    ] = field(
        default_factory=lambda: {
            0: set(),
            2: set(),
        }
    )

    def was_evaluated(
        self,
        dims: int,
        gp_ref: GeneComponentRef,
        sp_ref: GeneComponentRef,
    ) -> bool:
        return (gp_ref, sp_ref) in self.evaluated.setdefault(
            int(dims),
            set(),
        )

    def mark_evaluated(
        self,
        dims: int,
        gp_ref: GeneComponentRef,
        sp_ref: GeneComponentRef,
    ) -> None:
        self.evaluated.setdefault(
            int(dims),
            set(),
        ).add((gp_ref, sp_ref))

    def clear(self) -> None:
        self.evaluated.clear()
        self.evaluated.update({0: set(), 2: set()})


@dataclass
class BundledSTNode:
    """One semantic leaf or logical node in the bundled solution tree."""

    node_id: Any
    label: str
    sp_ref: GeneComponentRef | None = None
    dims: int | None = None
    gp_ref: GeneComponentRef | None = None
    solution_rule: str | None = None
    solution_params: dict[str, Any] = field(default_factory=dict)
    derivation: STSet | None = None
    innate: bool = False

    @property
    def directly_solved(self) -> bool:
        return self.innate or self.gp_ref is not None


@dataclass
class BundledSolutionTree:
    """Solution tree whose concrete nodes are gene-component references."""

    nodes: dict[Any, BundledSTNode] = field(default_factory=dict)
    roots: tuple[Any, ...] = ()
    _GP_meta: ProgramMeta | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _GP_X: ProgramX | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _SP_meta: ProgramMeta | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _SP_X: ProgramX | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __getitem__(self, node_id: Any) -> BundledSTNode:
        return self.nodes[node_id]

    def bind_environment(
        self,
        GP_meta: ProgramMeta,
        GP_X: ProgramX,
        SP_meta: ProgramMeta,
        SP_X: ProgramX,
    ) -> None:
        self._GP_meta = GP_meta
        self._GP_X = GP_X
        self._SP_meta = SP_meta
        self._SP_X = SP_X

    @classmethod
    def from_raw_output(
        cls,
        SP_meta: ProgramMeta,
        SP_X: ProgramX,
        *,
        raw_gidx: int = 0,
    ) -> "BundledSolutionTree":
        ref = component_ref(SP_meta, raw_gidx, ())
        dims = component_dims(SP_meta, SP_X, ref)
        node = BundledSTNode(
            node_id=ref,
            label="raw_output",
            sp_ref=ref,
            dims=dims,
        )
        return cls(
            nodes={ref: node},
            roots=(ref,),
        )

    def add_component(
        self,
        ref: GeneComponentRef,
        *,
        label: str,
        dims: int,
    ) -> None:
        self.nodes.setdefault(
            ref,
            BundledSTNode(
                node_id=ref,
                label=str(label),
                sp_ref=ref,
                dims=int(dims),
            ),
        )

    def _eval_requirement(
        self,
        requirement: Any,
        *,
        visiting: set[Any] | None = None,
    ) -> bool:
        if isinstance(requirement, STInverseRef):
            return True

        if isinstance(requirement, STNodeRef):
            return self.is_solved(
                requirement.node_id,
                visiting=visiting,
            )

        if not isinstance(requirement, STSet):
            raise TypeError(
                f"Unknown bundled ST requirement {type(requirement).__name__}."
            )

        if requirement.mode == "AND":
            return all(
                self._eval_requirement(
                    member,
                    visiting=visiting,
                )
                for member in requirement.members
            )

        return any(
            self._eval_requirement(
                member,
                visiting=visiting,
            )
            for member in requirement.members
        )

    def is_solved(
        self,
        node_id: Any,
        *,
        visiting: set[Any] | None = None,
    ) -> bool:
        node = self.nodes[node_id]

        if node.directly_solved:
            return True

        if node.derivation is None:
            return False

        visiting = set() if visiting is None else set(visiting)

        if node_id in visiting:
            return False

        visiting.add(node_id)

        return self._eval_requirement(
            node.derivation,
            visiting=visiting,
        )

    @property
    def solved(self) -> bool:
        return bool(self.roots) and all(
            self.is_solved(root)
            for root in self.roots
        )

    def mark_solution(
        self,
        sp_ref: GeneComponentRef,
        gp_ref: GeneComponentRef,
        *,
        rule: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        if sp_ref not in self.nodes:
            raise KeyError(f"No bundled ST node for {sp_ref!r}.")

        node = self.nodes[sp_ref]
        node.gp_ref = gp_ref
        node.solution_rule = str(rule)
        node.solution_params = dict(params or {})

    def _append_alternative(
        self,
        node_id: Any,
        branch: STSet,
    ) -> None:
        node = self.nodes[node_id]

        if node.derivation is None:
            node.derivation = STSet(
                mode="OR",
                members=[branch],
                label=f"{node.label} alternatives",
            )
            return

        if node.derivation.mode != "OR":
            node.derivation = STSet(
                mode="OR",
                members=[node.derivation, branch],
                label=f"{node.label} alternatives",
            )
            return

        node.derivation.members.append(branch)

    def initialize_output_partition(
        self,
        SP_meta: ProgramMeta,
        SP_X: ProgramX,
        *,
        shape_gidx: int,
        composite_gidx: int,
    ) -> None:
        """Build root <- shape AND composite using virtual component leaves."""
        if len(self.roots) != 1:
            raise ValueError(
                "Bundled initial output partition requires exactly one root."
            )

        shape_refs = component_refs_for_gene(
            SP_meta,
            SP_X,
            shape_gidx,
        )
        composite_refs = component_refs_for_gene(
            SP_meta,
            SP_X,
            composite_gidx,
        )

        if len(shape_refs) != 2:
            raise ValueError(
                "Bundled shape partition must expose exactly h and w."
            )
        if len(composite_refs) < 2 or len(composite_refs) % 2 != 0:
            raise ValueError(
                "Bundled composite partition must expose color/mask pairs."
            )

        for ref in (*shape_refs, *composite_refs):
            self.add_component(
                ref,
                label=f"sp:{ref.gene_id}:{ref.path}",
                dims=component_dims(SP_meta, SP_X, ref),
            )

        shape_id = "bundle:shape"
        composite_id = "bundle:composite"

        self.nodes[shape_id] = BundledSTNode(
            node_id=shape_id,
            label="shape",
            derivation=STSet(
                mode="OR",
                members=[
                    STSet(
                        mode="AND",
                        members=[
                            STNodeRef(ref)
                            for ref in shape_refs
                        ],
                        label="h AND w",
                        partition="and",
                    )
                ],
                label="shape alternatives",
            ),
        )

        self.nodes[composite_id] = BundledSTNode(
            node_id=composite_id,
            label="composite",
            derivation=STSet(
                mode="OR",
                members=[
                    STSet(
                        mode="AND",
                        members=[
                            STInverseRef("inv_partition_composite"),
                            *[
                                STNodeRef(ref)
                                for ref in composite_refs
                            ],
                        ],
                        label="assemble bundled composite",
                        partition="and",
                    )
                ],
                label="composite alternatives",
            ),
        )

        root = self.roots[0]
        self.nodes[root].derivation = STSet(
            mode="OR",
            members=[
                STSet(
                    mode="AND",
                    members=[
                        STInverseRef("inv_partition_shape"),
                        STNodeRef(shape_id),
                        STNodeRef(composite_id),
                    ],
                    label="shape AND composite",
                    partition="and",
                )
            ],
            label="root alternatives",
        )

    def register_generation(
        self,
        SP_meta: ProgramMeta,
        SP_X: ProgramX,
        *,
        source_gidx: int | tuple[int, ...],
        generated_gidx: int,
        info: BundledOperationInfo,
    ) -> None:
        """Attach reversible component-level alternatives from one bundle op.

        Each bundled operation manifest records which source component paths
        generated which output component paths. For reconstructive single-source
        operations, an inverse branch is attached to that exact source leaf.

        Initial shape/composite decomposition remains special because
        inv_partition_shape reconstructs jointly with composite content.
        """
        if info.partition not in {"and", "or"}:
            return

        if info.name in {
            "partition_shape",
            "partition_bool_trim",
        }:
            # These inverses require reconstruction context beyond their direct
            # emitted components. partition_shape is handled explicitly by the
            # initial shape+composite root branch; trim needs a source-shape
            # proof before it can become an executable ST alternative.
            return

        manifest = bundle_manifest(SP_meta, generated_gidx)

        if manifest is None:
            return

        source_gidxs = (
            (int(source_gidx),)
            if isinstance(source_gidx, int)
            else tuple(int(value) for value in source_gidx)
        )

        for ref in component_refs_for_gene(
            SP_meta,
            SP_X,
            generated_gidx,
        ):
            self.add_component(
                ref,
                label=f"sp:{ref.gene_id}:{ref.path}",
                dims=component_dims(SP_meta, SP_X, ref),
            )

        for application in manifest.get("applications", []):
            source_paths = [
                tuple(int(v) for v in path)
                for path in application["source_paths"]
            ]
            output_paths = [
                tuple(int(v) for v in path)
                for path in application["output_paths"]
            ]

            if len(source_gidxs) != 1 or len(source_paths) != 1:
                continue

            source_ref = component_ref(
                SP_meta,
                source_gidxs[0],
                source_paths[0],
            )

            if source_ref not in self.nodes:
                self.add_component(
                    source_ref,
                    label=f"sp:{source_ref.gene_id}:{source_ref.path}",
                    dims=component_dims(
                        SP_meta,
                        SP_X,
                        source_ref,
                    ),
                )

            output_refs = [
                component_ref(
                    SP_meta,
                    generated_gidx,
                    path,
                )
                for path in output_paths
            ]

            branch = STSet(
                mode="AND",
                members=[
                    STInverseRef(str(info.inverse_op)),
                    *[
                        STNodeRef(ref)
                        for ref in output_refs
                    ],
                ],
                label=f"{info.inverse_op} AND bundled children",
                partition=info.partition,
            )
            self._append_alternative(source_ref, branch)

    def unresolved_frontier_refs(self) -> tuple[GeneComponentRef, ...]:
        if not self.roots:
            return ()

        result: list[GeneComponentRef] = []
        seen: set[Any] = set()

        def visit_requirement(requirement: Any) -> None:
            if isinstance(requirement, STInverseRef):
                return
            if isinstance(requirement, STNodeRef):
                visit_node(requirement.node_id)
                return
            if isinstance(requirement, STSet):
                for member in requirement.members:
                    visit_requirement(member)
                return
            raise TypeError(
                f"Unknown bundled ST requirement {type(requirement).__name__}."
            )

        def visit_node(node_id: Any) -> None:
            if node_id in seen:
                return
            seen.add(node_id)

            if self.is_solved(node_id):
                return

            node = self.nodes[node_id]

            if node.sp_ref is not None:
                result.append(node.sp_ref)

            if node.derivation is not None:
                visit_requirement(node.derivation)

        for root in self.roots:
            visit_node(root)

        return tuple(result)


@dataclass(frozen=True)
class BundledComponentSolution:
    sp_ref: GeneComponentRef
    gp_ref: GeneComponentRef
    rule: str
    params: dict[str, Any]
    complexity_level: int


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.copy() if hasattr(value, "copy") else value
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    if isinstance(value, dict):
        return {
            str(key): _copy_value(item)
            for key, item in value.items()
        }
    return value


def _copy_params(params: dict[str, Any] | None) -> dict[str, Any]:
    return {
        str(key): _copy_value(value)
        for key, value in (params or {}).items()
        if str(key) != BUNDLE_META_KEY
    }


def _pack_gene(values: Iterable[Any]) -> np.ndarray:
    values = list(values)
    result = np.empty(len(values), dtype=object)

    for index, value in enumerate(values):
        result[index] = _copy_value(value)

    return result


def _value_dims(value: Any) -> int:
    return int(np.asarray(value).ndim)


def _gene_dims(gene: np.ndarray) -> int:
    dims = {
        _value_dims(value)
        for value in gene
    }

    if len(dims) != 1:
        raise ValueError(
            f"Component dims vary across samples: {sorted(dims)}."
        )

    return dims.pop()


def _atomic_dtype_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return (str(np.dtype(value.dtype)),)

        result: set[str] = set()
        for item in value.flat:
            result.update(_atomic_dtype_strings(item))
        return tuple(sorted(result))

    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for item in value:
            result.update(_atomic_dtype_strings(item))
        return tuple(sorted(result))

    return (str(np.dtype(np.asarray(value).dtype)),)


def _gene_dtype_strings(gene: np.ndarray) -> tuple[str, ...]:
    result: set[str] = set()

    for value in gene:
        result.update(_atomic_dtype_strings(value))

    return tuple(sorted(result))


def _object_matrix(rows: list[list[Any]]) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=object)

    width = len(rows[0])

    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError("Bundle rows must have one consistent positive width.")

    result = np.empty((len(rows), width), dtype=object)

    for row_idx, row in enumerate(rows):
        for col_idx, value in enumerate(row):
            result[row_idx, col_idx] = _copy_value(value)

    return result


def _group_width(op_name: str, output_count: int) -> int:
    requested = {
        "partition_shape": 2,
        "partition_composite": 2,
        "partition_bool_trim": 3,
        "partition_bool_subjects": 3,
    }.get(str(op_name), 1)

    if output_count < 1 or output_count % requested != 0:
        return 1

    return requested


def _normalize_source(
    info: OperationInfo,
    source_idx: Any,
) -> tuple[int, ...]:
    if isinstance(source_idx, np.generic):
        source_idx = source_idx.item()

    if isinstance(source_idx, np.ndarray):
        source_idx = source_idx.tolist()

    if isinstance(source_idx, (list, tuple)):
        indices = tuple(int(value) for value in source_idx)
    else:
        indices = (int(source_idx),)

    if len(indices) != info.source_count:
        raise ValueError(
            f"{info.name!r} expects {info.source_count} source gene(s), "
            f"got {len(indices)}."
        )

    if not info.ordered_sources and len(indices) > 1:
        indices = tuple(sorted(indices))

    return indices


def bundle_manifest(
    meta: ProgramMeta,
    gidx: int,
) -> dict[str, Any] | None:
    payload = meta.params[int(gidx)].get(BUNDLE_META_KEY)

    if payload is None:
        return None

    return payload


def is_bundled_gene(
    meta: ProgramMeta,
    gidx: int,
) -> bool:
    return bundle_manifest(meta, int(gidx)) is not None


def _gidx_from_gene_id(
    meta: ProgramMeta,
    gene_id: int,
) -> int:
    gene_id = int(gene_id)

    try:
        return meta.gene_id.index(gene_id)
    except ValueError as exc:
        raise KeyError(
            f"Stable gene ID {gene_id} is not live in {meta.side}."
        ) from exc


def component_ref(
    meta: ProgramMeta,
    gidx: int,
    path: Iterable[int] = (),
) -> GeneComponentRef:
    return GeneComponentRef(
        meta.stable_id(int(gidx)),
        tuple(int(value) for value in path),
    )


def _extract_component_value(
    value: Any,
    path: tuple[int, ...],
) -> Any:
    if not path:
        return _copy_value(value)

    current = value

    for index in path:
        current = current[index]

    return _copy_value(current)


def component_gene_by_gidx(
    X: ProgramX,
    gidx: int,
    path: Iterable[int] = (),
) -> np.ndarray:
    path = tuple(int(value) for value in path)

    return _pack_gene(
        _extract_component_value(value, path)
        for value in X[int(gidx)]
    )


def component_gene(
    meta: ProgramMeta,
    X: ProgramX,
    ref: GeneComponentRef,
) -> np.ndarray:
    gidx = _gidx_from_gene_id(meta, ref.gene_id)
    return component_gene_by_gidx(X, gidx, ref.path)


def component_paths_for_gene(
    meta: ProgramMeta,
    gidx: int,
) -> tuple[tuple[int, ...], ...]:
    manifest = bundle_manifest(meta, int(gidx))

    if manifest is None:
        return ((),)

    return tuple(
        tuple(int(value) for value in item["path"])
        for item in manifest.get("components", [])
    )


def component_refs_for_gene(
    meta: ProgramMeta,
    X: ProgramX,
    gidx: int,
) -> tuple[GeneComponentRef, ...]:
    del X
    return tuple(
        component_ref(meta, gidx, path)
        for path in component_paths_for_gene(meta, gidx)
    )


def component_dims(
    meta: ProgramMeta,
    X: ProgramX,
    ref: GeneComponentRef,
) -> int:
    gidx = _gidx_from_gene_id(meta, ref.gene_id)
    manifest = bundle_manifest(meta, gidx)

    if manifest is not None:
        for item in manifest.get("components", []):
            if tuple(item["path"]) == ref.path:
                return int(item["dims"])

    return _gene_dims(component_gene(meta, X, ref))


def component_dtype_strings(
    meta: ProgramMeta,
    X: ProgramX,
    ref: GeneComponentRef,
) -> tuple[str, ...]:
    gidx = _gidx_from_gene_id(meta, ref.gene_id)
    manifest = bundle_manifest(meta, gidx)

    if manifest is not None:
        for item in manifest.get("components", []):
            if tuple(item["path"]) == ref.path:
                return tuple(str(v) for v in item["dtypes"])

    return _gene_dtype_strings(component_gene(meta, X, ref))


def all_component_refs(
    meta: ProgramMeta,
    X: ProgramX,
    *,
    dims: int | None = None,
) -> tuple[GeneComponentRef, ...]:
    refs: list[GeneComponentRef] = []

    for gidx in range(len(X)):
        for ref in component_refs_for_gene(meta, X, gidx):
            if dims is not None and component_dims(meta, X, ref) != int(dims):
                continue
            refs.append(ref)

    return tuple(refs)


def _temp_program(
    side: str,
    source_genes: list[np.ndarray],
) -> tuple[ProgramMeta, ProgramX]:
    meta = ProgramMeta(side=side)
    X = ProgramX(
        side=side,
        sample_count=len(source_genes[0]),
    )

    for index, gene in enumerate(source_genes):
        gidx = X.append_gene(
            _copy_value(value)
            for value in gene
        )
        meta_idx = meta.append(
            source=-1,
            op=f"bundle_source_{index}",
            dims=_gene_dims(X[gidx]),
        )

        if gidx != meta_idx:
            raise RuntimeError(
                "Temporary bundled operation metadata/data diverged."
            )

    return meta, X


def _legacy_application(
    info: OperationInfo,
    side: str,
    source_genes: list[np.ndarray],
    params: dict[str, Any],
) -> list[np.ndarray] | None:
    temp_meta, temp_X = _temp_program(side, source_genes)

    source_arg: Any = (
        0
        if info.source_count == 1
        else tuple(range(info.source_count))
    )

    try:
        result = info.func(
            temp_meta,
            temp_X,
            source_arg,
            **_copy_params(params),
        )
    except (ValueError, TypeError, PermissionError, IndexError):
        return None

    if isinstance(result, (tuple, list)):
        output_gidxs = tuple(int(value) for value in result)
    else:
        output_gidxs = (int(result),)

    return [
        _copy_value(temp_X[gidx])
        for gidx in output_gidxs
    ]


def _transition_exists(
    meta: ProgramMeta,
    op_name: str,
    source_idx: tuple[int, ...],
    params: dict[str, Any],
) -> bool:
    source_value: Any = (
        source_idx[0]
        if len(source_idx) == 1
        else tuple(source_idx)
    )

    for gidx in range(len(meta)):
        if meta.op[gidx] != op_name:
            continue
        if meta.source[gidx] != source_value:
            continue

        existing = {
            key: value
            for key, value in meta.params[gidx].items()
            if key != BUNDLE_META_KEY
        }

        if values_exactly_equal(
            existing,
            _copy_params(params),
        ):
            return True

    return False


def _component_combinations(
    meta: ProgramMeta,
    X: ProgramX,
    source_gidxs: tuple[int, ...],
    *,
    ordered_sources: bool,
) -> Iterable[tuple[GeneComponentRef, ...]]:
    """Yield logical source tuples independently of physical bundle slots.

    A multi-source operation may legitimately draw two distinct components from
    the same stored bundle. Component identity, rather than physical gidx
    identity, therefore determines whether operands are distinct.
    """
    ref_lists = [
        component_refs_for_gene(meta, X, gidx)
        for gidx in source_gidxs
    ]
    seen: set[tuple[GeneComponentRef, ...]] = set()

    for refs in itertools.product(*ref_lists):
        # Legacy multi-source operations never used one exact gene twice. Keep
        # that semantic constraint at component granularity.
        if len(refs) > 1 and len(set(refs)) != len(refs):
            continue

        key = (
            tuple(refs)
            if ordered_sources
            else tuple(sorted(refs))
        )

        if key in seen:
            continue

        seen.add(key)
        yield tuple(refs)


def build_bundled_operation_values(
    meta: ProgramMeta,
    X: ProgramX,
    op: str | BundledOperationInfo,
    source_idx: Any,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Evaluate one bundled operation without appending it.

    Every legal component-level application of the legacy operation is run.
    The legacy outputs are grouped into rows and all rows are stored inside one
    object-valued gene.
    """
    if isinstance(op, str):
        refresh_bundled_registries()

    info = (
        BUNDLED_OP_REGISTRY[str(op)]
        if isinstance(op, str)
        else op
    )
    legacy = info.legacy
    source_gidxs = _normalize_source(legacy, source_idx)
    params = _copy_params(params)

    for gidx in source_gidxs:
        if gidx < 0 or gidx >= len(X):
            raise IndexError(
                f"Bundled source gidx {gidx} is outside the live program."
            )

    applications: list[dict[str, Any]] = []

    for source_refs in _component_combinations(
        meta,
        X,
        source_gidxs,
        ordered_sources=legacy.ordered_sources,
    ):
        source_genes = [
            component_gene(meta, X, ref)
            for ref in source_refs
        ]
        outputs = _legacy_application(
            legacy,
            meta.side,
            source_genes,
            params,
        )

        if not outputs:
            continue

        applications.append(
            {
                "source_refs": source_refs,
                "outputs": outputs,
            }
        )

    if not applications:
        raise ValueError(
            f"No legal component application for bundled operation "
            f"{legacy.name!r} on source {source_gidxs}."
        )

    sample_values: list[np.ndarray] = []
    components: list[dict[str, Any]] = []
    manifest_apps: list[dict[str, Any]] = []

    # Row allocation is shared across samples because each legacy application
    # has one fixed output count across all samples.
    row_cursor = 0

    for app_index, application in enumerate(applications):
        outputs = application["outputs"]
        width = _group_width(legacy.name, len(outputs))
        row_count = len(outputs) // width
        output_paths: list[tuple[int, int]] = []

        for slot, output_gene in enumerate(outputs):
            row = row_cursor + (slot // width)
            col = slot % width
            path = (row, col)
            output_paths.append(path)
            components.append(
                {
                    "path": list(path),
                    "dims": _gene_dims(output_gene),
                    "dtypes": list(
                        _gene_dtype_strings(output_gene)
                    ),
                    "legacy_output_slot": int(slot),
                    "application_index": int(app_index),
                }
            )

        manifest_apps.append(
            {
                "source_paths": [
                    list(ref.path)
                    for ref in application["source_refs"]
                ],
                "output_paths": [
                    list(path)
                    for path in output_paths
                ],
            }
        )
        row_cursor += row_count

    for sample_idx in range(X.sample_count):
        rows: list[list[Any]] = []

        for application in applications:
            outputs = application["outputs"]
            width = _group_width(legacy.name, len(outputs))

            for start in range(0, len(outputs), width):
                rows.append(
                    [
                        _copy_value(
                            outputs[start + offset][sample_idx]
                        )
                        for offset in range(width)
                    ]
                )

        sample_values.append(_object_matrix(rows))

    manifest = {
        "version": 1,
        "operation": legacy.name,
        "components": components,
        "applications": manifest_apps,
    }

    return sample_values, manifest


def apply_bundled_operation(
    meta: ProgramMeta,
    X: ProgramX,
    op: str | BundledOperationInfo,
    source_idx: Any,
    *,
    params: dict[str, Any] | None = None,
    reject_semantic_duplicate: bool = True,
) -> int:
    """Append exactly one bundled gene for one operation invocation."""
    if isinstance(op, str):
        refresh_bundled_registries()

    info = (
        BUNDLED_OP_REGISTRY[str(op)]
        if isinstance(op, str)
        else op
    )
    source_gidxs = _normalize_source(info.legacy, source_idx)
    params = _copy_params(params)

    if meta.side == "SP" and info.partition == "null":
        raise PermissionError(
            f"Bundled SP may not apply NULL operation {info.name!r}."
        )

    if _transition_exists(
        meta,
        info.name,
        source_gidxs,
        params,
    ):
        raise ValueError(
            f"Bundled transition {info.name!r} on {source_gidxs} "
            "already exists."
        )

    values, manifest = build_bundled_operation_values(
        meta,
        X,
        info,
        source_gidxs,
        params=params,
    )
    packed = _pack_gene(values)

    if (
        reject_semantic_duplicate
        and equivalent_gene_idx(X, packed) >= 0
    ):
        raise ValueError(
            f"Bundled operation {info.name!r} produced an existing gene."
        )

    gidx = X.append_gene(values)
    source_value: Any = (
        source_gidxs[0]
        if len(source_gidxs) == 1
        else tuple(source_gidxs)
    )
    meta_params = _copy_params(params)
    meta_params[BUNDLE_META_KEY] = manifest

    meta_gidx = meta.append(
        source=source_value,
        op=info.name,
        # Container dimensionality is intentionally not semantic. All solver
        # filtering uses component manifest dimensions instead.
        dims=2,
        params=meta_params,
    )

    if meta_gidx != gidx:
        raise RuntimeError(
            "Bundled operation metadata/data indices diverged."
        )

    return gidx


def try_bundled_operation(
    meta: ProgramMeta,
    X: ProgramX,
    op: str | BundledOperationInfo,
    source_idx: Any,
    *,
    params: dict[str, Any] | None = None,
) -> int | None:
    try:
        return apply_bundled_operation(
            meta,
            X,
            op,
            source_idx,
            params=params,
        )
    except (
        ValueError,
        TypeError,
        PermissionError,
        IndexError,
    ):
        return None


def _flatten_bundle_argument(value: Any) -> list[Any]:
    if (
        isinstance(value, np.ndarray)
        and value.dtype == object
        and value.ndim == 2
    ):
        return [
            _copy_value(item)
            for item in value.flat
        ]

    return [_copy_value(value)]


def apply_bundled_inverse(
    inverse: str | BundledInverseOperationInfo,
    *values: Any,
) -> Any:
    """Flatten bundled arguments and dispatch to the legacy inverse."""
    if isinstance(inverse, str):
        refresh_bundled_registries()

    info = (
        BUNDLED_INV_OP_REGISTRY[str(inverse)]
        if isinstance(inverse, str)
        else inverse
    )
    args: list[Any] = []

    for value in values:
        args.extend(_flatten_bundle_argument(value))

    return info.legacy.func(*args)


def apply_bundled_inverse_gene(
    meta: ProgramMeta,
    X: ProgramX,
    gidx: int,
    sample_idx: int,
    *,
    inverse: str | BundledInverseOperationInfo | None = None,
    context: Iterable[Any] = (),
) -> Any:
    """Apply an inverse once per component application stored in one gene.

    The bundle manifest preserves application boundaries. This matters when one
    operation call mapped over several compatible source components:

        bool_complement(bundle_of_masks)

    is one stored output gene but represents one inverse call per mask.

    For a partition such as partition_composite, all color/mask rows produced
    from one source component remain one application and are passed together to
    the inverse.

    context supplies any additional reconstruction arguments required by
    context-dependent inverses (for example the original shape for trim).
    """
    gidx = int(gidx)
    sample_idx = int(sample_idx)
    manifest = bundle_manifest(meta, gidx)

    if manifest is None:
        raise ValueError(
            "apply_bundled_inverse_gene requires a bundled generated gene."
        )

    op_name = str(meta.op[gidx])
    refresh_bundled_registries()

    if inverse is None:
        try:
            inverse_name = BUNDLED_OP_REGISTRY[op_name].inverse_op
        except KeyError as exc:
            raise KeyError(
                f"No bundled operation metadata for {op_name!r}."
            ) from exc

        if inverse_name is None:
            raise ValueError(
                f"Bundled operation {op_name!r} has no inverse."
            )

        inverse_info = BUNDLED_INV_OP_REGISTRY[inverse_name]
    elif isinstance(inverse, str):
        inverse_info = BUNDLED_INV_OP_REGISTRY[inverse]
    else:
        inverse_info = inverse

    value = X[gidx, sample_idx]
    context = tuple(_copy_value(item) for item in context)
    results: list[Any] = []

    for application in manifest.get("applications", []):
        outputs = [
            _extract_component_value(
                value,
                tuple(int(v) for v in path),
            )
            for path in application.get("output_paths", [])
        ]

        args = [
            *outputs,
            *context,
        ]
        results.append(
            inverse_info.legacy.func(*args)
        )

    if not results:
        raise ValueError(
            f"Bundled gene {gidx} has no inverse applications."
        )

    if len(results) == 1:
        return _copy_value(results[0])

    return _object_matrix(
        [[result] for result in results]
    )


def _build_bundled_registries() -> tuple[
    dict[str, BundledOperationInfo],
    dict[str, BundledInverseOperationInfo],
]:
    operations = {
        name: BundledOperationInfo(
            name=name,
            partition=info.partition,
            inverse_op=info.inverse_op,
            source_count=info.source_count,
            ordered_sources=info.ordered_sources,
            legacy=info,
        )
        for name, info in OP_REGISTRY.items()
    }
    inverses = {
        name: BundledInverseOperationInfo(
            name=name,
            forward_op=info.forward_op,
            reconstructive=info.reconstructive,
            legacy=info,
        )
        for name, info in INV_OP_REGISTRY.items()
    }
    return operations, inverses


LEGACY_OP_REGISTRY = OP_REGISTRY
LEGACY_INV_OP_REGISTRY = INV_OP_REGISTRY

BUNDLED_OP_REGISTRY, BUNDLED_INV_OP_REGISTRY = (
    _build_bundled_registries()
)


def refresh_bundled_registries() -> None:
    """Synchronize bundled adapters with the live legacy registries."""
    operations, inverses = _build_bundled_registries()

    BUNDLED_OP_REGISTRY.clear()
    BUNDLED_OP_REGISTRY.update(operations)

    BUNDLED_INV_OP_REGISTRY.clear()
    BUNDLED_INV_OP_REGISTRY.update(inverses)


def _source_gene_candidates(
    info: BundledOperationInfo,
    gene_count: int,
) -> list[Any]:
    if info.source_count == 1:
        return list(range(gene_count))

    indices = range(gene_count)

    # Repetition at the physical-gene level is intentional: one bundle can
    # contain multiple distinct semantic leaves that satisfy a binary op.
    # Exact repeated component refs are filtered later.
    if info.ordered_sources:
        return list(
            itertools.product(
                indices,
                repeat=info.source_count,
            )
        )

    return list(
        itertools.combinations_with_replacement(
            indices,
            info.source_count,
        )
    )


def _rng(
    rng: np.random.Generator | int | None,
) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng

    return np.random.default_rng(rng)


def GP_generate_bundled(
    GP_meta: ProgramMeta,
    GP_X: ProgramX,
    n_new_genes: int,
    *,
    rng: np.random.Generator | int | None = None,
    max_attempts_per_generation: int = 100,
    operation_names: Iterable[str] | None = None,
) -> list[int]:
    """Generate bundled GP genes; every accepted operation adds one gene."""
    if GP_meta.side != "GP" or GP_X.side != "GP":
        raise ValueError(
            "GP_generate_bundled requires GP-side ProgramMeta/ProgramX."
        )

    n_new_genes = int(n_new_genes)

    if n_new_genes < 0:
        raise ValueError("n_new_genes must be >= 0.")

    refresh_bundled_registries()
    rng = _rng(rng)
    allowed = (
        None
        if operation_names is None
        else {str(name) for name in operation_names}
    )
    infos = [
        info
        for name, info in BUNDLED_OP_REGISTRY.items()
        if allowed is None or name in allowed
    ]
    generated: list[int] = []
    failures = 0

    while len(generated) < n_new_genes:
        if not infos:
            break

        info = infos[int(rng.integers(len(infos)))]
        candidates = _source_gene_candidates(
            info,
            len(GP_X),
        )

        if not candidates:
            failures += 1
            continue

        source = candidates[
            int(rng.integers(len(candidates)))
        ]
        params = sample_operation_params(
            info.legacy,
            rng,
        )
        gidx = try_bundled_operation(
            GP_meta,
            GP_X,
            info,
            source,
            params=params,
        )

        if gidx is None:
            failures += 1
        else:
            generated.append(gidx)
            failures = 0

        if failures >= int(max_attempts_per_generation):
            break

    return generated


def SP_generate_bundled(
    SP_meta: ProgramMeta,
    SP_X: ProgramX,
    ST: BundledSolutionTree | None = None,
    *,
    rng: np.random.Generator | int | None = None,
    max_attempts: int = 100,
    operation_names: Iterable[str] | None = None,
) -> list[int]:
    """Generate one reversible bundled SP operation in exactly one gene."""
    if SP_meta.side != "SP" or SP_X.side != "SP":
        raise ValueError(
            "SP_generate_bundled requires SP-side ProgramMeta/ProgramX."
        )

    refresh_bundled_registries()
    rng = _rng(rng)
    allowed = (
        None
        if operation_names is None
        else {str(name) for name in operation_names}
    )
    infos = [
        info
        for name, info in BUNDLED_OP_REGISTRY.items()
        if info.partition != "null"
        and (allowed is None or name in allowed)
    ]

    for _ in range(int(max_attempts)):
        if not infos:
            return []

        info = infos[int(rng.integers(len(infos)))]
        candidates = _source_gene_candidates(
            info,
            len(SP_X),
        )

        if not candidates:
            continue

        source = candidates[
            int(rng.integers(len(candidates)))
        ]
        params = sample_operation_params(
            info.legacy,
            rng,
        )
        gidx = try_bundled_operation(
            SP_meta,
            SP_X,
            info,
            source,
            params=params,
        )

        if gidx is None:
            continue

        if ST is not None:
            ST.register_generation(
                SP_meta,
                SP_X,
                source_gidx=source,
                generated_gidx=gidx,
                info=info,
            )

        return [gidx]

    return []


def solve_bundled_frontier(
    GP_meta: ProgramMeta,
    GP_X: ProgramX,
    SP_meta: ProgramMeta,
    SP_X: ProgramX,
    ST: BundledSolutionTree,
    cache: ComponentPairEvaluationCache,
) -> list[BundledComponentSolution]:
    """Solve ST leaves by evaluating GP/SP component views.

    Stored genes remain bundled. Only temporary component views are extracted
    for 0D and 2D rule evaluation. Cache identity includes component path.
    """
    solved: list[BundledComponentSolution] = []
    gp_refs_by_dim = {
        dims: all_component_refs(
            GP_meta,
            GP_X,
            dims=dims,
        )
        for dims in (0, 2)
    }

    for sp_ref in ST.unresolved_frontier_refs():
        dims = component_dims(
            SP_meta,
            SP_X,
            sp_ref,
        )

        if dims not in {0, 2}:
            continue

        y_gene = component_gene(
            SP_meta,
            SP_X,
            sp_ref,
        )
        candidates: list[
            tuple[
                int,
                tuple[Any, ...],
                int,
                tuple[int, ...],
                GeneComponentRef,
                str,
                dict[str, Any],
            ]
        ] = []

        for gp_ref in gp_refs_by_dim[dims]:
            if cache.was_evaluated(
                dims,
                gp_ref,
                sp_ref,
            ):
                continue

            x_gene = component_gene(
                GP_meta,
                GP_X,
                gp_ref,
            )

            try:
                matches = (
                    _candidate_rule_solutions(
                        x_gene,
                        y_gene,
                    )
                    if dims == 0
                    else _candidate_2d_rule_solutions(
                        x_gene,
                        y_gene,
                    )
                )
            except (TypeError, ValueError, IndexError):
                matches = []

            cache.mark_evaluated(
                dims,
                gp_ref,
                sp_ref,
            )

            for level, parameter_key, rule, params in matches:
                candidates.append(
                    (
                        int(level),
                        tuple(parameter_key),
                        int(gp_ref.gene_id),
                        tuple(gp_ref.path),
                        gp_ref,
                        str(rule),
                        dict(params),
                    )
                )

        if not candidates:
            continue

        (
            level,
            _,
            _,
            _,
            gp_ref,
            rule,
            params,
        ) = min(
            candidates,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
            ),
        )

        ST.mark_solution(
            sp_ref,
            gp_ref,
            rule=rule,
            params=params,
        )
        solved.append(
            BundledComponentSolution(
                sp_ref=sp_ref,
                gp_ref=gp_ref,
                rule=rule,
                params=params,
                complexity_level=level,
            )
        )

    return solved



def _physical_source_tuple(source: Any) -> tuple[int, ...]:
    """Preserve physical source multiplicity for bundled lineage."""
    if isinstance(source, np.generic):
        source = source.item()

    if isinstance(source, np.ndarray):
        source = source.tolist()

    if isinstance(source, (list, tuple)):
        return tuple(int(value) for value in source if int(value) >= 0)

    value = int(source)
    return () if value < 0 else (value,)


def _legacy_params(meta: ProgramMeta, gidx: int) -> dict[str, Any]:
    return {
        str(key): _copy_value(value)
        for key, value in meta.params[int(gidx)].items()
        if str(key) != BUNDLE_META_KEY
    }


def materialize_legacy_program(
    meta: ProgramMeta,
    X: ProgramX,
) -> tuple[
    ProgramMeta,
    ProgramX,
    dict[GeneComponentRef, int],
]:
    """Expand one bundled program into a temporary legacy component program.

    This is intentionally a compiler bridge, not search storage. The bundled
    program remains untouched. Each semantic component is materialized once in
    a temporary ProgramMeta/ProgramX so the existing Kelschinator can compile
    the solved proof without learning about the bundled representation.

    Component lineage is reconstructed from the bundle manifest. In particular,
    a physical bundled gene that contains multiple logical applications becomes
    multiple legacy transition groups with the correct source component refs.
    """
    legacy_meta = ProgramMeta(side=meta.side)
    legacy_X = ProgramX(
        side=X.side,
        sample_count=X.sample_count,
    )
    ref_to_gidx: dict[GeneComponentRef, int] = {}

    for physical_gidx in range(len(X)):
        manifest = bundle_manifest(meta, physical_gidx)
        physical_sources = _physical_source_tuple(
            meta.source[physical_gidx]
        )

        if manifest is None:
            ref = component_ref(meta, physical_gidx, ())
            values = component_gene_by_gidx(
                X,
                physical_gidx,
                (),
            )

            if not physical_sources:
                legacy_source: Any = -1
            else:
                mapped_sources = tuple(
                    ref_to_gidx[
                        component_ref(meta, parent_gidx, ())
                    ]
                    for parent_gidx in physical_sources
                )
                legacy_source = (
                    mapped_sources[0]
                    if len(mapped_sources) == 1
                    else mapped_sources
                )

            gidx = legacy_X.append_gene(values)
            meta_gidx = legacy_meta.append(
                source=legacy_source,
                op=meta.op[physical_gidx],
                dims=_gene_dims(values),
                params=_legacy_params(meta, physical_gidx),
            )
            if gidx != meta_gidx:
                raise RuntimeError(
                    "Legacy materialization metadata/data diverged."
                )
            ref_to_gidx[ref] = gidx
            continue

        applications = manifest.get("applications", [])

        for component in manifest.get("components", []):
            path = tuple(
                int(value)
                for value in component["path"]
            )
            application_index = int(
                component["application_index"]
            )

            if (
                application_index < 0
                or application_index >= len(applications)
            ):
                raise RuntimeError(
                    "Bundled component references an invalid application."
                )

            application = applications[application_index]
            source_paths = [
                tuple(int(value) for value in source_path)
                for source_path in application.get(
                    "source_paths",
                    [],
                )
            ]

            if len(source_paths) != len(physical_sources):
                raise RuntimeError(
                    "Bundled manifest source-path arity does not match "
                    "physical source arity."
                )

            mapped_sources = tuple(
                ref_to_gidx[
                    component_ref(
                        meta,
                        parent_gidx,
                        source_path,
                    )
                ]
                for parent_gidx, source_path in zip(
                    physical_sources,
                    source_paths,
                )
            )
            legacy_source = (
                -1
                if not mapped_sources
                else mapped_sources[0]
                if len(mapped_sources) == 1
                else mapped_sources
            )

            ref = component_ref(
                meta,
                physical_gidx,
                path,
            )
            values = component_gene_by_gidx(
                X,
                physical_gidx,
                path,
            )
            gidx = legacy_X.append_gene(values)
            meta_gidx = legacy_meta.append(
                source=legacy_source,
                op=meta.op[physical_gidx],
                dims=int(component["dims"]),
                params=_legacy_params(meta, physical_gidx),
            )

            if gidx != meta_gidx:
                raise RuntimeError(
                    "Legacy materialization metadata/data diverged."
                )

            ref_to_gidx[ref] = gidx

    return legacy_meta, legacy_X, ref_to_gidx


def _translate_bundled_requirement(
    requirement: Any,
    node_id_map: dict[Any, Any],
) -> Any:
    if isinstance(requirement, STInverseRef):
        return STInverseRef(requirement.inverse_op)

    if isinstance(requirement, STNodeRef):
        return STNodeRef(
            node_id_map[requirement.node_id]
        )

    if isinstance(requirement, STSet):
        return STSet(
            mode=requirement.mode,
            members=[
                _translate_bundled_requirement(
                    member,
                    node_id_map,
                )
                for member in requirement.members
            ],
            label=requirement.label,
            partition=requirement.partition,
        )

    raise TypeError(
        f"Unsupported bundled ST requirement "
        f"{type(requirement).__name__}."
    )


def materialize_legacy_environment(
    GP_meta: ProgramMeta,
    GP_X: ProgramX,
    SP_meta: ProgramMeta,
    SP_X: ProgramX,
    ST: BundledSolutionTree,
) -> tuple[
    ProgramMeta,
    ProgramX,
    ProgramMeta,
    ProgramX,
    SolutionTree,
]:
    """Compile bundled GP/SP/ST state into an ephemeral legacy environment."""
    legacy_GP_meta, legacy_GP_X, gp_map = (
        materialize_legacy_program(
            GP_meta,
            GP_X,
        )
    )
    legacy_SP_meta, legacy_SP_X, sp_map = (
        materialize_legacy_program(
            SP_meta,
            SP_X,
        )
    )

    node_id_map: dict[Any, Any] = {}

    for node_id, node in ST.nodes.items():
        if node.sp_ref is not None:
            node_id_map[node_id] = sp_map[node.sp_ref]
        else:
            node_id_map[node_id] = node_id

    legacy_nodes: dict[Any, STNode] = {}

    for node_id, node in ST.nodes.items():
        translated_id = node_id_map[node_id]
        sp_gidx = (
            None
            if node.sp_ref is None
            else sp_map[node.sp_ref]
        )
        gp_gidx = (
            -1
            if node.gp_ref is None
            else gp_map[node.gp_ref]
        )

        if node.sp_ref is not None:
            physical_sp_gidx = _gidx_from_gene_id(
                SP_meta,
                node.sp_ref.gene_id,
            )
            op_name = SP_meta.op[physical_sp_gidx]
            source = (
                legacy_SP_meta.source[sp_gidx]
                if sp_gidx is not None
                else -1
            )
        else:
            op_name = "bundled_logical"
            source = -1

        legacy_nodes[translated_id] = STNode(
            node_id=translated_id,
            label=node.label,
            sp_gidx=sp_gidx,
            op=op_name,
            dims=node.dims,
            source=source,
            gp_gidx=gp_gidx,
            solution_rule=node.solution_rule,
            solution_params=dict(node.solution_params),
            derivation=(
                None
                if node.derivation is None
                else _translate_bundled_requirement(
                    node.derivation,
                    node_id_map,
                )
            ),
            innate=node.innate,
        )

    legacy_ST = SolutionTree(
        nodes=legacy_nodes,
        roots=tuple(
            node_id_map[root]
            for root in ST.roots
        ),
    )
    legacy_ST.bind_environment(
        legacy_GP_meta,
        legacy_GP_X,
        legacy_SP_X,
    )

    return (
        legacy_GP_meta,
        legacy_GP_X,
        legacy_SP_meta,
        legacy_SP_X,
        legacy_ST,
    )


def _bundled_prune_target_count(
    prune: int | float,
    gp_count: int,
) -> int:
    if isinstance(prune, bool):
        raise TypeError(
            "prune must be an int count or float proportion."
        )

    if isinstance(prune, (int, np.integer)):
        value = int(prune)
        if value < 0:
            raise ValueError("integer prune count must be >= 0.")
        return value

    if isinstance(prune, (float, np.floating)):
        value = float(prune)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "float prune proportion must be in [0, 1]."
            )
        return int(np.floor(value * gp_count))

    raise TypeError(
        "prune must be an int count or float proportion."
    )


def _bundled_solver_gene_ids(
    ST: BundledSolutionTree,
) -> set[int]:
    return {
        int(node.gp_ref.gene_id)
        for node in ST.nodes.values()
        if node.gp_ref is not None
    }


def _bundled_protected_gene_ids(
    GP_meta: ProgramMeta,
    ST: BundledSolutionTree,
) -> set[int]:
    live = {
        GP_meta.stable_id(gidx): gidx
        for gidx in range(len(GP_meta))
    }
    protected = _bundled_solver_gene_ids(ST)
    stack = list(protected)

    while stack:
        gene_id = stack.pop()
        gidx = live.get(gene_id)

        if gidx is None:
            continue

        for parent_gidx in _physical_source_tuple(
            GP_meta.source[gidx]
        ):
            parent_id = GP_meta.stable_id(parent_gidx)

            if parent_id not in protected:
                protected.add(parent_id)
                stack.append(parent_id)

    return protected


def _remap_physical_source_after_delete(
    source: Any,
    removed_gidx: int,
) -> Any:
    indices = _physical_source_tuple(source)

    if removed_gidx in indices:
        raise RuntimeError(
            "Cannot prune a bundled GP gene still used by a survivor."
        )

    if not indices:
        return -1

    remapped = tuple(
        index - 1 if index > removed_gidx else index
        for index in indices
    )
    return (
        remapped[0]
        if len(remapped) == 1
        else remapped
    )


def _delete_bundled_gp_gene(
    GP_meta: ProgramMeta,
    GP_X: ProgramX,
    gidx: int,
) -> None:
    gidx = int(gidx)

    for other_gidx, source in enumerate(GP_meta.source):
        if other_gidx == gidx:
            continue
        if gidx in _physical_source_tuple(source):
            raise RuntimeError(
                f"Bundled GP gene {gidx} is still referenced."
            )

    del GP_meta.source[gidx]
    del GP_meta.op[gidx]
    del GP_meta.dims[gidx]
    del GP_meta.params[gidx]
    del GP_meta.gene_id[gidx]
    del GP_X.genes[gidx]

    GP_meta.source[:] = [
        _remap_physical_source_after_delete(
            source,
            gidx,
        )
        for source in GP_meta.source
    ]


def GP_prune_bundled(
    GP_meta: ProgramMeta,
    GP_X: ProgramX,
    ST: BundledSolutionTree,
    prune: int | float = 5,
    *,
    rng: np.random.Generator | int | None = None,
) -> list[int]:
    """Prune whole bundled GP genes while protecting solved component lineage.

    A physical gene is removable only when:
      - no surviving physical gene uses it as a source;
      - none of its components solve ST;
      - it is outside the recursive ancestor closure of all ST-solving refs;
      - all of its own physical parents have gidx > 0, mirroring the legacy
        guard that preserves initialization-adjacent grammar structure.
    """
    if GP_meta.side != "GP" or GP_X.side != "GP":
        raise ValueError(
            "GP_prune_bundled requires GP-side ProgramMeta/ProgramX."
        )

    target = _bundled_prune_target_count(
        prune,
        len(GP_X),
    )

    if target == 0:
        return []

    rng = _rng(rng)
    original_indices = list(range(len(GP_X)))
    removed: list[int] = []

    while len(removed) < target:
        protected = _bundled_protected_gene_ids(
            GP_meta,
            ST,
        )
        referenced = {
            parent
            for source in GP_meta.source
            for parent in _physical_source_tuple(source)
        }
        eligible: list[int] = []

        for gidx in range(1, len(GP_X)):
            gene_id = GP_meta.stable_id(gidx)

            if gene_id in protected:
                continue
            if gidx in referenced:
                continue

            parents = _physical_source_tuple(
                GP_meta.source[gidx]
            )
            if not parents or any(parent <= 0 for parent in parents):
                continue

            eligible.append(gidx)

        if not eligible:
            break

        operation_count = max(
            1,
            len(BUNDLED_OP_REGISTRY),
        )
        denominator = np.log(float(operation_count + 1))
        scores = []

        for gidx in eligible:
            parents = _physical_source_tuple(
                GP_meta.source[gidx]
            )
            x = float(np.mean(np.asarray(parents, dtype=float)))
            scores.append(
                np.log(x + 1.0) / denominator
            )

        shifted = np.asarray(scores, dtype=float)
        shifted -= np.max(shifted)
        weights = np.exp(shifted)
        probabilities = weights / weights.sum()
        selected = eligible[
            int(rng.choice(len(eligible), p=probabilities))
        ]

        removed.append(
            original_indices[selected]
        )
        _delete_bundled_gp_gene(
            GP_meta,
            GP_X,
            selected,
        )
        del original_indices[selected]

    return removed


def _extract_pair_grid(
    sample: Any,
    field_name: str,
) -> np.ndarray:
    if isinstance(sample, dict):
        value = sample[field_name]
    else:
        value = getattr(sample, field_name)

    array = np.asarray(value)

    if array.ndim != 2:
        raise ValueError(
            f"Training {field_name} must be 2D."
        )

    return array.copy()


def _init_raw_program(
    side: str,
    grids: list[np.ndarray],
    *,
    raw_op: str,
) -> tuple[ProgramMeta, ProgramX]:
    meta = ProgramMeta(side=side)
    X = ProgramX(
        side=side,
        sample_count=len(grids),
    )
    gidx = X.append_gene(grids)
    meta_gidx = meta.append(
        source=-1,
        op=raw_op,
        dims=2,
    )

    if gidx != 0 or meta_gidx != 0:
        raise RuntimeError(
            "Bundled raw program must initialize at gene 0."
        )

    return meta, X


def init_env_bundled(
    grid_set: Iterable[Any],
) -> tuple[
    ProgramMeta,
    ProgramX,
    ProgramMeta,
    ProgramX,
    BundledSolutionTree,
]:
    """Initialize the one-slot bundled representation.

    For ordinary 2D ARC grids, each side begins with exactly:
      0. raw grid
      1. shape bundle:      [[h, w]]
      2. composite bundle:  [[color, mask], [color, mask], ...]

    No legacy multi-output genes are materialized.
    """
    grid_set = list(grid_set)

    if not grid_set:
        raise ValueError(
            "init_env_bundled requires at least one training sample."
        )

    inputs = [
        _extract_pair_grid(sample, "input")
        for sample in grid_set
    ]
    outputs = [
        _extract_pair_grid(sample, "output")
        for sample in grid_set
    ]

    GP_meta, GP_X = _init_raw_program(
        "GP",
        inputs,
        raw_op="raw_input",
    )
    SP_meta, SP_X = _init_raw_program(
        "SP",
        outputs,
        raw_op="raw_output",
    )

    gp_shape = apply_bundled_operation(
        GP_meta,
        GP_X,
        "partition_shape",
        0,
    )
    gp_composite = apply_bundled_operation(
        GP_meta,
        GP_X,
        "partition_composite",
        0,
    )
    sp_shape = apply_bundled_operation(
        SP_meta,
        SP_X,
        "partition_shape",
        0,
    )
    sp_composite = apply_bundled_operation(
        SP_meta,
        SP_X,
        "partition_composite",
        0,
    )

    if (gp_shape, gp_composite) != (1, 2):
        raise RuntimeError(
            "Unexpected bundled GP initialization ordering."
        )
    if (sp_shape, sp_composite) != (1, 2):
        raise RuntimeError(
            "Unexpected bundled SP initialization ordering."
        )

    ST = BundledSolutionTree.from_raw_output(
        SP_meta,
        SP_X,
    )
    ST.initialize_output_partition(
        SP_meta,
        SP_X,
        shape_gidx=sp_shape,
        composite_gidx=sp_composite,
    )
    ST.bind_environment(
        GP_meta,
        GP_X,
        SP_meta,
        SP_X,
    )

    return GP_meta, GP_X, SP_meta, SP_X, ST
