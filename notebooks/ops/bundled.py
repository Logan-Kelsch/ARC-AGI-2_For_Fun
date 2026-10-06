from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Any, Iterable

import numpy as np

from .environment import (
    ProgramMeta,
    ProgramX,
    STInverseRef,
    STNodeRef,
    STSet,
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

        if info.name == "partition_shape":
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
) -> Iterable[tuple[GeneComponentRef, ...]]:
    ref_lists = [
        component_refs_for_gene(meta, X, gidx)
        for gidx in source_gidxs
    ]

    return itertools.product(*ref_lists)


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
    info = (
        BUNDLED_INV_OP_REGISTRY[str(inverse)]
        if isinstance(inverse, str)
        else inverse
    )
    args: list[Any] = []

    for value in values:
        args.extend(_flatten_bundle_argument(value))

    return info.legacy.func(*args)


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


def _source_gene_candidates(
    info: BundledOperationInfo,
    gene_count: int,
) -> list[Any]:
    if info.source_count == 1:
        return list(range(gene_count))

    indices = range(gene_count)

    if info.ordered_sources:
        return list(
            itertools.permutations(
                indices,
                info.source_count,
            )
        )

    return list(
        itertools.combinations(
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
