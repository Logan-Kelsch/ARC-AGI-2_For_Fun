from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


Side = str
SourceRef = int | tuple[int, ...]


def _copy_value(value: Any) -> Any:
    """Copy heterogeneous gene values without changing their representation."""
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            copied = np.empty(value.shape, dtype=object)
            for index in np.ndindex(value.shape):
                copied[index] = _copy_value(value[index])
            return copied
        return value.copy()

    if isinstance(value, list):
        return [_copy_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)

    return value


def _pack_gene(values: Iterable[Any]) -> np.ndarray:
    """Pack one gene across samples into a 1D object array."""
    values = list(values)
    packed = np.empty(len(values), dtype=object)

    for sample_idx, value in enumerate(values):
        packed[sample_idx] = _copy_value(value)

    return packed


def _value_ndim(value: Any) -> int:
    """Return the dimensionality recorded in meta for one instantiated value."""
    if isinstance(value, np.ndarray):
        return int(value.ndim)

    return int(np.asarray(value).ndim)


def _gene_ndim(values: np.ndarray) -> int:
    """Require one structural dimensionality for a gene across all samples."""
    if len(values) == 0:
        raise ValueError("A gene must contain at least one sample.")

    dims = {_value_ndim(value) for value in values}

    if len(dims) != 1:
        raise ValueError(
            "All samples in one gene must have the same dimensionality; "
            f"got {sorted(dims)}."
        )

    return dims.pop()


def _normalize_source(source: Any) -> SourceRef:
    """Normalize a source reference while preserving single vs multi-source."""
    if isinstance(source, np.generic):
        source = source.item()

    if isinstance(source, np.ndarray):
        if source.ndim == 0:
            return _normalize_source(source.item())
        source = source.tolist()

    if isinstance(source, (list, tuple)):
        flattened: list[int] = []

        def collect(value: Any) -> None:
            if isinstance(value, np.generic):
                value = value.item()

            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    collect(value.item())
                else:
                    collect(value.tolist())
                return

            if isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
                return

            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    "Source references must contain integer gene indices."
                )

            flattened.append(int(value))

        collect(source)

        if not flattened:
            raise ValueError("A multi-source reference may not be empty.")

        if len(flattened) == 1:
            return flattened[0]

        return tuple(flattened)

    if isinstance(source, bool) or not isinstance(source, int):
        raise TypeError("Source reference must be an integer or index collection.")

    return int(source)


def source_indices(source: SourceRef) -> tuple[int, ...]:
    """Return non-negative source indices from one normalized source entry."""
    source = _normalize_source(source)

    if isinstance(source, int):
        return () if source == -1 else (source,)

    return tuple(index for index in source if index != -1)


@dataclass
class ProgramMeta:
    """Gene-parallel metadata for either GP or SP."""

    side: Side
    source: list[SourceRef] = field(default_factory=list)
    op: list[str] = field(default_factory=list)
    dims: list[int] = field(default_factory=list)
    params: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.side not in {"GP", "SP"}:
            raise ValueError("ProgramMeta.side must be 'GP' or 'SP'.")

    def __len__(self) -> int:
        return len(self.op)

    def append(
        self,
        *,
        source: Any,
        op: str,
        dims: int,
        params: dict[str, Any] | None = None,
    ) -> int:
        """Append metadata and return the newly allocated gene index."""
        dims = int(dims)
        if dims < 0:
            raise ValueError("dims must be non-negative.")

        source = _normalize_source(source)

        self.source.append(source)
        self.op.append(str(op))
        self.dims.append(dims)
        self.params.append(dict(params or {}))

        return len(self.op) - 1

    def row(self, gidx: int) -> dict[str, Any]:
        return {
            "gidx": int(gidx),
            "source": self.source[gidx],
            "op": self.op[gidx],
            "dims": self.dims[gidx],
            "params": dict(self.params[gidx]),
        }

    def rows(self) -> list[dict[str, Any]]:
        return [self.row(gidx) for gidx in range(len(self))]


@dataclass
class ProgramX:
    """Gene-major instantiated program data.

    The primary index is gene index and the secondary index is sample index:

        X[gidx][sample_idx]
        X[gidx, sample_idx]

    Each gene is retained as a 1D object array so different samples may contain
    arrays with different spatial shapes while still sharing one gene index.
    """

    side: Side
    sample_count: int
    genes: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.side not in {"GP", "SP"}:
            raise ValueError("ProgramX.side must be 'GP' or 'SP'.")
        if self.sample_count < 1:
            raise ValueError("ProgramX requires at least one sample.")

    def __len__(self) -> int:
        return len(self.genes)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.genes), self.sample_count)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError("ProgramX tuple indexing is X[gidx, sample_idx].")
            gidx, sample_idx = index
            return self.genes[gidx][sample_idx]

        return self.genes[index]

    def append_gene(self, values: Iterable[Any]) -> int:
        packed = _pack_gene(values)

        if len(packed) != self.sample_count:
            raise ValueError(
                "Gene must contain one value per sample: "
                f"{len(packed)} != {self.sample_count}."
            )

        self.genes.append(packed)
        return len(self.genes) - 1

    def gene(self, gidx: int) -> np.ndarray:
        return self.genes[gidx]

    def sample(self, sample_idx: int) -> list[Any]:
        return [gene[sample_idx] for gene in self.genes]

    def as_object_array(self) -> np.ndarray:
        """Materialize a 2D object array with axes [gene, sample]."""
        result = np.empty(self.shape, dtype=object)

        for gidx, gene in enumerate(self.genes):
            for sample_idx, value in enumerate(gene):
                result[gidx, sample_idx] = value

        return result


STNodeId = int | str


@dataclass(frozen=True)
class STNodeRef:
    """Reference to another solution-tree node."""

    node_id: STNodeId


@dataclass(frozen=True)
class STInverseRef:
    """Known inverse transformation required by one proof branch."""

    inverse_op: str

    @property
    def solved(self) -> bool:
        # The inverse function is part of the grammar, so knowing/possessing it
        # is not itself something GP must discover.
        return True


@dataclass
class STSet:
    """Boolean requirement set used to resolve one ST node."""

    mode: str
    members: list[Any] = field(default_factory=list)
    label: str = ""
    partition: str | None = None

    def __post_init__(self) -> None:
        self.mode = str(self.mode).upper()
        if self.mode not in {"AND", "OR"}:
            raise ValueError("STSet.mode must be 'AND' or 'OR'.")


@dataclass
class STNode:
    """One semantic target represented in the boolean solution tree."""

    node_id: STNodeId
    label: str
    sp_gidx: int | None = None
    op: str = ""
    dims: int | None = None
    source: SourceRef | None = None
    gp_gidx: int = -1
    derivation: STSet | None = None
    innate: bool = False

    @property
    def directly_solved(self) -> bool:
        return self.innate or self.gp_gidx >= 0


@dataclass
class SolutionTree:
    """Boolean proof tree describing how SP structure can resolve the output.

    Every semantic node may be solved directly by a matching GP gene, or by one
    of its alternative derivation branches.

    A derivation branch is expressed with STSet:

        AND(...)  every member is required
        OR(...)   any member is sufficient

    Reversible SP transformations add an alternative AND branch containing the
    known inverse operation and the transformed child node(s). The parent's
    direct GP solution remains an implicit OR alternative.

    Nodes created only to organize solution structure may have sp_gidx=None.
    """

    nodes: dict[STNodeId, STNode] = field(default_factory=dict)
    roots: tuple[STNodeId, ...] = (0,)

    @classmethod
    def from_sp_meta(cls, SP_meta: ProgramMeta) -> "SolutionTree":
        if SP_meta.side != "SP":
            raise ValueError("SolutionTree must be built from SP_meta.")

        tree = cls(nodes={}, roots=(0,) if len(SP_meta) else ())

        for gidx in range(len(SP_meta)):
            tree.nodes[gidx] = STNode(
                node_id=gidx,
                label=(
                    "root"
                    if gidx == 0
                    else f"sp_{gidx}:{SP_meta.op[gidx]}"
                ),
                sp_gidx=gidx,
                op=SP_meta.op[gidx],
                dims=SP_meta.dims[gidx],
                source=SP_meta.source[gidx],
            )

        return tree

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, node_id: STNodeId) -> STNode:
        return self.nodes[node_id]

    def _eval_requirement(
        self,
        requirement: Any,
        *,
        visiting: set[STNodeId] | None = None,
    ) -> bool:
        if isinstance(requirement, STNodeRef):
            return self.is_solved(
                requirement.node_id,
                visiting=visiting,
            )

        if isinstance(requirement, STInverseRef):
            return requirement.solved

        if isinstance(requirement, STSet):
            values = [
                self._eval_requirement(member, visiting=visiting)
                for member in requirement.members
            ]

            if requirement.mode == "AND":
                return all(values)

            return any(values)

        raise TypeError(
            "ST requirements must be STNodeRef, STInverseRef, or STSet."
        )

    def is_solved(
        self,
        node_id: STNodeId,
        *,
        visiting: set[STNodeId] | None = None,
    ) -> bool:
        node = self.nodes[node_id]

        if node.directly_solved:
            return True

        if node.derivation is None:
            return False

        visiting = set() if visiting is None else set(visiting)

        if node_id in visiting:
            raise ValueError(
                f"Cycle detected while evaluating ST node {node_id!r}."
            )

        visiting.add(node_id)

        return self._eval_requirement(
            node.derivation,
            visiting=visiting,
        )

    def needs_gp_solution(self, node_id: STNodeId) -> bool:
        """Whether this is an unresolved leaf with concrete SP data."""
        node = self.nodes[node_id]

        return (
            node.sp_gidx is not None
            and not self.is_solved(node_id)
            and node.derivation is None
            and not node.innate
        )

    @property
    def solved(self) -> bool:
        return bool(self.roots) and all(
            self.is_solved(root)
            for root in self.roots
        )

    def mark_solution(self, sp_gidx: int, gp_gidx: int) -> None:
        gp_gidx = int(gp_gidx)
        if gp_gidx < 0:
            raise ValueError("gp_gidx must be non-negative.")
        if sp_gidx not in self.nodes:
            raise KeyError(f"No ST node for SP gene {sp_gidx}.")
        self.nodes[sp_gidx].gp_gidx = gp_gidx

    def clear_solution(self, sp_gidx: int) -> None:
        if sp_gidx not in self.nodes:
            raise KeyError(f"No ST node for SP gene {sp_gidx}.")
        self.nodes[sp_gidx].gp_gidx = -1

    def sync(self, SP_meta: ProgramMeta) -> None:
        """Add new SP nodes without destroying boolean proof structure."""
        if SP_meta.side != "SP":
            raise ValueError("SolutionTree.sync requires SP_meta.")

        for gidx in range(len(SP_meta)):
            if gidx in self.nodes:
                node = self.nodes[gidx]
                node.op = SP_meta.op[gidx]
                node.dims = SP_meta.dims[gidx]
                node.source = SP_meta.source[gidx]
                continue

            self.nodes[gidx] = STNode(
                node_id=gidx,
                label=f"sp_{gidx}:{SP_meta.op[gidx]}",
                sp_gidx=gidx,
                op=SP_meta.op[gidx],
                dims=SP_meta.dims[gidx],
                source=SP_meta.source[gidx],
            )

    def _append_alternative(
        self,
        node_id: STNodeId,
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
        *,
        shape_gidxs: Iterable[int],
        composite_gidxs: Iterable[int],
    ) -> None:
        """Create the initial root = shape AND composite proof structure."""
        shape_gidxs = tuple(int(gidx) for gidx in shape_gidxs)
        composite_gidxs = tuple(int(gidx) for gidx in composite_gidxs)

        if 0 not in self.nodes:
            raise KeyError("ST root SP gene 0 is missing.")

        if len(shape_gidxs) != 2:
            raise ValueError(
                "Initial 2D shape partition requires exactly h and w genes."
            )

        for gidx in shape_gidxs:
            if gidx not in self.nodes:
                raise KeyError(f"Missing shape SP gene {gidx}.")

        for gidx in composite_gidxs:
            if gidx not in self.nodes:
                raise KeyError(f"Missing composite SP gene {gidx}.")

        h_gidx, w_gidx = shape_gidxs
        shape_id: STNodeId = "shape"
        composite_id: STNodeId = "composite"

        shape_branch = STSet(
            mode="AND",
            members=[
                STNodeRef(h_gidx),
                STNodeRef(w_gidx),
            ],
            label="h AND w",
            partition="and",
        )

        self.nodes[shape_id] = STNode(
            node_id=shape_id,
            label="shape",
            sp_gidx=None,
            op="logical_shape",
            dims=1,
            derivation=STSet(
                mode="OR",
                members=[shape_branch],
                label="shape alternatives",
            ),
        )

        self.nodes[h_gidx].label = "h"
        self.nodes[w_gidx].label = "w"

        composite_branch = STSet(
            mode="AND",
            members=[
                STInverseRef("inv_partition_composite"),
                *[
                    STNodeRef(gidx)
                    for gidx in composite_gidxs
                ],
            ],
            label="assemble composite",
            partition="and",
        )

        self.nodes[composite_id] = STNode(
            node_id=composite_id,
            label="composite",
            sp_gidx=None,
            op="logical_composite",
            dims=2,
            derivation=STSet(
                mode="OR",
                members=[composite_branch],
                label="composite alternatives",
            ),
        )

        root_branch = STSet(
            mode="AND",
            members=[
                STInverseRef("inv_partition_shape"),
                STNodeRef(shape_id),
                STNodeRef(composite_id),
            ],
            label="shape AND composite",
            partition="and",
        )

        self.nodes[0].derivation = STSet(
            mode="OR",
            members=[root_branch],
            label="root alternatives",
        )


    def register_generation(
        self,
        SP_meta: ProgramMeta,
        *,
        source_gidx: int,
        generated_gidxs: Iterable[int],
        partition: str,
        inverse_op: str,
        op_name: str,
    ) -> None:
        """Attach one reversible SP transformation as a proof alternative."""
        partition = str(partition).lower()

        if partition not in {"and", "or"}:
            raise ValueError(
                "Only AND/OR partition operations may extend ST."
            )

        self.sync(SP_meta)

        generated_gidxs = tuple(
            int(gidx) for gidx in generated_gidxs
        )

        if source_gidx not in self.nodes:
            raise KeyError(f"Missing source ST node {source_gidx}.")

        for gidx in generated_gidxs:
            if gidx not in self.nodes:
                raise KeyError(f"Missing generated ST node {gidx}.")

        branch = STSet(
            mode="AND",
            members=[
                STInverseRef(inverse_op),
                *[
                    STNodeRef(gidx)
                    for gidx in generated_gidxs
                ],
            ],
            label=f"{inverse_op} AND transformed data",
            partition=partition,
        )

        self._append_alternative(source_gidx, branch)

    def unresolved_frontier_nodes(
        self,
        node_id: STNodeId | None = None,
    ) -> tuple[STNodeId, ...]:
        """Return unresolved concrete SP nodes reachable from the solution root.

        Unlike unresolved_leaf_nodes(), this keeps an unresolved concrete parent
        in the frontier even when it also has derived alternatives. That matters
        for OR semantics: a source may still be solved directly by GP while its
        transformed children offer alternate solution paths.

        Solved nodes prune their entire subtree because no further proof below
        them is required.
        """
        if node_id is None:
            if not self.roots:
                return ()
            node_id = self.roots[0]

        result: list[STNodeId] = []
        seen: set[STNodeId] = set()

        def collect_requirement(requirement: Any) -> None:
            if isinstance(requirement, STInverseRef):
                return

            if isinstance(requirement, STSet):
                for member in requirement.members:
                    collect_requirement(member)
                return

            if isinstance(requirement, STNodeRef):
                collect_node(requirement.node_id)
                return

            raise TypeError("Unknown ST requirement type.")

        def collect_node(current_id: STNodeId) -> None:
            if current_id in seen:
                return

            seen.add(current_id)
            node = self.nodes[current_id]

            if self.is_solved(current_id):
                return

            if node.sp_gidx is not None:
                result.append(current_id)

            if node.derivation is not None:
                collect_requirement(node.derivation)

        collect_node(node_id)

        return tuple(result)

    def unresolved_leaf_nodes(
        self,
        node_id: STNodeId | None = None,
    ) -> tuple[STNodeId, ...]:
        """Return concrete unresolved leaves reachable from one solution node."""
        if node_id is None:
            if not self.roots:
                return ()
            node_id = self.roots[0]

        result: list[STNodeId] = []
        seen: set[STNodeId] = set()

        def collect_requirement(requirement: Any) -> None:
            if isinstance(requirement, STInverseRef):
                return

            if isinstance(requirement, STSet):
                for member in requirement.members:
                    collect_requirement(member)
                return

            if isinstance(requirement, STNodeRef):
                collect_node(requirement.node_id)
                return

            raise TypeError("Unknown ST requirement type.")

        def collect_node(current_id: STNodeId) -> None:
            if current_id in seen:
                return

            seen.add(current_id)
            node = self.nodes[current_id]

            if self.is_solved(current_id):
                return

            if node.derivation is None:
                if node.sp_gidx is not None:
                    result.append(current_id)
                return

            collect_requirement(node.derivation)

        collect_node(node_id)

        return tuple(result)

    def _format_requirement(self, requirement: Any) -> str:
        if isinstance(requirement, STNodeRef):
            return str(requirement.node_id)

        if isinstance(requirement, STInverseRef):
            return f"{requirement.inverse_op}✓"

        if isinstance(requirement, STSet):
            body = f" {requirement.mode} ".join(
                self._format_requirement(member)
                for member in requirement.members
            )
            return f"({body})"

        return repr(requirement)

    def rows(self) -> list[dict[str, Any]]:
        rows = []

        for node_id, node in self.nodes.items():
            rows.append(
                {
                    "node_id": node_id,
                    "label": node.label,
                    "sp_gidx": node.sp_gidx,
                    "op": node.op,
                    "dims": node.dims,
                    "source": node.source,
                    "gp_gidx": node.gp_gidx,
                    "directly_solved": node.directly_solved,
                    "solved": self.is_solved(node_id),
                    "needs_gp_solution": self.needs_gp_solution(node_id),
                    "derivation": (
                        None
                        if node.derivation is None
                        else self._format_requirement(node.derivation)
                    ),
                }
            )

        return rows


def _normalize_pool_dim_bound(
    name: str,
    value: int | None,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(
            f"{name} must be a non-negative integer or None."
        )

    value = int(value)

    if value < 0:
        raise ValueError(f"{name} must be >= 0.")

    return value


def _normalize_pool_dtype(dtype: Any | None) -> np.dtype | None:
    if dtype is None:
        return None

    try:
        return np.dtype(dtype)
    except TypeError as exc:
        raise TypeError(
            "dtype must be convertible to a NumPy dtype or None."
        ) from exc


def _atomic_value_dtypes(value: Any) -> set[np.dtype]:
    """Recursively collect atomic NumPy dtypes from one instantiated value."""
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return {np.dtype(value.dtype)}

        dtypes: set[np.dtype] = set()
        for item in value.flat:
            dtypes.update(_atomic_value_dtypes(item))
        return dtypes

    if isinstance(value, (list, tuple)):
        dtypes: set[np.dtype] = set()
        for item in value:
            dtypes.update(_atomic_value_dtypes(item))
        return dtypes

    return {np.dtype(np.asarray(value).dtype)}


def _gene_matches_dtype(
    gene: np.ndarray,
    dtype: np.dtype | None,
) -> bool:
    if dtype is None:
        return True

    observed: set[np.dtype] = set()

    for value in gene:
        observed.update(_atomic_value_dtypes(value))

    return bool(observed) and observed == {dtype}


def _validate_pool_filters(
    *,
    min_dim: int | None,
    max_dim: int | None,
    dtype: Any | None,
) -> tuple[int | None, int | None, np.dtype | None]:
    min_dim = _normalize_pool_dim_bound("min_dim", min_dim)
    max_dim = _normalize_pool_dim_bound("max_dim", max_dim)
    dtype = _normalize_pool_dtype(dtype)

    if (
        min_dim is not None
        and max_dim is not None
        and min_dim > max_dim
    ):
        raise ValueError(
            "min_dim may not be greater than max_dim."
        )

    return min_dim, max_dim, dtype


def _pack_selected_genes(
    genes: Iterable[np.ndarray],
) -> np.ndarray:
    genes = list(genes)
    result = np.empty(len(genes), dtype=object)

    for index, gene in enumerate(genes):
        result[index] = _copy_value(gene)

    return result


def get_ST_unsovled_frontier(
    ST: SolutionTree,
    SP_X: ProgramX,
    *,
    min_dim: int | None = None,
    max_dim: int | None = None,
    dtype: Any | None = None,
) -> np.ndarray:
    """Return instantiated SP data for unresolved Boolean-ST frontier nodes.

    The result is a 1D object ndarray with one entry per unresolved concrete ST
    node reachable from the root. Each entry is that SP gene's complete
    instantiated data across samples, i.e. the same 1D object array stored at:

        SP_X[sp_gidx]

    Filters:
      min_dim / max_dim:
        Inclusive dimensionality bounds.

      dtype:
        Exact atomic NumPy dtype required across the complete gene. Examples:

            dtype=bool
            dtype=np.bool_
            dtype=np.int64
            dtype="float64"

        dtype=None applies no dtype restriction.

    Examples:

        min_dim=0, max_dim=0
            scalar frontier genes of any dtype

        min_dim=2, max_dim=2, dtype=bool
            only 2D boolean frontier genes

        dtype=np.int64
            int64 frontier genes of any dimensionality

    Logical helper nodes such as "shape" and "composite" have no SP gene and
    are therefore never returned.
    """
    if SP_X.side != "SP":
        raise ValueError(
            "get_ST_unsovled_frontier requires SP_X.side == 'SP'."
        )

    min_dim, max_dim, dtype = _validate_pool_filters(
        min_dim=min_dim,
        max_dim=max_dim,
        dtype=dtype,
    )

    selected: list[np.ndarray] = []

    for node_id in ST.unresolved_frontier_nodes():
        node = ST[node_id]

        if node.sp_gidx is None:
            continue

        if node.sp_gidx < 0 or node.sp_gidx >= len(SP_X):
            raise IndexError(
                f"ST node {node_id!r} references SP gene {node.sp_gidx}, "
                f"outside range [0, {len(SP_X) - 1}]."
            )

        dims = node.dims

        if dims is None:
            continue
        if min_dim is not None and dims < min_dim:
            continue
        if max_dim is not None and dims > max_dim:
            continue

        gene = SP_X[node.sp_gidx]

        if not _gene_matches_dtype(gene, dtype):
            continue

        selected.append(gene)

    return _pack_selected_genes(selected)


# Correctly spelled alias for convenience; the requested public name above is
# retained exactly.
get_ST_unsolved_frontier = get_ST_unsovled_frontier


def get_GP_pool(
    GP_meta: ProgramMeta,
    GP_X: ProgramX,
    *,
    min_dim: int | None = None,
    max_dim: int | None = None,
    dtype: Any | None = None,
) -> np.ndarray:
    """Return instantiated GP genes matching dimensionality/dtype filters.

    The return value is a 1D object ndarray. Each entry is one complete GP gene
    across all training samples, equivalent to:

        GP_X[gidx]

    Filters are identical to get_ST_unsovled_frontier:

        min_dim / max_dim
            Inclusive dimensionality bounds.

        dtype
            Exact atomic NumPy dtype across the complete gene.

    Examples:

        get_GP_pool(
            GP_meta,
            GP_X,
            min_dim=0,
            max_dim=0,
        )
            -> all scalar GP genes, any dtype

        get_GP_pool(
            GP_meta,
            GP_X,
            min_dim=2,
            max_dim=2,
            dtype=bool,
        )
            -> only 2D boolean GP genes

        get_GP_pool(
            GP_meta,
            GP_X,
        )
            -> the complete GP pool
    """
    if GP_meta.side != "GP":
        raise ValueError(
            "get_GP_pool requires GP_meta.side == 'GP'."
        )
    if GP_X.side != "GP":
        raise ValueError(
            "get_GP_pool requires GP_X.side == 'GP'."
        )
    if len(GP_meta) != len(GP_X):
        raise ValueError(
            "GP_meta and GP_X must contain the same number of genes."
        )

    min_dim, max_dim, dtype = _validate_pool_filters(
        min_dim=min_dim,
        max_dim=max_dim,
        dtype=dtype,
    )

    selected: list[np.ndarray] = []

    for gidx in range(len(GP_X)):
        dims = GP_meta.dims[gidx]

        if min_dim is not None and dims < min_dim:
            continue
        if max_dim is not None and dims > max_dim:
            continue

        gene = GP_X[gidx]

        if not _gene_matches_dtype(gene, dtype):
            continue

        selected.append(gene)

    return _pack_selected_genes(selected)



def _extract_pair_grid(sample: Any, field_name: str) -> np.ndarray:
    if isinstance(sample, dict):
        if field_name not in sample:
            raise KeyError(f"Training sample has no {field_name!r} field.")
        value = sample[field_name]
    else:
        try:
            value = getattr(sample, field_name)
        except AttributeError as exc:
            raise AttributeError(
                f"Training sample has no {field_name!r} attribute."
            ) from exc

    array = np.asarray(value)

    if array.ndim != 2:
        raise ValueError(
            f"Training {field_name} must be a 2D grid, got {array.shape}."
        )

    return array.copy()


def _init_raw_program(
    side: Side,
    grids: list[np.ndarray],
    *,
    raw_op: str,
) -> tuple[ProgramMeta, ProgramX]:
    meta = ProgramMeta(side=side)
    X = ProgramX(side=side, sample_count=len(grids))

    gidx = X.append_gene(grids)
    if gidx != 0:
        raise RuntimeError("Raw gene must initialize at gene index 0.")

    dims = _gene_ndim(X[0])
    meta_gidx = meta.append(source=-1, op=raw_op, dims=dims)

    if meta_gidx != gidx:
        raise RuntimeError("Program metadata and X gene indices diverged.")

    return meta, X


def init_env(
    grid_set: Iterable[Any],
) -> tuple[ProgramMeta, ProgramX, ProgramMeta, ProgramX, SolutionTree]:
    """Initialize GP_meta, GP_X, SP_meta, SP_X, and ST.

    Initialization:
      GP gene 0 = input matrix
      SP gene 0 = output matrix

    Then both sides receive the default AND-partition operations:
      genes 1-2 = partition_shape(gene 0): h, w
      genes 3.. = partition_composite(gene 0)

    partition_composite creates two genes per distinct color observed across
    the complete sample set: one scalar int64 color ID and one 2D boolean
    presence mask.

    ST begins as a boolean proof:
        root = shape AND composite

    Later SP transformations may add reversible OR/AND alternatives. NULL
    partition operations are GP-only.
    """
    grid_set = list(grid_set)

    if not grid_set:
        raise ValueError("init_env requires at least one training sample.")

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

    # Local import avoids a module cycle: operations work on ProgramMeta/X,
    # while init_env is responsible for choosing the default operation sequence.
    from .ops import partition_composite, partition_shape

    partition_shape(GP_meta, GP_X, 0)
    partition_composite(GP_meta, GP_X, 0)

    # Build ST from the raw output first, then add the initial solution
    # decomposition explicitly as:
    #
    #     root <- shape AND composite
    #
    # Composite itself is reconstructed from every per-color ID/presence gene.
    ST = SolutionTree.from_sp_meta(SP_meta)

    shape_gidxs = partition_shape(SP_meta, SP_X, 0)
    composite_gidxs = partition_composite(SP_meta, SP_X, 0)

    ST.sync(SP_meta)
    ST.initialize_output_partition(
        shape_gidxs=shape_gidxs,
        composite_gidxs=composite_gidxs,
    )

    return GP_meta, GP_X, SP_meta, SP_X, ST
