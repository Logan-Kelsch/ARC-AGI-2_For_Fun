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


@dataclass
class STNode:
    """One SP gene represented in the solution tree."""

    sp_gidx: int
    op: str
    dims: int
    parents: tuple[int, ...]
    children: list[int] = field(default_factory=list)
    gp_gidx: int = -1

    @property
    def solved(self) -> bool:
        return self.gp_gidx >= 0


@dataclass
class SolutionTree:
    """Structural view of SP with GP solution indices attached.

    Each SP gene has one ST node. gp_gidx == -1 means the SP node is not yet
    explained by an exact GP gene across all samples.

    SP is currently expected to be tree-like because full-partition operations
    decompose prior SP genes. The representation also retains multiple parents
    so future full-partition operations may be multi-source without losing
    dependency information.
    """

    nodes: dict[int, STNode] = field(default_factory=dict)
    roots: tuple[int, ...] = ()

    @classmethod
    def from_sp_meta(cls, SP_meta: ProgramMeta) -> "SolutionTree":
        if SP_meta.side != "SP":
            raise ValueError("SolutionTree must be built from SP_meta.")

        nodes: dict[int, STNode] = {}

        for gidx in range(len(SP_meta)):
            parents = source_indices(SP_meta.source[gidx])
            nodes[gidx] = STNode(
                sp_gidx=gidx,
                op=SP_meta.op[gidx],
                dims=SP_meta.dims[gidx],
                parents=parents,
            )

        for gidx, node in nodes.items():
            for parent in node.parents:
                if parent not in nodes:
                    raise IndexError(
                        f"SP gene {gidx} references missing source gene {parent}."
                    )
                nodes[parent].children.append(gidx)

        roots = tuple(
            gidx
            for gidx, node in nodes.items()
            if not node.parents
        )

        return cls(nodes=nodes, roots=roots)

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, sp_gidx: int) -> STNode:
        return self.nodes[sp_gidx]

    def mark_solution(self, sp_gidx: int, gp_gidx: int) -> None:
        gp_gidx = int(gp_gidx)
        if gp_gidx < 0:
            raise ValueError("gp_gidx must be non-negative.")
        self.nodes[sp_gidx].gp_gidx = gp_gidx

    def clear_solution(self, sp_gidx: int) -> None:
        self.nodes[sp_gidx].gp_gidx = -1

    def sync(self, SP_meta: ProgramMeta) -> None:
        """Rebuild from SP structure while preserving known GP matches."""
        prior = {
            gidx: node.gp_gidx
            for gidx, node in self.nodes.items()
            if node.gp_gidx >= 0
        }

        rebuilt = SolutionTree.from_sp_meta(SP_meta)

        for gidx, gp_gidx in prior.items():
            if gidx in rebuilt.nodes:
                rebuilt.nodes[gidx].gp_gidx = gp_gidx

        self.nodes = rebuilt.nodes
        self.roots = rebuilt.roots

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "sp_gidx": gidx,
                "op": node.op,
                "dims": node.dims,
                "parents": node.parents,
                "children": tuple(node.children),
                "gp_gidx": node.gp_gidx,
                "solved": node.solved,
            }
            for gidx, node in sorted(self.nodes.items())
        ]


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

    Then both sides receive the default full-partition operations:
      gene 1 = partition_shape(gene 0)
      genes 2.. = partition_composite(gene 0)

    partition_composite creates two genes per distinct color observed across
    the complete sample set: one scalar int64 color ID and one 2D boolean
    presence mask.

    SP operation enforcement occurs inside notebooks.ops.ops: only operations
    registered as full_partition are legal on the SP side.
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

    partition_shape(SP_meta, SP_X, 0)
    partition_composite(SP_meta, SP_X, 0)

    ST = SolutionTree.from_sp_meta(SP_meta)

    return GP_meta, GP_X, SP_meta, SP_X, ST
