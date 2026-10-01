from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
import inspect
import itertools
from typing import Any, Callable, Iterable

import numpy as np

from .inv_ops import INV_OP_REGISTRY
from .environment import (
    ProgramMeta,
    ProgramX,
    SolutionTree,
    _gene_ndim,
    _normalize_source,
)


ParameterSampler = Callable[[np.random.Generator], dict[str, Any]]
GenerationValidator = Callable[
    [ProgramMeta, ProgramX, Any, dict[str, Any]],
    bool,
]
OutputCountEstimator = Callable[
    [ProgramMeta, ProgramX, Any, dict[str, Any]],
    int,
]


@dataclass
class OperationInfo:
    """Registered metadata controlling execution and random generation.

    min_dims_exclusive:
        Source gene dims must be strictly greater than this value.

    atomic_dtypes:
        If not None, every atomic value contained by the source gene across all
        samples must have one of these NumPy dtypes.

    parameter_sampler:
        Optional random parameter generator. Future parameterized operations can
        use this to supply values such as integer offsets, axis choices, etc.

    validator:
        Optional extra operation-specific predicate for constraints that cannot
        be expressed through dims/dtype metadata alone.
    """

    name: str
    partition: str
    inverse_op: str | None
    output_count: int | None
    output_count_estimator: OutputCountEstimator | None = None
    source_count: int = 1
    ordered_sources: bool = True
    min_dims_exclusive: int | None = None
    allowed_dims: tuple[int, ...] | None = None
    atomic_dtypes: tuple[np.dtype, ...] | None = None
    parameter_sampler: ParameterSampler | None = None
    validator: GenerationValidator | None = None
    func: Callable | None = field(default=None, repr=False, compare=False)


OP_REGISTRY: dict[str, OperationInfo] = {}


def _normalize_dtypes(
    dtypes: Iterable[Any] | None,
) -> tuple[np.dtype, ...] | None:
    if dtypes is None:
        return None
    return tuple(np.dtype(dtype) for dtype in dtypes)


def operation(
    *,
    partition: str = "null",
    inverse_op: str | None = None,
    output_count: int | None,
    output_count_estimator: OutputCountEstimator | None = None,
    source_count: int = 1,
    ordered_sources: bool = True,
    min_dims_exclusive: int | None = None,
    allowed_dims: Iterable[int] | None = None,
    atomic_dtypes: Iterable[Any] | None = None,
    parameter_sampler: ParameterSampler | None = None,
    validator: GenerationValidator | None = None,
):
    """Register a transformation operation.

    GP may use any registered operation.

    SP may use reversible AND/OR partition operations. NULL operations are
    excluded from SP generation.

    The decorator also validates every direct operation application, so manual
    notebook calls and random generation obey the same rules.
    """
    partition = str(partition).lower()
    if partition not in {"and", "or", "null"}:
        raise ValueError("partition must be 'and', 'or', or 'null'.")
    if partition in {"and", "or"} and not inverse_op:
        raise ValueError(
            "AND/OR partition operations require an inverse_op name."
        )

    if output_count is None and output_count_estimator is None:
        raise ValueError(
            "Dynamic-output operations require output_count_estimator."
        )
    if output_count is not None and output_count < 1:
        raise ValueError("output_count must be >= 1.")
    if source_count < 1:
        raise ValueError("source_count must be >= 1.")

    normalized_allowed_dims = (
        None
        if allowed_dims is None
        else tuple(sorted({int(dim) for dim in allowed_dims}))
    )
    if normalized_allowed_dims is not None and any(
        dim < 0 for dim in normalized_allowed_dims
    ):
        raise ValueError("allowed_dims must contain non-negative dimensions.")

    def decorator(func: Callable):
        info = OperationInfo(
            name=func.__name__,
            partition=partition,
            inverse_op=inverse_op,
            output_count=(
                None if output_count is None else int(output_count)
            ),
            output_count_estimator=output_count_estimator,
            source_count=int(source_count),
            ordered_sources=bool(ordered_sources),
            min_dims_exclusive=min_dims_exclusive,
            allowed_dims=normalized_allowed_dims,
            atomic_dtypes=_normalize_dtypes(atomic_dtypes),
            parameter_sampler=parameter_sampler,
            validator=validator,
        )
        signature = inspect.signature(func)

        @wraps(func)
        def wrapped(meta: ProgramMeta, X: ProgramX, *args, **kwargs):
            _validate_program_pair(meta, X)

            bound = signature.bind(meta, X, *args, **kwargs)
            bound.apply_defaults()

            source_idx = _normalize_source(bound.arguments["source_idx"])
            params = {
                name: value
                for name, value in bound.arguments.items()
                if name not in {"meta", "X", "source_idx"}
            }

            if info.inverse_op and info.inverse_op not in INV_OP_REGISTRY:
                raise RuntimeError(
                    f"Operation {info.name!r} references missing inverse "
                    f"{info.inverse_op!r}."
                )

            if meta.side == "SP" and info.partition == "null":
                raise PermissionError(
                    f"Operation {info.name!r} has NULL partition semantics "
                    "and cannot be applied to SP."
                )

            reason = generation_invalid_reason(
                meta,
                X,
                info,
                source_idx,
                params=params,
            )
            if reason is not None:
                raise ValueError(
                    f"Invalid generation for {info.name!r} on "
                    f"{meta.side} gene {source_idx}: {reason}"
                )

            expected_output_count = operation_output_count(
                info,
                meta,
                X,
                source_idx,
                params=params,
            )

            meta_before = len(meta)
            x_before = len(X)

            result = func(meta, X, *args, **kwargs)

            meta_added = len(meta) - meta_before
            x_added = len(X) - x_before

            if (
                meta_added != expected_output_count
                or x_added != expected_output_count
            ):
                raise RuntimeError(
                    f"Operation {info.name!r} expected "
                    f"{expected_output_count} outputs but added "
                    f"meta={meta_added}, X={x_added}."
                )

            if len(meta) != len(X):
                raise RuntimeError(
                    f"{meta.side} metadata/data gene indices diverged after "
                    f"{info.name!r}: {len(meta)} != {len(X)}."
                )

            # All outputs of one operation invocation share one exact
            # transformation signature.
            for gidx in range(meta_before, len(meta)):
                meta.params[gidx] = _copy_params(params)

            return result

        info.func = wrapped

        wrapped.operation_info = info
        wrapped.partition = info.partition
        wrapped.inverse_op = info.inverse_op
        wrapped.output_count = info.output_count

        OP_REGISTRY[info.name] = info
        return wrapped

    return decorator


def _copy_param_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.copy()

    if isinstance(value, list):
        return [_copy_param_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_copy_param_value(item) for item in value)

    if isinstance(value, dict):
        return {
            str(key): _copy_param_value(item)
            for key, item in value.items()
        }

    return value


def _copy_params(params: dict[str, Any] | None) -> dict[str, Any]:
    return {
        str(key): _copy_param_value(value)
        for key, value in (params or {}).items()
    }


def _freeze_value(value: Any):
    """Canonical hashable form for exact transition signatures."""
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            str(value.dtype),
            tuple(value.shape),
            tuple(_freeze_value(item) for item in value.flat),
        )

    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    (
                        str(key),
                        _freeze_value(item),
                    )
                    for key, item in value.items()
                )
            ),
        )

    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)

    return value


def _source_tuple(source_idx: Any) -> tuple[int, ...]:
    """Normalize one source reference into an index tuple."""
    normalized = _normalize_source(source_idx)
    if isinstance(normalized, int):
        return (normalized,)
    return tuple(int(index) for index in normalized)


def _canonical_source(
    info: OperationInfo,
    source_idx: Any,
):
    indices = _source_tuple(source_idx)

    if len(indices) != info.source_count:
        raise ValueError(
            f"{info.name!r} requires {info.source_count} source gene(s), "
            f"got {len(indices)}."
        )

    if not info.ordered_sources and len(indices) > 1:
        indices = tuple(sorted(indices))

    if info.source_count == 1:
        return indices[0]

    return indices


def _transition_signature(
    info: OperationInfo,
    source_idx: Any,
    params: dict[str, Any] | None,
):
    return (
        info.name,
        _canonical_source(info, source_idx),
        _freeze_value(params or {}),
    )


def _atomic_dtypes(value: Any) -> set[np.dtype]:
    """Recursively collect atomic NumPy dtypes from one instantiated value."""
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return {np.dtype(value.dtype)}

        dtypes: set[np.dtype] = set()
        for item in value.flat:
            dtypes.update(_atomic_dtypes(item))
        return dtypes

    if isinstance(value, (list, tuple)):
        dtypes: set[np.dtype] = set()
        for item in value:
            dtypes.update(_atomic_dtypes(item))
        return dtypes

    return {np.dtype(np.asarray(value).dtype)}


def gene_atomic_dtypes(X: ProgramX, gidx: int) -> tuple[np.dtype, ...]:
    """Return all atomic dtypes observed in one gene across all samples."""
    dtypes: set[np.dtype] = set()

    for value in X[gidx]:
        dtypes.update(_atomic_dtypes(value))

    return tuple(sorted(dtypes, key=str))


def values_exactly_equal(a: Any, b: Any) -> bool:
    """Exact structural equality including array shape, dtype, and contents."""
    if isinstance(a, np.generic):
        a = np.asarray(a)
    if isinstance(b, np.generic):
        b = np.asarray(b)

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False

        if a.dtype == object:
            return all(
                values_exactly_equal(x, y)
                for x, y in zip(a.flat, b.flat)
            )

        try:
            return bool(np.array_equal(a, b, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(a, b))

    if isinstance(a, dict) or isinstance(b, dict):
        if not isinstance(a, dict) or not isinstance(b, dict):
            return False
        if set(a) != set(b):
            return False
        return all(
            values_exactly_equal(a[key], b[key])
            for key in a
        )

    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if type(a) is not type(b):
            return False
        if len(a) != len(b):
            return False
        return all(
            values_exactly_equal(x, y)
            for x, y in zip(a, b)
        )

    if type(a) is not type(b):
        return False

    try:
        result = a == b
    except Exception:
        return False

    if isinstance(result, np.ndarray):
        return bool(np.all(result))
    return bool(result)


def genes_exactly_equal(
    gene_a: np.ndarray,
    gene_b: np.ndarray,
) -> bool:
    """Exact equality for two complete genes across every sample."""
    if not isinstance(gene_a, np.ndarray) or not isinstance(gene_b, np.ndarray):
        return False
    if gene_a.ndim != 1 or gene_b.ndim != 1:
        return False
    if len(gene_a) != len(gene_b):
        return False

    return all(
        values_exactly_equal(a, b)
        for a, b in zip(gene_a, gene_b)
    )


def equivalent_gene_idx(
    X: ProgramX,
    gene_values: np.ndarray,
    *,
    stop_before: int | None = None,
) -> int:
    """Return an exactly equivalent existing gene index, or -1."""
    end = len(X) if stop_before is None else int(stop_before)

    for gidx in range(end):
        if genes_exactly_equal(X[gidx], gene_values):
            return gidx

    return -1


def _resolve_operation(
    op: str | OperationInfo | Callable,
) -> OperationInfo:
    if isinstance(op, OperationInfo):
        return op

    if isinstance(op, str):
        try:
            return OP_REGISTRY[op]
        except KeyError as exc:
            raise KeyError(f"Unknown operation {op!r}.") from exc

    info = getattr(op, "operation_info", None)
    if isinstance(info, OperationInfo):
        return info

    raise TypeError(
        "op must be an operation name, OperationInfo, or registered callable."
    )


def operation_output_count(
    op: str | OperationInfo | Callable,
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: Any,
    *,
    params: dict[str, Any] | None = None,
) -> int:
    """Return the output count for one exact operation application."""
    info = _resolve_operation(op)

    if info.output_count is not None:
        return int(info.output_count)

    if info.output_count_estimator is None:
        raise RuntimeError(
            f"Operation {info.name!r} has no output-count definition."
        )

    count = int(
        info.output_count_estimator(
            meta,
            X,
            _canonical_source(info, source_idx),
            _copy_params(params),
        )
    )

    if count < 1:
        raise ValueError(
            f"Operation {info.name!r} estimated invalid output count {count}."
        )

    return count


def generation_exists(
    meta: ProgramMeta,
    op: str | OperationInfo | Callable,
    source_idx: Any,
    *,
    params: dict[str, Any] | None = None,
) -> bool:
    """Whether the exact (operation, source(s), parameters) already exists."""
    info = _resolve_operation(op)
    candidate = _transition_signature(info, source_idx, params)

    for gidx in range(len(meta)):
        if meta.op[gidx] != info.name:
            continue

        existing = _transition_signature(
            info,
            meta.source[gidx],
            meta.params[gidx],
        )
        if existing == candidate:
            return True

    return False


def generation_invalid_reason(
    meta: ProgramMeta,
    X: ProgramX,
    op: str | OperationInfo | Callable,
    source_idx: Any,
    *,
    params: dict[str, Any] | None = None,
) -> str | None:
    """Return None when a candidate generation is legal, else its reason."""
    _validate_program_pair(meta, X)
    info = _resolve_operation(op)

    try:
        canonical_source = _canonical_source(info, source_idx)
        source_tuple = _source_tuple(canonical_source)
    except (TypeError, ValueError) as exc:
        return str(exc)

    for index in source_tuple:
        if index < 0 or index >= len(X):
            return f"source index {index} is outside the current gene range"

    if meta.side == "SP" and info.partition == "null":
        return "SP may not use NULL partition operations"

    for index in source_tuple:
        source_dims = meta.dims[index]

        if (
            info.min_dims_exclusive is not None
            and source_dims <= info.min_dims_exclusive
        ):
            return (
                f"source gene {index} dims={source_dims} must be > "
                f"{info.min_dims_exclusive}"
            )

        if (
            info.allowed_dims is not None
            and source_dims not in info.allowed_dims
        ):
            return (
                f"source gene {index} dims={source_dims} must be in "
                f"{info.allowed_dims}"
            )

        if info.atomic_dtypes is not None:
            actual_dtypes = set(gene_atomic_dtypes(X, index))
            allowed_dtypes = set(info.atomic_dtypes)

            if not actual_dtypes:
                return f"source gene {index} has no atomic dtype"

            if not actual_dtypes.issubset(allowed_dtypes):
                return (
                    f"source gene {index} atomic dtype restriction failed: "
                    f"actual={sorted(map(str, actual_dtypes))}, "
                    f"allowed={sorted(map(str, allowed_dtypes))}"
                )

    params = _copy_params(params)

    if info.validator is not None:
        try:
            allowed = bool(
                info.validator(meta, X, canonical_source, params)
            )
        except Exception as exc:
            return f"custom validator raised {type(exc).__name__}: {exc}"

        if not allowed:
            return "custom validator rejected candidate"

    if generation_exists(
        meta,
        info,
        canonical_source,
        params=params,
    ):
        return "exact operation/source/parameter transition already exists"

    return None


def valid_generation(
    meta: ProgramMeta,
    X: ProgramX,
    op: str | OperationInfo | Callable,
    source_idx: Any,
    *,
    params: dict[str, Any] | None = None,
) -> bool:
    """Check whether an exact candidate transformation may be generated."""
    return generation_invalid_reason(
        meta,
        X,
        op,
        source_idx,
        params=params,
    ) is None


def sample_operation_params(
    op: str | OperationInfo | Callable,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Sample registered operation parameters for one generation attempt."""
    info = _resolve_operation(op)

    if info.parameter_sampler is None:
        return {}

    params = info.parameter_sampler(rng)

    if params is None:
        return {}
    if not isinstance(params, dict):
        raise TypeError(
            f"Parameter sampler for {info.name!r} must return a dict."
        )

    return _copy_params(params)


def _validate_program_pair(meta: ProgramMeta, X: ProgramX) -> None:
    if not isinstance(meta, ProgramMeta):
        raise TypeError("meta must be a ProgramMeta.")
    if not isinstance(X, ProgramX):
        raise TypeError("X must be a ProgramX.")
    if meta.side != X.side:
        raise ValueError(
            f"Program side mismatch: meta={meta.side}, X={X.side}."
        )
    if len(meta) != len(X):
        raise ValueError(
            f"{meta.side} metadata/data gene counts differ: "
            f"{len(meta)} != {len(X)}."
        )
    if not (
        len(meta.source)
        == len(meta.op)
        == len(meta.dims)
        == len(meta.params)
    ):
        raise ValueError(
            f"{meta.side} metadata fields are not parallel."
        )


def _validate_source(meta: ProgramMeta, X: ProgramX, source_idx: int) -> int:
    _validate_program_pair(meta, X)

    if isinstance(source_idx, bool) or not isinstance(
        source_idx,
        (int, np.integer),
    ):
        raise TypeError("source_idx must be an integer gene index.")

    source_idx = int(source_idx)

    if source_idx < 0 or source_idx >= len(X):
        raise IndexError(
            f"source_idx {source_idx} is outside {X.side}_X gene range "
            f"[0, {len(X) - 1}]."
        )

    return source_idx


def _validate_sources(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: Any,
    *,
    expected_count: int,
) -> tuple[int, ...]:
    _validate_program_pair(meta, X)
    indices = _source_tuple(source_idx)

    if len(indices) != expected_count:
        raise ValueError(
            f"Expected {expected_count} source genes, got {len(indices)}."
        )

    for index in indices:
        if index < 0 or index >= len(X):
            raise IndexError(
                f"source index {index} is outside {X.side}_X gene range "
                f"[0, {len(X) - 1}]."
            )

    return indices


def _bool_has_true_validator(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: Any,
    params: dict[str, Any],
) -> bool:
    """Require at least one True in every instantiated sample."""
    source = _source_tuple(source_idx)[0]

    return all(
        bool(np.any(np.asarray(X[source, sample_idx], dtype=bool)))
        for sample_idx in range(X.sample_count)
    )


def _bool_cavity_mask(array: np.ndarray) -> np.ndarray:
    """Return False cells enclosed by True using axis-adjacent connectivity."""
    array = np.asarray(array, dtype=bool)

    if array.ndim == 0:
        return np.asarray(False, dtype=bool)

    false_mask = np.logical_not(array)
    reachable = np.zeros(array.shape, dtype=bool)

    if not np.any(false_mask):
        return reachable

    stack: list[tuple[int, ...]] = []

    for axis in range(array.ndim):
        for boundary_index in (0, array.shape[axis] - 1):
            slicer = [slice(None)] * array.ndim
            slicer[axis] = boundary_index

            boundary_false = np.argwhere(false_mask[tuple(slicer)])

            for reduced_coord in boundary_false:
                coord = []
                reduced_pos = 0

                for dim in range(array.ndim):
                    if dim == axis:
                        coord.append(boundary_index)
                    else:
                        coord.append(int(reduced_coord[reduced_pos]))
                        reduced_pos += 1

                coord_tuple = tuple(coord)

                if not reachable[coord_tuple]:
                    reachable[coord_tuple] = True
                    stack.append(coord_tuple)

    while stack:
        coord = stack.pop()

        for axis in range(array.ndim):
            for delta in (-1, 1):
                neighbor = list(coord)
                neighbor[axis] += delta

                if (
                    neighbor[axis] < 0
                    or neighbor[axis] >= array.shape[axis]
                ):
                    continue

                neighbor_tuple = tuple(neighbor)

                if (
                    false_mask[neighbor_tuple]
                    and not reachable[neighbor_tuple]
                ):
                    reachable[neighbor_tuple] = True
                    stack.append(neighbor_tuple)

    return np.logical_and(false_mask, np.logical_not(reachable))


def _bool2_same_shape_validator(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: Any,
    params: dict[str, Any],
) -> bool:
    first, second = _source_tuple(source_idx)

    if first == second:
        return False

    return all(
        np.asarray(X[first, sample_idx]).shape
        == np.asarray(X[second, sample_idx]).shape
        for sample_idx in range(X.sample_count)
    )


def _bool_trim_validator(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: Any,
    params: dict[str, Any],
) -> bool:
    source = _source_tuple(source_idx)[0]

    for sample_idx in range(X.sample_count):
        array = np.asarray(X[source, sample_idx], dtype=bool)

        if array.ndim == 0 or any(size == 0 for size in array.shape):
            return False

        removable_boundary = False

        for axis in range(array.ndim):
            first_slice = np.take(array, 0, axis=axis)
            last_slice = np.take(array, -1, axis=axis)

            if not np.any(first_slice) or not np.any(last_slice):
                removable_boundary = True
                break

        if not removable_boundary:
            return False

    return True


def _trim_bool_array(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (offset, tight boolean bounding box) for one sample."""
    array = np.asarray(array, dtype=bool)

    true_coords = np.argwhere(array)

    if true_coords.size == 0:
        offset = np.zeros(array.ndim, dtype=np.int64)
        slices = tuple(slice(0, 0) for _ in range(array.ndim))
        return offset, array[slices].copy()

    starts = true_coords.min(axis=0).astype(np.int64, copy=False)
    stops = true_coords.max(axis=0) + 1

    slices = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(starts, stops)
    )

    return starts.copy(), array[slices].copy()


def _append_gene(
    meta: ProgramMeta,
    X: ProgramX,
    values,
    *,
    source,
    op_name: str,
) -> int:
    gidx = X.append_gene(values)
    dims = _gene_ndim(X[gidx])

    meta_gidx = meta.append(
        source=source,
        op=op_name,
        dims=dims,
        params={},
    )

    if meta_gidx != gidx:
        raise RuntimeError(
            f"{meta.side} metadata/data indices diverged: "
            f"meta={meta_gidx}, X={gidx}."
        )

    return gidx


def _partition_composite_colors(
    X: ProgramX,
    source_idx: int,
) -> np.ndarray:
    """Sorted union of categorical colors used across all samples."""
    colors = [
        np.asarray(value).reshape(-1)
        for value in X[source_idx]
    ]

    if not colors:
        return np.empty(0, dtype=np.int64)

    return np.unique(
        np.concatenate(colors)
    ).astype(np.int64, copy=False)


def _partition_composite_output_count(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: Any,
    params: dict[str, Any],
) -> int:
    source = _source_tuple(source_idx)[0]
    return 2 * len(_partition_composite_colors(X, source))


def _indiv_1dim_validator(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: Any,
    params: dict[str, Any],
) -> bool:
    """Require one consistent 1D length across all samples."""
    source = _source_tuple(source_idx)[0]

    lengths = [
        len(np.asarray(value))
        for value in X[source]
    ]

    return bool(lengths) and len(set(lengths)) == 1 and lengths[0] > 0


def _indiv_1dim_output_count(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: Any,
    params: dict[str, Any],
) -> int:
    source = _source_tuple(source_idx)[0]
    return len(np.asarray(X[source, 0]))


@operation(
    partition="and",
    inverse_op="inv_partition_shape",
    output_count=2,
    allowed_dims=(2,),
)
def partition_shape(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> tuple[int, int]:
    """Partition a 2D source into scalar height and width genes."""
    source_idx = _validate_source(meta, X, source_idx)

    heights = [
        np.int64(np.asarray(value).shape[0])
        for value in X[source_idx]
    ]
    widths = [
        np.int64(np.asarray(value).shape[1])
        for value in X[source_idx]
    ]

    h_gidx = _append_gene(
        meta,
        X,
        heights,
        source=source_idx,
        op_name="partition_shape",
    )
    w_gidx = _append_gene(
        meta,
        X,
        widths,
        source=source_idx,
        op_name="partition_shape",
    )

    return h_gidx, w_gidx


@operation(
    partition="and",
    inverse_op="inv_indiv_1dim",
    output_count=None,
    output_count_estimator=_indiv_1dim_output_count,
    allowed_dims=(1,),
    validator=_indiv_1dim_validator,
)
def indiv_1dim(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> tuple[int, ...]:
    """Partition a 1D source into one scalar gene per element position."""
    source_idx = _validate_source(meta, X, source_idx)
    length = len(np.asarray(X[source_idx, 0]))

    generated: list[int] = []

    for position in range(length):
        values = [
            np.asarray(value)[position]
            for value in X[source_idx]
        ]

        gidx = _append_gene(
            meta,
            X,
            values,
            source=source_idx,
            op_name="indiv_1dim",
        )
        generated.append(gidx)

    return tuple(generated)


@operation(
    partition="and",
    inverse_op="inv_partition_composite",
    output_count=None,
    output_count_estimator=_partition_composite_output_count,
    min_dims_exclusive=1,
    atomic_dtypes=(np.int64,),
)
def partition_composite(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> tuple[int, ...]:
    """Partition a categorical grid into one ID/mask pair per used color.

    The color set is the sorted union observed across all training samples.
    For each color, two gene-major outputs are created:

      1. scalar np.int64 color ID, constant across samples;
      2. 2D boolean presence mask for that color in each sample.

    If a color is absent from one sample, its presence value for that sample is
    an all-False matrix with the same spatial shape as the source sample.
    """
    source_idx = _validate_source(meta, X, source_idx)
    colors = _partition_composite_colors(X, source_idx)

    generated: list[int] = []

    for color in colors:
        color = np.int64(color)

        color_values = [
            np.int64(color)
            for _ in range(X.sample_count)
        ]
        presence_values = [
            (np.asarray(value) == color).astype(bool, copy=False)
            for value in X[source_idx]
        ]

        color_gidx = _append_gene(
            meta,
            X,
            color_values,
            source=source_idx,
            op_name="partition_composite",
        )
        presence_gidx = _append_gene(
            meta,
            X,
            presence_values,
            source=source_idx,
            op_name="partition_composite",
        )

        generated.extend([color_gidx, presence_gidx])

    return tuple(generated)



@operation(
    partition="null",
    inverse_op="inv_bool_sum",
    output_count=1,
    atomic_dtypes=(np.bool_,),
)
def bool_sum(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> int:
    """Count cumulative True values in each instantiated boolean source."""
    source_idx = _validate_source(meta, X, source_idx)

    values = [
        np.int64(np.count_nonzero(np.asarray(value, dtype=bool)))
        for value in X[source_idx]
    ]

    return _append_gene(
        meta,
        X,
        values,
        source=source_idx,
        op_name="bool_sum",
    )


@operation(
    partition="null",
    inverse_op="inv_bool_cavity",
    output_count=1,
    atomic_dtypes=(np.bool_,),
    validator=_bool_has_true_validator,
)
def bool_cavity(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> int:
    """Mark False regions fully enclosed by True values."""
    source_idx = _validate_source(meta, X, source_idx)

    values = [
        _bool_cavity_mask(np.asarray(value, dtype=bool))
        for value in X[source_idx]
    ]

    return _append_gene(
        meta,
        X,
        values,
        source=source_idx,
        op_name="bool_cavity",
    )


@operation(
    partition="or",
    inverse_op="inv_bool_complement",
    output_count=1,
    atomic_dtypes=(np.bool_,),
)
def bool_complement(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> int:
    """Boolean complement preserving the complete source shape."""
    source_idx = _validate_source(meta, X, source_idx)

    values = [
        np.logical_not(np.asarray(value, dtype=bool))
        for value in X[source_idx]
    ]

    return _append_gene(
        meta,
        X,
        values,
        source=source_idx,
        op_name="bool_complement",
    )


@operation(
    partition="null",
    inverse_op="inv_bool2_union",
    output_count=1,
    source_count=2,
    ordered_sources=False,
    atomic_dtypes=(np.bool_,),
    validator=_bool2_same_shape_validator,
)
def bool2_union(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: tuple[int, int],
) -> int:
    """Elementwise boolean union of two same-shaped boolean genes."""
    first, second = sorted(
        _validate_sources(
            meta,
            X,
            source_idx,
            expected_count=2,
        )
    )
    source = (first, second)

    values = [
        np.logical_or(
            np.asarray(X[first, sample_idx], dtype=bool),
            np.asarray(X[second, sample_idx], dtype=bool),
        )
        for sample_idx in range(X.sample_count)
    ]

    return _append_gene(
        meta,
        X,
        values,
        source=source,
        op_name="bool2_union",
    )


@operation(
    partition="null",
    inverse_op="inv_bool2_intersect",
    output_count=1,
    source_count=2,
    ordered_sources=False,
    atomic_dtypes=(np.bool_,),
    validator=_bool2_same_shape_validator,
)
def bool2_intersect(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: tuple[int, int],
) -> int:
    """Elementwise boolean intersection of two same-shaped boolean genes."""
    first, second = sorted(
        _validate_sources(
            meta,
            X,
            source_idx,
            expected_count=2,
        )
    )
    source = (first, second)

    values = [
        np.logical_and(
            np.asarray(X[first, sample_idx], dtype=bool),
            np.asarray(X[second, sample_idx], dtype=bool),
        )
        for sample_idx in range(X.sample_count)
    ]

    return _append_gene(
        meta,
        X,
        values,
        source=source,
        op_name="bool2_intersect",
    )


@operation(
    partition="or",
    inverse_op="inv_mat2_cwrotate",
    output_count=1,
    allowed_dims=(2,),
)
def mat2_cwrotate(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> int:
    """Rotate every 2D sample clockwise by 90 degrees."""
    source_idx = _validate_source(meta, X, source_idx)

    values = [
        np.rot90(np.asarray(value), k=-1).copy()
        for value in X[source_idx]
    ]

    return _append_gene(
        meta,
        X,
        values,
        source=source_idx,
        op_name="mat2_cwrotate",
    )


@operation(
    partition="or",
    inverse_op="inv_dim0_flip",
    output_count=1,
    min_dims_exclusive=0,
)
def dim0_flip(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> int:
    """Reverse values along dimension 0."""
    source_idx = _validate_source(meta, X, source_idx)

    values = [
        np.flip(np.asarray(value), axis=0).copy()
        for value in X[source_idx]
    ]

    return _append_gene(
        meta,
        X,
        values,
        source=source_idx,
        op_name="dim0_flip",
    )


@operation(
    partition="or",
    inverse_op="inv_dim1_flip",
    output_count=1,
    min_dims_exclusive=1,
)
def dim1_flip(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> int:
    """Reverse values along dimension 1."""
    source_idx = _validate_source(meta, X, source_idx)

    values = [
        np.flip(np.asarray(value), axis=1).copy()
        for value in X[source_idx]
    ]

    return _append_gene(
        meta,
        X,
        values,
        source=source_idx,
        op_name="dim1_flip",
    )


@operation(
    partition="or",
    inverse_op="inv_dim2_flip",
    output_count=1,
    min_dims_exclusive=2,
)
def dim2_flip(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> int:
    """Reverse values along dimension 2."""
    source_idx = _validate_source(meta, X, source_idx)

    values = [
        np.flip(np.asarray(value), axis=2).copy()
        for value in X[source_idx]
    ]

    return _append_gene(
        meta,
        X,
        values,
        source=source_idx,
        op_name="dim2_flip",
    )


@operation(
    partition="and",
    inverse_op="inv_partition_bool_trim",
    output_count=2,
    min_dims_exclusive=0,
    atomic_dtypes=(np.bool_,),
    validator=_bool_trim_validator,
)
def partition_bool_trim(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> tuple[int, int]:
    """Partition a boolean gene into offset and tight remaining structure."""
    source_idx = _validate_source(meta, X, source_idx)

    offsets = []
    trimmed_values = []

    for value in X[source_idx]:
        offset, trimmed = _trim_bool_array(
            np.asarray(value, dtype=bool)
        )
        offsets.append(offset)
        trimmed_values.append(trimmed)

    offset_gidx = _append_gene(
        meta,
        X,
        offsets,
        source=source_idx,
        op_name="partition_bool_trim",
    )
    data_gidx = _append_gene(
        meta,
        X,
        trimmed_values,
        source=source_idx,
        op_name="partition_bool_trim",
    )

    return offset_gidx, data_gidx

def _rng(
    rng: np.random.Generator | int | None,
) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _eligible_operation_infos(
    *,
    side: str,
    max_output_count: int | None = None,
    operation_names: Iterable[str] | None = None,
) -> list[OperationInfo]:
    allowed_names = (
        None
        if operation_names is None
        else {str(name) for name in operation_names}
    )

    infos = [
        info
        for info in OP_REGISTRY.values()
        if info.func is not None
        and (allowed_names is None or info.name in allowed_names)
        and (side != "SP" or info.partition != "null")
    ]
    return infos


def _candidate_signature(
    info: OperationInfo,
    source_idx: Any,
    params: dict[str, Any] | None,
):
    return _transition_signature(info, source_idx, params)


def _source_candidates(
    info: OperationInfo,
    gene_count: int,
) -> list[Any]:
    """Enumerate legal source-index tuples for one operation arity."""
    if info.source_count == 1:
        return list(range(gene_count))

    indices = range(gene_count)

    if info.ordered_sources:
        return [
            tuple(source)
            for source in itertools.permutations(indices, info.source_count)
        ]

    return [
        tuple(source)
        for source in itertools.combinations(indices, info.source_count)
    ]


def _parameterless_valid_candidates(
    meta: ProgramMeta,
    X: ProgramX,
    *,
    max_output_count: int | None = None,
    operation_names: Iterable[str] | None = None,
    excluded_signatures: set[Any] | None = None,
) -> list[tuple[OperationInfo, Any, dict[str, Any]]]:
    """Enumerate the complete finite candidate space for parameterless ops."""
    excluded_signatures = excluded_signatures or set()
    infos = _eligible_operation_infos(
        side=meta.side,
        max_output_count=max_output_count,
        operation_names=operation_names,
    )

    candidates = []

    for info in infos:
        if info.parameter_sampler is not None:
            continue

        for source_idx in _source_candidates(info, len(X)):
            params: dict[str, Any] = {}
            signature = _candidate_signature(info, source_idx, params)

            if signature in excluded_signatures:
                continue

            if (
                max_output_count is not None
                and operation_output_count(
                    info,
                    meta,
                    X,
                    source_idx,
                    params=params,
                ) > max_output_count
            ):
                continue

            if valid_generation(
                meta,
                X,
                info,
                source_idx,
                params=params,
            ):
                candidates.append((info, source_idx, params))

    return candidates


def _has_parameterized_ops(
    *,
    side: str,
    max_output_count: int | None,
    operation_names: Iterable[str] | None,
) -> bool:
    return any(
        info.parameter_sampler is not None
        for info in _eligible_operation_infos(
            side=side,
            max_output_count=max_output_count,
            operation_names=operation_names,
        )
    )


def _random_parameterized_candidate(
    meta: ProgramMeta,
    X: ProgramX,
    *,
    rng: np.random.Generator,
    max_output_count: int | None,
    max_attempts: int,
    operation_names: Iterable[str] | None,
    excluded_signatures: set[Any],
) -> tuple[OperationInfo, Any, dict[str, Any]] | None:
    infos = [
        info
        for info in _eligible_operation_infos(
            side=meta.side,
            max_output_count=max_output_count,
            operation_names=operation_names,
        )
        if info.parameter_sampler is not None
    ]

    if not infos:
        return None

    for _ in range(max_attempts):
        info = infos[int(rng.integers(len(infos)))]
        source_space = _source_candidates(info, len(X))
        if not source_space:
            continue
        source_idx = source_space[int(rng.integers(len(source_space)))]
        params = sample_operation_params(info, rng)
        signature = _candidate_signature(info, source_idx, params)

        if signature in excluded_signatures:
            continue

        if (
            max_output_count is not None
            and operation_output_count(
                info,
                meta,
                X,
                source_idx,
                params=params,
            ) > max_output_count
        ):
            continue

        if valid_generation(
            meta,
            X,
            info,
            source_idx,
            params=params,
        ):
            return info, source_idx, params

    return None


def _flatten_generated_indices(result: Any) -> list[int]:
    if isinstance(result, np.generic):
        result = result.item()

    if isinstance(result, int) and not isinstance(result, bool):
        return [int(result)]

    if isinstance(result, np.ndarray):
        result = result.tolist()

    if isinstance(result, (list, tuple)):
        indices: list[int] = []
        for item in result:
            indices.extend(_flatten_generated_indices(item))
        return indices

    raise TypeError(
        "Registered generation operations must return gene index/indices."
    )


def _rollback_appended_genes(
    meta: ProgramMeta,
    X: ProgramX,
    *,
    meta_len: int,
    x_len: int,
) -> None:
    """Rollback an append-only operation attempt."""
    del meta.source[meta_len:]
    del meta.op[meta_len:]
    del meta.dims[meta_len:]
    del meta.params[meta_len:]
    del X.genes[x_len:]


def _generated_outputs_are_novel(
    X: ProgramX,
    new_indices: list[int],
    *,
    existing_count: int,
) -> tuple[bool, str | None]:
    """Require every output gene to be novel versus all retained gene data."""
    accepted_new: list[int] = []

    for gidx in new_indices:
        duplicate_idx = equivalent_gene_idx(
            X,
            X[gidx],
            stop_before=existing_count,
        )
        if duplicate_idx >= 0:
            return (
                False,
                f"generated gene {gidx} exactly duplicates existing gene "
                f"{duplicate_idx}",
            )

        for prior_new_idx in accepted_new:
            if genes_exactly_equal(X[gidx], X[prior_new_idx]):
                return (
                    False,
                    f"generated genes {prior_new_idx} and {gidx} are exactly "
                    "equivalent",
                )

        accepted_new.append(gidx)

    return True, None


def _try_candidate_transactionally(
    meta: ProgramMeta,
    X: ProgramX,
    info: OperationInfo,
    source_idx: Any,
    params: dict[str, Any],
) -> tuple[list[int], str | None]:
    """Apply one candidate and keep it only when all outputs are novel."""
    expected_output_count = operation_output_count(
        info,
        meta,
        X,
        source_idx,
        params=params,
    )

    meta_before = len(meta)
    x_before = len(X)

    result = info.func(
        meta,
        X,
        source_idx,
        **params,
    )
    new_indices = _flatten_generated_indices(result)

    if len(new_indices) != expected_output_count:
        _rollback_appended_genes(
            meta,
            X,
            meta_len=meta_before,
            x_len=x_before,
        )
        raise RuntimeError(
            f"{info.name!r} returned {len(new_indices)} indices but "
            f"expected {expected_output_count}."
        )

    novel, reason = _generated_outputs_are_novel(
        X,
        new_indices,
        existing_count=x_before,
    )

    if not novel:
        _rollback_appended_genes(
            meta,
            X,
            meta_len=meta_before,
            x_len=x_before,
        )
        return [], reason

    return new_indices, None


def _generation_exhausted_message(side: str) -> str:
    return (
        f"{side} generation terminated: entire legal generation space was "
        "explored and no additional unique genes can be generated."
    )


def GP_generate(
    GP_meta: ProgramMeta,
    GP_X: ProgramX,
    n_new_genes: int,
    *,
    rng: np.random.Generator | int | None = None,
    max_attempts_per_generation: int = 500,
    operation_names: Iterable[str] | None = None,
) -> list[int]:
    """Generate novel GP genes while avoiding transition and data duplicates.

    For parameterless operations, every currently legal operation/source pair
    is explored at most once per call. Candidates whose instantiated outputs
    duplicate any retained gene are rolled back.

    If all finite legal candidates have been explored without producing another
    novel gene, generation terminates early with an explicit message.
    """
    _validate_program_pair(GP_meta, GP_X)

    if GP_meta.side != "GP":
        raise ValueError("GP_generate requires GP-side meta/X.")
    if isinstance(n_new_genes, bool) or int(n_new_genes) < 0:
        raise ValueError("n_new_genes must be a non-negative integer.")

    n_new_genes = int(n_new_genes)
    rng = _rng(rng)
    generated: list[int] = []
    rejected_signatures: set[Any] = set()
    parameterized_duplicate_rejections = 0

    while len(generated) < n_new_genes:
        remaining = n_new_genes - len(generated)
        accepted = False

        candidates = _parameterless_valid_candidates(
            GP_meta,
            GP_X,
            max_output_count=remaining,
            operation_names=operation_names,
            excluded_signatures=rejected_signatures,
        )
        rng.shuffle(candidates)

        for info, source_idx, params in candidates:
            signature = _candidate_signature(info, source_idx, params)
            rejected_signatures.add(signature)

            new_indices, _ = _try_candidate_transactionally(
                GP_meta,
                GP_X,
                info,
                source_idx,
                params,
            )

            if new_indices:
                generated.extend(new_indices)
                accepted = True
                break

        if accepted:
            continue

        parameterized = _has_parameterized_ops(
            side="GP",
            max_output_count=remaining,
            operation_names=operation_names,
        )

        if parameterized:
            candidate = _random_parameterized_candidate(
                GP_meta,
                GP_X,
                rng=rng,
                max_output_count=remaining,
                max_attempts=max_attempts_per_generation,
                operation_names=operation_names,
                excluded_signatures=rejected_signatures,
            )

            if candidate is not None:
                info, source_idx, params = candidate
                signature = _candidate_signature(info, source_idx, params)
                rejected_signatures.add(signature)

                new_indices, _ = _try_candidate_transactionally(
                    GP_meta,
                    GP_X,
                    info,
                    source_idx,
                    params,
                )

                if new_indices:
                    generated.extend(new_indices)
                    parameterized_duplicate_rejections = 0
                    continue

                parameterized_duplicate_rejections += 1
                if (
                    parameterized_duplicate_rejections
                    >= max_attempts_per_generation
                ):
                    print(
                        "GP generation terminated: no novel gene was found "
                        "within the sampled parameterized generation space."
                    )
                    break

                continue

            print(
                "GP generation terminated: no novel gene was found within the "
                "sampled parameterized generation space."
            )
            break

        print(_generation_exhausted_message("GP"))
        break

    return generated


def SP_generate(
    SP_meta: ProgramMeta,
    SP_X: ProgramX,
    ST: SolutionTree | None = None,
    *,
    rng: np.random.Generator | int | None = None,
    max_attempts: int = 500,
    operation_names: Iterable[str] | None = None,
) -> list[int]:
    """Generate one novel SP operation application.

    Legal parameterless SP candidates are exhaustively explored until one
    produces entirely novel output genes. Duplicate-data candidates are rolled
    back. If none remain, the function prints that the legal generation space
    has been exhausted and returns [].
    """
    _validate_program_pair(SP_meta, SP_X)

    if SP_meta.side != "SP":
        raise ValueError("SP_generate requires SP-side meta/X.")

    rng = _rng(rng)
    rejected_signatures: set[Any] = set()

    candidates = _parameterless_valid_candidates(
        SP_meta,
        SP_X,
        operation_names=operation_names,
        excluded_signatures=rejected_signatures,
    )
    rng.shuffle(candidates)

    for info, source_idx, params in candidates:
        signature = _candidate_signature(info, source_idx, params)
        rejected_signatures.add(signature)

        new_indices, _ = _try_candidate_transactionally(
            SP_meta,
            SP_X,
            info,
            source_idx,
            params,
        )

        if not new_indices:
            continue

        if ST is not None:
            ST.register_generation(
                SP_meta,
                source_gidx=int(source_idx),
                generated_gidxs=new_indices,
                partition=info.partition,
                inverse_op=info.inverse_op,
                op_name=info.name,
            )

        return new_indices

    if _has_parameterized_ops(
        side="SP",
        max_output_count=None,
        operation_names=operation_names,
    ):
        for _ in range(max_attempts):
            candidate = _random_parameterized_candidate(
                SP_meta,
                SP_X,
                rng=rng,
                max_output_count=None,
                max_attempts=1,
                operation_names=operation_names,
                excluded_signatures=rejected_signatures,
            )

            if candidate is None:
                continue

            info, source_idx, params = candidate
            signature = _candidate_signature(info, source_idx, params)
            rejected_signatures.add(signature)

            new_indices, _ = _try_candidate_transactionally(
                SP_meta,
                SP_X,
                info,
                source_idx,
                params,
            )

            if not new_indices:
                continue

            if ST is not None:
                ST.register_generation(
                    SP_meta,
                    source_gidx=int(source_idx),
                    generated_gidxs=new_indices,
                    partition=info.partition,
                    inverse_op=info.inverse_op,
                    op_name=info.name,
                )

            return new_indices

        print(
            "SP generation terminated: no novel gene was found within the "
            "sampled parameterized generation space."
        )
        return []

    print(_generation_exhausted_message("SP"))
    return []
