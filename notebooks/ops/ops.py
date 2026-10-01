from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
import inspect
from typing import Any, Callable, Iterable

import numpy as np

from .environment import (
    ProgramMeta,
    ProgramX,
    SolutionTree,
    _gene_ndim,
    _normalize_source,
)


ParameterSampler = Callable[[np.random.Generator], dict[str, Any]]
GenerationValidator = Callable[
    [ProgramMeta, ProgramX, int, dict[str, Any]],
    bool,
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
    full_partition: bool
    output_count: int
    min_dims_exclusive: int | None = None
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
    full_partition: bool,
    output_count: int,
    min_dims_exclusive: int | None = None,
    atomic_dtypes: Iterable[Any] | None = None,
    parameter_sampler: ParameterSampler | None = None,
    validator: GenerationValidator | None = None,
):
    """Register a transformation operation.

    GP may use any registered operation.

    SP may only use operations explicitly marked full_partition=True.

    The decorator also validates every direct operation application, so manual
    notebook calls and random generation obey the same rules.
    """
    if output_count < 1:
        raise ValueError("output_count must be >= 1.")

    def decorator(func: Callable):
        info = OperationInfo(
            name=func.__name__,
            full_partition=bool(full_partition),
            output_count=int(output_count),
            min_dims_exclusive=min_dims_exclusive,
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

            source_idx = int(bound.arguments["source_idx"])
            params = {
                name: value
                for name, value in bound.arguments.items()
                if name not in {"meta", "X", "source_idx"}
            }

            if meta.side == "SP" and not info.full_partition:
                raise PermissionError(
                    f"Operation {info.name!r} is not full_partition and "
                    "cannot be applied to SP."
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

            meta_before = len(meta)
            x_before = len(X)

            result = func(meta, X, *args, **kwargs)

            meta_added = len(meta) - meta_before
            x_added = len(X) - x_before

            if meta_added != info.output_count or x_added != info.output_count:
                raise RuntimeError(
                    f"Operation {info.name!r} declared {info.output_count} "
                    f"outputs but added meta={meta_added}, X={x_added}."
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
        wrapped.full_partition = info.full_partition
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


def _transition_signature(
    op_name: str,
    source_idx: int,
    params: dict[str, Any] | None,
):
    return (
        str(op_name),
        _normalize_source(source_idx),
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


def generation_exists(
    meta: ProgramMeta,
    op: str | OperationInfo | Callable,
    source_idx: int,
    *,
    params: dict[str, Any] | None = None,
) -> bool:
    """Whether the exact (operation, source, parameters) already exists."""
    info = _resolve_operation(op)
    candidate = _transition_signature(info.name, source_idx, params)

    for gidx in range(len(meta)):
        existing = _transition_signature(
            meta.op[gidx],
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
    source_idx: int,
    *,
    params: dict[str, Any] | None = None,
) -> str | None:
    """Return None when a candidate generation is legal, else its reason."""
    _validate_program_pair(meta, X)
    info = _resolve_operation(op)

    if isinstance(source_idx, bool) or not isinstance(
        source_idx,
        (int, np.integer),
    ):
        return "source_idx must be an integer"

    source_idx = int(source_idx)

    if source_idx < 0 or source_idx >= len(X):
        return "source_idx is outside the current gene range"

    if meta.side == "SP" and not info.full_partition:
        return "SP may only use full_partition operations"

    source_dims = meta.dims[source_idx]

    if (
        info.min_dims_exclusive is not None
        and source_dims <= info.min_dims_exclusive
    ):
        return (
            f"source dims={source_dims} must be > "
            f"{info.min_dims_exclusive}"
        )

    if info.atomic_dtypes is not None:
        actual_dtypes = set(gene_atomic_dtypes(X, source_idx))
        allowed_dtypes = set(info.atomic_dtypes)

        if not actual_dtypes:
            return "source has no atomic dtype"

        if not actual_dtypes.issubset(allowed_dtypes):
            return (
                "atomic dtype restriction failed: "
                f"actual={sorted(map(str, actual_dtypes))}, "
                f"allowed={sorted(map(str, allowed_dtypes))}"
            )

    params = _copy_params(params)

    if info.validator is not None:
        try:
            allowed = bool(
                info.validator(meta, X, source_idx, params)
            )
        except Exception as exc:
            return f"custom validator raised {type(exc).__name__}: {exc}"

        if not allowed:
            return "custom validator rejected candidate"

    if generation_exists(
        meta,
        info,
        source_idx,
        params=params,
    ):
        return "exact operation/source/parameter transition already exists"

    return None


def valid_generation(
    meta: ProgramMeta,
    X: ProgramX,
    op: str | OperationInfo | Callable,
    source_idx: int,
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


@operation(
    full_partition=True,
    output_count=1,
    min_dims_exclusive=0,
)
def partition_shape(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> int:
    """Partition one non-scalar gene into its per-sample shape vectors."""
    source_idx = _validate_source(meta, X, source_idx)

    shapes = [
        np.asarray(np.asarray(value).shape, dtype=np.int64)
        for value in X[source_idx]
    ]

    return _append_gene(
        meta,
        X,
        shapes,
        source=source_idx,
        op_name="partition_shape",
    )


@operation(
    full_partition=True,
    output_count=2,
    min_dims_exclusive=1,
    atomic_dtypes=(np.int64,),
)
def partition_composite(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> tuple[int, int]:
    """Partition an integer categorical grid into IDs and presence masks."""
    source_idx = _validate_source(meta, X, source_idx)

    color_values = []
    presence_values = []

    for sample_idx, value in enumerate(X[source_idx]):
        array = np.asarray(value)

        if array.ndim < 2:
            raise ValueError(
                "partition_composite requires source dims > 1; "
                f"sample {sample_idx} has shape {array.shape}."
            )

        colors = np.unique(array)

        presence = np.stack(
            [array == color for color in colors],
            axis=0,
        ).astype(bool, copy=False)

        color_values.append(colors.astype(np.int64, copy=False))
        presence_values.append(presence)

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

    return color_gidx, presence_gidx


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
        and (side != "SP" or info.full_partition)
        and (
            max_output_count is None
            or info.output_count <= max_output_count
        )
    ]
    return infos


def _random_valid_candidate(
    meta: ProgramMeta,
    X: ProgramX,
    *,
    rng: np.random.Generator,
    max_output_count: int | None = None,
    max_attempts: int = 500,
    operation_names: Iterable[str] | None = None,
) -> tuple[OperationInfo, int, dict[str, Any]] | None:
    """Randomly search the registry/source space for one valid candidate."""
    infos = _eligible_operation_infos(
        side=meta.side,
        max_output_count=max_output_count,
        operation_names=operation_names,
    )

    if not infos or len(X) == 0:
        return None

    for _ in range(max_attempts):
        info = infos[int(rng.integers(len(infos)))]
        source_idx = int(rng.integers(len(X)))
        params = sample_operation_params(info, rng)

        if valid_generation(
            meta,
            X,
            info,
            source_idx,
            params=params,
        ):
            return info, source_idx, params

    # Random attempts can miss a sparse legal space. Parameterless operations
    # get one deterministic fallback scan before reporting no candidate.
    parameterless_infos = [
        info
        for info in infos
        if info.parameter_sampler is None
    ]

    rng.shuffle(parameterless_infos)
    source_order = np.arange(len(X))
    rng.shuffle(source_order)

    for info in parameterless_infos:
        for source_idx in source_order:
            if valid_generation(
                meta,
                X,
                info,
                int(source_idx),
                params={},
            ):
                return info, int(source_idx), {}

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


def GP_generate(
    GP_meta: ProgramMeta,
    GP_X: ProgramX,
    n_new_genes: int,
    *,
    rng: np.random.Generator | int | None = None,
    max_attempts_per_generation: int = 500,
    operation_names: Iterable[str] | None = None,
) -> list[int]:
    """Randomly append up to exactly n_new_genes to GP.

    Each successful step randomly selects:
      1. a registered operation,
      2. an existing GP source gene,
      3. operation parameters (currently empty for the initial ops).

    Only candidates accepted by valid_generation are applied.

    Multi-output operations are considered only when their complete output fits
    within the remaining requested gene budget. If no valid candidate can fill
    the remaining budget, generation stops early and returns what was added.
    """
    _validate_program_pair(GP_meta, GP_X)

    if GP_meta.side != "GP":
        raise ValueError("GP_generate requires GP-side meta/X.")
    if isinstance(n_new_genes, bool) or int(n_new_genes) < 0:
        raise ValueError("n_new_genes must be a non-negative integer.")

    n_new_genes = int(n_new_genes)
    rng = _rng(rng)
    generated: list[int] = []

    while len(generated) < n_new_genes:
        remaining = n_new_genes - len(generated)

        candidate = _random_valid_candidate(
            GP_meta,
            GP_X,
            rng=rng,
            max_output_count=remaining,
            max_attempts=max_attempts_per_generation,
            operation_names=operation_names,
        )

        if candidate is None:
            break

        info, source_idx, params = candidate
        result = info.func(
            GP_meta,
            GP_X,
            source_idx,
            **params,
        )
        new_indices = _flatten_generated_indices(result)

        if len(new_indices) != info.output_count:
            raise RuntimeError(
                f"{info.name!r} returned {len(new_indices)} indices but "
                f"declares output_count={info.output_count}."
            )

        generated.extend(new_indices)

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
    """Apply one random valid full-partition generation step to SP.

    One generation step may create multiple genes when the chosen registered
    full-partition operation has output_count > 1.

    If ST is supplied it is synchronized after the SP structure grows.
    Returns the newly generated SP gene indices, or [] when no legal candidate
    remains.
    """
    _validate_program_pair(SP_meta, SP_X)

    if SP_meta.side != "SP":
        raise ValueError("SP_generate requires SP-side meta/X.")

    rng = _rng(rng)

    candidate = _random_valid_candidate(
        SP_meta,
        SP_X,
        rng=rng,
        max_attempts=max_attempts,
        operation_names=operation_names,
    )

    if candidate is None:
        return []

    info, source_idx, params = candidate
    result = info.func(
        SP_meta,
        SP_X,
        source_idx,
        **params,
    )
    new_indices = _flatten_generated_indices(result)

    if ST is not None:
        ST.sync(SP_meta)

    return new_indices
