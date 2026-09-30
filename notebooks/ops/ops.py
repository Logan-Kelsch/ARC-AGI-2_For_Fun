from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Callable

import numpy as np

from .environment import ProgramMeta, ProgramX, _gene_ndim


@dataclass(frozen=True)
class OperationInfo:
    """Static metadata attached to one GP transformation operation."""

    name: str
    full_partition: bool
    output_count: int


OP_REGISTRY: dict[str, OperationInfo] = {}


def operation(*, full_partition: bool, output_count: int):
    """Register a transformation and enforce GP/SP side constraints.

    GP may use any registered operation.

    SP may only use operations explicitly marked full_partition=True.
    """
    if output_count < 1:
        raise ValueError("output_count must be >= 1.")

    def decorator(func: Callable):
        info = OperationInfo(
            name=func.__name__,
            full_partition=bool(full_partition),
            output_count=int(output_count),
        )

        @wraps(func)
        def wrapped(meta: ProgramMeta, X: ProgramX, *args, **kwargs):
            _validate_program_pair(meta, X)

            if meta.side == "SP" and not info.full_partition:
                raise PermissionError(
                    f"Operation {info.name!r} is not full_partition and "
                    "cannot be applied to SP."
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

            return result

        wrapped.operation_info = info
        wrapped.full_partition = info.full_partition
        wrapped.output_count = info.output_count

        OP_REGISTRY[info.name] = info
        return wrapped

    return decorator


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
    )

    if meta_gidx != gidx:
        raise RuntimeError(
            f"{meta.side} metadata/data indices diverged: "
            f"meta={meta_gidx}, X={gidx}."
        )

    return gidx


@operation(full_partition=True, output_count=1)
def partition_shape(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> int:
    """Partition one gene into its per-sample shape vectors.

    For a 2D grid this produces:

        array([height, width])

    The operation is full_partition because shape is a complete structural
    partition of the dimensional extent of its source representation.
    """
    source_idx = _validate_source(meta, X, source_idx)

    shapes = [
        np.asarray(np.asarray(value).shape, dtype=int)
        for value in X[source_idx]
    ]

    return _append_gene(
        meta,
        X,
        shapes,
        source=source_idx,
        op_name="partition_shape",
    )


@operation(full_partition=True, output_count=2)
def partition_composite(
    meta: ProgramMeta,
    X: ProgramX,
    source_idx: int,
) -> tuple[int, int]:
    """Partition a categorical 2D grid into color IDs and presence masks.

    Produces two new genes:

      1. sorted 1D array of color IDs used in each sample;
      2. 3D boolean array shaped [num_colors, height, width].

    Presence channel i corresponds to color_ids[i].
    """
    source_idx = _validate_source(meta, X, source_idx)

    color_values = []
    presence_values = []

    for sample_idx, value in enumerate(X[source_idx]):
        array = np.asarray(value)

        if array.ndim != 2:
            raise ValueError(
                "partition_composite requires a 2D source for every sample; "
                f"sample {sample_idx} has shape {array.shape}."
            )
        if array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError(
                "partition_composite does not accept empty spatial dimensions."
            )

        colors = np.unique(array)

        presence = np.stack(
            [array == color for color in colors],
            axis=0,
        ).astype(bool, copy=False)

        color_values.append(colors.copy())
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
