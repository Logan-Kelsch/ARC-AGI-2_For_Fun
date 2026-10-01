from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class InverseOperationInfo:
    """Metadata for one reverse/reconstruction operation."""

    name: str
    forward_op: str
    reconstructive: bool
    func: Callable


INV_OP_REGISTRY: dict[str, InverseOperationInfo] = {}


def inverse_operation(
    forward_op: str,
    *,
    reconstructive: bool = True,
):
    """Register the inverse/relation-checker for one forward operation."""
    forward_op = str(forward_op)

    def decorator(func: Callable):
        info = InverseOperationInfo(
            name=func.__name__,
            forward_op=forward_op,
            reconstructive=bool(reconstructive),
            func=func,
        )
        INV_OP_REGISTRY[func.__name__] = info

        @wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)

        wrapped.inverse_info = info
        return wrapped

    return decorator


def _copy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.copy() if hasattr(value, "copy") else value
    return value


def _as_shape(shape: Any) -> tuple[int, ...]:
    array = np.asarray(shape, dtype=np.int64)

    if array.ndim != 1:
        raise ValueError("shape must be a 1D vector.")
    if np.any(array < 0):
        raise ValueError("shape values must be non-negative.")

    return tuple(int(value) for value in array)


@inverse_operation("partition_shape")
def inv_partition_shape(
    h: Any,
    w: Any,
    composite: Any,
):
    """Validate that reconstructed composite matches solved h and w."""
    expected = (
        int(np.asarray(h).item()),
        int(np.asarray(w).item()),
    )
    composite_array = np.asarray(composite)

    if composite_array.shape != expected:
        raise ValueError(
            f"Composite shape {composite_array.shape} does not match "
            f"solved shape {expected}."
        )

    return composite_array.copy()


@inverse_operation("indiv_1dim")
def inv_indiv_1dim(
    *values: Any,
) -> np.ndarray:
    """Reassemble scalar element genes into their original 1D array."""
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = tuple(values[0])

    if not values:
        raise ValueError("inv_indiv_1dim requires at least one value.")

    scalars = []

    for value in values:
        array = np.asarray(value)

        if array.ndim != 0:
            raise ValueError(
                "inv_indiv_1dim expects scalar element values."
            )

        scalars.append(array.item())

    return np.asarray(scalars)


@inverse_operation("partition_composite")
def inv_partition_composite(
    *color_parts: Any,
) -> np.ndarray:
    """Reassemble alternating color-ID / boolean-mask parts into int64 grid.

    Accepts:

        inv_partition_composite(color0, mask0, color1, mask1, ...)

    All masks must share one shape. Every spatial cell must belong to exactly
    one color mask.
    """
    if len(color_parts) == 1 and isinstance(
        color_parts[0],
        (list, tuple),
    ):
        color_parts = tuple(color_parts[0])

    if len(color_parts) < 2 or len(color_parts) % 2 != 0:
        raise ValueError(
            "Composite inverse requires alternating color ID / mask pairs."
        )

    ids = []
    masks = []

    for index in range(0, len(color_parts), 2):
        color = np.asarray(color_parts[index])

        if color.ndim != 0:
            raise ValueError("Color IDs must be scalar values.")

        mask = np.asarray(color_parts[index + 1], dtype=bool)

        if mask.ndim != 2:
            raise ValueError("Composite presence masks must be 2D.")

        ids.append(np.int64(color.item()))
        masks.append(mask)

    shape = masks[0].shape

    if any(mask.shape != shape for mask in masks):
        raise ValueError("All composite masks must have identical shape.")

    coverage = np.zeros(shape, dtype=np.int64)

    for mask in masks:
        coverage += mask.astype(np.int64)

    if not np.all(coverage == 1):
        raise ValueError(
            "Composite masks must cover every cell exactly once."
        )

    result = np.zeros(shape, dtype=np.int64)

    for color, mask in zip(ids, masks):
        result[mask] = color

    return result


@inverse_operation("bool_complement")
def inv_bool_complement(value: Any) -> np.ndarray:
    """Boolean complement is self-inverse."""
    return np.logical_not(np.asarray(value, dtype=bool))


@inverse_operation("mat2_cwrotate")
def inv_mat2_cwrotate(value: Any) -> np.ndarray:
    """Reverse a clockwise 90-degree rotation."""
    array = np.asarray(value)

    if array.ndim != 2:
        raise ValueError("inv_mat2_cwrotate requires a 2D value.")

    return np.rot90(array, k=1).copy()


@inverse_operation("dim0_flip")
def inv_dim0_flip(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim <= 0:
        raise ValueError("inv_dim0_flip requires dims > 0.")
    return np.flip(array, axis=0).copy()


@inverse_operation("dim1_flip")
def inv_dim1_flip(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim <= 1:
        raise ValueError("inv_dim1_flip requires dims > 1.")
    return np.flip(array, axis=1).copy()


@inverse_operation("dim2_flip")
def inv_dim2_flip(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim <= 2:
        raise ValueError("inv_dim2_flip requires dims > 2.")
    return np.flip(array, axis=2).copy()


@inverse_operation("partition_bool_trim")
def inv_partition_bool_trim(
    offset: Any,
    trimmed: Any,
    shape: Any,
) -> np.ndarray:
    """Reconstruct trimmed boolean data when the original shape is known."""
    offset = np.asarray(offset, dtype=np.int64)
    trimmed = np.asarray(trimmed, dtype=bool)
    shape = _as_shape(shape)

    if offset.ndim != 1:
        raise ValueError("offset must be a 1D vector.")
    if len(offset) != len(shape):
        raise ValueError("offset dimensionality must match shape.")
    if trimmed.ndim != len(shape):
        raise ValueError("trimmed dimensionality must match shape.")

    stops = offset + np.asarray(trimmed.shape, dtype=np.int64)

    if np.any(offset < 0) or np.any(stops > np.asarray(shape)):
        raise ValueError("trimmed data does not fit inside requested shape.")

    result = np.zeros(shape, dtype=bool)
    slices = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(offset, stops)
    )
    result[slices] = trimmed

    return result


def _axis_adjacent_cavity_mask(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=bool)

    if array.ndim == 0:
        return np.asarray(False, dtype=bool)

    false_mask = ~array
    reachable = np.zeros(array.shape, dtype=bool)
    stack: list[tuple[int, ...]] = []

    for axis in range(array.ndim):
        for boundary_index in (0, array.shape[axis] - 1):
            slicer = [slice(None)] * array.ndim
            slicer[axis] = boundary_index

            for reduced_coord in np.argwhere(false_mask[tuple(slicer)]):
                coord = []
                reduced_pos = 0

                for dim in range(array.ndim):
                    if dim == axis:
                        coord.append(boundary_index)
                    else:
                        coord.append(int(reduced_coord[reduced_pos]))
                        reduced_pos += 1

                coord = tuple(coord)

                if not reachable[coord]:
                    reachable[coord] = True
                    stack.append(coord)

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

                neighbor = tuple(neighbor)

                if false_mask[neighbor] and not reachable[neighbor]:
                    reachable[neighbor] = True
                    stack.append(neighbor)

    return false_mask & ~reachable


# NULL-partition operations are not used by ST to prove a parent. Their inverse
# functions are relation verifiers so every registered forward operation still
# has an explicit reverse/checking companion.


@inverse_operation("bool_sum", reconstructive=False)
def inv_bool_sum(
    count: Any,
    source: Any,
):
    source_array = np.asarray(source, dtype=bool)
    expected = np.int64(np.count_nonzero(source_array))

    if np.int64(count) != expected:
        raise ValueError("bool_sum inverse check failed.")

    return _copy(source)


@inverse_operation("bool_cavity", reconstructive=False)
def inv_bool_cavity(
    cavity: Any,
    source: Any,
):
    expected = _axis_adjacent_cavity_mask(
        np.asarray(source, dtype=bool)
    )

    if not np.array_equal(
        np.asarray(cavity, dtype=bool),
        expected,
    ):
        raise ValueError("bool_cavity inverse check failed.")

    return _copy(source)


@inverse_operation("bool2_union", reconstructive=False)
def inv_bool2_union(
    result: Any,
    left: Any,
    right: Any,
):
    expected = np.logical_or(
        np.asarray(left, dtype=bool),
        np.asarray(right, dtype=bool),
    )

    if not np.array_equal(
        np.asarray(result, dtype=bool),
        expected,
    ):
        raise ValueError("bool2_union inverse check failed.")

    return _copy(left), _copy(right)


@inverse_operation("bool2_intersect", reconstructive=False)
def inv_bool2_intersect(
    result: Any,
    left: Any,
    right: Any,
):
    expected = np.logical_and(
        np.asarray(left, dtype=bool),
        np.asarray(right, dtype=bool),
    )

    if not np.array_equal(
        np.asarray(result, dtype=bool),
        expected,
    ):
        raise ValueError("bool2_intersect inverse check failed.")

    return _copy(left), _copy(right)
