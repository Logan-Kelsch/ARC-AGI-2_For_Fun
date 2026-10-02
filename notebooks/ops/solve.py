from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from .environment import ProgramX, SolutionTree
from .ops import genes_exactly_equal


@dataclass(frozen=True)
class Scalar1GeneSolution:
    """One exact zero-residual scalar solution."""

    sp_gidx: int
    gp_gidx: int
    rule: str
    params: dict[str, Any]
    target_dtype: np.dtype
    complexity_level: int

    @property
    def expression(self) -> str:
        if self.rule == "identity":
            return "y = x"
        if self.rule == "negate":
            return "y = -x"
        if self.rule == "abs":
            return "y = abs(x)"
        if self.rule == "square":
            return "y = x^2"
        if self.rule == "add_constant":
            return f"y = x + {self.params['c']}"
        if self.rule == "constant_minus_x":
            return f"y = {self.params['c']} - x"
        if self.rule == "scale":
            return f"y = {self.params['c']} * x"
        if self.rule == "affine":
            return (
                f"y = {self.params['a']} * x + "
                f"{self.params['b']}"
            )
        return self.rule


def _scalar_dtype(gene: np.ndarray) -> np.dtype:
    if not isinstance(gene, np.ndarray) or gene.ndim != 1:
        raise ValueError("A scalar gene must be a 1D sample-axis ndarray.")
    if len(gene) == 0:
        raise ValueError("A scalar gene must contain at least one sample.")

    dtypes: set[np.dtype] = set()

    for value in gene:
        array = np.asarray(value)
        if array.ndim != 0:
            raise ValueError(
                "solve_0dim_1gene_basic requires only 0D genes."
            )
        dtypes.add(np.dtype(array.dtype))

    if len(dtypes) != 1:
        raise ValueError(
            "All samples of one scalar gene must share one dtype."
        )

    return next(iter(dtypes))


def _python_scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError("Expected scalar value.")
    return array.item()


def _is_numeric_dtype(dtype: np.dtype) -> bool:
    return bool(
        np.issubdtype(dtype, np.number)
        and not np.issubdtype(dtype, np.bool_)
    )


def _all_integral(values: list[Any]) -> bool:
    return all(
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        for value in values
    )


def _numeric_values(gene: np.ndarray) -> list[Any] | None:
    dtype = _scalar_dtype(gene)

    if not _is_numeric_dtype(dtype):
        return None

    return [_python_scalar(value) for value in gene]


def _as_exact_number(
    value: Any,
    *,
    integral_mode: bool,
):
    if integral_mode:
        return Fraction(int(value), 1)
    return value


def _numbers_equal(a: Any, b: Any) -> bool:
    try:
        result = a == b
    except Exception:
        return False

    if isinstance(result, np.ndarray):
        return bool(np.all(result))

    return bool(result)


def _prediction_matches_target(
    prediction: list[Any],
    target: list[Any],
    target_dtype: np.dtype,
) -> bool:
    if len(prediction) != len(target):
        return False

    for predicted, expected in zip(prediction, target):
        if not _numbers_equal(predicted, expected):
            return False

        # Verify the exact prediction can be represented as the target dtype
        # without the cast changing the target value.
        try:
            cast = np.asarray(predicted).astype(
                target_dtype,
                casting="unsafe",
                copy=False,
            ).item()
        except (TypeError, ValueError, OverflowError):
            return False

        if not _numbers_equal(cast, expected):
            return False

    return True


def _parameter_cost(value: Any) -> tuple[int, int, int]:
    """Small integers/rationals rank ahead of arbitrary floats."""
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return (0, abs(value.numerator), 1)
        return (
            1,
            abs(value.numerator) + value.denominator,
            value.denominator,
        )

    if isinstance(value, (int, np.integer)):
        return (0, abs(int(value)), 1)

    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value).is_integer():
            return (0, abs(int(value)), 1)
        return (2, 0, 0)

    return (3, 0, 0)


def _candidate_rule_solutions(
    x_gene: np.ndarray,
    y_gene: np.ndarray,
) -> list[tuple[int, tuple[Any, ...], str, dict[str, Any]]]:
    """Return exact rule matches ordered by explicit complexity level."""
    target_dtype = _scalar_dtype(y_gene)
    x_values = _numeric_values(x_gene)
    y_values = _numeric_values(y_gene)

    # Non-numeric scalar dtypes only participate in literal identity.
    if x_values is None or y_values is None:
        if all(
            _numbers_equal(
                _python_scalar(x),
                _python_scalar(y),
            )
            for x, y in zip(x_gene, y_gene)
        ):
            return [(0, (), "identity", {})]
        return []

    n = len(y_values)
    integral_mode = (
        _all_integral(x_values)
        and _all_integral(y_values)
    )

    x = [
        _as_exact_number(value, integral_mode=integral_mode)
        for value in x_values
    ]
    y = [
        _as_exact_number(value, integral_mode=integral_mode)
        for value in y_values
    ]

    matches: list[
        tuple[int, tuple[Any, ...], str, dict[str, Any]]
    ] = []

    def add_if_exact(
        level: int,
        rule: str,
        params: dict[str, Any],
        prediction: list[Any],
    ) -> None:
        if _prediction_matches_target(
            prediction,
            y,
            target_dtype,
        ):
            parameter_key = tuple(
                _parameter_cost(value)
                for value in params.values()
            )
            matches.append(
                (
                    level,
                    parameter_key,
                    rule,
                    params,
                )
            )

    add_if_exact(
        0,
        "identity",
        {},
        list(x),
    )
    add_if_exact(
        1,
        "negate",
        {},
        [-value for value in x],
    )
    add_if_exact(
        2,
        "abs",
        {},
        [abs(value) for value in x],
    )
    add_if_exact(
        3,
        "square",
        {},
        [value * value for value in x],
    )

    # One fitted constant is only considered when at least two samples exist.
    if n >= 2:
        c_add = y[0] - x[0]
        add_if_exact(
            4,
            "add_constant",
            {"c": c_add},
            [value + c_add for value in x],
        )

        c_reflect = y[0] + x[0]
        add_if_exact(
            5,
            "constant_minus_x",
            {"c": c_reflect},
            [c_reflect - value for value in x],
        )

        nonzero_idx = next(
            (
                index
                for index, value in enumerate(x)
                if value != 0
            ),
            None,
        )

        if nonzero_idx is not None:
            c_scale = y[nonzero_idx] / x[nonzero_idx]
            add_if_exact(
                6,
                "scale",
                {"c": c_scale},
                [c_scale * value for value in x],
            )

    # Two freely fitted parameters require at least three samples.
    if n >= 3:
        pair = None

        for i in range(n):
            for j in range(i + 1, n):
                if x[i] != x[j]:
                    pair = (i, j)
                    break
            if pair is not None:
                break

        if pair is not None:
            i, j = pair
            a = (y[j] - y[i]) / (x[j] - x[i])
            b = y[i] - a * x[i]

            add_if_exact(
                7,
                "affine",
                {"a": a, "b": b},
                [a * value + b for value in x],
            )

    return matches


def _find_gp_gidx(
    pool_gene: np.ndarray,
    GP_X: ProgramX,
) -> int:
    matches = [
        gidx
        for gidx in range(len(GP_X))
        if genes_exactly_equal(pool_gene, GP_X[gidx])
    ]

    if not matches:
        raise ValueError(
            "A GP_pool gene was not found in GP_X."
        )

    # Earlier GP genes are preferred when exact duplicates somehow exist.
    return min(matches)


def _map_frontier_to_sp_gidx(
    ST_frontier: np.ndarray,
    SP_X: ProgramX,
    ST: SolutionTree,
) -> list[int]:
    frontier_nodes = [
        node_id
        for node_id in ST.unresolved_frontier_nodes()
        if (
            ST[node_id].sp_gidx is not None
            and ST[node_id].dims == 0
        )
    ]

    unused = list(frontier_nodes)
    mapped: list[int] = []

    for frontier_gene in ST_frontier:
        match_pos = None

        for position, node_id in enumerate(unused):
            sp_gidx = ST[node_id].sp_gidx
            if genes_exactly_equal(
                frontier_gene,
                SP_X[sp_gidx],
            ):
                match_pos = position
                break

        if match_pos is None:
            raise ValueError(
                "An ST_frontier gene could not be mapped to an "
                "unresolved 0D ST node."
            )

        node_id = unused.pop(match_pos)
        sp_gidx = ST[node_id].sp_gidx

        if not isinstance(sp_gidx, int):
            raise RuntimeError(
                "Concrete frontier node has no integer SP gene index."
            )

        mapped.append(sp_gidx)

    return mapped


def solve_0dim_1gene_basic(
    GP_pool: np.ndarray,
    ST_frontier: np.ndarray,
    *,
    GP_X: ProgramX,
    SP_X: ProgramX,
    ST: SolutionTree,
) -> list[Scalar1GeneSolution]:
    """Solve unresolved scalar ST targets from one scalar GP gene.

    The function searches an explicit minimum-complexity hierarchy:

        0. y = x
        1. y = -x
        2. y = abs(x)
        3. y = x^2
        4. y = x + c
        5. y = c - x
        6. y = c*x
        7. y = a*x + b

    Every accepted relationship must have exactly zero residual across every
    training sample.

    One-parameter fitted rules require at least two samples. The affine rule
    requires at least three samples and at least two distinct x values.

    The target dtype is preserved: predictions must be exactly equal before
    casting and still exactly equal after casting to the frontier dtype.

    The first complexity level containing an exact solution wins. Within one
    level, simpler fitted constants rank first, followed by lower GP gene index.

    Successful solutions are written into ST using:

        gp_gidx
        solution_rule
        solution_params
    """
    if GP_X.side != "GP":
        raise ValueError(
            "solve_0dim_1gene_basic requires GP_X.side == 'GP'."
        )
    if SP_X.side != "SP":
        raise ValueError(
            "solve_0dim_1gene_basic requires SP_X.side == 'SP'."
        )

    if not isinstance(GP_pool, np.ndarray) or GP_pool.ndim != 1:
        raise ValueError("GP_pool must be a 1D ndarray.")
    if not isinstance(ST_frontier, np.ndarray) or ST_frontier.ndim != 1:
        raise ValueError("ST_frontier must be a 1D ndarray.")

    for gene in GP_pool:
        _scalar_dtype(gene)
    for gene in ST_frontier:
        _scalar_dtype(gene)

    gp_entries = [
        (
            _find_gp_gidx(gene, GP_X),
            gene,
        )
        for gene in GP_pool
    ]

    # Stable low-gidx ordering provides a deterministic tie-breaker and
    # naturally prefers earlier/shallower GP genes when rule complexity ties.
    gp_entries.sort(key=lambda item: item[0])

    target_sp_gidx = _map_frontier_to_sp_gidx(
        ST_frontier,
        SP_X,
        ST,
    )

    solved: list[Scalar1GeneSolution] = []

    for y_gene, sp_gidx in zip(
        ST_frontier,
        target_sp_gidx,
    ):
        # A prior solve can make another snapshot-frontier target irrelevant
        # through Boolean propagation.
        if ST.is_solved(sp_gidx):
            continue

        target_dtype = _scalar_dtype(y_gene)
        exact_candidates = []

        for gp_gidx, x_gene in gp_entries:
            for (
                level,
                parameter_key,
                rule,
                params,
            ) in _candidate_rule_solutions(
                x_gene,
                y_gene,
            ):
                exact_candidates.append(
                    (
                        level,
                        parameter_key,
                        gp_gidx,
                        rule,
                        params,
                    )
                )

        if not exact_candidates:
            continue

        (
            level,
            _,
            gp_gidx,
            rule,
            params,
        ) = min(
            exact_candidates,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            ),
        )

        ST.mark_solution(
            sp_gidx,
            gp_gidx,
            rule=rule,
            params=params,
        )

        solved.append(
            Scalar1GeneSolution(
                sp_gidx=sp_gidx,
                gp_gidx=gp_gidx,
                rule=rule,
                params=dict(params),
                target_dtype=target_dtype,
                complexity_level=level,
            )
        )

    return solved


@dataclass(frozen=True)
class Matrix1GeneSolution:
    """One exact zero-residual 2D solution from one GP gene."""

    sp_gidx: int
    gp_gidx: int
    rule: str
    params: dict[str, Any]
    target_dtype: np.dtype
    complexity_level: int

    @property
    def expression(self) -> str:
        if self.rule == "identity":
            return "Y = X"
        if self.rule == "embed_zero_structural":
            return (
                "Y = embed_zero("
                f"X, position={self.params['position']})"
            )
        if self.rule == "embed_zero_fixed":
            return (
                "Y = embed_zero("
                f"X, row={self.params['row']}, "
                f"col={self.params['col']})"
            )
        if self.rule == "crop_structural":
            return (
                "Y = crop("
                f"X, position={self.params['position']})"
            )
        if self.rule == "crop_fixed":
            return (
                "Y = crop("
                f"X, row={self.params['row']}, "
                f"col={self.params['col']})"
            )
        if self.rule == "shift_zero":
            return (
                "Y = shift_zero("
                f"X, dr={self.params['dr']}, "
                f"dc={self.params['dc']})"
            )
        if self.rule == "tile":
            return (
                "Y = tile("
                f"X, rows={self.params['rows']}, "
                f"cols={self.params['cols']})"
            )
        return self.rule


def _matrix_dtype(gene: np.ndarray) -> np.dtype:
    if not isinstance(gene, np.ndarray) or gene.ndim != 1:
        raise ValueError(
            "A 2D gene must be a 1D sample-axis ndarray."
        )
    if len(gene) == 0:
        raise ValueError(
            "A 2D gene must contain at least one sample."
        )

    dtypes: set[np.dtype] = set()

    for value in gene:
        array = np.asarray(value)

        if array.ndim != 2:
            raise ValueError(
                "solve_2dim_1gene_basic requires only 2D genes."
            )

        dtypes.add(np.dtype(array.dtype))

    if len(dtypes) != 1:
        raise ValueError(
            "All samples of one 2D gene must share one dtype."
        )

    return next(iter(dtypes))


def _cast_matrix_exact(
    value: Any,
    target_dtype: np.dtype,
) -> np.ndarray | None:
    array = np.asarray(value)

    if array.ndim != 2:
        return None

    try:
        cast = array.astype(
            target_dtype,
            casting="unsafe",
            copy=False,
        )
    except (TypeError, ValueError, OverflowError):
        return None

    try:
        equal = np.array_equal(
            cast,
            array,
            equal_nan=True,
        )
    except TypeError:
        equal = np.array_equal(cast, array)

    if not bool(equal):
        return None

    return np.asarray(cast, dtype=target_dtype).copy()


def _matrices_exactly_equal(
    predicted: np.ndarray,
    target: np.ndarray,
) -> bool:
    if (
        predicted.shape != target.shape
        or predicted.dtype != target.dtype
    ):
        return False

    try:
        return bool(
            np.array_equal(
                predicted,
                target,
                equal_nan=True,
            )
        )
    except TypeError:
        return bool(np.array_equal(predicted, target))


_SPATIAL_POSITIONS = (
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "center",
)


def _structural_offset(
    position: str,
    container_shape: tuple[int, int],
    item_shape: tuple[int, int],
) -> tuple[int, int] | None:
    ch, cw = container_shape
    ih, iw = item_shape

    if ih > ch or iw > cw:
        return None

    if position == "top_left":
        return (0, 0)

    if position == "top_right":
        return (0, cw - iw)

    if position == "bottom_left":
        return (ch - ih, 0)

    if position == "bottom_right":
        return (ch - ih, cw - iw)

    if position == "center":
        dh = ch - ih
        dw = cw - iw

        # Only call something "center" when it is unambiguous.
        if dh % 2 != 0 or dw % 2 != 0:
            return None

        return (dh // 2, dw // 2)

    raise ValueError(f"Unknown structural position {position!r}.")


def _embed_zero(
    source: np.ndarray,
    target_shape: tuple[int, int],
    row: int,
    col: int,
    target_dtype: np.dtype,
) -> np.ndarray | None:
    source = _cast_matrix_exact(source, target_dtype)

    if source is None:
        return None

    th, tw = target_shape
    sh, sw = source.shape

    if (
        row < 0
        or col < 0
        or row + sh > th
        or col + sw > tw
    ):
        return None

    result = np.zeros(
        target_shape,
        dtype=target_dtype,
    )
    result[
        row : row + sh,
        col : col + sw,
    ] = source

    return result


def _crop_at(
    source: np.ndarray,
    target_shape: tuple[int, int],
    row: int,
    col: int,
    target_dtype: np.dtype,
) -> np.ndarray | None:
    source = _cast_matrix_exact(source, target_dtype)

    if source is None:
        return None

    th, tw = target_shape
    sh, sw = source.shape

    if (
        row < 0
        or col < 0
        or row + th > sh
        or col + tw > sw
    ):
        return None

    return source[
        row : row + th,
        col : col + tw,
    ].copy()


def _shift_zero(
    source: np.ndarray,
    dr: int,
    dc: int,
    target_dtype: np.dtype,
) -> np.ndarray | None:
    source = _cast_matrix_exact(source, target_dtype)

    if source is None:
        return None

    h, w = source.shape
    result = np.zeros(
        source.shape,
        dtype=target_dtype,
    )

    src_r0 = max(0, -dr)
    src_r1 = min(h, h - dr)
    src_c0 = max(0, -dc)
    src_c1 = min(w, w - dc)

    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return result

    dst_r0 = src_r0 + dr
    dst_r1 = src_r1 + dr
    dst_c0 = src_c0 + dc
    dst_c1 = src_c1 + dc

    result[
        dst_r0:dst_r1,
        dst_c0:dst_c1,
    ] = source[
        src_r0:src_r1,
        src_c0:src_c1,
    ]

    return result


def _tile_matrix(
    source: np.ndarray,
    repeats: tuple[int, int],
    target_dtype: np.dtype,
) -> np.ndarray | None:
    source = _cast_matrix_exact(source, target_dtype)

    if source is None:
        return None

    rows, cols = repeats

    if rows < 1 or cols < 1:
        return None

    return np.tile(source, (rows, cols))


def _all_matrix_predictions_match(
    x_gene: np.ndarray,
    y_gene: np.ndarray,
    builder,
) -> bool:
    target_dtype = _matrix_dtype(y_gene)

    for x_value, y_value in zip(x_gene, y_gene):
        target = np.asarray(
            y_value,
            dtype=target_dtype,
        )
        predicted = builder(
            np.asarray(x_value),
            target,
            target_dtype,
        )

        if predicted is None:
            return False

        if not _matrices_exactly_equal(
            predicted,
            target,
        ):
            return False

    return True


def _candidate_2d_rule_solutions(
    x_gene: np.ndarray,
    y_gene: np.ndarray,
) -> list[
    tuple[
        int,
        tuple[Any, ...],
        str,
        dict[str, Any],
    ]
]:
    """Return exact 2D rules in explicit minimum-complexity order."""
    _matrix_dtype(x_gene)
    target_dtype = _matrix_dtype(y_gene)

    matches: list[
        tuple[
            int,
            tuple[Any, ...],
            str,
            dict[str, Any],
        ]
    ] = []

    def add(
        level: int,
        parameter_key: tuple[Any, ...],
        rule: str,
        params: dict[str, Any],
    ) -> None:
        matches.append(
            (
                level,
                parameter_key,
                rule,
                params,
            )
        )

    # Level 0: literal complete-matrix equality.
    identity_ok = True

    for x_value, y_value in zip(x_gene, y_gene):
        x_cast = _cast_matrix_exact(
            x_value,
            target_dtype,
        )
        target = np.asarray(
            y_value,
            dtype=target_dtype,
        )

        if (
            x_cast is None
            or not _matrices_exactly_equal(
                x_cast,
                target,
            )
        ):
            identity_ok = False
            break

    if identity_ok:
        add(0, (), "identity", {})

    # Level 1: zero-background structural embedding.
    for position_rank, position in enumerate(
        _SPATIAL_POSITIONS
    ):
        if _all_matrix_predictions_match(
            x_gene,
            y_gene,
            lambda x, y, dtype, p=position: (
                None
                if (
                    offset := _structural_offset(
                        p,
                        y.shape,
                        x.shape,
                    )
                ) is None
                else _embed_zero(
                    x,
                    y.shape,
                    offset[0],
                    offset[1],
                    dtype,
                )
            ),
        ):
            add(
                1,
                (position_rank,),
                "embed_zero_structural",
                {"position": position},
            )

    # Level 2: zero-background fixed embedding.
    x0 = np.asarray(x_gene[0])
    y0 = np.asarray(y_gene[0])

    if (
        x0.ndim == 2
        and y0.ndim == 2
        and x0.shape[0] <= y0.shape[0]
        and x0.shape[1] <= y0.shape[1]
    ):
        for row in range(
            y0.shape[0] - x0.shape[0] + 1
        ):
            for col in range(
                y0.shape[1] - x0.shape[1] + 1
            ):
                if _all_matrix_predictions_match(
                    x_gene,
                    y_gene,
                    lambda x, y, dtype, r=row, c=col: (
                        _embed_zero(
                            x,
                            y.shape,
                            r,
                            c,
                            dtype,
                        )
                    ),
                ):
                    add(
                        2,
                        (
                            abs(row) + abs(col),
                            row,
                            col,
                        ),
                        "embed_zero_fixed",
                        {
                            "row": row,
                            "col": col,
                        },
                    )

    # Level 3: structural crop.
    for position_rank, position in enumerate(
        _SPATIAL_POSITIONS
    ):
        if _all_matrix_predictions_match(
            x_gene,
            y_gene,
            lambda x, y, dtype, p=position: (
                None
                if (
                    offset := _structural_offset(
                        p,
                        x.shape,
                        y.shape,
                    )
                ) is None
                else _crop_at(
                    x,
                    y.shape,
                    offset[0],
                    offset[1],
                    dtype,
                )
            ),
        ):
            add(
                3,
                (position_rank,),
                "crop_structural",
                {"position": position},
            )

    # Level 4: fixed crop.
    if (
        x0.ndim == 2
        and y0.ndim == 2
        and y0.shape[0] <= x0.shape[0]
        and y0.shape[1] <= x0.shape[1]
    ):
        for row in range(
            x0.shape[0] - y0.shape[0] + 1
        ):
            for col in range(
                x0.shape[1] - y0.shape[1] + 1
            ):
                if _all_matrix_predictions_match(
                    x_gene,
                    y_gene,
                    lambda x, y, dtype, r=row, c=col: (
                        _crop_at(
                            x,
                            y.shape,
                            r,
                            c,
                            dtype,
                        )
                    ),
                ):
                    add(
                        4,
                        (
                            abs(row) + abs(col),
                            row,
                            col,
                        ),
                        "crop_fixed",
                        {
                            "row": row,
                            "col": col,
                        },
                    )

    # Level 5: same-canvas fixed translation with zero fill.
    if x0.shape == y0.shape:
        h0, w0 = x0.shape

        for dr in range(-(h0 - 1), h0):
            for dc in range(-(w0 - 1), w0):
                if dr == 0 and dc == 0:
                    continue

                if _all_matrix_predictions_match(
                    x_gene,
                    y_gene,
                    lambda x, y, dtype, r=dr, c=dc: (
                        None
                        if x.shape != y.shape
                        else _shift_zero(
                            x,
                            r,
                            c,
                            dtype,
                        )
                    ),
                ):
                    add(
                        5,
                        (
                            abs(dr) + abs(dc),
                            abs(dr),
                            abs(dc),
                            dr,
                            dc,
                        ),
                        "shift_zero",
                        {
                            "dr": dr,
                            "dc": dc,
                        },
                    )

    # Level 6: fixed integer tiling factors.
    if (
        x0.shape[0] > 0
        and x0.shape[1] > 0
        and y0.shape[0] % x0.shape[0] == 0
        and y0.shape[1] % x0.shape[1] == 0
    ):
        rows = y0.shape[0] // x0.shape[0]
        cols = y0.shape[1] // x0.shape[1]

        if (rows, cols) != (1, 1):
            if _all_matrix_predictions_match(
                x_gene,
                y_gene,
                lambda x, y, dtype, r=rows, c=cols: (
                    None
                    if (
                        y.shape[0]
                        != x.shape[0] * r
                        or y.shape[1]
                        != x.shape[1] * c
                    )
                    else _tile_matrix(
                        x,
                        (r, c),
                        dtype,
                    )
                ),
            ):
                add(
                    6,
                    (
                        rows * cols,
                        rows + cols,
                        rows,
                        cols,
                    ),
                    "tile",
                    {
                        "rows": rows,
                        "cols": cols,
                    },
                )

    return matches


def _map_2d_frontier_to_sp_gidx(
    ST_frontier: np.ndarray,
    SP_X: ProgramX,
    ST: SolutionTree,
) -> list[int]:
    frontier_nodes = [
        node_id
        for node_id in ST.unresolved_frontier_nodes()
        if (
            ST[node_id].sp_gidx is not None
            and ST[node_id].dims == 2
        )
    ]

    unused = list(frontier_nodes)
    mapped: list[int] = []

    for frontier_gene in ST_frontier:
        match_pos = None

        for position, node_id in enumerate(unused):
            sp_gidx = ST[node_id].sp_gidx

            if genes_exactly_equal(
                frontier_gene,
                SP_X[sp_gidx],
            ):
                match_pos = position
                break

        if match_pos is None:
            raise ValueError(
                "An ST_frontier gene could not be mapped to an "
                "unresolved 2D ST node."
            )

        node_id = unused.pop(match_pos)
        sp_gidx = ST[node_id].sp_gidx

        if not isinstance(sp_gidx, int):
            raise RuntimeError(
                "Concrete frontier node has no integer SP gene index."
            )

        mapped.append(sp_gidx)

    return mapped


def solve_2dim_1gene_basic(
    GP_pool: np.ndarray,
    ST_frontier: np.ndarray,
    *,
    GP_X: ProgramX,
    SP_X: ProgramX,
    ST: SolutionTree,
) -> list[Matrix1GeneSolution]:
    """Solve unresolved 2D ST targets from one 2D GP gene.

    Exact minimum-complexity hierarchy:

        0. Y = X
        1. zero-background structural embed
           top-left / top-right / bottom-left / bottom-right / center
        2. zero-background fixed embed(row, col)
        3. structural crop
        4. fixed crop(row, col)
        5. fixed translation shift_zero(dr, dc)
        6. fixed integer tile(rows, cols)

    A candidate is accepted only when it reconstructs the complete target
    exactly for every training sample.

    "Contains X somewhere" is therefore not enough: embedding only solves Y
    when all cells outside the embedded X are exactly zero / False.

    Target dtype is retained. Source matrices may only be cast to the target
    dtype when that cast preserves every value exactly.

    Successful solutions are immediately written into ST with gp_gidx,
    solution_rule, and solution_params.
    """
    if GP_X.side != "GP":
        raise ValueError(
            "solve_2dim_1gene_basic requires GP_X.side == 'GP'."
        )
    if SP_X.side != "SP":
        raise ValueError(
            "solve_2dim_1gene_basic requires SP_X.side == 'SP'."
        )

    if (
        not isinstance(GP_pool, np.ndarray)
        or GP_pool.ndim != 1
    ):
        raise ValueError("GP_pool must be a 1D ndarray.")

    if (
        not isinstance(ST_frontier, np.ndarray)
        or ST_frontier.ndim != 1
    ):
        raise ValueError(
            "ST_frontier must be a 1D ndarray."
        )

    for gene in GP_pool:
        _matrix_dtype(gene)

    for gene in ST_frontier:
        _matrix_dtype(gene)

    gp_entries = [
        (
            _find_gp_gidx(gene, GP_X),
            gene,
        )
        for gene in GP_pool
    ]
    gp_entries.sort(key=lambda item: item[0])

    target_sp_gidx = _map_2d_frontier_to_sp_gidx(
        ST_frontier,
        SP_X,
        ST,
    )

    solved: list[Matrix1GeneSolution] = []

    for y_gene, sp_gidx in zip(
        ST_frontier,
        target_sp_gidx,
    ):
        if ST.is_solved(sp_gidx):
            continue

        target_dtype = _matrix_dtype(y_gene)
        exact_candidates = []

        for gp_gidx, x_gene in gp_entries:
            for (
                level,
                parameter_key,
                rule,
                params,
            ) in _candidate_2d_rule_solutions(
                x_gene,
                y_gene,
            ):
                exact_candidates.append(
                    (
                        level,
                        parameter_key,
                        gp_gidx,
                        rule,
                        params,
                    )
                )

        if not exact_candidates:
            continue

        (
            level,
            _,
            gp_gidx,
            rule,
            params,
        ) = min(
            exact_candidates,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            ),
        )

        ST.mark_solution(
            sp_gidx,
            gp_gidx,
            rule=rule,
            params=params,
        )

        solved.append(
            Matrix1GeneSolution(
                sp_gidx=sp_gidx,
                gp_gidx=gp_gidx,
                rule=rule,
                params=dict(params),
                target_dtype=target_dtype,
                complexity_level=level,
            )
        )

    return solved
