from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

ARC_COLOR_COUNT = 10
INVALID_STATE = ARC_COLOR_COUNT
TRANSITION_STATE_COUNT = ARC_COLOR_COUNT + 1
TRANSITION_LABELS = tuple(range(ARC_COLOR_COUNT)) + ("INVALID",)

ARC_MAX_GRID_DIM = 30
DIMENSION_STATE_COUNT = ARC_MAX_GRID_DIM + 1
DIMENSION_LABELS = tuple(range(DIMENSION_STATE_COUNT))


def _matrix_is_diagonal(matrix: np.ndarray) -> bool:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False

    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    return bool(np.count_nonzero(off_diagonal) == 0)


def _state_is_used(matrix: np.ndarray, state: int) -> bool:
    matrix = np.asarray(matrix)
    return bool(
        np.count_nonzero(matrix[state, :])
        or np.count_nonzero(matrix[:, state])
    )


def _state_is_resolved(matrix: np.ndarray, state: int) -> bool:
    matrix = np.asarray(matrix)

    row = matrix[state, :].copy()
    col = matrix[:, state].copy()
    row[state] = 0
    col[state] = 0

    return bool(
        np.count_nonzero(row) == 0
        and np.count_nonzero(col) == 0
    )


@dataclass
class LossNode:
    """One node in the GP loss-decomposition tree.

    Leaf nodes hold transition matrices instead of scalar loss values.

    For all transition matrices:
      - rows are predicted states
      - columns are target states
      - diagonal entries are resolved transitions
      - off-diagonal entries are unresolved transitions

    Each leaf stores both:
      - matrix: aggregate transitions across every sample
      - sample_matrices: one transition matrix per sample

    solution_idx points to a discovered solution gene in the GP matrix.
    -1 means that no solving gene has been assigned yet.
    """

    name: str
    matrix: np.ndarray | None = None
    sample_matrices: tuple[np.ndarray, ...] = ()
    solution_idx: int = -1
    children: list["LossNode"] = field(default_factory=list)
    predicted: Any = None
    target: Any = None
    labels: tuple[Any, ...] = ()

    @property
    def loss(self) -> np.ndarray | None:
        """Compatibility alias for the aggregate transition matrix."""
        return self.matrix

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def sample_count(self) -> int:
        return len(self.sample_matrices)

    @property
    def sample_resolved(self) -> tuple[bool, ...]:
        if self.children:
            return ()

        return tuple(
            _matrix_is_diagonal(matrix)
            for matrix in self.sample_matrices
        )

    @property
    def resolved_sample_count(self) -> int:
        return sum(self.sample_resolved)

    @property
    def resolved(self) -> bool:
        """Whether this node is resolved across every supplied sample."""
        if self.children:
            return all(child.resolved for child in self.children)

        if self.matrix is None:
            return False

        if self.sample_matrices:
            return all(self.sample_resolved)

        return _matrix_is_diagonal(self.matrix)

    def find(self, name: str) -> "LossNode":
        """Find a uniquely named node anywhere below this node."""
        if self.name == name:
            return self

        for child in self.children:
            try:
                return child.find(name)
            except KeyError:
                pass

        raise KeyError(f"No loss node named {name!r}.")

    def walk(self) -> Iterable["LossNode"]:
        """Depth-first traversal including this node."""
        yield self
        for child in self.children:
            yield from child.walk()

    def sample_matrix(self, sample_idx: int) -> np.ndarray:
        if self.children:
            raise ValueError(
                f"Node {self.name!r} is not a transition-matrix leaf."
            )

        return self.sample_matrices[sample_idx]

    def state_status(
        self,
        state: int,
        *,
        sample_idx: int | None = None,
    ) -> str:
        """Return unused, resolved, or degenerate for one transition state."""
        if self.matrix is None:
            raise ValueError(
                f"Node {self.name!r} does not contain a transition matrix."
            )

        matrix = (
            self.matrix
            if sample_idx is None
            else self.sample_matrix(sample_idx)
        )

        if state < 0 or state >= matrix.shape[0]:
            raise IndexError(
                f"State {state} is outside matrix bounds {matrix.shape}."
            )

        if not _state_is_used(matrix, state):
            return "unused"

        if _state_is_resolved(matrix, state):
            return "resolved"

        return "degenerate"

    def state_resolved(
        self,
        state: int,
        *,
        sample_idx: int | None = None,
    ) -> bool:
        """True when a used state has no off-diagonal transitions."""
        return self.state_status(
            state,
            sample_idx=sample_idx,
        ) == "resolved"

    def degeneracies(
        self,
        state: int | None = None,
        *,
        sample_idx: int | None = None,
    ) -> list[tuple[int, int, int]]:
        """Return non-diagonal transitions as source, target, count tuples.

        If state is supplied, transitions are returned when that state appears
        on either side of the transition. This means a bad 3 -> 7 transition
        marks both state 3 and state 7 as degenerate.
        """
        if self.matrix is None:
            raise ValueError(
                f"Node {self.name!r} does not contain a transition matrix."
            )

        matrix = (
            self.matrix
            if sample_idx is None
            else self.sample_matrix(sample_idx)
        )

        transitions: list[tuple[int, int, int]] = []

        for source, target in np.argwhere(matrix > 0):
            source = int(source)
            target = int(target)

            if source == target:
                continue

            if state is not None and state not in {source, target}:
                continue

            transitions.append(
                (source, target, int(matrix[source, target]))
            )

        return transitions


@dataclass
class LossTree:
    """Hierarchical ARC loss tree across one or more samples.

    Structure:

        root
        ├── shape
        │   ├── h
        │   └── w
        └── composite

    h and w contain dimension transition matrices.
    composite contains categorical color transition matrices.
    """

    root: LossNode

    def __getitem__(self, name: str) -> LossNode:
        return self.root.find(name)

    @property
    def shape(self) -> LossNode:
        return self["shape"]

    @property
    def h(self) -> LossNode:
        return self["h"]

    @property
    def w(self) -> LossNode:
        return self["w"]

    @property
    def composite(self) -> LossNode:
        return self["composite"]

    @property
    def sample_count(self) -> int:
        return self.composite.sample_count

    @property
    def transition_matrix(self) -> np.ndarray:
        """Compatibility alias for aggregate composite transitions."""
        return self.composite.matrix

    @property
    def sample_transition_matrices(self) -> tuple[np.ndarray, ...]:
        return self.composite.sample_matrices

    @property
    def shape_resolved(self) -> bool:
        return self.shape.resolved

    @property
    def composite_resolved(self) -> bool:
        return self.composite.resolved

    @property
    def resolved(self) -> bool:
        return self.root.resolved

    def summary(self) -> dict[str, Any]:
        """Inspectable representation for notebooks and debugging."""
        return {
            "sample_count": self.sample_count,
            "resolved": self.resolved,
            "shape": {
                "resolved": self.shape.resolved,
                "solution_idx": self.shape.solution_idx,
                "h": {
                    "resolved": self.h.resolved,
                    "predicted": np.asarray(self.h.predicted).copy(),
                    "target": np.asarray(self.h.target).copy(),
                    "transition_matrix": self.h.matrix.copy(),
                    "sample_transition_matrices": tuple(
                        matrix.copy()
                        for matrix in self.h.sample_matrices
                    ),
                    "solution_idx": self.h.solution_idx,
                },
                "w": {
                    "resolved": self.w.resolved,
                    "predicted": np.asarray(self.w.predicted).copy(),
                    "target": np.asarray(self.w.target).copy(),
                    "transition_matrix": self.w.matrix.copy(),
                    "sample_transition_matrices": tuple(
                        matrix.copy()
                        for matrix in self.w.sample_matrices
                    ),
                    "solution_idx": self.w.solution_idx,
                },
            },
            "composite": {
                "resolved": self.composite.resolved,
                "transition_matrix": self.composite.matrix.copy(),
                "sample_transition_matrices": tuple(
                    matrix.copy()
                    for matrix in self.composite.sample_matrices
                ),
                "labels": TRANSITION_LABELS,
                "solution_idx": self.composite.solution_idx,
            },
            "root_solution_idx": self.root.solution_idx,
        }


def _validate_arc_grid(grid: Any, *, name: str) -> np.ndarray:
    """Return one validated 2D ARC grid."""
    try:
        array = np.asarray(grid)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric 2D grid.") from exc

    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} dimensions may not be empty.")
    if (
        array.shape[0] > ARC_MAX_GRID_DIM
        or array.shape[1] > ARC_MAX_GRID_DIM
    ):
        raise ValueError(
            f"{name} dimensions may not exceed "
            f"{ARC_MAX_GRID_DIM} x {ARC_MAX_GRID_DIM}."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain numeric ARC color indices.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if not np.all(array == np.floor(array)):
        raise ValueError(f"{name} must contain integer ARC color indices.")
    if np.any((array < 0) | (array >= ARC_COLOR_COUNT)):
        raise ValueError(
            f"{name} ARC colors must be integers in "
            f"[0, {ARC_COLOR_COUNT - 1}]."
        )

    return array.astype(np.int8, copy=False)


def _looks_like_single_grid(value: Any) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False

    return (
        array.ndim == 2
        and np.issubdtype(array.dtype, np.number)
    )


def _normalize_grid_samples(
    value: Any,
    *,
    name: str,
) -> tuple[np.ndarray, ...]:
    """Normalize one grid or a collection of grids into a sample tuple."""
    if _looks_like_single_grid(value):
        return (_validate_arc_grid(value, name=name),)

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            raise ValueError(
                f"{name} must be a grid or a collection of grids."
            )
        raw_samples = list(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_samples = list(value)
    else:
        raise ValueError(
            f"{name} must be a grid or a collection of grids."
        )

    if not raw_samples:
        raise ValueError(f"{name} contains no samples.")

    return tuple(
        _validate_arc_grid(sample, name=f"{name}[{sample_idx}]")
        for sample_idx, sample in enumerate(raw_samples)
    )


def color_transition_matrix(
    yhat: Any,
    y: Any,
) -> np.ndarray:
    """Build one 11x11 predicted-to-target color transition matrix.

    Rows are predicted states and columns are target states.

    States 0..9 are ARC colors. State 10 is INVALID:
      - INVALID -> target color means the target contains a pixel outside yhat
      - predicted color -> INVALID means yhat contains a pixel outside target

    A perfect prediction contains counts only on the normal color diagonal.
    """
    predicted = _validate_arc_grid(yhat, name="yhat")
    target = _validate_arc_grid(y, name="y")

    matrix = np.zeros(
        (TRANSITION_STATE_COUNT, TRANSITION_STATE_COUNT),
        dtype=np.int64,
    )

    predicted_h, predicted_w = predicted.shape
    target_h, target_w = target.shape

    for row in range(max(predicted_h, target_h)):
        for col in range(max(predicted_w, target_w)):
            in_predicted = row < predicted_h and col < predicted_w
            in_target = row < target_h and col < target_w

            if not in_predicted and not in_target:
                continue

            predicted_state = (
                int(predicted[row, col])
                if in_predicted
                else INVALID_STATE
            )
            target_state = (
                int(target[row, col])
                if in_target
                else INVALID_STATE
            )

            matrix[predicted_state, target_state] += 1

    return matrix


def dimension_transition_matrix(
    predicted_dimension: int,
    target_dimension: int,
) -> np.ndarray:
    """Build one dimension transition matrix.

    Dimension value d maps directly to matrix state d. State 0 is retained for
    direct indexing convenience but is unused by valid ARC grids.
    """
    predicted_dimension = int(predicted_dimension)
    target_dimension = int(target_dimension)

    for name, value in (
        ("predicted_dimension", predicted_dimension),
        ("target_dimension", target_dimension),
    ):
        if value < 1 or value > ARC_MAX_GRID_DIM:
            raise ValueError(
                f"{name} must be in [1, {ARC_MAX_GRID_DIM}], got {value}."
            )

    matrix = np.zeros(
        (DIMENSION_STATE_COUNT, DIMENSION_STATE_COUNT),
        dtype=np.int64,
    )
    matrix[predicted_dimension, target_dimension] = 1
    return matrix


def _aggregate_matrices(
    matrices: tuple[np.ndarray, ...],
) -> np.ndarray:
    if not matrices:
        raise ValueError("Cannot aggregate an empty matrix collection.")

    return np.sum(np.stack(matrices, axis=0), axis=0)


def loss_resolution(
    yhat: Any,
    y: Any,
) -> LossTree:
    """Resolve one or more prediction/target pairs into one loss tree.

    yhat and y may each be:
      - one 2D ARC grid
      - a list or tuple of 2D grids
      - a NumPy object array containing grids
      - a stacked 3D NumPy array of equally shaped grids

    This allows direct use with the current GP structure:

        tree = loss_resolution(gp_mat.input, gp_mat.output)

    Across N samples:
      - h stores N per-sample height transitions plus their aggregate matrix
      - w stores N per-sample width transitions plus their aggregate matrix
      - composite stores N per-sample color transitions plus their aggregate

    No scalar total loss is calculated.
    """
    predicted_samples = _normalize_grid_samples(yhat, name="yhat")
    target_samples = _normalize_grid_samples(y, name="y")

    if len(predicted_samples) != len(target_samples):
        raise ValueError(
            "yhat and y must contain the same number of samples: "
            f"{len(predicted_samples)} != {len(target_samples)}."
        )

    predicted_h = np.asarray(
        [grid.shape[0] for grid in predicted_samples],
        dtype=int,
    )
    target_h = np.asarray(
        [grid.shape[0] for grid in target_samples],
        dtype=int,
    )
    predicted_w = np.asarray(
        [grid.shape[1] for grid in predicted_samples],
        dtype=int,
    )
    target_w = np.asarray(
        [grid.shape[1] for grid in target_samples],
        dtype=int,
    )

    h_sample_matrices = tuple(
        dimension_transition_matrix(predicted, target)
        for predicted, target in zip(predicted_h, target_h)
    )
    w_sample_matrices = tuple(
        dimension_transition_matrix(predicted, target)
        for predicted, target in zip(predicted_w, target_w)
    )
    composite_sample_matrices = tuple(
        color_transition_matrix(predicted, target)
        for predicted, target in zip(predicted_samples, target_samples)
    )

    h_node = LossNode(
        name="h",
        matrix=_aggregate_matrices(h_sample_matrices),
        sample_matrices=h_sample_matrices,
        predicted=predicted_h,
        target=target_h,
        labels=DIMENSION_LABELS,
    )
    w_node = LossNode(
        name="w",
        matrix=_aggregate_matrices(w_sample_matrices),
        sample_matrices=w_sample_matrices,
        predicted=predicted_w,
        target=target_w,
        labels=DIMENSION_LABELS,
    )

    shape_node = LossNode(
        name="shape",
        children=[h_node, w_node],
        predicted=np.stack([predicted_h, predicted_w], axis=1),
        target=np.stack([target_h, target_w], axis=1),
    )

    composite_node = LossNode(
        name="composite",
        matrix=_aggregate_matrices(composite_sample_matrices),
        sample_matrices=composite_sample_matrices,
        predicted=predicted_samples,
        target=target_samples,
        labels=TRANSITION_LABELS,
    )

    root = LossNode(
        name="root",
        children=[shape_node, composite_node],
        predicted=predicted_samples,
        target=target_samples,
    )

    return LossTree(root=root)


def _transition_label(node: LossNode, state: int) -> str:
    if node.labels and state < len(node.labels):
        return str(node.labels[state])
    return str(state)


def _format_degeneracies(
    node: LossNode,
    transitions: list[tuple[int, int, int]],
) -> str:
    return ", ".join(
        f"{_transition_label(node, source)} -> "
        f"{_transition_label(node, target)} x{count}"
        for source, target, count in transitions
    )


def inspect_loss(
    tree: LossTree,
    *,
    show_samples: bool = True,
    show_unused_colors: bool = False,
) -> str:
    """Print and return a readable multi-sample loss-tree inspection.

    Every unresolved dimension, color, and INVALID state is marked with X.
    Sample-specific degeneracies are shown beneath the affected node/state.
    """
    if not isinstance(tree, LossTree):
        raise TypeError("inspect_loss expects a LossTree.")

    check = "✓"
    cross = "X"

    def marker(resolved: bool) -> str:
        return check if resolved else cross

    lines = [
        f"{marker(tree.resolved)} root | {tree.sample_count} sample(s)",
        f"├── {marker(tree.shape.resolved)} shape",
    ]

    for position, node in enumerate((tree.h, tree.w)):
        last = position == 1
        branch = "│   └──" if last else "│   ├──"
        continuation = "│       " if last else "│   │   "

        lines.append(
            f"{branch} {marker(node.resolved)} {node.name} | "
            f"{node.resolved_sample_count}/{tree.sample_count} samples resolved"
        )

        aggregate_transitions = node.degeneracies()
        if aggregate_transitions:
            lines.append(
                f"{continuation}X aggregate: "
                f"{_format_degeneracies(node, aggregate_transitions)}"
            )

        if show_samples:
            for sample_idx, is_resolved in enumerate(node.sample_resolved):
                if is_resolved:
                    continue

                sample_transitions = node.degeneracies(
                    sample_idx=sample_idx
                )
                lines.append(
                    f"{continuation}X sample {sample_idx}: "
                    f"{_format_degeneracies(node, sample_transitions)}"
                )

    composite = tree.composite
    lines.append(
        f"└── {marker(composite.resolved)} composite | "
        f"{composite.resolved_sample_count}/{tree.sample_count} "
        "samples resolved"
    )

    color_states = list(range(ARC_COLOR_COUNT))
    if (
        _state_is_used(composite.matrix, INVALID_STATE)
        or show_unused_colors
    ):
        color_states.append(INVALID_STATE)

    for state in color_states:
        status = composite.state_status(state)

        if status == "unused" and not show_unused_colors:
            continue

        label = _transition_label(composite, state)

        if status == "unused":
            lines.append(f"    ├── - {label} | unused")
            continue

        state_ok = status == "resolved"
        lines.append(
            f"    ├── {marker(state_ok)} {label}"
        )

        if not state_ok:
            aggregate_transitions = composite.degeneracies(state)
            lines.append(
                "    │   X aggregate: "
                + _format_degeneracies(
                    composite,
                    aggregate_transitions,
                )
            )

            if show_samples:
                for sample_idx in range(tree.sample_count):
                    sample_status = composite.state_status(
                        state,
                        sample_idx=sample_idx,
                    )
                    if sample_status != "degenerate":
                        continue

                    sample_transitions = composite.degeneracies(
                        state,
                        sample_idx=sample_idx,
                    )
                    lines.append(
                        f"    │   X sample {sample_idx}: "
                        + _format_degeneracies(
                            composite,
                            sample_transitions,
                        )
                    )

    report = "\n".join(lines)
    print(report)
    return report


resolve_loss_tree = loss_resolution
