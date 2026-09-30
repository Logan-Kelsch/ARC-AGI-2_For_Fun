from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

ARC_COLOR_COUNT = 10
INVALID_STATE = ARC_COLOR_COUNT
TRANSITION_STATE_COUNT = ARC_COLOR_COUNT + 1
TRANSITION_LABELS = tuple(range(ARC_COLOR_COUNT)) + ("INVALID",)


@dataclass
class LossNode:
    """One node in the discrete GP loss-decomposition tree.

    loss is intentionally heterogeneous:
      - root / shape: None because their loss is fully decomposed into children
      - h / w: integer absolute dimension error
      - composite: 11x11 categorical transition-count matrix

    solution_idx points to a discovered solution gene in the GP matrix.
    -1 means that no solving gene has been assigned yet.
    """

    name: str
    loss: Any = None
    solution_idx: int = -1
    children: list["LossNode"] = field(default_factory=list)
    predicted: Any = None
    target: Any = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def resolved(self) -> bool:
        """Whether this node's loss is completely resolved."""
        if self.children:
            return all(child.resolved for child in self.children)

        if self.name in {"h", "w"}:
            return int(self.loss) == 0

        if self.name == "composite":
            matrix = np.asarray(self.loss)
            if matrix.shape != (
                TRANSITION_STATE_COUNT,
                TRANSITION_STATE_COUNT,
            ):
                return False

            off_diagonal = matrix.copy()
            np.fill_diagonal(off_diagonal, 0)
            return bool(np.count_nonzero(off_diagonal) == 0)

        return self.loss in (None, 0)

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


@dataclass
class LossTree:
    """Fixed loss tree for comparing one ARC prediction to one target."""

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
    def resolved(self) -> bool:
        return self.root.resolved

    def summary(self) -> dict[str, Any]:
        """Small inspectable representation for notebooks/debugging."""
        return {
            "resolved": self.resolved,
            "shape": {
                "resolved": self.shape.resolved,
                "solution_idx": self.shape.solution_idx,
                "h": {
                    "predicted": self.h.predicted,
                    "target": self.h.target,
                    "loss": self.h.loss,
                    "solution_idx": self.h.solution_idx,
                },
                "w": {
                    "predicted": self.w.predicted,
                    "target": self.w.target,
                    "loss": self.w.loss,
                    "solution_idx": self.w.solution_idx,
                },
            },
            "composite": {
                "resolved": self.composite.resolved,
                "solution_idx": self.composite.solution_idx,
                "transition_matrix": self.composite.loss.copy(),
                "labels": TRANSITION_LABELS,
            },
            "root_solution_idx": self.root.solution_idx,
        }


def _validate_arc_grid(grid: Any, *, name: str) -> np.ndarray:
    """Return a 2D integer ARC color grid."""
    try:
        array = np.asarray(grid, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric 2D grid.") from exc

    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} dimensions may not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if not np.all(array == np.floor(array)):
        raise ValueError(f"{name} must contain integer ARC color indices.")
    if np.any((array < 0) | (array >= ARC_COLOR_COUNT)):
        raise ValueError(
            f"{name} ARC colors must be integers in [0, {ARC_COLOR_COUNT - 1}]."
        )

    return array.astype(np.int8)


def color_transition_matrix(
    yhat: Any,
    y: Any,
) -> np.ndarray:
    """Build the 11x11 target-to-prediction categorical transition matrix.

    Rows are target states and columns are predicted states.

    States 0..9 are normal ARC colors. State 10 is INVALID:
      - target color to INVALID: target pixel has no predicted coordinate
      - INVALID to predicted color: prediction has a pixel outside target shape

    Coordinates outside both grids are ignored.

    A perfect prediction therefore has counts only on the 0..9 diagonal and
    no transitions involving INVALID.
    """
    predicted = _validate_arc_grid(yhat, name="yhat")
    target = _validate_arc_grid(y, name="y")

    matrix = np.zeros(
        (TRANSITION_STATE_COUNT, TRANSITION_STATE_COUNT),
        dtype=np.int64,
    )

    predicted_h, predicted_w = predicted.shape
    target_h, target_w = target.shape

    max_h = max(predicted_h, target_h)
    max_w = max(predicted_w, target_w)

    for row in range(max_h):
        for col in range(max_w):
            in_target = row < target_h and col < target_w
            in_predicted = row < predicted_h and col < predicted_w

            if not in_target and not in_predicted:
                continue

            target_state = (
                int(target[row, col])
                if in_target
                else INVALID_STATE
            )
            predicted_state = (
                int(predicted[row, col])
                if in_predicted
                else INVALID_STATE
            )

            matrix[target_state, predicted_state] += 1

    return matrix


def loss_resolution(
    yhat: Any,
    y: Any,
) -> LossTree:
    """Resolve prediction error into the fixed GP loss tree.

    Tree structure:

        root
        |- shape
        |  |- h
        |  \- w
        \- composite

    h and w store absolute dimension errors.
    composite stores the categorical transition matrix.

    No scalar total loss is calculated. Parent nodes are resolved only when
    all of their child nodes are resolved.
    """
    predicted = _validate_arc_grid(yhat, name="yhat")
    target = _validate_arc_grid(y, name="y")

    predicted_h, predicted_w = predicted.shape
    target_h, target_w = target.shape

    h_node = LossNode(
        name="h",
        loss=abs(predicted_h - target_h),
        predicted=predicted_h,
        target=target_h,
    )
    w_node = LossNode(
        name="w",
        loss=abs(predicted_w - target_w),
        predicted=predicted_w,
        target=target_w,
    )

    shape_node = LossNode(
        name="shape",
        children=[h_node, w_node],
        predicted=np.array([predicted_h, predicted_w], dtype=int),
        target=np.array([target_h, target_w], dtype=int),
    )

    composite_node = LossNode(
        name="composite",
        loss=color_transition_matrix(predicted, target),
        predicted=predicted,
        target=target,
    )

    root = LossNode(
        name="root",
        children=[shape_node, composite_node],
        predicted=predicted,
        target=target,
    )

    return LossTree(root=root)


resolve_loss_tree = loss_resolution
