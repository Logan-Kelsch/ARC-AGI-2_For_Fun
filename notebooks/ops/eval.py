from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

ARC_COLOR_COUNT = 10
INVALID_STATE = ARC_COLOR_COUNT
TRANSITION_STATE_COUNT = ARC_COLOR_COUNT + 1


@dataclass(frozen=True)
class DimensionLoss:
    """One terminal output-shape loss component."""

    predicted: int
    target: int

    @property
    def delta(self) -> int:
        """Signed prediction error: predicted - target."""
        return self.predicted - self.target

    @property
    def resolved(self) -> bool:
        return self.predicted == self.target


@dataclass
class LossNode:
    """One node in the hierarchical ARC loss decomposition."""

    name: str
    value: object = None
    solution_idx: int = -1
    children: dict[str, "LossNode"] = field(default_factory=dict)

    def add_child(self, child: "LossNode") -> "LossNode":
        if child.name in self.children:
            raise ValueError(
                f"Loss node {self.name!r} already has child {child.name!r}."
            )
        self.children[child.name] = child
        return child

    def __getitem__(self, name: str) -> "LossNode":
        return self.children[name]

    def walk(self) -> Iterator["LossNode"]:
        """Depth-first traversal including this node."""
        yield self
        for child in self.children.values():
            yield from child.walk()


@dataclass
class LossTree:
    """Rooted decomposition of ARC output error.

    Tree structure:

        root
        |- shape
        |  |- h
        |  '- w
        '- composite

    h and w are terminal DimensionLoss nodes.

    composite.value is an 11x11 transition-count matrix. Rows are predicted
    states and columns are target states. States 0..9 are ARC colors and state
    10 is INVALID, used whenever one grid has a pixel at a coordinate and the
    other grid does not.
    """

    root: LossNode

    def __getitem__(self, name: str) -> LossNode:
        return self.root[name]

    def walk(self) -> Iterator[LossNode]:
        return self.root.walk()

    @property
    def transition_matrix(self) -> np.ndarray:
        return self["composite"].value

    @property
    def shape_resolved(self) -> bool:
        return (
            self["shape"]["h"].value.resolved
            and self["shape"]["w"].value.resolved
        )

    @property
    def composite_resolved(self) -> bool:
        matrix = self.transition_matrix
        diagonal = np.diag(np.diag(matrix))
        return bool(np.array_equal(matrix, diagonal))

    @property
    def resolved(self) -> bool:
        return self.shape_resolved and self.composite_resolved


def _validate_arc_grid(
    grid: np.ndarray | Sequence[Sequence[int]],
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(grid)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D grid, got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} may not have an empty spatial dimension.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain numeric ARC color indices.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if not np.all(array == np.floor(array)):
        raise ValueError(f"{name} colors must be integer-valued.")
    if np.any((array < 0) | (array >= ARC_COLOR_COUNT)):
        raise ValueError(
            f"{name} colors must be in [0, {ARC_COLOR_COUNT - 1}]."
        )

    return array.astype(np.int8, copy=False)


def color_transition_matrix(
    yhat: np.ndarray | Sequence[Sequence[int]],
    y: np.ndarray | Sequence[Sequence[int]],
) -> np.ndarray:
    """Build the categorical predicted-to-target pixel transition matrix.

    Matrix orientation:
        rows    = predicted state
        columns = target state

    The final row/column (index INVALID_STATE == 10) represents the absence of
    a corresponding pixel because the prediction and target shapes differ.

    A perfect prediction therefore contains counts only on the color diagonal.
    """
    predicted = _validate_arc_grid(yhat, name="yhat")
    target = _validate_arc_grid(y, name="y")

    pred_h, pred_w = predicted.shape
    target_h, target_w = target.shape

    matrix = np.zeros(
        (TRANSITION_STATE_COUNT, TRANSITION_STATE_COUNT),
        dtype=np.int64,
    )

    for row in range(max(pred_h, target_h)):
        for col in range(max(pred_w, target_w)):
            pred_exists = row < pred_h and col < pred_w
            target_exists = row < target_h and col < target_w

            if not pred_exists and not target_exists:
                continue

            pred_state = (
                int(predicted[row, col])
                if pred_exists
                else INVALID_STATE
            )
            target_state = (
                int(target[row, col])
                if target_exists
                else INVALID_STATE
            )

            matrix[pred_state, target_state] += 1

    return matrix


def loss_resolution(
    yhat: np.ndarray | Sequence[Sequence[int]],
    y: np.ndarray | Sequence[Sequence[int]],
) -> LossTree:
    """Resolve one prediction/target pair into the foundational loss tree.

    No scalar total-loss value is produced. Error remains decomposed into:
    output height, output width, and categorical pixel transitions.

    Every tree node starts with solution_idx=-1. Later GP search can write the
    gene index that resolves that exact node without changing this evaluator.
    """
    predicted = _validate_arc_grid(yhat, name="yhat")
    target = _validate_arc_grid(y, name="y")

    root = LossNode("root")

    shape = root.add_child(LossNode("shape"))
    shape.add_child(
        LossNode(
            "h",
            value=DimensionLoss(
                predicted=int(predicted.shape[0]),
                target=int(target.shape[0]),
            ),
        )
    )
    shape.add_child(
        LossNode(
            "w",
            value=DimensionLoss(
                predicted=int(predicted.shape[1]),
                target=int(target.shape[1]),
            ),
        )
    )

    root.add_child(
        LossNode(
            "composite",
            value=color_transition_matrix(predicted, target),
        )
    )

    return LossTree(root=root)
