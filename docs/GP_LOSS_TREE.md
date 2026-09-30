# GP loss tree

The GP loss system represents prediction error as a fixed decomposition tree
rather than collapsing all error into one scalar.

```text
root
├── shape
│   ├── h
│   └── w
└── composite
```

Every node has a `solution_idx` field. It defaults to `-1`, meaning no gene
in the GP matrix has yet been identified as the solution for that node.

## Shape

The shape branch is completely decomposed into two terminal nodes:

- `h`: absolute output-height error
- `w`: absolute output-width error

The shape node is resolved only when both are zero.

## Composite categorical error

The composite node contains an 11x11 transition-count matrix.

Rows are target states. Columns are predicted states.

States `0..9` correspond to the ten ARC colors. State `10` is `INVALID`.

For an ordinary overlapping pixel:

```text
matrix[target_color, predicted_color] += 1
```

A correct pixel therefore lands on the diagonal.

If the target contains a coordinate that does not exist in the prediction:

```text
target_color -> INVALID
```

If the prediction contains a coordinate that does not exist in the target:

```text
INVALID -> predicted_color
```

Coordinates outside both matrices are ignored.

A completely resolved composite node has no nonzero off-diagonal entries.

## Resolution

`loss_resolution(yhat, y)` returns a `LossTree`.

Example:

```python
tree = loss_resolution(yhat, y)

tree.h.loss
tree.w.loss
tree.composite.loss

tree.shape.resolved
tree.composite.resolved
tree.resolved
```

No total scalar loss is calculated. The root is resolved only when both the
shape branch and the composite branch are resolved.

When GP search discovers a gene that resolves a node:

```python
tree.composite.solution_idx = gene_idx
```

This keeps search bookkeeping attached directly to the loss location that the
gene solves.
