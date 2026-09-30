# GP loss tree

The GP loss system represents prediction error as a fixed decomposition tree
rather than collapsing all error into one scalar.

~~~text
root
├── shape
│   ├── h
│   └── w
└── composite
~~~

Every node has a solution_idx field. It defaults to -1, meaning no gene in
the GP matrix has yet been identified as the solution for that node.

The evaluator accepts either one prediction/target pair or multiple samples at
once. The current GP matrix can therefore be inspected directly:

~~~python
gp_mat = init_gp_mat(task.train)
tree = loss_resolution(gp_mat.input, gp_mat.output)
~~~

## Transition convention

All leaf transition matrices use the same orientation:

~~~text
rows    = predicted state
columns = target state
~~~

Diagonal entries are resolved transitions. Any off-diagonal entry is an
unresolved transition.

Every leaf stores:

~~~text
matrix             aggregate transitions across all samples
sample_matrices    one transition matrix per sample
~~~

This preserves both the task-wide overlap and the exact demonstration in which
a degeneracy occurs.

## Shape

Shape is fully decomposed into separate h and w leaves.

Each dimension has a transition matrix indexed directly by dimension value.
For example:

~~~text
h.matrix[3, 3] += 1    predicted height 3 and target height 3
h.matrix[3, 5] += 1    predicted height 3 but target height 5
~~~

The h node is resolved only when every height transition across every sample is
diagonal. The same rule applies independently to w.

Per-sample shape matrices are available as:

~~~python
tree.h.sample_matrices[i]
tree.w.sample_matrices[i]
~~~

## Composite categorical error

The composite node contains an 11x11 categorical transition matrix.

States 0..9 correspond to ARC colors. State 10 is INVALID.

For an ordinary overlapping pixel:

~~~text
matrix[predicted_color, target_color] += 1
~~~

If the target contains a coordinate that does not exist in the prediction:

~~~text
INVALID -> target_color
~~~

If the prediction contains a coordinate that does not exist in the target:

~~~text
predicted_color -> INVALID
~~~

The aggregate matrix is:

~~~python
tree.composite.matrix
~~~

A specific sample is:

~~~python
tree.composite.sample_matrices[i]
~~~

A completely resolved composite node contains no off-diagonal transitions in
any sample.

## Per-color resolution

A color is considered resolved only when no off-diagonal transition involving
that color exists on either side of the transition.

For example, both of these make color 3 unresolved:

~~~text
3 -> 7
2 -> 3
~~~

This prevents a color from being marked solved merely because its own predicted
row is clean while another color is incorrectly transitioning into it.

Useful checks:

~~~python
tree.composite.state_status(3)
tree.composite.state_status(3, sample_idx=1)
tree.composite.degeneracies(3)
tree.composite.degeneracies(3, sample_idx=1)
~~~

state_status returns one of:

~~~text
unused
resolved
degenerate
~~~

## Inspection

~~~python
inspect_loss(tree)
~~~

prints a readable task-wide view.

Resolved components receive a check mark. Every unresolved dimension, color,
or INVALID state receives X, followed by aggregate and sample-specific
off-diagonal transitions.

## Resolution

The full task is solved only when:

~~~text
h resolved across every sample
AND
w resolved across every sample
AND
composite resolved across every sample
~~~

No total scalar loss is calculated.

When GP search discovers a gene that resolves a node:

~~~python
tree.h.solution_idx = gene_idx
tree.composite.solution_idx = gene_idx
~~~

This keeps search bookkeeping attached directly to the loss location that the
gene solves.
