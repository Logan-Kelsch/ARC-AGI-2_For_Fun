# GP_Set evaluation tree

The GP_Set evaluation tree is a retained decomposition of the exact output
information that the gene matrix will eventually need to explain.

It is deliberately separate from the loss tree.

The loss tree describes observed input-to-output transition structure.

The GP evaluation tree asks a different question:

> Does the current GP_Set contain an exact gene representation of this portion
> of the output across every training sample?

The matching/search function is not implemented yet. This file establishes the
target structure and bookkeeping that function will use.

## Structure

~~~text
root
├── shape
│   ├── h
│   └── w
└── composite
    ├── color_id
    └── color_presence
~~~

The tree is attached directly to every GP_Set created by init_gp_mat:

~~~python
gp_mat = init_gp_mat(task.train)
tree = gp_mat.eval_tree
~~~

## Per-node retained state

Every GP_EvalNode contains:

~~~text
name
target
answer_present = False
gene_idx = -1
children
~~~

target is an outer object array with one exact target value for every training
sample.

Therefore for sample s:

~~~python
node.target[s]
~~~

is the exact output-side representation that a candidate GP gene must reproduce
for that sample.

## Exact across-sample requirement

A future evaluator will consider gene index i to answer a node only when:

~~~text
GP_Set.data[0][i] == node.target[0]
AND
GP_Set.data[1][i] == node.target[1]
AND
...
AND
GP_Set.data[L-1][i] == node.target[L-1]
~~~

using the same recursive exact-equivalence semantics already used by GP_Set.

There is no partial-match or similarity criterion at this level.

If the future evaluator finds an exact match, it can associate that gene:

~~~python
node.mark_answer(i)
~~~

which sets:

~~~text
answer_present = True
gene_idx = i
~~~

The current PR does not perform this search.

## Target representations

### root

The exact output grid for each sample.

~~~python
tree.root.target[s]
~~~

### shape

The output [height, width] vector.

~~~python
tree.shape.target[s]
~~~

This uses the same representation as the current shape-extraction gene.

### h

The output height scalar.

~~~python
tree.h.target[s]
~~~

### w

The output width scalar.

~~~python
tree.w.target[s]
~~~

### composite

The output categorical composition represented as:

~~~text
[
    color_id,
    color_presence,
]
~~~

for each sample.

### color_id

The sorted color IDs used in the output.

~~~python
tree.color_id.target[s]
~~~

This matches the representation produced by the current color-ID partition
operation.

### color_presence

One boolean presence mask per used output color.

~~~python
tree.color_presence.target[s]
~~~

This matches the representation produced by the current color-presence
partition operation.

## Example: output shape equals input shape

The current GP initialization creates an input-shape gene at index 1.

If every demonstration preserves shape, then the representations already align:

~~~python
for s in range(len(gp_mat.data)):
    gp_mat.data[s][1] == gp_mat.eval_tree.shape.target[s]
~~~

conceptually holds exactly across all samples.

A future evaluator will detect that fact and perform:

~~~python
gp_mat.eval_tree.shape.mark_answer(1)
~~~

The tree then retains the fact that the shape portion of the output is exactly
represented by gene index 1.

## Why this is retained inside GP_Set

As GP operations append genes, the interpretation space expands while the
output target decomposition stays fixed.

The evaluation tree therefore becomes persistent bookkeeping for iterative
constraint reduction:

~~~text
GP gene space grows
        |
        v
compare exact gene columns against retained output targets
        |
        v
mark newly represented evaluation nodes
        |
        v
focus future search on output information not yet represented
~~~

The next layer should implement the exact matching and propagation policy. This
foundation intentionally does not decide how a parent node should behave when
its children are individually represented.
