# GP_Set Evaluation Tree — Legacy Reference

> **Legacy architecture:** this document describes the earlier GP_Set evaluation-tree design. It is retained for historical/reference purposes and is **not** the current solving path.
>
> The current architecture uses GP_meta / GP_X, SP_meta / SP_X, the Boolean SolutionTree, exact typed frontier matching, and Kelschinator program distillation. See docs/ARCHITECTURE.md and docs/GP_SP_ENVIRONMENT.md.

---

The GP_Set evaluation tree is persistent bookkeeping for incrementally
constricting the output solution space as the gene matrix grows.

It is separate from the descriptive loss tree.

The evaluation tree asks:

> Which exact portions of the output are already represented somewhere in the
> GP_Set across every training sample, in what order were they discovered, and
> which portions remain available for future discovery?

The function that automatically finds these answers is intentionally not built
yet. This module only establishes the retained state and mutation rules that
future evaluator will use.

## Base structure

~~~text
root
├── shape
│   ├── h
│   └── w
└── composite
    ├── color_id          pool
    └── color_presence    pool
~~~

Every GP_Set created by init_gp_mat retains:

~~~python
gp_mat.eval_tree
~~~

## Node state

Every GP_EvalNode contains:

~~~text
name
target
answer_present = False
gene_idx = -1
solution_number = -1
terminal = False
children
~~~

For partitionable nodes it may also contain:

~~~text
remaining_target
is_pool
~~~

target always preserves the original output-side target.

remaining_target is mutable search state. Extracting a solved subset reduces
remaining_target without changing target.

## Exact across-sample target

Each node target is indexed by training sample:

~~~python
node.target[s]
~~~

A future evaluator should only claim an exact node/subset when one GP
representation accounts for that target across every sample according to the
project's exact heterogeneous equality semantics.

## Chronological solutions

The tree numbers discoveries globally, beginning at 1.

A whole retained node can be recorded with:

~~~python
tree.mark_terminal_solution("shape", gene_idx=1)
~~~

After the first discovery:

~~~text
shape.answer_present   True
shape.gene_idx         1
shape.solution_number  1
shape.terminal         True
~~~

The next discovered node or subset receives solution_number 2, then 3, and so
on.

Chronological history is available through:

~~~python
tree.terminal_nodes
~~~

which is ordered by solution_number.

## Terminal semantics

terminal means:

> this exact node/subset has already been accounted for and must not be offered
> as an independent target to later evaluation passes.

It does not delete the retained structure.

For example, if shape is terminal:

~~~text
root
├── shape  # terminal
│   ├── h
│   └── w
└── composite
~~~

then shape, h, and w are omitted from:

~~~python
tree.searchable_nodes()
~~~

because the shape branch has already been accounted for.

The parent root remains searchable. A future evaluator may still discover one
gene/program that represents the complete output.

Likewise, solving a subset under composite does not make composite terminal.
The complete composite representation can still be tested later.

## Color pools

color_id and color_presence are initialized as aligned pools.

Each has:

~~~python
node.target
node.remaining_target
~~~

At initialization, remaining_target is an independent copy of target.

Suppose black (ARC color 0) is discovered as one exact output subset. The
future evaluator can record that discovery with:

~~~python
terminal = tree.extract_color_terminal(
    color=0,
    gene_idx=i,
)
~~~

This does three things.

First, it creates a new retained terminal beneath composite:

~~~text
composite
├── color_id
├── color_presence
└── color_0
~~~

The new color_0 target contains both the color ID and the associated boolean
presence channel for each training sample.

Second, it assigns the next global solution number:

~~~text
color_0.answer_present   True
color_0.gene_idx         i
color_0.solution_number  2   # if shape was discovered first
color_0.terminal         True
~~~

Third, it removes color 0 from:

~~~python
tree.color_id.remaining_target
tree.color_presence.remaining_target
~~~

for every sample where it occurs.

The original full targets remain untouched:

~~~python
tree.color_id.target
tree.color_presence.target
tree.composite.target
~~~

so the complete composite can still be evaluated later.

## Absence across samples

A color subset can exist in some demonstrations and be absent in others.

The extracted terminal preserves that exact across-sample condition.

For a sample containing the color:

~~~text
[
    array([color]),
    one-channel presence mask
]
~~~

For a sample without the color:

~~~text
[
    empty color array,
    empty presence-channel array
]
~~~

Thus absence is represented explicitly rather than dropping the sample.

## Generic subset terminals

Colors are the first concrete pooled decomposition, but the mechanism is not
restricted to color.

Any later evaluation layer can retain a newly discovered subset beneath a
non-terminal parent:

~~~python
tree.add_terminal_subset(
    "composite",
    name="some_subset",
    target=per_sample_target,
    gene_idx=i,
    source_nodes=("some_pool",),
)
~~~

The new node receives the next solution number and becomes terminal while the
parent remains searchable.

This is the intended foundation for future decompositions of composite,
objects, relations, spatial regions, or other output representations.

## Example chronology

Suppose the initial GP_Set immediately reveals:

1. output shape is exactly input shape;
2. black's output ID/presence is exactly represented by another gene.

The retained evaluation state becomes conceptually:

~~~text
root
├── shape
│   solution #1
│   gene_idx = 1
│   terminal = True
│
└── composite
    ├── color_id          remaining colors only
    ├── color_presence    remaining presence channels only
    └── color_0
        solution #2
        gene_idx = j
        terminal = True
~~~

The two terminal findings are no longer independently searchable.

However:

~~~text
root
composite
remaining color pools
~~~

are still available to the future evaluator.

## Resetting evaluation state

~~~python
tree.clear_answers()
~~~

returns the evaluation tree to its initial undiscovered state:

- solution numbering restarts at 1;
- whole-node terminal flags are cleared;
- dynamic subset terminals are removed;
- color remaining pools are restored from their original targets.

## Current boundary

This code does not yet:

- scan GP_Set.data for matches;
- decide which gene should be tested first;
- propagate an answer automatically from child to parent;
- choose between a whole-node answer and a decomposition;
- generate new GP operations.

Those behaviors belong to the upcoming GP_Set evaluation function.

This layer only guarantees that once that function finds an exact portion of
the output, the discovery can be retained once, numbered chronologically,
associated with its gene index, removed from repeated search, and still coexist
with testable parent representations.


## Essential gene indices

Once terminal solutions have been recorded, the GP_Set can report the gene
indices that directly account for the retained output solution pieces.

~~~python
gp_mat.get_essential_gidx()
~~~

or equivalently:

~~~python
get_essential_gidx(gp_mat)
~~~

returns unique solution gene indices in discovery order.

For example, if:

~~~text
solution #1 -> shape   -> gene 8
solution #2 -> color_0 -> gene 13
solution #3 -> subset  -> gene 8
~~~

the result is:

~~~python
[8, 13]
~~~

because the same gene only needs to be retained once.

## Essential dependency tree

A directly essential solution gene may depend on earlier GP genes.

GP_Set.source is parallel to the gene indices:

~~~text
source[i] = -1
    gene i has no earlier GP source dependency

source[i] = 4
    gene i was built from gene 4

source[i] = [4, 7]
    gene i required both genes 4 and 7
~~~

Multi-source entries may also be nested lists, tuples, or NumPy arrays.

To recover the complete set of genes required to reconstruct all current
solutions:

~~~python
gp_mat.get_essential_gidx_tree()
~~~

or:

~~~python
get_essential_gidx_tree(gp_mat)
~~~

The function begins with every directly essential gene and recursively follows
source until each branch terminates at -1.

For example:

~~~text
solution uses gene 6

source[6] = [4, 5]
source[4] = 3
source[3] = [1, 2]
source[5] = [2, 0]
source[1] = 0
source[2] = 0
source[0] = -1
~~~

returns:

~~~python
[0, 1, 2, 3, 4, 5, 6]
~~~

The result is:

- unique;
- dependency-complete;
- ordered so dependencies appear before genes that use them;
- shared dependencies are included only once.

This list is intended to become the index basis for reconstructing a minimal
GP_Set/program containing only the genes needed by the retained solutions.

Cycles and references to nonexistent source indices raise errors rather than
silently producing an invalid reconstruction.
