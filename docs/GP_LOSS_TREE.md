# GP Loss Tree — Legacy Reference

> **Legacy architecture:** this document describes the earlier descriptive GP loss-tree experiment. It remains available for reference but does **not** define current solution status or the active solver architecture.
>
> Current solution state is represented by the Boolean SolutionTree and compiled through Kelschinator. See docs/ARCHITECTURE.md and docs/GP_SP_ENVIRONMENT.md.

---

The loss tree is a descriptive view of how current prediction states transition
to target states across the training demonstrations.

It does **not** decide whether the GP system has found a solution. That decision
belongs in a separate GP evaluation layer.

~~~text
root
├── shape
│   ├── h
│   └── w
└── composite
~~~

The evaluator accepts one prediction/target pair or multiple samples:

~~~python
gp_mat = init_gp_mat(task.train)
tree = loss_resolution(gp_mat.input, gp_mat.output)
~~~

## Transition convention

Every leaf transition matrix uses:

~~~text
rows    = predicted/source state
columns = target state
~~~

The aggregate matrix contains all supplied samples and sample_matrices preserves
one matrix per demonstration.

## Source-state nodes

Only source states containing a nonzero delta are represented beneath h, w, or
composite.

For example:

~~~text
1 -> 0
~~~

is represented once under source state 1.

Target state 0 does not receive a duplicate node simply because a transition
enters it.

Each TransitionStateNode contains:

~~~text
source
target_counts
sample_target_counts
uniform
target
sample_indices
solution_idx
~~~

Identity-only states such as 0 -> 0 are omitted from the nonzero-delta tree.

## Meaning of ✓ and X

The loss tree uses only structural checks.

~~~text
✓  source has exactly one observed target
X  source branches to more than one observed target
~~~

For example:

~~~text
sample 0: 1 -> 0 x6
sample 1: 1 -> 0 x6
sample 2: 1 -> 0 x5
sample 3: 1 -> 0 x6
sample 4: 1 -> 0 x5
~~~

produces:

~~~text
✓ 1 -> 0 x28
~~~

because every observed occurrence of source state 1 maps to the same target.

By contrast:

~~~text
1 -> 0 x20
1 -> 3 x8
~~~

produces:

~~~text
X 1 -> {0 x20, 3 x8}
~~~

because source state 1 is not functionally uniform.

A source does not need to appear in every sample. The check describes all
observed occurrences across the supplied demonstrations.

Programmatic access:

~~~python
state = tree.composite.source_state(1)

state.source
state.target_counts
state.sample_target_counts
state.uniform
state.target
state.sample_indices
state.solution_idx
~~~

## Shape

Height and width use the same source-state structure independently.

Example:

~~~text
predicted H 3 -> target H 5
predicted H 3 -> target H 5
~~~

becomes:

~~~text
✓ 3 -> 5 x2
~~~

while:

~~~text
predicted H 3 -> target H 5
predicted H 3 -> target H 7
~~~

becomes:

~~~text
X 3 -> {5 x1, 7 x1}
~~~

Access:

~~~python
tree.h.state_nodes
tree.w.state_nodes
~~~

## Composite

Composite uses ARC color states 0..9 plus INVALID at state 10.

Shape mismatch can therefore appear as:

~~~text
INVALID -> target_color
predicted_color -> INVALID
~~~

These are treated exactly like other source-state mappings.

## Inspection

~~~python
inspect_loss(tree)
~~~

prints source states directly under their structural location.

Conceptually:

~~~text
X root | 5 sample(s)
├── ✓ shape
│   ├── ✓ h
│   │   └── ✓ no nonzero delta
│   └── ✓ w
│       └── ✓ no nonzero delta
└── X composite
    ├── ✓ 1 -> 0 x28
    └── X 2 -> {3 x12, 4 x5}
~~~

The structural parent receives ✓ only when every nonzero-delta source beneath it
has a uniform one-target mapping. A parent receives X when at least one source
branches.

These marks do **not** mean that a GP program has been found. They describe only
the structure already visible in the demonstrations.

## Exact-match compatibility

The existing resolved properties remain available for code that needs to ask
whether the current predicted grids exactly equal the targets:

~~~python
tree.resolved
tree.shape.resolved
tree.composite.resolved
~~~

They are separate from the ✓ / X uniformity markers used by inspect_loss.

No scalar total loss is calculated.
