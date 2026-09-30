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

The evaluator accepts one prediction/target pair or many samples at once:

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

The aggregate matrix contains all samples and sample_matrices keeps one matrix
per demonstration.

## Source-state loss nodes

The loss tree now interprets each nonzero delta by the state it is coming FROM.

For example:

~~~text
1 -> 0
~~~

is represented once as source state 1. Target state 0 does not receive a
duplicate node merely because a transition enters it.

Each nonzero-delta source becomes a TransitionStateNode with:

~~~text
source
target_counts
sample_target_counts
solution_idx = -1
~~~

This gives GP search a precise place to attach the gene/program that explains
that functional mapping.

Identity-only states such as 0 -> 0 are omitted from this nonzero-delta tree.

## Solved vs unsolved

A source-state node is solved when it has exactly one observed target across
all occurrences in all supplied demonstrations.

For example:

~~~text
sample 0: 1 -> 0 x6
sample 1: 1 -> 0 x6
sample 2: 1 -> 0 x5
sample 3: 1 -> 0 x6
sample 4: 1 -> 0 x5

aggregate: 1 -> 0 x28
~~~

is solved because the functional dependency is uniform:

~~~text
1 -> 0
~~~

even though it is off-diagonal and therefore the raw input prediction is not
yet equal to the output.

A source is unsolved when the same source branches to multiple targets:

~~~text
1 -> 0 x20
1 -> 3 x8
~~~

which means color alone is not sufficient to determine the output state.

A source does not need to appear in every sample. Solved means every observed
occurrence across the full demonstration set agrees on one target. The source
node records exactly which samples supplied evidence.

Programmatic access:

~~~python
tree.composite.solved_states
tree.composite.unsolved_states

state = tree.composite.source_state(1)

state.source
state.target_counts
state.sample_target_counts
state.solved
state.target
state.sample_indices
state.solution_idx
~~~

## Shape

Height and width remain separate nodes.

Their source states are interpreted exactly like colors.

For example:

~~~text
predicted H 3 -> target H 5
predicted H 3 -> target H 5
~~~

creates a solved height source state:

~~~text
3 -> 5
~~~

while:

~~~text
3 -> 5
3 -> 7
~~~

creates an unsolved height source state because predicted height 3 does not yet
determine one target height.

Access:

~~~python
tree.h.solved_states
tree.h.unsolved_states

tree.w.solved_states
tree.w.unsolved_states
~~~

## Composite

Composite uses ARC color states 0..9 plus INVALID at state 10.

Shape mismatches therefore still appear categorically:

~~~text
INVALID -> target_color
predicted_color -> INVALID
~~~

The same source-state uniformity rules apply to INVALID.

## Inspection

~~~python
inspect_loss(tree)
~~~

prints the nonzero-delta tree partitioned into solved and unsolved mappings.

Conceptually:

~~~text
X root
├── shape
│   ├── h
│   │   ├── solved
│   │   └── unsolved
│   └── w
│       ├── solved
│       └── unsolved
└── composite
    ├── solved
    │   └── ✓ 1 -> 0 x28
    └── unsolved
        └── X 2 -> {3 x12, 4 x5}
~~~

With show_samples=True, each represented source also shows its observed
sample-specific transitions.

## Two meanings of resolution

The system intentionally distinguishes two ideas.

Exact output resolution asks whether the current predicted grids exactly equal
the outputs. This is still represented by:

~~~python
tree.resolved
tree.shape.resolved
tree.composite.resolved
~~~

Functional source resolution asks whether a nonzero-delta source already has a
uniform mapping that GP can model.

This is represented by:

~~~python
state.solved
tree.composite.solved_states
tree.composite.unsolved_states
~~~

Therefore an off-diagonal mapping such as 1 -> 0 can be functionally solved
while tree.composite.resolved remains False.

No scalar total loss is calculated.
