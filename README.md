# ARC-AGI-2 For Fun

An experimental **bidirectional symbolic program synthesis** system for ARC-AGI-2.

The project searches from both sides of each training pair:

- **forward from the input** by growing candidate symbolic programs;
- **backward from the known output** by decomposing the solution into reversible Boolean requirements;
- then **meets in the middle** by finding exact low-complexity relationships between the two semantic frontiers.

A fully solved Boolean proof can be distilled into an executable program and applied to unseen ARC test inputs.

This is an active research project. The architecture is intentionally explicit, inspectable, and exact rather than optimized for production use.

---

## Core architecture

~~~text
                              TRAINING PAIRS

                     input                         output
                       |                              |
                       v                              v
              FORWARD SYMBOLIC SEARCH       BACKWARD SYMBOLIC SEARCH
                       |                              |
                GP_meta / GP_X                 SP_meta / SP_X
                       |                              |
                       v                              v
                 candidate semantics          required semantics
                       |                              |
                       +--------- exact -------------+
                              matching
                                 |
                                 v
                         BOOLEAN SOLUTION TREE
                                 ST
                                 |
                    fully solved reconstructive proof
                                 |
                                 v
                           KELSCHINATOR
                                 |
                    distilled executable program
                                 |
                                 v
                       unseen test input -> y_hat
~~~

The system is therefore not simply evolving one expression until it matches an output.

It grows a **forward program space** and a **backward proof space**, then searches for exact semantic bridges between them.

---

## Main objects

The search environment contains five primary structures:

~~~text
GP_meta
GP_X

SP_meta
SP_X

ST
~~~

and one final compiled artifact:

~~~text
Kelschinator
~~~

### GP — Genetic Program side

GP grows transformations of the ARC input.

~~~python
GP_meta.source[gidx]
GP_meta.op[gidx]
GP_meta.dims[gidx]
GP_meta.params[gidx]

GP_X[gidx, sample_idx]
~~~

Each gene index represents one symbolic interpretation across every training sample.

GP is unrestricted by proof semantics: any registered operation that satisfies its source constraints may participate in forward search.

### SP — Solution Program side

SP begins from the known training outputs and grows **reversible solution representations**.

It uses the same gene-major representation:

~~~python
SP_meta.source[gidx]
SP_meta.op[gidx]
SP_meta.dims[gidx]
SP_meta.params[gidx]

SP_X[gidx, sample_idx]
~~~

SP is not a prediction. It is a structured search over representations of the known target.

### ST — Boolean Solution Tree

ST expresses what must be solved to reconstruct the output.

An ST node may be solved either:

1. **directly**, by a GP-backed symbolic relationship; or
2. **indirectly**, through a satisfied Boolean derivation branch.

ST uses nested:

~~~text
AND
OR
~~~

requirements plus known inverse transformations.

A node solved through GP records:

~~~text
gp_gidx
solution_rule
solution_params
~~~

so ST knows both **which GP gene supports the solution** and **how that GP gene maps to the target**.

---

## Bidirectional search

The two sides have complementary roles.

### Forward direction

~~~text
input
  |
  v
raw GP gene
  |
  +--> rotate
  +--> flip
  +--> partition
  +--> boolean operations
  +--> scalar summaries
  +--> ...
  |
  v
candidate semantic pool
~~~

GP asks:

> What useful symbolic representations can be generated from the input?

### Backward direction

~~~text
output
  |
  v
raw SP target
  |
  +--> shape AND composite
  |
  +--> reversible alternate representations
  |
  v
unresolved Boolean frontier
~~~

SP/ST asks:

> What smaller or alternate semantic pieces would be sufficient to reconstruct the output?

The search succeeds when enough of these two spaces can be connected exactly.

---

## Operation proof semantics

Every registered operation declares:

~~~text
partition = "and" | "or" | "null"
inverse_op
output_count
~~~

### AND

The generated children are jointly required to reconstruct the source.

Examples:

~~~text
partition_shape
partition_composite
partition_bool_trim
indiv_1dim
~~~

### OR

The generated representation is a reversible alternative way to solve the source.

Examples:

~~~text
bool_complement
mat2_cwrotate
dim0_flip
dim1_flip
dim2_flip
~~~

For example:

~~~text
C
OR
(
    inv_mat2_cwrotate
    AND rotated_C
)
~~~

If the rotated representation is solved, the original target is also solved.

### NULL

The transformation is useful on GP but cannot by itself form a reconstructive SP proof.

Examples:

~~~text
bool_sum
bool_mat_ident
bool_cavity
bool2_union
bool2_intersect
~~~

Information-losing and multi-source operations are currently NULL.

Reverse operations live in:

~~~text
notebooks/ops/inv_ops.py
~~~

and are registered in:

~~~python
INV_OP_REGISTRY
~~~

---

## Initial solution decomposition

Initialize from an ARC task:

~~~python
GP_meta, GP_X, SP_meta, SP_X, ST = init_env(task.train)
~~~

Gene 0 is the raw grid:

~~~text
GP_X[0] = training inputs
SP_X[0] = training outputs
~~~

Both sides are initially decomposed by shape and categorical composite structure.

### Shape

~~~python
h_gidx, w_gidx = partition_shape(meta, X, 0)
~~~

produces two scalar genes:

~~~text
h
w
~~~

ST represents:

~~~text
shape = h AND w
~~~

### Composite

~~~python
generated = partition_composite(meta, X, 0)
~~~

creates two genes per categorical color:

~~~text
color_id       scalar int64
color_presence 2D boolean matrix
~~~

For colors 0, 1, and 2:

~~~text
color_0_id
color_0_presence
color_1_id
color_1_presence
color_2_id
color_2_presence
~~~

The initial root proof is conceptually:

~~~text
ROOT
=
inv_partition_shape
AND
(
    h
    AND
    w
)
AND
(
    inv_partition_composite
    AND color_0_id
    AND color_0_presence
    AND color_1_id
    AND color_1_presence
    AND ...
)
~~~

For C colors, initialization creates:

~~~text
3 + 2*C
~~~

concrete genes on that side:

~~~text
0      raw grid
1      h
2      w
3+     color ID / presence pairs
~~~

### Select-residual color layering

When each training input and output have aligned spatial shape, SP also creates an input-relative select-residual representation by default.

For one input color x with observed output destinations:

~~~text
x -> {y1, y2, ..., yn}
~~~

SP creates one shared Boolean support for every non-empty destination subset A:

~~~text
U[x,A] = (input == x) AND (output in A)
~~~

ST then treats alternative base-layer choices as OR branches. For example:

~~~text
Layer[x,{0,1,2}]

OR
    select 0
    AND U[x,{0,1,2}]
    AND Layer[x,{1,2}]

OR
    select 1
    AND U[x,{0,1,2}]
    AND Layer[x,{0,2}]

OR
    select 2
    AND U[x,{0,1,2}]
    AND Layer[x,{0,1}]
~~~

This recursively carries smaller residual supports until a singleton destination remains.

The data expansion is a shared subset DAG rather than duplicated paint permutations, so n destinations require at most:

~~~text
2^n - 1
~~~

support masks for that source color.

The default initialization ceiling is 5 destinations per source color:

~~~python
init_env(
    task.train,
    select_residual_max_destinations=5,
)
~~~

If input/output shapes do not align, or a source color exceeds the configured ceiling, this root alternative is skipped and the ordinary absolute composite proof remains available.

---

## Random symbolic growth

### GP generation

~~~python
new_gp_gidx = GP_generate(
    GP_meta,
    GP_X,
    n_new_genes=10,
    rng=42,
)
~~~

GP generation:

- samples valid registered operations;
- obeys dimensionality, dtype, source-arity, and custom validators;
- prevents duplicate transformation signatures;
- prevents exact duplicate instantiated genes;
- supports fixed and dynamic output counts;
- stops when the legal search space is exhausted.

### SP generation

~~~python
new_sp_gidx = SP_generate(
    SP_meta,
    SP_X,
    ST,
    rng=42,
)
~~~

SP generation performs one reversible AND/OR transformation and automatically updates ST with the corresponding Boolean proof alternative.

NULL operations are excluded from SP.

---

## Typed computation pools

The matching layer does not compare every gene against every target blindly.

Both sides expose symmetric dimensionality and dtype filters.

### GP candidate pool

~~~python
GP_pool = get_GP_pool(
    GP_meta,
    GP_X,
    min_dim=2,
    max_dim=2,
    dtype=bool,
)
~~~

### Unresolved ST frontier

~~~python
ST_frontier = get_ST_unsovled_frontier(
    ST,
    SP_X,
    min_dim=2,
    max_dim=2,
    dtype=bool,
)
~~~

The correctly spelled alias is also available:

~~~python
get_ST_unsolved_frontier(...)
~~~

Both return 1D object arrays containing complete genes across all training samples.

Examples:

~~~python
# all scalar targets, any dtype
get_ST_unsovled_frontier(
    ST,
    SP_X,
    min_dim=0,
    max_dim=0,
)

# 2D boolean candidates only
get_GP_pool(
    GP_meta,
    GP_X,
    min_dim=2,
    max_dim=2,
    dtype=bool,
)
~~~

Solved ST subtrees are pruned from the active frontier.

For an unresolved OR relation:

~~~text
C
OR
(
    inverse
    AND transformed_C
)
~~~

both the original concrete target and its transformed alternative remain available until one route solves C.

---

## Exact minimum-complexity solving

The matching layer currently focuses on exact symbolic rules.

A candidate is accepted only when it has **zero residual across every training sample**.

The search is ordered by rule complexity rather than training fit quality: once a simpler exact family succeeds, more expressive families are unnecessary.

### 0D one-gene solver

~~~python
solutions_0d = solve_0dim_1gene_basic(
    GP_pool_0d,
    ST_frontier_0d,
    GP_X=GP_X,
    SP_X=SP_X,
    ST=ST,
)
~~~

Search order:

~~~text
0  y = x
1  y = -x
2  y = abs(x)
3  y = x^2
4  y = x + c
5  y = c - x
6  y = c*x
7  y = a*x + b
~~~

The solver:

- requires exact agreement across all demonstrations;
- preserves the target dtype;
- limits freely fitted rules by available sample count;
- prioritizes lower-complexity rules;
- writes successful solutions directly into ST.

### 2D one-gene solver

~~~python
solutions_2d = solve_2dim_1gene_basic(
    GP_pool_2d,
    ST_frontier_2d,
    GP_X=GP_X,
    SP_X=SP_X,
    ST=ST,
)
~~~

Current search hierarchy:

~~~text
0  Y = X

1  structural zero-background embed
   top-left / top-right / bottom-left / bottom-right / center

2  fixed zero-background embed(row, col)

3  structural crop

4  fixed crop(row, col)

5  zero-fill shift(dr, dc)

6  integer tile(rows, cols)
~~~

Containment alone is not considered a solution.

If a smaller GP matrix appears inside a larger SP matrix, the rule is accepted only when the candidate relationship reconstructs the **entire target exactly**.

---

## Kelschinator: solution distillation

Once ST is completely solved, the proof can be compiled into an executable program.

~~~python
kelschinator = Kelschinator()

fit_success = kelschinator.fit(ST)
~~~

A fit is accepted only if:

1. ST is fully solved;
2. the selected Boolean proof is reconstructive;
3. every required GP dependency can be replayed;
4. every symbolic matching rule can be executed;
5. the distilled pipeline reproduces **every training output exactly**.

Then:

~~~python
y_hat = kelschinator.transform(X_test)
~~~

applies the discovered program to a new input matrix.

The fitted Kelschinator freezes only the GP dependencies and ST reconstruction path actually required by the solved proof.

Useful inspection fields:

~~~python
kelschinator.is_fitted_
kelschinator.last_error_
kelschinator.pipeline_
~~~

Example:

~~~python
kelschinator = Kelschinator()

if kelschinator.fit(ST):
    y_hat = kelschinator.transform(task.test[0].input)

    for step in kelschinator.pipeline_:
        print(step)
else:
    print(kelschinator.last_error_)
~~~

This is the final synthesis step:

~~~text
search graph
   |
   v
solved Boolean proof
   |
   v
minimum required GP dependencies
   +
reconstruction operations
   |
   v
executable symbolic program
~~~

---

## One-call synthesis wrapper

For ordinary experimentation, the current search lifecycle is wrapped by:

~~~python
from notebooks.ops import synth

(
    solved_exactly,
    kelschinator,
    final_gp_len,
    final_sp_len,
    final_gp_ops,
    final_sp_ops,
    iterations,
) = synth(
    task_id,
    max_GP=1000,
    max_SP=1000,
    gen_size_GP=10,
)
~~~

`max_GP` and `max_SP` are loose ceilings: if a side is below its ceiling at the beginning of an iteration, that generation call is allowed to finish even if the resulting gene count crosses the limit.

The wrapper repeatedly grows GP/SP, reruns the currently implemented exact 0D/2D matchers, stops when ST solves or both search sides can no longer grow, distills the proof with Kelschinator, and finally checks every known test output for exact equality.

The operation counts report distinct operation applications rather than genes, so a multi-output partition is counted once.

---
## End-to-end notebook skeleton

~~~python
from notebooks.ops import (
    GP_generate,
    SP_generate,
    Kelschinator,
    get_GP_pool,
    get_ST_unsovled_frontier,
    init_env,
    solve_0dim_1gene_basic,
    solve_2dim_1gene_basic,
)


# Build bidirectional search environment.
GP_meta, GP_X, SP_meta, SP_X, ST = init_env(task.train)


# Grow candidate forward programs.
GP_generate(
    GP_meta,
    GP_X,
    n_new_genes=10,
    rng=42,
)


# Grow one reversible output-side representation.
SP_generate(
    SP_meta,
    SP_X,
    ST,
    rng=42,
)


# Scalar exact matching.
GP_pool_0d = get_GP_pool(
    GP_meta,
    GP_X,
    min_dim=0,
    max_dim=0,
)

ST_frontier_0d = get_ST_unsovled_frontier(
    ST,
    SP_X,
    min_dim=0,
    max_dim=0,
)

solve_0dim_1gene_basic(
    GP_pool_0d,
    ST_frontier_0d,
    GP_X=GP_X,
    SP_X=SP_X,
    ST=ST,
)


# 2D boolean exact matching.
GP_pool_2d = get_GP_pool(
    GP_meta,
    GP_X,
    min_dim=2,
    max_dim=2,
    dtype=bool,
)

ST_frontier_2d = get_ST_unsovled_frontier(
    ST,
    SP_X,
    min_dim=2,
    max_dim=2,
    dtype=bool,
)

solve_2dim_1gene_basic(
    GP_pool_2d,
    ST_frontier_2d,
    GP_X=GP_X,
    SP_X=SP_X,
    ST=ST,
)


# Compile only when the Boolean proof is complete.
kelschinator = Kelschinator()

if kelschinator.fit(ST):
    y_hat = kelschinator.transform(task.test[0].input)
~~~

The search loop itself is intentionally still experimental. GP/SP growth, pool selection, and solver calls can be scheduled differently as the search policy evolves.

---

## Current design principles

### Exact semantics over similarity

ARC colors and symbolic states are treated categorically unless an operation explicitly assigns another meaning.

The core match condition is exact cross-sample behavior, not continuous similarity.

### Minimum complexity before flexibility

Simple exact relationships are searched before more expressive ones.

The goal is to avoid explaining a tiny ARC training set with unnecessarily flexible rules.

### Reconstructive proofs

An ST branch only solves a target when the target can actually be reconstructed from the branch.

Partial containment or information-losing relationships do not count as complete solutions.

### Explicit search state

Programs, instantiated values, proof structure, solution rules, and inverse operations remain inspectable rather than hidden inside a latent model.

### Executability is part of correctness

A Boolean proof is not sufficient by itself.

Kelschinator must be able to replay the discovered symbolic program and exactly regenerate every training output before the solution is accepted as a fitted transformation.

---

## Operation library

Core operations currently include:

~~~text
partition_shape
partition_composite
indiv_1dim
partition_bool_trim

bool_complement
bool2_union
bool2_intersect
bool_sum
bool_mat_ident
bool_cavity

mat2_cwrotate
dim0_flip
dim1_flip
dim2_flip
~~~

The operation registry supports:

~~~text
single-source operations
multi-source operations
ordered/unordered sources
dimensionality restrictions
atomic dtype restrictions
parameter samplers
custom validators
fixed output counts
dynamic output counts
inverse operation registration
AND / OR / NULL proof semantics
~~~

---

## Repository structure

~~~text
notebooks/ops/
├── environment.py     GP/SP/ST data structures and pool/frontier helpers
├── ops.py             forward operation registry and program generation
├── inv_ops.py         inverse/reconstruction operation registry
├── solve.py           exact 0D and 2D symbolic matching
├── kelschinator.py    solved-proof -> executable-program distillation
├── GP.py              legacy GP_Set architecture / reference
├── loss.py            legacy descriptive loss-tree work / reference
├── grid_ops.py
└── alpha_ops.py

docs/
├── ARCHITECTURE.md        high-level current architecture
├── GP_SP_ENVIRONMENT.md   detailed current GP/SP/ST specification
├── GP_EVAL_TREE.md        legacy GP_Set evaluation-tree reference
├── GP_LOSS_TREE.md        legacy descriptive loss-tree reference
└── KAGGLE.md
~~~

New architecture work should generally target:

~~~text
environment.py
ops.py
inv_ops.py
solve.py
kelschinator.py
~~~

---

## Architecture documentation

For the current system overview:

~~~text
docs/ARCHITECTURE.md
~~~

For the detailed implementation model, proof semantics, generation rules, matching layer, and Kelschinator lifecycle:

~~~text
docs/GP_SP_ENVIRONMENT.md
~~~

The GP_Set evaluation-tree and loss-tree documents are retained as explicitly labeled legacy references.

---

## Quick start

~~~bash
git clone https://github.com/Logan-Kelsch/ARC-AGI-2_For_Fun.git
cd ARC-AGI-2_For_Fun

make setup
make data
make test
~~~

Then in a notebook:

~~~python
from notebooks.ops import init_env

GP_meta, GP_X, SP_meta, SP_X, ST = init_env(task.train)
~~~

---

## Research framing

The current architecture is best described as:

> **Bidirectional symbolic program synthesis with forward genetic-program search, backward reversible solution decomposition, exact semantic frontier matching, Boolean proof resolution, and executable program distillation.**

A shorter description is:

> **Bidirectional symbolic search with semantic frontier matching.**

The central research principle is:

> **Grow candidate interpretations from the input, grow reconstructive requirements from the output, meet them through the simplest exact semantic relationships available, and compile the satisfied proof into an executable program.**
