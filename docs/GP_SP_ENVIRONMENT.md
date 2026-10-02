# GP / SP / ST Architecture

This document describes the current experimental architecture for **bidirectional symbolic program synthesis** in ARC-AGI-2.

The system searches from both ends of each training pair:

~~~text
input
  |
  v
forward GP program growth
  |
  v
candidate semantics
  |
  +--------------------+
                       |
                 exact symbolic matching
                       |
  +--------------------+
  |
  v
required semantics
  ^
  |
backward SP/ST solution decomposition
  ^
  |
output
~~~

A solved Boolean proof is then compiled by **Kelschinator** into an executable transformation for unseen test inputs.

The older GP_Set and descriptive loss-tree code remains in the repository as reference. New architecture work should generally target:

~~~text
environment.py
ops.py
inv_ops.py
solve.py
kelschinator.py
~~~

---

# 1. Data model

The live search environment contains:

~~~text
GP_meta
GP_X

SP_meta
SP_X

ST
~~~

ProgramMeta is gene-parallel metadata. For every gene index gidx:

~~~python
meta.source[gidx]
meta.op[gidx]
meta.dims[gidx]
meta.params[gidx]
~~~

describe the instantiated data in:

~~~python
X[gidx]
~~~

source records the parent gene or genes, op records the registered transformation, dims records instantiated dimensionality, and params stores exact operation parameters.

ProgramX is gene-major:

~~~python
X[gidx]
X[gidx, sample_idx]
~~~

X[gidx] is a 1D object array containing the same symbolic gene across every training sample. Sample-specific matrix shapes are allowed while the gene meaning remains stable.

---

# 2. GP: forward symbolic search

GP begins from the input grids:

~~~text
GP_X[0] = training input matrices
~~~

Forward operations grow candidate symbolic interpretations:

~~~text
raw input
   |
   +--> partitions
   +--> masks
   +--> rotations
   +--> flips
   +--> boolean transforms
   +--> scalar summaries
   +--> ...
   |
   v
candidate semantic program space
~~~

GP may use any registered operation that satisfies its source constraints.

The purpose of GP is not to predict the output directly. It generates symbolic state that may exactly explain pieces of the output-side proof.

---

# 3. SP: backward symbolic search

SP begins from the known training outputs:

~~~text
SP_X[0] = training output matrices
~~~

SP explores alternate reconstructive representations of those targets.

Each operation declares:

~~~text
partition = "and" | "or" | "null"
inverse_op
output_count
~~~

## AND

An AND operation decomposes a source into children that are jointly sufficient to reconstruct or validate it.

Current examples:

~~~text
partition_shape
partition_composite
partition_bool_trim
indiv_1dim
~~~

A conceptual proof branch is:

~~~text
inverse
AND child_0
AND child_1
AND ...
~~~

## OR

An OR operation creates a reversible alternate representation.

Example:

~~~text
C
OR
(
    inv_mat2_cwrotate
    AND rotated_C
)
~~~

Current OR operations include:

~~~text
bool_complement
mat2_cwrotate
dim0_flip
dim1_flip
dim2_flip
~~~

## NULL

NULL operations are useful on GP but do not create a valid reconstructive SP proof.

Current examples:

~~~text
bool_sum
bool_cavity
bool2_union
bool2_intersect
~~~

Information-losing and multi-source transformations are currently NULL and GP-only.

---

# 4. ST: Boolean Solution Tree

SolutionTree is the proof layer connecting GP and SP.

ST uses:

~~~text
STNodeRef
STInverseRef
STSet(mode="AND")
STSet(mode="OR")
~~~

A semantic node may be solved:

1. directly from GP; or
2. through a satisfied Boolean derivation.

A direct solution records:

~~~text
gp_gidx
solution_rule
solution_params
~~~

For example:

~~~text
gp_gidx = 17
solution_rule = "add_constant"
solution_params = {"c": 2}
~~~

This means the target is reconstructed from GP gene 17 through y = x + 2.

A node can also remain without a direct GP index and become solved through a reversible branch:

~~~text
C
OR
(
    inverse
    AND transformed_C
)
~~~

Boolean solved state propagates toward the root.

---

# 5. Initialization

The main entry point is:

~~~python
GP_meta, GP_X, SP_meta, SP_X, ST = init_env(task.train)
~~~

Initialization creates raw gene 0:

~~~text
GP_X[0] = inputs
SP_X[0] = outputs
~~~

Then both sides receive partition_shape and partition_composite.

## partition_shape

~~~python
h_gidx, w_gidx = partition_shape(
    meta,
    X,
    source_idx,
)
~~~

A 2D source produces scalar int64 height and width genes:

~~~text
shape = h AND w
~~~

The inverse:

~~~python
inv_partition_shape(h, w, composite)
~~~

validates reconstructed dimensions.

## partition_composite

For every categorical color observed across the training set:

~~~text
color_id        scalar int64
color_presence  2D bool
~~~

The inverse:

~~~python
inv_partition_composite(
    color0,
    mask0,
    color1,
    mask1,
    ...
)
~~~

reassembles the categorical matrix exactly.

For C colors, each side initially contains:

~~~text
3 + 2*C genes
~~~

with:

~~~text
0      raw grid
1      h
2      w
3+     color ID / presence pairs
~~~

## Initial Boolean proof

Conceptually:

~~~text
ROOT
=
inv_partition_shape
AND shape
AND composite
~~~

where:

~~~text
shape = h AND w
~~~

and:

~~~text
composite
=
inv_partition_composite
AND color_0_id
AND color_0_presence
AND color_1_id
AND color_1_presence
AND ...
~~~

Known inverse operations are innate grammar knowledge and do not need to be discovered by GP.

## Select-residual initialization

When every training input/output pair has the same shape, init_env also builds an input-relative select-residual SP representation.

For one source color x with destination set A:

~~~text
x -> A
~~~

the concrete support target is:

~~~text
U[x,A] = (input == x) AND (output in A)
~~~

SP materializes every non-empty subset support once. ST then creates logical Layer[x,A] nodes whose OR branches choose one destination as the current base paint and recursively carry the remaining destination subset.

For example:

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

A singleton layer is reconstructed directly from its destination color and carried support.

This lets the solver explain a categorical partition through nested residual supports rather than requiring every direct x->y mask independently.

The subset supports are shared across all branches. Therefore n destinations create at most:

~~~text
2^n - 1
~~~

concrete support genes for that source color, rather than n! duplicated ordering paths.

The default expansion ceiling is:

~~~python
select_residual_max_destinations=5
~~~

and can be changed through init_env.

The complete select-residual reconstruction uses:

~~~text
inv_select_residual_leaf
inv_select_residual_step
inv_partition_select_residual
~~~

The inverse layer geometry enforces nested-support containment and exact final output coverage.

The select-residual route is attached to the root as a sibling OR alternative to the existing shape/composite representation.

init_env also binds GP_meta, GP_X, and SP_X to ST so a solved proof can later be distilled by Kelschinator.fit(ST).

---

# 6. Operation registry

Forward operations live in:

~~~text
notebooks/ops/ops.py
~~~

Inverse and reconstruction operations live in:

~~~text
notebooks/ops/inv_ops.py
~~~

The registries are:

~~~python
OP_REGISTRY
INV_OP_REGISTRY
~~~

Operations can declare:

~~~text
partition
inverse_op
output_count
output_count_estimator

source_count
ordered_sources

min_dims_exclusive
allowed_dims
atomic_dtypes

parameter_sampler
validator
~~~

This supports fixed and dynamic output counts, multi-source operations, commutative sources, dimensional restrictions, dtype restrictions, and custom validity rules.

---

# 7. Current operation families

Structural partitions:

~~~text
partition_shape
partition_composite
indiv_1dim
partition_bool_trim
~~~

Reversible unary transforms:

~~~text
bool_complement
mat2_cwrotate
dim0_flip
dim1_flip
dim2_flip
~~~

GP-only NULL transforms:

~~~text
bool_sum
bool_cavity
bool2_union
bool2_intersect
~~~

## indiv_1dim

A 1D source of length L becomes L scalar genes:

~~~text
[a, b, c]
->
a
b
c
~~~

On SP:

~~~text
source
OR
(
    inv_indiv_1dim
    AND element_0
    AND element_1
    AND element_2
)
~~~

All samples must have the same vector length so position semantics remain stable.

---

# 8. Program generation

## valid_generation

~~~python
valid_generation(
    meta,
    X,
    operation,
    source_idx,
    params={...},
)
~~~

checks side consistency, source existence, SP proof legality, dimensionality, atomic dtype, custom validators, and duplicate transition identity.

A transition is identified by:

~~~text
operation
source gene(s)
exact parameters
~~~

## Exact instantiated-gene novelty

Generation also rejects candidate outputs whose full cross-sample values exactly duplicate retained genes.

Exact equality includes representation, shape, dtype, contents, and all training samples.

Useful helpers:

~~~python
genes_exactly_equal(...)
equivalent_gene_idx(...)
~~~

Multi-output generation is transactional: if any output is redundant, the whole invocation is rolled back.

## GP_generate

~~~python
new_gidx = GP_generate(
    GP_meta,
    GP_X,
    n_new_genes=10,
    rng=0,
)
~~~

n_new_genes is an output-gene budget.

## SP_generate

~~~python
new_sp_gidx = SP_generate(
    SP_meta,
    SP_X,
    ST,
    rng=0,
)
~~~

One call performs one reversible SP transformation. When ST is supplied, the corresponding Boolean proof alternative is registered immediately.

---

# 9. Typed computation pools

The matching layer uses compatible slices of GP and the unresolved ST frontier.

## GP pool

~~~python
GP_pool = get_GP_pool(
    GP_meta,
    GP_X,
    min_dim=None,
    max_dim=None,
    dtype=None,
)
~~~

## ST frontier

~~~python
ST_frontier = get_ST_unsovled_frontier(
    ST,
    SP_X,
    min_dim=None,
    max_dim=None,
    dtype=None,
)
~~~

The correctly spelled alias is also available:

~~~python
get_ST_unsolved_frontier(...)
~~~

Examples:

~~~python
# every scalar candidate, any dtype
get_GP_pool(
    GP_meta,
    GP_X,
    min_dim=0,
    max_dim=0,
)

# unresolved 2D boolean targets only
get_ST_unsovled_frontier(
    ST,
    SP_X,
    min_dim=2,
    max_dim=2,
    dtype=bool,
)
~~~

Solved ST subtrees are pruned.

For an unresolved OR relation, both the original target and transformed target may remain candidates until one route solves the parent.

---

# 10. Exact semantic matching

The current solvers are ordered symbolic searches, not statistical regressors.

Training error is binary:

~~~text
zero residual       candidate
non-zero residual   reject
~~~

Among zero-residual candidates, lower-complexity rule families are preferred.

## Scalar one-gene solver

~~~python
solve_0dim_1gene_basic(
    GP_pool,
    ST_frontier,
    GP_X=GP_X,
    SP_X=SP_X,
    ST=ST,
)
~~~

Search hierarchy:

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

One-parameter fitted rules require at least two samples. Affine fitting requires at least three samples and at least two distinct source values.

## 2D one-gene solver

~~~python
solve_2dim_1gene_basic(
    GP_pool,
    ST_frontier,
    GP_X=GP_X,
    SP_X=SP_X,
    ST=ST,
)
~~~

Search hierarchy:

~~~text
0  identity
1  structural zero-background embedding
2  fixed zero-background embedding
3  structural crop
4  fixed crop
5  zero-fill translation
6  integer tiling
~~~

Structural positions include top-left, top-right, bottom-left, bottom-right, and center.

Containment alone is not a solution. A spatial rule must reconstruct the complete target exactly across every training sample.

Successful matches update ST immediately.

---

# 11. Minimum-complexity principle

The current solver policy is:

> Search the least expressive exact hypothesis class first.

For example:

~~~text
y = x
~~~

is preferred over:

~~~text
y = x + c
~~~

which is preferred over:

~~~text
y = a*x + b
~~~

when a simpler family already achieves exact agreement.

This is intended to reduce trivial interpolation and meaningless explanations in ARC's small training sets.

The same principle should guide future multi-gene, object-level, and higher-dimensional matching.

---

# 12. Kelschinator

Kelschinator turns a fully solved Boolean proof into an executable symbolic program.

~~~python
kelschinator = Kelschinator()

fit_success = kelschinator.fit(ST)
~~~

fit returns False when:

- ST is not fully solved;
- ST lacks its bound GP/SP environment;
- a selected proof branch is not reconstructive;
- a required GP operation cannot be replayed;
- a retained symbolic matching rule is not executable;
- the distilled program fails exact training replay.

A Boolean solved state is therefore necessary but not sufficient.

**Executability is part of correctness.**

## Distillation

A successful fit:

1. chooses a satisfied root proof;
2. follows only required Boolean branches;
3. identifies direct GP-backed ST nodes;
4. traces their GP source dependencies;
5. freezes only those GP operations;
6. freezes retained symbolic solution rules;
7. freezes required inverse reconstruction operations;
8. replays the compiled program on every training pair.

The result is a specific executable program rather than the full search graph.

## Training replay

For every training sample:

~~~text
compiled_program(input_i) == output_i
~~~

must hold exactly, including shape and dtype.

Only then:

~~~python
kelschinator.is_fitted_
~~~

becomes True.

## Test transformation

~~~python
y_hat = kelschinator.transform(X_test)
~~~

currently accepts one 2D input matrix and returns one reconstructed 2D output matrix.

Useful fields:

~~~python
kelschinator.is_fitted_
kelschinator.last_error_
kelschinator.pipeline_
~~~

The fitted Kelschinator no longer depends on later changes to ST's solved state.

---

# 13. End-to-end lifecycle

~~~text
1. init_env
   |
   v
GP / SP / ST
   |
   +-----------------------------------------------+
   |                                               |
   v                                               v
2. grow GP                                    3. grow SP
   |                                               |
   v                                               v
candidate semantics                       reversible target semantics
   |                                               |
   +-----------------------+-----------------------+
                           |
                           v
4. retrieve typed GP pool and unresolved ST frontier
                           |
                           v
5. run exact minimum-complexity symbolic matching
                           |
                           v
6. write direct solutions into ST
                           |
                           v
7. Boolean propagation
                           |
                      ST.solved?
                    /            \
                  no              yes
                  |                |
                  v                v
          continue search    8. Kelschinator.fit(ST)
                                   |
                                   v
                         exact training replay
                                   |
                              fit succeeds?
                             /            \
                           no              yes
                           |                |
                           v                v
                    continue search   9. transform(test_input)
~~~

This is the current meaning of bidirectional symbolic program synthesis in the project.

---

# 14. Minimal notebook example

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


GP_meta, GP_X, SP_meta, SP_X, ST = init_env(task.train)

GP_generate(
    GP_meta,
    GP_X,
    n_new_genes=10,
    rng=42,
)

SP_generate(
    SP_meta,
    SP_X,
    ST,
    rng=42,
)


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


kelschinator = Kelschinator()

if kelschinator.fit(ST):
    y_hat = kelschinator.transform(
        task.test[0].input
    )
~~~

The orchestration policy around generation, frontier prioritization, and solver scheduling remains experimental.

---

# 15. Current invariants

## Gene semantic consistency

One gene index retains one meaning across all training samples.

## Exactness

A solution relationship must match all demonstrations exactly.

## Dtype preservation

Matching and reconstruction preserve target dtype unless an exact safe conversion is part of the retained rule.

## Reconstructivity

A proof solves a target only when enough information exists to reconstruct that target.

## Boolean transparency

The reason a target is solved remains inspectable through ST.

## Search/program separation

The search graph may be large. The final Kelschinator retains only the program needed by the selected proof.

## Training replay

A distilled solution is not accepted unless it exactly reproduces all training outputs.

---

# 16. Research direction

The architecture now separates several research problems cleanly:

~~~text
operation grammar design
GP generation policy
SP decomposition policy
frontier prioritization
multi-gene exact matching
object-level relationships
solution complexity accounting
proof-path ranking
search-budget allocation
program simplification
generalization testing
~~~

The guiding principle is:

> **Grow symbolic possibilities forward from the input, grow reconstructive requirements backward from the output, connect them through the simplest exact semantic relationships available, and compile the satisfied proof into an executable program.**
