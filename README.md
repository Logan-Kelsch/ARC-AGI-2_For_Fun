# ARC-AGI-2 For Fun

An experimental **Genetic Programming / program-synthesis approach to ARC-AGI-2**.

The current architecture separates the search space over inputs from the exact partition structure of the known training outputs.

The five primary components are:

~~~text
GP_meta
GP_X

SP_meta
SP_X

ST
~~~

GP means Genetic Program, SP means Solution Program, and ST means Solution Tree.

This is an active research project. The architecture is intentionally explicit and interpretable rather than optimized for production use.

---

## Core idea

For every ARC training sample we know:

~~~text
input grid -> output grid
~~~

The system builds two program spaces:

~~~text
INPUT SIDE                         OUTPUT SIDE

GP_meta / GP_X                     SP_meta / SP_X
      |                                  |
      | arbitrary GP transforms          | reversible AND/OR transforms
      v                                  v
candidate interpretations          exact solution decomposition
      |                                  |
      +--------------- ST ---------------+
              exact cross-sample matches
~~~

GP is the expanding interpretation/search space.

SP is a complete decomposition of the observed training outputs.

ST records which SP genes are already represented exactly by GP genes across all training samples.

---

## GP_meta

GP_meta is parallel metadata for every GP gene.

For gene index i:

~~~python
GP_meta.source[i]
GP_meta.op[i]
GP_meta.dims[i]
~~~

describe:

~~~python
GP_X[i]
~~~

source identifies which earlier gene or genes were used.

op identifies the transformation operation.

dims records the dimensionality of the instantiated gene:

~~~text
0 scalar
1 vector
2 matrix
3 tensor / stack of matrices
...
~~~

---

## GP_X

GP_X contains the actual instantiated input-side gene values.

Its axes are gene-major:

~~~python
GP_X[gidx][sample_idx]
GP_X[gidx, sample_idx]
~~~

The same gene index means the same interpretation/operation for every training sample.

Because ARC sample shapes may differ, each gene stores its samples in a 1D object array.

---

## SP_meta and SP_X

SP has the same metadata/data structure as GP, but is instantiated from the known output grids.

The critical restriction is:

SP is restricted by proof semantics rather than a full-partition flag:

~~~text
AND   jointly reconstructive decomposition
OR    reversible alternative representation
NULL  GP-only / not a valid SP proof transform
~~~

SP may use AND/OR operations and rejects NULL operations.

---

## ST: Solution Tree

ST is derived directly from SP_meta.source.

Every SP gene receives a node containing:

~~~text
sp_gidx
op
dims
parents
children
gp_gidx
~~~

Initially:

~~~text
gp_gidx = -1
~~~

meaning no exact GP representation has yet been identified.

A future evaluator will compare GP and SP genes across every training sample. When GP gene j exactly reproduces SP gene i across all samples:

~~~python
ST.mark_solution(
    sp_gidx=i,
    gp_gidx=j,
)
~~~

ST therefore records how much of the exact solution program is already represented inside GP.

---

## Operations

Transformation operations live in:

~~~text
notebooks/ops/ops.py
~~~

and use one interface on either side:

~~~python
op(GP_meta, GP_X, source_idx)
op(SP_meta, SP_X, source_idx)
~~~

Each registered operation declares:

~~~text
partition = "and" | "or" | "null"
inverse_op
output_count
~~~

GP may use any registered operation. SP rejects NULL operations.

---

## Initial operations

### partition_shape

~~~python
h_gidx, w_gidx = partition_shape(
    meta,
    X,
    source_idx,
)
~~~

For a 2D grid, shape is now split into two scalar genes:

~~~text
grid -> height
     -> width
~~~

Both are required, so the SP proof represents shape as:

~~~text
shape = h AND w
~~~

### indiv_1dim

~~~python
element_gidxs = indiv_1dim(
    meta,
    X,
    source_idx,
)
~~~

For a source that is exactly 1D and length L across every sample, this creates
L scalar genes, one per element position.

~~~text
[a, b, c]
->
a
b
c
~~~

It is an AND partition. On SP, the original source remains an OR alternative:

~~~text
source
OR
(
    inv_indiv_1dim
    AND element_0
    AND element_1
    AND ...
)
~~~

The source length must be identical across samples so each generated gene index
has one stable positional meaning.

### partition_composite

~~~python
generated_gidx = partition_composite(
    meta,
    X,
    source_idx,
)
~~~

This creates two genes per distinct color observed across the complete sample
set:

~~~text
scalar int64 color ID
2D boolean presence mask
~~~

For three colors the partition produces six genes:

~~~text
color_1_id, color_1_presence,
color_2_id, color_2_presence,
color_3_id, color_3_presence
~~~

A color that is absent from one sample receives an all-False presence mask for
that sample, keeping gene indices aligned across demonstrations.

Both initial operations are AND partitions and are legal on SP.

---

## Initialization

The complete starting environment is:

~~~python
GP_meta, GP_X, SP_meta, SP_X, ST = init_env(task.train)
~~~

Gene 0 begins as:

~~~text
GP_X[0] = input grids across training samples
SP_X[0] = output grids across training samples
~~~

Then partition_shape and partition_composite are applied to gene 0 on both sides.
partition_shape creates scalar h and w genes.

The initial program layout is variable-length:

~~~text
gidx   operation                source   dims

0      raw_input/raw_output     -1       2
1      partition_shape           0       0   h
2      partition_shape           0       0   w
3      partition_composite       0       0   first color ID
4      partition_composite       0       2   first color mask
5      partition_composite       0       0   second color ID
6      partition_composite       0       2   second color mask
...    ...                        ...     ...
~~~

For C distinct colors, that side starts with `3 + 2*C` genes. ST is built
directly from the resulting SP structure, and every node begins unresolved with
gp_gidx=-1.

---

## Boolean ST solving

ST is now evaluated as a Boolean proof rather than a plain SP dependency tree.

The initial proof is:

~~~text
root = shape AND composite
~~~

A reversible SP transform creates an alternative branch. For example:

~~~text
composite_target
OR
(
    inverse_rotate
    AND rotated_target
)
~~~

Therefore the original target does not need a direct GP match if its transformed
representation is solved and the inverse is known.

Reverse transforms live in notebooks/ops/inv_ops.py.

---

## Expected search direction

~~~text
1. Initialize GP / SP / ST
        |
        v
2. SP defines the exact output decomposition
        |
        v
3. Expand GP with candidate transformation operations
        |
        v
4. Compare GP genes with SP genes across all samples
        |
        v
5. Mark exact matches in ST
        |
        v
6. Expand GP toward unresolved SP structure
        |
        v
7. Continue until the required SP representation is explained
        |
        v
8. Trace GP source dependencies
        |
        v
9. Reconstruct the minimal discovered program
        |
        v
10. Apply it to the unseen test input
~~~

Exact matching remains categorical and structural. The system does not assume ARC colors or arbitrary symbolic states have meaningful continuous distances.

---

## Quick start

~~~bash
git clone https://github.com/Logan-Kelsch/ARC-AGI-2_For_Fun.git
cd ARC-AGI-2_For_Fun

make setup
make data
make test
~~~

Inside a notebook:

~~~python
from notebooks.ops import init_env

GP_meta, GP_X, SP_meta, SP_X, ST = init_env(task.train)
~~~

---

## Experimental modules

~~~text
notebooks/ops/
├── environment.py
├── ops.py
├── GP.py          legacy GP_Set architecture / reference
├── loss.py        legacy descriptive loss work / reference
├── eval.py
├── grid_ops.py
└── alpha_ops.py
~~~

New architecture work should target environment.py, ops.py, and ST.

More detail is in:

~~~text
docs/GP_SP_ENVIRONMENT.md
~~~

---

## Current research principle

> **Grow interpretations on the input side, partition the known solution exactly on the output side, and let explicit structural matches progressively constrain the program search.**


### Unresolved ST frontier

~~~python
frontier = get_ST_unsovled_frontier(
    ST,
    SP_X,
    min_dim=0,
    max_dim=2,
)
~~~

returns a 1D object array containing the complete instantiated SP genes for
concrete unresolved nodes still reachable from the root proof.

Use min_dim=max_dim=0 for scalar targets and min_dim=max_dim=2 for matrix
targets. Solved nodes prune their lower proof subtrees. OR alternatives keep
both the original concrete target and its transformed target available until
one path resolves the source.

A correctly spelled alias, get_ST_unsolved_frontier, is also available.


### Typed ST / GP pools

The two core retrieval helpers now use the same filters:

~~~python
targets = get_ST_unsovled_frontier(
    ST,
    SP_X,
    min_dim=2,
    max_dim=2,
    dtype=bool,
)

candidates = get_GP_pool(
    GP_meta,
    GP_X,
    min_dim=2,
    max_dim=2,
    dtype=bool,
)
~~~

Both return 1D object arrays containing complete genes across all training
samples.

Use dtype=None for any datatype, including:

~~~python
get_ST_unsovled_frontier(
    ST,
    SP_X,
    min_dim=0,
    max_dim=0,
)
~~~

for every unresolved scalar target regardless of dtype.

This provides the compatible target/candidate pools used for later exact
solution matching.
