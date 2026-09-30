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
      | arbitrary GP transforms          | full-partition transforms only
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

> **SP may only use operations marked full_partition=True.**

SP is therefore not another free-form search space. It is a lossless decomposition of the solution representation.

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
full_partition
output_count
~~~

GP may use any registered operation.

SP calls are rejected unless the operation is full_partition.

---

## Initial operations

### partition_shape

~~~python
gidx = partition_shape(
    meta,
    X,
    source_idx,
)
~~~

For a 2D grid:

~~~text
grid -> [height, width]
~~~

This creates one 1D gene.

### partition_composite

~~~python
color_gidx, presence_gidx = partition_composite(
    meta,
    X,
    source_idx,
)
~~~

This generates two genes:

~~~text
1. sorted 1D array of colors used
2. 3D boolean array [num_colors, height, width]
~~~

Presence channel i corresponds to color_ids[i].

Both initial operations are full partitions and are therefore legal on SP.

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

The initial program layout is:

~~~text
gidx   operation                source   dims

0      raw_input/raw_output     -1       2
1      partition_shape           0       1
2      partition_composite       0       1   color IDs
3      partition_composite       0       3   color presence
~~~

The initial SP structure produces:

~~~text
SP g0: raw_output
├── SP g1: partition_shape
├── SP g2: partition_composite   dims=1
└── SP g3: partition_composite   dims=3
~~~

Every ST node begins unresolved with gp_gidx=-1.

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
