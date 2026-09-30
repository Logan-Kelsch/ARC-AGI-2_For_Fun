# GP / SP environment architecture

This is the new experimental architecture for ARC program synthesis.

The environment has exactly five primary components:

~~~text
GP_meta
GP_X
SP_meta
SP_X
ST
~~~

The older GP_Set / loss-tree implementation remains in the repository for
reference while this architecture is developed.

## 1. GP_meta

GP_meta is gene-parallel metadata for transformations of the input side.

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

source identifies the gene or genes required to generate i.

op identifies the registered transformation operation.

dims is the number of dimensions in the instantiated data for that gene.

Examples:

~~~text
dims = 0    scalar
dims = 1    vector
dims = 2    matrix
dims = 3    stack of matrices / tensor
~~~

## 2. GP_X

GP_X contains actual instantiated input-side gene data.

Its indexing order is:

~~~text
first index   gene index
second index  training sample index
~~~

so:

~~~python
GP_X[gidx][sample_idx]
GP_X[gidx, sample_idx]
~~~

refer to one instantiated gene value.

Different samples may have different spatial shapes because each gene stores a
1D object array across samples.

## 3. SP_meta

SP_meta has the same structure as GP_meta but describes transformations of the
known training outputs.

SP is constrained more strictly than GP:

> SP may only use operations registered with full_partition=True.

This ensures SP is a decomposition of the complete output representation rather
than an arbitrary feature search.

## 4. SP_X

SP_X is the output-side instantiated program data with the same gene-major
indexing:

~~~python
SP_X[gidx][sample_idx]
~~~

SP_X is not a predicted output. It is a fully observed decomposition of the
known training solution used to define what GP eventually needs to explain.

## 5. ST

ST is the SolutionTree derived from SP_meta.

Each SP gene has an ST node containing:

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

meaning no exact GP gene has yet been associated with that solution-program
node.

A future evaluator will compare GP_X against SP_X across all training samples.
When a GP gene exactly represents an SP gene, ST can record:

~~~python
ST.mark_solution(sp_gidx, gp_gidx)
~~~

ST is derived from SP dependencies rather than from a separately hand-written
loss hierarchy. As SP grows through full-partition operations, ST can be synced
from SP_meta.

## Registered operations

Transformation operations live in:

~~~text
notebooks/ops/ops.py
~~~

Operations use the same interface on either side:

~~~python
operation(GP_meta, GP_X, source_idx)
operation(SP_meta, SP_X, source_idx)
~~~

The operation layer checks that meta and X belong to the same side.

Every registered operation declares:

~~~text
full_partition
output_count
~~~

GP may use any registered transformation.

SP may only use operations marked:

~~~text
full_partition = True
~~~

## Initial operations

### partition_shape

~~~python
gidx = partition_shape(meta, X, source_idx)
~~~

For each sample, this transforms the selected source into its shape vector.

For a 2D grid:

~~~text
[[...],
 [...]]

->

[height, width]
~~~

It generates one gene with dims=1.

### partition_composite

~~~python
color_gidx, presence_gidx = partition_composite(
    meta,
    X,
    source_idx,
)
~~~

For each categorical 2D source it generates two genes.

First:

~~~text
1D sorted array of colors used
~~~

Second:

~~~text
3D boolean array:
[num_colors, height, width]
~~~

Presence channel i corresponds to color_ids[i].

This is the same categorical decomposition used by the previous
grid_dissection work, now expressed as a GP/SP operation.

Both initial operations are full partitions and are therefore legal on SP.

## init_env

~~~python
GP_meta, GP_X, SP_meta, SP_X, ST = init_env(task.train)
~~~

Initialization creates raw gene 0:

~~~text
GP_X[0] = input matrices across samples
SP_X[0] = output matrices across samples
~~~

Then it applies partition_shape and partition_composite to gene 0 on both
sides.

The initial gene table is therefore:

~~~text
gidx   op                     source   dims

0      raw_input/output       -1       2
1      partition_shape         0       1
2      partition_composite     0       1   color IDs
3      partition_composite     0       3   color presence
~~~

The initial ST is derived from SP_meta:

~~~text
SP g0: raw_output
├── SP g1: partition_shape
├── SP g2: partition_composite   dims=1
└── SP g3: partition_composite   dims=3
~~~

and every node begins with:

~~~text
gp_gidx = -1
~~~

## Direction from here

GP is the expanding search space.

SP is the exact output decomposition and may only expand through full
partitions.

ST is the bridge between them:

~~~text
GP_meta / GP_X
      |
      | exact cross-sample matches
      v
      ST
      ^
      |
SP_meta / SP_X
~~~

The next evaluation layer can therefore operate on a much simpler invariant:

> Find GP genes whose instantiated values exactly equal an SP gene across every
> training sample, then record the GP gene index in the corresponding ST node.
