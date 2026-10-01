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


## Random program generation

The environment now supports constrained random expansion of GP and SP.

### valid_generation

~~~python
valid_generation(
    meta,
    X,
    operation,
    source_idx,
    params={...},
)
~~~

checks a candidate transformation before it is instantiated.

The current checks are:

1. metadata and X belong to the same side and remain gene-parallel;
2. the source gene exists;
3. SP may only use operations marked full_partition;
4. the source satisfies the operation's dimensionality restriction;
5. the source satisfies the operation's atomic dtype restriction;
6. any operation-specific validator accepts the candidate;
7. the exact transformation signature has not already been used.

The exact transformation signature is:

~~~text
(operation, source gene index, exact parameters)
~~~

For multi-output operations, every output gene records the same signature.

ProgramMeta therefore now also retains:

~~~python
meta.params[gidx]
~~~

alongside source, op, and dims.

### Operation generation metadata

The operation decorator supports reusable search constraints:

~~~python
@operation(
    full_partition=True,
    output_count=1,
    min_dims_exclusive=0,
    atomic_dtypes=(np.int64,),
    parameter_sampler=...,
    validator=...,
)
~~~

The initial operations are configured as follows.

partition_shape:

~~~text
source dims > 0
dtype unrestricted
~~~

partition_composite:

~~~text
source dims > 1
atomic dtype == int64
~~~

Parameter samplers are currently unused by the initial two operations, but the
registry supports them for future integer/axis/kernel/etc. parameters.

### Atomic dtype

Generation restrictions use the atomic dtype of instantiated source values, not
the dtype of the outer object array used by ProgramX.

For example:

~~~text
2D np.int64 grid        -> atomic dtype int64
3D boolean mask stack   -> atomic dtype bool
1D float vector         -> atomic dtype float64
~~~

For nested object structures, atomic dtypes are collected recursively across all
samples.

### GP_generate

~~~python
new_gidx = GP_generate(
    GP_meta,
    GP_X,
    n_new_genes=20,
    rng=0,
)
~~~

GP_generate repeatedly:

~~~text
random registered operation
        +
random existing GP gene
        +
random operation parameters
        |
        v
valid_generation
        |
        v
instantiate operation
~~~

until the requested gene budget is filled or no valid generation remains.

The requested count is a gene-count budget, not an operation-count budget.
Therefore a two-output operation is only eligible when at least two gene slots
remain.

The returned list contains the newly created GP gene indices.

### SP_generate

~~~python
new_sp_gidx = SP_generate(
    SP_meta,
    SP_X,
    ST,
    rng=0,
)
~~~

SP_generate performs one random valid SP operation application.

Because SP only accepts full partitions, its candidate pool is automatically
restricted to full_partition operations.

One SP generation step may add multiple genes when the chosen operation has
multiple outputs.

If ST is supplied, it is automatically synchronized after SP grows.

### Duplicate prevention

Suppose the metadata already contains:

~~~text
op      = translate
source  = 7
params  = {"dx": 2, "dy": -1}
~~~

then that exact transition cannot be generated again.

However these remain distinct candidates:

~~~text
translate(source=7, dx=3, dy=-1)
translate(source=8, dx=2, dy=-1)
rotate(source=7, ...)
~~~

This allows the operation library to grow substantially without relying on
special-case duplicate rules for individual transformations.


## Exact instantiated-gene novelty

Generation now prevents a second form of redundancy in addition to duplicate
operation/source/parameter transitions.

A newly instantiated gene is retained only when its complete cross-sample data
is novel relative to every gene already present on that side.

Exact gene equality requires every sample value to match structurally:

~~~text
same representation type
same array shape
same array dtype
same contents
~~~

and the complete gene must match across every training sample.

Useful helpers:

~~~python
genes_exactly_equal(GP_X[i], GP_X[j])

equivalent_gene_idx(
    GP_X,
    candidate_gene_values,
)
~~~

equivalent_gene_idx returns the matching retained gene index, or -1 when the
candidate is novel.

### Transactional generation

Random generation applies candidates transactionally.

Conceptually:

~~~text
candidate operation
        |
        v
instantiate temporary output gene(s)
        |
        v
compare every output against all retained genes
        |
        +---- duplicate ----> rollback entire operation
        |
        '---- all novel ----> retain operation
~~~

For a multi-output operation, every output must be novel.

If one output duplicates an existing gene, the complete operation invocation is
rolled back rather than retaining only a partial partition.

Outputs from the same invocation are also checked against one another.

## Exhausting the legal generation space

For the current parameterless operation library, GP_generate and SP_generate
enumerate every legal operation/source candidate that remains after normal
generation constraints.

Candidates are shuffled before evaluation, so the search order remains random,
but every legal candidate can be tried once.

If all legal candidates either:

- were already instantiated as exact transitions, or
- instantiate data that exactly duplicates retained genes,

generation stops early and prints:

~~~text
GP generation terminated: entire legal generation space was explored and no additional unique genes can be generated.
~~~

or the corresponding SP message.

This means the generation loop does not continue retrying known-dead branches
after the current grammar is saturated.

Parameterized operations whose sampler has an effectively unbounded parameter
domain cannot honestly be called exhaustively searched. Those retain bounded
sampling behavior and print a separate sampled-parameter-space termination
message if no novel output is found.


## Boolean and spatial operation expansion

The operation registry now supports source arity directly.

~~~python
@operation(
    source_count=2,
    ordered_sources=False,
    ...
)
~~~

allows a transformation to consume multiple GP genes.

For commutative operations such as boolean union/intersection,
ordered_sources=False canonicalizes the source pair so:

~~~text
op(source=(4, 7))
op(source=(7, 4))
~~~

are the same transformation signature.

Generation enumerates source combinations automatically, so binary operations
participate in GP_generate without custom search code.

The registry also supports exact dimensional restrictions through:

~~~python
allowed_dims=(2,)
~~~

in addition to the existing dims > N rule.

### bool_complement

~~~text
source_count: 1
dtype: bool
dims: any
full_partition: False
~~~

Computes the boolean complement of the complete source tensor while preserving
its shape.

### bool2_union

~~~text
source_count: 2
dtype: bool for both sources
shape: identical per sample
dims: any
sources: unordered / commutative
full_partition: False
~~~

Computes elementwise logical OR.

### bool2_intersect

Same constraints as bool2_union, but computes elementwise logical AND.

### mat2_cwrotate

~~~text
source_count: 1
dims: exactly 2
dtype: unrestricted
full_partition: False
~~~

Rotates each instantiated matrix clockwise by 90 degrees.

### dim0_flip

~~~text
dims > 0
dtype unrestricted
~~~

Reverses values along axis 0.

### dim1_flip

~~~text
dims > 1
dtype unrestricted
~~~

Reverses values along axis 1.

### dim2_flip

~~~text
dims > 2
dtype unrestricted
~~~

Reverses values along axis 2.

### partition_bool_trim

~~~text
source_count: 1
dtype: bool
dims > 0
full_partition: True
output_count: 2
~~~

The operation is valid only when every training sample has at least one
all-False first or last boundary slice along at least one dimension.

For each sample it computes the tight N-dimensional bounding box containing all
True entries.

It produces:

~~~text
gene 1: int64 offset vector, one start index per dimension
gene 2: trimmed boolean structure
~~~

Examples:

~~~text
[False, True, True]

-> offset  [1]
-> data    [True, True]
~~~

and:

~~~text
[
    [False, False],
    [False, True]
]

-> offset  [1, 1]
-> data    [[True]]
~~~

For an all-False source, the offset is all zeros and the remaining structure is
empty along every dimension.

Because partition_bool_trim is marked full_partition=True, it is the only one
of this new operation group that SP_generate may use. The boolean set
operations, rotations, and flips remain GP-only transformations.
