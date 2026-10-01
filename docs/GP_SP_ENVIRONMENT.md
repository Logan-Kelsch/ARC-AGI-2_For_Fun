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

SP operations are classified by partition semantics:

~~~text
and   child outputs are jointly required to reconstruct/validate the parent
or    a reversible transformed representation is an alternative way to solve it
null  not a valid SP proof transformation
~~~

SP may use only AND/OR operations. NULL operations remain GP-only.

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

ST is a Boolean proof structure over SP targets rather than a simple dependency
tree. It records direct GP matches plus nested AND/OR alternatives created by
reversible SP transformations.

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
partition = "and" | "or" | "null"
inverse_op
output_count
~~~

GP may use any registered transformation.

SP may use only AND/OR operations. NULL operations are excluded because their
result does not provide a valid reversible proof of the source.

## Initial operations

### partition_shape

~~~python
h_gidx, w_gidx = partition_shape(meta, X, source_idx)
~~~

For a 2D source it generates two scalar np.int64 genes:

~~~text
height
width
~~~

Both have dims=0.

On ST, shape is represented as:

~~~text
shape = h AND w
~~~

and inv_partition_shape(h, w, composite) verifies that the reconstructed
composite has the solved dimensions.

### partition_composite

~~~python
generated_gidx = partition_composite(
    meta,
    X,
    source_idx,
)
~~~

The operation finds the sorted union of colors used across all training samples.

For each color it creates two separate gene-major outputs:

~~~text
color_k_id        scalar np.int64
color_k_presence  2D boolean matrix
~~~

So a three-color source creates six genes:

~~~text
color_1_id
color_1_presence
color_2_id
color_2_presence
color_3_id
color_3_presence
~~~

The color-ID gene is constant across samples. If that color is absent from a
particular sample, its corresponding presence gene contains an all-False matrix
with that sample's spatial shape.

This keeps the number and meaning of gene indices aligned across all samples
while exposing each categorical component independently to GP/SP matching.

partition_composite is a dynamic-output operation:

~~~text
output_count = 2 * number of distinct colors across all samples
~~~

Both initial operations are AND partitions and are legal on SP.

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

The initial gene table is variable-length because the composite partition now
creates one pair per distinct color:

~~~text
gidx   op                     source   dims

0      raw_input/output       -1       2
1      partition_shape         0       0   h
2      partition_shape         0       0   w
3      partition_composite     0       0   first color ID
4      partition_composite     0       2   first color presence
5      partition_composite     0       0   second color ID
6      partition_composite     0       2   second color presence
...    ...                      ...     ...
~~~

For C distinct colors, initialization creates:

~~~text
3 + 2*C total genes
~~~

on that side.

The initial ST is derived directly from however many SP genes this produces.

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
3. SP may only use AND/OR partition operations;
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
    partition="and",
    inverse_op="inv_example",
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
source dims == 2
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
Before selecting a candidate, generation asks the operation how many outputs it
would produce on that exact source. This supports both fixed-output operations
and dynamic operations such as partition_composite.

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

SP generation automatically excludes NULL operations and may sample reversible
AND/OR transformations.

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
partition: null/or as appropriate
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
partition: null/or as appropriate
~~~

Computes elementwise logical OR.

### bool2_intersect

Same constraints as bool2_union, but computes elementwise logical AND.

### mat2_cwrotate

~~~text
source_count: 1
dims: exactly 2
dtype: unrestricted
partition: null/or as appropriate
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
partition: and
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

partition_bool_trim is an AND partition. Reversible unary transforms such as
rotation and flips are OR partitions and are also SP-legal. Multi-source
boolean set operations remain NULL/GP-only.


## bool_sum

~~~text
source_count: 1
dtype: bool
dims: any
partition: null/or as appropriate
output dims: 0
~~~

Counts every True value in the instantiated source and emits one scalar
\`np.int64\` per training sample.

Examples:

~~~text
[True, False, True] -> 2

[[True, True],
 [False, True]] -> 3
~~~

The operation is GP-only.

## bool_cavity

~~~text
source_count: 1
dtype: bool
dims: any
requires: at least one True in every sample
partition: null/or as appropriate
output shape: identical to source
~~~

Produces a boolean mask containing only False regions that are fully enclosed
by True values.

Connectivity is axis-adjacent:

~~~text
1D -> left/right
2D -> 4-connectivity
3D -> 6-connectivity
N-D -> +/- 1 along one axis at a time
~~~

A False region is a cavity exactly when it cannot reach any boundary cell
through axis-adjacent False cells.

Examples:

~~~text
[False, True, False, True, False]

->

[False, False, True, False, False]
~~~

~~~text
[False, True, False, False, True]

->

[False, False, True, True, False]
~~~

~~~text
[
    [False, True,  False],
    [True,  False, True ],
    [False, True,  False],
]

->

[
    [False, False, False],
    [False, True,  False],
    [False, False, False],
]
~~~

The center remains a cavity even though it is diagonally adjacent to boundary
False cells, because diagonal adjacency is not used.

A scalar True is valid and produces scalar False. A source sample containing no
True values is not generation-valid for bool_cavity.

The operation is GP-only.


## Boolean Solution Tree

ST now represents a proof that the complete output can be reconstructed.

Every semantic ST node has two ways to become solved:

1. direct: an exact GP gene is attached through gp_gidx;
2. derived: one of its Boolean derivation branches evaluates True.

The initial environment is centered on:

~~~text
root
  =
shape
  AND
composite
~~~

Composite is itself:

~~~text
inv_partition_composite
AND color_0_id
AND color_0_presence
AND color_1_id
AND color_1_presence
AND ...
~~~

The inverse-operation requirements are innate: the transformation code is part
of the grammar and does not need to be discovered by GP.

### OR transformations

A reversible transform creates an alternative path instead of replacing the
original target.

If a composite child C is rotated clockwise into R:

~~~text
C
OR
(
    inv_mat2_cwrotate
    AND R
)
~~~

The direct C target may remain unsolved. If R is solved by a GP gene, the known
inverse makes C logically solved.

This propagates upward through the Boolean tree.

For example:

~~~text
shape: solved
AND
(
    composite leaf C: unsolved directly
    OR
    (
        inverse rotate: innately solved
        AND rotated C: solved
    )
)
~~~

is sufficient to solve the relevant composite branch and therefore the root.

### Partition classes

AND
: A decomposition whose required outputs are jointly used to explain the
  source. partition_shape and partition_composite are the initial examples.

OR
: A reversible alternate representation of one source. bool_complement,
  mat2_cwrotate, and axis flips are examples.

NULL
: A transformation that does not create a valid reversible single-source proof
  branch. Multi-source bool2_union/intersect are NULL. Information-losing
  bool_sum/bool_cavity are also NULL so ST cannot falsely infer their source
  from insufficient information.

### Inverse operations

Reverse functions live in:

~~~text
notebooks/ops/inv_ops.py
~~~

and are registered in INV_OP_REGISTRY.

Important examples:

~~~python
inv_partition_shape(h, w, composite)
inv_partition_composite(color0, mask0, color1, mask1, ...)
inv_bool_complement(value)
inv_mat2_cwrotate(value)
inv_dim0_flip(value)
inv_dim1_flip(value)
inv_dim2_flip(value)
~~~

NULL operations also have relation-checking inverse helpers for completeness,
but those inverses are marked non-reconstructive and are never used by ST as
solution branches.


## indiv_1dim

~~~text
source_count: 1
dims: exactly 1
dtype: unrestricted
partition: and
output_count: dynamic = source length L
inverse: inv_indiv_1dim
~~~

indiv_1dim splits a one-dimensional gene into one scalar gene per element
position.

~~~text
[a, b, c]

-> scalar a
-> scalar b
-> scalar c
~~~

Every training sample must have the same 1D length so generated position k has
the same semantic meaning across all samples.

On SP this creates the proof alternative:

~~~text
source
OR
(
    inv_indiv_1dim
    AND element_0
    AND element_1
    AND ...
    AND element_L-1
)
~~~

Therefore all generated element genes are jointly required to reconstruct the
source through the known inverse.

inv_indiv_1dim simply reassembles the scalar elements in positional order into
the original 1D NumPy array.


## Unresolved ST frontier

Use:

~~~python
frontier = get_ST_unsovled_frontier(
    ST,
    SP_X,
    min_dim=None,
    max_dim=None,
)
~~~

to retrieve the instantiated SP data that still represents unresolved concrete
solution-tree nodes.

The return value is always a 1D NumPy object array:

~~~text
frontier.shape == (L,)
frontier.dtype == object
~~~

Each entry is one complete SP gene across all training samples, equivalent to:

~~~python
SP_X[sp_gidx]
~~~

for that frontier node.

Logical helper nodes such as shape and composite are not returned because they
do not have instantiated SP data.

The traversal keeps unresolved concrete parents even when they also have OR
alternatives. For example:

~~~text
C
OR
(
    inv_rotate
    AND R
)
~~~

places both C and R in the unresolved frontier until C becomes solved through
either route.

Once a node is solved, its entire proof subtree is pruned from the frontier.

Dimension filters are inclusive:

~~~python
# scalar int/float/bool genes only
get_ST_unsovled_frontier(
    ST,
    SP_X,
    min_dim=0,
    max_dim=0,
)

# matrices only
get_ST_unsovled_frontier(
    ST,
    SP_X,
    min_dim=2,
    max_dim=2,
)

# all 1D and higher data
get_ST_unsovled_frontier(
    ST,
    SP_X,
    min_dim=1,
)
~~~

The exact requested name get_ST_unsovled_frontier is retained. A correctly
spelled alias, get_ST_unsolved_frontier, is exported as well.
