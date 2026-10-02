# ARC-AGI-2 Architecture

The current solver is an experimental **bidirectional symbolic program synthesis** system.

For implementation-level details, see:

~~~text
docs/GP_SP_ENVIRONMENT.md
~~~

## System view

~~~text
                 TRAINING INPUTS
                       |
                       v
                GP forward search
                       |
                       v
               candidate semantics
                       |
                       |
                exact matching
                       |
                       |
               required semantics
                       ^
                       |
               SP / ST backward
              reconstructive search
                       ^
                       |
                 TRAINING OUTPUTS


              solved Boolean proof
                       |
                       v
                  Kelschinator
                       |
                       v
                 executable program
                       |
                       v
                  TEST INPUT
                       |
                       v
                     y_hat
~~~

## Forward side: GP

GP grows symbolic programs from the input.

Each gene has:

~~~text
source
operation
dimensions
parameters
instantiated values across samples
~~~

The same gene index has the same semantic meaning for every training example.

GP can use reconstructive or information-losing operations because it is a candidate feature/program space.

## Backward side: SP

SP begins at the known output and explores reversible decompositions and alternate representations.

Operations are classified as:

~~~text
AND   children jointly reconstruct the source
OR    transformed representation is a reversible alternative
NULL  not a valid SP proof transformation
~~~

Only AND and OR operations grow SP.

## Proof side: ST

ST is a Boolean reconstructive proof.

A node can be solved:

~~~text
directly:
    GP gene + exact symbolic matching rule

or

indirectly:
    satisfied AND/OR derivation + known inverse operations
~~~

The default decomposition is based on output shape and categorical composition.

## Semantic meeting point

GP and ST expose compatible typed pools:

~~~python
get_GP_pool(...)
get_ST_unsovled_frontier(...)
~~~

Current exact matchers include:

~~~text
solve_0dim_1gene_basic
solve_2dim_1gene_basic
~~~

They search low-complexity rule families and accept only zero-residual behavior across every training sample.

## Final compilation

Once ST is fully solved:

~~~python
kelschinator = Kelschinator()
fit_success = kelschinator.fit(ST)
~~~

Kelschinator:

1. selects a satisfied reconstructive proof;
2. traces only the GP dependencies used by that proof;
3. freezes matching rules and inverse operations;
4. validates the distilled program on every training pair.

Then:

~~~python
y_hat = kelschinator.transform(X_test)
~~~

executes the synthesized program on an unseen input.

## Search philosophy

The current system prioritizes:

~~~text
exact semantics
minimum-complexity exact rules
explicit symbolic state
reconstructive proofs
training replay validation
~~~

rather than continuous similarity or a large black-box predictive model.

## Legacy architecture

The repository still contains earlier GP_Set, GP evaluation-tree, and loss-tree experiments.

Those remain useful historical/reference material but are not the current solver path.

Current work should generally target:

~~~text
notebooks/ops/environment.py
notebooks/ops/ops.py
notebooks/ops/inv_ops.py
notebooks/ops/solve.py
notebooks/ops/kelschinator.py
~~~
