# ARC-AGI-2 architecture

## What was removed

The previous repository targeted ARC-AGI-3 and therefore had concepts that no longer belong:

- interactive environments
- action enums
- game state
- transition histories
- environment scorecards
- world-model rollouts over action sequences
- an Agent adapter

ARC-AGI-2 is static program induction.

## Fixed outer loop

For one task:

```text
D = {(X1,Y1), ... , (Xn,Yn)}
test inputs = {T1, ... , Tm}
```

A candidate program `p` is evaluated on the demonstrations:

```text
p(Xi) -> Y_hat_i
```

Primary fitness:

```text
F_exact(p) = (1/n) * sum_i 1[p(Xi) = Yi]
```

The solver retains two candidate programs because the competition accepts two attempts per test input.

## Mapping to grammatical optimization

```text
grammar G
   ↓
partial program
   ↓ production rule
larger partial program
   ↓
...
   ↓
executable program p
   ↓
evaluate on demonstrations
   ↓
fitness / novelty / complexity / robustness
```

Genetic programming can mutate/crossover complete syntax trees.

MCTS can treat grammar derivation as a search tree:

- state: partial program
- action: legal grammar production
- transition: extend the program
- terminal state: executable program
- reward: demonstration performance plus regularization/generalization signals

This is where the stochastic MCTS idea belongs in ARC-AGI-2.

## Preventing task-level overfit

A program that fits a few demonstrations may still infer the wrong rule. Future fitness should consider:

- minimum description length / complexity penalty
- leave-one-demonstration-out consistency
- object-level invariants
- transformation equivalence classes
- robustness to synthetic variants
- reusable learned macros across public training tasks
- diversity between attempt 1 and attempt 2

The public evaluation set should remain an outer validation set rather than an inner-loop search signal.

## Current scaffold

The current `PrimitiveSearchSolver` searches only seven simple geometric programs. It exists to prove the contract, not to define the final approach.

The next serious layer should introduce a typed DSL and a search engine over its syntax trees.
