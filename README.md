# ARC-AGI-2 For Fun

This repository has been rebuilt for **ARC Prize 2026 — ARC-AGI-2**.

ARC-AGI-2 is not an interactive environment. There are no actions, game states, or rollout server. Each task provides a few demonstration input/output grid pairs and one or more test input grids. The solver must infer a transformation and produce exactly two candidate output grids for every test input.

The core loop is now:

```text
task
  ↓
demonstration input/output pairs
  ↓
candidate program / transformation generation
  ↓
exact + diagnostic scoring on demonstrations
  ↓
candidate ranking / search
  ↓
two retained programs
  ↓
apply to test input(s)
  ↓
attempt_1 + attempt_2
  ↓
submission.json
```

This is deliberately aligned with grammar-guided genetic programming / stochastic tree search. A future MCTS does not search game states; it searches the tree of **partial programs produced by grammar rules**.

## Quick start in WSL

```bash
git clone https://github.com/Logan-Kelsch/ARC-AGI-3_For_Fun.git
cd ARC-AGI-3_For_Fun

make setup
make data
make test
make evaluate
```

`make data` clones the official public ARC-AGI-2 repository into `data/ARC-AGI-2`.

The official public dataset contains 1,000 training tasks and 120 public evaluation tasks. The evaluation set should be treated as held out during method development.

## Inspect a task

```bash
make list-tasks SPLIT=training
make inspect TASK=<task_id> SPLIT=training
```

## Custom tensor playground

For manual modeling experiments, open:

`notebooks/02_custom_tensor_playground.ipynb`

It lets you select any public task and example, render the input/output, convert the ARC grid to a lossless `H × W × 10` tensor, apply arbitrary NumPy/model operations, render the decoded prediction, and evaluate it against the known target using exact match and diagnostic cell accuracy.

The third dimension is one channel per ARC color. Model outputs may be one-hot tensors, probabilities, logits, or arbitrary channel scores; they are decoded with `argmax`.

## Evaluate a replaceable solver

Two solver compartments are included:

- `null`: identity + zero-grid sanity baseline.
- `primitive`: tiny program search over identity, flips, rotations, and transpose.

```bash
make evaluate SOLVER=null SPLIT=evaluation
make evaluate SOLVER=primitive SPLIT=evaluation
```

The primitive solver is **not intended to be competitive**. It demonstrates the architecture we need for the real project.

## Kaggle path

The competition rerun swaps in unseen `arc-agi_test_challenges.json` tasks. The output must be `submission.json`, include every task id, and provide exactly `attempt_1` and `attempt_2` for each test input.

```bash
make notebook
make submit
make status
```

See `docs/KAGGLE.md`.

## Repository structure

```text
src/arc_agi2_fun/
  data.py
  scoring.py
  programs.py
  search.py
  solver.py
  registry.py
  evaluation.py
  submission.py
  visualize.py
  tensors.py

scripts/
notebooks/
docs/
tests/
```

## Design rule

The evaluator and Kaggle notebook call the **same solver implementation**. Search algorithms may change; task loading, exact scoring, and submission formatting should remain fixed.

The repository name still contains `ARC-AGI-3` because renaming the GitHub repository is separate from changing its contents.
