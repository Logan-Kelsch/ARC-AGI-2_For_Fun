# ARC-AGI-3 For Fun

A small research scaffold for ARC Prize 2026 — ARC-AGI-3.

The immediate goal is intentionally modest: prove the full loop from a compartmentalized local policy to a valid Kaggle submission before adding grammatical search, genetic programming, learned world models, or MCTS.

## Architecture

```text
ARC frame/history
      ↓
observation helpers
      ↓
policy
      ↓
competition adapter (MyAgent)
      ↓
official ARC environment
      ↓
scorecard / transition evidence
```

The first policy is a deterministic **null baseline**. It uses no learning, no game-specific strategy, and no randomness. It exists to validate plumbing and provide a frozen baseline.

The intended research path is:

```text
perception
  → evidence memory
  → grammar-generated transition hypotheses
  → hypothesis scoring
  → internal world model
  → stochastic / MCTS planning
  → one real ARC action
```

Real environment actions are evidence. Future MCTS rollouts should happen in an inferred internal model rather than treating the hidden competition environment as a free simulator.

## WSL quick start

Requirements:
- Ubuntu/WSL
- Python 3.12
- git
- make

```bash
git clone <this-repo>
cd ARC-AGI-3_For_Fun
make setup
make list-games
make verify-local
```

To inspect one game:

```bash
make play-local GAME=ls20 STEPS=80
```

To build the Kaggle deployment notebook:

```bash
make notebook
```

Before the first Kaggle push, place your token in `.kaggle/access_token` and replace `REPLACE_WITH_YOUR_USERNAME` in `notebooks/kernel-metadata.json`.

Then:

```bash
make submit
make status
```

After the Kaggle notebook commit completes, deliberately submit its generated `submission.parquet` to the competition. That competition rerun is the hidden evaluation; local public games are for development and are not an exact proxy for the hidden leaderboard distribution.

## Teaching notebook

Open `notebooks/01_null_agent_walkthrough.ipynb`.

It walks from the null policy through the local evaluation loop and shows exactly where grammar search / world-model induction / MCTS will later plug in.

## Repository layout

```text
agent/my_agent.py                    thin competition-facing adapter
src/arc_fun/observations.py          frame/grid helpers
src/arc_fun/policy.py                null policy + policy protocol
src/arc_fun/evaluation.py            transition/episode summaries
scripts/play_local.py                real local ARC engine loop
scripts/build_submission_notebook.py Kaggle notebook generator
scripts/slim_framework.py            trims optional agent-framework imports
notebooks/01_null_agent_walkthrough.ipynb
notebooks/kernel-metadata.json
tests/
```

## Design rule

The local evaluator, teaching notebook, and Kaggle `MyAgent` must all call the same policy implementation. We do not want a notebook solver and a separate competition solver drifting apart.
