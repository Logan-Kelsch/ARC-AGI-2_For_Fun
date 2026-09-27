# Getting the ARC-AGI-3 arcade locally

This project uses the official `arc-agi` Python package. You do not manually download a ZIP of games.

The first cache pass uses `OperationMode.NORMAL`: it contacts the ARC service, discovers the currently public environments, and materializes each game locally by calling `arcade.make(...)`. The package caches those environments under the project environment cache (normally `environment_files/`).

After that, the evaluator can use `OperationMode.OFFLINE`, so the inner experiment loop does not need the service. The evaluator resolves `environment_files/` from the repository root rather than from the current working directory, so terminal scripts and notebooks see the same cache.

## 1. WSL prerequisites

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv git make
```

Clone the repository and enter it:

```bash
git clone https://github.com/Logan-Kelsch/ARC-AGI-3_For_Fun.git
cd ARC-AGI-3_For_Fun
```

## 2. Install the project environment

```bash
make setup
```

This creates `.venv/`, installs `arc-agi` plus the local analysis/test dependencies, and clones the official ARC-AGI-3 agent framework used by the competition-facing adapter.

## 3. Download/cache the public arcade

```bash
make arcade-cache
```

The command prints one `OK` / `FAIL` line per game. Verify the offline cache with:

```bash
make arcade-list
```

You can delete `environment_files/` and rerun `make arcade-cache` whenever you want to rebuild the cache.

## 4. Evaluate the null policy on any cached game

```bash
make evaluate-game GAME=ls20
```

Useful controls:

```bash
make evaluate-game GAME=ls20 POLICY=null SEED=7 STEPS=80
make evaluate-game GAME=ls20 RENDER=terminal-fast
```

The evaluator reports final state, levels completed, real environment actions, policy compute time, the official local ARC score, and the raw official scorecard payload.

The score is calculated by the official ARC package on a real public ARC game. Public/local games are not the hidden Kaggle test distribution, so this is a faithful local evaluation mechanism, not a promise of the same leaderboard score.

## 5. Swap the policy without changing the evaluator

Policies are registered in `src/arc_fun/policy_registry.py`.

Today the registry contains:

```text
null -> NullPolicy
```

Add a new policy factory to the registry, then run:

```bash
make evaluate-game GAME=ls20 POLICY=my-policy
```

The environment loop and scorecard stay fixed while the policy compartment changes. That is the intended plug-in boundary for future grammar / GP / MCTS systems.

## 6. Notebook view

Open `notebooks/02_arcade_policy_evaluator.ipynb`.

It lists cached games, exposes `GAME_ID`, `POLICY_NAME`, `SEED`, and `MAX_ACTIONS`, and calls the exact same `evaluate_game(...)` function as the CLI.
