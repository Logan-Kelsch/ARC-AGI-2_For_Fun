# Kaggle submission workflow

ARC Prize 2026 — ARC-AGI-2 is a notebook-only code competition.

The competition rerun supplies an unseen `arc-agi_test_challenges.json`. Your notebook must write:

```text
/kaggle/working/submission.json
```

For every task id and every test input, the JSON must contain exactly two grids:

```json
{
  "task_id": [
    {
      "attempt_1": [[0]],
      "attempt_2": [[0]]
    }
  ]
}
```

If either attempt exactly matches the ground-truth output for that test input, that output receives credit.

## Build locally

```bash
make setup
make data
make evaluate
make notebook
```

The generated `notebooks/submission.ipynb` embeds the local package so the Kaggle rerun requires no internet.

## Push notebook

Put a Kaggle token in:

```text
.kaggle/access_token
```

Edit `notebooks/kernel-metadata.json` and replace `REPLACE_WITH_YOUR_USERNAME`.

Then:

```bash
make submit
make status
```

After Kaggle finishes the notebook commit, use the Kaggle UI to submit the generated `submission.json`.

## Important

The ordinary `arc-agi_test_challenges.json` visible while developing is a placeholder. Kaggle swaps in the unseen competition tasks during rerun. Do not write logic that depends on placeholder task ids.
