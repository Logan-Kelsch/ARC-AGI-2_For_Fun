from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "arc_agi2_fun"
NOTEBOOK = ROOT / "notebooks" / "submission.ipynb"
SOLVER_NAME = "primitive"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    package_sources = {
        path.name: path.read_text()
        for path in sorted(PACKAGE.glob("*.py"))
    }

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "kaggle": {
                "accelerator": "none",
                "isInternetEnabled": False,
                "isGpuEnabled": False,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-2 submission\n\n"
                "Generated from the same solver package used for local evaluation."
            ),
            code_cell(
                "PACKAGE_SOURCES = " + repr(package_sources) + "\n"
                "SOLVER_NAME = " + repr(SOLVER_NAME)
            ),
            code_cell(
                """from pathlib import Path
import sys

package_dir = Path("/kaggle/working/arc_agi2_fun")
package_dir.mkdir(parents=True, exist_ok=True)
for filename, source in PACKAGE_SOURCES.items():
    (package_dir / filename).write_text(source)

sys.path.insert(0, "/kaggle/working")
"""
            ),
            code_cell(
                """from arc_agi2_fun.data import find_kaggle_competition_dir, load_kaggle_challenges
from arc_agi2_fun.registry import create_solver
from arc_agi2_fun.submission import build_submission, write_submission

competition_dir = find_kaggle_competition_dir()
tasks = load_kaggle_challenges(
    competition_dir / "arc-agi_test_challenges.json"
)
solver = create_solver(SOLVER_NAME)
submission = build_submission(tasks, solver)
output = write_submission(submission, "/kaggle/working/submission.json")

print(f"solver={solver.name}")
print(f"tasks={len(tasks)}")
print(f"wrote={output}")
"""
            ),
        ],
    }


def main() -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(build(), indent=1))
    print(f"Wrote {NOTEBOOK}")


if __name__ == "__main__":
    main()
