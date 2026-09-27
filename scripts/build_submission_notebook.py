from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "submission.ipynb"
METADATA = ROOT / "notebooks" / "kernel-metadata.json"
AGENT = ROOT / "agent" / "my_agent.py"
PACKAGE = ROOT / "src" / "arc_fun"

ACCELERATOR = "t4"
ACCELERATORS = {
    "cpu": {"name": "none", "gpu": False},
    "t4": {"name": "nvidiaTeslaT4", "gpu": True},
    "p100": {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000", "gpu": True},
}


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


def package_payload() -> dict[str, str]:
    return {path.name: path.read_text() for path in sorted(PACKAGE.glob("*.py"))}


def build() -> dict:
    accel = ACCELERATORS[ACCELERATOR]
    agent_source = AGENT.read_text()
    package_sources = package_payload()

    install = code_cell(dedent(
        """
        !pip install --no-index --find-links \
            /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \
            arc-agi python-dotenv
        """
    ).strip())

    payload = code_cell(
        "AGENT_SOURCE = " + repr(agent_source) + "\n"
        "PACKAGE_SOURCES = " + repr(package_sources)
    )

    run = code_cell(dedent(
        """
        import os
        import shutil
        from pathlib import Path

        if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 --retry-max-time 600 http://gateway:8001/api/games

            framework = Path("/kaggle/working/ARC-AGI-3-Agents")
            shutil.copytree(
                "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents",
                framework,
                dirs_exist_ok=True,
            )

            package_dir = framework / "arc_fun"
            package_dir.mkdir(parents=True, exist_ok=True)
            for filename, source in PACKAGE_SOURCES.items():
                (package_dir / filename).write_text(source)

            template = framework / "agents" / "templates" / "my_agent.py"
            template.write_text(AGENT_SOURCE)

            (framework / "agents" / "__init__.py").write_text(
                '''from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            "random": Random,
            "myagent": MyAgent,
        }
        '''
            )

            (framework / ".env").write_text(
                '''SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        '''
            )

            %cd /kaggle/working/ARC-AGI-3-Agents
            !MPLBACKEND=agg python main.py --agent myagent
        """
    ).strip())

    dummy = code_cell(dedent(
        """
        import os
        if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            import pandas as pd
            pd.DataFrame(
                [["1_0", "1", True, 1]],
                columns=["row_id", "game_id", "end_of_game", "score"],
            ).to_parquet("/kaggle/working/submission.parquet", index=False)
        """
    ).strip())

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
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "cells": [
            markdown_cell(
                "# ARC-AGI-3 submission\n\nGenerated from the shared local policy code. "
                "Edit source files, not this notebook."
            ),
            install,
            payload,
            run,
            dummy,
        ],
    }


def main() -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(build(), indent=1))
    print(f"Wrote {NOTEBOOK}")

    if METADATA.exists():
        metadata = json.loads(METADATA.read_text())
        metadata["enable_gpu"] = ACCELERATORS[ACCELERATOR]["gpu"]
        METADATA.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
