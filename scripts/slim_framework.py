from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "vendor" / "ARC-AGI-3-Agents" / "agents" / "__init__.py"

MINIMAL = '''from typing import Type

from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    "random": Random,
}

__all__ = ["Agent", "Playback", "Swarm", "Random", "AVAILABLE_AGENTS"]
'''

if not TARGET.exists():
    raise SystemExit(f"Framework not found at {TARGET}; run setup after cloning.")

TARGET.write_text(MINIMAL)
print(f"Slimmed {TARGET}")
