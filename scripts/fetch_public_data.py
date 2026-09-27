from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default="data/ARC-AGI-2")
    args = parser.parse_args()

    destination = Path(args.destination)
    if (destination / ".git").exists():
        subprocess.run(
            ["git", "-C", str(destination), "pull", "--ff-only"],
            check=True,
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/arcprize/ARC-AGI-2.git",
                str(destination),
            ],
            check=True,
        )

    print(f"ARC-AGI-2 public data ready at {destination.resolve()}")


if __name__ == "__main__":
    main()
