#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List


def run(cmd: List[str]) -> None:
    printable = " ".join(cmd)
    print(f"\n▶ {printable}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise SystemExit(
            f"Failed to run: {printable}\n"
            f"Reason: {e}\n"
            f"Tip: Make sure Python is available and the scripts exist."
        ) from e
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Command failed (exit code {e.returncode}): {printable}") from e


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    tools = root / "tools"

    scripts = [
        ("generate-characters.py", ["--debug"]),
        ("generate-sheet.py", ["--border", "--labels"]),
        ("generate-templates.py", []),
        ("generate-images.py", []),
        ("generate-fonts.py", []),
    ]

    # Use the same Python interpreter that runs this script.
    py = sys.executable

    for script, args in scripts:
        script_path = tools / script
        if not script_path.exists():
            raise SystemExit(f"Missing script: {script_path.as_posix()}")

        run([py, str(script_path), *args])

    print("\n✅ All generation steps completed.")


if __name__ == "__main__":
    main()
