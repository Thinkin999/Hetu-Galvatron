#!/usr/bin/env python3
"""Reuse the analyzer from experiments/."""

from pathlib import Path
import runpy
import sys


def main() -> None:
    script = Path(__file__).resolve().parents[1] / "experiments" / "analyze_results.py"
    if not script.exists():
        raise SystemExit(f"Analyzer not found: {script}")
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
