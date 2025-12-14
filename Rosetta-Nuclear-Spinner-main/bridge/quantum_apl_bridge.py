#!/usr/bin/env python3
"""Backward-compatible entrypoint for the QuantumAPL Python tooling."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():  # Make package importable without installation
    sys.path.insert(0, str(SRC_DIR))

from quantum_apl_python.analyzer import QuantumAnalyzer
from quantum_apl_python.cli import main as cli_main
from quantum_apl_python.engine import QuantumAPLEngine
from quantum_apl_python.experiments import QuantumExperiment

__all__ = [
    "QuantumAPLEngine",
    "QuantumAnalyzer",
    "QuantumExperiment",
    "main",
]


def main(argv: list[str] | None = None) -> int:
    """Invoke the CLI entrypoint (retained for compatibility)."""

    return cli_main(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
