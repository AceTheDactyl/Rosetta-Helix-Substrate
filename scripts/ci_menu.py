#!/usr/bin/env python3
"""
KIRA System CI Terminal
=======================

User-facing CLI menu that threads the main Rosetta-Helix workflows
into one place. All commands run inside the local .venv when invoked
via `make ci-menu` or `python scripts/ci_menu.py`.

Workflows:
  1) KIRA server (Flask)
  2) Helix training (helix CLI)
  3) Nightly training
  4) WUMBO APL integrated
  5) WUMBO N0 integrated
  6) Physics verification
  7) Nuclear spinner simulation
  8) Visualization server
  9) Python test suite
  0) Exit
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a subprocess, streaming output."""
    print("\n$", " ".join(cmd))
    try:
        return subprocess.call(cmd, cwd=str(cwd) if cwd else None)
    except KeyboardInterrupt:
        return 130


def ensure_venv() -> Path:
    venv = ROOT / ".venv"
    if not venv.exists():
        print("[!] .venv not found. Run `make install` first.")
    return venv


def menu() -> None:
    ensure_venv()
    while True:
        print("\nKIRA System CI Terminal")
        print("=======================")
        print("1) Start KIRA server (localhost:5000)")
        print("2) Helix training (configs/full.yaml)")
        print("3) Nightly training run")
        print("4) WUMBO APL integrated run")
        print("5) WUMBO N0 integrated run")
        print("6) Physics verification suite")
        print("7) Nuclear spinner simulation")
        print("8) Visualization server (port 8765)")
        print("9) Python tests (pytest -q)")
        print("0) Exit")
        choice = input("Select> ").strip()

        if choice == "1":
            run([sys.executable, "kira-local-system/kira_server.py"], cwd=ROOT)
        elif choice == "2":
            helix = ROOT / ".venv/bin/helix"
            if helix.exists():
                run([str(helix), "train", "--config", "configs/full.yaml"], cwd=ROOT)
            else:
                run([sys.executable, "train_helix.py"], cwd=ROOT)
        elif choice == "3":
            run([sys.executable, "nightly_training_runner.py"], cwd=ROOT)
        elif choice == "4":
            run([sys.executable, "run_wumbo_apl_integrated.py"], cwd=ROOT)
        elif choice == "5":
            run([sys.executable, "run_wumbo_n0_integrated.py"], cwd=ROOT)
        elif choice == "6":
            run([sys.executable, "verify_physics.py"], cwd=ROOT)
        elif choice == "7":
            run([sys.executable, "scripts/nuclear_spinner.py"], cwd=ROOT)
        elif choice == "8":
            run([sys.executable, "visualization_server.py", "--port", "8765"], cwd=ROOT)
        elif choice == "9":
            run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
        elif choice == "0":
            print("Goodbye.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    menu()
