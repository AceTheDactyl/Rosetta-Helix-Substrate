"""
Run Directory Layout
====================

Standardized filesystem layout for training runs:

runs/<run_id>/
    resolved_config.yaml     # Exact config used
    env.json                 # Environment snapshot
    logs/
        train.log            # Human-readable training log
        events.jsonl         # Structured event log
    metrics/
        metrics.jsonl        # Step-by-step metrics
        summary.json         # Final metrics summary
    checkpoints/
        last.pt              # Most recent checkpoint
        best.pt              # Best checkpoint (by gate metric)
        step_*.pt            # Periodic checkpoints
    eval/
        gates.json           # Gate evaluation results
        results.json         # Full evaluation results
    exports/
        model.onnx           # Exported model (optional)
        model.pt             # TorchScript (optional)

Signature: run-layout|v0.1.0|helix
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


class RunLayout:
    """
    Manages the standardized directory layout for a training run.
    """

    def __init__(self, run_id: str, base_dir: str = "runs"):
        self.run_id = run_id
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / run_id

    # -------------------------------------------------------------------------
    # Directory paths
    # -------------------------------------------------------------------------

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def eval_dir(self) -> Path:
        return self.run_dir / "eval"

    @property
    def exports_dir(self) -> Path:
        return self.run_dir / "exports"

    # -------------------------------------------------------------------------
    # File paths
    # -------------------------------------------------------------------------

    @property
    def config_path(self) -> Path:
        return self.run_dir / "resolved_config.yaml"

    @property
    def env_path(self) -> Path:
        return self.run_dir / "env.json"

    @property
    def train_log_path(self) -> Path:
        return self.logs_dir / "train.log"

    @property
    def events_path(self) -> Path:
        return self.logs_dir / "events.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.metrics_dir / "metrics.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.metrics_dir / "summary.json"

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoints_dir / "last.pt"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoints_dir / "best.pt"

    @property
    def gates_path(self) -> Path:
        return self.eval_dir / "gates.json"

    @property
    def eval_results_path(self) -> Path:
        return self.eval_dir / "results.json"

    @property
    def report_path(self) -> Path:
        return self.run_dir / "report.json"

    def checkpoint_path(self, step: int) -> Path:
        """Get path for a step-specific checkpoint."""
        return self.checkpoints_dir / f"step_{step:08d}.pt"

    def export_path(self, format: str) -> Path:
        """Get path for an exported model."""
        extensions = {
            "onnx": "onnx",
            "torchscript": "pt",
            "pt": "pt",
            "bundle": "zip",
        }
        ext = extensions.get(format, format)
        return self.exports_dir / f"model.{ext}"

    # -------------------------------------------------------------------------
    # Directory management
    # -------------------------------------------------------------------------

    def create(self) -> "RunLayout":
        """Create all directories for the run."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.eval_dir.mkdir(exist_ok=True)
        self.exports_dir.mkdir(exist_ok=True)
        return self

    def exists(self) -> bool:
        """Check if run directory exists."""
        return self.run_dir.exists()

    def clean(self) -> None:
        """Remove the run directory and all contents."""
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)

    def list_checkpoints(self) -> list[Path]:
        """List all checkpoint files."""
        if not self.checkpoints_dir.exists():
            return []
        return sorted(self.checkpoints_dir.glob("*.pt"))

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the most recent checkpoint (last.pt or highest step)."""
        if self.last_checkpoint_path.exists():
            return self.last_checkpoint_path

        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[-1]
        return None

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate that the run directory has the minimum required structure.

        Returns (is_valid, list_of_issues)
        """
        issues = []

        if not self.run_dir.exists():
            issues.append(f"Run directory does not exist: {self.run_dir}")
            return False, issues

        # Check required files
        required = [
            (self.config_path, "resolved_config.yaml"),
            (self.env_path, "env.json"),
        ]

        for path, name in required:
            if not path.exists():
                issues.append(f"Missing required file: {name}")

        return len(issues) == 0, issues

    def __repr__(self) -> str:
        return f"RunLayout(run_id={self.run_id!r}, run_dir={self.run_dir!r})"
