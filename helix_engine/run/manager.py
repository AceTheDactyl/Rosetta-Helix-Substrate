"""
Run Manager
===========

Manages training runs: creation, saving, loading, resumption.

Signature: run-manager|v0.1.0|helix
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from helix_engine.core.contract import (
    EnvironmentSnapshot,
    ExitCode,
    GateResult,
    RunConfig,
    RunResult,
    RunStatus,
)
from helix_engine.run.layout import RunLayout


class RunManager:
    """
    Manages training run lifecycle:
    - Creating run directories
    - Saving configs and environment
    - Writing metrics
    - Managing checkpoints
    - Generating reports
    """

    def __init__(self, base_dir: str = "runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run(self, config: RunConfig) -> tuple[RunLayout, RunResult]:
        """
        Create a new training run.

        Returns (layout, result) tuple.
        """
        # Create layout
        layout = RunLayout(config.run_id, str(self.base_dir))
        layout.create()

        # Capture environment
        env = EnvironmentSnapshot.capture()

        # Save config
        with open(layout.config_path, "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)

        # Save environment
        with open(layout.env_path, "w") as f:
            json.dump(env.to_dict(), f, indent=2)

        # Initialize result
        result = RunResult(
            run_id=config.run_id,
            config=config,
            status=RunStatus.PENDING,
            environment=env,
            artifacts_dir=str(layout.run_dir),
            config_path=str(layout.config_path),
            metrics_path=str(layout.metrics_path),
        )

        return layout, result

    def start_run(self, result: RunResult) -> RunResult:
        """Mark a run as started."""
        result.status = RunStatus.RUNNING
        result.started_at = datetime.utcnow().isoformat()
        return result

    def complete_run(
        self,
        result: RunResult,
        layout: RunLayout,
        exit_code: ExitCode = ExitCode.SUCCESS,
        error_message: str = "",
    ) -> RunResult:
        """
        Mark a run as completed and generate final report.
        """
        result.completed_at = datetime.utcnow().isoformat()

        # Calculate duration
        if result.started_at:
            started = datetime.fromisoformat(result.started_at)
            completed = datetime.fromisoformat(result.completed_at)
            result.duration_seconds = (completed - started).total_seconds()

        result.exit_code = exit_code
        result.error_message = error_message

        if exit_code == ExitCode.SUCCESS:
            result.status = RunStatus.COMPLETED
        elif exit_code == ExitCode.INTERRUPTED:
            result.status = RunStatus.INTERRUPTED
        else:
            result.status = RunStatus.FAILED

        # Update checkpoint paths
        if layout.last_checkpoint_path.exists():
            result.checkpoints["last"] = str(layout.last_checkpoint_path)
        if layout.best_checkpoint_path.exists():
            result.checkpoints["best"] = str(layout.best_checkpoint_path)

        # Save final report
        result.save_report(str(layout.report_path))

        return result

    def load_run(self, run_id: str) -> Optional[tuple[RunLayout, RunResult]]:
        """
        Load an existing run.

        Returns (layout, result) or None if not found.
        """
        layout = RunLayout(run_id, str(self.base_dir))

        if not layout.exists():
            return None

        # Load config
        if layout.config_path.exists():
            with open(layout.config_path) as f:
                config_data = yaml.safe_load(f)
            config = RunConfig.from_dict(config_data)
        else:
            config = RunConfig(run_id=run_id)

        # Load environment
        env = None
        if layout.env_path.exists():
            with open(layout.env_path) as f:
                env_data = json.load(f)
            env = EnvironmentSnapshot(**env_data)

        # Load report if exists
        if layout.report_path.exists():
            with open(layout.report_path) as f:
                report_data = json.load(f)

            result = RunResult(
                run_id=run_id,
                config=config,
                status=report_data.get("status", RunStatus.PENDING),
                exit_code=ExitCode(report_data.get("exit_code", 0)),
                error_message=report_data.get("error_message", ""),
                started_at=report_data.get("started_at", ""),
                completed_at=report_data.get("completed_at", ""),
                duration_seconds=report_data.get("duration_seconds", 0.0),
                environment=env,
                total_steps=report_data.get("total_steps", 0),
                final_metrics=report_data.get("final_metrics", {}),
                gates_passed=report_data.get("gates_passed", False),
                artifacts_dir=str(layout.run_dir),
                config_path=str(layout.config_path),
                metrics_path=str(layout.metrics_path),
                checkpoints=report_data.get("checkpoints", {}),
                report_path=str(layout.report_path),
            )
        else:
            result = RunResult(
                run_id=run_id,
                config=config,
                environment=env,
                artifacts_dir=str(layout.run_dir),
                config_path=str(layout.config_path),
                metrics_path=str(layout.metrics_path),
            )

        return layout, result

    def list_runs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all runs, optionally filtered by status.
        """
        runs = []

        for run_dir in sorted(self.base_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            run_id = run_dir.name
            report_path = run_dir / "report.json"

            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)

                if status is None or report.get("status") == status:
                    runs.append({
                        "run_id": run_id,
                        "status": report.get("status"),
                        "started_at": report.get("started_at"),
                        "completed_at": report.get("completed_at"),
                        "exit_code": report.get("exit_code"),
                        "gates_passed": report.get("gates_passed"),
                    })
            else:
                # Run exists but no report yet
                if status is None or status == RunStatus.PENDING:
                    runs.append({
                        "run_id": run_id,
                        "status": RunStatus.PENDING,
                    })

        return runs

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and all its artifacts."""
        layout = RunLayout(run_id, str(self.base_dir))
        if layout.exists():
            layout.clean()
            return True
        return False

    def write_metrics(
        self,
        layout: RunLayout,
        step: int,
        metrics: Dict[str, Any],
    ) -> None:
        """
        Append metrics to the metrics.jsonl file.
        """
        entry = {
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            **metrics,
        }

        with open(layout.metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def write_summary(
        self,
        layout: RunLayout,
        summary: Dict[str, Any],
    ) -> None:
        """
        Write final metrics summary.
        """
        with open(layout.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def write_gates(
        self,
        layout: RunLayout,
        gate_results: List[GateResult],
    ) -> None:
        """
        Write gate evaluation results.
        """
        results = [
            {
                "name": g.name,
                "passed": g.passed,
                "expected": g.expected,
                "actual": g.actual,
                "message": g.message,
            }
            for g in gate_results
        ]

        with open(layout.gates_path, "w") as f:
            json.dump({
                "gates": results,
                "all_passed": all(g.passed for g in gate_results),
                "timestamp": datetime.utcnow().isoformat(),
            }, f, indent=2)

    def log_event(
        self,
        layout: RunLayout,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Log a structured event to events.jsonl.
        """
        entry = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }

        with open(layout.events_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
