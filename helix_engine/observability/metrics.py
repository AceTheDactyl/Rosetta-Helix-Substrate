"""
Metrics Collection and Writing
==============================

Collects and writes training metrics:
- Step-by-step metrics (metrics.jsonl)
- Running statistics
- Final summary

Signature: metrics|v0.1.0|helix
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetricsCollector:
    """
    Collects and aggregates training metrics.
    """

    def __init__(self):
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.step_metrics: List[Dict[str, Any]] = []
        self.current_step: int = 0

    def record(self, step: int, metrics: Dict[str, float]) -> None:
        """Record metrics for a training step."""
        self.current_step = step

        entry = {
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            **metrics,
        }
        self.step_metrics.append(entry)

        # Update history for aggregation
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.history[key].append(float(value))

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the most recent metrics."""
        if self.step_metrics:
            return self.step_metrics[-1]
        return None

    def get_running_mean(self, key: str, window: int = 100) -> Optional[float]:
        """Get running mean for a metric."""
        if key not in self.history:
            return None
        values = self.history[key][-window:]
        if not values:
            return None
        return sum(values) / len(values)

    def get_statistics(self, key: str) -> Optional[Dict[str, float]]:
        """Get statistics for a metric."""
        if key not in self.history or not self.history[key]:
            return None

        values = self.history[key]
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "latest": values[-1],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        summary = {
            "total_steps": self.current_step,
            "metrics_recorded": len(self.step_metrics),
            "metrics": {},
        }

        for key in self.history:
            stats = self.get_statistics(key)
            if stats:
                summary["metrics"][key] = stats

        return summary

    def clear(self) -> None:
        """Clear all collected metrics."""
        self.history.clear()
        self.step_metrics.clear()
        self.current_step = 0


class MetricsWriter:
    """
    Writes metrics to files in various formats.
    """

    def __init__(self, metrics_dir: str):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # JSONL file for step-by-step metrics
        self.jsonl_path = self.metrics_dir / "metrics.jsonl"
        self.summary_path = self.metrics_dir / "summary.json"

        # File handle for streaming writes
        self._jsonl_handle = None

    def open(self) -> None:
        """Open metrics files for writing."""
        self._jsonl_handle = open(self.jsonl_path, "a")

    def close(self) -> None:
        """Close metrics files."""
        if self._jsonl_handle:
            self._jsonl_handle.close()
            self._jsonl_handle = None

    def write_step(self, step: int, metrics: Dict[str, Any]) -> None:
        """Write metrics for a single step."""
        entry = {
            "step": step,
            "timestamp": datetime.utcnow().isoformat(),
            **metrics,
        }

        if self._jsonl_handle:
            self._jsonl_handle.write(json.dumps(entry) + "\n")
            self._jsonl_handle.flush()

    def write_summary(self, summary: Dict[str, Any]) -> None:
        """Write final metrics summary."""
        summary["generated_at"] = datetime.utcnow().isoformat()
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def read_metrics(self) -> List[Dict[str, Any]]:
        """Read all metrics from JSONL file."""
        metrics = []
        if self.jsonl_path.exists():
            with open(self.jsonl_path) as f:
                for line in f:
                    if line.strip():
                        metrics.append(json.loads(line))
        return metrics

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class MetricsAggregator:
    """
    Aggregates metrics across multiple runs for comparison.
    """

    def __init__(self):
        self.runs: Dict[str, Dict[str, Any]] = {}

    def add_run(self, run_id: str, summary: Dict[str, Any]) -> None:
        """Add a run's summary to the aggregator."""
        self.runs[run_id] = summary

    def compare(self, metric_name: str) -> Dict[str, Any]:
        """Compare a specific metric across runs."""
        comparison = {}

        for run_id, summary in self.runs.items():
            metrics = summary.get("metrics", {})
            if metric_name in metrics:
                comparison[run_id] = metrics[metric_name]

        if not comparison:
            return {"error": f"Metric '{metric_name}' not found in any run"}

        # Find best/worst
        values = [(run_id, data.get("mean", 0)) for run_id, data in comparison.items()]
        sorted_values = sorted(values, key=lambda x: x[1], reverse=True)

        return {
            "metric": metric_name,
            "runs": comparison,
            "best_run": sorted_values[0][0] if sorted_values else None,
            "worst_run": sorted_values[-1][0] if sorted_values else None,
        }

    def get_leaderboard(self, metric_name: str, ascending: bool = False) -> List[Dict[str, Any]]:
        """Get a sorted leaderboard for a metric."""
        entries = []

        for run_id, summary in self.runs.items():
            metrics = summary.get("metrics", {})
            if metric_name in metrics:
                entries.append({
                    "run_id": run_id,
                    "value": metrics[metric_name].get("mean", 0),
                    **metrics[metric_name],
                })

        return sorted(entries, key=lambda x: x["value"], reverse=not ascending)
