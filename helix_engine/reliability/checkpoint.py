"""
Checkpoint Management
=====================

Handles saving and loading of training checkpoints with:
- State preservation (model, optimizer, scheduler, step)
- Best checkpoint tracking
- Periodic checkpoints
- Resume support

Signature: checkpoint|v0.1.0|helix
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CheckpointState:
    """
    Complete state for a training checkpoint.
    """
    # Training progress
    step: int = 0
    epoch: int = 0

    # Core state
    model_state: Optional[Dict[str, Any]] = None
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None

    # Training state
    training_state: Dict[str, Any] = field(default_factory=dict)

    # Metrics at checkpoint
    metrics: Dict[str, float] = field(default_factory=dict)

    # Random states for reproducibility
    numpy_rng_state: Optional[Any] = None
    python_rng_state: Optional[Any] = None

    # Metadata
    created_at: str = ""
    run_id: str = ""
    checkpoint_type: str = "periodic"  # "periodic", "best", "last"


class CheckpointManager:
    """
    Manages checkpoints for a training run.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        run_id: str = "",
        keep_n_checkpoints: int = 5,
        best_metric: str = "negentropy",
        best_mode: str = "max",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_id = run_id
        self.keep_n_checkpoints = keep_n_checkpoints
        self.best_metric = best_metric
        self.best_mode = best_mode  # "max" or "min"

        # Track best value
        self.best_value: Optional[float] = None

        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def last_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "last.pt"

    @property
    def best_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best.pt"

    def checkpoint_path(self, step: int) -> Path:
        return self.checkpoint_dir / f"step_{step:08d}.pt"

    def save(
        self,
        state: CheckpointState,
        is_best: bool = False,
    ) -> Path:
        """
        Save a checkpoint.

        Always saves as last.pt, optionally saves as best.pt,
        and saves periodic checkpoints.
        """
        # Add metadata
        state.created_at = datetime.utcnow().isoformat()
        state.run_id = self.run_id

        # Capture random states
        import random
        state.python_rng_state = random.getstate()
        state.numpy_rng_state = np.random.get_state()

        # Convert to serializable format
        checkpoint_data = self._state_to_dict(state)

        # Save as last.pt
        self._save_checkpoint(checkpoint_data, self.last_checkpoint_path)
        state.checkpoint_type = "last"

        # Save as best.pt if this is the best
        if is_best:
            self._save_checkpoint(checkpoint_data, self.best_checkpoint_path)
            state.checkpoint_type = "best"

        # Save periodic checkpoint
        periodic_path = self.checkpoint_path(state.step)
        self._save_checkpoint(checkpoint_data, periodic_path)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return periodic_path

    def load(self, path: Optional[str] = None) -> Optional[CheckpointState]:
        """
        Load a checkpoint.

        If no path specified, loads the last checkpoint.
        """
        if path is None:
            if self.last_checkpoint_path.exists():
                path = str(self.last_checkpoint_path)
            else:
                return None
        else:
            path = str(path)

        if not os.path.exists(path):
            return None

        checkpoint_data = self._load_checkpoint(path)
        if checkpoint_data is None:
            return None

        return self._dict_to_state(checkpoint_data)

    def load_best(self) -> Optional[CheckpointState]:
        """Load the best checkpoint."""
        if self.best_checkpoint_path.exists():
            return self.load(str(self.best_checkpoint_path))
        return None

    def update_best(self, metrics: Dict[str, float]) -> bool:
        """
        Check if current metrics are best and update tracking.

        Returns True if this is the new best.
        """
        if self.best_metric not in metrics:
            return False

        current_value = metrics[self.best_metric]

        if self.best_value is None:
            self.best_value = current_value
            return True

        if self.best_mode == "max":
            is_best = current_value > self.best_value
        else:
            is_best = current_value < self.best_value

        if is_best:
            self.best_value = current_value

        return is_best

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = []

        for path in sorted(self.checkpoint_dir.glob("*.pt")):
            metadata = {
                "path": str(path),
                "name": path.name,
                "size_mb": path.stat().st_size / (1024 * 1024),
            }

            # Try to read step from filename
            if path.name.startswith("step_"):
                try:
                    step = int(path.stem.split("_")[1])
                    metadata["step"] = step
                except (ValueError, IndexError):
                    pass

            checkpoints.append(metadata)

        return checkpoints

    def restore_random_states(self, state: CheckpointState) -> None:
        """Restore random states for reproducibility."""
        import random

        if state.python_rng_state is not None:
            random.setstate(state.python_rng_state)

        if state.numpy_rng_state is not None:
            np.random.set_state(state.numpy_rng_state)

    def _save_checkpoint(self, data: Dict[str, Any], path: Path) -> None:
        """Save checkpoint data to file."""
        # Save to temp file first, then rename for atomicity
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "wb") as f:
            pickle.dump(data, f)
        shutil.move(str(temp_path), str(path))

    def _load_checkpoint(self, path: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data from file."""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, OSError, EOFError) as e:
            print(f"Warning: Failed to load checkpoint {path}: {e}")
            return None

    def _state_to_dict(self, state: CheckpointState) -> Dict[str, Any]:
        """Convert CheckpointState to dictionary."""
        return {
            "step": state.step,
            "epoch": state.epoch,
            "model_state": state.model_state,
            "optimizer_state": state.optimizer_state,
            "scheduler_state": state.scheduler_state,
            "training_state": state.training_state,
            "metrics": state.metrics,
            "numpy_rng_state": state.numpy_rng_state,
            "python_rng_state": state.python_rng_state,
            "created_at": state.created_at,
            "run_id": state.run_id,
            "checkpoint_type": state.checkpoint_type,
        }

    def _dict_to_state(self, data: Dict[str, Any]) -> CheckpointState:
        """Convert dictionary to CheckpointState."""
        return CheckpointState(
            step=data.get("step", 0),
            epoch=data.get("epoch", 0),
            model_state=data.get("model_state"),
            optimizer_state=data.get("optimizer_state"),
            scheduler_state=data.get("scheduler_state"),
            training_state=data.get("training_state", {}),
            metrics=data.get("metrics", {}),
            numpy_rng_state=data.get("numpy_rng_state"),
            python_rng_state=data.get("python_rng_state"),
            created_at=data.get("created_at", ""),
            run_id=data.get("run_id", ""),
            checkpoint_type=data.get("checkpoint_type", "periodic"),
        )

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old periodic checkpoints, keeping only the most recent N."""
        # Get all periodic checkpoints (exclude best.pt and last.pt)
        periodic = []
        for path in self.checkpoint_dir.glob("step_*.pt"):
            try:
                step = int(path.stem.split("_")[1])
                periodic.append((step, path))
            except (ValueError, IndexError):
                continue

        # Sort by step
        periodic.sort(key=lambda x: x[0])

        # Remove old ones
        while len(periodic) > self.keep_n_checkpoints:
            _, path = periodic.pop(0)
            path.unlink()
