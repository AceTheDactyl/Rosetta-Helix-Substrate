"""
Determinism Management
======================

Controls reproducibility through:
- Seed management
- Random state control
- Tolerance windows for floating point

Signature: determinism|v0.1.0|helix
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np


class DeterminismManager:
    """
    Manages deterministic training settings.
    """

    def __init__(
        self,
        seed: int = 42,
        deterministic: bool = False,
        tolerance: float = 1e-6,
    ):
        self.seed = seed
        self.deterministic = deterministic
        self.tolerance = tolerance
        self._initial_states: Dict[str, Any] = {}

    def setup(self) -> None:
        """
        Set up deterministic settings.

        Call this at the start of training.
        """
        # Set seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        # Capture initial states
        self._initial_states = {
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }

        # Try to set torch seed if available
        try:
            import torch
            torch.manual_seed(self.seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)

                if self.deterministic:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False

            self._initial_states["torch_rng"] = torch.get_rng_state()
            if torch.cuda.is_available():
                self._initial_states["cuda_rng"] = torch.cuda.get_rng_state_all()

        except ImportError:
            pass

    def reset(self) -> None:
        """
        Reset to initial random states.

        Useful for ensuring reproducibility across runs.
        """
        if "python_rng" in self._initial_states:
            random.setstate(self._initial_states["python_rng"])

        if "numpy_rng" in self._initial_states:
            np.random.set_state(self._initial_states["numpy_rng"])

        try:
            import torch
            if "torch_rng" in self._initial_states:
                torch.set_rng_state(self._initial_states["torch_rng"])
            if "cuda_rng" in self._initial_states:
                torch.cuda.set_rng_state_all(self._initial_states["cuda_rng"])
        except ImportError:
            pass

    def get_state(self) -> Dict[str, Any]:
        """Get current random states."""
        state = {
            "seed": self.seed,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
        }

        try:
            import torch
            state["torch_rng"] = torch.get_rng_state()
            if torch.cuda.is_available():
                state["cuda_rng"] = torch.cuda.get_rng_state_all()
        except ImportError:
            pass

        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore random states."""
        if "python_rng" in state:
            random.setstate(state["python_rng"])

        if "numpy_rng" in state:
            np.random.set_state(state["numpy_rng"])

        try:
            import torch
            if "torch_rng" in state:
                torch.set_rng_state(state["torch_rng"])
            if "cuda_rng" in state:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
        except ImportError:
            pass

    def check_within_tolerance(
        self,
        value: float,
        reference: float,
        name: str = "",
    ) -> tuple[bool, str]:
        """
        Check if value is within tolerance of reference.

        Returns (is_within, message)
        """
        diff = abs(value - reference)
        relative_diff = diff / max(abs(reference), 1e-10)

        if diff <= self.tolerance:
            return True, f"{name}: {value:.10f} == {reference:.10f} (diff={diff:.2e})"

        return False, f"{name}: {value:.10f} != {reference:.10f} (diff={diff:.2e}, rel={relative_diff:.2e})"

    def verify_reproducibility(
        self,
        run1_metrics: Dict[str, float],
        run2_metrics: Dict[str, float],
    ) -> tuple[bool, list[str]]:
        """
        Verify that two runs produced the same results within tolerance.

        Returns (all_match, list_of_mismatches)
        """
        mismatches = []

        all_keys = set(run1_metrics.keys()) | set(run2_metrics.keys())

        for key in all_keys:
            if key not in run1_metrics:
                mismatches.append(f"Missing in run1: {key}")
                continue
            if key not in run2_metrics:
                mismatches.append(f"Missing in run2: {key}")
                continue

            within, msg = self.check_within_tolerance(
                run1_metrics[key],
                run2_metrics[key],
                key,
            )

            if not within:
                mismatches.append(msg)

        return len(mismatches) == 0, mismatches
