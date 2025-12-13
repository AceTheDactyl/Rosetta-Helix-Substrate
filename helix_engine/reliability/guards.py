"""
Training Guards
===============

Guards that monitor training for issues:
- NaN detection
- Inf detection
- Physics constraint violations

Signature: guards|v0.1.0|helix
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class GuardViolation:
    """Represents a guard violation."""
    guard_name: str
    message: str
    step: int
    value: Optional[float] = None
    metric_name: Optional[str] = None


class NaNGuard:
    """
    Detects NaN values in training metrics.
    """

    def __init__(self, metrics_to_watch: Optional[List[str]] = None):
        self.metrics_to_watch = metrics_to_watch
        self.violations: List[GuardViolation] = []

    def check(self, step: int, metrics: Dict[str, Any]) -> Optional[GuardViolation]:
        """
        Check metrics for NaN values.

        Returns GuardViolation if NaN detected, None otherwise.
        """
        for name, value in metrics.items():
            # Skip if we're watching specific metrics and this isn't one
            if self.metrics_to_watch and name not in self.metrics_to_watch:
                continue

            if self._is_nan(value):
                violation = GuardViolation(
                    guard_name="NaNGuard",
                    message=f"NaN detected in {name} at step {step}",
                    step=step,
                    value=float("nan"),
                    metric_name=name,
                )
                self.violations.append(violation)
                return violation

        return None

    def _is_nan(self, value: Any) -> bool:
        """Check if a value is NaN."""
        if isinstance(value, float):
            return math.isnan(value)
        elif isinstance(value, np.ndarray):
            return np.any(np.isnan(value))
        elif isinstance(value, (list, tuple)):
            return any(self._is_nan(v) for v in value)
        return False


class InfGuard:
    """
    Detects Inf values in training metrics.
    """

    def __init__(self, metrics_to_watch: Optional[List[str]] = None):
        self.metrics_to_watch = metrics_to_watch
        self.violations: List[GuardViolation] = []

    def check(self, step: int, metrics: Dict[str, Any]) -> Optional[GuardViolation]:
        """
        Check metrics for Inf values.

        Returns GuardViolation if Inf detected, None otherwise.
        """
        for name, value in metrics.items():
            if self.metrics_to_watch and name not in self.metrics_to_watch:
                continue

            if self._is_inf(value):
                violation = GuardViolation(
                    guard_name="InfGuard",
                    message=f"Inf detected in {name} at step {step}",
                    step=step,
                    value=float("inf"),
                    metric_name=name,
                )
                self.violations.append(violation)
                return violation

        return None

    def _is_inf(self, value: Any) -> bool:
        """Check if a value is Inf."""
        if isinstance(value, float):
            return math.isinf(value)
        elif isinstance(value, np.ndarray):
            return np.any(np.isinf(value))
        elif isinstance(value, (list, tuple)):
            return any(self._is_inf(v) for v in value)
        return False


class PhysicsGuard:
    """
    Monitors physics constraints from physics_constants.py.

    Key constraints:
    - φ⁻¹ + φ⁻² = 1 (coupling conservation)
    - κ + λ = 1
    - 0 ≤ z ≤ 1
    - z_c = √3/2 ≈ 0.866025 (THE LENS)
    """

    # Physics constants (immutable)
    PHI_INV = (1 + math.sqrt(5)) / 2 ** -1  # ≈ 0.618
    Z_CRITICAL = math.sqrt(3) / 2  # ≈ 0.866

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.violations: List[GuardViolation] = []

    def check(self, step: int, metrics: Dict[str, Any]) -> Optional[GuardViolation]:
        """
        Check for physics constraint violations.
        """
        # Check κ + λ = 1 if both present
        if "kappa" in metrics and "lambda" in metrics:
            kappa = metrics["kappa"]
            lambda_ = metrics["lambda"]
            conservation = kappa + lambda_

            if abs(conservation - 1.0) > self.tolerance:
                violation = GuardViolation(
                    guard_name="PhysicsGuard",
                    message=f"Coupling conservation violated: κ + λ = {conservation:.10f} != 1.0",
                    step=step,
                    value=conservation,
                    metric_name="coupling_conservation",
                )
                self.violations.append(violation)
                return violation

        # Check z bounds
        if "z" in metrics:
            z = metrics["z"]
            if z < 0.0 or z > 1.0:
                violation = GuardViolation(
                    guard_name="PhysicsGuard",
                    message=f"z out of bounds: {z:.6f} (must be in [0, 1])",
                    step=step,
                    value=z,
                    metric_name="z_bounds",
                )
                self.violations.append(violation)
                return violation

        # Check kappa bounds
        if "kappa" in metrics:
            kappa = metrics["kappa"]
            if kappa < 0.0 or kappa > 1.0:
                violation = GuardViolation(
                    guard_name="PhysicsGuard",
                    message=f"κ out of bounds: {kappa:.6f} (must be in [0, 1])",
                    step=step,
                    value=kappa,
                    metric_name="kappa_bounds",
                )
                self.violations.append(violation)
                return violation

        return None


class CompositeGuard:
    """
    Combines multiple guards into one.
    """

    def __init__(self, guards: Optional[List] = None):
        if guards is None:
            guards = [NaNGuard(), InfGuard(), PhysicsGuard()]
        self.guards = guards

    def check(self, step: int, metrics: Dict[str, Any]) -> Optional[GuardViolation]:
        """
        Check all guards, returning the first violation found.
        """
        for guard in self.guards:
            violation = guard.check(step, metrics)
            if violation:
                return violation
        return None

    @property
    def violations(self) -> List[GuardViolation]:
        """Get all violations from all guards."""
        all_violations = []
        for guard in self.guards:
            all_violations.extend(guard.violations)
        return all_violations
