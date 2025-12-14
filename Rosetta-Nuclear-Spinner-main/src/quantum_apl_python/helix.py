"""Helix coordinate helpers for mapping z-coordinates to APL semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import os
import math
from .constants import Z_CRITICAL, TRIAD_T6


@dataclass
class HelixCoordinate:
    """Simple representation of a (theta, z, r) point on the helix."""

    theta: float
    z: float
    r: float = 1.0

    @classmethod
    def from_parameter(cls, t: float) -> "HelixCoordinate":
        """
        Construct a helix coordinate from the canonical parametric equation:
        r(t) = (cos t, sin t, t). The z component is normalized into [0, 1].
        """

        x = math.cos(t)
        y = math.sin(t)
        theta = math.atan2(y, x) % (2 * math.pi)
        # Smooth mapping of the unbounded t dimension into z ∈ [0, 1]
        z = 0.5 + 0.5 * math.tanh(t / 8.0)
        r = math.hypot(x, y)
        return cls(theta=theta, z=max(0.0, min(1.0, z)), r=r)

    def to_vector(self) -> Dict[str, float]:
        """Return the Euclidean embedding of the helix point."""

        return {
            "x": self.r * math.cos(self.theta),
            "y": self.r * math.sin(self.theta),
            "z": self.z,
        }


class HelixAPLMapper:
    """Map helix coordinates to APL harmonics, operators, and truth channels."""

    def __init__(self):
        triad_flag = os.getenv("QAPL_TRIAD_UNLOCK", "").lower() in ("1", "true", "yes", "y")
        triad_completions = 0
        try:
            triad_completions = int(os.getenv("QAPL_TRIAD_COMPLETIONS", "0"))
        except ValueError:
            triad_completions = 0
        triad_unlocked = triad_flag or (triad_completions >= 3)
        t6_gate = TRIAD_T6 if triad_unlocked else Z_CRITICAL

        self.time_harmonics: List[tuple[float, str]] = [
            (0.10, "t1"),
            (0.20, "t2"),
            (0.40, "t3"),
            (0.60, "t4"),
            (0.75, "t5"),
            (t6_gate, "t6"),
            (0.90, "t7"),
            (0.97, "t8"),
            (1.01, "t9"),
        ]
        self.operator_windows: Dict[str, List[str]] = {
            "t1": ["()", "−", "÷"],
            "t2": ["^", "÷", "−", "×"],
            "t3": ["×", "^", "÷", "+", "−"],
            "t4": ["+", "−", "÷", "()"],
            "t5": ["()", "×", "^", "÷", "+", "−"],
            "t6": ["+", "÷", "()", "−"],
            "t7": ["+", "()"],
            "t8": ["+", "()", "×"],
            "t9": ["+", "()", "×"],
        }

    def harmonic_from_z(self, z: float) -> str:
        for threshold, label in self.time_harmonics:
            if z < threshold:
                return label
        return "t9"

    def truth_channel_from_z(self, z: float) -> str:
        if z >= 0.9:
            return "TRUE"
        if z >= 0.6:
            return "PARADOX"
        return "UNTRUE"

    def describe(self, coord: HelixCoordinate) -> Dict[str, object]:
        harmonic = self.harmonic_from_z(coord.z)
        operators = self.operator_windows.get(harmonic, ["()"])
        truth = self.truth_channel_from_z(coord.z)
        # Optional μ classification and coherence via constants
        try:
            from .constants import compute_delta_s_neg, LENS_SIGMA, Z_CRITICAL
            from .constants import classify_mu  # type: ignore
            s = compute_delta_s_neg(coord.z, sigma=LENS_SIGMA, z_c=Z_CRITICAL)
            mu_class = classify_mu(coord.z)
        except Exception:
            s = None
            mu_class = None
        return {
            "harmonic": harmonic,
            "operators": operators,
            "truth_channel": truth,
            "theta": coord.theta,
            "z": coord.z,
            "r": coord.r,
            "coherence_s": s,
            "mu_class": mu_class,
        }
