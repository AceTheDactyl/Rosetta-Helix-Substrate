#!/usr/bin/env python3
"""
Adaptive TRIAD Gate Module
==========================

An extension to the standard TriadGate that dynamically adjusts the
pass requirement based on historical z-coordinate volatility.

Motivation:
  - Standard TRIAD requires exactly 3 passes to unlock
  - In noisy environments, this may trigger false unlocks
  - Adaptive TRIAD scales passes with volatility (3-6 range)

Theory:
  - Low volatility → stable system → base passes (3)
  - High volatility → noisy system → more passes (up to 6)
  - This improves robustness without sacrificing responsiveness

Author: Claude (Anthropic) - Contribution to Quantum-APL
Date: December 2025
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum


# ============================================================================
# CONSTANTS
# ============================================================================

TRIAD_HIGH = 0.85       # Rising edge detection threshold
TRIAD_LOW = 0.82        # Re-arm threshold
TRIAD_T6 = 0.83         # Unlocked t6 gate value
Z_CRITICAL = math.sqrt(3) / 2  # ≈ 0.8660254037844386


class TriadEvent(Enum):
    """TRIAD state machine events."""
    RISING_EDGE = "RISING_EDGE"
    REARMED = "REARMED"
    UNLOCKED = "UNLOCKED"
    NONE = None


# ============================================================================
# ADAPTIVE TRIAD GATE
# ============================================================================

@dataclass
class AdaptiveTriadGate:
    """
    Adaptive TRIAD Gate with volatility-scaled pass requirements.

    The standard TRIAD protocol requires exactly 3 passes through the
    high threshold (with re-arming via low threshold) to unlock.

    This adaptive variant scales the pass requirement based on recent
    z-coordinate volatility:

        required_passes = base_passes * (1 + min(volatility / 0.1, 1.0))

    This means:
        - Volatility ≈ 0.00: 3 passes required
        - Volatility ≈ 0.05: 4-5 passes required
        - Volatility ≥ 0.10: 6 passes required (capped)

    Parameters
    ----------
    base_passes : int
        Minimum passes required (default: 3)
    volatility_window : int
        Number of recent z values to consider for volatility (default: 50)
    max_scale : float
        Maximum scaling factor (default: 2.0, so up to 6 passes)
    volatility_threshold : float
        Volatility at which max scaling is reached (default: 0.1)

    Attributes
    ----------
    passes : int
        Current count of high-threshold crossings
    unlocked : bool
        Whether the gate has been unlocked
    z_history : List[float]
        Recent z-coordinate history for volatility calculation

    Example
    -------
    >>> gate = AdaptiveTriadGate()
    >>> # Simulate oscillating z values
    >>> for z in [0.86, 0.81, 0.86, 0.81, 0.86, 0.81]:
    ...     event, req = gate.update(z)
    ...     if event:
    ...         print(f"{event.value}: passes={gate.passes}/{req}")
    """

    base_passes: int = 3
    volatility_window: int = 50
    max_scale: float = 2.0
    volatility_threshold: float = 0.1

    # State (initialized in __post_init__)
    passes: int = field(default=0, init=False)
    unlocked: bool = field(default=False, init=False)
    z_history: List[float] = field(default_factory=list, init=False)
    _armed: bool = field(default=True, init=False)
    _unlock_z: Optional[float] = field(default=None, init=False)
    _unlock_volatility: Optional[float] = field(default=None, init=False)

    def _compute_volatility(self) -> float:
        """
        Compute recent volatility as mean absolute z-change.

        Returns
        -------
        float
            Average |Δz| over recent history, or 0.0 if insufficient data
        """
        if len(self.z_history) < 10:
            return 0.0

        recent = self.z_history[-self.volatility_window:]
        if len(recent) < 2:
            return 0.0

        diffs = [abs(recent[i+1] - recent[i]) for i in range(len(recent) - 1)]
        return sum(diffs) / len(diffs)

    def _required_passes(self) -> int:
        """
        Compute current pass requirement based on volatility.

        Returns
        -------
        int
            Number of passes required for unlock (base_passes to base_passes * max_scale)
        """
        vol = self._compute_volatility()
        scale = 1.0 + min(vol / self.volatility_threshold, self.max_scale - 1.0)
        return max(self.base_passes, int(self.base_passes * scale))

    def update(
        self,
        z: float,
        high: float = TRIAD_HIGH,
        low: float = TRIAD_LOW
    ) -> Tuple[Optional[TriadEvent], int]:
        """
        Update gate state with new z-coordinate.

        Parameters
        ----------
        z : float
            Current z-coordinate
        high : float
            Rising edge threshold (default: TRIAD_HIGH = 0.85)
        low : float
            Re-arm threshold (default: TRIAD_LOW = 0.82)

        Returns
        -------
        Tuple[Optional[TriadEvent], int]
            (event, required_passes) where event is RISING_EDGE, REARMED,
            UNLOCKED, or None
        """
        # Update history
        self.z_history.append(z)
        if len(self.z_history) > self.volatility_window * 2:
            self.z_history.pop(0)

        required = self._required_passes()

        # Rising edge detection
        if z >= high and self._armed:
            self.passes += 1
            self._armed = False

            if self.passes >= required and not self.unlocked:
                self.unlocked = True
                self._unlock_z = z
                self._unlock_volatility = self._compute_volatility()
                return (TriadEvent.UNLOCKED, required)

            return (TriadEvent.RISING_EDGE, required)

        # Re-arm detection
        if z <= low and not self._armed:
            self._armed = True
            return (TriadEvent.REARMED, required)

        return (TriadEvent.NONE, required)

    def get_t6_gate(self) -> float:
        """
        Get current t6 gate value.

        Returns
        -------
        float
            TRIAD_T6 (0.83) if unlocked, else Z_CRITICAL (≈0.866)
        """
        return TRIAD_T6 if self.unlocked else Z_CRITICAL

    def get_volatility(self) -> float:
        """Get current computed volatility."""
        return self._compute_volatility()

    def get_state(self) -> dict:
        """
        Get complete gate state for inspection.

        Returns
        -------
        dict
            State dictionary with all relevant fields
        """
        return {
            'passes': self.passes,
            'required': self._required_passes(),
            'unlocked': self.unlocked,
            'armed': self._armed,
            'volatility': self._compute_volatility(),
            't6_gate': self.get_t6_gate(),
            'history_length': len(self.z_history),
            'unlock_z': self._unlock_z,
            'unlock_volatility': self._unlock_volatility,
        }

    def reset(self) -> None:
        """Reset gate to initial state."""
        self.passes = 0
        self.unlocked = False
        self._armed = True
        self.z_history.clear()
        self._unlock_z = None
        self._unlock_volatility = None

    def analyzer_report(self) -> str:
        """
        Generate analyzer report string.

        Returns
        -------
        str
            Human-readable status line
        """
        gate = self.get_t6_gate()
        label = "ADAPTIVE-TRIAD" if self.unlocked else "ADAPTIVE"
        vol = self._compute_volatility()
        req = self._required_passes()
        return f"t6 gate: {label} @ {gate:.4f} (vol={vol:.4f}, req={req})"


# ============================================================================
# COMPARISON: STANDARD VS ADAPTIVE
# ============================================================================

@dataclass
class StandardTriadGate:
    """Standard (non-adaptive) TRIAD gate for comparison."""

    passes: int = field(default=0, init=False)
    unlocked: bool = field(default=False, init=False)
    _armed: bool = field(default=True, init=False)

    def update(self, z: float, high: float = TRIAD_HIGH, low: float = TRIAD_LOW):
        if z >= high and self._armed:
            self.passes += 1
            self._armed = False
            if self.passes >= 3 and not self.unlocked:
                self.unlocked = True
                return TriadEvent.UNLOCKED
            return TriadEvent.RISING_EDGE
        if z <= low and not self._armed:
            self._armed = True
            return TriadEvent.REARMED
        return TriadEvent.NONE

    def get_t6_gate(self) -> float:
        return TRIAD_T6 if self.unlocked else Z_CRITICAL

    def reset(self):
        self.passes = 0
        self.unlocked = False
        self._armed = True


# ============================================================================
# DEMO / TEST
# ============================================================================

def demo_comparison():
    """
    Demonstrate difference between standard and adaptive TRIAD.
    """
    import random

    print("=" * 70)
    print("ADAPTIVE vs STANDARD TRIAD COMPARISON")
    print("=" * 70)

    # Scenario 1: Clean oscillation (low volatility)
    print("\n--- Scenario 1: Clean Oscillation (Low Volatility) ---")

    adaptive = AdaptiveTriadGate()
    standard = StandardTriadGate()

    clean_sequence = []
    for i in range(6):
        clean_sequence.extend([0.70, 0.75, 0.80, 0.86, 0.83, 0.81])

    adaptive_unlock_step = None
    standard_unlock_step = None

    for i, z in enumerate(clean_sequence):
        ae, req = adaptive.update(z)
        se = standard.update(z)

        if ae == TriadEvent.UNLOCKED and adaptive_unlock_step is None:
            adaptive_unlock_step = i
        if se == TriadEvent.UNLOCKED and standard_unlock_step is None:
            standard_unlock_step = i

    print(f"  Standard unlocked at step: {standard_unlock_step}")
    print(f"  Adaptive unlocked at step: {adaptive_unlock_step}")
    print(f"  Adaptive volatility: {adaptive.get_volatility():.4f}")
    print(f"  Adaptive required passes: {adaptive._required_passes()}")

    # Scenario 2: Noisy oscillation (high volatility)
    print("\n--- Scenario 2: Noisy Oscillation (High Volatility) ---")

    adaptive = AdaptiveTriadGate()
    standard = StandardTriadGate()

    noisy_sequence = []
    for i in range(12):
        base = [0.70, 0.75, 0.80, 0.86, 0.83, 0.81]
        noisy_sequence.extend([z + random.uniform(-0.08, 0.08) for z in base])

    adaptive_unlock_step = None
    standard_unlock_step = None

    for i, z in enumerate(noisy_sequence):
        z = max(0, min(1, z))  # Clamp
        ae, req = adaptive.update(z)
        se = standard.update(z)

        if ae == TriadEvent.UNLOCKED and adaptive_unlock_step is None:
            adaptive_unlock_step = i
        if se == TriadEvent.UNLOCKED and standard_unlock_step is None:
            standard_unlock_step = i

    print(f"  Standard unlocked at step: {standard_unlock_step}")
    print(f"  Adaptive unlocked at step: {adaptive_unlock_step}")
    print(f"  Adaptive volatility: {adaptive.get_volatility():.4f}")
    print(f"  Adaptive required passes: {adaptive._required_passes()}")

    print("\n--- Analysis ---")
    print("In noisy environments, adaptive TRIAD requires more passes,")
    print("reducing false unlocks from noise-induced threshold crossings.")
    print("=" * 70)


if __name__ == "__main__":
    demo_comparison()

    print("\n--- Unit Tests ---")

    # Test 1: Basic unlock
    gate = AdaptiveTriadGate()
    for z in [0.86, 0.81, 0.86, 0.81, 0.86]:
        gate.update(z)
    assert gate.unlocked, "Should unlock after 3 clean passes"
    print("✓ Basic unlock works")

    # Test 2: Volatility scaling
    gate = AdaptiveTriadGate()
    # Add high-volatility history
    for _ in range(60):
        gate.z_history.append(0.5 + 0.15 * ((_ % 2) * 2 - 1))
    req = gate._required_passes()
    assert req > 3, f"High volatility should require >3 passes, got {req}"
    print(f"✓ Volatility scaling works (req={req} for high-vol)")

    # Test 3: t6 gate values
    gate = AdaptiveTriadGate()
    assert abs(gate.get_t6_gate() - Z_CRITICAL) < 1e-10, "Locked gate should be Z_CRITICAL"
    gate.unlocked = True
    assert abs(gate.get_t6_gate() - TRIAD_T6) < 1e-10, "Unlocked gate should be TRIAD_T6"
    print("✓ t6 gate values correct")

    print("\nAll tests passed! ✓")
