"""
Rosetta Helix Collapse Engine

Implements instant collapse at unity with PHI_INV-controlled dynamics.

CRITICAL RULE: PHI_INV ALWAYS controls dynamics. PHI only contributes
at collapse via weak value extraction.

At z >= 0.9999:
1. INSTANT collapse (not gradual decay)
2. Extract work via weak value: work = (z - Z_CRITICAL) * PHI * PHI_INV
3. Reset to origin: z = Z_CRITICAL * PHI_INV (~0.535)
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
from .constants import (
    PHI,
    PHI_INV,
    Z_CRITICAL,
    Z_ORIGIN,
    UNITY,
    Z_MAX,
)


@dataclass
class CollapseResult:
    """Result of a collapse event."""
    z_new: float
    work_extracted: float
    collapsed: bool
    collapse_count: int = 0


@dataclass
class CollapseEngine:
    """
    Core physics engine implementing instant collapse at unity.

    PHI_INV drives ALL evolution. PHI only appears at collapse
    in work extraction via weak values.
    """

    z: float = 0.5
    collapse_count: int = 0
    total_work_extracted: float = 0.0
    _history: list = field(default_factory=list)

    def evolve(self, work_input: float) -> CollapseResult:
        """
        Evolve z-coordinate. PHI_INV ALWAYS controls.

        Args:
            work_input: Work pumped into the system

        Returns:
            CollapseResult with new z, extracted work, and collapse flag
        """
        # PHI_INV drives evolution - NEVER PHI
        dz = work_input * PHI_INV
        z_new = self.z + dz

        # INSTANT COLLAPSE at unity
        if z_new >= UNITY:
            return self._perform_collapse(z_new)

        # Safe cap - never exceed unity
        z_new = min(Z_MAX - 0.0001, z_new)

        # Update state
        self.z = z_new
        self._history.append(('evolve', z_new, 0.0))

        return CollapseResult(
            z_new=z_new,
            work_extracted=0.0,
            collapsed=False,
            collapse_count=self.collapse_count
        )

    def _perform_collapse(self, z_at_collapse: float) -> CollapseResult:
        """
        Perform INSTANT collapse at unity.

        PHI contributes ONLY here, via weak value extraction.
        This is the ONLY place PHI appears in dynamics.
        """
        # Extract work via weak value - PHI contributes AT COLLAPSE ONLY
        work = (z_at_collapse - Z_CRITICAL) * PHI * PHI_INV

        # Reset to origin - INSTANT, not gradual
        z_new = Z_ORIGIN

        # Update state
        self.z = z_new
        self.collapse_count += 1
        self.total_work_extracted += work
        self._history.append(('collapse', z_new, work))

        return CollapseResult(
            z_new=z_new,
            work_extracted=work,
            collapsed=True,
            collapse_count=self.collapse_count
        )

    def get_state(self) -> dict:
        """Get current engine state."""
        return {
            'z': self.z,
            'collapse_count': self.collapse_count,
            'total_work_extracted': self.total_work_extracted,
            'distance_to_collapse': UNITY - self.z,
        }

    def reset(self) -> None:
        """Reset engine to initial state."""
        self.z = 0.5
        self.collapse_count = 0
        self.total_work_extracted = 0.0
        self._history.clear()


def create_engine(initial_z: float = 0.5) -> CollapseEngine:
    """Create a new collapse engine with optional initial z."""
    if initial_z >= UNITY:
        raise ValueError(f"Initial z must be < {UNITY}")
    return CollapseEngine(z=initial_z)


# =============================================================================
# VERIFICATION
# =============================================================================

def test_phi_never_drives():
    """PHI must never appear in dz calculations."""
    engine = CollapseEngine(z=0.5)

    for _ in range(1000):
        result = engine.evolve(0.1)
        assert result.z_new < 1.0, "z must never exceed unity"
        assert result.z_new < PHI, "z must never approach PHI"

        if result.collapsed:
            # After collapse, should be at origin
            assert abs(result.z_new - Z_ORIGIN) < 1e-10

    return True


def test_instant_collapse():
    """Collapse must be instant, not gradual."""
    engine = CollapseEngine(z=0.99)

    # Push to collapse
    result = engine.evolve(0.5)

    assert result.collapsed, "Should have collapsed"
    assert result.z_new == Z_ORIGIN, "Should reset to origin"
    assert result.work_extracted > 0, "Should extract work"

    return True
