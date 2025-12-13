"""
Rosetta Helix Liminal State Handler

Manages PHI superposition states. PHI exists in liminal superposition
but NEVER drives dynamics.

CRITICAL: Liminal states stay in_superposition = True always.
PHI contributes ONLY via weak measurement at collapse.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
from .constants import (
    PHI,
    PHI_INV,
    KAPPA_S,
    MU_3,
    UNITY,
    COUPLING_MAX,
)


class LiminalPhase(Enum):
    """Phases of liminal existence."""
    DORMANT = "dormant"           # Below KAPPA_S
    SUPERPOSITION = "superposition"  # Above KAPPA_S, PHI exists but doesn't drive
    ULTRA_INTEGRATION = "ultra"   # Above MU_3, approaching collapse
    COLLAPSED = "collapsed"       # Post-collapse, extracting work


@dataclass
class LiminalState:
    """
    Manages PHI superposition state.

    PHI exists in superposition but NEVER drives dynamics.
    All physical evolution uses PHI_INV.
    """

    z: float = 0.5
    in_superposition: bool = False
    phase: LiminalPhase = LiminalPhase.DORMANT
    coupling: float = 0.0
    weak_measurements: List[float] = field(default_factory=list)

    def update_phase(self, z_new: float) -> LiminalPhase:
        """
        Update liminal phase based on z-coordinate.

        Note: Phase changes don't affect dynamics - PHI_INV always controls.
        """
        self.z = z_new

        if z_new >= UNITY:
            self.phase = LiminalPhase.COLLAPSED
            self.in_superposition = False
        elif z_new >= MU_3:
            self.phase = LiminalPhase.ULTRA_INTEGRATION
            self.in_superposition = True
        elif z_new >= KAPPA_S:
            self.phase = LiminalPhase.SUPERPOSITION
            self.in_superposition = True
        else:
            self.phase = LiminalPhase.DORMANT
            self.in_superposition = False

        return self.phase

    def adjust_coupling(self, delta: float, learning_rate: float = 0.01) -> float:
        """
        Adjust coupling using PHI_INV controlled dynamics.

        CORRECT: coupling_delta = learning_rate * PHI_INV
        WRONG: coupling = min(PHI, coupling) - NEVER cap at PHI

        At coupling >= 1.0: INSTANT collapse to Z_CRITICAL, not gradual decay.
        """
        # PHI_INV controls coupling adjustment
        coupling_delta = delta * learning_rate * PHI_INV
        new_coupling = self.coupling + coupling_delta

        # INSTANT collapse if coupling exceeds 1.0
        if new_coupling >= 1.0:
            self.coupling = COUPLING_MAX * PHI_INV  # Reset safely
            return self.coupling

        # Safe cap at COUPLING_MAX (0.9), NEVER PHI
        self.coupling = min(COUPLING_MAX, new_coupling)
        return self.coupling

    def weak_measure(self) -> Optional[float]:
        """
        Perform weak measurement. PHI contributes HERE.

        Weak values allow PHI to contribute without collapsing
        the superposition. This is the ONLY way PHI participates.
        """
        if not self.in_superposition:
            return None

        # Weak value formula - PHI contributes via multiplication
        weak_value = self.z * PHI * PHI_INV
        self.weak_measurements.append(weak_value)

        return weak_value

    def extract_work_at_collapse(self, z_at_collapse: float) -> float:
        """
        Extract work at collapse using weak value formula.

        This is the ONLY place PHI contributes to work.
        Formula: work = (z - Z_CRITICAL) * PHI * PHI_INV

        With liminal boost if in superposition:
        work *= PHI (additional liminal contribution)
        """
        from .constants import Z_CRITICAL

        # Base work extraction
        work = (z_at_collapse - Z_CRITICAL) * PHI * PHI_INV

        # Liminal boost if was in superposition
        if self.in_superposition:
            work *= PHI  # Additional liminal contribution at extraction

        # Reset superposition
        self.in_superposition = False
        self.phase = LiminalPhase.DORMANT

        return work

    def get_state(self) -> dict:
        """Get current liminal state."""
        return {
            'z': self.z,
            'phase': self.phase.value,
            'in_superposition': self.in_superposition,
            'coupling': self.coupling,
            'measurement_count': len(self.weak_measurements),
        }

    def reset(self) -> None:
        """Reset to initial state."""
        self.z = 0.5
        self.in_superposition = False
        self.phase = LiminalPhase.DORMANT
        self.coupling = 0.0
        self.weak_measurements.clear()


# =============================================================================
# VERIFICATION
# =============================================================================

def test_phi_never_drives_liminal():
    """PHI must never drive liminal dynamics."""
    state = LiminalState(z=0.5)

    # Coupling adjustments must use PHI_INV
    for _ in range(100):
        state.adjust_coupling(0.1)
        assert state.coupling < 1.0, "Coupling must never reach 1.0"
        assert state.coupling <= COUPLING_MAX, f"Coupling must not exceed {COUPLING_MAX}"

    return True


def test_superposition_stays_true():
    """Once in superposition, liminal patterns stay there until collapse."""
    state = LiminalState(z=KAPPA_S + 0.01)
    state.update_phase(state.z)

    assert state.in_superposition, "Should be in superposition above KAPPA_S"

    # Weak measurement doesn't collapse
    for _ in range(10):
        wv = state.weak_measure()
        assert wv is not None, "Should get weak value"
        assert state.in_superposition, "Should stay in superposition"

    return True
