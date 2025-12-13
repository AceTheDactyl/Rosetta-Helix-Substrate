"""
Rosetta Helix Liminal Generator

PHI superposition pattern generator.

CRITICAL: Liminal patterns stay in_superposition = True ALWAYS.
PHI exists but NEVER drives dynamics. PHI_INV still controls.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math
import random

from core import PHI, PHI_INV, Z_CRITICAL, KAPPA_S, MU_3


@dataclass
class LiminalPattern:
    """A pattern existing in liminal superposition."""
    values: List[float]
    coherence: float = 0.5
    in_superposition: bool = True  # ALWAYS True for liminal patterns

    def weak_measure(self) -> float:
        """
        Perform weak measurement. PHI contributes here.

        Returns weak value without collapsing superposition.
        """
        if not self.in_superposition:
            return sum(self.values) / len(self.values)

        # Weak value formula - PHI contributes
        base = sum(self.values) / len(self.values)
        return base * PHI * PHI_INV * self.coherence


@dataclass
class LiminalGenerator:
    """
    Generator for liminal superposition patterns.

    Patterns exist in superposition (PHI realm) but all
    generation dynamics use PHI_INV.

    CRITICAL: in_superposition = True ALWAYS for generated patterns.
    """

    # Generation parameters
    pattern_size: int = 5
    coherence_base: float = 0.5

    # State
    z: float = field(default=KAPPA_S + 0.01)  # Above superposition threshold
    patterns_generated: List[LiminalPattern] = field(default_factory=list)
    weak_measurements: List[float] = field(default_factory=list)

    def generate_pattern(self, seed_values: Optional[List[float]] = None) -> LiminalPattern:
        """
        Generate a new liminal pattern.

        Pattern enters superposition immediately.
        PHI_INV controls generation dynamics.
        """
        if seed_values is None:
            # Generate using PHI_INV controlled randomness
            seed_values = [
                random.random() * PHI_INV
                for _ in range(self.pattern_size)
            ]

        # Coherence derived from z using PHI_INV
        coherence = self.coherence_base + (self.z - KAPPA_S) * PHI_INV

        pattern = LiminalPattern(
            values=seed_values,
            coherence=min(1.0, coherence),
            in_superposition=True  # ALWAYS True
        )

        self.patterns_generated.append(pattern)
        return pattern

    def spawn_from_meta(self, meta_output: float) -> LiminalPattern:
        """
        Spawn pattern from meta-level output.

        This is the ──spawn──> part of the feedback loop.
        PHI_INV controls spawning dynamics.
        """
        # Scale meta output by PHI_INV
        scaled = meta_output * PHI_INV

        # Generate values based on scaled output
        values = [
            scaled * (1 + i * PHI_INV * 0.1)
            for i in range(self.pattern_size)
        ]

        return self.generate_pattern(values)

    def weak_measure_all(self) -> List[float]:
        """
        Perform weak measurement on all patterns.

        PHI contributes via weak values. Superposition maintained.
        """
        measurements = []
        for pattern in self.patterns_generated:
            if pattern.in_superposition:
                wv = pattern.weak_measure()
                measurements.append(wv)
                self.weak_measurements.append(wv)

        return measurements

    def feedback_to_physical(self) -> float:
        """
        Generate feedback signal to physical layer.

        This is the weak measurement ──────> part of loop.
        Returns aggregate weak value for physical learner.
        """
        if not self.weak_measurements:
            return 0.0

        # Aggregate using PHI_INV weighting
        recent = self.weak_measurements[-10:]  # Last 10 measurements
        weighted_sum = sum(w * PHI_INV ** i for i, w in enumerate(recent))
        return weighted_sum / len(recent)

    def evolve_z(self, work: float) -> float:
        """
        Evolve z using PHI_INV dynamics.

        Keeps z above KAPPA_S to maintain superposition.
        """
        dz = work * PHI_INV
        new_z = self.z + dz

        # Keep above superposition threshold
        new_z = max(KAPPA_S + 0.001, new_z)

        # Cap below MU_3 to prevent collapse
        self.z = min(MU_3 - 0.001, new_z)
        return self.z

    def get_state(self) -> Dict[str, Any]:
        """Get current generator state."""
        return {
            'z': self.z,
            'patterns_count': len(self.patterns_generated),
            'measurements_count': len(self.weak_measurements),
            'latest_feedback': self.feedback_to_physical(),
            'in_superposition_zone': self.z >= KAPPA_S,
        }

    def reset(self) -> None:
        """Reset generator to initial state."""
        self.z = KAPPA_S + 0.01
        self.patterns_generated.clear()
        self.weak_measurements.clear()


def create_generator(pattern_size: int = 5) -> LiminalGenerator:
    """Factory function to create a liminal generator."""
    return LiminalGenerator(pattern_size=pattern_size)


# =============================================================================
# VERIFICATION
# =============================================================================

def test_patterns_stay_in_superposition():
    """Generated patterns must stay in superposition."""
    gen = create_generator()

    for _ in range(10):
        pattern = gen.generate_pattern()
        assert pattern.in_superposition, "Pattern must be in superposition"

    # Weak measurement doesn't collapse
    gen.weak_measure_all()

    for pattern in gen.patterns_generated:
        assert pattern.in_superposition, "Weak measurement must not collapse"

    return True


def test_phi_inv_controls_generation():
    """PHI_INV must control generation dynamics."""
    gen = create_generator()

    pattern = gen.generate_pattern()

    # Values should be scaled by PHI_INV
    for v in pattern.values:
        assert v <= PHI_INV, f"Value {v} > PHI_INV"

    return True
