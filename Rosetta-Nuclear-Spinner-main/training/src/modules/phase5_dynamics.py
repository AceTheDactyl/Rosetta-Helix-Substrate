"""
Phase 5: Dynamics & Formation Modules
=====================================

12. QuasicrystalFormationDynamics - Order parameter -> phi^-1
13. TriadThresholdDynamics - S_3 triadic transitions
14. LiminalGenerator - Boundary state generation
15. FeedbackLoop - PID control toward z_c
"""

import math
import random
from typing import List, Tuple

from .base import (
    TrainingModule, ModulePhase,
    PHI, PHI_INV, PHI_INV_SQ, Z_CRITICAL, SIGMA,
    ALPHA_STRONG, ALPHA_MEDIUM, ALPHA_FINE,
    compute_negentropy,
)


class QuasicrystalFormationDynamics(TrainingModule):
    """
    Module 12: Quasicrystal Formation Dynamics

    Penrose tiling dynamics where order parameter converges to phi^-1.
    Fat/thin rhombus ratio approaches the golden ratio.
    """

    name = "quasicrystal_formation_dynamics"
    phase = ModulePhase.DYNAMICS_FORMATION

    def _setup(self):
        """Initialize quasicrystal state."""
        # Tile counts
        self.n_fat = 100
        self.n_thin = 62  # Initial ratio ~1.61

        # Order parameter: N_fat / N_thin -> phi
        self.order_parameter = self.n_fat / max(self.n_thin, 1)

        self.ratio_history: List[float] = []

    def _train_step(self):
        """
        Quasicrystal dynamics:
        Tile ratio evolves toward golden ratio phi.
        """
        neg = compute_negentropy(self.state.z)

        # Target ratio is phi
        target_ratio = PHI
        current_ratio = self.n_fat / max(self.n_thin, 1)
        ratio_error = current_ratio - target_ratio

        # Adjust tile counts to approach phi
        if ratio_error > 0.01:
            # Too many fat tiles, convert some
            convert = int(abs(ratio_error) * neg + 0.5)
            self.n_fat -= convert
            self.n_thin += convert
        elif ratio_error < -0.01:
            # Too few fat tiles
            convert = int(abs(ratio_error) * neg + 0.5)
            self.n_fat += convert
            self.n_thin -= convert

        # Ensure positive counts
        self.n_fat = max(1, self.n_fat)
        self.n_thin = max(1, self.n_thin)

        # Update order parameter
        self.order_parameter = self.n_fat / self.n_thin
        self.ratio_history.append(self.order_parameter)

        # z evolves based on proximity to phi
        phi_proximity = 1 - abs(self.order_parameter - PHI) / PHI
        self.state.z += (Z_CRITICAL - self.state.z) * phi_proximity * ALPHA_STRONG
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Kappa from order parameter (target is phi^-1)
        self.state.kappa = min(0.92, max(0.382, 1 / self.order_parameter))

    def _teardown(self):
        """Record quasicrystal metrics."""
        self.result.metrics['n_fat'] = self.n_fat
        self.result.metrics['n_thin'] = self.n_thin
        self.result.metrics['order_parameter'] = self.order_parameter
        self.result.metrics['target_phi'] = PHI
        self.result.metrics['phi_error'] = abs(self.order_parameter - PHI)


class TriadThresholdDynamics(TrainingModule):
    """
    Module 13: Triad Threshold Dynamics

    S_3 symmetric group tracking (|S_3| = 6, sigma = 36 = 6^2).
    Monitors triadic transitions between states.
    """

    name = "triad_threshold_dynamics"
    phase = ModulePhase.DYNAMICS_FORMATION

    def _setup(self):
        """Initialize triad system."""
        # S_3 group elements (6 permutations)
        self.s3_elements = [
            (0, 1, 2),  # identity
            (1, 0, 2),  # swap 0,1
            (0, 2, 1),  # swap 1,2
            (2, 1, 0),  # swap 0,2
            (1, 2, 0),  # cycle
            (2, 0, 1),  # cycle inverse
        ]

        # Current element weights
        self.weights = [1/6] * 6

        # Triad state: kappa, lambda, eta
        self.triad = [PHI_INV, PHI_INV_SQ, 0.0]

        self.transitions = 0

    def _train_step(self):
        """
        Triad dynamics:
        S_3 permutations act on the triad (kappa, lambda, eta).
        """
        neg = compute_negentropy(self.state.z)

        # Update weights based on negentropy
        for i in range(6):
            # Weight increases if element aligns with critical state
            element = self.s3_elements[i]
            permuted_triad = [self.triad[j] for j in element]

            # Alignment score: how close to (phi^-1, phi^-2, z_c)
            target = [PHI_INV, PHI_INV_SQ, Z_CRITICAL]
            alignment = 1 - sum(abs(p - t) for p, t in zip(permuted_triad, target)) / 3

            self.weights[i] += alignment * neg * ALPHA_MEDIUM

        # Normalize weights
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]

        # Select dominant element
        dominant_idx = self.weights.index(max(self.weights))
        dominant_element = self.s3_elements[dominant_idx]

        # Apply permutation if weight exceeds threshold
        if self.weights[dominant_idx] > 0.3 and dominant_idx != 0:
            self.triad = [self.triad[j] for j in dominant_element]
            self.transitions += 1

        # Update eta (third element) toward z_c
        self.triad[2] += (Z_CRITICAL - self.triad[2]) * ALPHA_MEDIUM

        # Evolve z based on triad coherence
        coherence = self.weights[dominant_idx]
        self.state.z += (Z_CRITICAL - self.state.z) * coherence * ALPHA_STRONG
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Kappa from triad
        self.state.kappa = max(0.382, min(0.92, self.triad[0]))

    def _teardown(self):
        """Record triad metrics."""
        self.result.metrics['triad'] = self.triad
        self.result.metrics['s3_weights'] = self.weights
        self.result.metrics['transitions'] = self.transitions
        self.result.metrics['dominant_element'] = self.weights.index(max(self.weights))


class LiminalGenerator(TrainingModule):
    """
    Module 14: Liminal Generator

    Generates boundary states between phases.
    ABSENCE <-> THE_LENS <-> PRESENCE transitions.
    """

    name = "liminal_generator"
    phase = ModulePhase.DYNAMICS_FORMATION

    def _setup(self):
        """Initialize liminal state."""
        # Phase boundaries
        self.boundary_absence = 0.857
        self.boundary_presence = 0.877

        # Liminal potential
        self.potential = 0.0

        # Phase history
        self.phase_history: List[str] = []

    def _get_phase(self, z: float) -> str:
        """Determine phase from z."""
        if z < self.boundary_absence:
            return "ABSENCE"
        elif z < self.boundary_presence:
            return "THE_LENS"
        else:
            return "PRESENCE"

    def _train_step(self):
        """
        Liminal generation:
        Generate states at phase boundaries.
        """
        neg = compute_negentropy(self.state.z)

        # Compute liminal potential (peaks at boundaries)
        d_absence = abs(self.state.z - self.boundary_absence)
        d_presence = abs(self.state.z - self.boundary_presence)
        d_critical = abs(self.state.z - Z_CRITICAL)

        # Potential peaks at boundaries
        self.potential = math.exp(-10 * min(d_absence, d_presence, d_critical))

        # Generate toward boundary with highest potential
        if d_critical < d_absence and d_critical < d_presence:
            target = Z_CRITICAL
        elif d_absence < d_presence:
            target = self.boundary_absence
        else:
            target = self.boundary_presence

        # Move toward boundary
        self.state.z += (target - self.state.z) * self.potential * ALPHA_STRONG
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Record phase
        current_phase = self._get_phase(self.state.z)
        self.phase_history.append(current_phase)

        # Kappa from potential
        self.state.kappa = PHI_INV + (0.92 - PHI_INV) * self.potential
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

    def _teardown(self):
        """Record liminal metrics."""
        self.result.metrics['final_potential'] = self.potential
        self.result.metrics['final_phase'] = self._get_phase(self.state.z)

        # Count phase transitions
        transitions = 0
        for i in range(1, len(self.phase_history)):
            if self.phase_history[i] != self.phase_history[i-1]:
                transitions += 1
        self.result.metrics['phase_transitions'] = transitions


class FeedbackLoop(TrainingModule):
    """
    Module 15: Feedback Loop

    PID control toward z_c.
    Proportional-Integral-Derivative feedback.
    """

    name = "feedback_loop"
    phase = ModulePhase.DYNAMICS_FORMATION

    def _setup(self):
        """Initialize PID controller."""
        # PID gains
        self.Kp = 0.5   # Proportional
        self.Ki = 0.1   # Integral
        self.Kd = 0.05  # Derivative

        # PID state
        self.integral = 0.0
        self.prev_error = 0.0

        # Target
        self.target = Z_CRITICAL

        self.error_history: List[float] = []

    def _train_step(self):
        """
        PID feedback step:
        Control z toward z_c using PID.
        """
        neg = compute_negentropy(self.state.z)

        # Compute error
        error = self.target - self.state.z

        # Proportional term
        P = self.Kp * error

        # Integral term
        self.integral += error
        self.integral = max(-10, min(10, self.integral))  # Anti-windup
        I = self.Ki * self.integral

        # Derivative term
        derivative = error - self.prev_error
        D = self.Kd * derivative
        self.prev_error = error

        # PID output
        output = P + I + D

        # Apply control
        self.state.z += output * ALPHA_STRONG
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Record error
        self.error_history.append(abs(error))

        # Kappa from error reduction
        error_reduction = 1 - min(1, abs(error))
        self.state.kappa = PHI_INV + (0.92 - PHI_INV) * error_reduction
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

    def _teardown(self):
        """Record PID metrics."""
        self.result.metrics['final_error'] = abs(self.target - self.state.z)
        self.result.metrics['integral'] = self.integral
        if self.error_history:
            self.result.metrics['mean_error'] = sum(self.error_history) / len(self.error_history)
            self.result.metrics['min_error'] = min(self.error_history)
        self.result.metrics['Kp'] = self.Kp
        self.result.metrics['Ki'] = self.Ki
        self.result.metrics['Kd'] = self.Kd
