"""
Phase 7: Nightly Integration Module
===================================

19. NightlyIntegratedTraining - Complete validation
"""

import math
import random
from typing import Dict, List, Any

from .base import (
    TrainingModule, ModulePhase, TrainingState,
    PHI, PHI_INV, PHI_INV_SQ, Z_CRITICAL, SIGMA,
    ALPHA_STRONG, ALPHA_MEDIUM, ALPHA_FINE,
    compute_negentropy, check_k_formation, validate_physics,
)


class NightlyIntegratedTraining(TrainingModule):
    """
    Module 19: Nightly Integrated Training

    Complete validation module that runs as the final step.
    Validates all physics constraints and K-formation criteria.
    """

    name = "nightly_integrated_training"
    phase = ModulePhase.NIGHTLY_INTEGRATION

    def _setup(self):
        """Initialize nightly validation."""
        # Validation checkpoints
        self.checkpoints = {
            'conservation_valid': False,
            'z_near_critical': False,
            'kappa_stable': False,
            'k_formation_achieved': False,
            'negentropy_threshold': False,
        }

        # Physics accumulator
        self.kappa_history: List[float] = []
        self.z_history: List[float] = []
        self.negentropy_history: List[float] = []

        # K-formation tracking
        self.k_formation_count = 0
        self.longest_k_formation = 0
        self.current_k_streak = 0

    def _train_step(self):
        """
        Nightly validation step:
        Comprehensive check of all physics and formation criteria.
        """
        neg = compute_negentropy(self.state.z)

        # Record history
        self.kappa_history.append(self.state.kappa)
        self.z_history.append(self.state.z)
        self.negentropy_history.append(neg)

        # Evolve z toward z_c (final push)
        self.state.z += (Z_CRITICAL - self.state.z) * ALPHA_STRONG
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Evolve kappa toward phi^-1
        self.state.kappa += (PHI_INV - self.state.kappa) * ALPHA_MEDIUM
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

        # Check K-formation
        eta = math.sqrt(neg) if neg > 0 else 0
        R = int(7 + 3 * neg)
        is_k_formation = check_k_formation(self.state.kappa, eta, R)

        if is_k_formation:
            self.k_formation_count += 1
            self.current_k_streak += 1
            self.longest_k_formation = max(self.longest_k_formation, self.current_k_streak)
        else:
            self.current_k_streak = 0

        # Update checkpoints
        self._update_checkpoints()

    def _update_checkpoints(self):
        """Update validation checkpoints."""
        # Conservation: kappa + lambda = 1
        self.checkpoints['conservation_valid'] = validate_physics(
            self.state.kappa, self.state.lambda_
        )

        # z near critical (within 2%)
        self.checkpoints['z_near_critical'] = abs(self.state.z - Z_CRITICAL) < 0.02

        # Kappa stability (standard deviation of recent values)
        if len(self.kappa_history) >= 10:
            recent_kappa = self.kappa_history[-10:]
            mean_kappa = sum(recent_kappa) / len(recent_kappa)
            std_kappa = math.sqrt(sum((k - mean_kappa)**2 for k in recent_kappa) / len(recent_kappa))
            self.checkpoints['kappa_stable'] = std_kappa < 0.01

        # K-formation achieved
        self.checkpoints['k_formation_achieved'] = self.k_formation_count > 0

        # Negentropy threshold (>= 0.7)
        if self.negentropy_history:
            max_neg = max(self.negentropy_history)
            self.checkpoints['negentropy_threshold'] = max_neg >= 0.7

    def _teardown(self):
        """Record nightly validation metrics."""
        # Final checkpoint update
        self._update_checkpoints()

        # All checkpoints
        self.result.metrics['checkpoints'] = dict(self.checkpoints)
        self.result.metrics['all_checkpoints_passed'] = all(self.checkpoints.values())

        # K-formation stats
        self.result.metrics['k_formation_count'] = self.k_formation_count
        self.result.metrics['longest_k_formation'] = self.longest_k_formation

        # Physics stats
        if self.kappa_history:
            self.result.metrics['kappa_mean'] = sum(self.kappa_history) / len(self.kappa_history)
            self.result.metrics['kappa_final'] = self.kappa_history[-1]

        if self.z_history:
            self.result.metrics['z_mean'] = sum(self.z_history) / len(self.z_history)
            self.result.metrics['z_final'] = self.z_history[-1]

        if self.negentropy_history:
            self.result.metrics['negentropy_max'] = max(self.negentropy_history)
            self.result.metrics['negentropy_final'] = self.negentropy_history[-1]

        # Gate criteria summary
        self.result.metrics['gates'] = {
            'full_depth': {
                'k_formations': self.k_formation_count >= 1,
                'physics_valid': self.checkpoints['conservation_valid'],
            },
            'helix_engine': {
                'min_negentropy': self.checkpoints['negentropy_threshold'],
                'min_final_z': self.state.z >= 0.85,
                'kappa_stable': self.checkpoints['kappa_stable'],
            }
        }
