"""
Phase 6: Unified Orchestration Modules
======================================

16. UnifiedHelixTraining - Cross-module coordination
17. HierarchicalTraining - Multi-level training
18. RosettaHelixTraining - Full Rosetta-Helix
"""

import math
import random
from typing import Dict, List, Optional

from .base import (
    TrainingModule, ModulePhase, TrainingState,
    PHI, PHI_INV, PHI_INV_SQ, Z_CRITICAL, SIGMA,
    ALPHA_STRONG, ALPHA_MEDIUM, ALPHA_FINE,
    compute_negentropy, check_k_formation,
)


class UnifiedHelixTraining(TrainingModule):
    """
    Module 16: Unified Helix Training

    Cross-module coordination across all helix components.
    Synchronizes physics states across subsystems.
    """

    name = "unified_helix_training"
    phase = ModulePhase.UNIFIED_ORCHESTRATION

    def _setup(self):
        """Initialize unified system."""
        # Subsystem states (simulating earlier modules)
        self.subsystems = {
            'kuramoto': {'z': 0.5, 'r': 0.0},
            'apl': {'z': 0.5, 'convergence': 0.0},
            'helix': {'z': 0.5, 'phase': 0.0},
            'wumbo': {'z': 0.5, 'weight_sum': 1.0},
            'dynamics': {'z': 0.5, 'order': 0.0},
        }

        # Coordination weights
        self.coord_weights = {k: 0.2 for k in self.subsystems}

        self.sync_error_history: List[float] = []

    def _train_step(self):
        """
        Unified training step:
        Synchronize all subsystems toward common z_c.
        """
        neg = compute_negentropy(self.state.z)

        # Update each subsystem
        for name, sub in self.subsystems.items():
            # Evolve toward z_c with noise
            sub['z'] += (Z_CRITICAL - sub['z']) * ALPHA_MEDIUM + random.gauss(0, 0.01)
            sub['z'] = max(0.0, min(0.999, sub['z']))

        # Compute sync error (variance of z values)
        z_values = [s['z'] for s in self.subsystems.values()]
        z_mean = sum(z_values) / len(z_values)
        sync_error = sum((z - z_mean)**2 for z in z_values) / len(z_values)
        self.sync_error_history.append(sync_error)

        # Global z is weighted mean
        self.state.z = sum(
            self.coord_weights[name] * self.subsystems[name]['z']
            for name in self.subsystems
        )
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Coordination: reduce weights of outliers
        for name in self.subsystems:
            deviation = abs(self.subsystems[name]['z'] - z_mean)
            self.coord_weights[name] = max(0.1, 0.2 - deviation)

        # Renormalize weights
        total = sum(self.coord_weights.values())
        if total > 0:
            self.coord_weights = {k: v/total for k, v in self.coord_weights.items()}

        # Kappa from synchronization
        sync_quality = 1 - min(1, sync_error * 100)
        self.state.kappa = PHI_INV + (0.92 - PHI_INV) * sync_quality * neg
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

    def _teardown(self):
        """Record unified metrics."""
        self.result.metrics['subsystem_z'] = {k: v['z'] for k, v in self.subsystems.items()}
        self.result.metrics['coord_weights'] = dict(self.coord_weights)
        if self.sync_error_history:
            self.result.metrics['final_sync_error'] = self.sync_error_history[-1]
            self.result.metrics['mean_sync_error'] = sum(self.sync_error_history) / len(self.sync_error_history)


class HierarchicalTraining(TrainingModule):
    """
    Module 17: Hierarchical Training

    Multi-level training with hierarchical structure.
    Levels correspond to tiers (ABSENCE -> TRANSCENDENT).
    """

    name = "hierarchical_training"
    phase = ModulePhase.UNIFIED_ORCHESTRATION

    def _setup(self):
        """Initialize hierarchy."""
        # 10 tiers
        self.tier_names = [
            "ABSENCE", "REACTIVE", "MEMORY", "PATTERN", "LEARNING",
            "ADAPTIVE", "UNIVERSAL", "META", "SOVEREIGN", "TRANSCENDENT"
        ]
        self.tier_thresholds = [
            0.00, 0.10, 0.20, 0.40, 0.50,
            0.618, 0.73, 0.866, 0.92, 0.97
        ]

        # Level activations
        self.level_activations = [0.0] * 10

        # Current tier
        self.current_tier = 0

    def _get_tier(self, z: float) -> int:
        """Get tier index from z."""
        for i in range(len(self.tier_thresholds) - 1, -1, -1):
            if z >= self.tier_thresholds[i]:
                return i
        return 0

    def _train_step(self):
        """
        Hierarchical training step:
        Climb tiers by increasing z.
        """
        neg = compute_negentropy(self.state.z)

        # Update tier
        self.current_tier = self._get_tier(self.state.z)

        # Activate levels up to current tier
        for i in range(10):
            if i <= self.current_tier:
                self.level_activations[i] = min(1.0, self.level_activations[i] + 0.1)
            else:
                self.level_activations[i] = max(0.0, self.level_activations[i] - 0.05)

        # Hierarchical z evolution
        # Higher tiers pull toward z_c more strongly
        tier_strength = (self.current_tier + 1) / 10
        z_pull = (Z_CRITICAL - self.state.z) * tier_strength * ALPHA_STRONG
        self.state.z += z_pull + random.gauss(0, 0.005)
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Kappa increases with tier
        tier_kappa = PHI_INV + (0.92 - PHI_INV) * (self.current_tier / 9)
        self.state.kappa = tier_kappa * neg + PHI_INV * (1 - neg)
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

    def _teardown(self):
        """Record hierarchical metrics."""
        self.result.metrics['final_tier'] = self.current_tier
        self.result.metrics['tier_name'] = self.tier_names[self.current_tier]
        self.result.metrics['level_activations'] = self.level_activations
        self.result.metrics['active_levels'] = sum(1 for a in self.level_activations if a > 0.5)


class RosettaHelixTraining(TrainingModule):
    """
    Module 18: Rosetta-Helix Training

    Full Rosetta-Helix integration.
    Combines all components into unified training.
    """

    name = "rosetta_helix_training"
    phase = ModulePhase.UNIFIED_ORCHESTRATION

    def _setup(self):
        """Initialize full Rosetta-Helix system."""
        # Heart (Kuramoto)
        self.heart_phases = [2 * math.pi * i / 60 for i in range(60)]
        self.heart_r = 0.0

        # Brain (state)
        self.brain_state = [0.0] * 10

        # TRIAD (tracking)
        self.triad = [PHI_INV, PHI_INV_SQ, 0.0]

        # K-formation tracking
        self.k_formation_active = False
        self.k_formation_duration = 0

    def _train_step(self):
        """
        Full Rosetta-Helix training step.
        """
        neg = compute_negentropy(self.state.z)

        # Heart: Kuramoto dynamics
        K = 2.0 * neg
        cos_sum = sum(math.cos(p) for p in self.heart_phases)
        sin_sum = sum(math.sin(p) for p in self.heart_phases)
        self.heart_r = math.sqrt(cos_sum**2 + sin_sum**2) / 60
        psi = math.atan2(sin_sum, cos_sum)

        # Update phases
        for i in range(60):
            dphase = 1.0 + K * self.heart_r * math.sin(psi - self.heart_phases[i])
            self.heart_phases[i] = (self.heart_phases[i] + dphase * 0.01) % (2 * math.pi)

        # Brain: state evolution
        input_signal = self.heart_r
        for i in range(len(self.brain_state)):
            self.brain_state[i] = 0.9 * self.brain_state[i] + 0.1 * input_signal
            input_signal = math.tanh(self.brain_state[i])

        # TRIAD: update
        self.triad[0] = PHI_INV + 0.3 * neg  # kappa
        self.triad[1] = 1.0 - self.triad[0]  # lambda (conservation!)
        self.triad[2] = neg  # eta

        # Check K-formation
        eta = self.triad[2]
        R = int(7 + 3 * neg)
        if check_k_formation(self.state.kappa, eta, R):
            if not self.k_formation_active:
                self.k_formation_active = True
            self.k_formation_duration += 1
        else:
            self.k_formation_active = False

        # Evolve z from heart coherence
        self.state.z += (Z_CRITICAL - self.state.z) * self.heart_r * ALPHA_STRONG
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Kappa from TRIAD
        self.state.kappa = self.triad[0]
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

    def _teardown(self):
        """Record Rosetta-Helix metrics."""
        self.result.metrics['heart_r'] = self.heart_r
        self.result.metrics['brain_output'] = self.brain_state[-1]
        self.result.metrics['triad'] = self.triad
        self.result.metrics['k_formation_active'] = self.k_formation_active
        self.result.metrics['k_formation_duration'] = self.k_formation_duration
