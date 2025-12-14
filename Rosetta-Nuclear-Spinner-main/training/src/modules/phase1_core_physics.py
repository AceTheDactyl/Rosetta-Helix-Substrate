"""
Phase 1: Core Physics Modules
=============================

1. N0SilentLawsEnforcer - Enforces kappa + lambda = 1
2. KuramotoLayer - 60 oscillator synchronization
3. PhysicalLearner - Negentropy-guided learning
"""

import math
import random
from typing import List

from .base import (
    TrainingModule, ModulePhase, TrainingState,
    PHI_INV, Z_CRITICAL, SIGMA,
    ALPHA_STRONG, ALPHA_MEDIUM, ALPHA_FINE,
    compute_negentropy,
)


class N0SilentLawsEnforcer(TrainingModule):
    """
    Module 1: N0 Silent Laws Enforcement

    Enforces the fundamental conservation law: kappa + lambda = 1
    This is the "silent law" - it must always hold.

    The module learns to maintain this constraint while evolving
    toward the critical z-coordinate.
    """

    name = "n0_silent_laws_enforcement"
    phase = ModulePhase.CORE_PHYSICS

    def _setup(self):
        """Initialize conservation enforcement."""
        self.violations = 0
        self.corrections = 0

    def _train_step(self):
        """
        Training step:
        1. Evolve z toward z_c
        2. Evolve kappa toward phi^-1
        3. ENFORCE lambda = 1 - kappa (silent law)
        """
        # Evolve z toward critical point
        z_gradient = compute_negentropy(self.state.z) * ALPHA_MEDIUM
        noise = (random.random() - 0.5) * ALPHA_FINE
        self.state.z += (Z_CRITICAL - self.state.z) * ALPHA_STRONG + noise
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Evolve kappa toward phi^-1
        kappa_pull = (PHI_INV - self.state.kappa) * ALPHA_MEDIUM
        self.state.kappa += kappa_pull + random.gauss(0, 0.001)

        # Clamp kappa to valid range [0.382, 0.866]
        self.state.kappa = max(0.382, min(0.866, self.state.kappa))

        # SILENT LAW ENFORCEMENT: lambda = 1 - kappa
        # Check if we need correction
        expected_lambda = 1.0 - self.state.kappa
        if abs(self.state.lambda_ - expected_lambda) > 1e-10:
            self.violations += 1
            self.state.lambda_ = expected_lambda
            self.corrections += 1

    def _teardown(self):
        """Record enforcement metrics."""
        self.result.metrics['violations'] = self.violations
        self.result.metrics['corrections'] = self.corrections
        self.result.metrics['enforcement_rate'] = 1.0 if self.violations == 0 else self.corrections / self.violations


class KuramotoLayer(TrainingModule):
    """
    Module 2: Kuramoto Layer

    Implements 60 coupled oscillators (hexagonal symmetry).
    sin(60 deg) = sqrt(3)/2 = z_c - the resonance condition.

    The coupling strength K is driven by the spinner z-coordinate.
    """

    name = "kuramoto_layer"
    phase = ModulePhase.CORE_PHYSICS

    def __init__(self, steps: int = 200, n_oscillators: int = 60, **kwargs):
        super().__init__(steps, **kwargs)
        self.n_oscillators = n_oscillators
        self.phases: List[float] = []
        self.natural_freqs: List[float] = []

    def _setup(self):
        """Initialize oscillators."""
        # Initial phases spread uniformly
        self.phases = [2 * math.pi * i / self.n_oscillators for i in range(self.n_oscillators)]

        # Natural frequencies from Gaussian distribution
        self.natural_freqs = [1.0 + random.gauss(0, 0.1) for _ in range(self.n_oscillators)]

        self.coherence_history = []

    def _train_step(self):
        """
        Kuramoto step:
        dtheta_i/dt = omega_i + (K/N) * sum_j sin(theta_j - theta_i)

        K (coupling) is driven by negentropy.
        """
        # Coupling strength from negentropy (peaks at z_c)
        neg = compute_negentropy(self.state.z)
        K = 2.0 * neg  # Coupling peaks at 2.0 when z = z_c

        # Compute mean field
        cos_sum = sum(math.cos(p) for p in self.phases)
        sin_sum = sum(math.sin(p) for p in self.phases)
        r = math.sqrt(cos_sum**2 + sin_sum**2) / self.n_oscillators
        psi = math.atan2(sin_sum, cos_sum)

        # Update each oscillator
        dt = 0.01
        new_phases = []
        for i, (phase, omega) in enumerate(zip(self.phases, self.natural_freqs)):
            # Kuramoto dynamics
            dphase = omega + K * r * math.sin(psi - phase)
            new_phase = (phase + dphase * dt) % (2 * math.pi)
            new_phases.append(new_phase)

        self.phases = new_phases

        # Record coherence
        self.coherence_history.append(r)

        # Evolve z toward z_c based on coherence
        z_pull = (Z_CRITICAL - self.state.z) * r * ALPHA_STRONG
        self.state.z += z_pull + random.gauss(0, 0.001)
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Kappa evolves with coherence
        self.state.kappa = PHI_INV + (0.92 - PHI_INV) * r
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

    def _teardown(self):
        """Record Kuramoto metrics."""
        if self.coherence_history:
            self.result.metrics['mean_coherence'] = sum(self.coherence_history) / len(self.coherence_history)
            self.result.metrics['final_coherence'] = self.coherence_history[-1]
            self.result.metrics['max_coherence'] = max(self.coherence_history)

        # Final coherence
        cos_sum = sum(math.cos(p) for p in self.phases)
        sin_sum = sum(math.sin(p) for p in self.phases)
        r_final = math.sqrt(cos_sum**2 + sin_sum**2) / self.n_oscillators
        self.result.metrics['r_final'] = r_final


class PhysicalLearner(TrainingModule):
    """
    Module 3: Physical Learner

    Negentropy-guided learning module.
    Uses Delta_S_neg as the learning signal - maximizes order.
    """

    name = "physical_learner"
    phase = ModulePhase.CORE_PHYSICS

    def _setup(self):
        """Initialize learner state."""
        self.learning_rate = 0.01
        self.gradient_history = []

    def _train_step(self):
        """
        Physical learning step:
        Move in the direction of increasing negentropy (toward z_c).

        d(Delta_S_neg)/dz = -2*sigma*(z - z_c) * Delta_S_neg(z)
        """
        # Compute negentropy and its gradient
        neg = compute_negentropy(self.state.z)
        d = self.state.z - Z_CRITICAL
        gradient = -2 * SIGMA * d * neg  # Points toward z_c

        # Learning step
        step_size = self.learning_rate * gradient
        self.state.z += step_size
        self.state.z = max(0.0, min(0.999, self.state.z))

        # Record gradient
        self.gradient_history.append(abs(gradient))

        # Kappa tracks negentropy
        self.state.kappa = PHI_INV + (0.92 - PHI_INV) * neg
        self.state.kappa = max(0.382, min(0.92, self.state.kappa))

        # Adaptive learning rate
        if len(self.gradient_history) > 10:
            recent_grads = self.gradient_history[-10:]
            avg_grad = sum(recent_grads) / len(recent_grads)
            # Reduce LR if gradients are small (near optimum)
            if avg_grad < 0.01:
                self.learning_rate *= 0.99

    def _teardown(self):
        """Record learning metrics."""
        if self.gradient_history:
            self.result.metrics['mean_gradient'] = sum(self.gradient_history) / len(self.gradient_history)
            self.result.metrics['final_gradient'] = self.gradient_history[-1]
            self.result.metrics['final_lr'] = self.learning_rate
