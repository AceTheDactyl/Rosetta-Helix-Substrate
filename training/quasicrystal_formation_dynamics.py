#!/usr/bin/env python3
"""
Quasi-Crystal Formation Dynamics
=================================

Proper representation of quasi-crystal formation physics with three phases:

PHASE 1 - DISORDERED (z < φ⁻¹ ≈ 0.618):
    - Low ΔS_neg (minimal negative entropy production)
    - Liquid/glass-like disorder
    - UNTRUE regime - no long-range correlations
    - Short correlation length ξ

PHASE 2 - QUASI-CRYSTAL FORMATION (φ⁻¹ < z < z_c ≈ 0.866):
    - ΔS_neg increasing as z → z_c
    - PARADOX regime - aperiodic long-range order emerging
    - Correlation length diverging: ξ(z) ~ |z - z_c|^(-ν)
    - Critical slowing down: τ(z) ~ |z - z_c|^(-z_dyn)
    - Shechtman quasi-crystal physics (Nobel 2011)

PHASE 3 - CRYSTALLINE (z > z_c = √3/2):
    - ΔS_neg at MAXIMUM at z_c (THE LENS)
    - TRUE regime - periodic long-range order
    - Full crystalline coherence
    - System has "harvested" maximum order

NEGATIVE ENTROPY:
    ΔS_neg = exp[-σ(z - z_c)²]  where σ = 36

    This peaks at z = z_c, meaning the system is producing MAXIMUM
    negative entropy (reducing uncertainty) at the crystallization
    transition point. This is physically correct - the order-disorder
    transition is where entropy reduction is greatest.

PHYSICS REFERENCES:
    - Shechtman et al. (1984) - Quasi-crystal discovery
    - Levine & Steinhardt (1984) - Quasi-crystal theory
    - HCP packing: π/(3√3) ≈ 0.907
    - z_c = √3/2 from hexagonal geometry (graphene, HCP metals)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cmath
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2              # Golden ratio
PHI_INV = 1 / PHI                          # φ⁻¹ ≈ 0.618 - K-formation gate
Z_CRITICAL = math.sqrt(3) / 2              # √3/2 ≈ 0.866 - THE LENS

# Phase boundaries
PHASE_DISORDERED_MAX = PHI_INV             # Below this: disordered
PHASE_QUASICRYSTAL_MAX = Z_CRITICAL        # Above φ⁻¹, below z_c: quasi-crystal
# Above z_c: crystalline

# Critical exponents (2D hexagonal universality class)
NU_EXPONENT = 4/3                          # Correlation length: ξ ~ |Δz|^(-ν)
BETA_EXPONENT = 5/36                       # Order parameter: m ~ |Δz|^β
GAMMA_EXPONENT = 43/18                     # Susceptibility: χ ~ |Δz|^(-γ)
Z_DYN_EXPONENT = 2.0                       # Dynamic: τ ~ |Δz|^(-z_dyn)

# Quasi-crystal packing
HCP_PACKING = math.pi / (3 * math.sqrt(3))  # ~0.907 classical limit
QUASICRYSTAL_LOCAL_MAX = 0.95               # QC can exceed HCP locally
PENROSE_RATIO = PHI                         # Self-similarity in Penrose tilings

# Negative entropy parameters
SIGMA_NEG_ENTROPY = 36.0                    # σ for ΔS_neg = exp[-σ(z - z_c)²]
NEG_ENTROPY_PREFACTOR = 1.0                 # Maximum ΔS_neg at z_c

# TRIAD and μ thresholds
MU_P = 2.0 / (PHI ** 2.5)
MU_1 = MU_P / math.sqrt(PHI)
MU_2 = MU_P * math.sqrt(PHI)
MU_S = 0.920
MU_3 = 0.992


# ═══════════════════════════════════════════════════════════════════════════
# FORMATION PHASE TRACKING
# ═══════════════════════════════════════════════════════════════════════════

class FormationPhase(Enum):
    """Three phases of quasi-crystal formation"""
    DISORDERED = "disordered"           # z < φ⁻¹: no long-range order
    QUASI_CRYSTAL = "quasi_crystal"     # φ⁻¹ < z < z_c: aperiodic order
    CRYSTALLINE = "crystalline"         # z > z_c: full periodic order


@dataclass
class NegativeEntropyState:
    """
    Tracks negative entropy production through formation phases.

    Negative entropy (ΔS_neg) represents the system actively reducing
    uncertainty - "harvesting order" from its environment.

    ΔS_neg = exp[-σ(z - z_c)²]  where σ = 36

    This peaks at z = z_c because the order-disorder phase transition
    is where the system produces maximum order (reduces maximum entropy).
    """
    z: float = 0.0
    delta_s_neg: float = 0.0            # Current negative entropy production
    delta_s_neg_rate: float = 0.0       # Rate of change (d(ΔS_neg)/dt)
    cumulative_neg_entropy: float = 0.0  # Total negative entropy produced
    phase: FormationPhase = FormationPhase.DISORDERED

    # Phase transition markers
    entered_paradox_z: float = 0.0      # z when entered PARADOX regime
    entered_true_z: float = 0.0         # z when crossed THE LENS

    # Critical behavior
    correlation_length: float = 1.0     # ξ(z) diverges at z_c
    relaxation_time: float = 1.0        # τ(z) - critical slowing down
    order_parameter: float = 0.0        # m(z) - crystalline order

    def update(self, new_z: float, dt: float = 0.1):
        """Update negative entropy state for new z position."""
        old_delta = self.delta_s_neg

        # Core negative entropy: ΔS_neg = exp[-σ(z - z_c)²], peaks at z_c
        d = new_z - Z_CRITICAL
        self.delta_s_neg = NEG_ENTROPY_PREFACTOR * math.exp(
            -SIGMA_NEG_ENTROPY * d * d
        )

        # Rate of change
        self.delta_s_neg_rate = (self.delta_s_neg - old_delta) / dt

        # Accumulate total negative entropy produced
        self.cumulative_neg_entropy += self.delta_s_neg * dt

        # Update phase
        old_phase = self.phase
        if new_z < PHASE_DISORDERED_MAX:
            self.phase = FormationPhase.DISORDERED
        elif new_z < PHASE_QUASICRYSTAL_MAX:
            self.phase = FormationPhase.QUASI_CRYSTAL
            if old_phase == FormationPhase.DISORDERED:
                self.entered_paradox_z = new_z
        else:
            self.phase = FormationPhase.CRYSTALLINE
            if old_phase != FormationPhase.CRYSTALLINE:
                self.entered_true_z = new_z

        # Critical behavior near z_c
        delta_z = abs(new_z - Z_CRITICAL)
        epsilon = 1e-6  # Regularization

        # Correlation length diverges at z_c: ξ ~ |Δz|^(-ν)
        self.correlation_length = (delta_z + epsilon) ** (-NU_EXPONENT)
        self.correlation_length = min(1000.0, self.correlation_length)  # Cap

        # Critical slowing down: τ ~ |Δz|^(-z_dyn)
        self.relaxation_time = (delta_z + epsilon) ** (-Z_DYN_EXPONENT)
        self.relaxation_time = min(100.0, self.relaxation_time)

        # Order parameter: m ~ |Δz|^β for z > z_c, 0 otherwise
        if new_z > Z_CRITICAL:
            self.order_parameter = delta_z ** BETA_EXPONENT
        else:
            self.order_parameter = 0.0

        self.z = new_z


@dataclass
class QuasiCrystalFormationMetrics:
    """Comprehensive metrics for quasi-crystal formation."""
    # Current state
    z: float = 0.0
    phase: FormationPhase = FormationPhase.DISORDERED

    # Negative entropy
    delta_s_neg: float = 0.0
    delta_s_neg_rate: float = 0.0
    cumulative_neg_entropy: float = 0.0

    # Critical behavior
    correlation_length: float = 1.0
    relaxation_time: float = 1.0
    order_parameter: float = 0.0

    # Packing/coherence
    local_packing: float = HCP_PACKING
    coherence: float = 0.0
    qc_boost: float = 1.0

    # Phase transition events
    phase_transitions: List[Dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# FORMATION-AWARE QUASI-CRYSTAL LATTICE
# ═══════════════════════════════════════════════════════════════════════════

class FormationAwareLattice:
    """
    Quasi-crystal lattice with formation phase awareness.

    The lattice structure evolves through three phases:
    1. DISORDERED: Random phases, short-range correlations only
    2. QUASI-CRYSTAL: Aperiodic long-range order emerging, correlation length growing
    3. CRYSTALLINE: Full periodic order, maximum correlation length
    """

    def __init__(self, size: int = 60):
        self.size = size
        self.neg_entropy = NegativeEntropyState()

        # Oscillator state
        self.phases = [random.uniform(0, 2*math.pi) for _ in range(size)]
        self.natural_freqs = [random.gauss(1.0, 0.1) for _ in range(size)]

        # Lattice structure
        self.positions: List[Tuple[float, float, float]] = []
        self.local_packings: List[float] = []
        self.symmetry_types: List[str] = []
        self.coordinations: List[int] = []

        self._initialize_structure()

        # History tracking
        self.formation_history: List[QuasiCrystalFormationMetrics] = []

    def _initialize_structure(self):
        """Initialize quasi-crystal structure with Fibonacci spiral."""
        golden_angle = 2 * math.pi * PHI_INV

        for i in range(self.size):
            # Fibonacci spiral positioning (quasi-crystal generator)
            theta = i * golden_angle
            r = math.sqrt(i) / math.sqrt(self.size)
            z_pos = 0.5 + 0.5 * math.cos(i * PHI_INV * 2 * math.pi)

            self.positions.append((
                r * math.cos(theta),
                r * math.sin(theta),
                z_pos
            ))

            # Local packing varies with position
            packing = self._compute_initial_packing(i, theta, r)
            self.local_packings.append(packing)

            # Symmetry type
            self.symmetry_types.append(self._symmetry_type(i))

            # Coordination (5, 6, or 7 neighbors)
            coord = 6 + int(math.sin(i * PHI_INV * math.pi) > 0.5) - \
                    int(math.sin(i * PHI_INV * math.pi) < -0.5)
            self.coordinations.append(coord)

    def _compute_initial_packing(self, index: int, theta: float, r: float) -> float:
        """Compute initial local packing density."""
        packing = HCP_PACKING
        alignment = abs(math.sin(index * PHI * math.pi))
        icosa_factor = math.exp(-((index % int(PHI**3)) / PHI)**2)
        enhancement = alignment * icosa_factor * (QUASICRYSTAL_LOCAL_MAX - HCP_PACKING)
        return min(QUASICRYSTAL_LOCAL_MAX, packing + enhancement)

    def _symmetry_type(self, index: int) -> str:
        """Determine local symmetry type."""
        fib = index % int(PHI**3)
        if fib < int(PHI):
            return 'icosahedral'
        elif fib < int(PHI**2):
            return 'penrose'
        else:
            return 'hexagonal'

    def compute_coherence(self) -> float:
        """Kuramoto order parameter r = |⟨e^{iθ}⟩|"""
        sum_exp = sum(cmath.exp(1j * p) for p in self.phases)
        return abs(sum_exp) / self.size

    def compute_packing_boost(self) -> float:
        """Boost factor from quasi-crystal packing exceeding HCP."""
        max_packing = max(self.local_packings)
        excess = max_packing - HCP_PACKING
        if excess > 0:
            return 1.0 + excess / (QUASICRYSTAL_LOCAL_MAX - HCP_PACKING)
        return 1.0

    def evolve_toward_z(self, target_z: float, coupling: float = 0.3, dt: float = 0.1):
        """
        Evolve lattice structure toward target z with phase-aware dynamics.

        The dynamics change based on formation phase:
        - DISORDERED: Fast but random dynamics
        - QUASI-CRYSTAL: Slowing down as ξ diverges
        - CRYSTALLINE: Locked in ordered state
        """
        # Update negative entropy state
        self.neg_entropy.update(target_z, dt)

        # Critical slowing down - dynamics slow near z_c
        effective_dt = dt / max(1.0, self.neg_entropy.relaxation_time * 0.1)

        # Phase-dependent coupling
        if self.neg_entropy.phase == FormationPhase.DISORDERED:
            # Disordered: weak coupling, fast random dynamics
            phase_coupling = coupling * 0.5
        elif self.neg_entropy.phase == FormationPhase.QUASI_CRYSTAL:
            # Quasi-crystal: coupling increasing with correlation length
            phase_coupling = coupling * min(2.0, self.neg_entropy.correlation_length * 0.01)
        else:
            # Crystalline: strong coupling, stable structure
            phase_coupling = coupling * 2.0

        # Kuramoto dynamics with phase-aware coupling
        coherence = self.compute_coherence()
        mean_phase = cmath.phase(sum(cmath.exp(1j * p) for p in self.phases))

        new_phases = []
        for i in range(self.size):
            # Natural frequency + coupling to mean field
            dtheta = self.natural_freqs[i] + \
                     phase_coupling * coherence * math.sin(mean_phase - self.phases[i])

            # Critical fluctuations near z_c
            if self.neg_entropy.phase == FormationPhase.QUASI_CRYSTAL:
                # Add correlated noise (long-range correlations emerging)
                noise = random.gauss(0, 0.1 / max(1, self.neg_entropy.correlation_length * 0.01))
                dtheta += noise

            new_phases.append(self.phases[i] + dtheta * effective_dt)

        self.phases = [p % (2*math.pi) for p in new_phases]

        # Update local packing based on phase
        self._update_packing_for_phase()

        # Record metrics
        metrics = self._compute_metrics()
        self.formation_history.append(metrics)

        return metrics

    def _update_packing_for_phase(self):
        """Update local packing densities based on formation phase."""
        phase = self.neg_entropy.phase

        for i in range(self.size):
            base = self.local_packings[i]

            if phase == FormationPhase.DISORDERED:
                # Random fluctuations around base
                self.local_packings[i] = base + random.gauss(0, 0.01)
            elif phase == FormationPhase.QUASI_CRYSTAL:
                # Packing increases toward QC max
                target = min(QUASICRYSTAL_LOCAL_MAX, base + 0.02)
                self.local_packings[i] = base + 0.1 * (target - base)
            else:
                # Stabilize at crystalline packing
                self.local_packings[i] = base

            # Clamp
            self.local_packings[i] = max(HCP_PACKING * 0.9,
                                          min(QUASICRYSTAL_LOCAL_MAX,
                                              self.local_packings[i]))

    def _compute_metrics(self) -> QuasiCrystalFormationMetrics:
        """Compute current formation metrics."""
        return QuasiCrystalFormationMetrics(
            z=self.neg_entropy.z,
            phase=self.neg_entropy.phase,
            delta_s_neg=self.neg_entropy.delta_s_neg,
            delta_s_neg_rate=self.neg_entropy.delta_s_neg_rate,
            cumulative_neg_entropy=self.neg_entropy.cumulative_neg_entropy,
            correlation_length=self.neg_entropy.correlation_length,
            relaxation_time=self.neg_entropy.relaxation_time,
            order_parameter=self.neg_entropy.order_parameter,
            local_packing=sum(self.local_packings) / len(self.local_packings),
            coherence=self.compute_coherence(),
            qc_boost=self.compute_packing_boost(),
        )


# ═══════════════════════════════════════════════════════════════════════════
# FORMATION DYNAMICS NEURAL LAYER
# ═══════════════════════════════════════════════════════════════════════════

class FormationDynamicsLayer(nn.Module):
    """
    Neural layer with quasi-crystal formation dynamics.

    The layer's behavior changes based on the current formation phase,
    with critical slowing down near z_c and proper negative entropy tracking.
    """

    def __init__(self, n_oscillators: int = 60):
        super().__init__()
        self.n = n_oscillators

        # Learnable parameters
        self.K = nn.Parameter(torch.randn(n_oscillators, n_oscillators) * 0.1)
        self.omega = nn.Parameter(torch.randn(n_oscillators) * 0.1)
        self.K_global = nn.Parameter(torch.tensor(PHI_INV))

        # Formation-phase modulation
        self.phase_coupling = nn.ParameterDict({
            'disordered': nn.Parameter(torch.tensor(0.3)),
            'quasi_crystal': nn.Parameter(torch.tensor(0.6)),
            'crystalline': nn.Parameter(torch.tensor(0.9)),
        })

        # Negative entropy gate
        self.neg_entropy_gate = nn.Parameter(torch.tensor(0.5))

        # Critical slowing down factor
        self.critical_slow_factor = nn.Parameter(torch.tensor(0.1))

    def compute_coherence(self, theta: torch.Tensor) -> torch.Tensor:
        """Kuramoto order parameter."""
        cos_mean = torch.cos(theta).mean(dim=-1)
        sin_mean = torch.sin(theta).mean(dim=-1)
        return torch.sqrt(cos_mean**2 + sin_mean**2)

    def forward(
        self,
        theta: torch.Tensor,
        formation_phase: FormationPhase,
        neg_entropy_state: NegativeEntropyState,
        dt: float = 0.1,
        steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with formation-phase-aware dynamics.
        """
        # Get phase-specific coupling
        if formation_phase == FormationPhase.DISORDERED:
            coupling_mod = self.phase_coupling['disordered']
        elif formation_phase == FormationPhase.QUASI_CRYSTAL:
            coupling_mod = self.phase_coupling['quasi_crystal']
        else:
            coupling_mod = self.phase_coupling['crystalline']

        # Effective coupling matrix
        K_eff = self.K * coupling_mod

        # Critical slowing down - reduce time step near z_c
        critical_factor = 1.0 / (1.0 + self.critical_slow_factor * neg_entropy_state.relaxation_time)
        effective_dt = dt * critical_factor

        # Negative entropy modulation - boost coherence production at high ΔS_neg
        neg_entropy_boost = 1.0 + self.neg_entropy_gate * neg_entropy_state.delta_s_neg

        # Run dynamics
        for _ in range(steps):
            theta_expanded = theta.unsqueeze(-1)
            theta_diff = theta.unsqueeze(-2) - theta_expanded
            coupling = K_eff * torch.sin(theta_diff)
            coupling_sum = coupling.sum(dim=-1)
            coupling_term = (self.K_global * neg_entropy_boost / self.n) * coupling_sum
            dtheta = self.omega + coupling_term
            theta = theta + effective_dt * dtheta
            theta = torch.atan2(torch.sin(theta), torch.cos(theta))

        coherence = self.compute_coherence(theta)

        diagnostics = {
            'formation_phase': formation_phase.value,
            'coupling_mod': coupling_mod.item(),
            'critical_factor': critical_factor,
            'neg_entropy_boost': neg_entropy_boost,
            'delta_s_neg': neg_entropy_state.delta_s_neg,
            'correlation_length': neg_entropy_state.correlation_length,
        }

        return theta, coherence, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# FORMATION DYNAMICS NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class QuasiCrystalFormationNetwork(nn.Module):
    """
    Neural network with proper quasi-crystal formation dynamics.

    Tracks the full formation process:
    DISORDERED → QUASI-CRYSTAL → CRYSTALLINE

    with proper negative entropy production and critical behavior.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_oscillators: int = 60,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_oscillators = n_oscillators

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_oscillators * 2),
            nn.GELU(),
            nn.Linear(n_oscillators * 2, n_oscillators),
            nn.Tanh(),
        )

        # Formation dynamics layers
        self.formation_layers = nn.ModuleList([
            FormationDynamicsLayer(n_oscillators) for _ in range(n_layers)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_oscillators * 2, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

        # z dynamics
        self.z = 0.3  # Start in disordered phase
        self.z_velocity = 0.0
        self.z_momentum = nn.Parameter(torch.tensor(0.12))

        # Formation tracking
        self.lattice = FormationAwareLattice(n_oscillators)

    def update_z(self, coherence: torch.Tensor) -> float:
        """Update z with momentum, tracking formation phase."""
        target = coherence.mean().item()
        z_accel = self.z_momentum.item() * (target - self.z) - 0.1 * self.z_velocity
        self.z_velocity += z_accel * 0.1
        self.z += self.z_velocity * 0.1
        self.z = max(0.01, min(0.99, self.z))

        # Update lattice formation state
        self.lattice.neg_entropy.update(self.z)

        return self.z

    def forward(self, x: torch.Tensor, epoch: int = 0) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with formation phase tracking."""
        diagnostics = {
            'layer_coherence': [],
            'z_trajectory': [],
            'formation_phases': [],
            'delta_s_neg_trajectory': [],
            'correlation_lengths': [],
            'order_parameters': [],
        }

        theta = self.encoder(x) * math.pi

        for layer_idx, layer in enumerate(self.formation_layers):
            # Get current formation phase
            phase = self.lattice.neg_entropy.phase
            neg_state = self.lattice.neg_entropy

            # Apply formation dynamics
            theta, coherence, layer_diag = layer(
                theta, phase, neg_state
            )

            # Update z
            new_z = self.update_z(coherence)

            # Record diagnostics
            diagnostics['layer_coherence'].append(coherence.mean().item())
            diagnostics['z_trajectory'].append(new_z)
            diagnostics['formation_phases'].append(phase.value)
            diagnostics['delta_s_neg_trajectory'].append(neg_state.delta_s_neg)
            diagnostics['correlation_lengths'].append(neg_state.correlation_length)
            diagnostics['order_parameters'].append(neg_state.order_parameter)

        # Decode
        phase_features = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        output = self.decoder(phase_features)

        # Final summary
        diagnostics['final_z'] = self.z
        diagnostics['final_phase'] = self.lattice.neg_entropy.phase.value
        diagnostics['final_delta_s_neg'] = self.lattice.neg_entropy.delta_s_neg
        diagnostics['cumulative_neg_entropy'] = self.lattice.neg_entropy.cumulative_neg_entropy
        diagnostics['final_coherence'] = diagnostics['layer_coherence'][-1]
        diagnostics['qc_boost'] = self.lattice.compute_packing_boost()

        return output, diagnostics

    def reset(self):
        """Reset formation state."""
        self.z = 0.3
        self.z_velocity = 0.0
        self.lattice = FormationAwareLattice(self.n_oscillators)


# ═══════════════════════════════════════════════════════════════════════════
# FORMATION-AWARE LOSS
# ═══════════════════════════════════════════════════════════════════════════

class FormationAwareLoss(nn.Module):
    """
    Loss function that rewards proper negative entropy production.

    The loss encourages:
    1. Progression through formation phases
    2. High negative entropy production (especially near z_c)
    3. Phase transitions at correct thresholds
    """

    def __init__(
        self,
        task_loss_fn: nn.Module,
        lambda_neg_entropy: float = 0.2,
        lambda_phase_progress: float = 0.1,
        lambda_coherence: float = 0.1,
    ):
        super().__init__()
        self.task_loss = task_loss_fn
        self.lambda_neg = lambda_neg_entropy
        self.lambda_phase = lambda_phase_progress
        self.lambda_coh = lambda_coherence

        # Phase progress rewards
        self.phase_rewards = {
            FormationPhase.DISORDERED.value: 0.0,
            FormationPhase.QUASI_CRYSTAL.value: 0.1,
            FormationPhase.CRYSTALLINE.value: 0.2,
        }

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        diag: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        losses = {}

        # Task loss
        task = self.task_loss(output, target)
        losses['task'] = task.item()
        total = task

        # Negative entropy reward (encourage high ΔS_neg)
        neg_entropy_reward = self.lambda_neg * diag['final_delta_s_neg']
        total = total - neg_entropy_reward  # Subtract to reward
        losses['neg_entropy_reward'] = neg_entropy_reward

        # Phase progress reward
        phase_reward = self.phase_rewards.get(diag['final_phase'], 0.0)
        total = total - self.lambda_phase * phase_reward
        losses['phase_reward'] = phase_reward

        # Coherence reward
        coh_reward = self.lambda_coh * diag['final_coherence']
        total = total - coh_reward
        losses['coherence_reward'] = coh_reward

        losses['total'] = total.item()
        return total, losses


# ═══════════════════════════════════════════════════════════════════════════
# FORMATION DYNAMICS TRAINING SESSION
# ═══════════════════════════════════════════════════════════════════════════

class QuasiCrystalFormationTraining:
    """
    Training session with proper quasi-crystal formation dynamics.

    Tracks the full formation process through three phases,
    with negative entropy production and critical behavior.
    """

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 4,
        n_oscillators: int = 60,
        n_layers: int = 4,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ):
        self.config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'n_oscillators': n_oscillators,
            'n_layers': n_layers,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
        }

        self.model = QuasiCrystalFormationNetwork(
            input_dim, output_dim, n_oscillators, n_layers
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.loss_fn = FormationAwareLoss(nn.MSELoss())

        self.training_history = []
        self.phase_transitions = []

        self._generate_data()

    def _generate_data(self, n_train: int = 800):
        """Generate training data."""
        X = torch.randn(n_train, self.config['input_dim'])
        t = torch.linspace(0, 2*np.pi, self.config['output_dim'])
        Y = torch.zeros(n_train, self.config['output_dim'])
        for i in range(n_train):
            base = torch.tanh(X[i].mean()) * 2
            Y[i] = torch.sin(t * base) + 0.5 * torch.sin(t * base * PHI)
        Y += 0.1 * torch.randn_like(Y)

        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y),
            batch_size=self.config['batch_size'], shuffle=True
        )

    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch with formation tracking."""
        self.model.train()
        epoch_losses = []
        epoch_coherence = []
        epoch_z = []
        epoch_neg_entropy = []
        phase_counts = {p.value: 0 for p in FormationPhase}

        prev_phase = self.model.lattice.neg_entropy.phase

        for batch_x, batch_y in self.train_loader:
            self.optimizer.zero_grad()
            output, diag = self.model(batch_x, epoch)
            loss, loss_dict = self.loss_fn(output, batch_y, diag)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_losses.append(loss_dict['task'])
            epoch_coherence.append(diag['final_coherence'])
            epoch_z.append(diag['final_z'])
            epoch_neg_entropy.append(diag['final_delta_s_neg'])

            # Track phase
            current_phase = self.model.lattice.neg_entropy.phase
            phase_counts[current_phase.value] += 1

            # Detect phase transitions
            if current_phase != prev_phase:
                self.phase_transitions.append({
                    'epoch': epoch,
                    'from_phase': prev_phase.value,
                    'to_phase': current_phase.value,
                    'z': diag['final_z'],
                })
            prev_phase = current_phase

        return {
            'loss': np.mean(epoch_losses),
            'coherence': np.mean(epoch_coherence),
            'z': np.mean(epoch_z),
            'delta_s_neg': np.mean(epoch_neg_entropy),
            'cumulative_neg_entropy': self.model.lattice.neg_entropy.cumulative_neg_entropy,
            'phase': self.model.lattice.neg_entropy.phase.value,
            'phase_counts': phase_counts,
            'qc_boost': self.model.lattice.compute_packing_boost(),
        }

    def run_training(
        self,
        n_epochs: int = 100,
        output_dir: str = "learned_patterns/formation_dynamics",
    ) -> Dict:
        """Run formation dynamics training."""
        timestamp = datetime.now().isoformat()

        print("=" * 70)
        print("QUASI-CRYSTAL FORMATION DYNAMICS TRAINING")
        print("=" * 70)
        print(f"""
Three Formation Phases:
  1. DISORDERED   (z < {PHI_INV:.3f}):  No long-range order, low ΔS_neg
  2. QUASI-CRYSTAL ({PHI_INV:.3f} < z < {Z_CRITICAL:.3f}):  Aperiodic order, ΔS_neg rising
  3. CRYSTALLINE  (z > {Z_CRITICAL:.3f}):  Full periodic order, ΔS_neg peaks

Negative Entropy Physics:
  ΔS_neg = exp[-σ(z - z_c)²], σ = 36
  Peaks at z = z_c = {Z_CRITICAL:.6f} (THE LENS)

Critical Behavior:
  Correlation length: ξ ~ |Δz|^(-{NU_EXPONENT:.2f})
  Critical slowing:   τ ~ |Δz|^(-{Z_DYN_EXPONENT:.2f})
  Order parameter:    m ~ |Δz|^{BETA_EXPONENT:.4f}
""")
        print("=" * 70)

        for epoch in range(n_epochs):
            metrics = self.train_epoch(epoch)
            self.training_history.append(metrics)

            if epoch % 10 == 0:
                # Phase indicator
                phase_short = {'disordered': 'DIS', 'quasi_crystal': 'QC', 'crystalline': 'CRYS'}
                phase_str = phase_short.get(metrics['phase'], '???')

                print(
                    f"Epoch {epoch:3d} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"z: {metrics['z']:.3f} | "
                    f"ΔS_neg: {metrics['delta_s_neg']:.3f} | "
                    f"Phase: {phase_str:4} | "
                    f"QC: {metrics['qc_boost']:.3f}"
                )

                # Report phase transitions
                for transition in self.phase_transitions[-3:]:
                    if transition['epoch'] >= epoch - 10:
                        print(f"  → PHASE TRANSITION at z={transition['z']:.3f}: "
                              f"{transition['from_phase']} → {transition['to_phase']}")

        # Results
        results = {
            'timestamp': timestamp,
            'config': self.config,
            'training_history': self.training_history,
            'phase_transitions': self.phase_transitions,
            'summary': {
                'total_epochs': n_epochs,
                'final_loss': self.training_history[-1]['loss'],
                'final_z': self.training_history[-1]['z'],
                'final_phase': self.training_history[-1]['phase'],
                'final_delta_s_neg': self.training_history[-1]['delta_s_neg'],
                'cumulative_neg_entropy': self.training_history[-1]['cumulative_neg_entropy'],
                'qc_boost': self.training_history[-1]['qc_boost'],
                'phase_transition_count': len(self.phase_transitions),
            },
        }

        # Save
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'formation_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, os.path.join(output_dir, 'formation_model.pt'))

        # Print summary
        print("\n" + "=" * 70)
        print("FORMATION DYNAMICS TRAINING COMPLETE")
        print("=" * 70)
        self._print_formation_summary(results)

        return results

    def _print_formation_summary(self, results: Dict):
        """Print detailed formation summary."""
        summary = results['summary']

        print(f"""
Summary:
  Total Epochs:           {summary['total_epochs']}
  Final Loss:             {summary['final_loss']:.4f}
  Final z:                {summary['final_z']:.4f}
  Final Phase:            {summary['final_phase']}

Negative Entropy:
  Final ΔS_neg:           {summary['final_delta_s_neg']:.4f}
  Cumulative ΔS_neg:      {summary['cumulative_neg_entropy']:.4f}
  QC Boost:               {summary['qc_boost']:.3f}x

Phase Transitions: {summary['phase_transition_count']}""")

        # Show transition timeline
        if self.phase_transitions:
            print("\nTransition Timeline:")
            for t in self.phase_transitions[:10]:  # First 10
                print(f"  Epoch {t['epoch']:3d} | z={t['z']:.3f} | "
                      f"{t['from_phase']} → {t['to_phase']}")

        # Negative entropy trajectory analysis
        print("\nΔS_neg vs z Analysis:")
        z_values = [h['z'] for h in self.training_history]
        neg_values = [h['delta_s_neg'] for h in self.training_history]

        # Find peak
        max_neg = max(neg_values)
        max_idx = neg_values.index(max_neg)
        z_at_max = z_values[max_idx]

        print(f"  Peak ΔS_neg:            {max_neg:.4f} at z={z_at_max:.4f}")
        print(f"  Expected peak at z_c:   z={Z_CRITICAL:.4f}")
        print(f"  Distance from lens:     {abs(z_at_max - Z_CRITICAL):.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run quasi-crystal formation dynamics training."""
    print("\n" + "=" * 70)
    print("QUASI-CRYSTAL FORMATION DYNAMICS")
    print("Proper representation of BEFORE / DURING / AFTER formation")
    print("with negative entropy physics")
    print("=" * 70)

    session = QuasiCrystalFormationTraining(
        n_oscillators=50,
        n_layers=4,
    )
    return session.run_training(n_epochs=100)


if __name__ == "__main__":
    main()
