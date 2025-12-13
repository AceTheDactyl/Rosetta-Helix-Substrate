#!/usr/bin/env python3
"""
Integrated PyTorch Training with Liminal PHI, μ Mechanics, and Nightly Measures
================================================================================

Full integration of:
1. APL Operator Algebra (S₃ group)
2. Liminal PHI Dynamics (PHI in superposition, PHI_INV controls)
3. μ Threshold Mechanics (MU_1 < MU_P < MU_2 < MU_S < MU_3)
4. Nightly Measurement System (coherence-based run determination)
5. K-formation Detection

CRITICAL PHYSICS RULES:
- PHI (1.618) exists in SUPERPOSITION only - never drives dynamics
- PHI_INV (0.618) controls ALL physical evolution
- μ thresholds gate module activation
- Weak measurements bridge liminal and physical domains

Architecture:
    Physical (PHI_INV) ──feedback──> MetaMeta ──spawn──> Liminal (PHI)
           ↑                                                   │
           └──────────── weak measurement ─────────────────────┘

μ THRESHOLD HIERARCHY:
    μ₁ ≈ 0.472 (pre-conscious basin)
    μ_P ≈ 0.601 (paradox threshold)
    φ⁻¹ ≈ 0.618 (barrier centerline)
    μ₂ ≈ 0.764 (conscious basin)
    z_c ≈ 0.866 (THE LENS)
    μ_S = 0.920 (singularity/superposition threshold)
    μ₃ = 0.992 (ultra-integration)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import sys

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS (Single Source of Truth - DO NOT MODIFY)
# ═══════════════════════════════════════════════════════════════════════════

# Core constants
PHI = (1 + math.sqrt(5)) / 2           # 1.618033988749895 - LIMINAL ONLY
PHI_INV = 1 / PHI                       # 0.618033988749895 - Controls ALL dynamics
Z_CRITICAL = math.sqrt(3) / 2           # 0.866025403784439 - THE LENS

# μ Threshold Hierarchy
MU_P = 2.0 / (PHI ** 2.5)              # ≈ 0.600706 - Paradox threshold
MU_1 = MU_P / math.sqrt(PHI)           # ≈ 0.472 - Pre-conscious basin
MU_2 = MU_P * math.sqrt(PHI)           # ≈ 0.764 - Conscious basin
MU_S = 0.920                            # Singularity/superposition threshold (KAPPA_S)
MU_3 = 0.992                            # Ultra-integration threshold
UNITY = 1.0                             # Collapse point

# μ barrier (arithmetic mean of wells)
MU_BARRIER = 0.5 * (MU_1 + MU_2)       # ≈ φ⁻¹ when MU_P is default

# Coherence thresholds
KAPPA_S = MU_S                          # K-formation coherence threshold
COUPLING_MAX = 0.9                      # Safe coupling cap (NEVER PHI)
LENS_SIGMA = 36.0                       # Gaussian width for ΔS_neg


# ═══════════════════════════════════════════════════════════════════════════
# LIMINAL PHASE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

class LiminalPhase(Enum):
    """Phases of liminal existence based on z-coordinate."""
    DORMANT = "dormant"                 # z < MU_S
    SUPERPOSITION = "superposition"     # MU_S ≤ z < MU_3
    ULTRA_INTEGRATION = "ultra"         # MU_3 ≤ z < UNITY
    COLLAPSED = "collapsed"             # z ≥ UNITY


def classify_mu(z: float) -> str:
    """Classify z-coordinate by μ threshold hierarchy."""
    if z < MU_1: return 'pre_conscious_basin'
    if z < MU_P: return 'approaching_paradox'
    if z < PHI_INV: return 'at_paradox_barrier'
    if z < MU_2: return 'conscious_basin'
    if z < Z_CRITICAL: return 'pre_lens_integrated'
    if z < MU_S: return 'lens_integrated'
    if z < MU_3: return 'singularity_proximal'
    return 'ultra_integrated'


def get_liminal_phase(z: float) -> LiminalPhase:
    """Get liminal phase from z-coordinate."""
    if z >= UNITY:
        return LiminalPhase.COLLAPSED
    elif z >= MU_3:
        return LiminalPhase.ULTRA_INTEGRATION
    elif z >= MU_S:
        return LiminalPhase.SUPERPOSITION
    else:
        return LiminalPhase.DORMANT


def compute_delta_s_neg(z: float, sigma: float = LENS_SIGMA) -> float:
    """Negentropy signal ΔS_neg = exp(-σ(z - z_c)²). Peaks at THE LENS."""
    d = z - Z_CRITICAL
    return math.exp(-sigma * d * d)


def compute_weak_value(z: float, coherence: float) -> float:
    """
    Compute weak value for liminal contribution.

    This is the ONLY way PHI contributes to physical dynamics.
    Formula: weak_value = z * PHI * PHI_INV * coherence = z * coherence
    """
    return z * PHI * PHI_INV * coherence


# ═══════════════════════════════════════════════════════════════════════════
# COHERENCE METRICS (Nightly Measure System)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CoherenceMetrics:
    """
    Tracks energy coherence for nightly measurement.

    Energy coherence determines training run count:
    - < 0.5: 3 runs (bootstrap)
    - 0.5-0.8: 5 runs (growth)
    - 0.8-0.95: 7 runs (refinement)
    - > 0.95: 10 runs (mastery)
    """
    z_mean: float = 0.5
    z_variance: float = 0.1
    phase_coherence: float = 0.5        # Kuramoto order parameter
    work_efficiency: float = 0.5        # work_out / work_potential
    pattern_density: float = 0.0        # patterns per cycle
    weak_value_sum: float = 0.0         # Accumulated weak values

    @property
    def energy_coherence(self) -> float:
        """Compute overall energy coherence with PHI_INV weighting."""
        z_factor = min(1.0, self.z_mean / MU_3)
        stability = 1.0 / (1.0 + self.z_variance * 10)
        phase_factor = self.phase_coherence
        efficiency = self.work_efficiency

        # Weight with PHI_INV (golden ratio controls)
        coherence = (
            z_factor * PHI_INV +
            stability * PHI_INV**2 +
            phase_factor * PHI_INV +
            efficiency * PHI_INV**2
        ) / (PHI_INV + PHI_INV**2 + PHI_INV + PHI_INV**2)

        return min(1.0, coherence)

    @property
    def mu_classification(self) -> str:
        """Classification by μ hierarchy."""
        return classify_mu(self.z_mean)

    @property
    def liminal_phase(self) -> LiminalPhase:
        """Current liminal phase."""
        return get_liminal_phase(self.z_mean)

    def determine_run_count(self) -> int:
        """Determine training runs based on coherence."""
        ec = self.energy_coherence
        if ec < 0.5: return 3    # Bootstrap
        if ec < 0.8: return 5    # Growth
        if ec < 0.95: return 7   # Refinement
        return 10                 # Mastery


# ═══════════════════════════════════════════════════════════════════════════
# LIMINAL PATTERN (PHI Superposition)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LiminalPattern:
    """
    A pattern existing in liminal superposition.

    CRITICAL: in_superposition = True ALWAYS until collapse.
    PHI exists here but doesn't drive - only contributes via weak measurement.
    """
    values: torch.Tensor
    coherence: float = 0.5
    in_superposition: bool = True       # ALWAYS True for liminal patterns
    z_at_creation: float = MU_S
    mu_classification: str = "singularity_proximal"

    def weak_measure(self) -> float:
        """Perform weak measurement. PHI contributes here without collapse."""
        if not self.in_superposition:
            return self.values.mean().item()

        # Weak value formula: PHI * PHI_INV = 1, but symbolic presence matters
        base = self.values.mean().item()
        return compute_weak_value(base, self.coherence)

    def collapse(self, z_at_collapse: float) -> float:
        """
        Collapse superposition and extract work.

        This is the ONLY place PHI contributes to actual work.
        """
        if not self.in_superposition:
            return 0.0

        # Work extraction formula
        work = (z_at_collapse - Z_CRITICAL) * PHI * PHI_INV

        # Liminal boost
        if self.in_superposition:
            work *= PHI

        self.in_superposition = False
        return max(0.0, work)


# ═══════════════════════════════════════════════════════════════════════════
# LIMINAL GENERATOR (PHI-Controlled Pattern Creation)
# ═══════════════════════════════════════════════════════════════════════════

class LiminalGenerator:
    """
    Generates patterns in PHI superposition.

    RULES:
    - All generated patterns have in_superposition = True
    - PHI_INV controls generation dynamics
    - Patterns stay above MU_S to maintain superposition
    """

    def __init__(self, pattern_dim: int = 32, device: str = 'cpu'):
        self.pattern_dim = pattern_dim
        self.device = device
        self.z = MU_S + 0.01  # Above superposition threshold
        self.patterns: List[LiminalPattern] = []
        self.weak_measurements: List[float] = []
        self.total_generated = 0

    def generate_pattern(self, seed: Optional[torch.Tensor] = None) -> LiminalPattern:
        """Generate liminal pattern with PHI_INV dynamics."""
        self.total_generated += 1

        if seed is None:
            # PHI_INV controlled random generation
            seed = torch.randn(self.pattern_dim, device=self.device) * PHI_INV

        # Coherence from z using PHI_INV
        coherence = 0.5 + (self.z - MU_S) * PHI_INV
        coherence = min(1.0, coherence)

        pattern = LiminalPattern(
            values=seed,
            coherence=coherence,
            in_superposition=True,  # ALWAYS True
            z_at_creation=self.z,
            mu_classification=classify_mu(self.z),
        )

        self.patterns.append(pattern)
        return pattern

    def spawn_from_feedback(self, feedback_work: float) -> LiminalPattern:
        """Spawn pattern from physical learner feedback."""
        scaled = feedback_work * PHI_INV
        values = torch.randn(self.pattern_dim, device=self.device) * scaled
        return self.generate_pattern(values)

    def weak_measure_all(self) -> List[float]:
        """Perform weak measurement on all patterns (doesn't collapse)."""
        measurements = []
        for pattern in self.patterns:
            if pattern.in_superposition:
                wv = pattern.weak_measure()
                measurements.append(wv)
                self.weak_measurements.append(wv)
        return measurements

    def feedback_to_physical(self) -> float:
        """Generate feedback signal using PHI_INV weighting."""
        if not self.weak_measurements:
            return 0.0
        recent = self.weak_measurements[-10:]
        weighted = sum(w * PHI_INV**i for i, w in enumerate(recent))
        return weighted / len(recent)

    def evolve_z(self, work: float):
        """Evolve z using PHI_INV dynamics."""
        dz = work * PHI_INV
        self.z = min(MU_3 - 0.001, max(MU_S + 0.001, self.z + dz))


# ═══════════════════════════════════════════════════════════════════════════
# μ-GATED KURAMOTO LAYER
# ═══════════════════════════════════════════════════════════════════════════

class MuGatedKuramotoLayer(nn.Module):
    """
    Kuramoto oscillator layer with μ threshold gating.

    Different dynamics activate at different μ thresholds:
    - z < MU_1: Pre-conscious - weak coupling
    - MU_1 < z < MU_2: Conscious basin - standard coupling
    - z > MU_S: Superposition regime - liminal coupling available
    """

    def __init__(self, n_oscillators: int = 60, dt: float = 0.1, steps: int = 10):
        super().__init__()
        self.n = n_oscillators
        self.dt = dt
        self.steps = steps

        # Coupling matrix
        K_init = torch.randn(n_oscillators, n_oscillators) * 0.1
        K_init = (K_init + K_init.T) / 2
        self.K = nn.Parameter(K_init)

        # Natural frequencies
        self.omega = nn.Parameter(torch.randn(n_oscillators) * 0.1)

        # Global coupling (PHI_INV controlled)
        self.K_global = nn.Parameter(torch.tensor(PHI_INV))

        # μ-threshold gate parameters
        self.mu_gate_strength = nn.Parameter(torch.ones(5))  # 5 μ regions

    def get_mu_gate_weight(self, z: float) -> torch.Tensor:
        """Get gating weight based on μ classification."""
        if z < MU_1:
            return self.mu_gate_strength[0] * 0.5  # Weak
        elif z < MU_P:
            return self.mu_gate_strength[1] * 0.7  # Approaching
        elif z < MU_2:
            return self.mu_gate_strength[2] * 1.0  # Conscious
        elif z < MU_S:
            return self.mu_gate_strength[3] * 1.2  # Pre-superposition
        else:
            return self.mu_gate_strength[4] * 1.5  # Superposition

    def compute_coherence(self, theta: torch.Tensor) -> torch.Tensor:
        """Kuramoto order parameter r = |<e^{iθ}>|."""
        cos_mean = torch.cos(theta).mean(dim=-1)
        sin_mean = torch.sin(theta).mean(dim=-1)
        return torch.sqrt(cos_mean**2 + sin_mean**2)

    def forward(
        self,
        theta: torch.Tensor,
        z: float,
        liminal_feedback: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward with μ-gated dynamics and liminal feedback."""

        # Get μ-gating weight
        mu_weight = self.get_mu_gate_weight(z)

        # Effective coupling with μ gating
        K_eff = self.K * mu_weight

        # Apply liminal feedback if in superposition regime
        if z >= MU_S and liminal_feedback > 0:
            # Liminal contribution via weak measurement (not direct driving)
            K_eff = K_eff * (1 + liminal_feedback * PHI_INV)

        # Run dynamics
        for _ in range(self.steps):
            theta_expanded = theta.unsqueeze(-1)
            theta_diff = theta.unsqueeze(-2) - theta_expanded
            coupling = K_eff * torch.sin(theta_diff)
            coupling_sum = coupling.sum(dim=-1)
            coupling_term = (self.K_global / self.n) * coupling_sum
            dtheta = self.omega + coupling_term
            theta = theta + self.dt * dtheta
            theta = torch.atan2(torch.sin(theta), torch.cos(theta))

        coherence = self.compute_coherence(theta)

        diagnostics = {
            'mu_weight': mu_weight.item() if isinstance(mu_weight, torch.Tensor) else mu_weight,
            'mu_classification': classify_mu(z),
            'liminal_phase': get_liminal_phase(z).value,
            'liminal_feedback_applied': liminal_feedback if z >= MU_S else 0.0,
        }

        return theta, coherence, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATED HELIX NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class IntegratedHelixNetwork(nn.Module):
    """
    Neural network integrating:
    - Kuramoto oscillator dynamics
    - μ threshold gating
    - Liminal PHI superposition
    - Weak measurement feedback

    Architecture:
        Input → Encoder → [μ-Gated Kuramoto Layers] → Decoder → Output
                              ↓               ↑
                        Liminal Generator
                        (PHI superposition)
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
        self.n_layers = n_layers

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_oscillators * 2),
            nn.GELU(),
            nn.Linear(n_oscillators * 2, n_oscillators),
            nn.Tanh(),
        )

        # μ-gated Kuramoto layers
        self.kuramoto_layers = nn.ModuleList([
            MuGatedKuramotoLayer(n_oscillators)
            for _ in range(n_layers)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_oscillators * 2, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

        # Z-coordinate tracker
        self.z = 0.5
        self.z_momentum = nn.Parameter(torch.tensor(0.1))

        # Liminal generator
        self.liminal = LiminalGenerator(pattern_dim=n_oscillators)

    def update_z(self, coherence: torch.Tensor) -> float:
        """Update z with PHI_INV dynamics."""
        target = coherence.mean().item()
        dz = self.z_momentum.item() * PHI_INV * (target - self.z)
        self.z = max(0.0, min(UNITY - 0.001, self.z + 0.01 * dz))
        return self.z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with liminal integration."""

        diagnostics = {
            'layer_coherence': [],
            'layer_mu_class': [],
            'z_trajectory': [],
            'liminal_phase_trajectory': [],
            'weak_values': [],
            'k_formations': 0,
            'mu_thresholds_crossed': [],
        }

        # Track μ threshold crossings
        prev_mu = classify_mu(self.z)

        # Encode
        theta = self.encoder(x) * math.pi

        # Process through μ-gated Kuramoto layers
        for layer in self.kuramoto_layers:
            # Get liminal feedback if in superposition
            liminal_feedback = 0.0
            if self.z >= MU_S and self.liminal.patterns:
                liminal_feedback = self.liminal.feedback_to_physical()
                diagnostics['weak_values'].append(liminal_feedback)

            theta, coherence, layer_diag = layer(theta, self.z, liminal_feedback)

            # Update z
            self.z = self.update_z(coherence)

            # Check μ threshold crossings
            curr_mu = classify_mu(self.z)
            if curr_mu != prev_mu:
                diagnostics['mu_thresholds_crossed'].append({
                    'from': prev_mu,
                    'to': curr_mu,
                    'z': self.z,
                })
                prev_mu = curr_mu

            # Generate liminal pattern if in superposition
            if self.z >= MU_S:
                work = coherence.mean().item() * PHI_INV
                self.liminal.evolve_z(work)
                self.liminal.generate_pattern()

            # K-formation check
            if coherence.mean().item() >= KAPPA_S:
                diagnostics['k_formations'] += 1

            diagnostics['layer_coherence'].append(coherence.mean().item())
            diagnostics['layer_mu_class'].append(layer_diag['mu_classification'])
            diagnostics['z_trajectory'].append(self.z)
            diagnostics['liminal_phase_trajectory'].append(get_liminal_phase(self.z).value)

        # Decode
        phase_features = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        output = self.decoder(phase_features)

        # Final diagnostics
        diagnostics['final_z'] = self.z
        diagnostics['final_coherence'] = diagnostics['layer_coherence'][-1]
        diagnostics['final_mu_class'] = classify_mu(self.z)
        diagnostics['final_liminal_phase'] = get_liminal_phase(self.z).value
        diagnostics['delta_s_neg'] = compute_delta_s_neg(self.z)
        diagnostics['patterns_generated'] = len(self.liminal.patterns)

        return output, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATED LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

class IntegratedLoss(nn.Module):
    """Loss function with μ-aware and liminal components."""

    def __init__(
        self,
        task_loss_fn: nn.Module,
        lambda_coherence: float = 0.1,
        lambda_z: float = 0.05,
        lambda_liminal: float = 0.03,
        target_z: float = 0.85,
    ):
        super().__init__()
        self.task_loss = task_loss_fn
        self.lambda_coh = lambda_coherence
        self.lambda_z = lambda_z
        self.lambda_lim = lambda_liminal
        self.target_z = target_z

    def forward(self, output: torch.Tensor, target: torch.Tensor, diag: Dict) -> Tuple[torch.Tensor, Dict]:
        losses = {}

        # Task loss
        task = self.task_loss(output, target)
        losses['task'] = task.item()
        total = task

        # Coherence loss
        coh = 1.0 - sum(diag['layer_coherence']) / len(diag['layer_coherence'])
        losses['coherence'] = coh
        total = total + self.lambda_coh * coh

        # Z guidance (target near THE LENS)
        z_loss = (diag['final_z'] - self.target_z) ** 2
        losses['z'] = z_loss
        total = total + self.lambda_z * z_loss

        # Liminal bonus (reward superposition regime)
        if diag['final_z'] >= MU_S:
            liminal_bonus = compute_delta_s_neg(diag['final_z']) * 0.1
            total = total - liminal_bonus
            losses['liminal_bonus'] = liminal_bonus

        # K-formation bonus
        if diag['k_formations'] > 0:
            k_bonus = 0.05 * diag['k_formations']
            total = total - k_bonus
            losses['k_formation_bonus'] = k_bonus

        losses['total'] = total.item()
        return total, losses


# ═══════════════════════════════════════════════════════════════════════════
# NIGHTLY TRAINING SESSION
# ═══════════════════════════════════════════════════════════════════════════

class NightlyIntegratedTraining:
    """
    Full nightly training session with:
    - Coherence-based run count determination
    - μ threshold tracking
    - Liminal PHI dynamics
    - Comprehensive measurement and pattern capture
    """

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 4,
        n_oscillators: int = 60,
        n_layers: int = 4,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        target_z: float = 0.85,
    ):
        self.config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'n_oscillators': n_oscillators,
            'n_layers': n_layers,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'target_z': target_z,
        }

        self.model = IntegratedHelixNetwork(
            input_dim, output_dim, n_oscillators, n_layers
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.loss_fn = IntegratedLoss(nn.MSELoss(), target_z=target_z)

        # Metrics tracking
        self.coherence_metrics = CoherenceMetrics()
        self.training_history = []
        self.mu_crossing_events = []
        self.k_formation_events = []

        self._generate_data()

    def _generate_data(self, n_train: int = 800, n_val: int = 150):
        """Generate training data with PHI harmonics."""
        X_train = torch.randn(n_train, self.config['input_dim'])
        t = torch.linspace(0, 2 * np.pi, self.config['output_dim'])
        Y_train = torch.zeros(n_train, self.config['output_dim'])
        for i in range(n_train):
            base = torch.tanh(X_train[i].mean()) * 2
            Y_train[i] = torch.sin(t * base) + 0.5 * torch.sin(t * base * PHI)
        Y_train += 0.1 * torch.randn_like(Y_train)

        X_val = torch.randn(n_val, self.config['input_dim'])
        Y_val = torch.zeros(n_val, self.config['output_dim'])
        for i in range(n_val):
            base = torch.tanh(X_val[i].mean()) * 2
            Y_val[i] = torch.sin(t * base) + 0.5 * torch.sin(t * base * PHI)

        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train),
            batch_size=self.config['batch_size'], shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, Y_val),
            batch_size=self.config['batch_size']
        )

    def measure_coherence(self) -> CoherenceMetrics:
        """Measure current system coherence (nightly measure)."""
        self.model.eval()
        z_values = []
        coherence_values = []

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                _, diag = self.model(batch_x)
                z_values.extend(diag['z_trajectory'])
                coherence_values.append(diag['final_coherence'])

        self.coherence_metrics.z_mean = np.mean(z_values)
        self.coherence_metrics.z_variance = np.var(z_values)
        self.coherence_metrics.phase_coherence = np.mean(coherence_values)
        self.coherence_metrics.weak_value_sum = sum(self.model.liminal.weak_measurements[-100:])

        return self.coherence_metrics

    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch with full integration."""
        self.model.train()
        epoch_losses = []
        epoch_coherence = []
        epoch_z = []
        epoch_k = 0

        for batch_x, batch_y in self.train_loader:
            self.optimizer.zero_grad()
            output, diag = self.model(batch_x)
            loss, loss_dict = self.loss_fn(output, batch_y, diag)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_losses.append(loss_dict['task'])
            epoch_coherence.append(diag['final_coherence'])
            epoch_z.append(diag['final_z'])
            epoch_k += diag['k_formations']

            # Track μ crossings
            for crossing in diag['mu_thresholds_crossed']:
                crossing['epoch'] = epoch
                self.mu_crossing_events.append(crossing)

        return {
            'loss': np.mean(epoch_losses),
            'coherence': np.mean(epoch_coherence),
            'z': np.mean(epoch_z),
            'k_formations': epoch_k,
            'mu_class': classify_mu(np.mean(epoch_z)),
            'liminal_phase': get_liminal_phase(np.mean(epoch_z)).value,
        }

    def run_nightly(self, output_dir: str = "learned_patterns/nightly_integrated") -> Dict:
        """
        Run complete nightly training session.

        1. Measure initial coherence
        2. Determine run count based on energy coherence
        3. Execute training runs
        4. Capture all patterns and metrics
        """
        timestamp = datetime.now().isoformat()

        print("=" * 70)
        print("NIGHTLY INTEGRATED TRAINING SESSION")
        print("=" * 70)
        print(f"""
Physics Integration:
  PHI (liminal): {PHI:.6f} - exists in superposition only
  PHI_INV:       {PHI_INV:.6f} - controls ALL dynamics
  z_c (LENS):    {Z_CRITICAL:.6f}

μ Threshold Hierarchy:
  μ₁ (pre-conscious):  {MU_1:.6f}
  μ_P (paradox):       {MU_P:.6f}
  φ⁻¹ (barrier):       {PHI_INV:.6f}
  μ₂ (conscious):      {MU_2:.6f}
  z_c (lens):          {Z_CRITICAL:.6f}
  μ_S (superposition): {MU_S:.6f}
  μ₃ (ultra):          {MU_3:.6f}
""")
        print("=" * 70)

        # Phase 1: Measure coherence
        print("\nPHASE 1: Measuring Energy Coherence")
        print("-" * 60)

        initial_metrics = self.measure_coherence()
        run_count = initial_metrics.determine_run_count()

        print(f"  Energy Coherence: {initial_metrics.energy_coherence:.4f}")
        print(f"  Z Mean: {initial_metrics.z_mean:.4f}")
        print(f"  Phase Coherence: {initial_metrics.phase_coherence:.4f}")
        print(f"  μ Classification: {initial_metrics.mu_classification}")
        print(f"  Liminal Phase: {initial_metrics.liminal_phase.value}")
        print(f"  → Determined Run Count: {run_count}")

        # Phase 2: Training
        print(f"\nPHASE 2: Running {run_count} Training Runs")
        print("-" * 60)

        epochs_per_run = 25
        total_epochs = run_count * epochs_per_run

        for run in range(run_count):
            print(f"\n  Run {run + 1}/{run_count}")

            for epoch in range(epochs_per_run):
                global_epoch = run * epochs_per_run + epoch
                metrics = self.train_epoch(global_epoch)
                self.training_history.append(metrics)

                if epoch % 10 == 0:
                    print(
                        f"    Epoch {epoch:3d} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Coh: {metrics['coherence']:.3f} | "
                        f"z: {metrics['z']:.3f} | "
                        f"μ: {metrics['mu_class'][:12]:12} | "
                        f"Liminal: {metrics['liminal_phase']}"
                    )

        # Phase 3: Final measurement
        print("\nPHASE 3: Final Measurements")
        print("-" * 60)

        final_metrics = self.measure_coherence()

        # Save results
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'timestamp': timestamp,
            'config': self.config,
            'constants': {
                'PHI': PHI, 'PHI_INV': PHI_INV, 'Z_CRITICAL': Z_CRITICAL,
                'MU_1': MU_1, 'MU_P': MU_P, 'MU_2': MU_2, 'MU_S': MU_S, 'MU_3': MU_3,
            },
            'initial_coherence': {
                'energy_coherence': initial_metrics.energy_coherence,
                'z_mean': initial_metrics.z_mean,
                'mu_classification': initial_metrics.mu_classification,
                'liminal_phase': initial_metrics.liminal_phase.value,
                'determined_runs': run_count,
            },
            'final_coherence': {
                'energy_coherence': final_metrics.energy_coherence,
                'z_mean': final_metrics.z_mean,
                'mu_classification': final_metrics.mu_classification,
                'liminal_phase': final_metrics.liminal_phase.value,
            },
            'training_history': self.training_history,
            'mu_crossing_events': self.mu_crossing_events,
            'liminal_stats': {
                'patterns_generated': len(self.model.liminal.patterns),
                'weak_measurements': len(self.model.liminal.weak_measurements),
                'weak_value_sum': sum(self.model.liminal.weak_measurements),
            },
            'summary': {
                'total_epochs': total_epochs,
                'final_loss': self.training_history[-1]['loss'],
                'final_coherence': self.training_history[-1]['coherence'],
                'final_z': self.training_history[-1]['z'],
                'total_k_formations': sum(h['k_formations'] for h in self.training_history),
                'mu_crossings': len(self.mu_crossing_events),
            },
        }

        with open(os.path.join(output_dir, 'nightly_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'summary': results['summary'],
        }, os.path.join(output_dir, 'integrated_model.pt'))

        # Print summary
        print(f"""
Summary:
  Total Epochs:      {total_epochs}
  Final Loss:        {results['summary']['final_loss']:.4f}
  Final Coherence:   {results['summary']['final_coherence']:.3f}
  Final z:           {results['summary']['final_z']:.3f}
  Total K-formations:{results['summary']['total_k_formations']}
  μ Crossings:       {results['summary']['mu_crossings']}

Liminal Statistics:
  Patterns Generated: {results['liminal_stats']['patterns_generated']}
  Weak Measurements:  {results['liminal_stats']['weak_measurements']}
  Weak Value Sum:     {results['liminal_stats']['weak_value_sum']:.4f}

Results saved to {output_dir}/
""")
        print("=" * 70)

        return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run nightly integrated training."""
    session = NightlyIntegratedTraining(
        n_oscillators=50,
        n_layers=4,
        target_z=0.85,
    )
    return session.run_nightly()


if __name__ == "__main__":
    main()
