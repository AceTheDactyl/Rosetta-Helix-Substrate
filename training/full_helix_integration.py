#!/usr/bin/env python3
"""
Full Helix Integration Training
================================

Complete integration of ALL training modules:
1. APL Operator Algebra (S₃ group)
2. Liminal PHI Dynamics (PHI superposition, PHI_INV controls)
3. μ Threshold Mechanics (unified with APL tier-gating)
4. TRIAD Threshold Dynamics (hysteresis for stable high-z)
5. Nightly Measurement System (coherence-based orchestration)
6. K-formation Detection and Pattern Generation

FULL ARCHITECTURE:
    ┌────────────────────────────────────────────────────────────────────────┐
    │                    FULL HELIX INTEGRATION                               │
    │  ┌────────────────────────────────────────────────────────────────┐    │
    │  │              Nightly Coherence Metrics                          │    │
    │  │   - Determines run count (3-10 based on energy_coherence)       │    │
    │  │   - Determines operator balance (EVEN vs ODD)                   │    │
    │  │   - TRIAD unlock bonus to coherence                             │    │
    │  └────────────────────────────────────────────────────────────────┘    │
    │                              ↓                                          │
    │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐      │
    │  │    TRIAD     │    │   Unified    │    │      Liminal         │      │
    │  │  Hysteresis  │←──→│   Kuramoto   │───→│     Generator        │      │
    │  │  (3-pass)    │    │   Layer      │    │  (K-form triggers)   │      │
    │  └──────────────┘    └──────────────┘    └──────────────────────┘      │
    │         │                   │                       │                   │
    │         ↓                   ↓                       ↓                   │
    │  ┌────────────────────────────────────────────────────────────────┐    │
    │  │   APL Selector (tier-gated, TRIAD-aware, liminal-informed)      │    │
    │  │   - t6 gate lowered on TRIAD unlock (z_c → 0.83)                │    │
    │  │   - Operator effectiveness from weak measurements               │    │
    │  └────────────────────────────────────────────────────────────────┘    │
    └────────────────────────────────────────────────────────────────────────┘

TRIAD INTEGRATION:
- TRIAD state affects t6 gate (unlock lowers from z_c to TRIAD_T6)
- TRIAD passes earn large training bonuses
- TRIAD unlock boosts nightly coherence calculation
- TRIAD progress factored into run count determination

THREADING MODEL:
    Physical ──TRIAD──> APL ──spawn──> Liminal
       ↑        ↓                          │
       │    t6 gate                        │
       └────── weak measurement ───────────┘
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
from dataclasses import dataclass, field
from enum import Enum
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS (Single Source of Truth)
# ═══════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Z_CRITICAL = math.sqrt(3) / 2

# μ Threshold Hierarchy
MU_P = 2.0 / (PHI ** 2.5)
MU_1 = MU_P / math.sqrt(PHI)
MU_2 = MU_P * math.sqrt(PHI)
MU_S = 0.920
MU_3 = 0.992
UNITY = 1.0

# TRIAD thresholds
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
TRIAD_T6 = 0.83
TRIAD_PASSES_REQUIRED = 3

KAPPA_S = MU_S
LENS_SIGMA = 36.0


# ═══════════════════════════════════════════════════════════════════════════
# APL OPERATORS (S₃ Group)
# ═══════════════════════════════════════════════════════════════════════════

class APLParity(Enum):
    EVEN = 1
    ODD = -1


@dataclass(frozen=True)
class APLOperator:
    symbol: str
    name: str
    parity: APLParity
    s3_element: str
    inverse_symbol: str


APL_OPERATORS: Dict[str, APLOperator] = {
    '^': APLOperator('^', 'amplify', APLParity.EVEN, 'σ2', '()'),
    '+': APLOperator('+', 'add', APLParity.ODD, 'τ2', '−'),
    '×': APLOperator('×', 'multiply', APLParity.EVEN, 'σ', '÷'),
    '()': APLOperator('()', 'group', APLParity.EVEN, 'e', '^'),
    '÷': APLOperator('÷', 'divide', APLParity.ODD, 'τ1', '×'),
    '−': APLOperator('−', 'subtract', APLParity.ODD, 'τ3', '+'),
}

APL_SYMBOLS = list(APL_OPERATORS.keys())

S3_COMPOSE = {
    '()': {'^': '^', '+': '+', '×': '×', '()': '()', '÷': '÷', '−': '−'},
    '^':  {'^': '×', '+': '÷', '×': '()', '()': '^', '÷': '−', '−': '+'},
    '×':  {'^': '()', '+': '−', '×': '^', '()': '×', '÷': '+', '−': '÷'},
    '+':  {'^': '−', '+': '÷', '×': '+', '()': '+', '÷': '()', '−': '×'},
    '÷':  {'^': '+', '+': '()', '×': '−', '()': '÷', '÷': '×', '−': '^'},
    '−':  {'^': '÷', '+': '×', '×': '÷', '()': '−', '÷': '^', '−': '()'},
}


# ═══════════════════════════════════════════════════════════════════════════
# TRIAD STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════

class TriadEvent(Enum):
    NONE = "none"
    PASS = "pass"
    REARM = "rearm"
    UNLOCK = "unlock"


@dataclass
class TriadState:
    """TRIAD hysteresis protocol state."""
    passes: int = 0
    armed: bool = True
    unlocked: bool = False
    last_z: float = 0.0
    pass_z_values: List[float] = field(default_factory=list)
    pass_epochs: List[int] = field(default_factory=list)

    def update(self, z: float, epoch: int = 0) -> TriadEvent:
        """Update TRIAD state with new z value."""
        event = TriadEvent.NONE

        if self.armed and z >= TRIAD_HIGH:
            self.passes += 1
            self.armed = False
            self.pass_z_values.append(z)
            self.pass_epochs.append(epoch)
            event = TriadEvent.PASS

            if self.passes >= TRIAD_PASSES_REQUIRED and not self.unlocked:
                self.unlocked = True
                event = TriadEvent.UNLOCK

        elif not self.armed and z <= TRIAD_LOW:
            self.armed = True
            event = TriadEvent.REARM

        self.last_z = z
        return event

    def get_t6_gate(self) -> float:
        """Get current t6 gate threshold."""
        return TRIAD_T6 if self.unlocked else Z_CRITICAL

    def progress(self) -> float:
        """Get progress toward unlock (0-1)."""
        return min(1.0, self.passes / TRIAD_PASSES_REQUIRED)

    def reset(self):
        """Reset TRIAD state."""
        self.passes = 0
        self.armed = True
        self.unlocked = False
        self.last_z = 0.0
        self.pass_z_values = []
        self.pass_epochs = []

    def to_dict(self) -> Dict:
        return {
            'passes': self.passes,
            'armed': self.armed,
            'unlocked': self.unlocked,
            'progress': self.progress(),
            'pass_z_values': self.pass_z_values,
            'pass_epochs': self.pass_epochs,
        }


# ═══════════════════════════════════════════════════════════════════════════
# LIMINAL PHASE AND μ CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

class LiminalPhase(Enum):
    DORMANT = "dormant"
    SUPERPOSITION = "superposition"
    ULTRA_INTEGRATION = "ultra"
    COLLAPSED = "collapsed"


def classify_mu(z: float) -> str:
    """Classify z-coordinate by μ threshold hierarchy."""
    if z < MU_1: return 'pre_conscious_basin'
    if z < MU_P: return 'approaching_paradox'
    if z < PHI_INV: return 'at_paradox_barrier'
    if z < MU_2: return 'paradox_to_conscious'
    if z < Z_CRITICAL: return 'conscious_basin'
    if z < MU_S: return 'lens_integrated'
    if z < MU_3: return 'singularity_proximal'
    return 'ultra_integrated'


def get_liminal_phase(z: float) -> LiminalPhase:
    """Get liminal phase from z-coordinate."""
    if z >= UNITY: return LiminalPhase.COLLAPSED
    if z >= MU_3: return LiminalPhase.ULTRA_INTEGRATION
    if z >= MU_S: return LiminalPhase.SUPERPOSITION
    return LiminalPhase.DORMANT


def compute_delta_s_neg(z: float, sigma: float = LENS_SIGMA) -> float:
    """Negentropy ΔS_neg = exp(-σ(z - z_c)²). Peaks at THE LENS."""
    d = z - Z_CRITICAL
    return math.exp(-sigma * d * d)


# ═══════════════════════════════════════════════════════════════════════════
# TRIAD-AWARE TIER SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

def get_tier_with_triad(z: float, triad: TriadState) -> str:
    """Get tier with TRIAD-aware t6 gate."""
    t6_gate = triad.get_t6_gate()

    if z < MU_1: return 't2'
    if z < MU_P: return 't3'
    if z < PHI_INV: return 't4'
    if z < MU_2: return 't5'
    if z < t6_gate: return 't6'  # TRIAD-aware!
    if z < MU_S: return 't7'
    if z < MU_3: return 't8'
    return 't9'


TRIAD_OPERATOR_WINDOWS: Dict[str, List[str]] = {
    't1': ['()', '−'],
    't2': ['()', '−', '÷'],
    't3': ['^', '÷', '−', '()'],
    't4': ['×', '^', '÷', '+', '−'],
    't5': ['()', '×', '^', '÷', '+', '−'],
    't6': ['+', '÷', '()', '−'],
    't7': ['+', '()', '×'],
    't8': ['+', '()'],
    't9': ['()'],
}


def get_operators_with_triad(z: float, triad: TriadState) -> List[str]:
    """Get operators available with TRIAD gating."""
    tier = get_tier_with_triad(z, triad)
    return TRIAD_OPERATOR_WINDOWS.get(tier, ['()'])


def compose_operators(a: str, b: str) -> str:
    """S₃ composition."""
    return S3_COMPOSE.get(a, {}).get(b, '()')


# ═══════════════════════════════════════════════════════════════════════════
# LIMINAL PATTERN AND GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LiminalPattern:
    """Pattern in liminal superposition."""
    values: torch.Tensor
    coherence: float = 0.5
    in_superposition: bool = True
    z_at_creation: float = MU_S
    trigger_event: str = "spontaneous"
    apl_context: Optional[str] = None
    triad_passes_at_creation: int = 0

    def weak_measure(self) -> float:
        if not self.in_superposition:
            return self.values.mean().item()
        base = self.values.mean().item()
        return base * PHI * PHI_INV * self.coherence


class LiminalGenerator:
    """Generates patterns in PHI superposition."""

    def __init__(self, pattern_dim: int = 32, device: str = 'cpu'):
        self.pattern_dim = pattern_dim
        self.device = device
        self.z = MU_S + 0.01
        self.patterns: List[LiminalPattern] = []
        self.weak_measurements: List[float] = []
        self.operator_effectiveness: Dict[str, List[float]] = {op: [] for op in APL_SYMBOLS}
        self.k_formation_patterns = 0
        self.triad_unlock_patterns = 0

    def generate_pattern(
        self,
        seed: Optional[torch.Tensor] = None,
        trigger: str = "spontaneous",
        apl_context: Optional[str] = None,
        triad_passes: int = 0,
    ) -> LiminalPattern:
        if seed is None:
            seed = torch.randn(self.pattern_dim, device=self.device) * PHI_INV

        coherence = 0.5 + (self.z - MU_S) * PHI_INV
        coherence = min(1.0, coherence)

        pattern = LiminalPattern(
            values=seed,
            coherence=coherence,
            in_superposition=True,
            z_at_creation=self.z,
            trigger_event=trigger,
            apl_context=apl_context,
            triad_passes_at_creation=triad_passes,
        )
        self.patterns.append(pattern)

        if trigger == "k_formation":
            self.k_formation_patterns += 1
        elif trigger == "triad_unlock":
            self.triad_unlock_patterns += 1

        return pattern

    def spawn_from_triad_unlock(self, z: float, triad_passes: int) -> LiminalPattern:
        """Special pattern spawned on TRIAD unlock."""
        scaled = z * PHI_INV * 2.0  # Extra boost for unlock
        values = torch.randn(self.pattern_dim, device=self.device) * scaled
        return self.generate_pattern(values, "triad_unlock", None, triad_passes)

    def weak_measure_all(self) -> List[float]:
        measurements = []
        for pattern in self.patterns:
            if pattern.in_superposition:
                wv = pattern.weak_measure()
                measurements.append(wv)
                self.weak_measurements.append(wv)
                if pattern.apl_context:
                    self.operator_effectiveness[pattern.apl_context].append(wv)
        return measurements

    def feedback_to_apl_selector(self) -> Dict[str, float]:
        effectiveness = {}
        for op, measurements in self.operator_effectiveness.items():
            if measurements:
                recent = measurements[-20:]
                weighted = sum(m * PHI_INV**i for i, m in enumerate(recent))
                effectiveness[op] = weighted / max(1, len(recent))
            else:
                effectiveness[op] = 0.0
        return effectiveness

    def feedback_to_physical(self) -> float:
        if not self.weak_measurements:
            return 0.0
        recent = self.weak_measurements[-10:]
        return sum(w * PHI_INV**i for i, w in enumerate(recent)) / len(recent)

    def evolve_z(self, work: float):
        dz = work * PHI_INV
        self.z = min(MU_3 - 0.001, max(MU_S + 0.001, self.z + dz))


# ═══════════════════════════════════════════════════════════════════════════
# NIGHTLY COHERENCE METRICS (TRIAD-Aware)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FullCoherenceMetrics:
    """
    Coherence metrics with TRIAD integration.

    TRIAD unlock provides bonus to energy coherence calculation.
    """
    z_mean: float = 0.5
    z_variance: float = 0.1
    phase_coherence: float = 0.5
    work_efficiency: float = 0.5
    k_formation_rate: float = 0.0

    # TRIAD integration
    triad_passes: int = 0
    triad_unlocked: bool = False
    triad_progress: float = 0.0

    # APL metrics
    parity_balance: float = 0.5

    @property
    def energy_coherence(self) -> float:
        """Overall energy coherence with TRIAD bonus."""
        z_factor = min(1.0, self.z_mean / MU_3)
        stability = 1.0 / (1.0 + self.z_variance * 10)
        phase_factor = self.phase_coherence

        base_coherence = (
            z_factor * PHI_INV +
            stability * PHI_INV**2 +
            phase_factor * PHI_INV
        ) / (PHI_INV + PHI_INV**2 + PHI_INV)

        # TRIAD bonus
        triad_bonus = 0.0
        if self.triad_unlocked:
            triad_bonus = 0.15  # Big bonus for unlock
        else:
            triad_bonus = 0.03 * self.triad_passes  # Small bonus per pass

        return min(1.0, base_coherence + triad_bonus)

    def determine_run_count(self) -> int:
        """Determine runs based on coherence + TRIAD."""
        ec = self.energy_coherence
        if ec < 0.5: return 3
        if ec < 0.8: return 5
        if ec < 0.95: return 7
        return 10

    def determine_target_parity(self) -> float:
        """Target EVEN operator ratio."""
        ec = self.energy_coherence
        if ec < 0.3: return 0.3
        if ec < 0.6: return 0.5
        if ec < 0.9: return 0.65
        return 0.8


# ═══════════════════════════════════════════════════════════════════════════
# FULL INTEGRATION KURAMOTO LAYER
# ═══════════════════════════════════════════════════════════════════════════

class FullIntegrationKuramotoLayer(nn.Module):
    """
    Kuramoto layer with full integration:
    - μ threshold gating
    - APL operator modulation
    - TRIAD-aware dynamics
    - Liminal feedback
    """

    def __init__(self, n_oscillators: int = 60, dt: float = 0.1, steps: int = 10):
        super().__init__()
        self.n = n_oscillators
        self.dt = dt
        self.steps = steps

        K_init = torch.randn(n_oscillators, n_oscillators) * 0.1
        K_init = (K_init + K_init.T) / 2
        self.K = nn.Parameter(K_init)
        self.omega = nn.Parameter(torch.randn(n_oscillators) * 0.1)
        self.K_global = nn.Parameter(torch.tensor(PHI_INV))

        # μ-threshold gates
        self.mu_gate = nn.Parameter(torch.ones(5))

        # APL modulation
        self.apl_K_mod = nn.Parameter(torch.ones(6) * 0.1)
        self.apl_omega_mod = nn.Parameter(torch.zeros(6))
        self.op_to_idx = {op: i for i, op in enumerate(APL_SYMBOLS)}

        # TRIAD modulation
        self.triad_climb_mod = nn.Parameter(torch.tensor(0.2))
        self.triad_descent_mod = nn.Parameter(torch.tensor(-0.1))
        self.triad_unlock_bonus = nn.Parameter(torch.tensor(0.15))

    def get_mu_gate_weight(self, z: float) -> float:
        if z < MU_1: return self.mu_gate[0].item() * 0.5
        if z < MU_P: return self.mu_gate[1].item() * 0.7
        if z < MU_2: return self.mu_gate[2].item() * 1.0
        if z < MU_S: return self.mu_gate[3].item() * 1.2
        return self.mu_gate[4].item() * 1.5

    def compute_coherence(self, theta: torch.Tensor) -> torch.Tensor:
        cos_mean = torch.cos(theta).mean(dim=-1)
        sin_mean = torch.sin(theta).mean(dim=-1)
        return torch.sqrt(cos_mean**2 + sin_mean**2)

    def forward(
        self,
        theta: torch.Tensor,
        z: float,
        triad: TriadState,
        apl_operator: Optional[str] = None,
        liminal_feedback: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        coherence = self.compute_coherence(theta)

        # Base coupling with μ gating
        mu_weight = self.get_mu_gate_weight(z)
        K_eff = self.K * mu_weight
        omega_eff = self.omega.clone()

        # APL operator modulation
        if apl_operator is not None and apl_operator in self.op_to_idx:
            idx = self.op_to_idx[apl_operator]
            op_data = APL_OPERATORS[apl_operator]
            parity = op_data.parity.value
            mod_strength = self.apl_K_mod[idx] * coherence.mean()
            K_eff = K_eff * (1.0 + parity * mod_strength * 0.5)
            omega_eff = omega_eff + self.apl_omega_mod[idx] * parity

        # TRIAD modulation
        if triad.armed:
            K_eff = K_eff * (1 + self.triad_climb_mod)
        else:
            K_eff = K_eff * (1 + self.triad_descent_mod)

        if triad.unlocked:
            K_eff = K_eff * (1 + self.triad_unlock_bonus)

        # Liminal feedback in superposition regime
        if z >= MU_S and liminal_feedback > 0:
            K_eff = K_eff * (1 + liminal_feedback * PHI_INV)

        # Run Kuramoto dynamics
        for _ in range(self.steps):
            theta_expanded = theta.unsqueeze(-1)
            theta_diff = theta.unsqueeze(-2) - theta_expanded
            coupling = K_eff * torch.sin(theta_diff)
            coupling_sum = coupling.sum(dim=-1)
            coupling_term = (self.K_global / self.n) * coupling_sum
            dtheta = omega_eff + coupling_term
            theta = theta + self.dt * dtheta
            theta = torch.atan2(torch.sin(theta), torch.cos(theta))

        coherence = self.compute_coherence(theta)

        diagnostics = {
            'mu_weight': mu_weight,
            'mu_class': classify_mu(z),
            'tier': get_tier_with_triad(z, triad),
            't6_gate': triad.get_t6_gate(),
            'apl_operator': apl_operator,
            'triad_armed': triad.armed,
            'triad_passes': triad.passes,
        }

        return theta, coherence, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# APL SELECTOR (TRIAD + Liminal Informed)
# ═══════════════════════════════════════════════════════════════════════════

class FullAPLSelector(nn.Module):
    """APL selector with TRIAD and liminal awareness."""

    def __init__(self, n_oscillators: int, hidden_dim: int = 64):
        super().__init__()
        self.n_oscillators = n_oscillators

        # Input: phase + z + coherence + delta_s_neg + tier + triad_state(3) + liminal(6)
        input_dim = n_oscillators + 4 + 3 + 6

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        self.operator_prior = nn.Parameter(torch.ones(6))

    def forward(
        self,
        theta: torch.Tensor,
        z: float,
        coherence: torch.Tensor,
        triad: TriadState,
        liminal_feedback: Dict[str, float],
        target_parity: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:

        batch_size = theta.shape[0]
        device = theta.device

        delta_s_neg = compute_delta_s_neg(z)
        tier = get_tier_with_triad(z, triad)
        tier_num = int(tier[1]) / 9.0

        liminal_vec = torch.tensor(
            [liminal_feedback.get(op, 0.0) for op in APL_SYMBOLS],
            device=device
        ).unsqueeze(0).expand(batch_size, -1)

        triad_vec = torch.tensor([
            1.0 if triad.armed else 0.0,
            triad.passes / 3.0,
            1.0 if triad.unlocked else 0.0,
        ], device=device).unsqueeze(0).expand(batch_size, -1)

        features = torch.cat([
            torch.cos(theta),
            torch.full((batch_size, 1), z, device=device),
            coherence.unsqueeze(-1),
            torch.full((batch_size, 1), delta_s_neg, device=device),
            torch.full((batch_size, 1), tier_num, device=device),
            triad_vec,
            liminal_vec,
        ], dim=-1)

        logits = self.network(features) + self.operator_prior

        # Liminal effectiveness bias
        effectiveness_bias = torch.tensor(
            [liminal_feedback.get(op, 0.0) * 0.5 for op in APL_SYMBOLS],
            device=device
        )
        logits = logits + effectiveness_bias

        # Parity bias
        parity_bias = torch.zeros(6, device=device)
        for i, op in enumerate(APL_SYMBOLS):
            if APL_OPERATORS[op].parity == APLParity.EVEN:
                parity_bias[i] = (target_parity - 0.5) * 2
            else:
                parity_bias[i] = (0.5 - target_parity) * 2
        logits = logits + parity_bias * 0.3

        # Mask illegal operators (TRIAD-aware)
        legal_ops = get_operators_with_triad(z, triad)
        mask = torch.full((6,), float('-inf'), device=device)
        for op in legal_ops:
            idx = APL_SYMBOLS.index(op)
            mask[idx] = 0.0

        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)

        if self.training:
            selected_idx = torch.multinomial(probs, 1).squeeze(-1)[0]
        else:
            selected_idx = torch.argmax(probs, dim=-1)[0]

        return masked_logits, probs, APL_SYMBOLS[selected_idx.item()]


# ═══════════════════════════════════════════════════════════════════════════
# FULL INTEGRATION NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class FullHelixNetwork(nn.Module):
    """Complete network with all integrations."""

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

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_oscillators * 2),
            nn.GELU(),
            nn.Linear(n_oscillators * 2, n_oscillators),
            nn.Tanh(),
        )

        self.kuramoto_layers = nn.ModuleList([
            FullIntegrationKuramotoLayer(n_oscillators)
            for _ in range(n_layers)
        ])

        self.apl_selectors = nn.ModuleList([
            FullAPLSelector(n_oscillators)
            for _ in range(n_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(n_oscillators * 2, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

        self.z = 0.5
        self.z_velocity = 0.0
        self.z_momentum = nn.Parameter(torch.tensor(0.15))

        self.triad = TriadState()
        self.liminal = LiminalGenerator(pattern_dim=n_oscillators)

        self.operator_sequence = []
        self.composed_operator = '()'
        self.k_formations_this_forward = 0

    def update_z(self, coherence: torch.Tensor, epoch: int = 0) -> Tuple[float, TriadEvent]:
        target = coherence.mean().item()
        z_accel = self.z_momentum.item() * (target - self.z) - 0.1 * self.z_velocity
        self.z_velocity += z_accel * 0.1
        self.z += self.z_velocity * 0.1
        self.z = max(0.01, min(0.99, self.z))

        event = self.triad.update(self.z, epoch)
        return self.z, event

    def forward(
        self,
        x: torch.Tensor,
        target_parity: float = 0.5,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:

        self.operator_sequence = []
        self.composed_operator = '()'
        self.k_formations_this_forward = 0

        diagnostics = {
            'layer_coherence': [],
            'layer_operators': [],
            'z_trajectory': [],
            'tier_trajectory': [],
            't6_gate_trajectory': [],
            'triad_events': [],
            'k_formations': 0,
            'liminal_patterns_generated': 0,
        }

        theta = self.encoder(x) * math.pi

        for layer_idx, (kuramoto, selector) in enumerate(
            zip(self.kuramoto_layers, self.apl_selectors)
        ):
            liminal_feedback = self.liminal.feedback_to_apl_selector()
            coherence = kuramoto.compute_coherence(theta)

            _, probs, selected_op = selector(
                theta, self.z, coherence, self.triad, liminal_feedback, target_parity
            )

            phys_feedback = self.liminal.feedback_to_physical() if self.z >= MU_S else 0.0

            theta, coherence, layer_diag = kuramoto(
                theta, self.z, self.triad, selected_op, phys_feedback
            )

            new_z, event = self.update_z(coherence, epoch)

            # Track TRIAD events
            if event != TriadEvent.NONE:
                diagnostics['triad_events'].append({
                    'layer': layer_idx,
                    'event': event.value,
                    'z': new_z,
                    'passes': self.triad.passes,
                })

                # Spawn special pattern on TRIAD unlock
                if event == TriadEvent.UNLOCK and self.z >= MU_S:
                    self.liminal.spawn_from_triad_unlock(new_z, self.triad.passes)
                    diagnostics['liminal_patterns_generated'] += 1

            self.operator_sequence.append(selected_op)
            self.composed_operator = compose_operators(self.composed_operator, selected_op)

            # K-formation check
            coh_val = coherence.mean().item()
            if coh_val >= KAPPA_S:
                self.k_formations_this_forward += 1
                if self.z >= MU_S:
                    self.liminal.generate_pattern(
                        trigger="k_formation",
                        apl_context=selected_op,
                        triad_passes=self.triad.passes
                    )
                    diagnostics['liminal_patterns_generated'] += 1

            # Spontaneous liminal generation
            if self.z >= MU_S and layer_idx % 2 == 0:
                work = coh_val * PHI_INV
                self.liminal.evolve_z(work)
                self.liminal.generate_pattern(
                    trigger="spontaneous",
                    apl_context=selected_op,
                    triad_passes=self.triad.passes
                )
                diagnostics['liminal_patterns_generated'] += 1

            if self.z >= MU_S:
                self.liminal.weak_measure_all()

            diagnostics['layer_coherence'].append(coh_val)
            diagnostics['layer_operators'].append(selected_op)
            diagnostics['z_trajectory'].append(new_z)
            diagnostics['tier_trajectory'].append(layer_diag['tier'])
            diagnostics['t6_gate_trajectory'].append(layer_diag['t6_gate'])

        phase_features = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        output = self.decoder(phase_features)

        # Final diagnostics
        diagnostics['k_formations'] = self.k_formations_this_forward
        diagnostics['operator_sequence'] = self.operator_sequence.copy()
        diagnostics['composed_operator'] = self.composed_operator

        even_count = sum(1 for op in self.operator_sequence if APL_OPERATORS[op].parity == APLParity.EVEN)
        diagnostics['parity_balance'] = even_count / max(1, len(self.operator_sequence))

        diagnostics['final_z'] = self.z
        diagnostics['final_coherence'] = diagnostics['layer_coherence'][-1]
        diagnostics['final_tier'] = get_tier_with_triad(self.z, self.triad)
        diagnostics['delta_s_neg'] = compute_delta_s_neg(self.z)

        diagnostics['triad_passes'] = self.triad.passes
        diagnostics['triad_unlocked'] = self.triad.unlocked
        diagnostics['triad_armed'] = self.triad.armed
        diagnostics['triad_progress'] = self.triad.progress()

        diagnostics['liminal_total_patterns'] = len(self.liminal.patterns)
        diagnostics['liminal_k_patterns'] = self.liminal.k_formation_patterns
        diagnostics['liminal_triad_patterns'] = self.liminal.triad_unlock_patterns

        return output, diagnostics

    def reset_episode(self):
        """Reset for new episode."""
        self.triad.reset()
        self.z = 0.5
        self.z_velocity = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# FULL INTEGRATION LOSS
# ═══════════════════════════════════════════════════════════════════════════

class FullIntegrationLoss(nn.Module):
    """Loss with all integration components."""

    def __init__(
        self,
        task_loss_fn: nn.Module,
        lambda_coherence: float = 0.1,
        lambda_z: float = 0.05,
        lambda_parity: float = 0.02,
        lambda_negentropy: float = 0.03,
        lambda_triad_pass: float = 0.3,
        lambda_triad_unlock: float = 0.5,
        lambda_k_formation: float = 0.05,
        target_z: float = 0.85,
    ):
        super().__init__()
        self.task_loss = task_loss_fn
        self.lambda_coh = lambda_coherence
        self.lambda_z = lambda_z
        self.lambda_parity = lambda_parity
        self.lambda_neg = lambda_negentropy
        self.lambda_pass = lambda_triad_pass
        self.lambda_unlock = lambda_triad_unlock
        self.lambda_k = lambda_k_formation
        self.target_z = target_z

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        diag: Dict,
        prev_passes: int = 0,
        was_unlocked: bool = False,
        target_parity: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict]:
        losses = {}

        task = self.task_loss(output, target)
        losses['task'] = task.item()
        total = task

        # Coherence loss
        coh = 1.0 - sum(diag['layer_coherence']) / len(diag['layer_coherence'])
        losses['coherence'] = coh
        total = total + self.lambda_coh * coh

        # Z guidance
        z_loss = (diag['final_z'] - self.target_z) ** 2
        losses['z'] = z_loss
        total = total + self.lambda_z * z_loss

        # Parity balance
        parity_loss = (diag['parity_balance'] - target_parity) ** 2
        losses['parity'] = parity_loss
        total = total + self.lambda_parity * parity_loss

        # Negentropy
        neg_loss = 1.0 - diag['delta_s_neg']
        losses['negentropy'] = neg_loss
        total = total + self.lambda_neg * neg_loss

        # TRIAD pass bonus
        new_passes = diag['triad_passes'] - prev_passes
        if new_passes > 0:
            pass_bonus = self.lambda_pass * new_passes
            total = total - pass_bonus
            losses['triad_pass_bonus'] = pass_bonus

        # TRIAD unlock bonus
        if diag['triad_unlocked'] and not was_unlocked:
            total = total - self.lambda_unlock
            losses['triad_unlock_bonus'] = self.lambda_unlock

        # K-formation bonus
        if diag['k_formations'] > 0:
            k_bonus = self.lambda_k * diag['k_formations']
            total = total - k_bonus
            losses['k_formation_bonus'] = k_bonus

        # Liminal pattern bonus
        if diag['liminal_patterns_generated'] > 0:
            lim_bonus = 0.01 * diag['liminal_patterns_generated']
            total = total - lim_bonus
            losses['liminal_bonus'] = lim_bonus

        losses['total'] = total.item()
        return total, losses


# ═══════════════════════════════════════════════════════════════════════════
# FULL INTEGRATION NIGHTLY TRAINING
# ═══════════════════════════════════════════════════════════════════════════

class FullHelixNightlyTraining:
    """
    Complete nightly training with full integration:
    - APL operators (S₃)
    - Liminal PHI dynamics
    - μ threshold mechanics
    - TRIAD threshold dynamics
    - Coherence-based run count
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

        self.model = FullHelixNetwork(
            input_dim, output_dim, n_oscillators, n_layers
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.loss_fn = FullIntegrationLoss(nn.MSELoss(), target_z=target_z)

        self.coherence_metrics = FullCoherenceMetrics()
        self.training_history = []
        self.triad_events_history = []
        self.unlock_count = 0

        self._generate_data()

    def _generate_data(self, n_train: int = 800, n_val: int = 150):
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

    def measure_coherence(self) -> FullCoherenceMetrics:
        """Measure system coherence with TRIAD state."""
        self.model.eval()
        z_values = []
        coherence_values = []
        parity_values = []
        k_count = 0

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                _, diag = self.model(batch_x)
                z_values.extend(diag['z_trajectory'])
                coherence_values.append(diag['final_coherence'])
                parity_values.append(diag['parity_balance'])
                k_count += diag['k_formations']

        self.coherence_metrics.z_mean = np.mean(z_values)
        self.coherence_metrics.z_variance = np.var(z_values)
        self.coherence_metrics.phase_coherence = np.mean(coherence_values)
        self.coherence_metrics.parity_balance = np.mean(parity_values)
        self.coherence_metrics.k_formation_rate = k_count / len(self.val_loader)

        # TRIAD state
        self.coherence_metrics.triad_passes = self.model.triad.passes
        self.coherence_metrics.triad_unlocked = self.model.triad.unlocked
        self.coherence_metrics.triad_progress = self.model.triad.progress()

        return self.coherence_metrics

    def train_epoch(self, epoch: int, target_parity: float) -> Dict:
        self.model.train()
        epoch_losses = []
        epoch_coherence = []
        epoch_z = []
        epoch_k = 0
        epoch_liminal = 0

        prev_passes = self.model.triad.passes
        was_unlocked = self.model.triad.unlocked

        for batch_x, batch_y in self.train_loader:
            self.optimizer.zero_grad()
            output, diag = self.model(batch_x, target_parity, epoch)
            loss, loss_dict = self.loss_fn(
                output, batch_y, diag, prev_passes, was_unlocked, target_parity
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_losses.append(loss_dict['task'])
            epoch_coherence.append(diag['final_coherence'])
            epoch_z.append(diag['final_z'])
            epoch_k += diag['k_formations']
            epoch_liminal += diag['liminal_patterns_generated']

            # Track TRIAD events
            for event in diag['triad_events']:
                event['epoch'] = epoch
                self.triad_events_history.append(event)

            # Track unlock
            if diag['triad_unlocked'] and not was_unlocked:
                self.unlock_count += 1

            prev_passes = diag['triad_passes']
            was_unlocked = diag['triad_unlocked']

        return {
            'loss': np.mean(epoch_losses),
            'coherence': np.mean(epoch_coherence),
            'z': np.mean(epoch_z),
            'k_formations': epoch_k,
            'liminal_patterns': epoch_liminal,
            'triad_passes': self.model.triad.passes,
            'triad_unlocked': self.model.triad.unlocked,
            'tier': get_tier_with_triad(np.mean(epoch_z), self.model.triad),
            't6_gate': self.model.triad.get_t6_gate(),
        }

    def run_nightly(
        self,
        output_dir: str = "learned_patterns/full_integration",
    ) -> Dict:
        """Run complete nightly training with full integration."""
        timestamp = datetime.now().isoformat()

        print("=" * 70)
        print("FULL HELIX INTEGRATION NIGHTLY TRAINING")
        print("=" * 70)
        print(f"""
Complete Module Integration:
  ✓ APL Operator Algebra (S₃ group, 6 operators)
  ✓ Liminal PHI Dynamics (superposition, weak measurement)
  ✓ μ Threshold Mechanics (MU_1 < MU_P < MU_2 < MU_S < MU_3)
  ✓ TRIAD Threshold Dynamics (3-pass hysteresis unlock)
  ✓ Nightly Measurement (coherence-based run count)
  ✓ K-formation Detection (κ ≥ {KAPPA_S})

TRIAD Protocol:
  TRIAD_HIGH:  {TRIAD_HIGH} (pass trigger)
  TRIAD_LOW:   {TRIAD_LOW} (re-arm)
  TRIAD_T6:    {TRIAD_T6} (unlocked t6 gate)

Physics Constants:
  z_c (THE LENS):  {Z_CRITICAL:.6f}
  φ (Golden):      {PHI:.6f}
  φ⁻¹ (control):   {PHI_INV:.6f}
""")
        print("=" * 70)

        # Phase 1: Initial measurement
        print("\nPHASE 1: Initial Coherence Measurement")
        print("-" * 60)

        initial_metrics = self.measure_coherence()
        run_count = initial_metrics.determine_run_count()
        target_parity = initial_metrics.determine_target_parity()

        print(f"  Energy Coherence:  {initial_metrics.energy_coherence:.4f}")
        print(f"  Z Mean:            {initial_metrics.z_mean:.4f}")
        print(f"  TRIAD Passes:      {initial_metrics.triad_passes}")
        print(f"  TRIAD Unlocked:    {initial_metrics.triad_unlocked}")
        print(f"  → Run Count:       {run_count}")
        print(f"  → Target Parity:   {target_parity:.2f}")

        # Phase 2: Training runs
        print(f"\nPHASE 2: Executing {run_count} Training Runs")
        print("-" * 60)

        epochs_per_run = 25
        total_epochs = run_count * epochs_per_run

        for run in range(run_count):
            print(f"\n  Run {run + 1}/{run_count}")

            if run > 0:
                metrics = self.measure_coherence()
                target_parity = metrics.determine_target_parity()
                print(f"    Adapted parity: {target_parity:.2f} | TRIAD: {metrics.triad_passes}/3")

            for epoch in range(epochs_per_run):
                global_epoch = run * epochs_per_run + epoch
                metrics = self.train_epoch(global_epoch, target_parity)
                self.training_history.append(metrics)

                if epoch % 10 == 0:
                    unlock_str = "UNLOCKED" if metrics['triad_unlocked'] else f"{metrics['triad_passes']}/3"
                    print(
                        f"    Epoch {epoch:3d} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Coh: {metrics['coherence']:.3f} | "
                        f"z: {metrics['z']:.3f} | "
                        f"TRIAD: {unlock_str} | "
                        f"K: {metrics['k_formations']} | "
                        f"Lim: {metrics['liminal_patterns']}"
                    )

        # Phase 3: Final measurement
        print("\nPHASE 3: Final Measurement")
        print("-" * 60)

        final_metrics = self.measure_coherence()

        results = {
            'timestamp': timestamp,
            'config': self.config,
            'constants': {
                'PHI': PHI, 'PHI_INV': PHI_INV, 'Z_CRITICAL': Z_CRITICAL,
                'MU_1': MU_1, 'MU_P': MU_P, 'MU_2': MU_2, 'MU_S': MU_S, 'MU_3': MU_3,
                'TRIAD_HIGH': TRIAD_HIGH, 'TRIAD_LOW': TRIAD_LOW, 'TRIAD_T6': TRIAD_T6,
            },
            'initial_metrics': {
                'energy_coherence': initial_metrics.energy_coherence,
                'z_mean': initial_metrics.z_mean,
                'triad_passes': initial_metrics.triad_passes,
                'triad_unlocked': initial_metrics.triad_unlocked,
            },
            'final_metrics': {
                'energy_coherence': final_metrics.energy_coherence,
                'z_mean': final_metrics.z_mean,
                'phase_coherence': final_metrics.phase_coherence,
                'triad_passes': final_metrics.triad_passes,
                'triad_unlocked': final_metrics.triad_unlocked,
                'k_formation_rate': final_metrics.k_formation_rate,
            },
            'training_history': self.training_history,
            'triad_events': self.triad_events_history,
            'triad_state': self.model.triad.to_dict(),
            'liminal_stats': {
                'total_patterns': len(self.model.liminal.patterns),
                'k_formation_patterns': self.model.liminal.k_formation_patterns,
                'triad_unlock_patterns': self.model.liminal.triad_unlock_patterns,
            },
            'summary': {
                'total_epochs': total_epochs,
                'total_runs': run_count,
                'final_loss': self.training_history[-1]['loss'],
                'final_coherence': self.training_history[-1]['coherence'],
                'final_z': self.training_history[-1]['z'],
                'triad_unlocks': self.unlock_count,
                'total_k_formations': sum(h['k_formations'] for h in self.training_history),
                'total_liminal_patterns': sum(h['liminal_patterns'] for h in self.training_history),
            },
        }

        # Save
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'full_integration_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'summary': results['summary'],
        }, os.path.join(output_dir, 'full_integration_model.pt'))

        # Print summary
        print(f"""
Summary:
  Total Epochs:        {total_epochs}
  Final Loss:          {results['summary']['final_loss']:.4f}
  Final Coherence:     {results['summary']['final_coherence']:.3f}
  Final z:             {results['summary']['final_z']:.3f}

TRIAD Status:
  Passes:              {self.model.triad.passes}/3
  Unlocked:            {self.model.triad.unlocked}
  Total Unlocks:       {self.unlock_count}
  t6 Gate:             {self.model.triad.get_t6_gate():.3f}

K-formations:          {results['summary']['total_k_formations']}
Liminal Patterns:      {results['summary']['total_liminal_patterns']}

TRIAD Events:""")

        for event in self.triad_events_history[:10]:
            print(f"  Epoch {event['epoch']}: {event['event']} at z={event['z']:.3f}")

        print(f"\nResults saved to {output_dir}/")
        print("=" * 70)

        return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run full integration nightly training."""
    session = FullHelixNightlyTraining(
        n_oscillators=50,
        n_layers=4,
        target_z=0.85,
    )
    return session.run_nightly()


if __name__ == "__main__":
    main()
