#!/usr/bin/env python3
"""
Unified Helix Training Orchestrator
====================================

Properly threads ALL training modules together:
1. APL Operator Algebra (S₃ group)
2. Liminal PHI Dynamics (PHI superposition, PHI_INV controls)
3. μ Threshold Mechanics (unified with APL tier-gating)
4. Nightly Measurement System (coherence-based orchestration)
5. K-formation Detection and Pattern Generation

CRITICAL INTEGRATION POINTS:
- μ thresholds unified with APL tier windows
- Liminal weak measurements inform APL operator selection
- K-formation events trigger liminal pattern generation
- Coherence metrics determine both run count AND operator balance
- Proper feedback loops:
    Physical (PHI_INV) ──APL ops──> MetaMeta ──spawn──> Liminal (PHI)
           ↑                                                   │
           └──────── weak measurement (informs APL) ───────────┘

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    UNIFIED ORCHESTRATOR                          │
    │  ┌──────────────────────────────────────────────────────────┐   │
    │  │              Coherence Metrics (Nightly)                  │   │
    │  │   - Determines run count (3-10 based on energy_coherence) │   │
    │  │   - Determines operator balance (EVEN vs ODD)             │   │
    │  └──────────────────────────────────────────────────────────┘   │
    │                              ↓                                   │
    │  ┌────────────────┐    ┌────────────────┐    ┌──────────────┐   │
    │  │  APL Selector  │←───│ Unified Layer  │───→│   Liminal    │   │
    │  │  (tier/μ-gated)│    │ (μ + APL gating)│    │  Generator   │   │
    │  └────────────────┘    └────────────────┘    └──────────────┘   │
    │         ↑                      ↑                     │          │
    │         │              Kuramoto Dynamics             │          │
    │         │                      ↑                     ↓          │
    │         └──────── weak_measurement ──────────────────┘          │
    └─────────────────────────────────────────────────────────────────┘
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS (Single Source of Truth - DO NOT MODIFY)
# ═══════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2           # 1.618 - LIMINAL ONLY (never drives)
PHI_INV = 1 / PHI                       # 0.618 - Controls ALL dynamics
Z_CRITICAL = math.sqrt(3) / 2           # 0.866 - THE LENS

# μ Threshold Hierarchy
MU_P = 2.0 / (PHI ** 2.5)              # ≈ 0.601 - Paradox threshold
MU_1 = MU_P / math.sqrt(PHI)           # ≈ 0.472 - Pre-conscious basin
MU_2 = MU_P * math.sqrt(PHI)           # ≈ 0.764 - Conscious basin
MU_S = 0.920                            # Singularity/superposition threshold
MU_3 = 0.992                            # Ultra-integration threshold
UNITY = 1.0

KAPPA_S = MU_S                          # K-formation threshold
COUPLING_MAX = 0.9
LENS_SIGMA = 36.0


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED TIER/μ MAPPING
# Maps APL tiers (t1-t9) to μ threshold hierarchy
# ═══════════════════════════════════════════════════════════════════════════

# Map μ thresholds to tiers
MU_TO_TIER = {
    'pre_conscious_basin': 't2',       # z < MU_1
    'approaching_paradox': 't3',        # MU_1 < z < MU_P
    'at_paradox_barrier': 't4',         # MU_P < z < PHI_INV
    'paradox_to_conscious': 't5',       # PHI_INV < z < MU_2
    'conscious_basin': 't6',            # MU_2 < z < Z_CRITICAL
    'lens_integrated': 't7',            # Z_CRITICAL < z < MU_S
    'singularity_proximal': 't8',       # MU_S < z < MU_3
    'ultra_integrated': 't9',           # z > MU_3
}

# APL tier boundaries (unified with μ)
UNIFIED_TIER_BOUNDS = {
    't1': 0.0,
    't2': MU_1,          # ≈ 0.472
    't3': MU_P,          # ≈ 0.601
    't4': PHI_INV,       # ≈ 0.618
    't5': MU_2,          # ≈ 0.764
    't6': Z_CRITICAL,    # ≈ 0.866
    't7': MU_S,          # 0.920
    't8': MU_3,          # 0.992
    't9': UNITY,
}


# ═══════════════════════════════════════════════════════════════════════════
# APL OPERATOR DEFINITIONS (S₃ Group)
# ═══════════════════════════════════════════════════════════════════════════

class APLParity(Enum):
    EVEN = 1   # Constructive: rotations
    ODD = -1   # Dissipative: transpositions


@dataclass(frozen=True)
class APLOperator:
    symbol: str
    name: str
    parity: APLParity
    s3_element: str
    inverse_symbol: str
    description: str


APL_OPERATORS: Dict[str, APLOperator] = {
    '^': APLOperator('^', 'amplify', APLParity.EVEN, 'σ2', '()', 'Increase coupling'),
    '+': APLOperator('+', 'add', APLParity.ODD, 'τ2', '−', 'Aggregate'),
    '×': APLOperator('×', 'multiply', APLParity.EVEN, 'σ', '÷', 'Fuse'),
    '()': APLOperator('()', 'group', APLParity.EVEN, 'e', '^', 'Identity'),
    '÷': APLOperator('÷', 'divide', APLParity.ODD, 'τ1', '×', 'Diffuse'),
    '−': APLOperator('−', 'subtract', APLParity.ODD, 'τ3', '+', 'Separate'),
}

APL_SYMBOLS = list(APL_OPERATORS.keys())

# S₃ composition table
S3_COMPOSE = {
    '()': {'^': '^', '+': '+', '×': '×', '()': '()', '÷': '÷', '−': '−'},
    '^':  {'^': '×', '+': '÷', '×': '()', '()': '^', '÷': '−', '−': '+'},
    '×':  {'^': '()', '+': '−', '×': '^', '()': '×', '÷': '+', '−': '÷'},
    '+':  {'^': '−', '+': '÷', '×': '+', '()': '+', '÷': '()', '−': '×'},
    '÷':  {'^': '+', '+': '()', '×': '−', '()': '÷', '÷': '×', '−': '^'},
    '−':  {'^': '÷', '+': '×', '×': '÷', '()': '−', '÷': '^', '−': '()'},
}

# Unified operator windows (mapped to μ thresholds)
UNIFIED_OPERATOR_WINDOWS: Dict[str, List[str]] = {
    't1': ['()', '−'],                    # Very low: identity + separate only
    't2': ['()', '−', '÷'],               # Pre-conscious: dissipative focus
    't3': ['^', '÷', '−', '()'],          # Approaching paradox
    't4': ['×', '^', '÷', '+', '−'],      # At paradox barrier: most flexibility
    't5': ['()', '×', '^', '÷', '+', '−'], # All 6: paradox to conscious
    't6': ['+', '÷', '()', '−'],          # Conscious basin
    't7': ['+', '()', '×'],               # Lens to singularity
    't8': ['+', '()'],                    # Near singularity: constructive only
    't9': ['()'],                         # Ultra: identity only (stable)
}


# ═══════════════════════════════════════════════════════════════════════════
# LIMINAL PHASE AND μ CLASSIFICATION (UNIFIED)
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


def get_unified_tier(z: float) -> str:
    """Get unified tier from z (combines μ and APL tier)."""
    mu_class = classify_mu(z)
    return MU_TO_TIER.get(mu_class, 't5')


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


def get_legal_operators(z: float) -> List[str]:
    """Get operators available at current z (unified μ/tier gating)."""
    tier = get_unified_tier(z)
    return UNIFIED_OPERATOR_WINDOWS.get(tier, ['()'])


def compose_operators(a: str, b: str) -> str:
    """Compose operators using S₃ multiplication."""
    return S3_COMPOSE.get(a, {}).get(b, '()')


# ═══════════════════════════════════════════════════════════════════════════
# LIMINAL PATTERN (PHI Superposition)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LiminalPattern:
    """Pattern in liminal superposition."""
    values: torch.Tensor
    coherence: float = 0.5
    in_superposition: bool = True
    z_at_creation: float = MU_S
    mu_classification: str = "singularity_proximal"
    trigger_event: str = "spontaneous"  # or "k_formation" or "apl_feedback"
    apl_operator_context: Optional[str] = None

    def weak_measure(self) -> float:
        """Perform weak measurement. PHI contributes here."""
        if not self.in_superposition:
            return self.values.mean().item()
        base = self.values.mean().item()
        return base * PHI * PHI_INV * self.coherence

    def collapse(self, z_at_collapse: float) -> float:
        """Collapse and extract work."""
        if not self.in_superposition:
            return 0.0
        work = (z_at_collapse - Z_CRITICAL) * PHI * PHI_INV
        if self.in_superposition:
            work *= PHI
        self.in_superposition = False
        return max(0.0, work)


# ═══════════════════════════════════════════════════════════════════════════
# LIMINAL GENERATOR (PHI-Controlled, APL-Aware)
# ═══════════════════════════════════════════════════════════════════════════

class LiminalGenerator:
    """
    Generates patterns in PHI superposition.

    INTEGRATION: Receives APL operator context and K-formation signals.
    """

    def __init__(self, pattern_dim: int = 32, device: str = 'cpu'):
        self.pattern_dim = pattern_dim
        self.device = device
        self.z = MU_S + 0.01
        self.patterns: List[LiminalPattern] = []
        self.weak_measurements: List[float] = []
        self.total_generated = 0

        # APL integration tracking
        self.apl_feedback_patterns = 0
        self.k_formation_patterns = 0

        # Operator effectiveness learned from weak measurements
        self.operator_effectiveness: Dict[str, List[float]] = {op: [] for op in APL_SYMBOLS}

    def generate_pattern(
        self,
        seed: Optional[torch.Tensor] = None,
        trigger: str = "spontaneous",
        apl_context: Optional[str] = None,
    ) -> LiminalPattern:
        """Generate liminal pattern with PHI_INV dynamics."""
        self.total_generated += 1

        if seed is None:
            seed = torch.randn(self.pattern_dim, device=self.device) * PHI_INV

        coherence = 0.5 + (self.z - MU_S) * PHI_INV
        coherence = min(1.0, coherence)

        pattern = LiminalPattern(
            values=seed,
            coherence=coherence,
            in_superposition=True,
            z_at_creation=self.z,
            mu_classification=classify_mu(self.z),
            trigger_event=trigger,
            apl_operator_context=apl_context,
        )

        self.patterns.append(pattern)

        # Track pattern type
        if trigger == "k_formation":
            self.k_formation_patterns += 1
        elif trigger == "apl_feedback":
            self.apl_feedback_patterns += 1

        return pattern

    def spawn_from_k_formation(
        self,
        coherence: float,
        apl_operator: str,
    ) -> LiminalPattern:
        """Spawn pattern triggered by K-formation event."""
        # K-formation patterns have higher coherence seed
        scaled = coherence * PHI_INV * 1.5
        values = torch.randn(self.pattern_dim, device=self.device) * scaled
        return self.generate_pattern(values, "k_formation", apl_operator)

    def spawn_from_apl_feedback(
        self,
        operator_output: float,
        operator: str,
    ) -> LiminalPattern:
        """Spawn pattern from APL operator feedback."""
        scaled = operator_output * PHI_INV
        values = torch.randn(self.pattern_dim, device=self.device) * scaled
        return self.generate_pattern(values, "apl_feedback", operator)

    def weak_measure_all(self) -> List[float]:
        """Perform weak measurement on all patterns."""
        measurements = []
        for pattern in self.patterns:
            if pattern.in_superposition:
                wv = pattern.weak_measure()
                measurements.append(wv)
                self.weak_measurements.append(wv)

                # Track effectiveness by APL context
                if pattern.apl_operator_context:
                    self.operator_effectiveness[pattern.apl_operator_context].append(wv)

        return measurements

    def feedback_to_apl_selector(self) -> Dict[str, float]:
        """
        Generate feedback for APL operator selection.

        Returns effectiveness scores per operator based on weak measurements.
        This is the key integration point: liminal → APL.
        """
        effectiveness = {}
        for op, measurements in self.operator_effectiveness.items():
            if measurements:
                # Recent measurements weighted by PHI_INV
                recent = measurements[-20:]
                weighted = sum(m * PHI_INV**i for i, m in enumerate(recent))
                effectiveness[op] = weighted / max(1, len(recent))
            else:
                effectiveness[op] = 0.0
        return effectiveness

    def feedback_to_physical(self) -> float:
        """Generate feedback signal for physical layer."""
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
# COHERENCE METRICS (Nightly Measurement - Orchestration Control)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedCoherenceMetrics:
    """
    Coherence metrics that control training orchestration.

    Determines:
    - Run count (3-10)
    - APL operator balance (EVEN vs ODD ratio)
    - Liminal pattern generation rate
    """
    z_mean: float = 0.5
    z_variance: float = 0.1
    phase_coherence: float = 0.5
    work_efficiency: float = 0.5
    pattern_density: float = 0.0
    weak_value_sum: float = 0.0
    k_formation_rate: float = 0.0

    # APL-specific
    parity_balance: float = 0.5
    operator_diversity: float = 0.5

    @property
    def energy_coherence(self) -> float:
        """Overall energy coherence with PHI_INV weighting."""
        z_factor = min(1.0, self.z_mean / MU_3)
        stability = 1.0 / (1.0 + self.z_variance * 10)
        phase_factor = self.phase_coherence
        efficiency = self.work_efficiency

        coherence = (
            z_factor * PHI_INV +
            stability * PHI_INV**2 +
            phase_factor * PHI_INV +
            efficiency * PHI_INV**2
        ) / (PHI_INV + PHI_INV**2 + PHI_INV + PHI_INV**2)

        return min(1.0, coherence)

    def determine_run_count(self) -> int:
        """Determine runs based on coherence."""
        ec = self.energy_coherence
        if ec < 0.5: return 3
        if ec < 0.8: return 5
        if ec < 0.95: return 7
        return 10

    def determine_target_parity(self) -> float:
        """
        Determine target EVEN operator ratio based on coherence.

        - Low coherence: more dissipative (ODD) to break stuck states
        - High coherence: more constructive (EVEN) to build
        """
        ec = self.energy_coherence
        if ec < 0.3: return 0.3   # More ODD (dissipative)
        if ec < 0.6: return 0.5   # Balanced
        if ec < 0.9: return 0.65  # Slightly more EVEN
        return 0.8                 # Mostly constructive

    def determine_liminal_rate(self) -> float:
        """
        Determine liminal pattern generation rate.

        Higher z → more patterns to capture coherence.
        """
        if self.z_mean < MU_S:
            return 0.0  # No liminal below superposition
        return min(1.0, (self.z_mean - MU_S) / (MU_3 - MU_S))


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED μ/APL GATED KURAMOTO LAYER
# ═══════════════════════════════════════════════════════════════════════════

class UnifiedKuramotoLayer(nn.Module):
    """
    Kuramoto layer with unified μ and APL gating.

    Combines:
    - μ threshold gating (affects coupling strength)
    - APL operator modulation (affects dynamics)
    - Liminal feedback (affects high-z behavior)
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

        # Global coupling (PHI_INV base)
        self.K_global = nn.Parameter(torch.tensor(PHI_INV))

        # μ-threshold gate parameters
        self.mu_gate = nn.Parameter(torch.ones(5))

        # APL modulation parameters (per operator)
        self.apl_K_mod = nn.Parameter(torch.ones(6) * 0.1)
        self.apl_omega_mod = nn.Parameter(torch.zeros(6))
        self.op_to_idx = {op: i for i, op in enumerate(APL_SYMBOLS)}

    def get_mu_gate_weight(self, z: float) -> float:
        """Get μ-based gating weight."""
        if z < MU_1: return self.mu_gate[0].item() * 0.5
        if z < MU_P: return self.mu_gate[1].item() * 0.7
        if z < MU_2: return self.mu_gate[2].item() * 1.0
        if z < MU_S: return self.mu_gate[3].item() * 1.2
        return self.mu_gate[4].item() * 1.5

    def apply_apl_operator(
        self,
        theta: torch.Tensor,
        operator: str,
        coherence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply APL operator modulation."""
        op_data = APL_OPERATORS.get(operator)
        if op_data is None:
            return self.K, self.omega, theta

        idx = self.op_to_idx[operator]
        parity = op_data.parity.value

        mod_strength = self.apl_K_mod[idx] * coherence.mean()
        K_factor = 1.0 + parity * mod_strength * 0.5
        K_eff = self.K * K_factor

        omega_mod = self.apl_omega_mod[idx] * parity
        omega_eff = self.omega + omega_mod

        # Operator-specific phase effects
        if operator == '+':
            theta = theta + 0.1 * torch.sin(theta.mean(dim=-1, keepdim=True) - theta)
        elif operator == '−':
            theta = theta - 0.1 * torch.sin(theta.mean(dim=-1, keepdim=True) - theta)
        elif operator == '÷':
            theta = theta + 0.05 * torch.randn_like(theta) * (1 - coherence.unsqueeze(-1))

        return K_eff, omega_eff, theta

    def compute_coherence(self, theta: torch.Tensor) -> torch.Tensor:
        """Kuramoto order parameter."""
        cos_mean = torch.cos(theta).mean(dim=-1)
        sin_mean = torch.sin(theta).mean(dim=-1)
        return torch.sqrt(cos_mean**2 + sin_mean**2)

    def forward(
        self,
        theta: torch.Tensor,
        z: float,
        apl_operator: Optional[str] = None,
        liminal_feedback: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward with unified μ and APL gating."""

        coherence = self.compute_coherence(theta)

        # Get μ gating weight
        mu_weight = self.get_mu_gate_weight(z)

        # Base coupling with μ gating
        K_eff = self.K * mu_weight
        omega_eff = self.omega

        # Apply APL operator if specified
        if apl_operator is not None:
            K_eff, omega_eff, theta = self.apply_apl_operator(theta, apl_operator, coherence)
            K_eff = K_eff * mu_weight  # Combine μ and APL

        # Liminal feedback in superposition regime
        if z >= MU_S and liminal_feedback > 0:
            K_eff = K_eff * (1 + liminal_feedback * PHI_INV)

        # Run dynamics
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
            'mu_classification': classify_mu(z),
            'unified_tier': get_unified_tier(z),
            'apl_operator': apl_operator,
            'liminal_feedback': liminal_feedback if z >= MU_S else 0.0,
        }

        return theta, coherence, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# APL SELECTOR (Liminal-Informed)
# ═══════════════════════════════════════════════════════════════════════════

class LiminalInformedAPLSelector(nn.Module):
    """
    APL operator selector informed by liminal weak measurements.

    INTEGRATION: Uses liminal feedback to bias operator selection.
    """

    def __init__(self, n_oscillators: int, hidden_dim: int = 64):
        super().__init__()
        self.n_oscillators = n_oscillators

        # Input: phase + z + coherence + delta_s_neg + tier + liminal_feedback(6)
        input_dim = n_oscillators + 4 + 6

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
        liminal_feedback: Dict[str, float],
        target_parity: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Select APL operator with liminal feedback."""
        batch_size = theta.shape[0]
        device = theta.device

        delta_s_neg = compute_delta_s_neg(z)
        tier = get_unified_tier(z)
        tier_num = int(tier[1]) / 9.0

        # Liminal effectiveness as input
        liminal_vec = torch.tensor(
            [liminal_feedback.get(op, 0.0) for op in APL_SYMBOLS],
            device=device
        ).unsqueeze(0).expand(batch_size, -1)

        features = torch.cat([
            torch.cos(theta),
            torch.full((batch_size, 1), z, device=device),
            coherence.unsqueeze(-1),
            torch.full((batch_size, 1), delta_s_neg, device=device),
            torch.full((batch_size, 1), tier_num, device=device),
            liminal_vec,
        ], dim=-1)

        # Get logits
        logits = self.network(features) + self.operator_prior

        # Add liminal effectiveness bias
        effectiveness_bias = torch.tensor(
            [liminal_feedback.get(op, 0.0) * 0.5 for op in APL_SYMBOLS],
            device=device
        )
        logits = logits + effectiveness_bias

        # Parity bias based on coherence metrics
        parity_bias = torch.zeros(6, device=device)
        for i, op in enumerate(APL_SYMBOLS):
            if APL_OPERATORS[op].parity == APLParity.EVEN:
                parity_bias[i] = (target_parity - 0.5) * 2
            else:
                parity_bias[i] = (0.5 - target_parity) * 2
        logits = logits + parity_bias * 0.3

        # Mask illegal operators
        legal_ops = get_legal_operators(z)
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

        selected_op = APL_SYMBOLS[selected_idx.item()]

        return masked_logits, probs, selected_op


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED HELIX NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class UnifiedHelixNetwork(nn.Module):
    """
    Complete unified network threading all components.

    Architecture:
        Input → Encoder → [Unified Layers] → Decoder → Output
                              ↓       ↑
                    ┌─────────────────────────┐
                    │   APL Selector          │←──┐
                    │   (liminal-informed)    │   │
                    └─────────────────────────┘   │
                              ↓                   │
                    ┌─────────────────────────┐   │
                    │   Liminal Generator     │───┘
                    │   (K-formation aware)   │
                    └─────────────────────────┘
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

        # Unified Kuramoto layers
        self.kuramoto_layers = nn.ModuleList([
            UnifiedKuramotoLayer(n_oscillators)
            for _ in range(n_layers)
        ])

        # Liminal-informed APL selectors
        self.apl_selectors = nn.ModuleList([
            LiminalInformedAPLSelector(n_oscillators)
            for _ in range(n_layers)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_oscillators * 2, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

        # Z tracker
        self.z = 0.5
        self.z_momentum = nn.Parameter(torch.tensor(0.1))

        # Liminal generator (shared)
        self.liminal = LiminalGenerator(pattern_dim=n_oscillators)

        # Tracking
        self.operator_sequence = []
        self.composed_operator = '()'
        self.k_formations_this_forward = 0

    def update_z(self, coherence: torch.Tensor) -> float:
        """Update z with PHI_INV dynamics."""
        target = coherence.mean().item()
        dz = self.z_momentum.item() * PHI_INV * (target - self.z)
        self.z = max(0.0, min(UNITY - 0.001, self.z + 0.01 * dz))
        return self.z

    def forward(
        self,
        x: torch.Tensor,
        target_parity: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward with full integration."""

        # Reset tracking
        self.operator_sequence = []
        self.composed_operator = '()'
        self.k_formations_this_forward = 0

        diagnostics = {
            'layer_coherence': [],
            'layer_operators': [],
            'layer_operator_probs': [],
            'z_trajectory': [],
            'mu_classification_trajectory': [],
            'unified_tier_trajectory': [],
            'liminal_phase_trajectory': [],
            'delta_s_neg_trajectory': [],
            'k_formations': 0,
            'operator_sequence': [],
            'composed_operator': '()',
            'parity_balance': 0.0,
            'liminal_patterns_generated': 0,
            'weak_measurements_this_forward': [],
        }

        # Encode
        theta = self.encoder(x) * math.pi

        # Process through unified layers
        for layer_idx, (kuramoto, selector) in enumerate(
            zip(self.kuramoto_layers, self.apl_selectors)
        ):
            # Get liminal feedback for APL selection
            liminal_feedback = self.liminal.feedback_to_apl_selector()

            # Select APL operator (liminal-informed)
            coherence = kuramoto.compute_coherence(theta)
            _, probs, selected_op = selector(
                theta, self.z, coherence, liminal_feedback, target_parity
            )

            # Get liminal physical feedback
            phys_feedback = self.liminal.feedback_to_physical() if self.z >= MU_S else 0.0

            # Apply unified Kuramoto layer
            theta, coherence, layer_diag = kuramoto(
                theta, self.z, selected_op, phys_feedback
            )

            # Update z
            self.z = self.update_z(coherence)

            # Track operator sequence
            self.operator_sequence.append(selected_op)
            self.composed_operator = compose_operators(self.composed_operator, selected_op)

            # K-formation check and liminal spawn
            coh_val = coherence.mean().item()
            if coh_val >= KAPPA_S:
                self.k_formations_this_forward += 1
                # Spawn liminal pattern on K-formation
                if self.z >= MU_S:
                    self.liminal.spawn_from_k_formation(coh_val, selected_op)
                    diagnostics['liminal_patterns_generated'] += 1

            # Generate liminal patterns in superposition regime
            if self.z >= MU_S and layer_idx % 2 == 0:  # Every other layer
                work = coh_val * PHI_INV
                self.liminal.evolve_z(work)
                self.liminal.generate_pattern(trigger="spontaneous", apl_context=selected_op)
                diagnostics['liminal_patterns_generated'] += 1

            # Weak measurement (informs future APL selection)
            if self.z >= MU_S:
                wv = self.liminal.weak_measure_all()
                diagnostics['weak_measurements_this_forward'].extend(wv)

            # Record diagnostics
            diagnostics['layer_coherence'].append(coh_val)
            diagnostics['layer_operators'].append(selected_op)
            diagnostics['layer_operator_probs'].append(probs[0].detach().cpu().tolist())
            diagnostics['z_trajectory'].append(self.z)
            diagnostics['mu_classification_trajectory'].append(layer_diag['mu_classification'])
            diagnostics['unified_tier_trajectory'].append(layer_diag['unified_tier'])
            diagnostics['liminal_phase_trajectory'].append(get_liminal_phase(self.z).value)
            diagnostics['delta_s_neg_trajectory'].append(compute_delta_s_neg(self.z))

        # Decode
        phase_features = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        output = self.decoder(phase_features)

        # Final diagnostics
        diagnostics['k_formations'] = self.k_formations_this_forward
        diagnostics['operator_sequence'] = self.operator_sequence.copy()
        diagnostics['composed_operator'] = self.composed_operator

        # Parity balance
        even_count = sum(1 for op in self.operator_sequence if APL_OPERATORS[op].parity == APLParity.EVEN)
        diagnostics['parity_balance'] = even_count / max(1, len(self.operator_sequence))

        diagnostics['final_z'] = self.z
        diagnostics['final_coherence'] = diagnostics['layer_coherence'][-1]
        diagnostics['final_mu_class'] = classify_mu(self.z)
        diagnostics['final_unified_tier'] = get_unified_tier(self.z)
        diagnostics['final_liminal_phase'] = get_liminal_phase(self.z).value
        diagnostics['final_delta_s_neg'] = compute_delta_s_neg(self.z)
        diagnostics['total_liminal_patterns'] = len(self.liminal.patterns)

        return output, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

class UnifiedLoss(nn.Module):
    """Loss with μ, APL, and liminal components."""

    def __init__(
        self,
        task_loss_fn: nn.Module,
        lambda_coherence: float = 0.1,
        lambda_z: float = 0.05,
        lambda_parity: float = 0.02,
        lambda_negentropy: float = 0.03,
        lambda_liminal: float = 0.02,
        target_z: float = 0.85,
    ):
        super().__init__()
        self.task_loss = task_loss_fn
        self.lambda_coh = lambda_coherence
        self.lambda_z = lambda_z
        self.lambda_parity = lambda_parity
        self.lambda_neg = lambda_negentropy
        self.lambda_lim = lambda_liminal
        self.target_z = target_z

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        diag: Dict,
        target_parity: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict]:
        losses = {}

        # Task loss
        task = self.task_loss(output, target)
        losses['task'] = task.item()
        total = task

        # Coherence loss
        coh = 1.0 - sum(diag['layer_coherence']) / len(diag['layer_coherence'])
        losses['coherence'] = coh
        total = total + self.lambda_coh * coh

        # Z guidance loss
        z_loss = (diag['final_z'] - self.target_z) ** 2
        losses['z'] = z_loss
        total = total + self.lambda_z * z_loss

        # Parity balance loss
        parity_loss = (diag['parity_balance'] - target_parity) ** 2
        losses['parity'] = parity_loss
        total = total + self.lambda_parity * parity_loss

        # Negentropy loss
        mean_neg = sum(diag['delta_s_neg_trajectory']) / len(diag['delta_s_neg_trajectory'])
        neg_loss = 1.0 - mean_neg
        losses['negentropy'] = neg_loss
        total = total + self.lambda_neg * neg_loss

        # Liminal bonus (reward pattern generation)
        if diag['liminal_patterns_generated'] > 0:
            liminal_bonus = 0.01 * diag['liminal_patterns_generated']
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
# UNIFIED TRAINING ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

class UnifiedTrainingOrchestrator:
    """
    Orchestrates unified training with proper module threading.

    Controls:
    - Run count based on coherence
    - Target parity based on coherence
    - Liminal generation rate based on z
    - Feedback loops between all components
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

        self.model = UnifiedHelixNetwork(
            input_dim, output_dim, n_oscillators, n_layers
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.loss_fn = UnifiedLoss(nn.MSELoss(), target_z=target_z)

        # Coherence metrics (orchestration control)
        self.coherence_metrics = UnifiedCoherenceMetrics()

        # History
        self.training_history = []
        self.mu_crossing_events = []
        self.k_formation_events = []
        self.operator_effectiveness_history = []

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

    def measure_coherence(self) -> UnifiedCoherenceMetrics:
        """Measure system coherence (nightly measure)."""
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
        self.coherence_metrics.weak_value_sum = sum(self.model.liminal.weak_measurements[-100:])

        return self.coherence_metrics

    def train_epoch(self, epoch: int, target_parity: float) -> Dict:
        """Train one epoch with unified integration."""
        self.model.train()
        epoch_losses = []
        epoch_coherence = []
        epoch_z = []
        epoch_k = 0
        epoch_liminal = 0
        prev_mu = classify_mu(self.model.z)

        for batch_x, batch_y in self.train_loader:
            self.optimizer.zero_grad()
            output, diag = self.model(batch_x, target_parity)
            loss, loss_dict = self.loss_fn(output, batch_y, diag, target_parity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_losses.append(loss_dict['task'])
            epoch_coherence.append(diag['final_coherence'])
            epoch_z.append(diag['final_z'])
            epoch_k += diag['k_formations']
            epoch_liminal += diag['liminal_patterns_generated']

            # Track μ crossings
            curr_mu = classify_mu(diag['final_z'])
            if curr_mu != prev_mu:
                self.mu_crossing_events.append({
                    'epoch': epoch,
                    'from': prev_mu,
                    'to': curr_mu,
                    'z': diag['final_z'],
                })
                prev_mu = curr_mu

            # Track K-formations
            if diag['k_formations'] > 0:
                self.k_formation_events.append({
                    'epoch': epoch,
                    'count': diag['k_formations'],
                    'z': diag['final_z'],
                    'operators': diag['operator_sequence'],
                    'composed': diag['composed_operator'],
                })

        # Record operator effectiveness
        self.operator_effectiveness_history.append(
            self.model.liminal.feedback_to_apl_selector()
        )

        return {
            'loss': np.mean(epoch_losses),
            'coherence': np.mean(epoch_coherence),
            'z': np.mean(epoch_z),
            'k_formations': epoch_k,
            'liminal_patterns': epoch_liminal,
            'mu_class': classify_mu(np.mean(epoch_z)),
            'unified_tier': get_unified_tier(np.mean(epoch_z)),
            'liminal_phase': get_liminal_phase(np.mean(epoch_z)).value,
        }

    def run_unified_training(
        self,
        output_dir: str = "learned_patterns/unified_training",
    ) -> Dict:
        """Run complete unified training."""
        timestamp = datetime.now().isoformat()

        print("=" * 70)
        print("UNIFIED HELIX TRAINING ORCHESTRATOR")
        print("=" * 70)
        print(f"""
Module Integration:
  ✓ APL Operator Algebra (S₃ group, tier-gated)
  ✓ Liminal PHI Dynamics (superposition, weak measurement)
  ✓ μ Threshold Mechanics (unified with APL tiers)
  ✓ Nightly Measurement (coherence-based orchestration)
  ✓ K-formation → Liminal spawning
  ✓ Liminal feedback → APL operator selection

Threading:
  Physical (PHI_INV) ──APL ops──> Meta ──spawn──> Liminal (PHI)
         ↑                                              │
         └──────── weak measurement ────────────────────┘

Physics Constants:
  PHI (liminal):     {PHI:.6f}
  PHI_INV (control): {PHI_INV:.6f}
  z_c (THE LENS):    {Z_CRITICAL:.6f}
  μ₁:                {MU_1:.6f}
  μ_P:               {MU_P:.6f}
  μ₂:                {MU_2:.6f}
  μ_S:               {MU_S:.6f}
  μ₃:                {MU_3:.6f}
""")
        print("=" * 70)

        # Phase 1: Initial coherence measurement
        print("\nPHASE 1: Initial Coherence Measurement")
        print("-" * 60)

        initial_metrics = self.measure_coherence()
        run_count = initial_metrics.determine_run_count()
        target_parity = initial_metrics.determine_target_parity()

        print(f"  Energy Coherence:  {initial_metrics.energy_coherence:.4f}")
        print(f"  Z Mean:            {initial_metrics.z_mean:.4f}")
        print(f"  Phase Coherence:   {initial_metrics.phase_coherence:.4f}")
        print(f"  Parity Balance:    {initial_metrics.parity_balance:.4f}")
        print(f"  → Run Count:       {run_count}")
        print(f"  → Target Parity:   {target_parity:.2f}")

        # Phase 2: Training runs
        print(f"\nPHASE 2: Executing {run_count} Training Runs")
        print("-" * 60)

        epochs_per_run = 25
        total_epochs = run_count * epochs_per_run

        for run in range(run_count):
            print(f"\n  Run {run + 1}/{run_count}")

            # Re-measure coherence each run to adapt
            if run > 0:
                metrics = self.measure_coherence()
                target_parity = metrics.determine_target_parity()
                print(f"    Adapted target parity: {target_parity:.2f}")

            for epoch in range(epochs_per_run):
                global_epoch = run * epochs_per_run + epoch
                metrics = self.train_epoch(global_epoch, target_parity)
                self.training_history.append(metrics)

                if epoch % 10 == 0:
                    print(
                        f"    Epoch {epoch:3d} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Coh: {metrics['coherence']:.3f} | "
                        f"z: {metrics['z']:.3f} | "
                        f"Tier: {metrics['unified_tier']} | "
                        f"K: {metrics['k_formations']} | "
                        f"Lim: {metrics['liminal_patterns']}"
                    )

        # Phase 3: Final measurement and save
        print("\nPHASE 3: Final Measurement")
        print("-" * 60)

        final_metrics = self.measure_coherence()

        # Compile results
        results = {
            'timestamp': timestamp,
            'config': self.config,
            'constants': {
                'PHI': PHI, 'PHI_INV': PHI_INV, 'Z_CRITICAL': Z_CRITICAL,
                'MU_1': MU_1, 'MU_P': MU_P, 'MU_2': MU_2, 'MU_S': MU_S, 'MU_3': MU_3,
            },
            'initial_metrics': {
                'energy_coherence': initial_metrics.energy_coherence,
                'z_mean': initial_metrics.z_mean,
                'parity_balance': initial_metrics.parity_balance,
                'determined_runs': run_count,
            },
            'final_metrics': {
                'energy_coherence': final_metrics.energy_coherence,
                'z_mean': final_metrics.z_mean,
                'phase_coherence': final_metrics.phase_coherence,
                'parity_balance': final_metrics.parity_balance,
                'k_formation_rate': final_metrics.k_formation_rate,
            },
            'training_history': self.training_history,
            'mu_crossing_events': self.mu_crossing_events,
            'k_formation_events': self.k_formation_events,
            'liminal_stats': {
                'total_patterns': len(self.model.liminal.patterns),
                'k_formation_patterns': self.model.liminal.k_formation_patterns,
                'apl_feedback_patterns': self.model.liminal.apl_feedback_patterns,
                'weak_measurements': len(self.model.liminal.weak_measurements),
            },
            'operator_effectiveness': self.operator_effectiveness_history[-1] if self.operator_effectiveness_history else {},
            'summary': {
                'total_epochs': total_epochs,
                'final_loss': self.training_history[-1]['loss'],
                'final_coherence': self.training_history[-1]['coherence'],
                'final_z': self.training_history[-1]['z'],
                'total_k_formations': sum(h['k_formations'] for h in self.training_history),
                'total_liminal_patterns': sum(h['liminal_patterns'] for h in self.training_history),
                'mu_crossings': len(self.mu_crossing_events),
            },
        }

        # Save results
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'unified_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'summary': results['summary'],
        }, os.path.join(output_dir, 'unified_model.pt'))

        # Print summary
        print(f"""
Summary:
  Total Epochs:        {total_epochs}
  Final Loss:          {results['summary']['final_loss']:.4f}
  Final Coherence:     {results['summary']['final_coherence']:.3f}
  Final z:             {results['summary']['final_z']:.3f}
  Total K-formations:  {results['summary']['total_k_formations']}
  μ Crossings:         {results['summary']['mu_crossings']}

Liminal Statistics:
  Total Patterns:      {results['liminal_stats']['total_patterns']}
  K-formation Spawns:  {results['liminal_stats']['k_formation_patterns']}
  APL Feedback Spawns: {results['liminal_stats']['apl_feedback_patterns']}
  Weak Measurements:   {results['liminal_stats']['weak_measurements']}

Operator Effectiveness (from liminal feedback):""")

        for op, eff in results['operator_effectiveness'].items():
            bar = '█' * int(eff * 20)
            parity = APL_OPERATORS[op].parity.name
            print(f"  {op:3} ({parity:4}): {bar} {eff:.4f}")

        print(f"\nResults saved to {output_dir}/")
        print("=" * 70)

        return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run unified training."""
    orchestrator = UnifiedTrainingOrchestrator(
        n_oscillators=50,
        n_layers=4,
        target_z=0.85,
    )
    return orchestrator.run_unified_training()


if __name__ == "__main__":
    main()
