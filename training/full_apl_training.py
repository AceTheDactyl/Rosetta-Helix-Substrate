#!/usr/bin/env python3
"""
Full APL (Array Programming Language) Training
===============================================

Complete integration with the S₃ group algebra structure:

S₃ GROUP (6 OPERATORS):
    Symbol  Name   S₃ Element  Parity  Inverse  Order  Cybernetic
    ------  ----   ----------  ------  -------  -----  -----------
    ()      grp    e (identity) EVEN    ^        1      CLOSURE
    ×       mul    σ (3-cycle)  EVEN    ÷        3      INTEGRATION
    ^       amp    σ² (3-cycle) EVEN    ()       3      GAIN
    +       add    τ₂ (swap)    ODD     −        2      AGGREGATION
    −       sub    τ₃ (swap)    ODD     +        2      DIFFERENTIATION
    ÷       div    τ₁ (swap)    ODD     ×        2      NOISE

GROUP PROPERTIES:
    - Closed under composition (op₁ ∘ op₂ always yields valid op)
    - Every operator has an inverse
    - Associative: (a ∘ b) ∘ c = a ∘ (b ∘ c)
    - Identity: () ∘ op = op ∘ () = op

TIER-GATED WINDOWS:
    t1 (0.00-0.10): [(), −, ÷]
    t2 (0.10-0.20): [^, ÷, −, ×]
    t3 (0.20-0.40): [×, ^, ÷, +, −]
    t4 (0.40-0.60): [+, −, ÷, ()]
    t5 (0.60-0.75): [ALL] - Full access at PARADOX threshold
    t6 (0.75-0.866): [+, ÷, (), −] - Near THE LENS
    t7 (0.866-0.92): [+, ()]
    t8 (0.92-0.97): [+, (), ×]
    t9 (0.97-1.00): [+, (), ×]

TRUTH CHANNEL BIASING:
    TRUE:    Favor ^, +, × (constructive/aggregation)
    UNTRUE:  Favor ÷, −, () (dissipative/isolation)
    PARADOX: Favor (), × (containment/integration)

PARITY WEIGHTING:
    High ΔS_neg: Favor EVEN parity (constructive)
    Low ΔS_neg:  Favor ODD parity (dissipative)
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
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Z_CRITICAL = math.sqrt(3) / 2

# μ Thresholds
MU_P = 2.0 / (PHI ** 2.5)
MU_S = 0.920
MU_3 = 0.992

# TRIAD
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82

# Negative entropy
SIGMA_NEG_ENTROPY = 36.0  # σ for ΔS_neg = exp[-σ(z - z_c)²]

# Truth channel boundaries (aligned with phase regime mapping)
Z_PRESENCE_MIN = 0.877  # TRUE threshold (upper bound of THE_LENS phase)


# ═══════════════════════════════════════════════════════════════════════════
# S₃ GROUP ALGEBRA
# ═══════════════════════════════════════════════════════════════════════════

class Parity(Enum):
    """Operator parity in S₃ group"""
    EVEN = 1    # Constructive: identity and 3-cycles
    ODD = -1    # Dissipative: transpositions


class CyberneticRole(Enum):
    """Cybernetic function of each operator"""
    CLOSURE = "closure"              # () - Establish/relax containment
    INTEGRATION = "integration"       # × - Join, entangle, mix
    GAIN = "gain"                    # ^ - Amplify, pump, bias
    NOISE = "noise"                  # ÷ - Diffuse, randomize, reset
    AGGREGATION = "aggregation"      # + - Aggregate, route, converge
    DIFFERENTIATION = "differentiation"  # − - Split, isolate, fork


@dataclass
class S3Element:
    """Element of the S₃ symmetric group"""
    name: str
    permutation: Tuple[int, int, int]  # How it permutes [0, 1, 2]
    parity: Parity
    sign: int  # +1 for even, -1 for odd


# The 6 elements of S₃
S3_ELEMENTS: Dict[str, S3Element] = {
    "e":   S3Element("identity",    (0, 1, 2), Parity.EVEN, +1),
    "σ":   S3Element("3-cycle",     (1, 2, 0), Parity.EVEN, +1),
    "σ2":  S3Element("3-cycle-inv", (2, 0, 1), Parity.EVEN, +1),
    "τ1":  S3Element("swap-12",     (1, 0, 2), Parity.ODD,  -1),
    "τ2":  S3Element("swap-23",     (0, 2, 1), Parity.ODD,  -1),
    "τ3":  S3Element("swap-13",     (2, 1, 0), Parity.ODD,  -1),
}


@dataclass
class APLOperator:
    """Complete APL operator definition"""
    symbol: str
    name: str
    s3_element: str
    parity: Parity
    inverse: str
    order: int  # Order in group (1, 2, or 3)
    cybernetic: CyberneticRole
    description: str


# Complete operator definitions
APL_OPERATORS: Dict[str, APLOperator] = {
    "()": APLOperator("()", "grp", "e",   Parity.EVEN, "^",  1, CyberneticRole.CLOSURE,
                      "Establish/relax containment surfaces"),
    "×":  APLOperator("×",  "mul", "σ",   Parity.EVEN, "÷",  3, CyberneticRole.INTEGRATION,
                      "Join, entangle, or mix subsystems"),
    "^":  APLOperator("^",  "amp", "σ2",  Parity.EVEN, "()", 3, CyberneticRole.GAIN,
                      "Apply gain, pumping, or bias"),
    "+":  APLOperator("+",  "add", "τ2",  Parity.ODD,  "−",  2, CyberneticRole.AGGREGATION,
                      "Aggregate, route, or converge flows"),
    "÷":  APLOperator("÷",  "div", "τ1",  Parity.ODD,  "×",  2, CyberneticRole.NOISE,
                      "Diffuse, randomize, or reset phases"),
    "−":  APLOperator("−",  "sub", "τ3",  Parity.ODD,  "+",  2, CyberneticRole.DIFFERENTIATION,
                      "Split, isolate, or fork structures"),
}

APL_SYMBOLS = list(APL_OPERATORS.keys())


def compose_s3(a: str, b: str) -> str:
    """Compose two S₃ elements: (a ∘ b)(i) = a(b(i))"""
    elem_a = S3_ELEMENTS[a]
    elem_b = S3_ELEMENTS[b]

    # Apply b first, then a
    result_perm = tuple(elem_a.permutation[elem_b.permutation[i]] for i in range(3))

    # Find matching element
    for name, elem in S3_ELEMENTS.items():
        if elem.permutation == result_perm:
            return name

    return "e"  # Fallback to identity


def compose_operators(a: str, b: str) -> str:
    """Compose two APL operators using S₃ group multiplication"""
    s3_a = APL_OPERATORS[a].s3_element
    s3_b = APL_OPERATORS[b].s3_element
    s3_result = compose_s3(s3_a, s3_b)

    # Map S₃ element back to operator
    for sym, op in APL_OPERATORS.items():
        if op.s3_element == s3_result:
            return sym

    return "()"  # Identity fallback


def compose_sequence(operators: List[str]) -> str:
    """Compose a sequence of operators left-to-right"""
    if not operators:
        return "()"
    result = operators[0]
    for op in operators[1:]:
        result = compose_operators(result, op)
    return result


def get_inverse(symbol: str) -> str:
    """Get the inverse operator"""
    return APL_OPERATORS[symbol].inverse


def get_order(symbol: str) -> int:
    """Get the order of an operator (how many times to apply to get identity)"""
    return APL_OPERATORS[symbol].order


# ═══════════════════════════════════════════════════════════════════════════
# TIER-GATED OPERATOR WINDOWS
# ═══════════════════════════════════════════════════════════════════════════

# Tier boundaries (z values)
TIER_BOUNDS = [0.10, 0.20, 0.40, 0.60, 0.75, Z_CRITICAL, 0.92, 0.97]
TIER_NAMES = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]

# Operator windows per tier
TIER_WINDOWS: Dict[str, List[str]] = {
    "t1": ["()", "−", "÷"],
    "t2": ["^", "÷", "−", "×"],
    "t3": ["×", "^", "÷", "+", "−"],
    "t4": ["+", "−", "÷", "()"],
    "t5": ["()", "×", "^", "÷", "+", "−"],  # All operators
    "t6": ["+", "÷", "()", "−"],
    "t7": ["+", "()"],
    "t8": ["+", "()", "×"],
    "t9": ["+", "()", "×"],
}


def get_tier(z: float, triad_unlocked: bool = False) -> str:
    """Get tier name from z coordinate"""
    t6_gate = TRIAD_LOW if triad_unlocked else Z_CRITICAL
    bounds = TIER_BOUNDS.copy()
    bounds[5] = t6_gate

    for i, bound in enumerate(bounds):
        if z < bound:
            return TIER_NAMES[i]
    return "t9"


def get_operator_window(z: float, triad_unlocked: bool = False) -> List[str]:
    """Get available operators for current z position"""
    tier = get_tier(z, triad_unlocked)
    return TIER_WINDOWS.get(tier, ["()"])


# ═══════════════════════════════════════════════════════════════════════════
# TRUTH CHANNEL BIASING
# ═══════════════════════════════════════════════════════════════════════════

class TruthChannel(Enum):
    TRUE = "TRUE"
    PARADOX = "PARADOX"
    UNTRUE = "UNTRUE"


def get_truth_channel(z: float) -> TruthChannel:
    """Determine truth channel from z position"""
    if z >= Z_PRESENCE_MIN:
        return TruthChannel.TRUE
    elif z >= PHI_INV:
        return TruthChannel.PARADOX
    else:
        return TruthChannel.UNTRUE


# Truth channel bias weights
TRUTH_BIAS: Dict[str, Dict[str, float]] = {
    "TRUE": {
        "^": 1.5, "+": 1.4, "×": 1.0,
        "()": 0.9, "÷": 0.7, "−": 0.7
    },
    "UNTRUE": {
        "÷": 1.5, "−": 1.4, "()": 1.0,
        "+": 0.9, "^": 0.7, "×": 0.7
    },
    "PARADOX": {
        "()": 1.5, "×": 1.4,
        "+": 1.0, "^": 1.0,
        "÷": 0.9, "−": 0.9
    },
}


@dataclass
class TruthDistribution:
    """Distribution over truth values"""
    TRUE: float
    PARADOX: float
    UNTRUE: float

    def as_list(self) -> List[float]:
        return [self.TRUE, self.PARADOX, self.UNTRUE]

    @classmethod
    def from_list(cls, values: List[float]) -> 'TruthDistribution':
        return cls(TRUE=values[0], PARADOX=values[1], UNTRUE=values[2])

    def normalize(self) -> 'TruthDistribution':
        total = self.TRUE + self.PARADOX + self.UNTRUE
        if total > 0:
            return TruthDistribution(
                self.TRUE / total,
                self.PARADOX / total,
                self.UNTRUE / total
            )
        return TruthDistribution(1/3, 1/3, 1/3)


def apply_s3_to_truth(dist: TruthDistribution, s3_element: str) -> TruthDistribution:
    """Apply S₃ permutation to truth distribution"""
    elem = S3_ELEMENTS[s3_element]
    values = dist.as_list()
    permuted = [values[elem.permutation[i]] for i in range(3)]
    return TruthDistribution.from_list(permuted)


def apply_operator_to_truth(dist: TruthDistribution, operator: str) -> TruthDistribution:
    """Apply APL operator's S₃ action to truth distribution"""
    s3_elem = APL_OPERATORS[operator].s3_element
    return apply_s3_to_truth(dist, s3_elem)


# ═══════════════════════════════════════════════════════════════════════════
# OPERATOR WEIGHTING
# ═══════════════════════════════════════════════════════════════════════════

def compute_delta_s_neg(z: float) -> float:
    """Negative entropy: ΔS_neg = exp[-σ(z - z_c)²], peaks at z_c"""
    d = z - Z_CRITICAL
    return math.exp(-SIGMA_NEG_ENTROPY * d * d)


def compute_operator_weight(
    operator: str,
    z: float,
    delta_s_neg: float,
    truth_channel: TruthChannel,
) -> float:
    """
    Compute weight for operator selection.

    Combines:
    1. Truth channel bias
    2. Parity-based coherence weighting
    3. Tier membership bonus
    """
    op = APL_OPERATORS[operator]

    # Base weight from truth bias
    truth_bias = TRUTH_BIAS[truth_channel.value].get(operator, 1.0)
    weight = truth_bias

    # Parity weighting: high ΔS_neg favors EVEN, low favors ODD
    if op.parity == Parity.EVEN:
        parity_boost = delta_s_neg
    else:
        parity_boost = 1 - delta_s_neg
    weight *= (0.8 + 0.4 * parity_boost)

    # Tier membership bonus
    window = get_operator_window(z)
    if operator in window:
        weight *= 1.2
    else:
        weight *= 0.5  # Penalty for out-of-window

    return weight


def select_operator(
    z: float,
    delta_s_neg: float,
    truth_channel: TruthChannel,
    temperature: float = 1.0,
) -> str:
    """Select operator using weighted sampling"""
    window = get_operator_window(z)
    if not window:
        return "()"

    weights = []
    for op in window:
        w = compute_operator_weight(op, z, delta_s_neg, truth_channel)
        weights.append(w)

    # Softmax with temperature
    weights = np.array(weights)
    weights = np.exp(weights / temperature)
    weights = weights / weights.sum()

    return np.random.choice(window, p=weights)


# ═══════════════════════════════════════════════════════════════════════════
# APL OPERATOR HANDLERS (Effect on Dynamics)
# ═══════════════════════════════════════════════════════════════════════════

def apply_operator_to_phases(
    phases: torch.Tensor,
    operator: str,
    coupling: float,
) -> torch.Tensor:
    """Apply APL operator effect to oscillator phases (handles batched input)"""
    op = APL_OPERATORS[operator]

    # Handle batched input: phases is [batch, n_oscillators]
    if phases.dim() == 1:
        phases = phases.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, n_osc = phases.shape

    if operator == "()":
        # CLOSURE: Stabilize current state (no change)
        result = phases

    elif operator == "×":
        # INTEGRATION: Pull phases toward mean (increase coupling)
        sin_mean = torch.sin(phases).mean(dim=-1, keepdim=True)
        cos_mean = torch.cos(phases).mean(dim=-1, keepdim=True)
        mean_phase = torch.atan2(sin_mean, cos_mean)
        delta = 0.1 * coupling * torch.sin(mean_phase - phases)
        result = phases + delta

    elif operator == "^":
        # GAIN: Amplify phase differences from mean
        sin_mean = torch.sin(phases).mean(dim=-1, keepdim=True)
        cos_mean = torch.cos(phases).mean(dim=-1, keepdim=True)
        mean_phase = torch.atan2(sin_mean, cos_mean)
        diff = phases - mean_phase
        result = phases + 0.1 * coupling * diff

    elif operator == "+":
        # AGGREGATION: Pull phases toward local neighbors
        # Use convolution-like smoothing instead of sorting
        padded = F.pad(phases, (1, 1), mode='circular')
        smoothed = (padded[:, :-2] + phases + padded[:, 2:]) / 3
        result = phases + 0.1 * coupling * (smoothed - phases)

    elif operator == "÷":
        # NOISE: Add phase noise (decoherence)
        noise = torch.randn_like(phases) * 0.1 * coupling
        result = phases + noise

    elif operator == "−":
        # DIFFERENTIATION: Push phases apart from neighbors
        padded = F.pad(phases, (1, 1), mode='circular')
        diff_left = phases - padded[:, :-2]
        diff_right = phases - padded[:, 2:]
        # Amplify differences
        result = phases + 0.05 * coupling * (diff_left + diff_right)

    else:
        result = phases

    if squeeze_output:
        result = result.squeeze(0)

    return result


def apply_operator_to_coupling(
    coupling_matrix: torch.Tensor,
    operator: str,
    strength: float = 0.1,
) -> torch.Tensor:
    """Apply APL operator effect to coupling matrix"""
    op = APL_OPERATORS[operator]

    if operator == "()":
        # CLOSURE: Return to baseline (slight decay toward identity)
        eye = torch.eye(coupling_matrix.shape[0])
        return coupling_matrix * 0.95 + eye * 0.05

    elif operator == "×":
        # INTEGRATION: Increase all couplings
        return coupling_matrix * (1 + strength)

    elif operator == "^":
        # GAIN: Amplify strong couplings, reduce weak
        threshold = coupling_matrix.abs().mean()
        mask = coupling_matrix.abs() > threshold
        result = coupling_matrix.clone()
        result[mask] *= (1 + strength)
        result[~mask] *= (1 - strength * 0.5)
        return result

    elif operator == "+":
        # AGGREGATION: Average neighboring couplings
        kernel = torch.ones(3, 3) / 9
        # Simple smoothing
        return coupling_matrix * 0.9 + coupling_matrix.mean() * 0.1

    elif operator == "÷":
        # NOISE: Add noise to couplings
        noise = torch.randn_like(coupling_matrix) * strength
        return coupling_matrix + noise

    elif operator == "−":
        # DIFFERENTIATION: Increase variance in couplings
        mean_c = coupling_matrix.mean()
        return coupling_matrix + strength * (coupling_matrix - mean_c)

    return coupling_matrix


# ═══════════════════════════════════════════════════════════════════════════
# APL-AWARE KURAMOTO LAYER
# ═══════════════════════════════════════════════════════════════════════════

class APLKuramotoLayer(nn.Module):
    """Kuramoto layer with full S₃ APL operator algebra"""

    def __init__(self, n_oscillators: int = 60):
        super().__init__()
        self.n = n_oscillators

        # Coupling matrix
        K_init = torch.randn(n_oscillators, n_oscillators) * 0.1
        K_init = (K_init + K_init.T) / 2
        self.K = nn.Parameter(K_init)

        # Natural frequencies
        self.omega = nn.Parameter(torch.randn(n_oscillators) * 0.1)

        # Global coupling
        self.K_global = nn.Parameter(torch.tensor(PHI_INV))

        # Operator effect strengths (learnable)
        self.op_strengths = nn.ParameterDict({
            sym: nn.Parameter(torch.tensor(0.1)) for sym in APL_SYMBOLS
        })

        # Truth distribution (learnable starting point)
        self.truth_logits = nn.Parameter(torch.zeros(3))

    def compute_coherence(self, theta: torch.Tensor) -> torch.Tensor:
        cos_mean = torch.cos(theta).mean(dim=-1)
        sin_mean = torch.sin(theta).mean(dim=-1)
        return torch.sqrt(cos_mean**2 + sin_mean**2)

    def get_truth_distribution(self) -> TruthDistribution:
        probs = F.softmax(self.truth_logits, dim=0)
        return TruthDistribution(
            TRUE=probs[0].item(),
            PARADOX=probs[1].item(),
            UNTRUE=probs[2].item()
        )

    def forward(
        self,
        theta: torch.Tensor,
        z: float,
        applied_operators: List[str],
        dt: float = 0.1,
        steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        # Compute context
        delta_s_neg = compute_delta_s_neg(z)
        truth_channel = get_truth_channel(z)
        tier = get_tier(z)
        window = get_operator_window(z)

        # Get truth distribution
        truth_dist = self.get_truth_distribution()

        # Build effective coupling
        K_eff = self.K.clone()

        # Apply each operator's effect to coupling
        for op_sym in applied_operators:
            strength = self.op_strengths[op_sym].item()
            K_eff = apply_operator_to_coupling(K_eff, op_sym, strength)

        # Run Kuramoto dynamics
        for step in range(steps):
            # Apply operators to phases periodically
            if step % 3 == 0 and applied_operators:
                op = applied_operators[step // 3 % len(applied_operators)]
                strength = self.op_strengths[op].item()
                theta = apply_operator_to_phases(theta, op, strength)

            # Standard Kuramoto
            theta_expanded = theta.unsqueeze(-1)
            theta_diff = theta.unsqueeze(-2) - theta_expanded
            coupling = K_eff * torch.sin(theta_diff)
            coupling_sum = coupling.sum(dim=-1)
            coupling_term = (self.K_global / self.n) * coupling_sum
            dtheta = self.omega + coupling_term
            theta = theta + dt * dtheta
            theta = torch.atan2(torch.sin(theta), torch.cos(theta))

        coherence = self.compute_coherence(theta)

        # Apply operators to truth distribution
        final_truth = truth_dist
        for op in applied_operators:
            final_truth = apply_operator_to_truth(final_truth, op)

        diagnostics = {
            'tier': tier,
            'window': window,
            'truth_channel': truth_channel.value,
            'delta_s_neg': delta_s_neg,
            'applied_operators': applied_operators,
            'composed_result': compose_sequence(applied_operators) if applied_operators else "()",
            'initial_truth': truth_dist,
            'final_truth': final_truth,
        }

        return theta, coherence, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# APL SEQUENCE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

class APLSequenceGenerator:
    """Generates APL operator sequences based on context"""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.sequence_history: List[List[str]] = []

    def generate_sequence(
        self,
        z: float,
        delta_s_neg: float,
        truth_channel: TruthChannel,
        length: int = 3,
    ) -> List[str]:
        """Generate a sequence of operators"""
        sequence = []
        for _ in range(length):
            op = select_operator(z, delta_s_neg, truth_channel, self.temperature)
            sequence.append(op)
        self.sequence_history.append(sequence)
        return sequence

    def generate_balanced_sequence(
        self,
        z: float,
        length: int = 4,
    ) -> List[str]:
        """Generate sequence that returns to identity"""
        sequence = []
        current = "()"

        for i in range(length - 1):
            window = get_operator_window(z)
            op = random.choice(window)
            sequence.append(op)
            current = compose_operators(current, op)

        # Add inverse to return to identity
        inverse = get_inverse(current)
        sequence.append(inverse)

        return sequence


# ═══════════════════════════════════════════════════════════════════════════
# APL HELIX NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class APLHelixNetwork(nn.Module):
    """Neural network with full S₃ APL operator algebra"""

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

        # APL Kuramoto layers
        self.apl_layers = nn.ModuleList([
            APLKuramotoLayer(n_oscillators) for _ in range(n_layers)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_oscillators * 2, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

        # z dynamics
        self.z = 0.5
        self.z_velocity = 0.0
        self.z_momentum = nn.Parameter(torch.tensor(0.15))

        # APL sequence generator
        self.apl_generator = APLSequenceGenerator()

        # Operator usage tracking
        self.operator_usage: Dict[str, int] = {sym: 0 for sym in APL_SYMBOLS}
        self.composition_results: Dict[str, int] = {}

    def update_z(self, coherence: torch.Tensor) -> float:
        target = coherence.mean().item()
        z_accel = self.z_momentum.item() * (target - self.z) - 0.1 * self.z_velocity
        self.z_velocity += z_accel * 0.1
        self.z += self.z_velocity * 0.1
        self.z = max(0.01, min(0.99, self.z))
        return self.z

    def forward(self, x: torch.Tensor, epoch: int = 0) -> Tuple[torch.Tensor, Dict]:
        diagnostics = {
            'layer_coherence': [],
            'z_trajectory': [],
            'tier_trajectory': [],
            'operators_applied': [],
            'compositions': [],
            'truth_evolution': [],
        }

        theta = self.encoder(x) * math.pi

        for layer_idx, apl_layer in enumerate(self.apl_layers):
            # Generate operator sequence based on current context
            delta_s_neg = compute_delta_s_neg(self.z)
            truth_channel = get_truth_channel(self.z)

            # Generate 2-3 operators per layer
            operators = self.apl_generator.generate_sequence(
                self.z, delta_s_neg, truth_channel,
                length=random.randint(2, 3)
            )

            # Apply APL layer
            theta, coherence, layer_diag = apl_layer(
                theta, self.z, operators
            )

            # Track operator usage
            for op in operators:
                self.operator_usage[op] += 1

            # Track composition
            composed = layer_diag['composed_result']
            self.composition_results[composed] = self.composition_results.get(composed, 0) + 1

            # Update z
            new_z = self.update_z(coherence)

            # Record diagnostics
            diagnostics['layer_coherence'].append(coherence.mean().item())
            diagnostics['z_trajectory'].append(new_z)
            diagnostics['tier_trajectory'].append(layer_diag['tier'])
            diagnostics['operators_applied'].append(operators)
            diagnostics['compositions'].append(composed)
            diagnostics['truth_evolution'].append({
                'initial': layer_diag['initial_truth'],
                'final': layer_diag['final_truth'],
            })

        # Decode
        phase_features = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        output = self.decoder(phase_features)

        # Final diagnostics
        diagnostics['final_z'] = self.z
        diagnostics['final_coherence'] = diagnostics['layer_coherence'][-1]
        diagnostics['final_tier'] = get_tier(self.z)
        diagnostics['final_truth_channel'] = get_truth_channel(self.z).value
        diagnostics['final_delta_s_neg'] = compute_delta_s_neg(self.z)
        diagnostics['operator_usage'] = dict(self.operator_usage)
        diagnostics['composition_results'] = dict(self.composition_results)

        return output, diagnostics

    def reset(self):
        self.z = 0.5
        self.z_velocity = 0.0
        self.operator_usage = {sym: 0 for sym in APL_SYMBOLS}
        self.composition_results = {}


# ═══════════════════════════════════════════════════════════════════════════
# APL-AWARE LOSS
# ═══════════════════════════════════════════════════════════════════════════

class APLLoss(nn.Module):
    """Loss with S₃ group algebra rewards"""

    def __init__(
        self,
        task_loss_fn: nn.Module,
        lambda_coherence: float = 0.1,
        lambda_parity_balance: float = 0.05,
        lambda_composition: float = 0.05,
    ):
        super().__init__()
        self.task_loss = task_loss_fn
        self.lambda_coh = lambda_coherence
        self.lambda_parity = lambda_parity_balance
        self.lambda_comp = lambda_composition

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

        # Coherence reward
        coh = sum(diag['layer_coherence']) / len(diag['layer_coherence'])
        total = total - self.lambda_coh * coh
        losses['coherence_reward'] = coh

        # Parity balance reward (mix of even/odd)
        usage = diag['operator_usage']
        even_count = sum(usage.get(s, 0) for s in ["()", "×", "^"])
        odd_count = sum(usage.get(s, 0) for s in ["+", "−", "÷"])
        total_ops = even_count + odd_count + 1e-6
        balance = 1 - abs(even_count - odd_count) / total_ops
        total = total - self.lambda_parity * balance
        losses['parity_balance'] = balance

        # Composition variety reward
        n_unique = len(diag['composition_results'])
        comp_reward = min(1.0, n_unique / 6.0)  # Up to 6 unique compositions
        total = total - self.lambda_comp * comp_reward
        losses['composition_variety'] = comp_reward

        losses['total'] = total.item()
        return total, losses


# ═══════════════════════════════════════════════════════════════════════════
# APL TRAINING SESSION
# ═══════════════════════════════════════════════════════════════════════════

class FullAPLTraining:
    """Training session with complete S₃ APL operator algebra"""

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

        self.model = APLHelixNetwork(
            input_dim, output_dim, n_oscillators, n_layers
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.loss_fn = APLLoss(nn.MSELoss())

        self.training_history = []
        self.operator_history: Dict[str, List[int]] = {sym: [] for sym in APL_SYMBOLS}

        self._generate_data()

    def _generate_data(self, n_train: int = 800):
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
        self.model.train()
        epoch_losses = []
        epoch_coherence = []
        epoch_z = []

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

        # Record operator usage
        for sym in APL_SYMBOLS:
            self.operator_history[sym].append(self.model.operator_usage.get(sym, 0))

        return {
            'loss': np.mean(epoch_losses),
            'coherence': np.mean(epoch_coherence),
            'z': np.mean(epoch_z),
            'tier': get_tier(np.mean(epoch_z)),
            'truth_channel': get_truth_channel(np.mean(epoch_z)).value,
            'delta_s_neg': compute_delta_s_neg(np.mean(epoch_z)),
            'operator_usage': dict(self.model.operator_usage),
            'composition_results': dict(self.model.composition_results),
        }

    def run_training(
        self,
        n_epochs: int = 100,
        output_dir: str = "learned_patterns/apl_training",
    ) -> Dict:
        timestamp = datetime.now().isoformat()

        print("=" * 70)
        print("FULL S₃ APL OPERATOR ALGEBRA TRAINING")
        print("=" * 70)
        print(f"""
S₃ GROUP OPERATORS (6 total):
  EVEN (Constructive):
    ()  grp  identity     order=1  CLOSURE
    ×   mul  3-cycle      order=3  INTEGRATION
    ^   amp  3-cycle-inv  order=3  GAIN

  ODD (Dissipative):
    +   add  swap-23      order=2  AGGREGATION
    −   sub  swap-13      order=2  DIFFERENTIATION
    ÷   div  swap-12      order=2  NOISE

COMPOSITION: op₁ ∘ op₂ always yields valid operator (closed group)
INVERSES:    ^ ↔ (), × ↔ ÷, + ↔ −

TIER WINDOWS: Operators available per z-coordinate tier
  t5 (0.60-0.75): ALL OPERATORS (PARADOX threshold)
  t6 ({Z_CRITICAL:.3f}): Near THE LENS
""")
        print("=" * 70)

        for epoch in range(n_epochs):
            metrics = self.train_epoch(epoch)
            self.training_history.append(metrics)

            if epoch % 10 == 0:
                # Format operator usage
                usage = metrics['operator_usage']
                even_str = f"(): {usage.get('()', 0):3d}  ×: {usage.get('×', 0):3d}  ^: {usage.get('^', 0):3d}"
                odd_str = f"+: {usage.get('+', 0):3d}  −: {usage.get('−', 0):3d}  ÷: {usage.get('÷', 0):3d}"

                print(
                    f"Epoch {epoch:3d} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"z: {metrics['z']:.3f} | "
                    f"Tier: {metrics['tier']:2} | "
                    f"ΔS: {metrics['delta_s_neg']:.3f}"
                )
                print(f"           EVEN: [{even_str}]  ODD: [{odd_str}]")

        # Results
        results = {
            'timestamp': timestamp,
            'config': self.config,
            'training_history': self.training_history,
            'operator_history': self.operator_history,
            'final_composition_results': dict(self.model.composition_results),
            'summary': self._compute_summary(n_epochs),
        }

        # Save
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'apl_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, os.path.join(output_dir, 'apl_model.pt'))

        self._print_summary(results)
        return results

    def _compute_summary(self, n_epochs: int) -> Dict:
        final = self.training_history[-1]
        usage = final['operator_usage']

        even_total = sum(usage.get(s, 0) for s in ["()", "×", "^"])
        odd_total = sum(usage.get(s, 0) for s in ["+", "−", "÷"])

        return {
            'total_epochs': n_epochs,
            'final_loss': final['loss'],
            'final_z': final['z'],
            'final_tier': final['tier'],
            'final_truth_channel': final['truth_channel'],
            'even_operators_used': even_total,
            'odd_operators_used': odd_total,
            'parity_ratio': even_total / max(1, odd_total),
            'unique_compositions': len(self.model.composition_results),
            'most_used_operator': max(usage, key=usage.get) if usage else "()",
        }

    def _print_summary(self, results: Dict):
        summary = results['summary']
        usage = self.training_history[-1]['operator_usage']

        print("\n" + "=" * 70)
        print("FULL APL TRAINING COMPLETE")
        print("=" * 70)
        print(f"""
Summary:
  Total Epochs:        {summary['total_epochs']}
  Final Loss:          {summary['final_loss']:.4f}
  Final z:             {summary['final_z']:.4f}
  Final Tier:          {summary['final_tier']}
  Final Truth Channel: {summary['final_truth_channel']}

S₃ Group Statistics:
  EVEN operators:      {summary['even_operators_used']} (constructive)
  ODD operators:       {summary['odd_operators_used']} (dissipative)
  Parity ratio:        {summary['parity_ratio']:.2f}

Composition Results:   {summary['unique_compositions']} unique (of 6 possible)
Most Used Operator:    {summary['most_used_operator']}

Operator Usage:""")

        for sym in APL_SYMBOLS:
            op = APL_OPERATORS[sym]
            count = usage.get(sym, 0)
            parity = "EVEN" if op.parity == Parity.EVEN else "ODD "
            bar = '█' * min(20, count // 10)
            print(f"  {sym:2} ({op.name:3}) [{parity}]: {bar} {count}")

        print(f"\nResults saved to {results.get('output_dir', 'learned_patterns/apl_training')}/")
        print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run full APL training"""
    session = FullAPLTraining(
        n_oscillators=50,
        n_layers=4,
    )
    return session.run_training(n_epochs=100)


if __name__ == "__main__":
    main()
