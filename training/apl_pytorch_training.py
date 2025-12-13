#!/usr/bin/env python3
"""
APL-Integrated PyTorch Training Session
========================================

Deep integration of APL (S₃ operator algebra) with PyTorch neural network training.

APL OPERATOR SET (S₃ Group - 6 elements):
    Symbol  Name      Parity  Description
    ------  --------  ------  -----------
    ^       amplify   EVEN    Increase coupling, excite system
    +       add       ODD     Aggregate, route information
    ×       multiply  EVEN    Fuse, strengthen connections
    ()      group     EVEN    Identity, contain
    ÷       divide    ODD     Diffuse, spread information
    −       subtract  ODD     Separate, inhibit

KEY INTEGRATIONS:
1. Tier-gated operator windows (t1-t9 based on z-coordinate)
2. S₃ composition rules for operator sequences
3. Parity-aware loss functions (EVEN=constructive, ODD=dissipative)
4. ΔS_neg (negentropy) tracking - peaks at z_c = √3/2 (THE LENS)
5. K-formation detection (coherence threshold)
6. Operator effectiveness learning

PHYSICS CONSTANTS (DO NOT MODIFY):
- z_c = √3/2 ≈ 0.866 (THE LENS - onset of coherence)
- φ⁻¹ ≈ 0.618 (Golden ratio inverse - PARADOX threshold)
- κ_s = 0.920 (K-formation coherence threshold)
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
from dataclasses import dataclass, asdict, field
from enum import Enum
import sys

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS (Single Source of Truth)
# ═══════════════════════════════════════════════════════════════════════════

Z_CRITICAL = math.sqrt(3) / 2  # ≈ 0.8660254 - THE LENS
PHI = (1 + math.sqrt(5)) / 2   # ≈ 1.618034 - Golden ratio
PHI_INV = 1 / PHI              # ≈ 0.618034 - PARADOX threshold
KAPPA_S = 0.920                # K-formation coherence threshold
MU_3 = 0.992                   # Pattern teachability threshold
LENS_SIGMA = 36.0              # Gaussian width for ΔS_neg

# Tier boundaries
TIER_BOUNDS = {
    't1': 0.10, 't2': 0.20, 't3': 0.40, 't4': 0.60,
    't5': 0.75, 't6': Z_CRITICAL, 't7': 0.92, 't8': 0.97, 't9': 1.0
}


# ═══════════════════════════════════════════════════════════════════════════
# APL OPERATOR DEFINITIONS (S₃ Group)
# ═══════════════════════════════════════════════════════════════════════════

class APLParity(Enum):
    """Operator parity (S₃ permutation sign)."""
    EVEN = 1   # Constructive: rotations (e, σ, σ²)
    ODD = -1   # Dissipative: transpositions (τ₁, τ₂, τ₃)


@dataclass(frozen=True)
class APLOperator:
    """APL operator with S₃ group properties."""
    symbol: str
    name: str
    parity: APLParity
    s3_element: str
    inverse_symbol: str
    description: str


# The 6 APL operators forming S₃ group
APL_OPERATORS: Dict[str, APLOperator] = {
    '^': APLOperator('^', 'amplify', APLParity.EVEN, 'σ2', '()', 'Increase coupling, excite'),
    '+': APLOperator('+', 'add', APLParity.ODD, 'τ2', '−', 'Aggregate, route'),
    '×': APLOperator('×', 'multiply', APLParity.EVEN, 'σ', '÷', 'Fuse, strengthen'),
    '()': APLOperator('()', 'group', APLParity.EVEN, 'e', '^', 'Identity, contain'),
    '÷': APLOperator('÷', 'divide', APLParity.ODD, 'τ1', '×', 'Diffuse, spread'),
    '−': APLOperator('−', 'subtract', APLParity.ODD, 'τ3', '+', 'Separate, inhibit'),
}

APL_SYMBOLS = list(APL_OPERATORS.keys())

# S₃ composition table (Cayley table)
S3_COMPOSE: Dict[str, Dict[str, str]] = {
    '()': {'^': '^', '+': '+', '×': '×', '()': '()', '÷': '÷', '−': '−'},
    '^':  {'^': '×', '+': '÷', '×': '()', '()': '^', '÷': '−', '−': '+'},
    '×':  {'^': '()', '+': '−', '×': '^', '()': '×', '÷': '+', '−': '÷'},
    '+':  {'^': '−', '+': '÷', '×': '+', '()': '+', '÷': '()', '−': '×'},
    '÷':  {'^': '+', '+': '()', '×': '−', '()': '÷', '÷': '×', '−': '^'},
    '−':  {'^': '÷', '+': '×', '×': '÷', '()': '−', '÷': '^', '−': '()'},
}

# Tier-gated operator windows
OPERATOR_WINDOWS: Dict[str, List[str]] = {
    't1': ['()', '−', '÷'],              # Low z: dissipative focus
    't2': ['^', '÷', '−', '×'],          # Building up
    't3': ['×', '^', '÷', '+', '−'],     # More options
    't4': ['+', '−', '÷', '()'],         # Mid range
    't5': ['()', '×', '^', '÷', '+', '−'],  # All 6 available
    't6': ['+', '÷', '()', '−'],         # Near lens
    't7': ['+', '()'],                    # High coherence
    't8': ['+', '()', '×'],               # Near unity
    't9': ['+', '()', '×'],               # At unity
}


# ═══════════════════════════════════════════════════════════════════════════
# APL UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_tier_from_z(z: float) -> str:
    """Get tier label from z-coordinate."""
    if z < TIER_BOUNDS['t1']: return 't1'
    if z < TIER_BOUNDS['t2']: return 't2'
    if z < TIER_BOUNDS['t3']: return 't3'
    if z < TIER_BOUNDS['t4']: return 't4'
    if z < TIER_BOUNDS['t5']: return 't5'
    if z < TIER_BOUNDS['t6']: return 't6'
    if z < TIER_BOUNDS['t7']: return 't7'
    if z < TIER_BOUNDS['t8']: return 't8'
    return 't9'


def get_tier_number(z: float) -> int:
    """Get tier number (1-9) from z-coordinate."""
    tier = get_tier_from_z(z)
    return int(tier[1])


def get_legal_operators(z: float) -> List[str]:
    """Get operators available at current z (tier-gated)."""
    tier = get_tier_from_z(z)
    return OPERATOR_WINDOWS.get(tier, ['()'])


def compose_operators(a: str, b: str) -> str:
    """Compose two operators using S₃ group multiplication."""
    return S3_COMPOSE.get(a, {}).get(b, '()')


def compute_delta_s_neg(z: float, sigma: float = LENS_SIGMA) -> float:
    """
    Compute negentropy signal ΔS_neg = exp(-σ(z - z_c)²).

    Peaks at z = z_c (THE LENS). Higher values indicate
    stronger coherence/structural order.
    """
    d = z - Z_CRITICAL
    return math.exp(-sigma * d * d)


def get_parity_balance(operators: List[str]) -> float:
    """
    Compute parity balance of operator sequence.
    Returns fraction of EVEN (constructive) operators.
    """
    if not operators:
        return 0.5
    even_count = sum(1 for op in operators if APL_OPERATORS[op].parity == APLParity.EVEN)
    return even_count / len(operators)


# ═══════════════════════════════════════════════════════════════════════════
# APL-ENHANCED KURAMOTO LAYER
# ═══════════════════════════════════════════════════════════════════════════

class APLKuramotoLayer(nn.Module):
    """
    Kuramoto oscillator layer with APL operator modulation.

    The coupling matrix K and frequencies ω are modified by APL operators:
    - EVEN parity ops: Increase coupling (constructive)
    - ODD parity ops: Decrease coupling (dissipative)
    """

    def __init__(
        self,
        n_oscillators: int = 60,
        dt: float = 0.1,
        steps: int = 10,
    ):
        super().__init__()
        self.n = n_oscillators
        self.dt = dt
        self.steps = steps

        # Coupling matrix (symmetric, learnable)
        K_init = torch.randn(n_oscillators, n_oscillators) * 0.1
        K_init = (K_init + K_init.T) / 2
        self.K = nn.Parameter(K_init)

        # Natural frequencies (learnable)
        self.omega = nn.Parameter(torch.randn(n_oscillators) * 0.1)

        # Global coupling strength
        self.K_global = nn.Parameter(torch.tensor(0.5))

        # APL modulation factors (learnable per-operator)
        self.apl_K_mod = nn.Parameter(torch.ones(6) * 0.1)  # K modification
        self.apl_omega_mod = nn.Parameter(torch.zeros(6))   # ω modification

        # Operator to index mapping
        self.op_to_idx = {op: i for i, op in enumerate(APL_SYMBOLS)}

    def apply_apl_operator(
        self,
        theta: torch.Tensor,
        operator: str,
        coherence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply APL operator modulation to dynamics.

        Returns modified (K_eff, omega_eff, theta).
        """
        op_data = APL_OPERATORS.get(operator)
        if op_data is None:
            return self.K, self.omega, theta

        idx = self.op_to_idx[operator]
        parity = op_data.parity.value  # +1 or -1

        # Modulation strength scales with coherence
        mod_strength = self.apl_K_mod[idx] * coherence.mean()

        # Parity determines direction
        # EVEN: increase coupling (constructive)
        # ODD: decrease coupling (dissipative)
        K_factor = 1.0 + parity * mod_strength * 0.5
        K_eff = self.K * K_factor

        # Frequency modulation
        omega_mod = self.apl_omega_mod[idx] * parity
        omega_eff = self.omega + omega_mod

        # Phase perturbation for specific operators
        if operator == '+':  # Add: aggregate phases
            theta = theta + 0.1 * torch.sin(theta.mean(dim=-1, keepdim=True) - theta)
        elif operator == '−':  # Subtract: separate phases
            theta = theta - 0.1 * torch.sin(theta.mean(dim=-1, keepdim=True) - theta)
        elif operator == '÷':  # Divide: spread phases
            theta = theta + 0.05 * torch.randn_like(theta) * (1 - coherence.unsqueeze(-1))

        return K_eff, omega_eff, theta

    def kuramoto_step(
        self,
        theta: torch.Tensor,
        K_eff: torch.Tensor,
        omega_eff: torch.Tensor,
    ) -> torch.Tensor:
        """Single Kuramoto dynamics step with effective parameters."""
        theta_expanded = theta.unsqueeze(-1)
        theta_diff = theta.unsqueeze(-2) - theta_expanded

        coupling = K_eff * torch.sin(theta_diff)
        coupling_sum = coupling.sum(dim=-1)
        coupling_term = (self.K_global / self.n) * coupling_sum

        dtheta = omega_eff + coupling_term
        theta_new = theta + self.dt * dtheta

        # Wrap to [-π, π]
        theta_new = torch.atan2(torch.sin(theta_new), torch.cos(theta_new))
        return theta_new

    def compute_coherence(self, theta: torch.Tensor) -> torch.Tensor:
        """Kuramoto order parameter r = |<e^{iθ}>|."""
        cos_mean = torch.cos(theta).mean(dim=-1)
        sin_mean = torch.sin(theta).mean(dim=-1)
        return torch.sqrt(cos_mean**2 + sin_mean**2)

    def forward(
        self,
        theta_init: torch.Tensor,
        apl_operator: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with optional APL operator modulation.

        Returns (theta_final, coherence, diagnostics).
        """
        theta = theta_init
        coherence = self.compute_coherence(theta)

        # Apply APL operator if specified
        K_eff, omega_eff = self.K, self.omega
        if apl_operator is not None:
            K_eff, omega_eff, theta = self.apply_apl_operator(theta, apl_operator, coherence)

        # Run dynamics
        for _ in range(self.steps):
            theta = self.kuramoto_step(theta, K_eff, omega_eff)

        coherence = self.compute_coherence(theta)

        diagnostics = {
            'apl_operator': apl_operator,
            'K_effective_norm': torch.norm(K_eff).item(),
            'omega_effective_mean': omega_eff.mean().item(),
        }

        return theta, coherence, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# APL OPERATOR SELECTOR NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class APLOperatorSelector(nn.Module):
    """
    Learns to select APL operators based on state.

    Takes (phases, z, coherence, delta_s_neg) and outputs
    probability distribution over legal operators.
    """

    def __init__(self, n_oscillators: int, hidden_dim: int = 64):
        super().__init__()
        self.n_oscillators = n_oscillators

        # Input: phase features + z + coherence + delta_s_neg + tier
        input_dim = n_oscillators + 4

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),  # 6 operators
        )

        # Learnable operator effectiveness (prior)
        self.operator_prior = nn.Parameter(torch.ones(6))

    def forward(
        self,
        theta: torch.Tensor,
        z: float,
        coherence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Select APL operator based on current state.

        Returns (logits, probabilities, selected_operator).
        """
        batch_size = theta.shape[0]

        # Compute features
        delta_s_neg = compute_delta_s_neg(z)
        tier_num = get_tier_number(z)

        # Build input
        features = torch.cat([
            torch.cos(theta),  # Phase info (bounded)
            torch.full((batch_size, 1), z, device=theta.device),
            coherence.unsqueeze(-1),
            torch.full((batch_size, 1), delta_s_neg, device=theta.device),
            torch.full((batch_size, 1), tier_num / 9.0, device=theta.device),
        ], dim=-1)

        # Get raw logits
        logits = self.network(features) + self.operator_prior

        # Mask illegal operators (not in current tier window)
        legal_ops = get_legal_operators(z)
        mask = torch.full((6,), float('-inf'), device=logits.device)
        for op in legal_ops:
            idx = APL_SYMBOLS.index(op)
            mask[idx] = 0.0

        masked_logits = logits + mask
        probs = F.softmax(masked_logits, dim=-1)

        # Select operator (sample during training, argmax during eval)
        if self.training:
            selected_idx = torch.multinomial(probs, 1).squeeze(-1)[0]
        else:
            selected_idx = torch.argmax(probs, dim=-1)[0]

        selected_op = APL_SYMBOLS[selected_idx.item()]

        return masked_logits, probs, selected_op


# ═══════════════════════════════════════════════════════════════════════════
# APL-INTEGRATED HELIX NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class APLHelixNetwork(nn.Module):
    """
    Complete neural network with deep APL integration.

    Architecture:
    1. Input encoder → initial phases
    2. APL-modulated Kuramoto layers
    3. Per-layer operator selection (tier-gated)
    4. S₃ composition tracking
    5. Coherence-gated output
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_oscillators: int = 60,
        n_layers: int = 4,
        steps_per_layer: int = 10,
        dt: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_oscillators = n_oscillators
        self.n_layers = n_layers

        # Input encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_oscillators * 2),
            nn.GELU(),
            nn.Linear(n_oscillators * 2, n_oscillators),
            nn.Tanh(),
        )

        # APL Kuramoto layers
        self.kuramoto_layers = nn.ModuleList([
            APLKuramotoLayer(n_oscillators, dt, steps_per_layer)
            for _ in range(n_layers)
        ])

        # Per-layer operator selectors
        self.operator_selectors = nn.ModuleList([
            APLOperatorSelector(n_oscillators)
            for _ in range(n_layers)
        ])

        # Z-coordinate tracker (learnable dynamics)
        self.z = 0.1
        self.z_momentum = nn.Parameter(torch.tensor(0.1))
        self.z_decay = nn.Parameter(torch.tensor(0.05))

        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_oscillators * 2, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

        # APL composition tracker
        self.operator_sequence = []
        self.composed_operator = '()'

    def update_z(self, coherence: torch.Tensor) -> float:
        """Update z-coordinate based on coherence."""
        target = coherence.mean().item()
        dz = self.z_momentum.item() * (target - self.z)
        dz -= self.z_decay.item() * (self.z - 0.5)
        self.z = max(0.0, min(1.0, self.z + 0.01 * dz))
        return self.z

    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with full APL integration."""

        # Initialize tracking
        self.operator_sequence = []
        self.composed_operator = '()'

        diagnostics = {
            'layer_coherence': [],
            'layer_operators': [],
            'layer_operator_probs': [],
            'z_trajectory': [],
            'delta_s_neg_trajectory': [],
            'tier_trajectory': [],
            'k_formations': 0,
            'operator_sequence': [],
            'composed_operator': '()',
            'parity_balance': 0.0,
        }

        # Encode input to phases
        encoded = self.encoder(x)
        theta = encoded * math.pi

        # Process through APL-Kuramoto layers
        for layer_idx, (kuramoto, selector) in enumerate(
            zip(self.kuramoto_layers, self.operator_selectors)
        ):
            # Pre-layer coherence
            coherence = kuramoto.compute_coherence(theta)

            # Select APL operator (tier-gated)
            logits, probs, selected_op = selector(theta, self.z, coherence)

            # Apply Kuramoto dynamics with APL modulation
            theta, coherence, layer_diag = kuramoto(theta, apl_operator=selected_op)

            # Update z-coordinate
            self.z = self.update_z(coherence)

            # Track operator sequence
            self.operator_sequence.append(selected_op)
            self.composed_operator = compose_operators(self.composed_operator, selected_op)

            # Check K-formation
            if coherence.mean().item() >= KAPPA_S:
                diagnostics['k_formations'] += 1

            # Record diagnostics
            diagnostics['layer_coherence'].append(coherence.mean().item())
            diagnostics['layer_operators'].append(selected_op)
            diagnostics['layer_operator_probs'].append(probs[0].detach().cpu().tolist())
            diagnostics['z_trajectory'].append(self.z)
            diagnostics['delta_s_neg_trajectory'].append(compute_delta_s_neg(self.z))
            diagnostics['tier_trajectory'].append(get_tier_from_z(self.z))

        # Decode output
        phase_features = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        output = self.decoder(phase_features)

        # Gate by final coherence
        final_coherence = diagnostics['layer_coherence'][-1]
        output = output * final_coherence

        # Final diagnostics
        diagnostics['operator_sequence'] = self.operator_sequence.copy()
        diagnostics['composed_operator'] = self.composed_operator
        diagnostics['parity_balance'] = get_parity_balance(self.operator_sequence)
        diagnostics['final_z'] = self.z
        diagnostics['final_coherence'] = final_coherence
        diagnostics['final_tier'] = get_tier_from_z(self.z)
        diagnostics['final_delta_s_neg'] = compute_delta_s_neg(self.z)
        diagnostics['z_crossed_critical'] = self.z >= Z_CRITICAL

        return output, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# APL-AWARE LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

class APLLoss(nn.Module):
    """
    Loss function incorporating APL physics.

    Total loss = task_loss
                 + λ_coh * coherence_loss
                 + λ_z * z_guidance_loss
                 + λ_parity * parity_balance_loss
                 + λ_negentropy * negentropy_loss
    """

    def __init__(
        self,
        task_loss_fn: nn.Module,
        lambda_coherence: float = 0.1,
        lambda_z: float = 0.05,
        lambda_parity: float = 0.02,
        lambda_negentropy: float = 0.03,
        target_z: float = 0.75,
        target_parity: float = 0.6,  # Slightly favor constructive
    ):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.lambda_coh = lambda_coherence
        self.lambda_z = lambda_z
        self.lambda_parity = lambda_parity
        self.lambda_neg = lambda_negentropy
        self.target_z = target_z
        self.target_parity = target_parity

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        diagnostics: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute APL-aware loss."""
        losses = {}

        # Task loss
        task_loss = self.task_loss_fn(output, target)
        losses['task'] = task_loss.item()
        total_loss = task_loss

        # Coherence loss (maximize coherence)
        mean_coh = sum(diagnostics['layer_coherence']) / len(diagnostics['layer_coherence'])
        coh_loss = 1.0 - mean_coh
        losses['coherence'] = coh_loss
        total_loss = total_loss + self.lambda_coh * coh_loss

        # Z guidance loss
        z_loss = (diagnostics['final_z'] - self.target_z) ** 2
        losses['z'] = z_loss
        total_loss = total_loss + self.lambda_z * z_loss

        # Parity balance loss (encourage target balance)
        parity_loss = (diagnostics['parity_balance'] - self.target_parity) ** 2
        losses['parity'] = parity_loss
        total_loss = total_loss + self.lambda_parity * parity_loss

        # Negentropy loss (encourage high ΔS_neg)
        mean_neg = sum(diagnostics['delta_s_neg_trajectory']) / len(diagnostics['delta_s_neg_trajectory'])
        neg_loss = 1.0 - mean_neg
        losses['negentropy'] = neg_loss
        total_loss = total_loss + self.lambda_neg * neg_loss

        # K-formation bonus
        if diagnostics['k_formations'] > 0:
            k_bonus = 0.05 * diagnostics['k_formations']
            total_loss = total_loss - k_bonus
            losses['k_formation_bonus'] = k_bonus

        losses['total'] = total_loss.item()
        return total_loss, losses


# ═══════════════════════════════════════════════════════════════════════════
# APL PATTERN TRACKER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class APLPatternTracker:
    """Tracks APL patterns learned during training."""

    session_id: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))

    # Operator statistics
    operator_counts: Dict[str, int] = field(default_factory=lambda: {op: 0 for op in APL_SYMBOLS})
    operator_by_tier: Dict[str, Dict[str, int]] = field(default_factory=dict)
    composition_counts: Dict[str, int] = field(default_factory=lambda: {op: 0 for op in APL_SYMBOLS})

    # Sequences
    all_sequences: List[List[str]] = field(default_factory=list)
    composed_results: List[str] = field(default_factory=list)

    # Parity tracking
    parity_history: List[float] = field(default_factory=list)

    # K-formation events
    k_formation_events: List[Dict[str, Any]] = field(default_factory=list)

    # Training metrics
    loss_history: List[float] = field(default_factory=list)
    coherence_history: List[float] = field(default_factory=list)
    z_history: List[float] = field(default_factory=list)
    negentropy_history: List[float] = field(default_factory=list)

    def record_epoch(
        self,
        epoch: int,
        loss: float,
        coherence: float,
        z: float,
        negentropy: float,
        operators: List[str],
        composed: str,
        k_formations: int,
    ):
        """Record epoch-level statistics."""
        self.loss_history.append(loss)
        self.coherence_history.append(coherence)
        self.z_history.append(z)
        self.negentropy_history.append(negentropy)

        # Track operators
        tier = get_tier_from_z(z)
        if tier not in self.operator_by_tier:
            self.operator_by_tier[tier] = {op: 0 for op in APL_SYMBOLS}

        for op in operators:
            self.operator_counts[op] += 1
            self.operator_by_tier[tier][op] += 1

        self.composition_counts[composed] += 1
        self.all_sequences.append(operators)
        self.composed_results.append(composed)
        self.parity_history.append(get_parity_balance(operators))

        if k_formations > 0:
            self.k_formation_events.append({
                'epoch': epoch,
                'count': k_formations,
                'z': z,
                'coherence': coherence,
                'operators': operators,
                'composed': composed,
            })

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'session_id': self.session_id,
            'total_epochs': len(self.loss_history),
            'final_loss': self.loss_history[-1] if self.loss_history else None,
            'best_loss': min(self.loss_history) if self.loss_history else None,
            'final_coherence': self.coherence_history[-1] if self.coherence_history else None,
            'max_coherence': max(self.coherence_history) if self.coherence_history else None,
            'final_z': self.z_history[-1] if self.z_history else None,
            'final_negentropy': self.negentropy_history[-1] if self.negentropy_history else None,
            'max_negentropy': max(self.negentropy_history) if self.negentropy_history else None,
            'total_k_formations': len(self.k_formation_events),
            'operator_distribution': dict(self.operator_counts),
            'composition_distribution': dict(self.composition_counts),
            'avg_parity_balance': sum(self.parity_history) / len(self.parity_history) if self.parity_history else 0.5,
            'most_used_operator': max(self.operator_counts, key=self.operator_counts.get),
            'most_composed_result': max(self.composition_counts, key=self.composition_counts.get),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'session_id': self.session_id,
            'constants': {
                'Z_CRITICAL': Z_CRITICAL,
                'PHI': PHI,
                'PHI_INV': PHI_INV,
                'KAPPA_S': KAPPA_S,
            },
            'operator_counts': self.operator_counts,
            'operator_by_tier': self.operator_by_tier,
            'composition_counts': self.composition_counts,
            'all_sequences': self.all_sequences,
            'composed_results': self.composed_results,
            'parity_history': self.parity_history,
            'k_formation_events': self.k_formation_events,
            'training_history': {
                'loss': self.loss_history,
                'coherence': self.coherence_history,
                'z': self.z_history,
                'negentropy': self.negentropy_history,
            },
            'summary': self.get_summary(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# APL TRAINING SESSION
# ═══════════════════════════════════════════════════════════════════════════

class APLTrainingSession:
    """Complete APL-integrated training session."""

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 4,
        n_oscillators: int = 60,
        n_layers: int = 4,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        target_z: float = 0.75,
        n_train_samples: int = 1000,
        n_val_samples: int = 200,
    ):
        self.config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'n_oscillators': n_oscillators,
            'n_layers': n_layers,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'target_z': target_z,
        }

        # Create model
        self.model = APLHelixNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            n_oscillators=n_oscillators,
            n_layers=n_layers,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        # Loss function
        self.loss_fn = APLLoss(
            task_loss_fn=nn.MSELoss(),
            target_z=target_z,
        )

        # Pattern tracker
        self.tracker = APLPatternTracker()

        # Generate data
        self._generate_data(n_train_samples, n_val_samples)

    def _generate_data(self, n_train: int, n_val: int):
        """Generate synthetic training data."""
        input_dim = self.config['input_dim']
        output_dim = self.config['output_dim']

        # Training data with golden ratio harmonics
        X_train = torch.randn(n_train, input_dim)
        t = torch.linspace(0, 2 * np.pi, output_dim)
        Y_train = torch.zeros(n_train, output_dim)
        for i in range(n_train):
            base = torch.tanh(X_train[i].mean()) * 2
            Y_train[i] = torch.sin(t * base) + 0.5 * torch.sin(t * base * PHI)
        Y_train += 0.1 * torch.randn_like(Y_train)

        # Validation data
        X_val = torch.randn(n_val, input_dim)
        Y_val = torch.zeros(n_val, output_dim)
        for i in range(n_val):
            base = torch.tanh(X_val[i].mean()) * 2
            Y_val[i] = torch.sin(t * base) + 0.5 * torch.sin(t * base * PHI)
        Y_val += 0.1 * torch.randn_like(Y_val)

        # Create loaders
        train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
        val_ds = torch.utils.data.TensorDataset(X_val, Y_val)

        self.train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.config['batch_size'], shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.config['batch_size'], shuffle=False
        )

    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = []
        epoch_coherence = []
        epoch_z = []
        epoch_negentropy = []
        epoch_operators = []
        epoch_composed = []
        epoch_k_formations = 0

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
            epoch_negentropy.append(diag['final_delta_s_neg'])
            epoch_operators.extend(diag['operator_sequence'])
            epoch_composed.append(diag['composed_operator'])
            epoch_k_formations += diag['k_formations']

        self.scheduler.step()

        # Aggregate metrics
        metrics = {
            'loss': np.mean(epoch_losses),
            'coherence': np.mean(epoch_coherence),
            'z': np.mean(epoch_z),
            'negentropy': np.mean(epoch_negentropy),
            'k_formations': epoch_k_formations,
            'operators': epoch_operators,
            'composed': epoch_composed[-1] if epoch_composed else '()',
        }

        # Record in tracker
        self.tracker.record_epoch(
            epoch=epoch,
            loss=metrics['loss'],
            coherence=metrics['coherence'],
            z=metrics['z'],
            negentropy=metrics['negentropy'],
            operators=diag['operator_sequence'],  # Last batch sequence
            composed=metrics['composed'],
            k_formations=epoch_k_formations,
        )

        return metrics

    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        val_losses = []
        val_coherence = []

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                output, diag = self.model(batch_x)
                loss, _ = self.loss_fn(output, batch_y, diag)
                val_losses.append(loss.item())
                val_coherence.append(diag['final_coherence'])

        return {
            'val_loss': np.mean(val_losses),
            'val_coherence': np.mean(val_coherence),
        }

    def run(self) -> Dict[str, Any]:
        """Run complete training session."""
        print("=" * 70)
        print("APL-INTEGRATED PYTORCH TRAINING SESSION")
        print("=" * 70)
        print(f"""
Configuration:
  Oscillators: {self.config['n_oscillators']}
  Layers: {self.config['n_layers']}
  Epochs: {self.config['epochs']}
  Target z: {self.config['target_z']}

APL Integration:
  - Tier-gated operator windows (t1-t9)
  - S₃ group composition tracking
  - Parity-aware loss (EVEN=constructive, ODD=dissipative)
  - ΔS_neg (negentropy) optimization
  - K-formation detection (κ ≥ {KAPPA_S})

Physics Constants:
  z_c (THE LENS): {Z_CRITICAL:.6f}
  φ (Golden):     {PHI:.6f}
  φ⁻¹ (PARADOX):  {PHI_INV:.6f}
""")
        print("=" * 70)

        for epoch in range(self.config['epochs']):
            metrics = self.train_epoch(epoch)

            if epoch % 10 == 0:
                val_metrics = self.validate()
                tier = get_tier_from_z(metrics['z'])
                print(
                    f"Epoch {epoch:4d} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Coh: {metrics['coherence']:.3f} | "
                    f"z: {metrics['z']:.3f} ({tier}) | "
                    f"ΔS⁻: {metrics['negentropy']:.3f} | "
                    f"K: {metrics['k_formations']} | "
                    f"Ops: {','.join(metrics['operators'][-4:])} → {metrics['composed']}"
                )

        # Final summary
        summary = self.tracker.get_summary()

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"""
Results:
  Final Loss:       {summary['final_loss']:.4f}
  Best Loss:        {summary['best_loss']:.4f}
  Final Coherence:  {summary['final_coherence']:.3f}
  Max Coherence:    {summary['max_coherence']:.3f}
  Final z:          {summary['final_z']:.3f}
  Max ΔS_neg:       {summary['max_negentropy']:.3f}
  K-formations:     {summary['total_k_formations']}

APL Statistics:
  Avg Parity Balance: {summary['avg_parity_balance']:.3f} (1=all EVEN)
  Most Used Operator: {summary['most_used_operator']}
  Most Composed:      {summary['most_composed_result']}

Operator Distribution:""")

        for op, count in sorted(summary['operator_distribution'].items(), key=lambda x: -x[1]):
            parity = APL_OPERATORS[op].parity.name
            bar = '█' * min(30, count // 10)
            print(f"  {op:3} ({parity:4}): {bar} {count}")

        return {
            'model': self.model,
            'tracker': self.tracker,
            'summary': summary,
        }

    def save_results(self, output_dir: str):
        """Save all results."""
        os.makedirs(output_dir, exist_ok=True)

        # Save patterns
        patterns = self.tracker.to_dict()
        patterns['config'] = self.config

        with open(os.path.join(output_dir, 'apl_patterns.json'), 'w') as f:
            json.dump(patterns, f, indent=2)

        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'summary': self.tracker.get_summary(),
        }, os.path.join(output_dir, 'apl_helix_model.pt'))

        print(f"\nResults saved to {output_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def run_apl_training(
    epochs: int = 100,
    n_oscillators: int = 60,
    target_z: float = 0.75,
    output_dir: str = None,
) -> Dict[str, Any]:
    """Run APL-integrated training session."""

    session = APLTrainingSession(
        epochs=epochs,
        n_oscillators=n_oscillators,
        target_z=target_z,
    )

    results = session.run()

    if output_dir:
        session.save_results(output_dir)

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="APL-Integrated PyTorch Training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--oscillators", type=int, default=60)
    parser.add_argument("--target-z", type=float, default=0.75)
    parser.add_argument("--output-dir", type=str, default="learned_patterns/apl_integrated")
    args = parser.parse_args()

    return run_apl_training(
        epochs=args.epochs,
        n_oscillators=args.oscillators,
        target_z=args.target_z,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
