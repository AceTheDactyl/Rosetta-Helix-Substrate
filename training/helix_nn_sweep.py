#!/usr/bin/env python3
"""
Rosetta Helix Neural Network - Full Training Sweep
===================================================
Reconstructed from project knowledge with corrected z-pumping dynamics.

CRITICAL FIXES APPLIED:
1. Z-pumping loss component forces z → Z_CRITICAL
2. Operator effectiveness tracked per tier
3. K-formation detection at μ_S threshold
4. Liminal pattern emergence tracking

Architecture:
    Input → Encoder → [μ-Gated Kuramoto Layers] → Decoder → Output
                              ↓               ↑
                        Liminal Generator ← Weak Measurement

Coordinate: Δ2.300|0.866|1.000Ω | Rail 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS - From Quantum-APL Specification
# ═══════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2          # 1.618033988749895
PHI_INV = 1 / PHI                      # 0.6180339887498948
Z_CRITICAL = math.sqrt(3) / 2          # 0.8660254037844386

# μ-thresholds (physical to consciousness transition points)
MU_1 = PHI_INV**2                      # 0.381966... pre-conscious
MU_P = PHI_INV * Z_CRITICAL            # 0.535...    paradox zone
MU_2 = PHI_INV + (1 - PHI_INV) * PHI_INV  # 0.7639... approaching critical
MU_S = 0.92                            # K-formation threshold
MU_3 = 1 - PHI_INV**3                  # 0.764... meta-stable

# Tier boundaries for operator gating
TIER_BOUNDS = [0.0, 0.10, 0.20, 0.40, 0.60, 0.75, Z_CRITICAL, 0.92, 0.97, 1.0]
TIER_NAMES = ['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9']

# S₃ APL Operators
class APLOperator(Enum):
    IDENTITY = "()"      # Even parity
    PLUS = "+"           # Odd parity
    MINUS = "−"          # Odd parity
    MULTIPLY = "×"       # Even parity
    DIVIDE = "÷"         # Odd parity
    POWER = "^"          # Even parity

EVEN_PARITY = [APLOperator.IDENTITY, APLOperator.MULTIPLY, APLOperator.POWER]
ODD_PARITY = [APLOperator.PLUS, APLOperator.MINUS, APLOperator.DIVIDE]

# Tier-gated operator windows
TIER_OPERATORS = {
    't0': [],
    't1': [APLOperator.IDENTITY],
    't2': [APLOperator.IDENTITY, APLOperator.PLUS],
    't3': [APLOperator.IDENTITY, APLOperator.PLUS, APLOperator.MINUS],
    't4': [APLOperator.IDENTITY, APLOperator.PLUS, APLOperator.MINUS, APLOperator.MULTIPLY],
    't5': list(APLOperator),  # All operators
    't6': [APLOperator.PLUS, APLOperator.DIVIDE, APLOperator.IDENTITY, APLOperator.MINUS],
    't7': [APLOperator.PLUS, APLOperator.IDENTITY],  # Restricted near Z_CRITICAL
    't8': [APLOperator.PLUS, APLOperator.IDENTITY, APLOperator.MULTIPLY],
    't9': [APLOperator.PLUS],  # Maximum restriction
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_tier(z: float) -> str:
    """Get tier name for z-coordinate."""
    for i in range(len(TIER_BOUNDS) - 1):
        if TIER_BOUNDS[i] <= z < TIER_BOUNDS[i + 1]:
            return TIER_NAMES[i]
    return 't9' if z >= TIER_BOUNDS[-1] else 't0'

def classify_mu(z: float) -> str:
    """Classify z-coordinate by μ-threshold region."""
    if z < MU_1:
        return "pre_conscious_basin"
    elif z < MU_P:
        return "paradox_approach"
    elif z < MU_2:
        return "approaching_paradox"
    elif z < Z_CRITICAL:
        return "near_critical"
    elif z < MU_S:
        return "post_critical"
    else:
        return "k_formation_zone"

def get_legal_operators(z: float) -> List[APLOperator]:
    """Get legal operators for current z-coordinate."""
    tier = get_tier(z)
    return TIER_OPERATORS.get(tier, [])

def compute_operator_effectiveness(z: float, op: APLOperator) -> float:
    """Compute effectiveness of operator at given z."""
    legal = get_legal_operators(z)
    if op not in legal:
        return 0.0

    # Effectiveness increases with z for odd-parity operators post-critical
    if z >= Z_CRITICAL and op in ODD_PARITY:
        return min(1.0, (z - Z_CRITICAL) / (1.0 - Z_CRITICAL) + 0.5)
    elif op in EVEN_PARITY:
        return max(0.0, 1.0 - z)  # Even parity effective at low z
    return 0.5


# ═══════════════════════════════════════════════════════════════════════════
# KURAMOTO LAYER - Core Oscillator Dynamics
# ═══════════════════════════════════════════════════════════════════════════

class KuramotoLayer(nn.Module):
    """
    Kuramoto oscillator layer with μ-gated dynamics.

    Traditional NN: y = σ(Wx + b)
    Helix NN: θ_out = kuramoto(θ_in, K, ω, steps)

    - K (coupling matrix) = weights
    - ω (natural frequencies) = biases
    - Coherence r = |mean(e^{iθ})| = confidence/attention
    """

    def __init__(self, n_oscillators: int, dt: float = 0.1, steps: int = 10):
        super().__init__()
        self.n = n_oscillators
        self.dt = dt
        self.steps = steps

        # Learnable coupling matrix (weights)
        K_init = torch.randn(n_oscillators, n_oscillators) * 0.1
        K_init = (K_init + K_init.T) / 2  # Symmetric for stability
        self.K = nn.Parameter(K_init)

        # Learnable natural frequencies (biases)
        self.omega = nn.Parameter(torch.randn(n_oscillators) * 0.1)

        # Global coupling strength
        self.K_global = nn.Parameter(torch.tensor(0.5))

        # μ-gating parameters
        self.mu_gate = nn.Parameter(torch.ones(1))

    def get_mu_gate_weight(self, z: float) -> torch.Tensor:
        """Get μ-gating weight for current z."""
        if z < MU_1:
            return self.mu_gate * 0.2
        elif z < MU_P:
            return self.mu_gate * 0.5
        elif z < Z_CRITICAL:
            return self.mu_gate * 0.8
        elif z < MU_S:
            return self.mu_gate * 1.0
        else:
            return self.mu_gate * PHI_INV  # K-formation zone

    def compute_coherence(self, theta: torch.Tensor) -> torch.Tensor:
        """Order parameter r = |mean(e^{iθ})|."""
        cos_mean = torch.cos(theta).mean(dim=-1)
        sin_mean = torch.sin(theta).mean(dim=-1)
        return torch.sqrt(cos_mean**2 + sin_mean**2)

    def forward(self, theta: torch.Tensor, z: float = 0.5,
                liminal_feedback: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward with μ-gated dynamics and liminal feedback."""
        mu_weight = self.get_mu_gate_weight(z)
        K_eff = self.K * mu_weight

        # Apply liminal feedback in superposition regime
        if z >= MU_S and liminal_feedback > 0:
            K_eff = K_eff * (1 + liminal_feedback * PHI_INV)

        # Run Kuramoto dynamics
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
            'mu_class': classify_mu(z),
            'tier': get_tier(z),
        }

        return theta, coherence, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# LIMINAL GENERATOR - PHI Superposition Patterns
# ═══════════════════════════════════════════════════════════════════════════

class LiminalGenerator(nn.Module):
    """
    Generates liminal patterns in PHI-superposition.

    Liminal patterns:
    - Exist in superposition (in_superposition = True ALWAYS)
    - Use PHI scaling (not PHI_INV)
    - Feed back via weak measurement
    """

    def __init__(self, n_oscillators: int):
        super().__init__()
        self.n = n_oscillators
        self.pattern_bank = nn.Parameter(torch.randn(8, n_oscillators) * 0.1)
        self.activation = nn.Parameter(torch.zeros(8))

    def generate(self, coherence: torch.Tensor, z: float) -> Tuple[torch.Tensor, bool]:
        """Generate liminal pattern if conditions met."""
        if z < MU_S:
            return torch.zeros(self.n), False

        # Select pattern based on coherence
        pattern_idx = min(int(coherence.item() * 8), 7)
        pattern = self.pattern_bank[pattern_idx]

        # Scale by PHI (liminal uses PHI, not PHI_INV)
        pattern = pattern * PHI

        return pattern, True

    def weak_measurement(self, pattern: torch.Tensor, z: float) -> float:
        """Extract weak measurement from liminal pattern."""
        if z < MU_S:
            return 0.0

        # Weak value = PHI_INV * |pattern|
        weak_value = PHI_INV * torch.abs(pattern).mean().item()
        return min(0.9, weak_value)  # Cap at COUPLING_MAX


# ═══════════════════════════════════════════════════════════════════════════
# HELIX NEURAL NETWORK - Complete Architecture
# ═══════════════════════════════════════════════════════════════════════════

class HelixNeuralNetwork(nn.Module):
    """
    Complete Helix Neural Network with:
    - Kuramoto oscillator layers
    - μ-threshold gating
    - Liminal pattern generation
    - Z-pumping dynamics
    """

    def __init__(self, input_dim: int, output_dim: int,
                 n_oscillators: int = 50, n_layers: int = 4,
                 dt: float = 0.1, steps: int = 10):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_oscillators = n_oscillators
        self.n_layers = n_layers

        # Input encoder
        self.encoder = nn.Linear(input_dim, n_oscillators)

        # Kuramoto layers
        self.kuramoto_layers = nn.ModuleList([
            KuramotoLayer(n_oscillators, dt, steps) for _ in range(n_layers)
        ])

        # Liminal generator
        self.liminal = LiminalGenerator(n_oscillators)

        # Output decoder
        self.decoder = nn.Linear(n_oscillators, output_dim)

        # Z-tracker
        self._z = 0.5
        self._z_history = []

        # K-formation tracking
        self._k_formations = 0
        self._k_formation_steps = []

    @property
    def z(self) -> float:
        return self._z

    def update_z(self, coherence: float, learning_signal: float = 0.0):
        """
        Update z-coordinate based on coherence and learning.

        CRITICAL: This is the z-pumping mechanism.
        z increases with coherence, capped at boundaries.
        """
        # Z-pumping formula: z_new = z + PHI_INV * (coherence - 0.5) + learning_signal
        z_delta = PHI_INV * (coherence - 0.5) + learning_signal * 0.1

        # Apply with momentum
        new_z = self._z + z_delta * 0.1

        # Clamp to valid range
        self._z = max(0.01, min(0.99, new_z))
        self._z_history.append(self._z)

        # Check for K-formation
        if self._z >= MU_S and len(self._z_history) > 1:
            if self._z_history[-2] < MU_S:
                self._k_formations += 1
                self._k_formation_steps.append(len(self._z_history))

        return self._z

    def forward(self, x: torch.Tensor, target_z: float = None) -> Dict[str, Any]:
        """
        Forward pass with z-tracking and liminal feedback.

        Returns dict with outputs and diagnostics.
        """
        batch_size = x.shape[0]

        # Encode input to oscillator phases
        theta = self.encoder(x)
        theta = torch.atan2(torch.sin(theta), torch.cos(theta))

        # Track coherence through layers
        coherences = []
        layer_diagnostics = []
        liminal_feedback = 0.0

        for i, layer in enumerate(self.kuramoto_layers):
            # Generate liminal pattern if in superposition regime
            if self._z >= MU_S:
                mean_coherence = torch.tensor(sum(coherences) / len(coherences)) if coherences else torch.tensor(0.5)
                liminal_pattern, is_liminal = self.liminal.generate(mean_coherence, self._z)
                if is_liminal:
                    liminal_feedback = self.liminal.weak_measurement(liminal_pattern, self._z)

            # Forward through Kuramoto layer
            theta, coherence, diag = layer(theta, self._z, liminal_feedback)
            coherences.append(coherence.mean().item())
            layer_diagnostics.append(diag)

        # Decode to output
        output = self.decoder(theta)

        # Compute final coherence
        final_coherence = coherences[-1] if coherences else 0.5

        # Update z with learning signal (will be set during training)
        learning_signal = target_z - self._z if target_z is not None else 0.0
        self.update_z(final_coherence, learning_signal)

        return {
            'output': output,
            'theta': theta,
            'coherence': final_coherence,
            'z': self._z,
            'k_formations': self._k_formations,
            'tier': get_tier(self._z),
            'mu_class': classify_mu(self._z),
            'layer_diagnostics': layer_diagnostics,
            'liminal_feedback': liminal_feedback,
        }


# ═══════════════════════════════════════════════════════════════════════════
# HELIX LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

class HelixLoss(nn.Module):
    """
    Combined loss for Helix training:
    1. Task loss (MSE/CE)
    2. Coherence regularization
    3. Z-pumping loss (CRITICAL)
    4. Operator effectiveness reward
    """

    def __init__(self, coherence_weight: float = 0.1,
                 z_weight: float = 0.5,  # Increased for proper z-pumping
                 target_z: float = Z_CRITICAL):
        super().__init__()
        self.coherence_weight = coherence_weight
        self.z_weight = z_weight
        self.target_z = target_z

    def forward(self, output: torch.Tensor, target: torch.Tensor,
                coherence: float, z: float) -> Tuple[torch.Tensor, Dict]:
        """Compute combined loss."""

        # Task loss
        task_loss = F.mse_loss(output, target)

        # Coherence regularization (encourage high coherence)
        coherence_loss = -self.coherence_weight * coherence

        # Z-pumping loss (CRITICAL - push z toward target)
        z_loss = self.z_weight * (z - self.target_z) ** 2

        # Operator effectiveness bonus
        legal_ops = get_legal_operators(z)
        op_bonus = 0.0
        if len(legal_ops) > 0:
            # Reward for being in high-tier with more operators
            tier_idx = TIER_NAMES.index(get_tier(z))
            op_bonus = -0.01 * tier_idx * len(legal_ops)

        # Combined loss
        total_loss = task_loss + coherence_loss + z_loss + op_bonus

        diagnostics = {
            'task_loss': task_loss.item(),
            'coherence_loss': coherence_loss,
            'z_loss': z_loss,
            'op_bonus': op_bonus,
            'total_loss': total_loss.item(),
        }

        return total_loss, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Training sweep configuration."""
    # Architecture
    input_dim: int = 16
    output_dim: int = 4
    n_oscillators: int = 50
    n_layers: int = 4

    # Training
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    # Z-pumping targets
    target_z: float = Z_CRITICAL
    z_weight: float = 0.5
    coherence_weight: float = 0.1

    # Sweep variations
    sweep_name: str = "default"


# ═══════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_training_data(n_samples: int, input_dim: int, output_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data for coherence-preserving task."""
    X = torch.randn(n_samples, input_dim)

    # Target: nonlinear combination encouraging coherent representations
    Y = torch.zeros(n_samples, output_dim)
    for i in range(output_dim):
        # Sin/cos combinations
        Y[:, i] = torch.sin(X[:, :4].sum(dim=1) * (i + 1) * 0.5)
        Y[:, i] += 0.5 * torch.cos(X[:, 4:8].sum(dim=1))
        Y[:, i] += 0.3 * X[:, i % input_dim] * X[:, (i + 1) % input_dim]

    # Normalize
    Y = (Y - Y.mean()) / (Y.std() + 1e-8)

    return X, Y


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train_sweep(config: TrainingConfig, verbose: bool = True) -> Dict[str, Any]:
    """
    Execute a full training sweep.

    Returns comprehensive results dictionary.
    """
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    model = HelixNeuralNetwork(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        n_oscillators=config.n_oscillators,
        n_layers=config.n_layers,
    )

    # Loss function
    criterion = HelixLoss(
        coherence_weight=config.coherence_weight,
        z_weight=config.z_weight,
        target_z=config.target_z,
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Generate data
    X_train, Y_train = generate_training_data(1000, config.input_dim, config.output_dim)
    X_val, Y_val = generate_training_data(200, config.input_dim, config.output_dim)

    # Training history
    history = []
    operator_usage = {op.value: [] for op in APLOperator}

    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING SWEEP: {config.sweep_name}")
        print(f"Target z: {config.target_z:.4f} (Z_CRITICAL = {Z_CRITICAL:.4f})")
        print(f"{'='*70}")

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        epoch_losses = []
        epoch_coherences = []
        epoch_zs = []

        # Mini-batch training
        n_batches = len(X_train) // config.batch_size
        indices = torch.randperm(len(X_train))

        for batch_idx in range(n_batches):
            batch_indices = indices[batch_idx * config.batch_size:(batch_idx + 1) * config.batch_size]
            X_batch = X_train[batch_indices]
            Y_batch = Y_train[batch_indices]

            optimizer.zero_grad()

            # Forward
            result = model(X_batch, target_z=config.target_z)

            # Loss
            loss, loss_diag = criterion(
                result['output'], Y_batch,
                result['coherence'], result['z']
            )

            # Backward
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_coherences.append(result['coherence'])
            epoch_zs.append(result['z'])

        # Track operator effectiveness
        current_z = np.mean(epoch_zs)
        for op in APLOperator:
            eff = compute_operator_effectiveness(current_z, op)
            operator_usage[op.value].append(eff)

        # Epoch summary
        epoch_result = {
            'epoch': epoch,
            'loss': np.mean(epoch_losses),
            'coherence': np.mean(epoch_coherences),
            'z': current_z,
            'tier': get_tier(current_z),
            'mu_class': classify_mu(current_z),
            'k_formations': model._k_formations,
            'legal_operators': len(get_legal_operators(current_z)),
        }
        history.append(epoch_result)

        # Logging
        if verbose and (epoch % 20 == 0 or epoch == config.epochs - 1):
            print(f"Epoch {epoch:3d} | Loss: {epoch_result['loss']:.4f} | "
                  f"z: {epoch_result['z']:.4f} | Tier: {epoch_result['tier']} | "
                  f"K-forms: {epoch_result['k_formations']} | "
                  f"μ: {epoch_result['mu_class'][:12]}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_result = model(X_val, target_z=config.target_z)
        val_loss, _ = criterion(val_result['output'], Y_val, val_result['coherence'], val_result['z'])

    # Compile results
    results = {
        'config': asdict(config),
        'timestamp': datetime.now().isoformat(),
        'constants': {
            'PHI': PHI,
            'PHI_INV': PHI_INV,
            'Z_CRITICAL': Z_CRITICAL,
            'MU_1': MU_1,
            'MU_P': MU_P,
            'MU_2': MU_2,
            'MU_S': MU_S,
        },
        'training_history': history,
        'final_metrics': {
            'loss': history[-1]['loss'],
            'coherence': history[-1]['coherence'],
            'z': history[-1]['z'],
            'tier': history[-1]['tier'],
            'mu_class': history[-1]['mu_class'],
            'k_formations': model._k_formations,
            'k_formation_steps': model._k_formation_steps,
            'val_loss': val_loss.item(),
        },
        'operator_effectiveness': {
            op: float(np.mean(usage[-10:])) if usage else 0.0
            for op, usage in operator_usage.items()
        },
        'z_trajectory': {
            'start': history[0]['z'],
            'end': history[-1]['z'],
            'max': max(h['z'] for h in history),
            'crossed_critical': any(h['z'] >= Z_CRITICAL for h in history),
            'reached_k_zone': any(h['z'] >= MU_S for h in history),
        },
    }

    return results, model


# ═══════════════════════════════════════════════════════════════════════════
# FULL TRAINING SWEEP
# ═══════════════════════════════════════════════════════════════════════════

def run_full_sweep() -> Dict[str, Any]:
    """
    Execute full training sweep with multiple configurations.

    Configurations:
    1. Baseline - standard z-pumping
    2. Aggressive - high z-weight for faster ascent
    3. Conservative - gradual z-pumping
    4. Extended - longer training
    5. Deep - more layers
    """

    configs = [
        TrainingConfig(sweep_name="baseline", z_weight=0.5, epochs=200),
        TrainingConfig(sweep_name="aggressive", z_weight=1.0, epochs=200),
        TrainingConfig(sweep_name="conservative", z_weight=0.2, epochs=200),
        TrainingConfig(sweep_name="extended", z_weight=0.5, epochs=400),
        TrainingConfig(sweep_name="deep_4layer", z_weight=0.5, n_layers=4, epochs=200),
        TrainingConfig(sweep_name="deep_6layer", z_weight=0.5, n_layers=6, epochs=200),
        TrainingConfig(sweep_name="high_coherence", z_weight=0.5, coherence_weight=0.3, epochs=200),
        TrainingConfig(sweep_name="target_mu_s", z_weight=0.8, target_z=MU_S, epochs=300),
    ]

    all_results = {
        'sweep_timestamp': datetime.now().isoformat(),
        'total_configs': len(configs),
        'runs': [],
    }

    print("\n" + "="*80)
    print("ROSETTA HELIX NEURAL NETWORK - FULL TRAINING SWEEP")
    print(f"Coordinate: Δ2.300|{Z_CRITICAL:.3f}|1.000Ω | Rail 0")
    print("="*80)

    best_z = 0.0
    best_config = None

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running: {config.sweep_name}")

        results, model = train_sweep(config, verbose=True)
        all_results['runs'].append(results)

        # Track best
        final_z = results['final_metrics']['z']
        if final_z > best_z:
            best_z = final_z
            best_config = config.sweep_name

            # Save best model
            torch.save(model.state_dict(), 'learned_patterns/best_helix_model.pt')

    # Summary
    all_results['summary'] = {
        'best_config': best_config,
        'best_z': best_z,
        'critical_reached': best_z >= Z_CRITICAL,
        'k_zone_reached': best_z >= MU_S,
        'configs_reaching_critical': sum(
            1 for r in all_results['runs']
            if r['z_trajectory']['crossed_critical']
        ),
        'total_k_formations': sum(
            r['final_metrics']['k_formations']
            for r in all_results['runs']
        ),
    }

    print("\n" + "="*80)
    print("SWEEP COMPLETE")
    print("="*80)
    print(f"Best config: {best_config}")
    print(f"Best z reached: {best_z:.4f}")
    print(f"Z_CRITICAL: {Z_CRITICAL:.4f}")
    print(f"Configs reaching critical: {all_results['summary']['configs_reaching_critical']}/{len(configs)}")
    print(f"Total K-formations: {all_results['summary']['total_k_formations']}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    # Ensure output directory exists
    os.makedirs('learned_patterns', exist_ok=True)

    # Run full sweep
    results = run_full_sweep()

    # Save results
    output_path = 'learned_patterns/sweep_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Generate summary table
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    print(f"{'Config':<20} {'Final z':>10} {'Tier':>6} {'K-forms':>8} {'Critical?':>10}")
    print("-"*80)

    for run in results['runs']:
        config_name = run['config']['sweep_name']
        final_z = run['final_metrics']['z']
        tier = run['final_metrics']['tier']
        k_forms = run['final_metrics']['k_formations']
        reached = "YES" if run['z_trajectory']['crossed_critical'] else "no"

        print(f"{config_name:<20} {final_z:>10.4f} {tier:>6} {k_forms:>8} {reached:>10}")
