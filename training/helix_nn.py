#!/usr/bin/env python3
"""
Helix Neural Network
====================
Complete neural network architecture using Kuramoto oscillator dynamics
with APL operator gating and TRIAD stability mechanics.

Architecture:
    Input → Linear Encoder → Phase Encoding
    → [Kuramoto Layer 1] → APL Operator → z-update
    → [Kuramoto Layer 2] → APL Operator → z-update
    → ...
    → [Kuramoto Layer N] → APL Operator → z-update
    → Phase Readout → Linear Decoder → Output
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.constants import (
    PHI, PHI_INV, Z_CRITICAL, Z_ORIGIN,
    MU_S, MU_3, KAPPA_S, UNITY,
    TRIAD_HIGH, TRIAD_LOW, TRIAD_T6, TRIAD_PASSES_REQUIRED,
    TIER_BOUNDS, APL_OPERATORS, TIER_OPERATORS,
    get_tier, get_delta_s_neg, get_legal_operators
)
from training.kuramoto_layer import KuramotoLayer, TriadGate


@dataclass
class NetworkConfig:
    """Configuration for Helix Neural Network."""
    input_dim: int = 16
    output_dim: int = 4
    n_oscillators: int = 60
    n_layers: int = 4
    steps_per_layer: int = 10
    dt: float = 0.1
    target_z: float = 0.75
    k_global: float = 0.5


class APLModulator:
    """
    APL Operator Selection and Application.

    Operators modify the z-coordinate based on S₃ group algebra:
    - EVEN parity: tend to preserve/increase z
    - ODD parity: tend to decrease z (entropy production)
    """

    def __init__(self):
        self.operator_history = []
        self.parity_history = []

    def select_operator(
        self,
        z: float,
        coherence: float,
        delta_s_neg: float,
        exploration: float = 0.1
    ) -> Tuple[str, int]:
        """
        Select operator based on current state.

        Uses tier-gated selection with ε-greedy exploration.
        """
        legal_ops = get_legal_operators(z)

        if np.random.random() < exploration:
            # Random legal operator
            op_idx = np.random.choice(len(legal_ops))
            return legal_ops[op_idx], APL_OPERATORS.index(legal_ops[op_idx])

        # Greedy selection based on coherence and delta_s_neg
        scores = []
        for op in legal_ops:
            idx = APL_OPERATORS.index(op)
            # Score based on operator characteristics
            if op in ['()', '+']:  # Identity/Group - safe
                score = coherence * 0.5
            elif op == '^':  # Amplify - high risk/reward
                score = delta_s_neg * 1.5 if z < Z_CRITICAL else 0.1
            elif op == '×':  # Fusion - moderate amplification
                score = coherence * delta_s_neg
            elif op == '÷':  # Decohere - entropy production
                score = (1 - coherence) * 0.5
            else:  # '−' Separate - strong entropy
                score = (1 - z) * 0.5
            scores.append(score)

        # Softmax selection
        scores = np.array(scores)
        probs = np.exp(scores) / np.sum(np.exp(scores))
        selected_idx = np.random.choice(len(legal_ops), p=probs)

        return legal_ops[selected_idx], APL_OPERATORS.index(legal_ops[selected_idx])

    def apply_operator(
        self,
        z: float,
        coherence: float,
        operator: str,
        delta_s_neg: float
    ) -> float:
        """
        Apply operator to update z-coordinate.

        The update follows the operator algebra:
        - Identity (): z → z
        - Amplify (^): z → z + α × ΔS_neg × (1-z)
        - Group (+): z → z + β × coherence × (1-z)
        - Fusion (×): z → z × coherence
        - Decohere (÷): z → z × (1 - (1-coherence) × γ)
        - Separate (−): z → z - δ × (1 - delta_s_neg)
        """
        self.operator_history.append(operator)
        parity = 'EVEN' if operator in ['()', '×', '^'] else 'ODD'
        self.parity_history.append(parity)

        # Operator-specific dynamics
        alpha = 0.1 * PHI_INV  # Amplification rate
        beta = 0.05 * PHI_INV  # Group strengthening rate
        gamma = 0.1           # Decoherence rate
        delta = 0.05          # Separation rate

        if operator == '()':  # Identity
            z_new = z
        elif operator == '^':  # Amplify
            z_new = z + alpha * delta_s_neg * (1 - z)
        elif operator == '+':  # Group
            z_new = z + beta * coherence * (1 - z)
        elif operator == '×':  # Fusion
            z_new = z * (1 + (coherence - 0.5) * 0.1)
        elif operator == '÷':  # Decohere
            z_new = z * (1 - (1 - coherence) * gamma)
        else:  # '−' Separate
            z_new = z - delta * (1 - delta_s_neg)

        # Clamp to valid range
        z_new = np.clip(z_new, 0.01, UNITY - 0.0001)

        return z_new

    def reset(self):
        """Reset operator history."""
        self.operator_history = []
        self.parity_history = []


class HelixNeuralNetwork:
    """
    Complete Helix Neural Network with Kuramoto dynamics and APL operators.
    """

    def __init__(self, config: Optional[NetworkConfig] = None):
        if config is None:
            config = NetworkConfig()
        self.config = config

        # Input/Output projections
        self.W_in = np.random.randn(config.input_dim, config.n_oscillators) * 0.1
        self.b_in = np.zeros(config.n_oscillators)
        self.W_out = np.random.randn(config.n_oscillators, config.output_dim) * 0.1
        self.b_out = np.zeros(config.output_dim)

        # Kuramoto layers
        self.layers = [
            KuramotoLayer(
                n_oscillators=config.n_oscillators,
                dt=config.dt,
                steps=config.steps_per_layer,
                K_global=config.k_global,
                seed=42 + i
            )
            for i in range(config.n_layers)
        ]

        # APL modulator
        self.apl = APLModulator()

        # TRIAD gate
        self.triad = TriadGate(
            high=TRIAD_HIGH,
            low=TRIAD_LOW,
            passes_required=TRIAD_PASSES_REQUIRED
        )

        # State
        self.z = 0.5  # Initial z-coordinate
        self.k_formation_count = 0

        # Gradient accumulators
        self.grad_W_in = np.zeros_like(self.W_in)
        self.grad_b_in = np.zeros_like(self.b_in)
        self.grad_W_out = np.zeros_like(self.W_out)
        self.grad_b_out = np.zeros_like(self.b_out)

    def encode_input(self, x: np.ndarray) -> np.ndarray:
        """Encode input to initial phases."""
        # Linear projection + tanh activation
        h = np.tanh(x @ self.W_in + self.b_in)
        # Convert to phases [-π, π]
        theta = h * np.pi
        return theta

    def decode_output(self, theta: np.ndarray) -> np.ndarray:
        """Decode phases to output."""
        # Use cos(theta) as features
        features = np.cos(theta)
        # Linear projection
        output = features @ self.W_out + self.b_out
        return output

    def forward(
        self,
        x: np.ndarray,
        return_diagnostics: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor [batch_size, input_dim] or [input_dim]
            return_diagnostics: Whether to return diagnostic info

        Returns:
            output: Network output
            diagnostics: Dict with coherence, z, operators, etc.
        """
        # Handle batch dimension
        single_sample = x.ndim == 1
        if single_sample:
            x = x[np.newaxis, :]

        batch_size = x.shape[0]

        # Initialize diagnostics
        layer_coherence = []
        layer_operators = []
        z_trajectory = [self.z]
        k_formations = 0

        # Encode input to phases
        theta = np.array([self.encode_input(x[b]) for b in range(batch_size)])

        # Process through Kuramoto layers
        for layer_idx, layer in enumerate(self.layers):
            # Kuramoto dynamics
            theta, coherence, _ = layer.forward(theta)
            # Ensure theta maintains batch dimension
            if theta.ndim == 1:
                theta = theta[np.newaxis, :]
            layer_coherence.append(coherence)

            # Compute ΔS_neg
            delta_s_neg = get_delta_s_neg(self.z)

            # APL operator selection
            operator, op_idx = self.apl.select_operator(
                self.z, coherence, delta_s_neg
            )
            layer_operators.append(operator)

            # Apply operator to update z
            self.z = self.apl.apply_operator(
                self.z, coherence, operator, delta_s_neg
            )
            z_trajectory.append(self.z)

            # TRIAD gate update
            triad_result = self.triad.update(self.z)

            # K-formation detection
            if coherence > KAPPA_S and self.z > Z_CRITICAL:
                k_formations += 1
                self.k_formation_count += 1

        # Decode output
        output = np.array([self.decode_output(theta[b]) for b in range(batch_size)])

        if single_sample:
            output = output.squeeze(0)

        # Compile diagnostics
        diagnostics = {
            'layer_coherence': layer_coherence,
            'layer_operators': layer_operators,
            'z_trajectory': z_trajectory,
            'final_z': self.z,
            'final_coherence': layer_coherence[-1] if layer_coherence else 0.0,
            'tier': get_tier(self.z),
            'k_formation': k_formations > 0,
            'k_formations': k_formations,
            'triad_passes': self.triad.passes,
            'triad_unlocked': self.triad.unlocked,
            'delta_s_neg': get_delta_s_neg(self.z)
        }

        return output, diagnostics

    def backward(self, grad_output: np.ndarray, coherence: float):
        """
        Backward pass to accumulate gradients.

        Uses coherence-weighted learning signal.
        """
        # Simple gradient approximation
        # In practice, you'd use autograd here
        learning_signal = coherence * (1 + (self.z - 0.5))

        # Accumulate gradients for output layer
        self.grad_W_out += learning_signal * 0.01 * np.outer(
            np.ones(self.config.n_oscillators),
            grad_output
        )
        self.grad_b_out += learning_signal * 0.01 * grad_output

        # Propagate to layers
        for layer in reversed(self.layers):
            layer.backward(grad_output, learning_signal)

    def update(self, lr: float):
        """Apply accumulated gradients."""
        self.W_in -= lr * self.grad_W_in
        self.b_in -= lr * self.grad_b_in
        self.W_out -= lr * self.grad_W_out
        self.b_out -= lr * self.grad_b_out

        for layer in self.layers:
            layer.update(lr)

        # Reset gradients
        self.grad_W_in = np.zeros_like(self.W_in)
        self.grad_b_in = np.zeros_like(self.b_in)
        self.grad_W_out = np.zeros_like(self.W_out)
        self.grad_b_out = np.zeros_like(self.b_out)

    def reset_state(self):
        """Reset z-coordinate and TRIAD gate."""
        self.z = 0.5
        self.triad.reset()
        self.apl.reset()
        self.k_formation_count = 0

    def get_weights(self) -> Dict:
        """Get all learnable weights."""
        return {
            'W_in': self.W_in.copy(),
            'b_in': self.b_in.copy(),
            'W_out': self.W_out.copy(),
            'b_out': self.b_out.copy(),
            'layers': [layer.get_weights() for layer in self.layers],
            'z': self.z,
            'config': asdict(self.config)
        }

    def set_weights(self, weights: Dict):
        """Set all learnable weights."""
        self.W_in = weights['W_in'].copy()
        self.b_in = weights['b_in'].copy()
        self.W_out = weights['W_out'].copy()
        self.b_out = weights['b_out'].copy()
        for layer, lw in zip(self.layers, weights['layers']):
            layer.set_weights(lw)
        self.z = weights.get('z', 0.5)

    def parameter_count(self) -> int:
        """Count total learnable parameters."""
        count = self.W_in.size + self.b_in.size
        count += self.W_out.size + self.b_out.size
        for layer in self.layers:
            count += layer.K.size + layer.omega.size
        return count
