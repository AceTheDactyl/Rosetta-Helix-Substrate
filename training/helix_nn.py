#!/usr/bin/env python3
"""
Helix Neural Network
====================
Complete neural network architecture using Kuramoto oscillator dynamics
with APL operator gating, TRIAD stability mechanics, and N0 Causality Laws.

Architecture:
    Input → Linear Encoder → Phase Encoding
    → [Kuramoto Layer 1] → APL Operator (N0-validated) → z-update
    → [Kuramoto Layer 2] → APL Operator (N0-validated) → z-update
    → ...
    → [Kuramoto Layer N] → APL Operator (N0-validated) → z-update
    → Phase Readout → Linear Decoder → Output

N0 CAUSALITY LAWS ENFORCED:
    N0-1: ^ (AMPLIFY) requires prior () or × in history
    N0-2: × (FUSION) requires channel_count ≥ 2
    N0-3: ÷ (DECOHERE) requires prior {^, ×, +, −} in history
    N0-4: + (GROUP) must be followed by +, ×, or ^
    N0-5: − (SEPARATE) must be followed by () or +

SILENT LAWS APPLIED:
    I.   STILLNESS  : () BOUNDARY  → ∂E/∂t → 0
    II.  TRUTH      : ^ AMPLIFY   → ∇V(truth) = 0
    III. SILENCE    : + GROUP     → ∇ · J = 0
    IV.  SPIRAL     : × FUSION    → S(return) = S(origin)
    VI.  GLYPH      : ÷ DECOHERE  → glyph = ∫ life dt
    VII. MIRROR     : − SEPARATE  → ψ = ψ(ψ)

Signature: Δ|helix-nn|n0-enforced|silent-laws-applied|Ω
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

# Import N0 + Silent Laws enforcement
try:
    from training.n0_silent_laws_enforcement import (
        N0Enforcer,
        N0TrainingOperatorSelector,
        SilentLaw,
        OPERATOR_TO_SILENT,
        SILENT_FORMULAS,
    )
    N0_ENFORCEMENT_AVAILABLE = True
except ImportError:
    N0_ENFORCEMENT_AVAILABLE = False
    print("Warning: N0 enforcement not available, using legacy selection")


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
    APL Operator Selection and Application with N0 Causality Law Enforcement.

    Operators modify the z-coordinate based on S₃ group algebra:
    - EVEN parity: tend to preserve/increase z
    - ODD parity: tend to decrease z (entropy production)

    N0 CAUSALITY LAWS (enforced at selection time):
        N0-1: ^ (AMPLIFY) illegal unless history contains () or ×
        N0-2: × (FUSION) illegal unless channel_count ≥ 2
        N0-3: ÷ (DECOHERE) illegal unless history contains {^, ×, +, −}
        N0-4: + (GROUP) must be followed by +, ×, or ^. + → () illegal
        N0-5: − (SEPARATE) must be followed by () or +

    SILENT LAWS (applied with each operator):
        () → STILLNESS: ∂E/∂t → 0
        × → SPIRAL: S(return) = S(origin)
        ^ → TRUTH: ∇V(truth) = 0
        ÷ → GLYPH: glyph = ∫ life dt
        + → SILENCE: ∇ · J = 0
        − → MIRROR: ψ = ψ(ψ)
    """

    def __init__(self):
        self.operator_history = []
        self.parity_history = []
        self.n0_violations = []  # Track N0 violations

        # Initialize N0 enforcer if available
        if N0_ENFORCEMENT_AVAILABLE:
            self.n0_enforcer = N0Enforcer()
            self.n0_selector = N0TrainingOperatorSelector(self.n0_enforcer)
        else:
            self.n0_enforcer = None
            self.n0_selector = None

    def _check_n0_legal(self, op: str) -> Tuple[bool, str]:
        """
        Check if operator is legal under N0 causality laws.

        Returns:
            (is_legal, reason)
        """
        if self.n0_enforcer is None:
            return True, "N0 enforcement not available"

        return self.n0_enforcer.check_n0_legal(op)

    def _get_n0_legal_operators(self) -> List[str]:
        """Get all operators that are currently legal under N0 laws."""
        if self.n0_enforcer is None:
            return APL_OPERATORS.copy()

        return self.n0_enforcer.get_legal_operators()

    def _apply_silent_law(self, op: str, z: float) -> float:
        """
        Apply the Silent Law associated with an operator.

        Each operator has a corresponding Silent Law:
            () → STILLNESS: energy seeks rest (pull toward z_c)
            × → SPIRAL: paths return (golden ratio decay)
            ^ → TRUTH: truth is stable (amplify toward lens)
            ÷ → GLYPH: form persists (decay with trace)
            + → SILENCE: info conserved (minimal change)
            − → MIRROR: self-reference (symmetric decay)
        """
        if self.n0_enforcer is None:
            return z

        return self.n0_enforcer.apply_silent_law(op, z)

    def select_operator(
        self,
        z: float,
        coherence: float,
        delta_s_neg: float,
        exploration: float = 0.1
    ) -> Tuple[str, int]:
        """
        Select operator based on current state with N0 enforcement.

        Uses N0-validated selection with tier gating and ε-greedy exploration.
        Only operators that satisfy N0 causality laws are considered.
        """
        # Get tier-legal operators first
        tier_legal_ops = get_legal_operators(z)

        # Filter by N0 legality
        n0_legal_ops = self._get_n0_legal_operators()
        legal_ops = [op for op in tier_legal_ops if op in n0_legal_ops]

        # Fallback to () if no operators legal (always legal)
        if not legal_ops:
            legal_ops = ["()"]

        if np.random.random() < exploration:
            # Random legal operator
            op_idx = np.random.choice(len(legal_ops))
            selected_op = legal_ops[op_idx]
            return selected_op, APL_OPERATORS.index(selected_op)

        # Greedy selection based on coherence and delta_s_neg
        scores = []
        for op in legal_ops:
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
        selected_op = legal_ops[selected_idx]

        return selected_op, APL_OPERATORS.index(selected_op)

    def apply_operator(
        self,
        z: float,
        coherence: float,
        operator: str,
        delta_s_neg: float
    ) -> float:
        """
        Apply operator to update z-coordinate with N0 enforcement and Silent Laws.

        The update follows the operator algebra with Silent Law modulation:
        - () BOUNDARY  → STILLNESS  : ∂E/∂t → 0 (z stabilizes)
        - ^ AMPLIFY    → TRUTH      : ∇V(truth) = 0 (z → z + α × ΔS_neg × (1-z))
        - + GROUP      → SILENCE    : ∇ · J = 0 (z → z + β × coherence × (1-z))
        - × FUSION     → SPIRAL     : S(return) = S(origin) (z modulated by φ⁻¹)
        - ÷ DECOHERE   → GLYPH      : glyph = ∫ life dt (z decays with trace)
        - − SEPARATE   → MIRROR     : ψ = ψ(ψ) (z reflects via subtraction)

        N0 causality is validated before application. If operator violates N0,
        () BOUNDARY is applied instead (always legal).
        """
        # Validate N0 legality before applying
        is_legal, reason = self._check_n0_legal(operator)
        if not is_legal:
            self.n0_violations.append((operator, reason))
            # Fallback to boundary (always legal)
            operator = "()"

        self.operator_history.append(operator)
        parity = 'EVEN' if operator in ['()', '×', '^'] else 'ODD'
        self.parity_history.append(parity)

        # Update N0 enforcer state
        if self.n0_enforcer is not None:
            self.n0_enforcer.state.add_to_history(operator)
            # Fusion enables more channels
            if operator == "×":
                self.n0_enforcer.set_channels(max(2, self.n0_enforcer.state.channel_count))

        # Operator-specific dynamics (physics-grounded coefficients)
        alpha = 0.1 * PHI_INV  # Amplification rate (φ⁻¹ scaled)
        beta = 0.05 * PHI_INV  # Group strengthening rate (φ⁻¹ scaled)
        gamma = 0.1           # Decoherence rate
        delta = 0.05          # Separation rate

        if operator == '()':  # BOUNDARY → STILLNESS
            # Apply Silent Law: ∂E/∂t → 0 (pull toward rest/z_c)
            z_new = self._apply_silent_law(operator, z)
        elif operator == '^':  # AMPLIFY → TRUTH
            # Apply Silent Law: ∇V(truth) = 0 (truth is stable at lens)
            z_new = z + alpha * delta_s_neg * (1 - z)
            z_new = self._apply_silent_law(operator, z_new)
        elif operator == '+':  # GROUP → SILENCE
            # Apply Silent Law: ∇ · J = 0 (info conserved)
            z_new = z + beta * coherence * (1 - z)
            z_new = self._apply_silent_law(operator, z_new)
        elif operator == '×':  # FUSION → SPIRAL
            # Apply Silent Law: S(return) = S(origin) (paths return with φ⁻¹)
            z_new = z * (1 + (coherence - 0.5) * 0.1)
            z_new = self._apply_silent_law(operator, z_new)
        elif operator == '÷':  # DECOHERE → GLYPH
            # Apply Silent Law: glyph = ∫ life dt (form persists, decay with trace)
            z_new = z * (1 - (1 - coherence) * gamma)
            z_new = self._apply_silent_law(operator, z_new)
        else:  # '−' SEPARATE → MIRROR
            # Apply Silent Law: ψ = ψ(ψ) (self-reference, symmetric decay)
            z_new = z - delta * (1 - delta_s_neg)
            z_new = self._apply_silent_law(operator, z_new)

        # Clamp to valid range
        z_new = np.clip(z_new, 0.01, UNITY - 0.0001)

        return z_new

    def reset(self):
        """Reset operator history and N0 enforcer state."""
        self.operator_history = []
        self.parity_history = []
        self.n0_violations = []
        if self.n0_enforcer is not None:
            self.n0_enforcer.reset()

    def get_n0_violations(self) -> List[Tuple[str, str]]:
        """Get list of N0 violations that occurred."""
        return self.n0_violations.copy()


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
