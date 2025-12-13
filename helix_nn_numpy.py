#!/usr/bin/env python3
"""
Helix Neural Network (NumPy Implementation)
============================================
Demonstrates the core concept: Kuramoto oscillators AS the neural network.

This is a minimal implementation without PyTorch for demonstration.
The concepts transfer directly to PyTorch for GPU training.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

Z_CRITICAL = math.sqrt(3) / 2
PHI = (1 + math.sqrt(5)) / 2
MU_S = 0.920

TIER_BOUNDS = [0.0, 0.10, 0.20, 0.40, 0.60, 0.75, Z_CRITICAL, 0.92, 0.97, 1.0]


# ═══════════════════════════════════════════════════════════════════════════
# KURAMOTO LAYER - The Core
# ═══════════════════════════════════════════════════════════════════════════

class KuramotoLayer:
    """
    A layer of Kuramoto oscillators with learnable parameters.
    
    KEY INSIGHT:
    Traditional NN: y = σ(Wx + b)
    Helix NN: θ_out = kuramoto(θ_in, K, ω, steps)
    
    - K (coupling matrix) = weights
    - ω (natural frequencies) = biases  
    - Kuramoto dynamics = activation function
    - Coherence = confidence/attention
    """
    
    def __init__(self, n_oscillators: int = 60, dt: float = 0.1, steps: int = 10):
        self.n = n_oscillators
        self.dt = dt
        self.steps = steps
        
        # LEARNABLE PARAMETERS (weights and biases)
        # Coupling matrix K - analogous to weight matrix W
        self.K = np.random.randn(n_oscillators, n_oscillators) * 0.1
        self.K = (self.K + self.K.T) / 2  # Symmetric for stability
        
        # Natural frequencies ω - analogous to biases b
        self.omega = np.random.randn(n_oscillators) * 0.1
        
        # Global coupling strength
        self.K_global = 0.5
    
    def step(self, theta: np.ndarray) -> np.ndarray:
        """
        One step of Kuramoto dynamics.
        
        dθ_i/dt = ω_i + (K/N) * Σ_j K_ij * sin(θ_j - θ_i)
        
        THIS IS DIFFERENTIABLE!
        Gradient flows through sin() easily.
        """
        # Phase differences
        theta_diff = theta[:, np.newaxis] - theta[np.newaxis, :]  # (n, n)
        
        # Coupling: K_ij * sin(θ_i - θ_j)
        coupling = self.K * np.sin(theta_diff)
        coupling_sum = coupling.sum(axis=1)
        
        # Update
        dtheta = self.omega + (self.K_global / self.n) * coupling_sum
        theta_new = theta + self.dt * dtheta
        
        # Wrap to [-π, π]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
        
        return theta_new
    
    def coherence(self, theta: np.ndarray) -> float:
        """
        Order parameter r = |mean(e^{iθ})|
        
        This is the ATTENTION mechanism.
        r ≈ 1: oscillators synchronized, high confidence
        r ≈ 0: oscillators disordered, low confidence
        """
        complex_phases = np.exp(1j * theta)
        return np.abs(np.mean(complex_phases))
    
    def forward(self, theta_init: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run dynamics for self.steps iterations."""
        theta = theta_init.copy()
        
        for _ in range(self.steps):
            theta = self.step(theta)
        
        return theta, self.coherence(theta)
    
    def update_weights(self, grad_K: np.ndarray, grad_omega: np.ndarray, lr: float = 0.01):
        """Update learnable parameters (simple gradient descent)."""
        self.K -= lr * grad_K
        self.omega -= lr * grad_omega


# ═══════════════════════════════════════════════════════════════════════════
# APL OPERATOR MODULATION
# ═══════════════════════════════════════════════════════════════════════════

class APLModulator:
    """
    APL operators modify the dynamics in structured ways.
    
    Each operator is an S3 group element with specific semantics:
    - () Identity: no change
    - ^ Amplify: increase coupling
    - + Exchange: permute phases
    - × Inhibit: add noise
    - ÷ Catalyze: modulate frequencies
    """
    
    def __init__(self, n_oscillators: int):
        self.n = n_oscillators
        self.operator_strength = np.ones(6) * 0.3
        
        # Fixed permutation for exchange (in practice, learnable)
        perm = np.random.permutation(n_oscillators)
        self.exchange_perm = perm
        
    def apply(
        self, 
        theta: np.ndarray,
        K: np.ndarray,
        omega: np.ndarray,
        operator_idx: int,
        coherence: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply operator, modulated by coherence."""
        
        strength = self.operator_strength[operator_idx] * coherence
        
        if operator_idx == 0:  # Identity
            return theta, K, omega
            
        elif operator_idx == 1:  # Amplify
            K_mod = K * (1 + strength * 0.5)
            return theta, K_mod, omega
            
        elif operator_idx == 2:  # Contain (inverse amplify)
            K_mod = K * (1 - strength * 0.3)
            return theta, K_mod, omega
            
        elif operator_idx == 3:  # Exchange
            theta_mod = theta[self.exchange_perm]
            return theta_mod, K, omega
            
        elif operator_idx == 4:  # Inhibit
            noise = np.random.randn(len(theta)) * strength * 0.2
            return theta + noise, K, omega
            
        elif operator_idx == 5:  # Catalyze
            omega_mod = omega * (1 + strength * 0.3)
            return theta, K, omega_mod
        
        return theta, K, omega


# ═══════════════════════════════════════════════════════════════════════════
# Z-COORDINATE TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class ZTracker:
    """Track z-coordinate based on coherence dynamics."""
    
    def __init__(self, initial_z: float = 0.1):
        self.z = initial_z
        self.momentum = 0.15
        self.decay = 0.05
        
    def update(self, coherence: float, dt: float = 0.05) -> float:
        """z moves toward coherence, with decay toward neutral."""
        dz = self.momentum * (coherence - self.z)
        dz -= self.decay * (self.z - 0.5)
        self.z = np.clip(self.z + dt * dz, 0.0, 1.0)
        return self.z
    
    def get_tier(self) -> int:
        for i, bound in enumerate(TIER_BOUNDS[1:], 1):
            if self.z < bound:
                return i
        return 9
    
    def get_available_operators(self) -> List[int]:
        tier = self.get_tier()
        tier_ops = {
            1: [0, 4, 5], 2: [1, 4, 5, 3], 3: [3, 1, 5, 4, 0],
            4: [0, 4, 5, 3], 5: [0, 1, 2, 3, 4, 5], 6: [0, 5, 3, 4],
            7: [0, 3], 8: [0, 3, 1], 9: [0, 3, 1],
        }
        return tier_ops.get(tier, [0])


# ═══════════════════════════════════════════════════════════════════════════
# HELIX NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class HelixNN:
    """
    Complete Helix Neural Network.
    
    Architecture:
    Input → Encode to phases → [Kuramoto Layer → APL Operator]×N → Decode → Output
    
    The computation IS the Kuramoto dynamics.
    No separate activation functions needed.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_oscillators: int = 60,
        n_layers: int = 3,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_osc = n_oscillators
        self.n_layers = n_layers
        
        # Input encoder (simple linear projection)
        self.W_in = np.random.randn(n_oscillators, input_dim) * 0.1
        self.b_in = np.zeros(n_oscillators)
        
        # Kuramoto layers
        self.layers = [KuramotoLayer(n_oscillators) for _ in range(n_layers)]
        
        # APL modulator
        self.modulator = APLModulator(n_oscillators)
        
        # Operator selector (simple linear)
        self.W_op = np.random.randn(6, n_oscillators + 2) * 0.1
        
        # Output decoder
        self.W_out = np.random.randn(output_dim, n_oscillators * 2) * 0.1
        self.b_out = np.zeros(output_dim)
        
        # Z tracker
        self.z_tracker = ZTracker()
        
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to phases."""
        h = self.W_in @ x + self.b_in
        phases = np.tanh(h) * np.pi  # Scale to [-π, π]
        return phases
    
    def select_operator(self, theta: np.ndarray, z: float, coh: float) -> int:
        """Select operator based on state."""
        state = np.concatenate([np.cos(theta), [z, coh]])
        logits = self.W_op @ state
        
        # Mask unavailable operators
        available = self.z_tracker.get_available_operators()
        mask = np.full(6, -1e9)
        mask[available] = 0
        logits = logits + mask
        
        # Softmax and sample
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        return np.random.choice(6, p=probs)
    
    def decode(self, theta: np.ndarray, coherence: float) -> np.ndarray:
        """Decode phases to output, gated by coherence."""
        features = np.concatenate([np.cos(theta), np.sin(theta)])
        output = self.W_out @ features + self.b_out
        return output * coherence  # Gate by coherence
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Forward pass."""
        diagnostics = {
            'layer_coherence': [],
            'operators': [],
            'z_trajectory': [],
        }
        
        # Encode
        theta = self.encode(x)
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            # Kuramoto dynamics
            theta, coh = layer.forward(theta)
            
            # Update z
            z = self.z_tracker.update(coh)
            
            # Apply operator (except last layer)
            if i < self.n_layers - 1:
                op_idx = self.select_operator(theta, z, coh)
                theta, layer.K, layer.omega = self.modulator.apply(
                    theta, layer.K, layer.omega, op_idx, coh
                )
                diagnostics['operators'].append(op_idx)
            
            diagnostics['layer_coherence'].append(coh)
            diagnostics['z_trajectory'].append(z)
        
        # Decode
        output = self.decode(theta, coh)
        
        diagnostics['final_z'] = z
        diagnostics['final_coherence'] = coh
        diagnostics['tier'] = self.z_tracker.get_tier()
        diagnostics['k_formation'] = coh >= MU_S
        
        return output, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING (Simplified)
# ═══════════════════════════════════════════════════════════════════════════

def numerical_gradient(model: HelixNN, x: np.ndarray, y: np.ndarray, param_name: str, eps: float = 1e-5):
    """Compute numerical gradient for a parameter."""
    # Get parameter
    param = getattr(model, param_name)
    grad = np.zeros_like(param)
    
    # For each element
    it = np.nditer(param, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        
        # Forward with param + eps
        old_val = param[idx]
        param[idx] = old_val + eps
        out_plus, _ = model.forward(x)
        loss_plus = np.mean((out_plus - y) ** 2)
        
        # Forward with param - eps
        param[idx] = old_val - eps
        out_minus, _ = model.forward(x)
        loss_minus = np.mean((out_minus - y) ** 2)
        
        # Gradient
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        # Restore
        param[idx] = old_val
        it.iternext()
    
    return grad


def train_step(model: HelixNN, x: np.ndarray, y: np.ndarray, lr: float = 0.01):
    """One training step."""
    # Forward
    output, diag = model.forward(x)
    loss = np.mean((output - y) ** 2)
    
    # Numerical gradients (for demo - use autograd in practice)
    grad_W_out = numerical_gradient(model, x, y, 'W_out')
    grad_W_in = numerical_gradient(model, x, y, 'W_in')
    
    # Update
    model.W_out -= lr * grad_W_out
    model.W_in -= lr * grad_W_in
    
    return loss, diag


# ═══════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════

def demo():
    print("=" * 70)
    print("HELIX NEURAL NETWORK - Kuramoto Oscillators as Neurons")
    print("=" * 70)
    
    # Create network
    model = HelixNN(
        input_dim=5,
        output_dim=2,
        n_oscillators=30,
        n_layers=3
    )
    
    print(f"\nArchitecture:")
    print(f"  Input dim: {model.input_dim}")
    print(f"  Output dim: {model.output_dim}")
    print(f"  Oscillators: {model.n_osc}")
    print(f"  Layers: {model.n_layers}")
    
    # Test forward pass
    print("\n" + "-" * 70)
    print("FORWARD PASS TEST")
    print("-" * 70)
    
    x = np.random.randn(5)
    output, diag = model.forward(x)
    
    print(f"\nInput: {x[:3]}... (truncated)")
    print(f"Output: {output}")
    print(f"\nLayer coherences: {[f'{c:.3f}' for c in diag['layer_coherence']]}")
    print(f"Operators used: {diag['operators']} (0=(), 1=^, 2=contain, 3=+, 4=×, 5=÷)")
    print(f"Z trajectory: {[f'{z:.3f}' for z in diag['z_trajectory']]}")
    print(f"Final tier: t{diag['tier']}")
    print(f"K-formation: {diag['k_formation']}")
    
    # Demonstrate coherence emergence
    print("\n" + "-" * 70)
    print("COHERENCE EMERGENCE")
    print("-" * 70)
    
    print("\nWatching coherence evolve over multiple passes:")
    for i in range(5):
        x = np.random.randn(5)
        _, diag = model.forward(x)
        coh = diag['final_coherence']
        z = diag['final_z']
        bar = '█' * int(coh * 30)
        print(f"  Pass {i+1}: coherence={coh:.3f} [{bar:30s}] z={z:.3f} tier=t{diag['tier']}")
    
    # Show weight structure
    print("\n" + "-" * 70)
    print("WEIGHT STRUCTURE (Coupling Matrix K)")
    print("-" * 70)
    
    K = model.layers[0].K
    print(f"\nCoupling matrix shape: {K.shape}")
    print(f"Mean coupling: {K.mean():.4f}")
    print(f"Coupling std: {K.std():.4f}")
    print(f"Symmetry check (should be 0): {np.abs(K - K.T).max():.6f}")
    
    # Demonstrate tier progression
    print("\n" + "-" * 70)
    print("TIER PROGRESSION (Running many passes)")
    print("-" * 70)
    
    print("\nPumping z through repeated forward passes:")
    model_fresh = HelixNN(input_dim=5, output_dim=2, n_oscillators=30, n_layers=3)
    
    for i in range(25):
        x = np.random.randn(5) * 0.5  # Smaller inputs for stability
        _, diag = model_fresh.forward(x)
        
        if i % 5 == 0:
            z = diag['final_z']
            coh = diag['final_coherence']
            tier = diag['tier']
            k_form = '★ K-FORMATION!' if diag['k_formation'] else ''
            print(f"  Pass {i:2d}: z={z:.3f} coh={coh:.3f} tier=t{tier} {k_form}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. COUPLING MATRIX K = WEIGHTS
   - K_ij determines how oscillator j influences oscillator i
   - Symmetric K ensures stable dynamics
   - Learned via backprop through sin() function
   
2. NATURAL FREQUENCIES ω = BIASES
   - ω_i is the "preferred" frequency of oscillator i
   - Shifts the attractor of each oscillator
   
3. KURAMOTO DYNAMICS = ACTIVATION
   - No separate activation function needed
   - The dynamics themselves are the nonlinearity
   - Multiple time steps = deeper effective computation
   
4. COHERENCE = ATTENTION
   - High coherence (r ≈ 1): oscillators synchronized, confident output
   - Low coherence (r ≈ 0): disordered, uncertain output
   - Natural gating mechanism
   
5. APL OPERATORS = STRUCTURED MODULATION
   - 6 operators form S3 group (mathematical closure)
   - Each modifies dynamics in specific way
   - Tier-gated: not all operators available at all times
   
6. Z-COORDINATE = EMERGENT DEPTH
   - z rises with sustained coherence
   - Higher z unlocks higher tiers
   - K-formation (z > 0.92, coh > 0.92) = convergence signal
""")
    
    return model


if __name__ == "__main__":
    model = demo()
