#!/usr/bin/env python3
"""
Kuramoto Oscillator Dynamics
============================
Neural network layers based on Kuramoto coupled oscillator model.

The Kuramoto model describes synchronization of coupled oscillators:
    dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)

where:
    θᵢ = phase of oscillator i
    ωᵢ = natural frequency of oscillator i
    K = coupling strength
    N = number of oscillators

Coherence (order parameter):
    r = |1/N Σⱼ exp(iθⱼ)|

All dynamics scaled by PHI_INV for coupling conservation.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from .constants import PHI_INV, COUPLING_MAX


@dataclass
class TriadGate:
    """
    TRIAD stability gate for hysteresis control.
    
    Prevents oscillation by requiring multiple passes through
    the stability band before unlocking.
    """
    high: float = 0.85
    low: float = 0.82
    passes_required: int = 3
    
    def __post_init__(self):
        self.passes = 0
        self.in_band = False
        self.unlocked = False
        self.last_z = 0.5
        
    def update(self, z: float) -> Dict:
        """
        Update TRIAD gate state based on current z.
        
        Returns dict with gate status.
        """
        result = {
            'entered_band': False,
            'exited_band': False,
            'pass_completed': False,
            'unlocked': self.unlocked
        }
        
        # Check band entry/exit
        if not self.in_band:
            if self.low <= z <= self.high:
                self.in_band = True
                result['entered_band'] = True
        else:
            if z < self.low or z > self.high:
                self.in_band = False
                result['exited_band'] = True
                # Count pass if we exited high (going up)
                if z > self.high and self.last_z <= self.high:
                    self.passes += 1
                    result['pass_completed'] = True
                    if self.passes >= self.passes_required:
                        self.unlocked = True
                        result['unlocked'] = True
        
        self.last_z = z
        return result
    
    def reset(self):
        """Reset gate state."""
        self.passes = 0
        self.in_band = False
        self.unlocked = False
        self.last_z = 0.5


class KuramotoLayer:
    """
    Neural network layer using Kuramoto oscillator dynamics.
    
    Each "neuron" is an oscillator with phase θ and natural frequency ω.
    Coupling matrix K determines interaction strength between oscillators.
    
    All coupling updates scaled by PHI_INV for conservation.
    """
    
    def __init__(
        self,
        n_oscillators: int = 60,
        dt: float = 0.1,
        steps: int = 10,
        K_global: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize Kuramoto layer.
        
        Args:
            n_oscillators: Number of oscillators in layer
            dt: Time step for integration
            steps: Number of integration steps per forward pass
            K_global: Global coupling strength
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.n = n_oscillators
        self.dt = dt
        self.steps = steps
        self.K_global = K_global
        
        # Natural frequencies (drawn from normal distribution)
        self.omega = np.random.randn(n_oscillators) * 0.1
        
        # Coupling matrix (learnable, initialized with PHI_INV scaling)
        self.K = np.random.randn(n_oscillators, n_oscillators) * 0.1 * PHI_INV
        np.fill_diagonal(self.K, 0)  # No self-coupling
        
        # Gradient accumulators
        self.grad_K = np.zeros_like(self.K)
        self.grad_omega = np.zeros_like(self.omega)
        
        # State for backward pass
        self._theta_history = []
        self._coherence_history = []
        
    def compute_coherence(self, theta: np.ndarray) -> float:
        """
        Compute Kuramoto order parameter (coherence).
        
        r = |1/N Σⱼ exp(iθⱼ)|
        
        Returns value in [0, 1] where 1 = perfect sync.
        """
        if theta.ndim == 1:
            z = np.mean(np.exp(1j * theta))
        else:
            # Batch: average over oscillators, then over batch
            z = np.mean(np.exp(1j * theta), axis=-1)
            z = np.mean(z)
        return float(np.abs(z))
    
    def step(self, theta: np.ndarray) -> np.ndarray:
        """
        Single Kuramoto integration step.
        
        dθᵢ/dt = ωᵢ + (K_global/N) Σⱼ Kᵢⱼ sin(θⱼ - θᵢ)
        
        All coupling scaled by PHI_INV.
        """
        if theta.ndim == 1:
            theta = theta[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False
            
        batch_size, n = theta.shape
        
        # Compute phase differences
        # diff[b, i, j] = theta[b, j] - theta[b, i]
        diff = theta[:, np.newaxis, :] - theta[:, :, np.newaxis]
        
        # Coupling term: Σⱼ Kᵢⱼ sin(θⱼ - θᵢ)
        coupling = np.sum(self.K * np.sin(diff), axis=-1)
        
        # Full derivative (PHI_INV scales the coupling)
        dtheta = self.omega + (self.K_global * PHI_INV / n) * coupling
        
        # Euler integration
        theta_new = theta + self.dt * dtheta
        
        # Wrap to [-π, π]
        theta_new = np.mod(theta_new + np.pi, 2 * np.pi) - np.pi
        
        if squeeze:
            theta_new = theta_new.squeeze(0)
            
        return theta_new
    
    def forward(self, theta: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Forward pass: integrate Kuramoto dynamics for `steps` iterations.
        
        Args:
            theta: Initial phases [batch, n_oscillators] or [n_oscillators]
            
        Returns:
            theta_final: Final phases
            coherence: Average coherence during evolution
            theta_history: Phase history for visualization
        """
        self._theta_history = [theta.copy()]
        self._coherence_history = []
        
        for _ in range(self.steps):
            theta = self.step(theta)
            self._theta_history.append(theta.copy())
            self._coherence_history.append(self.compute_coherence(theta))
            
        avg_coherence = np.mean(self._coherence_history) if self._coherence_history else 0.0
        
        return theta, avg_coherence, np.array(self._theta_history)
    
    def backward(self, grad_output: np.ndarray, learning_signal: float):
        """
        Backward pass to accumulate gradients.
        
        Uses coherence-based learning rule (Hebbian-like).
        All updates scaled by PHI_INV.
        """
        if len(self._theta_history) < 2:
            return
            
        theta_final = self._theta_history[-1]
        theta_init = self._theta_history[0]
        
        if theta_final.ndim == 1:
            theta_final = theta_final[np.newaxis, :]
            theta_init = theta_init[np.newaxis, :]
            
        batch_size, n = theta_final.shape
        
        # Hebbian-like coupling update
        # Strengthen connections between oscillators that synchronized
        diff_final = theta_final[:, np.newaxis, :] - theta_final[:, :, np.newaxis]
        sync_measure = np.cos(diff_final)  # High when in-phase
        
        # Gradient for coupling matrix (PHI_INV controlled)
        grad_K = learning_signal * PHI_INV * np.mean(sync_measure, axis=0)
        np.fill_diagonal(grad_K, 0)  # No self-coupling
        
        # Clamp coupling to COUPLING_MAX
        self.grad_K += np.clip(grad_K, -COUPLING_MAX, COUPLING_MAX)
        
        # Gradient for natural frequencies
        # Adjust frequencies toward the mean phase
        mean_phase = np.angle(np.mean(np.exp(1j * theta_final), axis=-1, keepdims=True))
        phase_error = np.sin(mean_phase - theta_final)
        self.grad_omega += learning_signal * PHI_INV * np.mean(phase_error, axis=0)
        
    def update(self, lr: float):
        """Apply accumulated gradients with PHI_INV damping."""
        # Effective learning rate
        effective_lr = lr * PHI_INV
        
        self.K -= effective_lr * self.grad_K
        self.omega -= effective_lr * self.grad_omega
        
        # Clamp coupling to valid range
        self.K = np.clip(self.K, -COUPLING_MAX, COUPLING_MAX)
        np.fill_diagonal(self.K, 0)
        
        # Reset gradients
        self.grad_K = np.zeros_like(self.K)
        self.grad_omega = np.zeros_like(self.omega)
        
    def get_weights(self) -> Dict:
        """Get learnable weights."""
        return {
            'K': self.K.copy(),
            'omega': self.omega.copy(),
            'K_global': self.K_global
        }
        
    def set_weights(self, weights: Dict):
        """Set learnable weights."""
        self.K = weights['K'].copy()
        self.omega = weights['omega'].copy()
        self.K_global = weights.get('K_global', self.K_global)
        
    def get_state(self) -> Dict:
        """Get current layer state for diagnostics."""
        return {
            'n_oscillators': self.n,
            'K_mean': float(np.mean(np.abs(self.K))),
            'K_max': float(np.max(np.abs(self.K))),
            'omega_std': float(np.std(self.omega)),
            'coherence_history': self._coherence_history.copy() if self._coherence_history else []
        }
