#!/usr/bin/env python3
"""
Kuramoto Oscillator Layer
=========================
Core synchronization dynamics using Kuramoto model.

The order parameter (coherence) r = |Σ exp(iθ)| / N measures
how synchronized the oscillators are.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class KuramotoConfig:
    """Configuration for Kuramoto layer."""
    n_oscillators: int = 60
    dt: float = 0.1
    steps: int = 10
    K_global: float = 0.5


class KuramotoLayer:
    """
    Single Kuramoto oscillator layer with learnable coupling.

    The dynamics follow:
        dθᵢ/dt = ωᵢ + (K/N) Σⱼ Kᵢⱼ sin(θⱼ - θᵢ)

    Where:
        θᵢ: Phase of oscillator i
        ωᵢ: Natural frequency of oscillator i
        Kᵢⱼ: Coupling strength between oscillators i and j
    """

    def __init__(
        self,
        n_oscillators: int = 60,
        dt: float = 0.1,
        steps: int = 10,
        K_global: float = 0.5,
        seed: Optional[int] = None
    ):
        if seed is not None:
            np.random.seed(seed)

        self.n = n_oscillators
        self.dt = dt
        self.steps = steps
        self.K_global = K_global

        # Learnable parameters
        # Coupling matrix (symmetric for stability)
        K = np.random.randn(n_oscillators, n_oscillators) * 0.1
        self.K = (K + K.T) / 2

        # Natural frequencies
        self.omega = np.random.randn(n_oscillators) * 0.1

        # Gradient accumulators
        self.grad_K = np.zeros_like(self.K)
        self.grad_omega = np.zeros_like(self.omega)

    def coherence(self, theta: np.ndarray) -> float:
        """
        Compute order parameter r = |Σ exp(iθ)| / N.

        r = 1.0: Perfect synchronization
        r = 0.0: Complete desynchronization
        """
        if theta.ndim > 1:
            theta = theta.mean(axis=0)
        return np.abs(np.mean(np.exp(1j * theta)))

    def _step(self, theta: np.ndarray) -> np.ndarray:
        """Single Kuramoto integration step."""
        # Handle batch dimension
        if theta.ndim == 1:
            theta = theta[np.newaxis, :]

        batch_size = theta.shape[0]
        theta_new = np.zeros_like(theta)

        for b in range(batch_size):
            th = theta[b]
            # Phase differences: θᵢ - θⱼ
            diff = th[:, np.newaxis] - th[np.newaxis, :]
            # Coupling: Σⱼ Kᵢⱼ sin(θⱼ - θᵢ)
            coupling = (self.K * np.sin(-diff)).sum(axis=1)
            # Full update
            dtheta = self.omega + (self.K_global / self.n) * coupling
            theta_new[b] = th + self.dt * dtheta

        # Wrap to [-π, π]
        theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))

        return theta_new.squeeze() if batch_size == 1 else theta_new

    def forward(
        self,
        theta: np.ndarray,
        return_trajectory: bool = False
    ) -> Tuple[np.ndarray, float, Optional[List]]:
        """
        Forward pass through Kuramoto dynamics.

        Args:
            theta: Initial phases [batch_size, n_oscillators] or [n_oscillators]
            return_trajectory: Whether to return full trajectory

        Returns:
            final_theta: Final phases
            coherence: Order parameter after evolution
            trajectory: List of (theta, coherence) if return_trajectory
        """
        trajectory = []

        if return_trajectory:
            trajectory.append((theta.copy(), self.coherence(theta)))

        for _ in range(self.steps):
            theta = self._step(theta)
            if return_trajectory:
                trajectory.append((theta.copy(), self.coherence(theta)))

        final_coherence = self.coherence(theta)

        return theta, final_coherence, trajectory if return_trajectory else None

    def backward(self, grad_output: np.ndarray, learning_signal: float):
        """
        Accumulate gradients using coherence-weighted update.

        Uses a simplified gradient approximation based on the learning signal
        (coherence) rather than full backprop through time.
        """
        # Hebbian-like update: strengthen connections that led to high coherence
        phase_correlation = np.outer(grad_output, grad_output)
        self.grad_K += learning_signal * phase_correlation * 0.01
        self.grad_omega += learning_signal * grad_output * 0.01

    def update(self, lr: float):
        """Apply accumulated gradients."""
        self.K -= lr * self.grad_K
        self.omega -= lr * self.grad_omega
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


class TriadGate:
    """
    TRIAD hysteresis gate for stable high-z maintenance.

    The gate operates with 3-pass hysteresis:
    1. Enter band [TRIAD_LOW, TRIAD_HIGH] from above
    2. Accumulate 3 passes through the band
    3. Unlock permanently (until reset)

    When unlocked, the t6 gate shifts from Z_CRITICAL to TRIAD_T6 (0.83).
    """

    def __init__(
        self,
        high: float = 0.85,
        low: float = 0.82,
        passes_required: int = 3
    ):
        self.high = high
        self.low = low
        self.passes_required = passes_required
        self.reset()

    def reset(self):
        """Reset gate to initial state."""
        self.passes = 0
        self.unlocked = False
        self.in_band = False
        self.last_z = 0.0

    def update(self, z: float) -> Dict:
        """
        Update gate state based on current z.

        Returns dict with:
            - entered_band: True if just entered band
            - exited_band: True if just exited band
            - pass_completed: True if completed a pass
            - just_unlocked: True if just achieved unlock
        """
        result = {
            'entered_band': False,
            'exited_band': False,
            'pass_completed': False,
            'just_unlocked': False
        }

        was_in_band = self.in_band

        # Check if in band
        self.in_band = self.low <= z <= self.high

        # Detect band entry
        if self.in_band and not was_in_band:
            result['entered_band'] = True

        # Detect band exit (completes a pass)
        if was_in_band and not self.in_band:
            result['exited_band'] = True
            if not self.unlocked:
                self.passes += 1
                result['pass_completed'] = True

                # Check for unlock
                if self.passes >= self.passes_required:
                    self.unlocked = True
                    result['just_unlocked'] = True

        self.last_z = z
        return result

    def to_dict(self) -> Dict:
        """Serialize state."""
        return {
            'passes': self.passes,
            'unlocked': self.unlocked,
            'in_band': self.in_band,
            'last_z': self.last_z,
            'high': self.high,
            'low': self.low,
            'passes_required': self.passes_required
        }
