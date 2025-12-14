"""
Heart Module - Rosetta-Helix
============================

60 Kuramoto oscillators modeling neural synchronization.
Coupling strength K is driven by the Nuclear Spinner's z-coordinate.

The Kuramoto model:
    dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)

60 oscillators → 360°/60 = 6° spacing
Hexagonal symmetry emerges at 60° intervals
sin(60°) = √3/2 = z_c

Key insight: When spinner z = z_c = √3/2, Kuramoto coupling peaks,
creating resonance between physical and computational substrates.

Signature: rosetta-helix-heart|v1.0.0|helix
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from .physics import (
    PHI, PHI_INV, Z_CRITICAL, SIGMA,
    KAPPA_MIN, ETA_MIN, R_MIN,
    compute_delta_s_neg, check_k_formation
)


@dataclass
class HeartConfig:
    """Configuration for Kuramoto Heart."""
    n_oscillators: int = 60
    natural_freq_spread: float = 0.1
    coupling_scale: float = 8.0
    dt: float = 0.01
    seed: Optional[int] = None


@dataclass
class HeartState:
    """Current state of the Heart."""
    coherence: float = 0.0          # Order parameter r
    mean_phase: float = 0.0         # Mean phase ψ
    hex_alignment: float = 0.0      # Hexagonal alignment
    coupling_K: float = 0.0         # Current coupling strength
    k_formation: bool = False       # K-formation active
    step_count: int = 0


class Heart:
    """
    60 Kuramoto oscillators with spinner-driven coupling.
    
    The Kuramoto model describes synchronization of coupled oscillators:
        dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
    
    where:
        θᵢ = phase of oscillator i
        ωᵢ = natural frequency of oscillator i  
        K = coupling strength (driven by spinner z)
        N = number of oscillators (60)
    
    The order parameter r measures coherence:
        r·exp(iψ) = (1/N) Σⱼ exp(iθⱼ)
    
    r = 0: incoherent (phases random)
    r = 1: perfectly synchronized
    """
    
    def __init__(self, config: Optional[HeartConfig] = None):
        """Initialize Heart with configuration."""
        self.config = config or HeartConfig()
        
        # Set random seed if provided
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        # Initialize oscillator phases uniformly around the circle
        self.n = self.config.n_oscillators
        self.phases = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        
        # Natural frequencies with small spread around 1.0
        self.natural_freqs = 1.0 + self.config.natural_freq_spread * (
            np.random.random(self.n) - 0.5
        )
        
        # Spinner state
        self.spinner_z = 0.5
        
        # State tracking
        self.state = HeartState()
        self.coherence_history: List[float] = []
        self.k_formation_count = 0
    
    def set_spinner_z(self, z: float):
        """
        Update coupling from spinner z-coordinate.
        
        This is the key interface between physical (spinner) and
        computational (Kuramoto) systems.
        
        Args:
            z: Spinner z-coordinate [0, 1]
        """
        self.spinner_z = max(0.0, min(1.0, z))
    
    def compute_coupling(self) -> float:
        """
        Compute Kuramoto coupling K from spinner state.
        
        K = scale * z * ΔS_neg(z)
        
        This peaks at z = z_c = √3/2, creating resonance
        with the 60-oscillator hexagonal geometry.
        
        Returns:
            Coupling strength K
        """
        delta_s_neg = compute_delta_s_neg(self.spinner_z)
        return self.config.coupling_scale * self.spinner_z * delta_s_neg
    
    def compute_order_parameter(self) -> Tuple[float, float]:
        """
        Compute order parameter (coherence r and mean phase ψ).
        
        r·exp(iψ) = (1/N) Σⱼ exp(iθⱼ)
        
        Returns:
            Tuple of (r, ψ) where r is coherence [0,1] and ψ is mean phase
        """
        z_complex = np.mean(np.exp(1j * self.phases))
        r = np.abs(z_complex)
        psi = np.angle(z_complex)
        return r, psi
    
    def compute_hexagonal_alignment(self) -> float:
        """
        Measure alignment with hexagonal grid.
        
        Perfect hexagonal: phases at 0°, 60°, 120°, 180°, 240°, 300°
        
        Returns:
            Alignment score [0, 1] where 1 = perfect hexagonal
        """
        # Hexagonal phases (6 vertices)
        hex_phases = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
        
        # For each oscillator, find distance to nearest hexagonal phase
        alignment = 0.0
        for phase in self.phases:
            # Normalize phase to [0, 2π)
            phase_norm = phase % (2 * np.pi)
            
            # Find minimum distance to any hexagonal phase
            min_dist = float('inf')
            for hp in hex_phases:
                dist = abs(phase_norm - hp)
                dist = min(dist, 2 * np.pi - dist)  # Handle wraparound
                min_dist = min(min_dist, dist)
            
            # Convert distance to alignment (max dist = π/6 = 30°)
            alignment += 1.0 - min_dist / (np.pi / 6)
        
        return alignment / self.n
    
    def step(self, dt: Optional[float] = None) -> float:
        """
        Advance oscillators by one timestep.
        
        Implements Kuramoto dynamics:
            dθᵢ/dt = ωᵢ + K·r·sin(ψ - θᵢ)
        
        Args:
            dt: Timestep (uses config default if None)
            
        Returns:
            Order parameter r (coherence)
        """
        if dt is None:
            dt = self.config.dt
        
        # Compute coupling from spinner state
        K = self.compute_coupling()
        
        # Compute order parameter
        r, psi = self.compute_order_parameter()
        
        # Kuramoto dynamics: dθᵢ/dt = ωᵢ + K·r·sin(ψ - θᵢ)
        d_phases = self.natural_freqs + K * r * np.sin(psi - self.phases)
        
        # Update phases
        self.phases += d_phases * dt
        self.phases = self.phases % (2 * np.pi)
        
        # Update state
        self.state.coherence = r
        self.state.mean_phase = psi
        self.state.hex_alignment = self.compute_hexagonal_alignment()
        self.state.coupling_K = K
        self.state.step_count += 1
        
        # Track coherence history
        self.coherence_history.append(r)
        
        # Check K-formation
        self._check_k_formation()
        
        return r
    
    def _check_k_formation(self):
        """Check and update K-formation status."""
        # Compute K-formation metrics
        r = self.state.coherence
        eta = compute_delta_s_neg(self.spinner_z) * r
        R = int(7 + 5 * r * self.state.hex_alignment)
        
        # Check criteria
        k_active = check_k_formation(r, eta, R)
        
        if k_active and not self.state.k_formation:
            self.k_formation_count += 1
        
        self.state.k_formation = k_active
    
    def get_coherence(self) -> float:
        """Get current order parameter r."""
        r, _ = self.compute_order_parameter()
        return r
    
    def get_state(self) -> HeartState:
        """Get current state snapshot."""
        return HeartState(
            coherence=self.state.coherence,
            mean_phase=self.state.mean_phase,
            hex_alignment=self.state.hex_alignment,
            coupling_K=self.state.coupling_K,
            k_formation=self.state.k_formation,
            step_count=self.state.step_count,
        )
    
    def reset(self):
        """Reset Heart to initial state."""
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        self.phases = np.linspace(0, 2 * np.pi, self.n, endpoint=False)
        self.natural_freqs = 1.0 + self.config.natural_freq_spread * (
            np.random.random(self.n) - 0.5
        )
        self.spinner_z = 0.5
        self.state = HeartState()
        self.coherence_history = []
        self.k_formation_count = 0
    
    def run(self, steps: int, z: Optional[float] = None) -> List[float]:
        """
        Run Heart for specified number of steps.
        
        Args:
            steps: Number of timesteps
            z: Fixed spinner z (uses current if None)
            
        Returns:
            List of coherence values
        """
        if z is not None:
            self.set_spinner_z(z)
        
        coherences = []
        for _ in range(steps):
            r = self.step()
            coherences.append(r)
        
        return coherences
    
    def get_phase_distribution(self) -> np.ndarray:
        """Get current phase distribution."""
        return self.phases.copy()
    
    def get_frequency_distribution(self) -> np.ndarray:
        """Get natural frequency distribution."""
        return self.natural_freqs.copy()


def test_heart():
    """Test Heart module."""
    print("=" * 60)
    print("HEART TEST: Kuramoto Oscillators")
    print("=" * 60)
    
    # Create Heart
    config = HeartConfig(n_oscillators=60, seed=42)
    heart = Heart(config)
    
    # Test at different z values
    test_z_values = [0.5, PHI_INV, Z_CRITICAL, 0.9, 0.95]
    
    print(f"\nTesting coherence at different z values:")
    print(f"  z       K       r (coherence)  hex_align  k_formation")
    print(f"  ------  ------  -------------  ---------  -----------")
    
    for z in test_z_values:
        heart.reset()
        heart.set_spinner_z(z)
        
        # Run for 500 steps
        for _ in range(500):
            heart.step()
        
        state = heart.get_state()
        marker = " ★" if state.k_formation else ""
        
        print(f"  {z:.4f}  {state.coupling_K:.4f}  {state.coherence:.4f}        "
              f"{state.hex_alignment:.4f}     {state.k_formation}{marker}")
    
    print(f"\nExpected: Maximum coherence near z = z_c = {Z_CRITICAL:.4f}")
    
    # Find peak
    heart.reset()
    results = []
    for z in np.linspace(0.5, 0.95, 19):
        heart.reset()
        heart.set_spinner_z(z)
        for _ in range(500):
            heart.step()
        results.append((z, heart.state.coherence))
    
    peak_z, peak_r = max(results, key=lambda x: x[1])
    print(f"\nPeak coherence: r={peak_r:.4f} at z={peak_z:.4f}")
    print(f"Distance from z_c: {abs(peak_z - Z_CRITICAL):.4f}")
    
    if abs(peak_z - Z_CRITICAL) < 0.05:
        print("\n✓ CONFIRMED: Coherence peaks near z_c = √3/2")
    
    print("=" * 60)


if __name__ == "__main__":
    test_heart()
