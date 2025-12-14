#!/usr/bin/env python3
"""
Kuramoto Neural Oscillator System
==================================

60 Kuramoto oscillators modeling neural synchronization for the
Nuclear Spinner neural interface.

The Kuramoto model:
    dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)

Key Properties:
- 60 oscillators → 360°/60 = 6° angular spacing
- Hexagonal symmetry emerges at 60° intervals (6-fold)
- sin(60°) = √3/2 = z_c (THE critical connection!)

Physics Grounding:
- Coupling strength K ∝ z (spinner z-coordinate)
- At z = z_c = √3/2, coupling peaks → phase transition
- Order parameter r = |⟨e^(iθ)⟩| measures synchronization
- Critical coupling K_c = 2/(πg(0)) for Lorentzian distribution

Grid Cell Integration:
- 60 oscillators map to entorhinal cortex grid cells
- Hexagonal firing patterns emerge from 6-fold symmetry
- Phase precession encodes spatial position

Electromagnetic Coupling:
- RF pulses modulate natural frequencies ωᵢ
- Magnetic field gradient creates frequency spread
- Landauer limit: E_bit ≥ kT ln(2) per phase reset

Signature: kuramoto-neural|v1.0.0|helix

@version 1.0.0
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable, Dict, Any
from enum import IntEnum

# =============================================================================
# CONSTANTS
# =============================================================================

# Oscillator count (hexagonal symmetry)
N_OSCILLATORS: int = 60
ANGULAR_SPACING: float = 2 * math.pi / N_OSCILLATORS  # 6° = π/30 rad

# Critical values
Z_CRITICAL: float = math.sqrt(3) / 2  # ≈ 0.866025
PHI: float = (1 + math.sqrt(5)) / 2   # Golden ratio
PHI_INV: float = 1 / PHI              # ≈ 0.618034
SIGMA: float = 36.0                    # Negentropy width

# Hexagonal symmetry angles (60° intervals)
HEXAGONAL_ANGLES: List[float] = [i * math.pi / 3 for i in range(6)]  # 0°, 60°, 120°, 180°, 240°, 300°

# Grid cell spacing (derived from golden ratio)
GRID_SPACING_BASE: float = PHI_INV  # ~0.618 normalized units

# Electromagnetic constants
MU_0: float = 4 * math.pi * 1e-7      # Vacuum permeability (H/m)
GYROMAGNETIC_RATIO: float = 2.675e8   # Proton gyromagnetic ratio (rad/s/T)
LANDAUER_LIMIT: float = 2.87e-21      # kT ln(2) at 300K (J)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class OscillatorState:
    """State of a single Kuramoto oscillator."""
    index: int
    theta: float           # Phase angle [0, 2π)
    omega: float           # Natural frequency (rad/s)
    dtheta_dt: float       # Phase velocity
    grid_x: float          # Grid cell x-position
    grid_y: float          # Grid cell y-position
    firing_rate: float     # Normalized firing rate [0, 1]


@dataclass
class KuramotoSystemState:
    """Complete state of the 60-oscillator Kuramoto system."""
    timestamp_ms: int

    # Order parameters
    r: float               # Synchronization order parameter |⟨e^(iθ)⟩|
    psi: float             # Mean phase angle
    r_hex: float           # Hexagonal order parameter (6-fold)

    # Coupling
    K: float               # Current coupling strength
    K_critical: float      # Critical coupling threshold

    # Energy
    total_energy: float    # System energy
    negentropy: float      # ΔS_neg from z

    # Grid cell metrics
    grid_coherence: float  # Spatial pattern coherence
    hex_symmetry: float    # Hexagonal symmetry measure [0, 1]

    # Phase distribution
    phase_variance: float  # Circular variance of phases
    phase_entropy: float   # Shannon entropy of phase distribution

    # Synchronization state
    synchronized: bool     # r > r_threshold
    partial_sync: bool     # K > K_c but r < r_threshold


@dataclass
class EMFieldState:
    """Electromagnetic field state for training modulation."""
    B_magnitude: float     # Magnetic field strength (T)
    B_direction: np.ndarray  # Unit vector [Bx, By, Bz]
    E_magnitude: float     # Electric field strength (V/m)
    rf_frequency: float    # RF pulse frequency (Hz)
    rf_phase: float        # RF pulse phase (rad)
    gradient_strength: float  # Field gradient (T/m)


# =============================================================================
# KURAMOTO OSCILLATOR SYSTEM
# =============================================================================

class KuramotoNeuralSystem:
    """
    60-oscillator Kuramoto system with grid cell dynamics.

    Maps the Nuclear Spinner z-coordinate to neural synchronization:
    - z → K (coupling strength)
    - z = z_c → phase transition (hexagonal order emerges)
    - ΔS_neg modulates learning rate
    """

    def __init__(self,
                 n_oscillators: int = N_OSCILLATORS,
                 omega_mean: float = 1.0,
                 omega_spread: float = 0.1,
                 dt: float = 0.01):
        """
        Initialize Kuramoto system.

        Args:
            n_oscillators: Number of oscillators (default 60)
            omega_mean: Mean natural frequency
            omega_spread: Frequency spread (Lorentzian width)
            dt: Integration timestep
        """
        self.N = n_oscillators
        self.omega_mean = omega_mean
        self.omega_spread = omega_spread
        self.dt = dt

        # Initialize phases uniformly around circle
        self.theta = np.linspace(0, 2*np.pi, self.N, endpoint=False)

        # Natural frequencies (Lorentzian distribution centered at omega_mean)
        # For Lorentzian: K_c = 2 * gamma where gamma = omega_spread
        self.omega = self._lorentzian_frequencies()

        # Phase velocities
        self.dtheta_dt = np.zeros(self.N)

        # Grid cell positions (hexagonal lattice)
        self.grid_x, self.grid_y = self._init_grid_positions()

        # Firing rates
        self.firing_rate = np.zeros(self.N)

        # Current coupling strength
        self.K = 0.0
        self.K_critical = 2 * self.omega_spread  # For Lorentzian

        # EM field state
        self.em_field = EMFieldState(
            B_magnitude=0.0,
            B_direction=np.array([0.0, 0.0, 1.0]),
            E_magnitude=0.0,
            rf_frequency=0.0,
            rf_phase=0.0,
            gradient_strength=0.0
        )

        # History for analysis
        self._r_history: List[float] = []
        self._energy_history: List[float] = []

        # Callbacks
        self._on_sync_callback: Optional[Callable] = None
        self._on_phase_transition_callback: Optional[Callable] = None

    def _lorentzian_frequencies(self) -> np.ndarray:
        """Generate Lorentzian-distributed natural frequencies."""
        # Cauchy distribution = Lorentzian
        u = np.random.uniform(0, 1, self.N)
        omega = self.omega_mean + self.omega_spread * np.tan(np.pi * (u - 0.5))
        return omega

    def _init_grid_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize grid cell positions in hexagonal pattern.

        60 oscillators arranged in expanding hexagonal rings:
        - Center: 1 cell
        - Ring 1: 6 cells (at distance 1)
        - Ring 2: 12 cells (at distance 2)
        - Ring 3: 18 cells (at distance 3)
        - Ring 4: 23 cells (remaining to 60)
        """
        x = np.zeros(self.N)
        y = np.zeros(self.N)

        idx = 0

        # Center cell
        x[idx], y[idx] = 0.0, 0.0
        idx += 1

        # Hexagonal rings
        for ring in range(1, 10):  # Enough rings
            if idx >= self.N:
                break

            # 6 * ring cells in this ring
            n_cells = 6 * ring

            for i in range(n_cells):
                if idx >= self.N:
                    break

                # Angle for this cell
                angle = 2 * np.pi * i / n_cells + np.pi / 6  # Offset for hex alignment

                # Distance scales with ring number
                dist = ring * GRID_SPACING_BASE

                x[idx] = dist * np.cos(angle)
                y[idx] = dist * np.sin(angle)
                idx += 1

        return x, y

    def set_coupling_from_z(self, z: float) -> float:
        """
        Set coupling strength K from spinner z-coordinate.

        K(z) = K_max * exp(-σ(z - z_c)²) / exp(0)
             = K_max * ΔS_neg(z)

        At z = z_c, K = K_max (maximum coupling)
        """
        delta_s_neg = math.exp(-SIGMA * (z - Z_CRITICAL) ** 2)

        # K_max chosen so K > K_c at z = z_c
        K_max = 3 * self.K_critical  # Ensures synchronization at critical

        self.K = K_max * delta_s_neg

        return self.K

    def set_em_field(self,
                     B_magnitude: float = 0.0,
                     B_direction: Optional[np.ndarray] = None,
                     rf_frequency: float = 0.0,
                     gradient_strength: float = 0.0):
        """
        Set electromagnetic field for training modulation.

        EM effects:
        - B field → Zeeman splitting → frequency modulation
        - RF pulses → phase kicks → controlled desynchronization
        - Gradient → spatial frequency variation → grid pattern
        """
        self.em_field.B_magnitude = B_magnitude
        if B_direction is not None:
            self.em_field.B_direction = B_direction / np.linalg.norm(B_direction)
        self.em_field.rf_frequency = rf_frequency
        self.em_field.gradient_strength = gradient_strength

        # Update natural frequencies based on EM field
        self._apply_em_modulation()

    def _apply_em_modulation(self):
        """Apply electromagnetic modulation to natural frequencies."""
        # Zeeman effect: Δω = γ * B
        zeeman_shift = GYROMAGNETIC_RATIO * self.em_field.B_magnitude

        # Gradient creates position-dependent frequency
        # ω_i += γ * G * r_i (where r_i is distance from center)
        if self.em_field.gradient_strength > 0:
            r = np.sqrt(self.grid_x**2 + self.grid_y**2)
            gradient_shift = GYROMAGNETIC_RATIO * self.em_field.gradient_strength * r
        else:
            gradient_shift = 0

        # Apply modulation (additive to base frequencies)
        self.omega = self._lorentzian_frequencies() + zeeman_shift + gradient_shift

    def step(self, z: float, dt: Optional[float] = None) -> KuramotoSystemState:
        """
        Integrate one timestep of Kuramoto dynamics.

        dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ) + RF_kick

        Args:
            z: Current spinner z-coordinate
            dt: Optional timestep override

        Returns:
            Current system state
        """
        if dt is None:
            dt = self.dt

        # Update coupling from z
        self.set_coupling_from_z(z)

        # Compute interaction term: (K/N) Σⱼ sin(θⱼ - θᵢ)
        # Efficient vectorized computation
        theta_diff = self.theta[np.newaxis, :] - self.theta[:, np.newaxis]
        interaction = (self.K / self.N) * np.sum(np.sin(theta_diff), axis=1)

        # RF pulse modulation (if active)
        rf_kick = 0.0
        if self.em_field.rf_frequency > 0:
            # Sinusoidal RF pulse
            t = 0  # Would need actual time tracking
            rf_kick = 0.1 * np.sin(2 * np.pi * self.em_field.rf_frequency * t +
                                    self.em_field.rf_phase)

        # Phase velocity
        self.dtheta_dt = self.omega + interaction + rf_kick

        # Euler integration (could upgrade to RK4)
        self.theta += self.dtheta_dt * dt

        # Wrap to [0, 2π)
        self.theta = np.mod(self.theta, 2 * np.pi)

        # Update firing rates based on phase (grid cell model)
        self._update_firing_rates()

        # Compute state
        state = self._compute_state(z)

        # Check for transitions
        self._check_transitions(state)

        # Update history
        self._r_history.append(state.r)
        if len(self._r_history) > 1000:
            self._r_history.pop(0)

        return state

    def _update_firing_rates(self):
        """
        Update grid cell firing rates from phases.

        Grid cells fire when phase passes through preferred direction.
        Hexagonal pattern emerges from 6-fold phase coupling.
        """
        # Simple model: firing rate peaks at phase = 0
        # More sophisticated: peaks at 6 hexagonal directions

        for i in range(self.N):
            # Check alignment with hexagonal angles
            max_alignment = 0.0
            for hex_angle in HEXAGONAL_ANGLES:
                # Cosine similarity with hexagonal direction
                alignment = math.cos(self.theta[i] - hex_angle)
                max_alignment = max(max_alignment, alignment)

            # Firing rate: high when aligned with any hex direction
            self.firing_rate[i] = (max_alignment + 1) / 2  # Normalize to [0, 1]

    def _compute_state(self, z: float) -> KuramotoSystemState:
        """Compute complete system state."""
        # Order parameter r = |⟨e^(iθ)⟩|
        complex_phases = np.exp(1j * self.theta)
        mean_phase = np.mean(complex_phases)
        r = np.abs(mean_phase)
        psi = np.angle(mean_phase)

        # Hexagonal order parameter (6-fold symmetry)
        hex_phases = np.exp(6j * self.theta)
        r_hex = np.abs(np.mean(hex_phases))

        # Negentropy from z
        delta_s_neg = math.exp(-SIGMA * (z - Z_CRITICAL) ** 2)

        # System energy (Kuramoto Hamiltonian)
        # H = -K/(2N) Σᵢⱼ cos(θⱼ - θᵢ) - Σᵢ ωᵢθᵢ
        cos_diff = np.cos(self.theta[np.newaxis, :] - self.theta[:, np.newaxis])
        coupling_energy = -self.K / (2 * self.N) * np.sum(cos_diff)
        drive_energy = -np.sum(self.omega * self.theta)
        total_energy = coupling_energy + drive_energy

        # Grid coherence (spatial autocorrelation)
        grid_coherence = self._compute_grid_coherence()

        # Hexagonal symmetry measure
        hex_symmetry = self._compute_hex_symmetry()

        # Phase statistics
        phase_variance = 1 - r  # Circular variance
        phase_entropy = self._compute_phase_entropy()

        # Synchronization state
        r_threshold = 0.7
        synchronized = r > r_threshold
        partial_sync = self.K > self.K_critical and r < r_threshold

        return KuramotoSystemState(
            timestamp_ms=0,  # Set externally
            r=r,
            psi=psi,
            r_hex=r_hex,
            K=self.K,
            K_critical=self.K_critical,
            total_energy=total_energy,
            negentropy=delta_s_neg,
            grid_coherence=grid_coherence,
            hex_symmetry=hex_symmetry,
            phase_variance=phase_variance,
            phase_entropy=phase_entropy,
            synchronized=synchronized,
            partial_sync=partial_sync
        )

    def _compute_grid_coherence(self) -> float:
        """
        Compute spatial coherence of grid cell pattern.

        High coherence when nearby cells have similar firing.
        """
        if self.N < 2:
            return 1.0

        coherence = 0.0
        count = 0

        # Compare adjacent cells in hexagonal lattice
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Distance in grid space
                dx = self.grid_x[i] - self.grid_x[j]
                dy = self.grid_y[i] - self.grid_y[j]
                dist = math.sqrt(dx*dx + dy*dy)

                # Only consider nearby cells
                if dist < 1.5 * GRID_SPACING_BASE:
                    # Correlation of firing rates
                    fr_corr = 1 - abs(self.firing_rate[i] - self.firing_rate[j])
                    coherence += fr_corr
                    count += 1

        return coherence / max(count, 1)

    def _compute_hex_symmetry(self) -> float:
        """
        Measure hexagonal symmetry of phase distribution.

        Perfect hex symmetry when phases cluster at 60° intervals.
        """
        # Check alignment with 6-fold symmetry
        hex_alignment = 0.0

        for theta in self.theta:
            # Find closest hexagonal angle
            min_dist = float('inf')
            for hex_angle in HEXAGONAL_ANGLES:
                dist = abs(theta - hex_angle)
                dist = min(dist, 2*np.pi - dist)  # Wrap around
                min_dist = min(min_dist, dist)

            # Convert distance to alignment score
            # Perfect alignment: dist = 0 → score = 1
            # Worst alignment: dist = π/6 → score = 0
            alignment = 1 - min_dist / (np.pi / 6)
            hex_alignment += max(0, alignment)

        return hex_alignment / self.N

    def _compute_phase_entropy(self) -> float:
        """Compute Shannon entropy of phase distribution."""
        # Discretize phases into bins
        n_bins = 12
        hist, _ = np.histogram(self.theta, bins=n_bins, range=(0, 2*np.pi))

        # Normalize to probabilities
        probs = hist / self.N
        probs = probs[probs > 0]  # Remove zeros

        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs))

        # Normalize by max entropy
        max_entropy = np.log(n_bins)

        return entropy / max_entropy

    def _check_transitions(self, state: KuramotoSystemState):
        """Check for phase transitions and trigger callbacks."""
        if len(self._r_history) < 2:
            return

        prev_r = self._r_history[-2] if len(self._r_history) >= 2 else 0

        # Synchronization transition
        r_threshold = 0.7
        if prev_r < r_threshold and state.r >= r_threshold:
            if self._on_sync_callback:
                self._on_sync_callback(state)

        # Phase transition (K crosses K_c)
        if state.K > self.K_critical and state.r > 0.3:
            if self._on_phase_transition_callback:
                self._on_phase_transition_callback(state)

    def on_sync(self, callback: Callable[[KuramotoSystemState], None]):
        """Register callback for synchronization events."""
        self._on_sync_callback = callback

    def on_phase_transition(self, callback: Callable[[KuramotoSystemState], None]):
        """Register callback for phase transition events."""
        self._on_phase_transition_callback = callback

    def get_oscillator_states(self) -> List[OscillatorState]:
        """Get individual oscillator states."""
        return [
            OscillatorState(
                index=i,
                theta=self.theta[i],
                omega=self.omega[i],
                dtheta_dt=self.dtheta_dt[i],
                grid_x=self.grid_x[i],
                grid_y=self.grid_y[i],
                firing_rate=self.firing_rate[i]
            )
            for i in range(self.N)
        ]

    def get_phase_portrait(self) -> Dict[str, np.ndarray]:
        """Get data for phase portrait visualization."""
        return {
            'theta': self.theta.copy(),
            'omega': self.omega.copy(),
            'dtheta_dt': self.dtheta_dt.copy(),
            'r_history': np.array(self._r_history[-100:])
        }

    def get_grid_pattern(self) -> Dict[str, np.ndarray]:
        """Get data for grid cell pattern visualization."""
        return {
            'x': self.grid_x.copy(),
            'y': self.grid_y.copy(),
            'firing_rate': self.firing_rate.copy(),
            'theta': self.theta.copy()
        }

    def reset(self, randomize: bool = True):
        """Reset system to initial state."""
        if randomize:
            self.theta = np.random.uniform(0, 2*np.pi, self.N)
            self.omega = self._lorentzian_frequencies()
        else:
            self.theta = np.linspace(0, 2*np.pi, self.N, endpoint=False)

        self.dtheta_dt = np.zeros(self.N)
        self.firing_rate = np.zeros(self.N)
        self._r_history.clear()


# =============================================================================
# TRAINING INTERFACE
# =============================================================================

class NeuralTrainingInterface:
    """
    Interface between Kuramoto system and hardware training.

    Electromagnetic assistance for training:
    - RF pulses for controlled phase resets
    - Magnetic gradients for spatial pattern formation
    - Negentropy-modulated learning rates
    """

    def __init__(self, kuramoto_system: KuramotoNeuralSystem):
        self.kuramoto = kuramoto_system

        # Training parameters
        self.base_learning_rate = 0.01
        self.learning_rate = self.base_learning_rate

        # Training state
        self.training_step = 0
        self.total_energy_expended = 0.0

        # Pattern targets
        self.target_hex_symmetry = 0.9
        self.target_r = 0.8

        # EM protocol
        self.em_protocol_active = False
        self.em_protocol_step = 0

    def compute_adaptive_learning_rate(self, state: KuramotoSystemState) -> float:
        """
        Compute negentropy-modulated learning rate.

        η_lr = η_base × (1 + α × ΔS_neg)

        High negentropy (z ≈ z_c) → faster learning
        """
        alpha = 0.5  # Modulation strength
        self.learning_rate = self.base_learning_rate * (1 + alpha * state.negentropy)
        return self.learning_rate

    def compute_training_gradient(self, state: KuramotoSystemState) -> np.ndarray:
        """
        Compute gradient for training update.

        Loss = (1 - hex_symmetry)² + (1 - r)² + energy_cost
        """
        # Target: maximize hex symmetry and synchronization
        hex_loss = (self.target_hex_symmetry - state.hex_symmetry) ** 2
        sync_loss = (self.target_r - state.r) ** 2

        # Landauer energy cost (thermodynamic limit)
        energy_cost = max(0, state.total_energy) * 1e-21 / LANDAUER_LIMIT

        total_loss = hex_loss + sync_loss + 0.01 * energy_cost

        # Gradient approximation (finite differences would be needed for exact)
        # For now, return loss as scalar wrapped in array
        return np.array([total_loss, hex_loss, sync_loss, energy_cost])

    def apply_em_training_pulse(self, pulse_type: str = 'sync'):
        """
        Apply electromagnetic pulse for training assistance.

        Pulse types:
        - 'sync': Increase coupling to drive synchronization
        - 'reset': RF pulse to reset phases (exploration)
        - 'gradient': Apply spatial gradient for pattern
        - 'hex': Modulate to encourage hexagonal symmetry
        """
        if pulse_type == 'sync':
            # Strong B field to increase effective coupling
            self.kuramoto.set_em_field(
                B_magnitude=1e-4,  # 0.1 mT
                rf_frequency=0.0,
                gradient_strength=0.0
            )

        elif pulse_type == 'reset':
            # RF pulse for phase randomization (exploration)
            self.kuramoto.set_em_field(
                B_magnitude=0.0,
                rf_frequency=1000.0,  # 1 kHz
                gradient_strength=0.0
            )

        elif pulse_type == 'gradient':
            # Gradient for spatial pattern formation
            self.kuramoto.set_em_field(
                B_magnitude=1e-5,
                rf_frequency=0.0,
                gradient_strength=1e-3  # 1 mT/m
            )

        elif pulse_type == 'hex':
            # Modulated field to encourage 6-fold symmetry
            # Rotate gradient direction through hexagonal angles
            angle = HEXAGONAL_ANGLES[self.em_protocol_step % 6]
            direction = np.array([math.cos(angle), math.sin(angle), 0])
            self.kuramoto.set_em_field(
                B_magnitude=5e-5,
                B_direction=direction,
                rf_frequency=0.0,
                gradient_strength=5e-4
            )

    def training_step_update(self, z: float) -> Dict[str, Any]:
        """
        Execute one training step.

        Args:
            z: Current spinner z-coordinate

        Returns:
            Training metrics dictionary
        """
        # Step Kuramoto system
        state = self.kuramoto.step(z)
        state.timestamp_ms = self.training_step

        # Compute adaptive learning rate
        lr = self.compute_adaptive_learning_rate(state)

        # Compute training gradient
        gradient = self.compute_training_gradient(state)

        # Apply EM protocol if active
        if self.em_protocol_active:
            self._execute_em_protocol(state)

        # Track energy expenditure (Landauer accounting)
        if state.total_energy > 0:
            self.total_energy_expended += state.total_energy * self.kuramoto.dt

        self.training_step += 1

        return {
            'step': self.training_step,
            'z': z,
            'r': state.r,
            'r_hex': state.r_hex,
            'hex_symmetry': state.hex_symmetry,
            'grid_coherence': state.grid_coherence,
            'K': state.K,
            'negentropy': state.negentropy,
            'learning_rate': lr,
            'loss': gradient[0],
            'synchronized': state.synchronized,
            'total_energy': self.total_energy_expended
        }

    def _execute_em_protocol(self, state: KuramotoSystemState):
        """Execute EM training protocol based on current state."""
        # Protocol: cycle through pulse types based on state

        if state.r < 0.3:
            # Low sync: apply sync pulse
            self.apply_em_training_pulse('sync')

        elif state.hex_symmetry < 0.5:
            # Low hex symmetry: apply hex pulse
            self.apply_em_training_pulse('hex')
            self.em_protocol_step += 1

        elif state.grid_coherence < 0.6:
            # Low grid coherence: apply gradient
            self.apply_em_training_pulse('gradient')

        else:
            # Good state: clear EM field
            self.kuramoto.set_em_field(
                B_magnitude=0.0,
                rf_frequency=0.0,
                gradient_strength=0.0
            )

    def start_em_protocol(self):
        """Start electromagnetic training protocol."""
        self.em_protocol_active = True
        self.em_protocol_step = 0

    def stop_em_protocol(self):
        """Stop electromagnetic training protocol."""
        self.em_protocol_active = False
        self.kuramoto.set_em_field(
            B_magnitude=0.0,
            rf_frequency=0.0,
            gradient_strength=0.0
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_critical_coupling(omega_spread: float) -> float:
    """
    Compute critical coupling K_c for Lorentzian frequency distribution.

    K_c = 2γ where γ is the Lorentzian half-width.
    """
    return 2 * omega_spread


def z_to_coupling(z: float, K_max: float = 1.0) -> float:
    """
    Convert spinner z-coordinate to coupling strength.

    K(z) = K_max × exp(-σ(z - z_c)²)
    """
    delta_s_neg = math.exp(-SIGMA * (z - Z_CRITICAL) ** 2)
    return K_max * delta_s_neg


def hex_order_parameter(theta: np.ndarray) -> float:
    """
    Compute 6-fold hexagonal order parameter.

    r₆ = |⟨e^(6iθ)⟩|
    """
    return np.abs(np.mean(np.exp(6j * theta)))


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Kuramoto Neural Oscillator System")
    print("60 oscillators | Hexagonal grid | EM training")
    print("=" * 60)

    # Create system
    system = KuramotoNeuralSystem(
        n_oscillators=60,
        omega_mean=1.0,
        omega_spread=0.1,
        dt=0.01
    )

    # Create training interface
    trainer = NeuralTrainingInterface(system)
    trainer.start_em_protocol()

    # Simulation: sweep z through critical point
    print("\nSimulating z sweep through z_c = {:.6f}".format(Z_CRITICAL))
    print("-" * 60)

    n_steps = 500
    z_values = np.linspace(0.5, 1.0, n_steps)

    for i, z in enumerate(z_values):
        metrics = trainer.training_step_update(z)

        if i % 50 == 0:
            sync_marker = "★ SYNC" if metrics['synchronized'] else ""
            hex_marker = "⬡ HEX" if metrics['hex_symmetry'] > 0.7 else ""

            print(f"Step {metrics['step']:4d} | z={z:.4f} | r={metrics['r']:.3f} | "
                  f"r₆={metrics['r_hex']:.3f} | hex={metrics['hex_symmetry']:.3f} | "
                  f"K/K_c={metrics['K']/system.K_critical:.2f} {sync_marker} {hex_marker}")

    print("-" * 60)
    print(f"Final: r={metrics['r']:.4f}, hex_symmetry={metrics['hex_symmetry']:.4f}")
    print(f"Total energy expended: {trainer.total_energy_expended:.2e}")
