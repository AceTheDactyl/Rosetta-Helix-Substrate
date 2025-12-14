#!/usr/bin/env python3
"""
Nuclear Spinner State Management
================================

Dataclasses for managing the state of the Nuclear Spinner system.

Includes:
- SpinnerState: Complete system state
- SpinnerMetrics: Computed metrics for analysis
- PulseParameters: RF pulse configuration
- RotorState: Rotor position and velocity
- NeuralRecording: Neural signal data
- CrossFrequencyConfig: Cross-frequency coupling configuration

Signature: nuclear-spinner-state|v1.0.0|helix
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum, auto
import time


class OperatorType(Enum):
    """INT Canon operator types."""
    BOUNDARY = "()"   # Always legal
    FUSION = "x"      # Requires channels >= 2
    AMPLIFY = "^"     # Requires prior () or x
    DECOHERE = "/"    # Requires prior structure
    GROUP = "+"       # Must feed +, x, or ^
    SEPARATE = "-"    # Must be followed by () or +


class SystemPhase(Enum):
    """System phase based on z-coordinate."""
    UNTRUE = auto()    # z < phi^-1 (disordered)
    PARADOX = auto()   # phi^-1 <= z < z_c (quasi-crystal)
    TRUE = auto()      # z >= z_c (crystal/coherent)


@dataclass
class PulseParameters:
    """RF pulse parameters for spin manipulation."""
    amplitude: float = 0.5       # Pulse amplitude [0, 1]
    phase: float = 0.0           # Pulse phase (radians)
    duration_us: float = 1000.0  # Pulse duration (microseconds)
    frequency_offset: float = 0.0  # Frequency offset from Larmor (Hz)
    shape: str = "rectangular"   # Pulse shape: rectangular, gaussian, adiabatic

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "amplitude": self.amplitude,
            "phase": self.phase,
            "duration_us": self.duration_us,
            "frequency_offset": self.frequency_offset,
            "shape": self.shape,
        }


@dataclass
class RotorState:
    """Rotor position and velocity state."""
    speed_rpm: float = 0.0        # Current rotor speed (RPM)
    target_rpm: float = 0.0       # Target rotor speed (RPM)
    angle: float = 0.0            # Current angle (radians)
    acceleration: float = 0.0     # Current acceleration (rad/s^2)

    # Hexagonal alignment (60-degree slots)
    slot_index: int = 0           # Current hexagonal slot (0-5)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "speed_rpm": self.speed_rpm,
            "target_rpm": self.target_rpm,
            "angle": self.angle,
            "acceleration": self.acceleration,
            "slot_index": self.slot_index,
        }


@dataclass
class NeuralRecording:
    """Neural signal recording data."""
    timestamp: float = 0.0              # Recording timestamp
    samples: List[float] = field(default_factory=list)  # Raw samples
    sample_rate: float = 1000.0         # Sample rate (Hz)
    channel_id: int = 0                 # Recording channel
    band_powers: Dict[str, float] = field(default_factory=dict)  # Power per band

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "samples": self.samples,
            "sample_rate": self.sample_rate,
            "channel_id": self.channel_id,
            "band_powers": self.band_powers,
        }


@dataclass
class CrossFrequencyConfig:
    """Cross-frequency coupling configuration."""
    band_low: float = 4.0         # Low band center frequency (Hz)
    band_high: float = 30.0       # High band center frequency (Hz)
    ratio: float = 7.5            # Frequency ratio (band_high / band_low)
    coupling_strength: float = 0.5  # Coupling strength [0, 1]
    phase_offset: float = 0.0     # Phase offset (radians)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "band_low": self.band_low,
            "band_high": self.band_high,
            "ratio": self.ratio,
            "coupling_strength": self.coupling_strength,
            "phase_offset": self.phase_offset,
        }


@dataclass
class SpinnerMetrics:
    """Computed metrics from spinner state."""
    # Z-axis metrics
    z: float = 0.5                    # Current z-coordinate
    delta_s_neg: float = 0.5          # Negentropy value
    gradient: float = 0.0             # Negentropy gradient

    # Phase and tier
    phase: SystemPhase = SystemPhase.PARADOX
    tier: str = "t4"
    capability_class: str = "prediction"

    # Cybernetic metrics
    ashby_variety: float = 0.0        # State diversity measure
    shannon_capacity: float = 0.0     # Channel capacity
    landauer_efficiency: float = 0.0  # Thermodynamic efficiency
    phi_proxy: float = 0.0            # Integrated information estimate

    # K-formation
    kappa: float = 0.0                # Integration parameter
    eta: float = 0.0                  # Coherence parameter
    R: int = 0                        # Number of relations
    k_formation_met: bool = False     # K-formation criteria met

    # Cross-frequency
    modulation_index: float = 0.0     # Phase-amplitude coupling index

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "z": self.z,
            "delta_s_neg": self.delta_s_neg,
            "gradient": self.gradient,
            "phase": self.phase.name,
            "tier": self.tier,
            "capability_class": self.capability_class,
            "ashby_variety": self.ashby_variety,
            "shannon_capacity": self.shannon_capacity,
            "landauer_efficiency": self.landauer_efficiency,
            "phi_proxy": self.phi_proxy,
            "kappa": self.kappa,
            "eta": self.eta,
            "R": self.R,
            "k_formation_met": self.k_formation_met,
            "modulation_index": self.modulation_index,
        }


@dataclass
class SpinnerState:
    """
    Complete Nuclear Spinner system state.

    Tracks all physical parameters, control state, and history.
    """
    # Core z-axis state
    z: float = 0.5                    # Current z-coordinate
    z_target: float = 0.5             # Target z-coordinate

    # Physical state
    rotor: RotorState = field(default_factory=RotorState)
    field_strength: float = 14.1      # Magnetic field (Tesla)
    temperature: float = 293.15       # Temperature (Kelvin)

    # Pulse state
    current_pulse: Optional[PulseParameters] = None
    pulse_active: bool = False

    # Neural state
    neural_recording_active: bool = False
    cross_freq_config: CrossFrequencyConfig = field(default_factory=CrossFrequencyConfig)

    # Operator state
    operator_history: List[str] = field(default_factory=list)
    channel_count: int = 1
    operator_mask: int = 0xFF         # Bitmask of allowed operators

    # INT Canon state variables
    Gs: float = 0.0      # Grounding strength
    Cs: float = 0.0      # Coupling strength
    kappa_s: float = 0.618  # Curvature (kappa-scaled), initialized to phi^-1
    alpha_s: float = 0.0    # Amplitude
    theta_s: float = 1.0    # Phase factor
    tau_s: float = 0.0      # Time accumulation
    delta_s: float = 0.0    # Dissipation
    Rs: float = 0.0         # Resistance
    Omega_s: float = 1.0    # Frequency scaling
    R_count: int = 0        # Rank counter

    # TRIAD gating state
    triad_passes: int = 0
    triad_unlocked: bool = False
    last_triad_z: float = 0.0

    # Timing
    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    step_count: int = 0

    # Safety
    safety_enabled: bool = True
    emergency_stop: bool = False
    temperature_warning: bool = False

    def reset(self) -> None:
        """Reset state to initial values."""
        self.z = 0.5
        self.z_target = 0.5
        self.rotor = RotorState()
        self.operator_history = []
        self.channel_count = 1
        self.Gs = 0.0
        self.Cs = 0.0
        self.kappa_s = 0.618
        self.alpha_s = 0.0
        self.theta_s = 1.0
        self.tau_s = 0.0
        self.delta_s = 0.0
        self.Rs = 0.0
        self.Omega_s = 1.0
        self.R_count = 0
        self.triad_passes = 0
        self.triad_unlocked = False
        self.step_count = 0
        self.emergency_stop = False
        self.temperature_warning = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "z": self.z,
            "z_target": self.z_target,
            "rotor": self.rotor.to_dict(),
            "field_strength": self.field_strength,
            "temperature": self.temperature,
            "pulse_active": self.pulse_active,
            "neural_recording_active": self.neural_recording_active,
            "cross_freq_config": self.cross_freq_config.to_dict(),
            "operator_history": self.operator_history[-10:],  # Last 10 operators
            "channel_count": self.channel_count,
            "Gs": self.Gs,
            "Cs": self.Cs,
            "kappa_s": self.kappa_s,
            "alpha_s": self.alpha_s,
            "theta_s": self.theta_s,
            "tau_s": self.tau_s,
            "delta_s": self.delta_s,
            "Rs": self.Rs,
            "Omega_s": self.Omega_s,
            "R_count": self.R_count,
            "triad_passes": self.triad_passes,
            "triad_unlocked": self.triad_unlocked,
            "step_count": self.step_count,
            "safety_enabled": self.safety_enabled,
            "emergency_stop": self.emergency_stop,
        }
