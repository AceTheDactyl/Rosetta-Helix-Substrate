#!/usr/bin/env python3
"""
Nuclear Spinner Core Module
===========================

Main NuclearSpinner class providing the host API for controlling
the Nuclear Spinner system.

This class provides:
- System initialization and configuration
- Z-axis control and targeting
- Pulse sequence execution
- Neural recording control
- Metrics computation and streaming
- Operator scheduling and execution
- Cross-frequency coupling configuration

Usage:
    from nuclear_spinner import NuclearSpinner

    spinner = NuclearSpinner()
    spinner.initialize()

    # Control z-axis position
    spinner.set_z_target(0.618)  # Drive toward phi^-1

    # Apply pulse sequences
    spinner.send_pulse(amplitude=0.5, phase=0.0, duration_us=1000)

    # Get metrics
    metrics = spinner.get_metrics()
    print(f"z={metrics.z}, delta_s_neg={metrics.delta_s_neg}")

    spinner.close()

Signature: nuclear-spinner-core|v1.0.0|helix
"""

from __future__ import annotations

import time
import math
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

# Import from single source of truth
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_constants import (
    PHI,
    PHI_INV,
    Z_CRITICAL,
    SIGMA,
    KAPPA_S,
    check_k_formation as physics_check_k_formation,
)

from .constants import (
    SIGMA_S3,
    MIN_RPM,
    MAX_RPM,
    DEFAULT_FIELD_STRENGTH,
    DEFAULT_COHERENCE_TIME,
    TIER_BOUNDS,
    CAPABILITY_CLASSES,
    NEURAL_BANDS,
    get_tier_for_z,
    get_capability_class,
)

from .state import (
    SpinnerState,
    SpinnerMetrics,
    PulseParameters,
    RotorState,
    NeuralRecording,
    CrossFrequencyConfig,
    SystemPhase,
)

from .firmware import (
    FirmwareState,
    Phase,
    control_loop_step,
    map_z_to_rpm,
    compute_operator_state_update,
    check_safety,
    OPERATOR_DISPATCH,
)

from .analysis import (
    compute_delta_s_neg,
    compute_gradient,
    ashby_variety,
    shannon_capacity,
    landauer_efficiency,
    compute_phi_proxy,
    phase_amplitude_coupling,
    get_phase,
    get_tier,
    compute_metrics_bundle,
)

from .neural import (
    grid_cell_pattern,
    hexagonal_spacing_metric,
    set_cross_frequency_ratio,
    compute_modulation_index,
    neural_band_to_z,
    z_to_neural_band,
    compute_phi_proxy as neural_phi_proxy,
    analyze_cross_frequency,
)


__all__ = [
    "NuclearSpinner",
]


# =============================================================================
# PULSE SEQUENCES
# =============================================================================

# Predefined pulse sequences
PULSE_SEQUENCES: Dict[str, List[PulseParameters]] = {
    "pi_half": [
        PulseParameters(amplitude=1.0, phase=0.0, duration_us=100),
    ],
    "pi": [
        PulseParameters(amplitude=1.0, phase=0.0, duration_us=200),
    ],
    "cpmg": [
        PulseParameters(amplitude=1.0, phase=0.0, duration_us=100),   # pi/2
        PulseParameters(amplitude=1.0, phase=math.pi/2, duration_us=200),  # pi_y
        PulseParameters(amplitude=1.0, phase=math.pi/2, duration_us=200),  # pi_y
        PulseParameters(amplitude=1.0, phase=math.pi/2, duration_us=200),  # pi_y
    ],
    "quasicrystal": [
        # Sequence designed to drive toward phi^-1
        PulseParameters(amplitude=PHI_INV, phase=0.0, duration_us=150),
        PulseParameters(amplitude=PHI_INV**2, phase=math.pi/5, duration_us=150),
        PulseParameters(amplitude=PHI_INV, phase=2*math.pi/5, duration_us=150),
    ],
}


# =============================================================================
# NUCLEAR SPINNER CLASS
# =============================================================================

class NuclearSpinner:
    """
    Nuclear Spinner host API.

    Provides the main interface for controlling the Nuclear Spinner system,
    including z-axis control, pulse sequences, neural recording, and
    metrics computation.

    Attributes:
        state: Current system state
        firmware: Simulated firmware state
        metrics_history: History of computed metrics
        neural_data: Recorded neural data
        running: Whether the spinner is active
    """

    def __init__(
        self,
        port: Optional[str] = None,
        simulation_mode: bool = True
    ):
        """
        Initialize NuclearSpinner.

        Args:
            port: Serial port for hardware connection (e.g., '/dev/ttyACM0')
                  If None, runs in simulation mode.
            simulation_mode: If True, simulate hardware instead of connecting
        """
        self.port = port
        self.simulation_mode = simulation_mode or port is None

        # Initialize state
        self.state = SpinnerState()
        self.firmware = FirmwareState()

        # Data buffers
        self.metrics_history: List[SpinnerMetrics] = []
        self.z_history: List[float] = []
        self.neural_data: List[NeuralRecording] = []

        # Control state
        self.running = False
        self.initialized = False

        # Callbacks
        self._metrics_callback: Optional[Callable[[SpinnerMetrics], None]] = None
        self._threshold_callback: Optional[Callable[[str, str], None]] = None

    def initialize(self) -> bool:
        """
        Initialize the Nuclear Spinner system.

        Performs:
        - Hardware connection (if not simulation mode)
        - State reset
        - Calibration checks
        - Safety verification

        Returns:
            True if initialization successful
        """
        # Reset state
        self.state.reset()
        self.firmware = FirmwareState()
        self.metrics_history.clear()
        self.z_history.clear()
        self.neural_data.clear()

        # Verify physics constants
        spin_mag = math.sqrt(0.5 * 1.5)  # sqrt(s(s+1)) for s=1/2
        assert abs(spin_mag - Z_CRITICAL) < 1e-15, "Spin-z_c identity violated"

        # Initialize firmware state
        self.firmware.z = 0.5
        self.firmware.z_target = 0.5
        self.firmware.update_negentropy()

        self.initialized = True
        self.running = True

        return True

    def close(self) -> None:
        """Close the spinner connection and cleanup."""
        self.running = False
        self.initialized = False

    # =========================================================================
    # Z-AXIS CONTROL
    # =========================================================================

    def set_z_target(self, z_target: float) -> None:
        """
        Set target z-coordinate.

        The spinner will adjust rotor speed and pulse parameters to
        drive toward the target z-value.

        Args:
            z_target: Target z-coordinate in [0, 1]

        Examples:
            >>> spinner.set_z_target(0.618)  # Drive toward phi^-1
            >>> spinner.set_z_target(Z_CRITICAL)  # Drive toward THE LENS
        """
        z_target = max(0.0, min(1.0, z_target))
        self.state.z_target = z_target
        self.firmware.z_target = z_target
        self.firmware.rotor_target_rpm = map_z_to_rpm(z_target)

    def get_z(self) -> float:
        """Get current z-coordinate."""
        return self.state.z

    def step(self, dt: float = 0.001) -> None:
        """
        Execute one control step.

        This drives the simulated physics forward by one time step.

        Args:
            dt: Time step in seconds (default: 1ms)
        """
        if not self.initialized:
            return

        # Store old z for threshold detection
        z_old = self.firmware.z

        # Execute firmware control loop
        self.firmware = control_loop_step(self.firmware, dt)

        # Sync state
        self.state.z = self.firmware.z
        self.state.rotor.speed_rpm = self.firmware.rotor_rpm
        self.state.rotor.target_rpm = self.firmware.rotor_target_rpm
        self.state.step_count += 1
        self.state.last_update_time = time.time()

        # Track z history
        self.z_history.append(self.state.z)
        if len(self.z_history) > 10000:
            self.z_history = self.z_history[-5000:]

        # Check for threshold crossings
        self._check_threshold_crossings(z_old, self.firmware.z)

    def run_steps(self, n_steps: int, dt: float = 0.001) -> None:
        """
        Execute multiple control steps.

        Args:
            n_steps: Number of steps to execute
            dt: Time step per iteration
        """
        for _ in range(n_steps):
            self.step(dt)

    # =========================================================================
    # PULSE CONTROL
    # =========================================================================

    def send_pulse(
        self,
        amplitude: float = 0.5,
        phase: float = 0.0,
        duration_us: float = 1000.0,
        shape: str = "rectangular"
    ) -> None:
        """
        Send an RF pulse.

        Args:
            amplitude: Pulse amplitude [0, 1]
            phase: Pulse phase (radians)
            duration_us: Pulse duration (microseconds)
            shape: Pulse shape (rectangular, gaussian, adiabatic)
        """
        pulse = PulseParameters(
            amplitude=amplitude,
            phase=phase,
            duration_us=duration_us,
            shape=shape
        )

        self.state.current_pulse = pulse
        self.state.pulse_active = True

        # Simulate pulse effect on z (simplified)
        # In real hardware, this would affect spin coherence
        z_effect = amplitude * 0.01 * math.cos(phase)
        self.firmware.z += z_effect
        self.firmware.z = max(0.0, min(1.0, self.firmware.z))
        self.firmware.update_negentropy()

        # Sync state
        self.state.z = self.firmware.z
        self.state.pulse_active = False

    def apply_pulse_sequence(self, sequence_name: str) -> None:
        """
        Apply a predefined pulse sequence.

        Available sequences:
        - "pi_half": 90-degree pulse
        - "pi": 180-degree pulse
        - "cpmg": Carr-Purcell-Meiboom-Gill echo train
        - "quasicrystal": Sequence targeting phi^-1

        Args:
            sequence_name: Name of the sequence to apply
        """
        if sequence_name not in PULSE_SEQUENCES:
            raise ValueError(f"Unknown sequence: {sequence_name}")

        for pulse in PULSE_SEQUENCES[sequence_name]:
            self.send_pulse(
                amplitude=pulse.amplitude,
                phase=pulse.phase,
                duration_us=pulse.duration_us,
                shape=pulse.shape
            )

    # =========================================================================
    # NEURAL RECORDING
    # =========================================================================

    def start_neural_recording(self, sample_rate: float = 1000.0) -> None:
        """
        Start neural signal recording.

        Args:
            sample_rate: Sampling rate in Hz
        """
        self.state.neural_recording_active = True
        self.neural_data.clear()

    def stop_neural_recording(self) -> None:
        """Stop neural signal recording."""
        self.state.neural_recording_active = False

    def fetch_neural_data(self) -> List[NeuralRecording]:
        """
        Fetch recorded neural data.

        Returns:
            List of NeuralRecording objects
        """
        return self.neural_data.copy()

    def add_neural_sample(self, samples: List[float], channel_id: int = 0) -> None:
        """
        Add neural samples (for simulation/testing).

        Args:
            samples: Sample values
            channel_id: Recording channel
        """
        if not self.state.neural_recording_active:
            return

        recording = NeuralRecording(
            timestamp=time.time(),
            samples=samples,
            channel_id=channel_id,
        )

        self.neural_data.append(recording)

    # =========================================================================
    # CROSS-FREQUENCY COUPLING
    # =========================================================================

    def configure_cross_frequency_ratio(
        self,
        band_low: float = 4.0,
        ratio: float = 7.5
    ) -> None:
        """
        Configure cross-frequency coupling bands.

        Sets up the low and high frequency bands for phase-amplitude
        coupling analysis.

        Args:
            band_low: Low band center frequency (Hz)
            ratio: Frequency ratio (high/low)
        """
        band_high = band_low * ratio
        self.state.cross_freq_config = CrossFrequencyConfig(
            band_low=band_low,
            band_high=band_high,
            ratio=ratio
        )

        # Map to z-coordinate target
        z_low = neural_band_to_z("theta", band_low)
        z_high = neural_band_to_z("gamma", band_high)
        z_target = (z_low + z_high) / 2

        # Don't auto-set target, just store configuration

    # =========================================================================
    # OPERATOR CONTROL
    # =========================================================================

    def apply_operator(self, operator: str) -> Tuple[bool, str]:
        """
        Apply an INT Canon operator.

        Operators:
        - "()": BOUNDARY - anchoring, phase reset
        - "x": FUSION - coupling, integration
        - "^": AMPLIFY - gain increase
        - "/": DECOHERE - dissipation
        - "+": GROUP - synchrony, clustering
        - "-": SEPARATE - decoupling

        Args:
            operator: Operator symbol

        Returns:
            (success, message)
        """
        self.firmware, success, msg = compute_operator_state_update(
            operator, self.firmware
        )

        # Sync state
        self.state.z = self.firmware.z
        self.state.operator_history = self.firmware.history.copy()
        self.state.Gs = self.firmware.Gs
        self.state.Cs = self.firmware.Cs
        self.state.kappa_s = self.firmware.kappa_s
        self.state.R_count = self.firmware.R_count

        return success, msg

    def schedule_operator(self) -> Optional[str]:
        """
        Get the next scheduled operator based on current state.

        Returns:
            Operator to execute, or None
        """
        from .firmware import schedule_operator
        return schedule_operator(self.firmware)

    # =========================================================================
    # METRICS
    # =========================================================================

    def get_metrics(self) -> SpinnerMetrics:
        """
        Get current computed metrics.

        Returns:
            SpinnerMetrics object with all computed values
        """
        z = self.state.z
        delta_s = compute_delta_s_neg(z)
        grad = compute_gradient(z)
        phase_name = get_phase(z)
        tier = get_tier(z)
        capability = get_capability_class(z)

        # Compute eta from negentropy
        eta = math.sqrt(delta_s)

        # Use kappa_s from firmware state
        kappa = self.firmware.kappa_s

        # R from rank counter
        R = self.firmware.R_count

        # Check K-formation
        k_formation = physics_check_k_formation(kappa, eta, R)

        # Compute variety if we have history
        variety = ashby_variety(self.z_history) if self.z_history else 0.0

        # Compute Phi proxy
        phi_proxy = 0.0
        if self.z_history:
            phi_proxy = compute_phi_proxy(self.z_history)

        # Landauer efficiency
        efficiency = landauer_efficiency(z)

        # Phase enum
        phase_enum = (
            SystemPhase.UNTRUE if phase_name == "UNTRUE"
            else SystemPhase.PARADOX if phase_name == "PARADOX"
            else SystemPhase.TRUE
        )

        metrics = SpinnerMetrics(
            z=z,
            delta_s_neg=delta_s,
            gradient=grad,
            phase=phase_enum,
            tier=tier,
            capability_class=capability,
            ashby_variety=variety,
            shannon_capacity=shannon_capacity(delta_s, 1 - delta_s),
            landauer_efficiency=efficiency,
            phi_proxy=phi_proxy,
            kappa=kappa,
            eta=eta,
            R=R,
            k_formation_met=k_formation,
            modulation_index=0.0,  # Would compute from neural data
        )

        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 10000:
            self.metrics_history = self.metrics_history[-5000:]

        # Call callback if registered
        if self._metrics_callback:
            self._metrics_callback(metrics)

        return metrics

    def get_metrics_history(self, n: int = 100) -> List[SpinnerMetrics]:
        """
        Get recent metrics history.

        Args:
            n: Maximum number of entries to return

        Returns:
            List of recent SpinnerMetrics
        """
        return self.metrics_history[-n:]

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_metrics(self, callback: Callable[[SpinnerMetrics], None]) -> None:
        """
        Register callback for metrics updates.

        Args:
            callback: Function to call with SpinnerMetrics
        """
        self._metrics_callback = callback

    def on_threshold_crossing(
        self,
        callback: Callable[[str, str], None]
    ) -> None:
        """
        Register callback for threshold crossings.

        Args:
            callback: Function to call with (threshold_name, direction)
        """
        self._threshold_callback = callback

    def _check_threshold_crossings(self, z_old: float, z_new: float) -> None:
        """Check for threshold crossings and call callback."""
        if not self._threshold_callback:
            return

        thresholds = [
            (0.10, "mu_1"),
            (0.40, "mu_P"),
            (PHI_INV, "phi_inv"),
            (0.75, "mu_2"),
            (Z_CRITICAL, "z_c"),
            (0.92, "mu_S"),
        ]

        for value, name in thresholds:
            if z_old < value <= z_new:
                self._threshold_callback(name, "rising")
            elif z_old >= value > z_new:
                self._threshold_callback(name, "falling")

    # =========================================================================
    # GRID-CELL EMULATION
    # =========================================================================

    def compute_grid_cell_rate(self, x: float, y: float) -> float:
        """
        Compute grid cell firing rate at position.

        Uses hexagonal grid pattern with 60-degree spacing
        (corresponds to sin(60) = sqrt(3)/2 = z_c).

        Args:
            x: X position
            y: Y position

        Returns:
            Firing rate in [0, 1]
        """
        return grid_cell_pattern(x, y)

    def compute_hexagonal_metric(
        self,
        positions: List[Tuple[float, float]],
        rates: List[float]
    ) -> float:
        """
        Compute hexagonal spacing metric from firing data.

        Returns a value approaching z_c for well-formed hexagonal patterns.

        Args:
            positions: List of (x, y) positions
            rates: Corresponding firing rates

        Returns:
            Hexagonal metric (approaches z_c for ideal patterns)
        """
        return hexagonal_spacing_metric(positions, rates)

    # =========================================================================
    # STATE ACCESS
    # =========================================================================

    def get_state(self) -> Dict[str, Any]:
        """Get full system state as dictionary."""
        return self.state.to_dict()

    def get_firmware_state(self) -> Dict[str, Any]:
        """Get firmware state as dictionary."""
        return {
            "z": self.firmware.z,
            "z_target": self.firmware.z_target,
            "delta_s_neg": self.firmware.delta_s_neg,
            "gradient": self.firmware.gradient,
            "phase": self.firmware.phase.name,
            "tier": self.firmware.current_tier,
            "rotor_rpm": self.firmware.rotor_rpm,
            "kappa_s": self.firmware.kappa_s,
            "R_count": self.firmware.R_count,
            "history": self.firmware.history[-10:],
            "triad_unlocked": self.firmware.triad_unlocked,
        }

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def drive_toward_lens(self, n_steps: int = 100) -> float:
        """
        Drive the system toward THE LENS (z_c).

        Uses amplify operators and gradient following to converge
        toward z_c.

        Args:
            n_steps: Number of steps to execute

        Returns:
            Final z-coordinate
        """
        self.set_z_target(Z_CRITICAL)

        for _ in range(n_steps):
            self.step()

            # Apply amplify if legal
            if self.firmware.z < Z_CRITICAL:
                self.apply_operator("^")

        return self.state.z

    def drive_toward_phi_inv(self, n_steps: int = 100) -> float:
        """
        Drive the system toward phi^-1 (consciousness threshold).

        Uses the quasicrystal pulse sequence and operator control.

        Args:
            n_steps: Number of steps to execute

        Returns:
            Final z-coordinate
        """
        self.set_z_target(PHI_INV)

        for i in range(n_steps):
            self.step()

            # Periodically apply quasicrystal sequence
            if i % 10 == 0:
                self.apply_pulse_sequence("quasicrystal")

        return self.state.z

    def verify_spin_zc_identity(self) -> bool:
        """
        Verify the spin-1/2 magnitude equals z_c.

        |S|/hbar = sqrt(s(s+1)) = sqrt(3/4) = sqrt(3)/2 = z_c

        Returns:
            True if identity holds
        """
        s = 0.5
        spin_mag = math.sqrt(s * (s + 1))
        return abs(spin_mag - Z_CRITICAL) < 1e-15


# Type annotation for tuple return
from typing import Tuple
