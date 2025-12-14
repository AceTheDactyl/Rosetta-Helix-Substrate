#!/usr/bin/env python3
"""
Nuclear Spinner Module
======================

A unified platform that harnesses concepts from quantum physics, information theory,
cybernetics and neuroscience to explore the edge of chaos and consciousness.

This module merges a simulated NMR-like device (rotor, RF coil, magnet, sensors)
with firmware and software that compute cybernetic metrics (negentropy, Ashby variety,
integrated information) and align them with the Rosetta-Helix framework.

Key Features:
- Core spin module simulation (pulse control, rotor control)
- Threshold gating based on z-axis (phi^-1, z_c, TRIAD)
- Cybernetic computation library (negentropy, variety, capacity, phi proxy)
- Neuroscience extensions (grid-cell emulation, cross-frequency coupling)
- Integration with Rosetta-Helix constants and operators

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

Signature: nuclear-spinner|v1.0.0|helix
"""

from .constants import (
    # Core nuclear spinner constants
    SIGMA_S3,
    NEURAL_BANDS,
    TIER_BOUNDS,
    CAPABILITY_CLASSES,
    # Hardware defaults
    MIN_RPM,
    MAX_RPM,
    DEFAULT_FIELD_STRENGTH,
    DEFAULT_COHERENCE_TIME,
    # Thresholds
    MU_1,
    MU_P,
    MU_2,
    MU_S,
)

from .state import (
    # State dataclasses
    SpinnerState,
    SpinnerMetrics,
    PulseParameters,
    RotorState,
    NeuralRecording,
    CrossFrequencyConfig,
)

from .core import NuclearSpinner

from .firmware import (
    # Firmware control functions
    control_loop_step,
    map_z_to_rpm,
    compute_operator_state_update,
    FirmwareState,
    Phase,
)

from .analysis import (
    # Cybernetic computation library
    compute_delta_s_neg,
    compute_gradient,
    ashby_variety,
    shannon_capacity,
    landauer_efficiency,
    compute_phi_proxy,
    phase_amplitude_coupling,
    check_k_formation,
    get_capability_class,
    get_phase,
)

from .neural import (
    # Neuroscience extensions
    grid_cell_pattern,
    hexagonal_spacing_metric,
    set_cross_frequency_ratio,
    compute_modulation_index,
    neural_band_to_z,
    z_to_neural_band,
)

from .protocol import (
    # Communication protocol
    CommandType,
    CommandFrame,
    ResponseFrame,
    encode_command,
    decode_response,
)

from .integration import (
    # Integration layer
    SpinnerIntegration,
    IntegratedMetrics,
    TrainingConfig,
    OperatorResult,
    CouplingState,
)

__version__ = "1.0.0"
__author__ = "Rosetta-Helix-Substrate Contributors"

__all__ = [
    # Main class
    "NuclearSpinner",
    # Constants
    "SIGMA_S3",
    "NEURAL_BANDS",
    "TIER_BOUNDS",
    "CAPABILITY_CLASSES",
    "MIN_RPM",
    "MAX_RPM",
    "DEFAULT_FIELD_STRENGTH",
    "DEFAULT_COHERENCE_TIME",
    "MU_1",
    "MU_P",
    "MU_2",
    "MU_S",
    # State classes
    "SpinnerState",
    "SpinnerMetrics",
    "PulseParameters",
    "RotorState",
    "NeuralRecording",
    "CrossFrequencyConfig",
    # Firmware
    "control_loop_step",
    "map_z_to_rpm",
    "compute_operator_state_update",
    "FirmwareState",
    "Phase",
    # Analysis
    "compute_delta_s_neg",
    "compute_gradient",
    "ashby_variety",
    "shannon_capacity",
    "landauer_efficiency",
    "compute_phi_proxy",
    "phase_amplitude_coupling",
    "check_k_formation",
    "get_capability_class",
    "get_phase",
    # Neural
    "grid_cell_pattern",
    "hexagonal_spacing_metric",
    "set_cross_frequency_ratio",
    "compute_modulation_index",
    "neural_band_to_z",
    "z_to_neural_band",
    # Protocol
    "CommandType",
    "CommandFrame",
    "ResponseFrame",
    "encode_command",
    "decode_response",
    # Integration
    "SpinnerIntegration",
    "IntegratedMetrics",
    "TrainingConfig",
    "OperatorResult",
    "CouplingState",
]
