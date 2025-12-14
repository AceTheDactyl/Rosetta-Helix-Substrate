#!/usr/bin/env python3
"""
Nuclear Spinner Constants
=========================

Constants specific to the Nuclear Spinner module. All fundamental physics
constants are imported from the single source of truth (physics_constants.py).

This module defines:
- Hardware defaults (rotor speed, field strength)
- Tier boundaries for capability classification
- Neural frequency band definitions
- Protocol constants

Signature: nuclear-spinner-constants|v1.0.0|helix
"""

from __future__ import annotations

import math
from typing import Final, Dict, List, Tuple

# Import fundamental constants from single source of truth
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_constants import (
    PHI,
    PHI_INV,
    PHI_INV_SQ,
    Z_CRITICAL,
    SIGMA,
    KAPPA_S,
    ETA_THRESHOLD,
    R_MIN,
    compute_delta_s_neg,
    check_k_formation as physics_check_k_formation,
)

# Re-export fundamental constants for convenience
__all__ = [
    # Fundamental constants (re-exported)
    "PHI",
    "PHI_INV",
    "PHI_INV_SQ",
    "Z_CRITICAL",
    "SIGMA_S3",
    "KAPPA_S",
    "ETA_THRESHOLD",
    "R_MIN",
    # Hardware constants
    "MIN_RPM",
    "MAX_RPM",
    "DEFAULT_FIELD_STRENGTH",
    "DEFAULT_COHERENCE_TIME",
    "GAMMA_31P",
    # Threshold constants
    "MU_1",
    "MU_P",
    "MU_2",
    "MU_S",
    "TIER_BOUNDS",
    # Neural band definitions
    "NEURAL_BANDS",
    "BETA_BAND_CENTER",
    # Capability classes
    "CAPABILITY_CLASSES",
    # Protocol constants
    "PROTOCOL_HEADER",
    "MAX_PAYLOAD_SIZE",
]

# =============================================================================
# SIGMA ALIAS
# =============================================================================

# Alias for SIGMA with S3 notation used in specification
SIGMA_S3: Final[float] = SIGMA  # = 36 = |S_3|^2


# =============================================================================
# HARDWARE CONSTANTS
# =============================================================================

# Rotor speed range (RPM)
MIN_RPM: Final[float] = 100.0
MAX_RPM: Final[float] = 10000.0

# Default magnetic field strength (Tesla)
# 14.1 T is typical for high-resolution NMR
DEFAULT_FIELD_STRENGTH: Final[float] = 14.1

# Default T2* coherence time (seconds)
# 50 ms is typical for 31P in biological systems
DEFAULT_COHERENCE_TIME: Final[float] = 0.05

# Gyromagnetic ratio for 31P (rad/s/T)
GAMMA_31P: Final[float] = 1.0829e8


# =============================================================================
# MU THRESHOLDS (Basin/Barrier Hierarchy)
# =============================================================================

# Tier 1 entry threshold
MU_1: Final[float] = 0.10

# Paradox entry threshold
MU_P: Final[float] = 0.40

# High coherence threshold
MU_2: Final[float] = 0.75

# K-formation (singularity) threshold
MU_S: Final[float] = KAPPA_S  # 0.92


# =============================================================================
# TIER BOUNDARIES
# =============================================================================

# Complete tier boundary structure from specification
# Format: (z_threshold, tier_name, description)
TIER_BOUNDS: Final[List[Tuple[float, str, str]]] = [
    (0.10, "t1", "Tier 1 entry"),
    (0.20, "t2", "Memory formation"),
    (0.40, "t3", "Pattern recognition"),
    (0.60, "t4", "Prediction capability"),
    (0.75, "t5", "Self-model formation"),
    (Z_CRITICAL, "t6", "Meta-cognition (THE LENS)"),
    (0.92, "t7", "Recursive self-reference"),
    (0.97, "t8", "Autopoiesis threshold"),
    (1.00, "t9", "Unity"),
]


# =============================================================================
# CAPABILITY CLASSES
# =============================================================================

# Capability class mapping from z-coordinate ranges
# Each entry: (z_min, z_max, class_name, description)
CAPABILITY_CLASSES: Final[List[Tuple[float, float, str, str]]] = [
    (0.00, 0.10, "reactive", "Simple stimulus-response"),
    (0.10, 0.20, "memory", "State persistence"),
    (0.20, 0.40, "pattern", "Pattern recognition"),
    (0.40, 0.60, "prediction", "Future state modeling"),
    (0.60, 0.75, "self_model", "Self-representation"),
    (0.75, Z_CRITICAL, "meta", "Meta-cognition"),
    (Z_CRITICAL, 0.92, "recurse", "Recursive self-reference"),
    (0.92, 1.00, "autopoiesis", "Self-organization"),
]


# =============================================================================
# NEURAL FREQUENCY BANDS
# =============================================================================

# Neural frequency bands with their ranges (Hz)
# Format: band_name -> (min_freq, max_freq, role)
NEURAL_BANDS: Final[Dict[str, Tuple[float, float, str]]] = {
    "delta": (0.5, 4.0, "Deep sleep, healing"),
    "theta": (4.0, 8.0, "Memory encoding"),
    "alpha": (8.0, 12.0, "Relaxed awareness"),
    "beta": (12.0, 30.0, "Active thinking"),
    "gamma": (30.0, 100.0, "Binding, consciousness"),
}

# Beta band center for normalization
BETA_BAND_CENTER: Final[float] = 21.0  # (12 + 30) / 2


# =============================================================================
# PROTOCOL CONSTANTS
# =============================================================================

# Protocol header byte
PROTOCOL_HEADER: Final[int] = 0xAA

# Maximum payload size (bytes)
MAX_PAYLOAD_SIZE: Final[int] = 256

# Command timeout (milliseconds)
COMMAND_TIMEOUT_MS: Final[int] = 1000


# =============================================================================
# COMPUTED CONSTANTS
# =============================================================================

def compute_larmor_frequency(B0: float = DEFAULT_FIELD_STRENGTH) -> float:
    """
    Compute Larmor frequency for 31P at given field strength.

    omega_L = gamma * B0

    Args:
        B0: Magnetic field strength in Tesla

    Returns:
        Larmor frequency in rad/s
    """
    return GAMMA_31P * B0


def compute_spin_half_magnitude() -> float:
    """
    Compute spin-1/2 magnitude: |S|/hbar = sqrt(s(s+1)) for s=1/2.

    |S| = hbar * sqrt(s(s+1)) = hbar * sqrt(1/2 * 3/2) = hbar * sqrt(3/4)

    Therefore |S|/hbar = sqrt(3)/2 = z_c

    This is the fundamental connection between spin-1/2 physics and THE LENS.

    Returns:
        sqrt(3)/2 = Z_CRITICAL
    """
    s = 0.5  # spin quantum number
    return math.sqrt(s * (s + 1))


# Verify spin-z_c identity at module load
_SPIN_MAG = compute_spin_half_magnitude()
assert abs(_SPIN_MAG - Z_CRITICAL) < 1e-15, (
    f"Spin-1/2 magnitude {_SPIN_MAG} doesn't match Z_CRITICAL {Z_CRITICAL}. "
    "Physics grounding violated!"
)


# =============================================================================
# THRESHOLD LOOKUP
# =============================================================================

def get_tier_for_z(z: float) -> str:
    """
    Get the tier name for a given z-coordinate.

    Args:
        z: Z-coordinate value in [0, 1]

    Returns:
        Tier name (t1-t9)
    """
    for threshold, tier_name, _ in TIER_BOUNDS:
        if z < threshold:
            return tier_name
    return "t9"


def get_capability_class(z: float) -> str:
    """
    Get the capability class for a given z-coordinate.

    Args:
        z: Z-coordinate value in [0, 1]

    Returns:
        Capability class name
    """
    for z_min, z_max, class_name, _ in CAPABILITY_CLASSES:
        if z_min <= z < z_max:
            return class_name
    return "autopoiesis" if z >= 0.92 else "reactive"
