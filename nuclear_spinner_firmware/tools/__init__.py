"""
Nuclear Spinner Python Tools
============================

Physics constants and extended computations for the Nuclear Spinner.

Usage:
    from tools import constants
    from tools.extended_physics_constants_v1 import QuasicrystalState

Signature: nuclear-spinner-tools|v1.0.0|helix
"""

from .constants import (
    PHI,
    PHI_INV,
    PHI_INV_SQ,
    Z_CRITICAL,
    SIGMA,
    LENS_SIGMA,
    compute_delta_s_neg,
    compute_delta_s_neg_derivative,
    get_phase,
    get_tier,
    is_critical,
    check_k_formation,
    validate_all_constants,
)

__all__ = [
    "PHI",
    "PHI_INV",
    "PHI_INV_SQ",
    "Z_CRITICAL",
    "SIGMA",
    "LENS_SIGMA",
    "compute_delta_s_neg",
    "compute_delta_s_neg_derivative",
    "get_phase",
    "get_tier",
    "is_critical",
    "check_k_formation",
    "validate_all_constants",
]

__version__ = "1.0.0"
