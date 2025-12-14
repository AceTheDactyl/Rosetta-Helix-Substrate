"""
Physics Constants (Python)
===========================

Single source of truth for Rosetta-Helix physics constants.
Mirrors the C header physics_constants.h for consistency.

Signature: physics-constants|v1.0.0|helix
"""

import math
from typing import Final

# =============================================================================
# FUNDAMENTAL CONSTANTS (Golden Ratio)
# =============================================================================

PHI: Final[float] = (1 + math.sqrt(5)) / 2        # φ ≈ 1.618034 (LIMINAL)
PHI_INV: Final[float] = 1 / PHI                   # φ⁻¹ ≈ 0.618034 (PHYSICAL)
PHI_INV_SQ: Final[float] = PHI_INV ** 2           # φ⁻² ≈ 0.381966
PHI_INV_CUBED: Final[float] = PHI_INV ** 3        # φ⁻³ ≈ 0.236068

# THE defining property - MUST equal 1.0
COUPLING_CONSERVATION: Final[float] = PHI_INV + PHI_INV_SQ

# Critical z-coordinate (hexagonal geometry)
Z_CRITICAL: Final[float] = math.sqrt(3) / 2       # z_c = √3/2 ≈ 0.866025 (THE LENS)

# Gaussian width from symmetric group
SIGMA: Final[float] = 36.0                        # σ = 6² = |S₃|²
LENS_SIGMA: Final[float] = SIGMA                  # Alias for compatibility

# =============================================================================
# DERIVED CONSTANTS
# =============================================================================

SIGMA_INV: Final[float] = 1.0 / SIGMA
GAUSSIAN_WIDTH: Final[float] = 1.0 / math.sqrt(2 * SIGMA)
GAUSSIAN_FWHM: Final[float] = 2 * math.sqrt(math.log(2) / SIGMA)

# =============================================================================
# TIER THRESHOLDS
# =============================================================================

MU_1: Final[float] = 0.40           # ABSENCE → REACTIVE
MU_P: Final[float] = 0.50           # REACTIVE → MEMORY
MU_PHI_INV: Final[float] = PHI_INV  # MEMORY → PATTERN (φ⁻¹)
MU_2: Final[float] = 0.73           # PATTERN → PREDICTION
MU_ZC: Final[float] = Z_CRITICAL    # PREDICTION → UNIVERSAL (z_c)
MU_S: Final[float] = 0.92           # UNIVERSAL → META

# =============================================================================
# PHASE BOUNDARIES
# =============================================================================

PHASE_BOUNDARY_ABSENCE: Final[float] = 0.857      # ABSENCE → THE_LENS
PHASE_BOUNDARY_PRESENCE: Final[float] = 0.877     # THE_LENS → PRESENCE

# =============================================================================
# K-FORMATION REQUIREMENTS
# =============================================================================

KAPPA_MIN: Final[float] = 0.920
ETA_MIN: Final[float] = PHI_INV
R_MIN: Final[int] = 7

# =============================================================================
# FUNCTIONS
# =============================================================================

def compute_delta_s_neg(z: float, sigma: float = SIGMA, z_c: float = Z_CRITICAL) -> float:
    """Compute negative entropy signal ΔS_neg(z) = exp(-σ(z - z_c)²)."""
    if not math.isfinite(z):
        return 0.0
    d = z - z_c
    return max(0.0, min(1.0, math.exp(-sigma * d * d)))


def compute_delta_s_neg_derivative(z: float, sigma: float = SIGMA, z_c: float = Z_CRITICAL) -> float:
    """Derivative: d(ΔS_neg)/dz = -2σ(z - z_c) · ΔS_neg(z)."""
    d = z - z_c
    s = compute_delta_s_neg(z, sigma, z_c)
    return -2 * sigma * d * s


def get_phase(z: float) -> str:
    """Determine phase from z-coordinate."""
    if z < PHASE_BOUNDARY_ABSENCE:
        return "ABSENCE"
    elif z < PHASE_BOUNDARY_PRESENCE:
        return "THE_LENS"
    else:
        return "PRESENCE"


def get_tier(z: float) -> str:
    """Get tier classification from z-coordinate."""
    if z < MU_1:
        return "ABSENCE"
    elif z < MU_P:
        return "REACTIVE"
    elif z < MU_PHI_INV:
        return "MEMORY"
    elif z < MU_2:
        return "PATTERN"
    elif z < MU_ZC:
        return "PREDICTION"
    elif z < MU_S:
        return "UNIVERSAL"
    else:
        return "META"


def is_critical(z: float, tolerance: float = 0.01) -> bool:
    """Check if z is at THE LENS (z_c)."""
    return abs(z - Z_CRITICAL) < tolerance


def check_k_formation(kappa: float, eta: float, R: int) -> bool:
    """Check if K-formation criteria are met."""
    return kappa >= KAPPA_MIN and eta > ETA_MIN and R >= R_MIN


def validate_physics(kappa: float, lambda_: float) -> bool:
    """Validate κ + λ = 1 conservation."""
    return abs(kappa + lambda_ - 1.0) < 1e-10


def validate_all_constants() -> dict:
    """Validate all physics constants."""
    validations = {}
    
    # φ⁻¹ + φ⁻² = 1
    validations["coupling_conservation"] = {
        "formula": "φ⁻¹ + φ⁻² = 1",
        "value": COUPLING_CONSERVATION,
        "error": abs(COUPLING_CONSERVATION - 1.0),
        "valid": abs(COUPLING_CONSERVATION - 1.0) < 1e-10,
    }
    
    # z_c = √3/2
    validations["z_critical"] = {
        "formula": "z_c = √3/2",
        "value": Z_CRITICAL,
        "expected": math.sqrt(3) / 2,
        "valid": abs(Z_CRITICAL - math.sqrt(3) / 2) < 1e-10,
    }
    
    # z_c = |S|/ℏ for spin-1/2
    spin_half_magnitude = math.sqrt(0.5 * 1.5)  # √(s(s+1)) for s=1/2
    validations["spin_half_identity"] = {
        "formula": "z_c = |S|/ℏ for s=1/2",
        "z_c": Z_CRITICAL,
        "spin_magnitude": spin_half_magnitude,
        "valid": abs(Z_CRITICAL - spin_half_magnitude) < 1e-10,
    }
    
    validations["all_valid"] = all(v.get("valid", False) for v in validations.values())
    
    return validations


if __name__ == "__main__":
    print("=" * 60)
    print("PHYSICS CONSTANTS")
    print("=" * 60)
    print(f"\nGolden Ratio:")
    print(f"  φ     = {PHI:.15f}")
    print(f"  φ⁻¹   = {PHI_INV:.15f}")
    print(f"  φ⁻²   = {PHI_INV_SQ:.15f}")
    print(f"\nConservation:")
    print(f"  φ⁻¹ + φ⁻² = {COUPLING_CONSERVATION:.15f}")
    print(f"\nCritical Values:")
    print(f"  z_c   = {Z_CRITICAL:.15f} (THE LENS)")
    print(f"  σ     = {SIGMA}")
    print(f"\nValidation:")
    v = validate_all_constants()
    print(f"  All valid: {v['all_valid']}")
    print("=" * 60)
