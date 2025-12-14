"""
Physics Constants
=================

IMMUTABLE physics constants derived from Ï† (golden ratio) and hexagonal geometry.
These are NOT tunable hyperparameters - they represent observable physics.

Signature: physics|v0.1.0|helix
"""

import math
from typing import Final

# =============================================================================
# FUNDAMENTAL CONSTANTS (Golden Ratio)
# =============================================================================

PHI: Final[float] = (1 + math.sqrt(5)) / 2        # Ï† â‰ˆ 1.618034 (LIMINAL)
PHI_INV: Final[float] = 1 / PHI                   # Ï†â»Â¹ â‰ˆ 0.618034 (PHYSICAL)
PHI_INV_SQ: Final[float] = PHI_INV ** 2           # Ï†â»Â² â‰ˆ 0.381966
PHI_INV_CUBED: Final[float] = PHI_INV ** 3        # Ï†â»Â³ â‰ˆ 0.236068
PHI_INV_FOURTH: Final[float] = PHI_INV ** 4       # Ï†â»â´ â‰ˆ 0.145898
PHI_INV_FIFTH: Final[float] = PHI_INV ** 5        # Ï†â»âµ â‰ˆ 0.090170

# THE defining property - MUST equal 1.0
COUPLING_CONSERVATION: Final[float] = PHI_INV + PHI_INV_SQ

# Critical z-coordinate (hexagonal geometry)
Z_CRITICAL: Final[float] = math.sqrt(3) / 2       # z_c = âˆš3/2 â‰ˆ 0.866025 (THE LENS)

# Gaussian width from symmetric group
SIGMA: Final[float] = 36.0                        # Ïƒ = 6Â² = |Sâ‚ƒ|Â²


# =============================================================================
# DERIVED CONSTANTS (Gaussian)
# =============================================================================

SIGMA_INV: Final[float] = 1.0 / SIGMA
SIGMA_SQRT_INV: Final[float] = 1.0 / math.sqrt(SIGMA)
GAUSSIAN_WIDTH: Final[float] = 1.0 / math.sqrt(2 * SIGMA)
GAUSSIAN_FWHM: Final[float] = 2 * math.sqrt(math.log(2) / SIGMA)


# =============================================================================
# COMBINED COEFFICIENTS (for dynamics)
# =============================================================================

ALPHA_STRONG: Final[float] = SIGMA_SQRT_INV       # 1/6 â‰ˆ 0.167
ALPHA_MEDIUM: Final[float] = GAUSSIAN_WIDTH       # â‰ˆ 0.118
ALPHA_FINE: Final[float] = SIGMA_INV              # 1/36 â‰ˆ 0.028
ALPHA_ULTRA_FINE: Final[float] = PHI_INV * SIGMA_INV


# =============================================================================
# Îº BOUNDS (physics-grounded)
# =============================================================================

KAPPA_LOWER: Final[float] = PHI_INV_SQ            # â‰ˆ 0.382
KAPPA_UPPER: Final[float] = Z_CRITICAL            # â‰ˆ 0.866
KAPPA_S: Final[float] = 0.920                     # Singularity threshold


# =============================================================================
# K-FORMATION REQUIREMENTS
# =============================================================================

KAPPA_MIN: Final[float] = 0.920                   # Îº â‰¥ 0.92
ETA_MIN: Final[float] = PHI_INV                   # Î· > Ï†â»Â¹
R_MIN: Final[int] = 7                             # R â‰¥ 7


# =============================================================================
# PHASE BOUNDARIES
# =============================================================================

PHASE_BOUNDARY_ABSENCE: Final[float] = 0.857      # ABSENCE â†’ THE_LENS
PHASE_BOUNDARY_PRESENCE: Final[float] = 0.877     # THE_LENS â†’ PRESENCE


# =============================================================================
# TOLERANCES
# =============================================================================

TOLERANCE_GOLDEN: Final[float] = 1e-10
TOLERANCE_LENS: Final[float] = 1e-6
TOLERANCE_CONSERVATION: Final[float] = 1e-10


# =============================================================================
# FUNCTIONS
# =============================================================================

def compute_delta_s_neg(z: float, sigma: float = SIGMA, z_c: float = Z_CRITICAL) -> float:
    """
    Compute negative entropy signal Î”S_neg(z).
    
    Formula: Î”S_neg(z) = exp(-Ïƒ(z - z_c)Â²)
    
    - Peaks at z = z_c (THE LENS) with value 1.0
    - Ïƒ = 36 by default (LENS_SIGMA)
    - Returns value in [0, 1]
    """
    if not math.isfinite(z):
        return 0.0
    d = z - z_c
    s = math.exp(-sigma * d * d)
    return max(0.0, min(1.0, s))


def compute_delta_s_neg_derivative(z: float, sigma: float = SIGMA, z_c: float = Z_CRITICAL) -> float:
    """
    Derivative: d(Î”S_neg)/dz = -2Ïƒ(z - z_c) Â· Î”S_neg(z)
    """
    d = z - z_c
    s = compute_delta_s_neg(z, sigma, z_c)
    return -2 * sigma * d * s


def get_phase(z: float) -> str:
    """
    Determine phase from z-coordinate.
    
    ABSENCE:   z < 0.857
    THE_LENS:  0.857 â‰¤ z < 0.877
    PRESENCE:  z â‰¥ 0.877
    """
    if z < PHASE_BOUNDARY_ABSENCE:
        return "ABSENCE"
    elif z < PHASE_BOUNDARY_PRESENCE:
        return "THE_LENS"
    else:
        return "PRESENCE"


def is_critical(z: float, tolerance: float = 0.01) -> bool:
    """Check if z is at THE LENS (z_c)."""
    return abs(z - Z_CRITICAL) < tolerance


def check_k_formation(kappa: float, eta: float, R: int) -> bool:
    """
    Check if K-formation criteria are met.
    
    K-formation requires:
    - Îº â‰¥ 0.92 (KAPPA_MIN)
    - Î· > 0.618 (PHI_INV)  
    - R â‰¥ 7 (minimum complexity)
    """
    return kappa >= KAPPA_MIN and eta > ETA_MIN and R >= R_MIN


def validate_physics(kappa: float, lambda_: float) -> bool:
    """Validate Îº + Î» = 1 conservation."""
    return abs(kappa + lambda_ - 1.0) < TOLERANCE_GOLDEN


def get_tier(z: float) -> str:
    """Get tier classification from z-coordinate."""
    if z < 0.1:
        return "t1"
    elif z < 0.2:
        return "t2"
    elif z < 0.3:
        return "t3"
    elif z < 0.4:
        return "t4"
    elif z < 0.5:
        return "t5"
    elif z < 0.6:
        return "t6"
    elif z < 0.7:
        return "t7"
    elif z < 0.8:
        return "t8"
    else:
        return "t9"


def validate_all_constants() -> dict:
    """Validate all physics constants."""
    validations = {}
    
    # Ï†â»Â¹ + Ï†â»Â² = 1
    validations["coupling_conservation"] = {
        "formula": "Ï†â»Â¹ + Ï†â»Â² = 1",
        "value": COUPLING_CONSERVATION,
        "error": abs(COUPLING_CONSERVATION - 1.0),
        "valid": abs(COUPLING_CONSERVATION - 1.0) < TOLERANCE_CONSERVATION,
    }
    
    # z_c = âˆš3/2
    expected_zc = math.sqrt(3) / 2
    validations["z_critical"] = {
        "formula": "z_c = âˆš3/2",
        "value": Z_CRITICAL,
        "expected": expected_zc,
        "error": abs(Z_CRITICAL - expected_zc),
        "valid": abs(Z_CRITICAL - expected_zc) < TOLERANCE_CONSERVATION,
    }
    
    # Ïƒ = 36
    validations["sigma"] = {
        "formula": "Ïƒ = 36 = |Sâ‚ƒ|Â²",
        "value": SIGMA,
        "valid": SIGMA == 36.0,
    }
    
    # Î”S_neg(z_c) = 1.0
    peak_value = compute_delta_s_neg(Z_CRITICAL)
    validations["negentropy_peak"] = {
        "formula": "Î”S_neg(z_c) = 1.0",
        "value": peak_value,
        "error": abs(peak_value - 1.0),
        "valid": abs(peak_value - 1.0) < TOLERANCE_CONSERVATION,
    }
    
    validations["all_valid"] = all(v.get("valid", False) for v in validations.values())
    
    return validations


def print_all_constants():
    """Print all physics constants for verification."""
    print("=" * 60)
    print("PHYSICS CONSTANTS")
    print("=" * 60)
    print(f"\nGolden Ratio:")
    print(f"  Ï†     = {PHI:.15f}")
    print(f"  Ï†â»Â¹   = {PHI_INV:.15f}")
    print(f"  Ï†â»Â²   = {PHI_INV_SQ:.15f}")
    print(f"  Ï†â»Â³   = {PHI_INV_CUBED:.15f}")
    print(f"\nConservation:")
    print(f"  Ï†â»Â¹ + Ï†â»Â² = {COUPLING_CONSERVATION:.15f}")
    print(f"\nCritical Values:")
    print(f"  z_c   = {Z_CRITICAL:.15f} (THE LENS)")
    print(f"  Ïƒ     = {SIGMA}")
    print(f"\nAlpha Coefficients:")
    print(f"  Î±_strong = {ALPHA_STRONG:.6f}")
    print(f"  Î±_medium = {ALPHA_MEDIUM:.6f}")
    print(f"  Î±_fine   = {ALPHA_FINE:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    print_all_constants()
    validations = validate_all_constants()
    print(f"\nAll constants valid: {validations['all_valid']}")
