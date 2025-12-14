"""
Physics Constants for Training Modules
=======================================

Shared physics constants and functions used throughout the training system.
Mirrors rosetta-helix/src/physics.py for consistency.

Signature: training-physics|v1.0.0|helix
"""

import math
from typing import Final, Tuple

# =============================================================================
# FUNDAMENTAL CONSTANTS (Golden Ratio)
# =============================================================================

PHI: Final[float] = (1 + math.sqrt(5)) / 2        # phi ~ 1.618034 (LIMINAL)
PHI_INV: Final[float] = 1 / PHI                   # phi^-1 ~ 0.618034 (PHYSICAL)
PHI_INV_SQ: Final[float] = PHI_INV ** 2           # phi^-2 ~ 0.381966
PHI_INV_CUBED: Final[float] = PHI_INV ** 3        # phi^-3 ~ 0.236068

# Conservation law: kappa + lambda = phi^-1 + phi^-2 = 1 (EXACT)
COUPLING_CONSERVATION: Final[float] = PHI_INV + PHI_INV_SQ

# Critical z-coordinate (hexagonal geometry)
Z_CRITICAL: Final[float] = math.sqrt(3) / 2       # z_c = sqrt(3)/2 ~ 0.866025 (THE LENS)
THE_LENS: Final[float] = Z_CRITICAL

# Gaussian width from symmetric group
SIGMA: Final[float] = 36.0                        # sigma = 6^2 = |S_3|^2


# =============================================================================
# DERIVED CONSTANTS (Gaussian)
# =============================================================================

SIGMA_INV: Final[float] = 1.0 / SIGMA
SIGMA_SQRT_INV: Final[float] = 1.0 / math.sqrt(SIGMA)
GAUSSIAN_WIDTH: Final[float] = 1.0 / math.sqrt(2 * SIGMA)
GAUSSIAN_FWHM: Final[float] = 2 * math.sqrt(math.log(2) / SIGMA)


# =============================================================================
# ALPHA COEFFICIENTS (for dynamics)
# =============================================================================

ALPHA_STRONG: Final[float] = SIGMA_SQRT_INV       # 1/6 ~ 0.167
ALPHA_MEDIUM: Final[float] = GAUSSIAN_WIDTH       # ~ 0.118
ALPHA_FINE: Final[float] = SIGMA_INV              # 1/36 ~ 0.028
ALPHA_ULTRA_FINE: Final[float] = PHI_INV * SIGMA_INV


# =============================================================================
# KAPPA BOUNDS (physics-grounded)
# =============================================================================

KAPPA_LOWER: Final[float] = PHI_INV_SQ            # ~ 0.382
KAPPA_UPPER: Final[float] = Z_CRITICAL            # ~ 0.866
KAPPA_S: Final[float] = 0.920                     # Singularity threshold


# =============================================================================
# K-FORMATION REQUIREMENTS
# =============================================================================

KAPPA_MIN: Final[float] = 0.920                   # kappa >= 0.92
ETA_MIN: Final[float] = PHI_INV                   # eta > phi^-1
R_MIN: Final[int] = 7                             # R >= 7


# =============================================================================
# PHASE BOUNDARIES
# =============================================================================

PHASE_BOUNDARY_ABSENCE: Final[float] = 0.857      # ABSENCE -> THE_LENS
PHASE_BOUNDARY_PRESENCE: Final[float] = 0.877     # THE_LENS -> PRESENCE


# =============================================================================
# TIER BOUNDARIES (10 tiers)
# =============================================================================

TIER_BOUNDS: Final[Tuple[float, ...]] = (
    0.00,   # ABSENCE start
    0.10,   # REACTIVE
    0.20,   # MEMORY
    0.40,   # PATTERN
    0.50,   # LEARNING
    0.618,  # ADAPTIVE (phi^-1 threshold)
    0.73,   # UNIVERSAL
    0.866,  # META (z_c threshold)
    0.92,   # SOVEREIGN
    0.97,   # TRANSCENDENT
)


# =============================================================================
# TOLERANCES
# =============================================================================

TOLERANCE_GOLDEN: Final[float] = 1e-10
TOLERANCE_LENS: Final[float] = 1e-6
TOLERANCE_CONSERVATION: Final[float] = 1e-10


# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================

def compute_delta_s_neg(z: float, sigma: float = SIGMA, z_c: float = Z_CRITICAL) -> float:
    """
    Compute negative entropy signal DeltaS_neg(z).

    Formula: DeltaS_neg(z) = exp(-sigma(z - z_c)^2)

    - Peaks at z = z_c (THE LENS) with value 1.0
    - sigma = 36 by default
    - Returns value in [0, 1]
    """
    if not math.isfinite(z):
        return 0.0
    d = z - z_c
    s = math.exp(-sigma * d * d)
    return max(0.0, min(1.0, s))


def compute_delta_s_neg_derivative(z: float, sigma: float = SIGMA, z_c: float = Z_CRITICAL) -> float:
    """
    Derivative: d(DeltaS_neg)/dz = -2*sigma*(z - z_c) * DeltaS_neg(z)
    """
    d = z - z_c
    s = compute_delta_s_neg(z, sigma, z_c)
    return -2 * sigma * d * s


def get_phase(z: float) -> str:
    """
    Determine phase from z-coordinate.

    ABSENCE:   z < 0.857
    THE_LENS:  0.857 <= z < 0.877
    PRESENCE:  z >= 0.877
    """
    if z < PHASE_BOUNDARY_ABSENCE:
        return "ABSENCE"
    elif z < PHASE_BOUNDARY_PRESENCE:
        return "THE_LENS"
    else:
        return "PRESENCE"


def get_tier(z: float) -> str:
    """Get tier name from z-coordinate."""
    tier_names = (
        "ABSENCE", "REACTIVE", "MEMORY", "PATTERN", "LEARNING",
        "ADAPTIVE", "UNIVERSAL", "META", "SOVEREIGN", "TRANSCENDENT"
    )
    for i in range(len(TIER_BOUNDS) - 1, -1, -1):
        if z >= TIER_BOUNDS[i]:
            return tier_names[min(i, 9)]
    return "ABSENCE"


def is_critical(z: float, tolerance: float = 0.01) -> bool:
    """Check if z is at THE LENS (z_c)."""
    return abs(z - Z_CRITICAL) < tolerance


def check_k_formation(kappa: float, eta: float, R: int) -> bool:
    """
    Check if K-formation criteria are met.

    K-formation requires:
    - kappa >= 0.92 (KAPPA_MIN)
    - eta > 0.618 (PHI_INV)
    - R >= 7 (minimum complexity)
    """
    return kappa >= KAPPA_MIN and eta > ETA_MIN and R >= R_MIN


def validate_physics(kappa: float, lambda_: float) -> bool:
    """Validate kappa + lambda = 1 conservation."""
    return abs(kappa + lambda_ - 1.0) < TOLERANCE_GOLDEN
