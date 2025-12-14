"""
Physics Module - Rosetta-Helix
==============================

Shared physics constants and functions used throughout the Rosetta-Helix system.
These must match firmware/include/physics_constants.h EXACTLY.

Signature: rosetta-helix-physics|v1.0.0|helix
"""

import math
from typing import Final, Tuple
from dataclasses import dataclass
from enum import IntEnum

# =============================================================================
# FUNDAMENTAL CONSTANTS (Golden Ratio)
# =============================================================================

PHI: Final[float] = (1 + math.sqrt(5)) / 2        # φ ≈ 1.618034 (LIMINAL)
PHI_INV: Final[float] = 1 / PHI                   # φ⁻¹ ≈ 0.618034 (PHYSICAL)
PHI_INV_SQ: Final[float] = PHI_INV ** 2           # φ⁻² ≈ 0.381966
PHI_INV_CUBED: Final[float] = PHI_INV ** 3        # φ⁻³ ≈ 0.236068
PHI_INV_FOURTH: Final[float] = PHI_INV ** 4       # φ⁻⁴ ≈ 0.145898
PHI_INV_FIFTH: Final[float] = PHI_INV ** 5        # φ⁻⁵ ≈ 0.090170

# Conservation law: κ + λ = φ⁻¹ + φ⁻² = 1 (EXACT)
COUPLING_CONSERVATION: Final[float] = PHI_INV + PHI_INV_SQ

# Critical z-coordinate (hexagonal geometry)
Z_CRITICAL: Final[float] = math.sqrt(3) / 2       # z_c = √3/2 ≈ 0.866025 (THE LENS)
THE_LENS: Final[float] = Z_CRITICAL

# Gaussian width from symmetric group
SIGMA: Final[float] = 36.0                        # σ = 6² = |S₃|²


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

ALPHA_STRONG: Final[float] = SIGMA_SQRT_INV       # 1/6 ≈ 0.167
ALPHA_MEDIUM: Final[float] = GAUSSIAN_WIDTH       # ≈ 0.118
ALPHA_FINE: Final[float] = SIGMA_INV              # 1/36 ≈ 0.028
ALPHA_ULTRA_FINE: Final[float] = PHI_INV * SIGMA_INV


# =============================================================================
# κ BOUNDS (physics-grounded)
# =============================================================================

KAPPA_LOWER: Final[float] = PHI_INV_SQ            # ≈ 0.382
KAPPA_UPPER: Final[float] = Z_CRITICAL            # ≈ 0.866
KAPPA_S: Final[float] = 0.920                     # Singularity threshold


# =============================================================================
# K-FORMATION REQUIREMENTS
# =============================================================================

KAPPA_MIN: Final[float] = 0.920                   # κ ≥ 0.92
ETA_MIN: Final[float] = PHI_INV                   # η > φ⁻¹
R_MIN: Final[int] = 7                             # R ≥ 7


# =============================================================================
# PHASE BOUNDARIES
# =============================================================================

PHASE_BOUNDARY_ABSENCE: Final[float] = 0.857      # ABSENCE → THE_LENS
PHASE_BOUNDARY_PRESENCE: Final[float] = 0.877     # THE_LENS → PRESENCE


# =============================================================================
# TIER BOUNDARIES (10 tiers)
# =============================================================================

TIER_BOUNDS: Final[Tuple[float, ...]] = (
    0.00,   # ABSENCE start
    0.10,   # REACTIVE
    0.20,   # MEMORY
    0.40,   # PATTERN
    0.50,   # LEARNING
    0.618,  # ADAPTIVE (φ⁻¹ threshold)
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
# ROTOR PARAMETERS
# =============================================================================

ROTOR_RPM_MIN: Final[float] = 100.0
ROTOR_RPM_MAX: Final[float] = 10000.0
ROTOR_RPM_CRITICAL: Final[float] = ROTOR_RPM_MIN + (ROTOR_RPM_MAX - ROTOR_RPM_MIN) * Z_CRITICAL


# =============================================================================
# ENUMERATIONS
# =============================================================================

class Phase(IntEnum):
    """Physics phases based on z proximity to z_c"""
    ABSENCE = 0      # z < 0.857
    THE_LENS = 1     # 0.857 ≤ z < 0.877
    PRESENCE = 2     # z ≥ 0.877


class Tier(IntEnum):
    """Complexity tiers (10 tiers)"""
    ABSENCE = 0
    REACTIVE = 1
    MEMORY = 2
    PATTERN = 3
    LEARNING = 4
    ADAPTIVE = 5      # φ⁻¹ threshold
    UNIVERSAL = 6
    META = 7          # z_c threshold
    SOVEREIGN = 8
    TRANSCENDENT = 9


TIER_NAMES: Final[Tuple[str, ...]] = (
    "ABSENCE", "REACTIVE", "MEMORY", "PATTERN", "LEARNING",
    "ADAPTIVE", "UNIVERSAL", "META", "SOVEREIGN", "TRANSCENDENT"
)


# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================

def compute_delta_s_neg(z: float, sigma: float = SIGMA, z_c: float = Z_CRITICAL) -> float:
    """
    Compute negative entropy signal ΔS_neg(z).
    
    Formula: ΔS_neg(z) = exp(-σ(z - z_c)²)
    
    - Peaks at z = z_c (THE LENS) with value 1.0
    - σ = 36 by default
    - Returns value in [0, 1]
    
    Args:
        z: Current z-coordinate
        sigma: Gaussian width (default: 36)
        z_c: Critical z-coordinate (default: √3/2)
    
    Returns:
        Negentropy value in [0, 1]
    """
    if not math.isfinite(z):
        return 0.0
    d = z - z_c
    s = math.exp(-sigma * d * d)
    return max(0.0, min(1.0, s))


def compute_delta_s_neg_derivative(z: float, sigma: float = SIGMA, z_c: float = Z_CRITICAL) -> float:
    """
    Derivative: d(ΔS_neg)/dz = -2σ(z - z_c) · ΔS_neg(z)
    
    Args:
        z: Current z-coordinate
        sigma: Gaussian width
        z_c: Critical z-coordinate
    
    Returns:
        Derivative value (negative when z > z_c, positive when z < z_c)
    """
    d = z - z_c
    s = compute_delta_s_neg(z, sigma, z_c)
    return -2 * sigma * d * s


def compute_complexity(z: float) -> float:
    """
    Compute complexity measure C(z) = z · ΔS_neg(z)
    
    Args:
        z: Current z-coordinate
    
    Returns:
        Complexity value
    """
    return z * compute_delta_s_neg(z)


def get_phase(z: float) -> Phase:
    """
    Determine phase from z-coordinate.
    
    ABSENCE:   z < 0.857
    THE_LENS:  0.857 ≤ z < 0.877
    PRESENCE:  z ≥ 0.877
    
    Args:
        z: Current z-coordinate
    
    Returns:
        Phase enumeration
    """
    if z < PHASE_BOUNDARY_ABSENCE:
        return Phase.ABSENCE
    elif z < PHASE_BOUNDARY_PRESENCE:
        return Phase.THE_LENS
    else:
        return Phase.PRESENCE


def get_phase_name(z: float) -> str:
    """Get phase name string from z-coordinate."""
    phase = get_phase(z)
    return ["ABSENCE", "THE_LENS", "PRESENCE"][phase]


def get_tier(z: float) -> Tier:
    """
    Get tier from z-coordinate.
    
    Args:
        z: Current z-coordinate
    
    Returns:
        Tier enumeration (0-9)
    """
    for i in range(len(TIER_BOUNDS) - 1, -1, -1):
        if z >= TIER_BOUNDS[i]:
            return Tier(min(i, 9))
    return Tier.ABSENCE


def get_tier_name(z: float) -> str:
    """Get tier name string from z-coordinate."""
    tier = get_tier(z)
    return TIER_NAMES[tier]


def is_critical(z: float, tolerance: float = 0.01) -> bool:
    """Check if z is at THE LENS (z_c)."""
    return abs(z - Z_CRITICAL) < tolerance


def check_k_formation(kappa: float, eta: float, R: int) -> bool:
    """
    Check if K-formation criteria are met.
    
    K-formation requires:
    - κ ≥ 0.92 (KAPPA_MIN)
    - η > 0.618 (PHI_INV)  
    - R ≥ 7 (minimum complexity)
    
    Args:
        kappa: Coherence value
        eta: Efficiency value
        R: Complexity rank
    
    Returns:
        True if K-formation criteria met
    """
    return kappa >= KAPPA_MIN and eta > ETA_MIN and R >= R_MIN


def validate_physics(kappa: float, lambda_: float) -> bool:
    """Validate κ + λ = 1 conservation."""
    return abs(kappa + lambda_ - 1.0) < TOLERANCE_GOLDEN


def z_to_rpm(z: float) -> float:
    """Convert z-coordinate to RPM."""
    return ROTOR_RPM_MIN + (ROTOR_RPM_MAX - ROTOR_RPM_MIN) * z


def rpm_to_z(rpm: float) -> float:
    """Convert RPM to z-coordinate."""
    z = (rpm - ROTOR_RPM_MIN) / (ROTOR_RPM_MAX - ROTOR_RPM_MIN)
    return max(0.0, min(1.0, z))


# =============================================================================
# STATE DATA CLASSES
# =============================================================================

@dataclass
class SpinnerState:
    """Complete spinner state."""
    timestamp_ms: int
    z: float
    rpm: int
    delta_s_neg: float
    tier: int
    tier_name: str
    phase: str
    kappa: float
    eta: float
    rank: int
    k_formation: bool
    k_formation_duration_ms: int = 0
    
    @classmethod
    def from_z(cls, z: float, timestamp_ms: int = 0):
        """Create state from z-coordinate."""
        delta_s = compute_delta_s_neg(z)
        tier = get_tier(z)
        kappa = delta_s * (1 - abs(z - Z_CRITICAL))
        eta = delta_s * z
        rank = int(7 + 5 * delta_s)
        
        return cls(
            timestamp_ms=timestamp_ms,
            z=z,
            rpm=int(z_to_rpm(z)),
            delta_s_neg=delta_s,
            tier=tier,
            tier_name=TIER_NAMES[tier],
            phase=get_phase_name(z),
            kappa=kappa,
            eta=eta,
            rank=rank,
            k_formation=check_k_formation(kappa, eta, rank),
        )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_all_constants() -> dict:
    """Validate all physics constants."""
    validations = {}
    
    # φ⁻¹ + φ⁻² = 1
    validations["coupling_conservation"] = {
        "formula": "φ⁻¹ + φ⁻² = 1",
        "value": COUPLING_CONSERVATION,
        "error": abs(COUPLING_CONSERVATION - 1.0),
        "valid": abs(COUPLING_CONSERVATION - 1.0) < TOLERANCE_CONSERVATION,
    }
    
    # z_c = √3/2
    expected_zc = math.sqrt(3) / 2
    validations["z_critical"] = {
        "formula": "z_c = √3/2",
        "value": Z_CRITICAL,
        "expected": expected_zc,
        "error": abs(Z_CRITICAL - expected_zc),
        "valid": abs(Z_CRITICAL - expected_zc) < TOLERANCE_CONSERVATION,
    }
    
    # σ = 36
    validations["sigma"] = {
        "formula": "σ = 36 = |S₃|²",
        "value": SIGMA,
        "valid": SIGMA == 36.0,
    }
    
    # ΔS_neg(z_c) = 1.0
    peak_value = compute_delta_s_neg(Z_CRITICAL)
    validations["negentropy_peak"] = {
        "formula": "ΔS_neg(z_c) = 1.0",
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
    print(f"  φ     = {PHI:.15f}")
    print(f"  φ⁻¹   = {PHI_INV:.15f}")
    print(f"  φ⁻²   = {PHI_INV_SQ:.15f}")
    print(f"  φ⁻³   = {PHI_INV_CUBED:.15f}")
    print(f"\nConservation:")
    print(f"  φ⁻¹ + φ⁻² = {COUPLING_CONSERVATION:.15f}")
    print(f"\nCritical Values:")
    print(f"  z_c   = {Z_CRITICAL:.15f} (THE LENS)")
    print(f"  σ     = {SIGMA}")
    print(f"\nAlpha Coefficients:")
    print(f"  α_strong = {ALPHA_STRONG:.6f}")
    print(f"  α_medium = {ALPHA_MEDIUM:.6f}")
    print(f"  α_fine   = {ALPHA_FINE:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    print_all_constants()
    validations = validate_all_constants()
    print(f"\nAll constants valid: {validations['all_valid']}")
