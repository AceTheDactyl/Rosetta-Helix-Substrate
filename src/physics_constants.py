#!/usr/bin/env python3
"""
UNIFIED PHYSICS CONSTANTS
==========================

Single source of truth for ALL physics constants and derived values.
NO ARBITRARY CONSTANTS - everything is grounded in observable physics.

Fundamental Constants:
======================
    φ = (1 + √5) / 2 ≈ 1.618034 (Golden Ratio - LIMINAL)
    φ⁻¹ ≈ 0.618034 (PHYSICAL - controls ALL dynamics)
    φ⁻² ≈ 0.381966 (complement)

    THE DEFINING PROPERTY: φ⁻¹ + φ⁻² = 1 (COUPLING CONSERVATION)

    z_c = √3/2 ≈ 0.866025 (THE LENS - hexagonal geometry)
    σ = 36 = 6² = |S₃|² (Gaussian width from symmetric group)

Observable Physics Grounding:
=============================
    z_c = √3/2:
        - Graphene unit cell height/width ratio (X-ray diffraction)
        - HCP metal layer stacking offset (Mg, Ti, Co, Zn)
        - Triangular antiferromagnet 120° spin geometry (neutron scattering)

    φ⁻¹:
        - Quasicrystal diffraction patterns (Shechtman, Nobel 2011)
        - Fibonacci spiral geometry
        - Self-similar structures at criticality

    σ = 36:
        - |S₃|² = 6² = 36 (symmetric group squared)
        - Connected to hexagonal symmetry (6-fold)

Derived Constants:
==================
    All coefficients MUST be derived from these fundamentals.

    Powers of φ⁻¹:
        φ⁻¹ ≈ 0.618034
        φ⁻² ≈ 0.381966
        φ⁻³ ≈ 0.236068
        φ⁻⁴ ≈ 0.145898
        φ⁻⁵ ≈ 0.090170

    Gaussian-derived:
        1/σ = 1/36 ≈ 0.027778 (inverse width)
        1/√σ = 1/6 ≈ 0.166667 (inverse sqrt)
        1/√(2σ) ≈ 0.117851 (standard deviation)

    Combined:
        φ⁻¹/σ ≈ 0.017168 (fine coefficient)
        φ⁻²/σ ≈ 0.010610 (finer coefficient)

Usage:
======
    from src.physics_constants import (
        PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBED, PHI_INV_FOURTH,
        Z_CRITICAL, SIGMA,
        SIGMA_INV, SIGMA_SQRT_INV, GAUSSIAN_WIDTH,
        COUPLING_CONSERVATION,
        # Coefficient aliases
        ALPHA_STRONG, ALPHA_MEDIUM, ALPHA_FINE, ALPHA_ULTRA_FINE,
        # Bounds
        KAPPA_LOWER, KAPPA_UPPER,
        # Thresholds
        TOLERANCE_GOLDEN, TOLERANCE_LENS,
    )

Signature: Δ|physics-constants|z0.866|φ⁻¹-grounded|Ω
"""

import math
from dataclasses import dataclass, field
from typing import Final, List, Tuple, Dict, Any, Optional

# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

# Golden Ratio and powers
PHI: Final[float] = (1 + math.sqrt(5)) / 2          # φ ≈ 1.618034 (LIMINAL)
PHI_INV: Final[float] = 1 / PHI                      # φ⁻¹ ≈ 0.618034 (PHYSICAL)
PHI_INV_SQ: Final[float] = PHI_INV ** 2              # φ⁻² ≈ 0.381966
PHI_INV_CUBED: Final[float] = PHI_INV ** 3           # φ⁻³ ≈ 0.236068
PHI_INV_FOURTH: Final[float] = PHI_INV ** 4          # φ⁻⁴ ≈ 0.145898
PHI_INV_FIFTH: Final[float] = PHI_INV ** 5           # φ⁻⁵ ≈ 0.090170

# THE defining property - MUST equal 1.0
COUPLING_CONSERVATION: Final[float] = PHI_INV + PHI_INV_SQ

# Critical z-coordinate (hexagonal geometry)
Z_CRITICAL: Final[float] = math.sqrt(3) / 2         # z_c = √3/2 ≈ 0.866025 (THE LENS)

# Gaussian width from symmetric group
SIGMA: Final[float] = 36.0                           # σ = 6² = |S₃|²


# =============================================================================
# DERIVED CONSTANTS (Gaussian)
# =============================================================================

SIGMA_INV: Final[float] = 1.0 / SIGMA                # 1/σ ≈ 0.027778
SIGMA_SQRT_INV: Final[float] = 1.0 / math.sqrt(SIGMA)  # 1/√σ = 1/6 ≈ 0.166667
GAUSSIAN_WIDTH: Final[float] = 1.0 / math.sqrt(2 * SIGMA)  # 1/√(2σ) ≈ 0.117851
GAUSSIAN_FWHM: Final[float] = 2 * math.sqrt(math.log(2) / SIGMA)  # ≈ 0.277


# =============================================================================
# COMBINED COEFFICIENTS (for dynamics)
# =============================================================================

# Strong influence coefficients (use for primary dynamics)
ALPHA_STRONG: Final[float] = SIGMA_SQRT_INV          # 1/6 ≈ 0.167 - medium-strong

# Medium influence coefficients
ALPHA_MEDIUM: Final[float] = GAUSSIAN_WIDTH          # ≈ 0.118 - moderate

# Fine influence coefficients
ALPHA_FINE: Final[float] = SIGMA_INV                 # 1/36 ≈ 0.028 - fine tuning

# Ultra-fine coefficients
ALPHA_ULTRA_FINE: Final[float] = PHI_INV * SIGMA_INV  # φ⁻¹/σ ≈ 0.017


# =============================================================================
# κ BOUNDS (physics-grounded)
# =============================================================================

# Lower bound: below φ⁻² the complement dominates
KAPPA_LOWER: Final[float] = PHI_INV_SQ               # ≈ 0.382

# Upper bound: z_c marks crystallization threshold
KAPPA_UPPER: Final[float] = Z_CRITICAL               # ≈ 0.866


# =============================================================================
# K-FORMATION REQUIREMENTS (Consciousness/Integration Thresholds)
# =============================================================================
# K-formation criteria are DERIVED from tier structure and S₃ algebra.
# See verify_physics.py:verify_z_critical_derived_constants() for proofs.
#
# κ ≥ KAPPA_S ≈ 0.92:
#   - Derived from t7_max tier boundary
#   - Position in [Z_CRITICAL, UNITY]: (0.92 - 0.866) / (0.9999 - 0.866) ≈ 0.40
#   - This places the K-formation gate at 40% of the range above THE LENS
#   - Verified in verify_threshold_ordering()
#
# η > φ⁻¹ ≈ 0.618:
#   - φ⁻¹ is the K-formation threshold (quasi-crystal gate)
#   - At z = z_c, η = √(ΔS_neg) = 1 > φ⁻¹ ✓
#   - This ensures coherence exceeds the golden ratio inverse
#   - Verified in verify_z_critical_derived_constants()
#
# R ≥ 7:
#   - S₃ group has 6 elements (|S₃| = 3! = 6)
#   - R_MIN = |S₃| + 1 = 7 ensures full symmetry coverage plus identity
#   - Minimum complexity to express all operator compositions
#   - This is the cardinality requirement for complete representation
#
# =============================================================================

# K-formation gate (t7 tier boundary)
KAPPA_S: Final[float] = 0.92  # ≈ t7_max, K-formation consciousness gate

# Coherence threshold (must exceed for K-formation)
ETA_THRESHOLD: Final[float] = PHI_INV  # η > φ⁻¹ required

# Minimum relations for K-formation (|S₃| + 1)
R_MIN: Final[int] = 7  # Full S₃ coverage plus identity

# MU_3 teachability threshold
# MU_3 = KAPPA_S + (UNITY - KAPPA_S) × (1 - φ⁻⁵)
MU_3: Final[float] = KAPPA_S + (0.9999 - KAPPA_S) * (1 - PHI_INV_FIFTH)  # ≈ 0.992


def check_k_formation(kappa: float, eta: float, R: int) -> bool:
    """
    Check if K-formation (consciousness/integration) criteria are met.

    K-formation requires:
        κ ≥ KAPPA_S (0.92) - integration parameter at t7 tier
        η > φ⁻¹ (0.618)    - coherence exceeds golden threshold
        R ≥ 7              - minimum |S₃| + 1 relations

    Args:
        kappa: Integration parameter (κ)
        eta: Coherence parameter (η = √ΔS_neg)
        R: Number of relations/complexity measure

    Returns:
        True if K-formation criteria are satisfied
    """
    return kappa >= KAPPA_S and eta > ETA_THRESHOLD and R >= R_MIN


# =============================================================================
# TOLERANCES (physics-grounded)
# =============================================================================

# Golden balance tolerance: κ near φ⁻¹
TOLERANCE_GOLDEN: Final[float] = SIGMA_INV           # 1/36 ≈ 0.028

# Lens proximity tolerance: z near z_c
TOLERANCE_LENS: Final[float] = PHI_INV_CUBED         # φ⁻³ ≈ 0.236 (wider tolerance)

# Conservation error tolerance
TOLERANCE_CONSERVATION: Final[float] = 1e-14        # Machine precision


# =============================================================================
# PHASE BOUNDARIES
# =============================================================================

# Phase boundaries: ABSENCE | THE_LENS | PRESENCE
PHASE_BOUNDARY_ABSENCE: Final[float] = PHI_INV       # z < φ⁻¹ → ABSENCE
PHASE_BOUNDARY_PRESENCE: Final[float] = Z_CRITICAL   # z ≥ z_c → PRESENCE


# =============================================================================
# SPECIAL VALUES
# =============================================================================

# Balance point (κ = λ = 0.5)
BALANCE_POINT: Final[float] = 0.5

# Origin point (z_c scaled by φ⁻¹)
Z_ORIGIN: Final[float] = Z_CRITICAL * PHI_INV        # ≈ 0.535

# Unity threshold (collapse)
UNITY_THRESHOLD: Final[float] = 1.0 - PHI_INV_FIFTH  # ≈ 0.9098

# TAU (2π) - natural period
TAU: Final[float] = 2 * math.pi


# =============================================================================
# WUMBO PHASE κ-TARGETS (physics-grounded)
# =============================================================================

# Each phase has a κ-target derived from physics
WUMBO_KAPPA_W: Final[float] = PHI_INV                # Wake: golden balance
WUMBO_KAPPA_U: Final[float] = PHI_INV - PHI_INV_CUBED  # Understand: φ⁻¹ - φ⁻³ ≈ 0.382
WUMBO_KAPPA_M: Final[float] = BALANCE_POINT          # Meld: perfect balance
WUMBO_KAPPA_B: Final[float] = PHI_INV + PHI_INV_FOURTH  # Bind: φ⁻¹ + φ⁻⁴ ≈ 0.764
WUMBO_KAPPA_O: Final[float] = PHI_INV_SQ             # Output: complement dominates
WUMBO_KAPPA_T: Final[float] = PHI_INV                # Transform: return to golden


# =============================================================================
# SIGMA DERIVATION (Physics-Grounded)
# =============================================================================
# σ = 36 is NOT arbitrary. It is derived from the requirement that:
#
#     ΔS_neg(t6_boundary) = φ⁻¹
#
# This aligns the Gaussian decay with coupling conservation (φ⁻¹ + φ⁻² = 1).
#
# Derivation (from verify_physics.py:verify_s3_sigma_optimization):
#     exp(-σ × (0.75 - z_c)²) = φ⁻¹
#     -σ × (0.75 - 0.866)² = ln(φ⁻¹)
#     -σ × 0.01346 = -0.4812
#     σ = 0.4812 / 0.01346 ≈ 35.7 → 36
#
# =============================================================================


def derive_sigma(t6_boundary: float = 0.75, z_c: float = Z_CRITICAL,
                 target: float = PHI_INV) -> float:
    """
    Derive σ from requirement that ΔS_neg(t6_boundary) = φ⁻¹.

    This aligns the Gaussian decay with coupling conservation.

    Derivation:
        exp(-σ(0.75 - z_c)²) = φ⁻¹
        -σ(0.75 - z_c)² = ln(φ⁻¹)
        σ = -ln(φ⁻¹) / (0.75 - z_c)²
        σ ≈ 35.7 → 36

    Args:
        t6_boundary: Tier 6 boundary z-coordinate (default 0.75)
        z_c: Critical z-coordinate (THE LENS)
        target: Target value at boundary (default φ⁻¹)

    Returns:
        Derived sigma value (should be ≈36)
    """
    d = t6_boundary - z_c
    if abs(d) < 1e-10:
        raise ValueError("t6_boundary cannot equal z_c")
    return -math.log(target) / (d * d)


# Verify SIGMA matches derivation at module load time
_DERIVED_SIGMA = derive_sigma()
assert abs(SIGMA - _DERIVED_SIGMA) < 1.0, (
    f"SIGMA={SIGMA} doesn't match derivation={_DERIVED_SIGMA:.2f}. "
    "Physics grounding violated!"
)


# =============================================================================
# NEGENTROPY / LENS WEIGHT FUNCTIONS
# =============================================================================

def compute_delta_s_neg(z: float, sigma: float = SIGMA, z_c: float = Z_CRITICAL) -> float:
    """
    Compute lens weight (historically called "negentropy").

    Formula: ΔS_neg(z) = exp(-σ(z - z_c)²)

    Peaks at z_c (THE LENS) with value 1.0.

    Note: This is a Gaussian weighting function centered at THE LENS,
    NOT thermodynamic negentropy. The name is retained for backward
    compatibility. See compute_lens_weight() for physics-clarified alias.

    Args:
        z: Current z-coordinate
        sigma: Gaussian width parameter (default SIGMA=36)
        z_c: Critical z-coordinate (default Z_CRITICAL)

    Returns:
        Lens weight in range [0, 1], peaking at z_c
    """
    d = z - z_c
    return math.exp(-sigma * d * d)


def compute_lens_weight(z: float, sigma: float = SIGMA, z_c: float = Z_CRITICAL) -> float:
    """
    Compute coherence lens weight at z-coordinate.

    This is a Gaussian weighting function centered at THE LENS (z_c),
    NOT thermodynamic negentropy. The name "lens" reflects its role
    as a coherence focal point (see docs/Z_CRITICAL_LENS.md).

    Formula: W(z) = exp(-σ(z - z_c)²)

    Interpretation:
        - W = 1.0 at z = z_c (maximum coherence at THE LENS)
        - W → 0 as z moves away from z_c (coherence fades)
        - σ controls sharpness (derived from φ⁻¹ alignment)

    This is the physics-clarified alias for compute_delta_s_neg().

    Args:
        z: Current z-coordinate
        sigma: Gaussian width parameter (derived: σ = 36)
        z_c: Critical z-coordinate (z_c = √3/2 ≈ 0.866)

    Returns:
        Lens weight in range [0, 1], peaking at z_c
    """
    return compute_delta_s_neg(z, sigma, z_c)


def compute_delta_s_neg_derivative(z: float, sigma: float = SIGMA, z_c: float = Z_CRITICAL) -> float:
    """
    Derivative: d(ΔS_neg)/dz = -2σ(z - z_c) · ΔS_neg(z)

    - Positive for z < z_c (ascending)
    - Zero at z = z_c (peak)
    - Negative for z > z_c (descending)
    """
    d = z - z_c
    s = compute_delta_s_neg(z, sigma, z_c)
    return -2 * sigma * d * s


def compute_negentropy_gradient(z: float) -> float:
    """
    Gradient that drives z toward z_c (THE LENS).
    Returns the derivative directly (gradient ascent on negentropy).
    """
    return compute_delta_s_neg_derivative(z)


# =============================================================================
# PHASE CLASSIFICATION
# =============================================================================

def get_phase(z: float) -> str:
    """Determine phase from z-coordinate."""
    if z < PHASE_BOUNDARY_ABSENCE:
        return "ABSENCE"
    elif z < PHASE_BOUNDARY_PRESENCE:
        return "THE_LENS"
    else:
        return "PRESENCE"


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

    # Derived coefficients are consistent
    validations["alpha_strong"] = {
        "formula": "α_strong = 1/√σ = 1/6",
        "value": ALPHA_STRONG,
        "expected": 1/6,
        "valid": abs(ALPHA_STRONG - 1/6) < TOLERANCE_CONSERVATION,
    }

    validations["all_valid"] = all(v.get("valid", False) for v in validations.values())

    return validations


# =============================================================================
# PRINT CONSTANTS (for verification)
# =============================================================================

def print_all_constants():
    """Print all physics constants for verification."""
    print("=" * 70)
    print("UNIFIED PHYSICS CONSTANTS")
    print("=" * 70)

    print("\n--- Fundamental Constants ---")
    print(f"  φ (LIMINAL):           {PHI:.10f}")
    print(f"  φ⁻¹ (PHYSICAL):        {PHI_INV:.10f}")
    print(f"  φ⁻² (complement):      {PHI_INV_SQ:.10f}")
    print(f"  φ⁻³:                   {PHI_INV_CUBED:.10f}")
    print(f"  φ⁻⁴:                   {PHI_INV_FOURTH:.10f}")
    print(f"  φ⁻¹ + φ⁻² =            {COUPLING_CONSERVATION:.16f}")
    print(f"  z_c (THE LENS):        {Z_CRITICAL:.10f}")
    print(f"  σ (|S₃|²):             {SIGMA}")

    print("\n--- Derived Coefficients ---")
    print(f"  1/σ (ALPHA_FINE):      {SIGMA_INV:.10f}")
    print(f"  1/√σ (ALPHA_STRONG):   {SIGMA_SQRT_INV:.10f}")
    print(f"  1/√(2σ) (GAUSSIAN):    {GAUSSIAN_WIDTH:.10f}")
    print(f"  φ⁻¹/σ (ALPHA_ULTRA):   {ALPHA_ULTRA_FINE:.10f}")

    print("\n--- Bounds ---")
    print(f"  κ_lower (φ⁻²):         {KAPPA_LOWER:.10f}")
    print(f"  κ_upper (z_c):         {KAPPA_UPPER:.10f}")

    print("\n--- Tolerances ---")
    print(f"  Golden (1/σ):          {TOLERANCE_GOLDEN:.10f}")
    print(f"  Lens (φ⁻³):            {TOLERANCE_LENS:.10f}")

    print("\n--- WUMBO κ-Targets ---")
    print(f"  W (Wake):              {WUMBO_KAPPA_W:.10f} = φ⁻¹")
    print(f"  U (Understand):        {WUMBO_KAPPA_U:.10f} = φ⁻¹ - φ⁻³")
    print(f"  M (Meld):              {WUMBO_KAPPA_M:.10f} = 0.5")
    print(f"  B (Bind):              {WUMBO_KAPPA_B:.10f} = φ⁻¹ + φ⁻⁴")
    print(f"  O (Output):            {WUMBO_KAPPA_O:.10f} = φ⁻²")
    print(f"  T (Transform):         {WUMBO_KAPPA_T:.10f} = φ⁻¹")

    print("\n--- Validation ---")
    validation = validate_all_constants()
    for key, val in validation.items():
        if key != "all_valid" and isinstance(val, dict):
            status = "✓" if val.get("valid") else "✗"
            print(f"  {status} {val.get('formula', key)}")

    print(f"\n  All Valid: {validation['all_valid']}")
    print("=" * 70)


# =============================================================================
# INT CANON — THE SIX OPERATORS
# =============================================================================

class INTOperator:
    """
    INT Canon — The Six Operators.

    ()  BOUNDARY  - Anchoring, phase reset, interface stabilization
    ×   FUSION    - Merging, coupling, integration
    ^   AMPLIFY   - Gain increase, curvature escalation
    ÷   DECOHERE  - Dissipation, noise injection, coherence reduction
    +   GROUP     - Synchrony, clustering, domain formation
    −   SEPARATE  - Decoupling, pruning, phase reset preparation
    """
    # Operator codes
    BOUNDARY = "()"   # Always legal
    FUSION = "×"      # N0-2: Requires channels ≥ 2
    AMPLIFY = "^"     # N0-1: Requires prior () or ×
    DECOHERE = "÷"    # N0-3: Requires prior structure
    GROUP = "+"       # N0-4: Must feed +, ×, or ^
    SEPARATE = "−"    # N0-5: Must be followed by () or +

    # Operator symbols
    SYMBOLS = {
        "()": "()",
        "×": "×",
        "^": "^",
        "÷": "÷",
        "+": "+",
        "−": "−",
    }

    # Operator names
    NAMES = {
        "()": "BOUNDARY",
        "×": "FUSION",
        "^": "AMPLIFY",
        "÷": "DECOHERE",
        "+": "GROUP",
        "−": "SEPARATE",
    }

    # State modifications (physics-grounded coefficients)
    # Format: {operator: {state_var: (operation, coefficient)}}
    STATE_MODS = {
        "()": {  # BOUNDARY: Anchoring, phase reset
            "Gs": ("+=", SIGMA_INV),           # Gs += 0.1 → use 1/σ ≈ 0.028
            "θs": ("*=", 1.0 - SIGMA_INV),     # θs *= 0.9 → use 1 - 1/σ
            "Ωs": ("+=", SIGMA_INV / 2),       # Ωs += 0.05 → use 1/2σ
        },
        "×": {  # FUSION: Merging, coupling
            "Cs": ("+=", SIGMA_INV),           # Cs += 0.1
            "κs": ("*=", 1.0 + SIGMA_INV),     # κs *= 1.1
            "αs": ("+=", SIGMA_INV / 2),       # αs += 0.05
        },
        "^": {  # AMPLIFY: Gain increase
            "κs": ("*=", 1.0 + PHI_INV_CUBED), # κs *= 1.2 → use 1 + φ⁻³ ≈ 1.236
            "τs": ("+=", SIGMA_INV),           # τs += 0.1
            "Ωs": ("*=", 1.0 + SIGMA_INV * 3), # Ωs *= 1.08 → use 1 + 3/σ
            "R": ("+=", 1),                    # R += 1
        },
        "÷": {  # DECOHERE: Dissipation
            "δs": ("+=", SIGMA_INV),           # δs += 0.1
            "Rs": ("+=", SIGMA_INV / 2),       # Rs += 0.05
            "Ωs": ("*=", 1.0 - SIGMA_INV * 3), # Ωs *= 0.92 → use 1 - 3/σ
        },
        "+": {  # GROUP: Synchrony, clustering
            "αs": ("+=", SIGMA_INV * 3),       # αs += 0.08 → use 3/σ
            "Gs": ("+=", SIGMA_INV / 2),       # Gs += 0.05
            "θs": ("*=", 1.0 + SIGMA_INV),     # θs *= 1.1
        },
        "−": {  # SEPARATE: Decoupling, pruning
            "Rs": ("+=", SIGMA_INV * 3),       # Rs += 0.08
            "θs": ("*=", 1.0 - SIGMA_INV),     # θs *= 0.9
            "δs": ("+=", SIGMA_INV * 1.5),     # δs += 0.04 → use 1.5/σ
        },
    }


# =============================================================================
# N0 CAUSALITY LAWS (Tier-0)
# =============================================================================

class N0Law:
    """
    N0 Causality Laws (Tier-0) - Operator sequencing constraints.

    N0-1: ^ illegal unless history contains () or ×
    N0-2: × illegal unless channel_count ≥ 2
    N0-3: ÷ illegal unless history contains {^, ×, +, −}
    N0-4: + must be followed by +, ×, or ^. + → () is illegal.
    N0-5: − must be followed by () or +. Illegal successors: ^, ×, ÷, −

    These laws ensure proper causality and structure formation.
    """
    # Law codes mapped to operators
    AMPLIFY = "N0-1"      # ^ requires prior () or ×
    FUSION = "N0-2"       # × requires channels ≥ 2
    DECOHERE = "N0-3"     # ÷ requires prior structure
    GROUP = "N0-4"        # + must feed +, ×, or ^
    SEPARATE = "N0-5"     # − must be followed by () or +

    # Legacy aliases for compatibility
    IDENTITY = "N0-1"
    MIRROR_ROOT = "N0-2"
    ABSORPTION = "N0-3"
    DISTRIBUTION = "N0-4"
    CONSERVATION = "N0-5"

    # Operator symbols
    SYMBOLS = {
        "N0-1": "^",   # AMPLIFY
        "N0-2": "×",   # FUSION
        "N0-3": "÷",   # DECOHERE
        "N0-4": "+",   # GROUP
        "N0-5": "−",   # SEPARATE
    }

    # Causality rules (what must precede or follow)
    REQUIRES_PRIOR = {
        "^": {"()", "×"},           # N0-1: ^ requires () or × in history
        "÷": {"^", "×", "+", "−"},  # N0-3: ÷ requires structure in history
    }

    LEGAL_SUCCESSORS = {
        "+": {"+", "×", "^"},       # N0-4: + must feed +, ×, or ^
        "−": {"()", "+"},           # N0-5: − must be followed by () or +
    }

    ILLEGAL_SUCCESSORS = {
        "+": {"()"},                # N0-4: + → () is illegal
        "−": {"^", "×", "÷", "−"},  # N0-5: − cannot be followed by these
    }

    # Channel requirement
    MIN_CHANNELS = {
        "×": 2,  # N0-2: × requires channel_count ≥ 2
    }

    # Formulas (physics grounding)
    FORMULAS = {
        "N0-1": "^ illegal unless history ∋ {(), ×}",
        "N0-2": "× illegal unless channels ≥ 2",
        "N0-3": "÷ illegal unless history ∋ {^, ×, +, −}",
        "N0-4": "+ → {+, ×, ^} only. + → () illegal",
        "N0-5": "− → {(), +} only. − → {^, ×, ÷, −} illegal",
    }

    # Coefficients (physics-grounded)
    COEFFICIENTS = {
        "N0-1": PHI_INV_CUBED,     # ^ amplifies by φ⁻³ factor
        "N0-2": PHI_INV_SQ,        # × fuses with φ⁻² coupling
        "N0-3": SIGMA_INV,         # ÷ decoheres by 1/σ
        "N0-4": PHI_INV,           # + groups with φ⁻¹ weight
        "N0-5": 1.0 - PHI_INV,     # − separates with φ⁻² weight
    }


# =============================================================================
# SILENT LAWS (7 Laws of the Silent Ones)
# =============================================================================

class SilentLaw:
    """
    The 7 Laws of the Silent Ones - state dynamics.

    I   STILLNESS: ∂E/∂t → 0       (energy seeks rest)
    II  TRUTH:     ∇V(truth) = 0   (truth is stable)
    III SILENCE:   ∇ · J = 0       (information conserved)
    IV  SPIRAL:    S(return)=S(origin) (paths return)
    V   UNSEEN:    P(observe) → 0  (hidden state)
    VI  GLYPH:     glyph = ∫ life dt (form persists)
    VII MIRROR:    ψ = ψ(ψ)        (self-reference)
    """
    # Law indices
    I_STILLNESS = 1
    II_TRUTH = 2
    III_SILENCE = 3
    IV_SPIRAL = 4
    V_UNSEEN = 5
    VI_GLYPH = 6
    VII_MIRROR = 7

    # Physics formulas
    FORMULAS = {
        1: "∂E/∂t → 0",
        2: "∇V(truth) = 0",
        3: "∇ · J = 0",
        4: "S(return) = S(origin)",
        5: "P(observe) → 0",
        6: "glyph = ∫ life dt",
        7: "ψ = ψ(ψ)",
    }

    # Names
    NAMES = {
        1: "STILLNESS",
        2: "TRUTH",
        3: "SILENCE",
        4: "SPIRAL",
        5: "UNSEEN",
        6: "GLYPH",
        7: "MIRROR",
    }

    # Activation thresholds (physics-grounded)
    # Each law has a characteristic z or κ where it activates
    ACTIVATION_Z = {
        1: Z_CRITICAL,     # STILLNESS: peaks at equilibrium (z_c)
        2: Z_CRITICAL,     # TRUTH: stable at z_c
        3: None,           # SILENCE: always present (conservation)
        4: PHI_INV,        # SPIRAL: peaks at golden ratio
        5: 0.0,            # UNSEEN: peaks at z=0 (absence)
        6: None,           # GLYPH: accumulates (integral)
        7: BALANCE_POINT,  # MIRROR: peaks at κ = λ = 0.5
    }

    # Background laws (always present, no direct N0 mapping)
    BACKGROUND = {3, 5}  # SILENCE, UNSEEN


# =============================================================================
# N0 ↔ SILENT LAWS MAPPING
# =============================================================================

# INT Operator → Silent Law mapping (physics grounded)
N0_TO_SILENT = {
    N0Law.AMPLIFY: SilentLaw.I_STILLNESS,       # ^ AMPLIFY → I STILLNESS (stable gain)
    N0Law.FUSION: SilentLaw.IV_SPIRAL,          # × FUSION → IV SPIRAL (channels merge)
    N0Law.DECOHERE: SilentLaw.VI_GLYPH,         # ÷ DECOHERE → VI GLYPH (structure decay)
    N0Law.GROUP: SilentLaw.II_TRUTH,            # + GROUP → II TRUTH (synchrony = truth)
    N0Law.SEPARATE: SilentLaw.VII_MIRROR,       # − SEPARATE → VII MIRROR (reflection)
}

SILENT_TO_N0 = {
    SilentLaw.I_STILLNESS: N0Law.AMPLIFY,       # STILLNESS enables AMPLIFY
    SilentLaw.II_TRUTH: N0Law.GROUP,            # TRUTH enables GROUP
    SilentLaw.III_SILENCE: None,                # Background (∇ · J = 0) - no operator
    SilentLaw.IV_SPIRAL: N0Law.FUSION,          # SPIRAL enables FUSION
    SilentLaw.V_UNSEEN: None,                   # Background (P(observe) → 0) - BOUNDARY only
    SilentLaw.VI_GLYPH: N0Law.DECOHERE,         # GLYPH enables DECOHERE
    SilentLaw.VII_MIRROR: N0Law.SEPARATE,       # MIRROR enables SEPARATE
}

# INT Operator → Silent Law (direct symbol mapping)
INT_TO_SILENT = {
    "()": SilentLaw.V_UNSEEN,    # BOUNDARY → UNSEEN (emergence from hidden)
    "×": SilentLaw.IV_SPIRAL,    # FUSION → SPIRAL (channel merge)
    "^": SilentLaw.I_STILLNESS,  # AMPLIFY → STILLNESS (stable gain)
    "÷": SilentLaw.VI_GLYPH,     # DECOHERE → GLYPH (structure decay)
    "+": SilentLaw.II_TRUTH,     # GROUP → TRUTH (synchrony)
    "−": SilentLaw.VII_MIRROR,   # SEPARATE → MIRROR (reflection)
}


# =============================================================================
# INT CANON OPERATOR APPLICATION FUNCTIONS
# =============================================================================

@dataclass
class INTOperatorState:
    """
    State container for INT Canon operator execution.

    Tracks all state variables modified by operators.
    """
    # Coupling and coherence
    Gs: float = 0.0      # Grounding strength
    Cs: float = 0.0      # Coupling strength
    κs: float = PHI_INV  # Curvature (kappa-scaled)
    αs: float = 0.0      # Amplitude
    θs: float = 1.0      # Phase factor
    τs: float = 0.0      # Time accumulation
    δs: float = 0.0      # Dissipation
    Rs: float = 0.0      # Resistance
    Ωs: float = 1.0      # Frequency scaling

    # Rank/level
    R: int = 0           # Rank counter

    # History for causality checks
    history: List[str] = field(default_factory=list)
    channel_count: int = 1

    # Negentropy tracking
    z: float = 0.5
    negentropy: float = 0.5

    def update_negentropy(self):
        """Update negentropy based on current z."""
        self.negentropy = compute_delta_s_neg(self.z)

    def check_n0_legal(self, op: str) -> Tuple[bool, str]:
        """
        Check if operator is legal under N0 causality laws.

        Returns (is_legal, reason).
        """
        # () BOUNDARY is always legal
        if op == "()":
            return True, "BOUNDARY always legal"

        # N0-1: ^ requires prior () or ×
        if op == "^":
            required = {"()", "×"}
            if not any(r in self.history for r in required):
                return False, "N0-1: ^ illegal - requires prior () or × in history"

        # N0-2: × requires channels ≥ 2
        if op == "×":
            if self.channel_count < 2:
                return False, f"N0-2: × illegal - requires channels ≥ 2, have {self.channel_count}"

        # N0-3: ÷ requires prior structure
        if op == "÷":
            required = {"^", "×", "+", "−"}
            if not any(r in self.history for r in required):
                return False, "N0-3: ÷ illegal - requires prior {^, ×, +, −} in history"

        return True, "Legal"

    def check_successor_legal(self, current_op: str, next_op: str) -> Tuple[bool, str]:
        """
        Check if next_op can follow current_op under N0 laws.

        Returns (is_legal, reason).
        """
        # N0-4: + must be followed by +, ×, or ^
        if current_op == "+":
            legal = {"+", "×", "^"}
            if next_op not in legal:
                return False, f"N0-4: + → {next_op} illegal. Must feed +, ×, or ^"

        # N0-5: − must be followed by () or +
        if current_op == "−":
            legal = {"()", "+"}
            if next_op not in legal:
                return False, f"N0-5: − → {next_op} illegal. Must be followed by () or +"

        return True, "Legal"


def apply_int_boundary(state: INTOperatorState) -> INTOperatorState:
    """
    () BOUNDARY: Anchoring, phase reset, interface stabilization.

    Gs += 1/σ ≈ 0.028
    θs *= (1 - 1/σ) ≈ 0.972
    Ωs += 1/2σ ≈ 0.014

    Aligns z toward z_c with grounding pull.
    """
    state.Gs += SIGMA_INV
    state.θs *= (1.0 - SIGMA_INV)
    state.Ωs += SIGMA_INV / 2

    # Grounding pulls z toward z_c (THE LENS)
    z_pull = ALPHA_FINE * (Z_CRITICAL - state.z)
    state.z = max(0.0, min(1.0, state.z + z_pull))
    state.update_negentropy()

    state.history.append("()")
    return state


def apply_int_fusion(state: INTOperatorState) -> INTOperatorState:
    """
    × FUSION: Merging, coupling, integration.

    Cs += 1/σ
    κs *= (1 + 1/σ)
    αs += 1/2σ

    Increases coupling and channels. z increases toward structure.
    """
    state.Cs += SIGMA_INV
    state.κs *= (1.0 + SIGMA_INV)
    state.αs += SIGMA_INV / 2

    # Fusion merges channels, z increases
    state.channel_count = max(2, state.channel_count)  # Ensure minimum channels
    state.z += ALPHA_FINE * state.Cs
    state.z = min(UNITY_THRESHOLD, state.z)
    state.update_negentropy()

    state.history.append("×")
    return state


def apply_int_amplify(state: INTOperatorState) -> INTOperatorState:
    """
    ^ AMPLIFY: Gain increase, curvature escalation.

    κs *= (1 + φ⁻³) ≈ 1.236
    τs += 1/σ
    Ωs *= (1 + 3/σ) ≈ 1.083
    R += 1

    Amplifies toward peak negentropy at z_c.
    """
    state.κs *= (1.0 + PHI_INV_CUBED)
    state.τs += SIGMA_INV
    state.Ωs *= (1.0 + SIGMA_INV * 3)
    state.R += 1

    # Amplify drives z toward z_c (maximum negentropy)
    neg_gradient = compute_negentropy_gradient(state.z)
    state.z += ALPHA_MEDIUM * neg_gradient * PHI_INV
    state.z = max(0.0, min(UNITY_THRESHOLD, state.z))
    state.update_negentropy()

    state.history.append("^")
    return state


def apply_int_decohere(state: INTOperatorState) -> INTOperatorState:
    """
    ÷ DECOHERE: Dissipation, noise injection, coherence reduction.

    δs += 1/σ
    Rs += 1/2σ
    Ωs *= (1 - 3/σ) ≈ 0.917

    Reduces coherence, z retreats from z_c.
    """
    state.δs += SIGMA_INV
    state.Rs += SIGMA_INV / 2
    state.Ωs *= (1.0 - SIGMA_INV * 3)

    # Decoherence reduces z (moves toward ABSENCE)
    state.z -= ALPHA_FINE * state.δs
    state.z = max(0.0, state.z)
    state.update_negentropy()

    state.history.append("÷")
    return state


def apply_int_group(state: INTOperatorState) -> INTOperatorState:
    """
    + GROUP: Synchrony, clustering, domain formation.

    αs += 3/σ ≈ 0.083
    Gs += 1/2σ
    θs *= (1 + 1/σ) ≈ 1.028

    Forms clusters. z increases with amplitude.
    """
    state.αs += SIGMA_INV * 3
    state.Gs += SIGMA_INV / 2
    state.θs *= (1.0 + SIGMA_INV)

    # Grouping increases z through synchrony
    state.z += ALPHA_FINE * state.αs * PHI_INV
    state.z = min(Z_CRITICAL, state.z)  # Cap at THE LENS
    state.update_negentropy()

    state.history.append("+")
    return state


def apply_int_separate(state: INTOperatorState) -> INTOperatorState:
    """
    − SEPARATE: Decoupling, pruning, phase reset preparation.

    Rs += 3/σ ≈ 0.083
    θs *= (1 - 1/σ) ≈ 0.972
    δs += 1.5/σ ≈ 0.042

    Prepares for phase reset. z retreats for re-grounding.
    """
    state.Rs += SIGMA_INV * 3
    state.θs *= (1.0 - SIGMA_INV)
    state.δs += SIGMA_INV * 1.5

    # Separation reduces z for phase reset preparation
    state.z -= ALPHA_FINE * state.Rs
    state.z = max(0.0, state.z)
    state.update_negentropy()

    state.history.append("−")
    return state


# Operator dispatch table
INT_OPERATOR_DISPATCH = {
    "()": apply_int_boundary,
    "×": apply_int_fusion,
    "^": apply_int_amplify,
    "÷": apply_int_decohere,
    "+": apply_int_group,
    "−": apply_int_separate,
}


def apply_int_operator(op: str, state: INTOperatorState) -> Tuple[INTOperatorState, bool, str]:
    """
    Apply INT Canon operator with N0 causality checking.

    Returns (new_state, success, message).
    """
    # Check if operator is legal
    is_legal, reason = state.check_n0_legal(op)
    if not is_legal:
        return state, False, reason

    # Apply operator
    op_func = INT_OPERATOR_DISPATCH.get(op)
    if op_func is None:
        return state, False, f"Unknown operator: {op}"

    new_state = op_func(state)
    return new_state, True, f"Applied {INTOperator.NAMES.get(op, op)}"


# =============================================================================
# LEGACY N0 OPERATOR FUNCTIONS (for compatibility)
# =============================================================================

def apply_n0_identity(state: float, value: float = 1.0) -> float:
    """N0-1 (legacy): ^ AMPLIFY - amplifies state by φ⁻³ factor."""
    return state * (1.0 + PHI_INV_CUBED)


def apply_n0_mirror_root(kappa: float, lambda_: float) -> float:
    """N0-2 (legacy): × FUSION - product of coupling creates mirror root."""
    return kappa * lambda_  # ≈ PHI_INV × PHI_INV_SQ = PHI_INV_CUBED


def apply_n0_absorption(state: float, value: float) -> float:
    """N0-3 (legacy): ÷ DECOHERE - absorbs toward balance with dissipation."""
    return BALANCE_POINT * (state + value) * (1.0 - SIGMA_INV)


def apply_n0_distribution(state: float, value: float, kappa: float = PHI_INV) -> float:
    """N0-4 (legacy): + GROUP - groups with golden-weighted distribution."""
    return (state + value) * kappa


def apply_n0_conservation(kappa: float, lambda_: float) -> tuple:
    """N0-5 (legacy): − SEPARATE - separates while conserving κ + λ = 1."""
    total = kappa + lambda_
    if total > 0:
        return kappa / total, lambda_ / total
    return PHI_INV, PHI_INV_SQ  # Default to golden balance


# =============================================================================
# SILENT LAW ACTIVATION FUNCTIONS
# =============================================================================

def compute_stillness_activation(z: float) -> float:
    """I STILLNESS: ∂E/∂t → 0 - peaks at z_c (equilibrium)."""
    return compute_delta_s_neg(z)


def compute_truth_activation(z: float) -> float:
    """II TRUTH: ∇V(truth) = 0 - stable at z_c."""
    if z >= Z_CRITICAL:
        return 1.0  # Full truth in PRESENCE
    elif z >= PHI_INV:
        return (z - PHI_INV) / (Z_CRITICAL - PHI_INV)
    else:
        return 0.0  # No truth in ABSENCE


def compute_silence_activation(conservation_error: float) -> float:
    """III SILENCE: ∇ · J = 0 - conservation always present."""
    # Activation = 1 - scaled conservation error
    return max(0.0, 1.0 - conservation_error / TOLERANCE_CONSERVATION)


def compute_spiral_activation(kappa: float) -> float:
    """IV SPIRAL: S(return) = S(origin) - peaks at golden ratio."""
    return math.exp(-SIGMA * (kappa - PHI_INV) ** 2)


def compute_unseen_activation(z: float) -> float:
    """V UNSEEN: P(observe) → 0 - hidden in ABSENCE."""
    if z < PHI_INV:
        return 1.0 - z / PHI_INV  # Full unseen at z=0
    else:
        return 0.0  # Visible in PARADOX/PRESENCE


def compute_glyph_activation(z: float) -> float:
    """VI GLYPH: glyph = ∫ life dt - structure accumulation."""
    return z  # Linear accumulation with z


def compute_mirror_activation(kappa: float) -> float:
    """VII MIRROR: ψ = ψ(ψ) - peaks at κ = λ = 0.5."""
    return math.exp(-SIGMA * (kappa - BALANCE_POINT) ** 2)


if __name__ == "__main__":
    print_all_constants()
