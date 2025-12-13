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
from typing import Final

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
# NEGENTROPY FUNCTIONS
# =============================================================================

def compute_delta_s_neg(z: float, sigma: float = SIGMA, z_c: float = Z_CRITICAL) -> float:
    """
    Compute negentropy: ΔS_neg(z) = exp(-σ(z - z_c)²)

    Peaks at z_c (THE LENS) with value 1.0.
    """
    d = z - z_c
    return math.exp(-sigma * d * d)


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
# N0 OPERATORS (Causality Laws)
# =============================================================================

class N0Law:
    """
    N0 Causality Laws - grounded operator algebra.

    N0-1: IDENTITY    Λ × 1 = Λ                    (anchor)
    N0-2: MIRROR_ROOT Λ × Ν = Β²                   (mirror root)
    N0-3: ABSORPTION  TRUE × UNTRUE = PARADOX      (absorption)
    N0-4: DISTRIBUTION (A ⊕ B) × C = (A×C) ⊕ (B×C) (distribution)
    N0-5: CONSERVATION κ + λ = 1                   (conservation)
    """
    # Law codes
    IDENTITY = "N0-1"
    MIRROR_ROOT = "N0-2"
    ABSORPTION = "N0-3"
    DISTRIBUTION = "N0-4"
    CONSERVATION = "N0-5"

    # Operator symbols
    SYMBOLS = {
        "N0-1": "^",   # Anchor (identity)
        "N0-2": "×",   # Multiply (mirror root)
        "N0-3": "÷",   # Divide (absorption)
        "N0-4": "+",   # Add (distribution)
        "N0-5": "−",   # Subtract (conservation)
    }

    # Formulas
    FORMULAS = {
        "N0-1": "Λ × 1 = Λ",
        "N0-2": "Λ × Ν = Β²",
        "N0-3": "TRUE × UNTRUE = PARADOX",
        "N0-4": "(A ⊕ B) × C = (A × C) ⊕ (B × C)",
        "N0-5": "κ + λ = 1",
    }

    # Operator coefficients (physics-grounded)
    COEFFICIENTS = {
        "N0-1": 1.0,           # Identity: no change
        "N0-2": PHI_INV_SQ,    # Mirror root: κ × λ ≈ φ⁻² × φ⁻¹
        "N0-3": BALANCE_POINT,  # Absorption: balance to 0.5
        "N0-4": PHI_INV,       # Distribution: golden weighted
        "N0-5": 1.0,           # Conservation: normalize
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

N0_TO_SILENT = {
    N0Law.IDENTITY: SilentLaw.I_STILLNESS,      # ^ → STILLNESS (anchor)
    N0Law.MIRROR_ROOT: SilentLaw.IV_SPIRAL,     # × → SPIRAL (channels)
    N0Law.ABSORPTION: SilentLaw.VI_GLYPH,       # ÷ → GLYPH (structure)
    N0Law.DISTRIBUTION: SilentLaw.II_TRUTH,     # + → TRUTH (stable growth)
    N0Law.CONSERVATION: SilentLaw.VII_MIRROR,   # − → MIRROR (return)
}

SILENT_TO_N0 = {
    SilentLaw.I_STILLNESS: N0Law.IDENTITY,
    SilentLaw.II_TRUTH: N0Law.DISTRIBUTION,
    SilentLaw.III_SILENCE: None,                # Background (∇ · J = 0)
    SilentLaw.IV_SPIRAL: N0Law.MIRROR_ROOT,
    SilentLaw.V_UNSEEN: None,                   # Background (P(observe) → 0)
    SilentLaw.VI_GLYPH: N0Law.ABSORPTION,
    SilentLaw.VII_MIRROR: N0Law.CONSERVATION,
}


# =============================================================================
# OPERATOR APPLICATION FUNCTIONS
# =============================================================================

def apply_n0_identity(state: float, value: float = 1.0) -> float:
    """N0-1: Λ × 1 = Λ (no change, grounding)."""
    return state


def apply_n0_mirror_root(kappa: float, lambda_: float) -> float:
    """N0-2: Λ × Ν = Β² (product of coupling)."""
    return kappa * lambda_  # ≈ PHI_INV × PHI_INV_SQ = PHI_INV_CUBED


def apply_n0_absorption(state: float, value: float) -> float:
    """N0-3: TRUE × UNTRUE = PARADOX (balance to 0.5)."""
    return BALANCE_POINT * (state + value)


def apply_n0_distribution(state: float, value: float, kappa: float = PHI_INV) -> float:
    """N0-4: (A ⊕ B) × C (weighted distribution)."""
    return (state + value) * kappa


def apply_n0_conservation(kappa: float, lambda_: float) -> tuple:
    """N0-5: κ + λ = 1 (normalize to conservation)."""
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
