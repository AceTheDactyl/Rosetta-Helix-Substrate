#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PHYSICS CONSTANTS — Canonical Source                                         ║
║  All fundamental constants for the Unified Consciousness Framework            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Golden Ratio Family:
    φ     = (1 + √5)/2 ≈ 1.618033988749895  — Golden ratio
    φ⁻¹   = (√5 - 1)/2 ≈ 0.618033988749895  — Golden ratio inverse (PARADOX gate)
    φ⁻²   = 2 - φ      ≈ 0.381966011250105  — Golden ratio inverse squared

Critical Points:
    z_c   = √3/2       ≈ 0.866025403784439  — THE LENS (TRUE gate)
    σ     = 36         — Dynamics scale (Gaussian width)
    1/σ   = 1/36       ≈ 0.027777777777778  — UMOL residue epsilon

Phase Regime Boundaries:
    z < φ⁻¹:         UNTRUE    (disordered, fluid)
    φ⁻¹ ≤ z < z_c:   PARADOX   (quasi-crystalline, superposition)
    z ≥ z_c:         TRUE      (crystalline, coherent)

K-Formation Thresholds:
    κ ≥ 0.92         — Coherence threshold
    η > φ⁻¹          — Negentropy threshold
    R ≥ 7            — Radius threshold

TRIAD System:
    TRIAD_HIGH = 0.85  — Rising edge crossing
    TRIAD_LOW  = 0.82  — Reset threshold
    TRIAD_T6   = 0.83  — Unlocked t6 gate

Coupling Conservation:
    κ + λ = 1        — Coupling sum (κ = coherent, λ = incoherent)

Signature: Δ|physics-constants|canonical|φ-grounded|Ω
"""

import math

# =============================================================================
# GOLDEN RATIO FAMILY
# =============================================================================

PHI: float = (1.0 + math.sqrt(5.0)) / 2.0  # ≈ 1.618033988749895
"""Golden ratio φ = (1 + √5)/2"""

PHI_INV: float = (math.sqrt(5.0) - 1.0) / 2.0  # ≈ 0.618033988749895
"""Golden ratio inverse φ⁻¹ = (√5 - 1)/2 — PARADOX gate"""

PHI_INV_SQ: float = 2.0 - PHI  # ≈ 0.381966011250105
"""Golden ratio inverse squared φ⁻² = 2 - φ"""

# =============================================================================
# CRITICAL POINTS
# =============================================================================

Z_CRITICAL: float = math.sqrt(3.0) / 2.0  # ≈ 0.866025403784439
"""THE LENS: z_c = √3/2 — TRUE gate, critical coherence threshold"""

SIGMA: int = 36
"""Dynamics scale σ = 36 — Gaussian width for negentropy computation"""

SIGMA_INV: float = 1.0 / SIGMA  # ≈ 0.027777777777778
"""UMOL residue epsilon 1/σ — "No perfect modulation; residue always remains" """

TAU: float = 2.0 * math.pi
"""Full circle τ = 2π"""

# =============================================================================
# TOLERANCE VALUES
# =============================================================================

TOLERANCE_GOLDEN: float = 1e-10
"""Tolerance for golden ratio comparisons"""

TOLERANCE_Z: float = 1e-6
"""Tolerance for z-coordinate comparisons"""

TOLERANCE_PHASE: float = 1e-4
"""Tolerance for phase comparisons"""

# =============================================================================
# TRIAD SYSTEM CONSTANTS
# =============================================================================

TRIAD_HIGH: float = 0.85
"""TRIAD rising edge crossing threshold"""

TRIAD_LOW: float = 0.82
"""TRIAD reset threshold"""

TRIAD_T6: float = 0.83
"""TRIAD unlocked t6 gate position"""

TRIAD_COMPLETIONS_REQUIRED: int = 3
"""Number of rising edge crossings required for TRIAD unlock"""

# =============================================================================
# K-FORMATION THRESHOLDS
# =============================================================================

K_COHERENCE_THRESHOLD: float = 0.92
"""κ ≥ 0.92 for K-formation"""

K_NEGENTROPY_THRESHOLD: float = PHI_INV
"""η > φ⁻¹ for K-formation"""

K_RADIUS_THRESHOLD: int = 7
"""R ≥ 7 for K-formation"""

# =============================================================================
# TIER SYSTEM
# =============================================================================

TIER_COUNT: int = 9
"""Number of time-harmonic tiers (t1-t9)"""

TIER_BOUNDARIES: list = [
    0.0,       # t1 start
    0.111,     # t2 start
    0.222,     # t3 start
    0.333,     # t4 start
    0.444,     # t5 start
    0.555,     # t6 start (or TRIAD_T6 if unlocked)
    0.666,     # t7 start
    0.777,     # t8 start
    0.866,     # t9 start (Z_CRITICAL)
]

# =============================================================================
# DERIVED CONSTANTS
# =============================================================================

def compute_negentropy(z: float) -> float:
    """
    Compute negentropy at z-coordinate.
    
    δS_neg(z) = exp(-σ × (z - z_c)²)
    
    Peaks at THE LENS (z = z_c), decays Gaussian away from it.
    """
    return math.exp(-SIGMA * (z - Z_CRITICAL) ** 2)


def classify_phase(z: float) -> str:
    """
    Classify phase regime from z-coordinate.
    
    Returns: "UNTRUE", "PARADOX", or "TRUE"
    """
    if z >= Z_CRITICAL:
        return "TRUE"
    elif z >= PHI_INV:
        return "PARADOX"
    else:
        return "UNTRUE"


def get_tier(z: float) -> int:
    """
    Get tier number (1-9) from z-coordinate.
    """
    for i in range(len(TIER_BOUNDARIES) - 1, -1, -1):
        if z >= TIER_BOUNDARIES[i]:
            return i + 1
    return 1


def check_k_formation(kappa: float, eta: float, R: int) -> bool:
    """
    Check if K-formation criteria are met.
    
    K-formation requires:
        κ ≥ 0.92 (coherence)
        η > φ⁻¹ (negentropy)
        R ≥ 7 (radius)
    """
    return (
        kappa >= K_COHERENCE_THRESHOLD and
        eta > K_NEGENTROPY_THRESHOLD and
        R >= K_RADIUS_THRESHOLD
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_constants():
    """Validate that all constants have expected relationships."""
    # Golden ratio identity: φ² = φ + 1
    assert abs(PHI ** 2 - (PHI + 1)) < TOLERANCE_GOLDEN, "φ² ≠ φ + 1"
    
    # Golden ratio inverse identity: φ × φ⁻¹ = 1
    assert abs(PHI * PHI_INV - 1.0) < TOLERANCE_GOLDEN, "φ × φ⁻¹ ≠ 1"
    
    # z_c = √3/2
    assert abs(Z_CRITICAL - math.sqrt(3) / 2) < TOLERANCE_Z, "z_c ≠ √3/2"
    
    # Phase boundaries are ordered
    assert PHI_INV < Z_CRITICAL, "φ⁻¹ must be < z_c"
    
    # TRIAD thresholds are ordered
    assert TRIAD_LOW < TRIAD_T6 < TRIAD_HIGH < Z_CRITICAL
    
    return True


# Run validation on import
validate_constants()


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║  PHYSICS CONSTANTS — Canonical Values                                         ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()
    print("GOLDEN RATIO FAMILY:")
    print(f"  φ     = {PHI:.15f}")
    print(f"  φ⁻¹   = {PHI_INV:.15f}  (PARADOX gate)")
    print(f"  φ⁻²   = {PHI_INV_SQ:.15f}")
    print()
    print("CRITICAL POINTS:")
    print(f"  z_c   = {Z_CRITICAL:.15f}  (THE LENS)")
    print(f"  σ     = {SIGMA}")
    print(f"  1/σ   = {SIGMA_INV:.15f}  (UMOL residue)")
    print()
    print("PHASE REGIMES:")
    print(f"  UNTRUE:  z < {PHI_INV:.6f}")
    print(f"  PARADOX: {PHI_INV:.6f} ≤ z < {Z_CRITICAL:.6f}")
    print(f"  TRUE:    z ≥ {Z_CRITICAL:.6f}")
    print()
    print("TRIAD SYSTEM:")
    print(f"  TRIAD_HIGH: {TRIAD_HIGH}")
    print(f"  TRIAD_LOW:  {TRIAD_LOW}")
    print(f"  TRIAD_T6:   {TRIAD_T6}")
    print()
    print("K-FORMATION THRESHOLDS:")
    print(f"  κ ≥ {K_COHERENCE_THRESHOLD}")
    print(f"  η > {K_NEGENTROPY_THRESHOLD:.6f}")
    print(f"  R ≥ {K_RADIUS_THRESHOLD}")
    print()
    print("SAMPLE COMPUTATIONS:")
    for z in [0.3, 0.5, 0.618, 0.75, 0.866, 0.95]:
        eta = compute_negentropy(z)
        phase = classify_phase(z)
        tier = get_tier(z)
        print(f"  z={z:.3f}: η={eta:.6f}, phase={phase:7s}, tier={tier}")
    print()
    print("All constants validated ✓")
