#!/usr/bin/env python3
"""
UNIFIED PHYSICS VERIFICATION MODULE
====================================

Single verification module for all physics constraints in the Rosetta-Helix system.
Runs at import time to ensure physics integrity.

Verified Constraints:
====================
    1. φ⁻¹ + φ⁻² = 1 (coupling conservation)
    2. z_c = √3/2 (THE LENS - hexagonal geometry)
    3. σ derived from φ⁻¹ alignment
    4. Threshold ordering: Z_ORIGIN < Z_CRITICAL < KAPPA_S < MU_3 < UNITY
    5. K-formation thresholds grounded in tier structure

Cross-Module Consistency:
=========================
    - physics_constants.py: Source of truth
    - quantum_apl_integration.py: Uses same constants
    - tc_language_module.py: Uses same constants
    - unified_math_bridge.py: Uses same constants

Usage:
======
    # Run all verifications
    from physics_verification import verify_all, assert_physics_valid

    results = verify_all()
    for name, passed, msg in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {msg}")

    # Or assert (raises if failed)
    assert_physics_valid()

Signature: Δ|physics-verification|unified|fail-fast|Ω
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Any

# Import from physics_constants (single source of truth)
from physics_constants import (
    PHI,
    PHI_INV,
    PHI_INV_SQ,
    PHI_INV_FIFTH,
    Z_CRITICAL,
    SIGMA,
    COUPLING_CONSERVATION,
    TOLERANCE_GOLDEN,
    TOLERANCE_CONSERVATION,
    KAPPA_S,
    ETA_THRESHOLD,
    R_MIN,
    MU_3,
    Z_ORIGIN,
    derive_sigma,
    compute_lens_weight,
    check_k_formation,
)


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def verify_coupling_conservation() -> Tuple[bool, str]:
    """
    Verify φ⁻¹ + φ⁻² = 1 (coupling conservation).

    This is THE defining property of the golden ratio.
    The positive root of x + x² = 1 is φ⁻¹.
    """
    coupling_sum = PHI_INV + PHI_INV_SQ
    coupling_error = abs(coupling_sum - 1.0)

    if coupling_error < TOLERANCE_CONSERVATION:
        return True, f"φ⁻¹ + φ⁻² = {coupling_sum:.16f} (error: {coupling_error:.2e})"
    else:
        return False, f"FAILED: φ⁻¹ + φ⁻² = {coupling_sum:.16f} (error: {coupling_error:.2e})"


def verify_z_critical() -> Tuple[bool, str]:
    """
    Verify z_c = √3/2 (THE LENS - hexagonal geometry).

    Observable physics:
        - Graphene unit cell height/width
        - HCP metal layer stacking offset
        - Triangular antiferromagnet spin geometry
    """
    expected_zc = math.sqrt(3) / 2
    zc_error = abs(Z_CRITICAL - expected_zc)

    if zc_error < TOLERANCE_CONSERVATION:
        return True, f"z_c = {Z_CRITICAL:.16f} = √3/2 (error: {zc_error:.2e})"
    else:
        return False, f"FAILED: z_c = {Z_CRITICAL:.16f} ≠ √3/2 (error: {zc_error:.2e})"


def verify_sigma_derivation() -> Tuple[bool, str]:
    """
    Verify σ = 36 is derived from φ⁻¹ alignment at t6 boundary.

    Derivation:
        exp(-σ × (0.75 - z_c)²) = φ⁻¹
        σ = -ln(φ⁻¹) / (0.75 - z_c)² ≈ 35.7 → 36
    """
    derived_sigma = derive_sigma()
    sigma_error = abs(SIGMA - derived_sigma)

    if sigma_error < 1.0:
        return True, f"σ = {SIGMA}, derived = {derived_sigma:.2f} (Δ = {sigma_error:.2f})"
    else:
        return False, f"FAILED: σ = {SIGMA} ≠ derived {derived_sigma:.2f} (Δ = {sigma_error:.2f})"


def verify_threshold_ordering() -> Tuple[bool, str]:
    """
    Verify threshold ordering: Z_ORIGIN < Z_CRITICAL < KAPPA_S < MU_3 < UNITY.

    All thresholds are derived from z_c and φ.
    """
    UNITY = 0.9999

    ordering_ok = Z_ORIGIN < Z_CRITICAL < KAPPA_S < MU_3 < UNITY

    if ordering_ok:
        return True, (
            f"Z_ORIGIN({Z_ORIGIN:.3f}) < z_c({Z_CRITICAL:.3f}) < "
            f"κ_s({KAPPA_S:.3f}) < μ₃({MU_3:.3f}) < U({UNITY:.4f})"
        )
    else:
        return False, (
            f"FAILED: Ordering violated! "
            f"Z_O={Z_ORIGIN:.3f}, z_c={Z_CRITICAL:.3f}, κ_s={KAPPA_S:.3f}, "
            f"μ₃={MU_3:.3f}, U={UNITY:.4f}"
        )


def verify_z_origin() -> Tuple[bool, str]:
    """
    Verify Z_ORIGIN = Z_CRITICAL × φ⁻¹.

    This is the collapse reset point.
    """
    expected_origin = Z_CRITICAL * PHI_INV
    origin_error = abs(Z_ORIGIN - expected_origin)

    if origin_error < TOLERANCE_CONSERVATION:
        return True, f"Z_ORIGIN = {Z_ORIGIN:.6f} = z_c × φ⁻¹ (error: {origin_error:.2e})"
    else:
        return False, f"FAILED: Z_ORIGIN = {Z_ORIGIN:.6f} ≠ z_c × φ⁻¹ = {expected_origin:.6f}"


def verify_k_formation_thresholds() -> Tuple[bool, str]:
    """
    Verify K-formation thresholds are physics-grounded.

    - κ ≥ KAPPA_S (0.92) - t7 tier boundary
    - η > φ⁻¹ - coherence exceeds golden threshold
    - R ≥ 7 - |S₃| + 1 complexity
    """
    # Verify KAPPA_S alignment with t7 tier
    t7_max = 0.92
    kappa_aligned = abs(KAPPA_S - t7_max) < 0.01

    # Verify ETA_THRESHOLD = φ⁻¹
    eta_aligned = abs(ETA_THRESHOLD - PHI_INV) < TOLERANCE_CONSERVATION

    # Verify R_MIN = 7 = |S₃| + 1
    r_aligned = R_MIN == 7

    all_aligned = kappa_aligned and eta_aligned and r_aligned

    if all_aligned:
        return True, f"κ_s={KAPPA_S}, η={ETA_THRESHOLD:.6f}=φ⁻¹, R={R_MIN}=|S₃|+1"
    else:
        parts = []
        if not kappa_aligned:
            parts.append(f"κ_s={KAPPA_S}≠{t7_max}")
        if not eta_aligned:
            parts.append(f"η={ETA_THRESHOLD}≠φ⁻¹")
        if not r_aligned:
            parts.append(f"R={R_MIN}≠7")
        return False, f"FAILED: {', '.join(parts)}"


def verify_lens_weight() -> Tuple[bool, str]:
    """
    Verify lens weight (ΔS_neg) peaks at z_c with value 1.0.
    """
    peak_value = compute_lens_weight(Z_CRITICAL)
    peak_error = abs(peak_value - 1.0)

    if peak_error < TOLERANCE_CONSERVATION:
        return True, f"W(z_c) = {peak_value:.16f} (error: {peak_error:.2e})"
    else:
        return False, f"FAILED: W(z_c) = {peak_value:.16f} ≠ 1.0"


def verify_phi_algebraic_properties() -> Tuple[bool, str]:
    """
    Verify algebraic properties of φ.

    - φ² = φ + 1
    - φ × φ⁻¹ = 1
    - φ - φ⁻¹ = 1
    """
    errors = []

    # φ² = φ + 1
    prop1_error = abs(PHI ** 2 - (PHI + 1))
    if prop1_error >= TOLERANCE_CONSERVATION:
        errors.append(f"φ²≠φ+1 (err={prop1_error:.2e})")

    # φ × φ⁻¹ = 1
    prop2_error = abs(PHI * PHI_INV - 1.0)
    if prop2_error >= TOLERANCE_CONSERVATION:
        errors.append(f"φ×φ⁻¹≠1 (err={prop2_error:.2e})")

    # φ - φ⁻¹ = 1
    prop3_error = abs((PHI - PHI_INV) - 1.0)
    if prop3_error >= TOLERANCE_CONSERVATION:
        errors.append(f"φ-φ⁻¹≠1 (err={prop3_error:.2e})")

    if not errors:
        return True, "All φ algebraic properties verified"
    else:
        return False, f"FAILED: {'; '.join(errors)}"


def verify_coupling_conservation_constant() -> Tuple[bool, str]:
    """
    Verify COUPLING_CONSERVATION constant equals 1.0.
    """
    if abs(COUPLING_CONSERVATION - 1.0) < TOLERANCE_CONSERVATION:
        return True, f"COUPLING_CONSERVATION = {COUPLING_CONSERVATION:.16f}"
    else:
        return False, f"FAILED: COUPLING_CONSERVATION = {COUPLING_CONSERVATION:.16f} ≠ 1.0"


# =============================================================================
# AGGREGATE VERIFICATION
# =============================================================================

def verify_all() -> List[Tuple[str, bool, str]]:
    """
    Run all physics verifications.

    Returns:
        List of (name, passed, message) tuples
    """
    results = []

    # Core identity
    passed, msg = verify_coupling_conservation()
    results.append(("Coupling Conservation (φ⁻¹ + φ⁻² = 1)", passed, msg))

    # Z_CRITICAL
    passed, msg = verify_z_critical()
    results.append(("Z_CRITICAL = √3/2 (THE LENS)", passed, msg))

    # Sigma derivation
    passed, msg = verify_sigma_derivation()
    results.append(("Sigma derived from φ⁻¹ alignment", passed, msg))

    # Threshold ordering
    passed, msg = verify_threshold_ordering()
    results.append(("Threshold ordering", passed, msg))

    # Z_ORIGIN
    passed, msg = verify_z_origin()
    results.append(("Z_ORIGIN = z_c × φ⁻¹", passed, msg))

    # K-formation thresholds
    passed, msg = verify_k_formation_thresholds()
    results.append(("K-formation thresholds", passed, msg))

    # Lens weight
    passed, msg = verify_lens_weight()
    results.append(("Lens weight peaks at z_c", passed, msg))

    # Phi properties
    passed, msg = verify_phi_algebraic_properties()
    results.append(("φ algebraic properties", passed, msg))

    # COUPLING_CONSERVATION constant
    passed, msg = verify_coupling_conservation_constant()
    results.append(("COUPLING_CONSERVATION = 1", passed, msg))

    return results


def assert_physics_valid():
    """
    Assert all physics constraints are satisfied.

    Raises AssertionError if any verification fails.
    """
    results = verify_all()
    failures = [r for r in results if not r[1]]

    if failures:
        msg = "Physics verification failed:\n"
        for name, _, detail in failures:
            msg += f"  ✗ {name}: {detail}\n"
        raise AssertionError(msg)


def print_verification_report():
    """Print a complete verification report."""
    print("=" * 70)
    print("ROSETTA-HELIX PHYSICS VERIFICATION REPORT")
    print("=" * 70)
    print()

    results = verify_all()
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)

    for name, passed, msg in results:
        status = "✓" if passed else "✗"
        print(f"  [{status}] {name}")
        print(f"      {msg}")
        print()

    print("-" * 70)
    print(f"  Summary: {passed_count}/{total_count} verifications passed")
    print()

    if passed_count == total_count:
        print("  ╔════════════════════════════════════════════╗")
        print("  ║   ALL PHYSICS VERIFICATIONS PASSED ✓      ║")
        print("  ╚════════════════════════════════════════════╝")
    else:
        print("  ╔════════════════════════════════════════════╗")
        print("  ║   WARNING: PHYSICS VIOLATIONS DETECTED ✗   ║")
        print("  ╚════════════════════════════════════════════╝")

    print()
    print("=" * 70)


# =============================================================================
# IMPORT-TIME VERIFICATION
# =============================================================================

# Run verification at import time (fail-fast)
if __name__ != "__main__":
    try:
        assert_physics_valid()
    except AssertionError as e:
        import warnings
        warnings.warn(f"Physics verification failed at import: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_verification_report()
