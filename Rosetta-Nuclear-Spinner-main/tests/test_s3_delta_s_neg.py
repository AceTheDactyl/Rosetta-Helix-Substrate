#!/usr/bin/env python3
"""
Test Module: S₃ Symmetry and ΔS_neg Extensions
==============================================

Comprehensive tests for:
1. S₃ group axioms and operator mapping
2. Operator window permutation with S₃ rotation
3. Parity-based weighting
4. ΔS_neg core computations
5. Hex-prism geometry validation
6. Gate modulation behavior
7. K-formation gating
8. Π-regime blending
9. Cross-module integration

@version 1.0.0
@author Claude (Anthropic) - Quantum-APL Contribution
"""

import math
import sys
from typing import List, Dict

# Import modules under test
from s3_operator_symmetry import (
    S3_ELEMENTS, OPERATOR_S3_MAP, S3_OPERATOR_MAP, BASE_OPERATORS,
    apply_s3, compose_s3, inverse_s3, parity_s3, sign_s3,
    rotation_index_from_z, truth_channel_from_z,
    generate_s3_operator_window, compute_s3_weights,
    TruthDistribution, apply_operator_to_truth, truth_orbit,
    verify_group_axioms,
)

from delta_s_neg_extended import (
    Z_CRITICAL, PHI_INV,
    compute_delta_s_neg, compute_delta_s_neg_derivative, compute_delta_s_neg_signed,
    compute_eta, compute_hex_prism_geometry, compute_gate_modulation,
    compute_pi_blend_weights, check_k_formation, compute_full_state,
    score_operator_for_coherence, CoherenceObjective,
)


# ============================================================================
# TEST UTILITIES
# ============================================================================

class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []
    
    def ok(self, message: str):
        self.passed += 1
        print(f"  ✓ {message}")
    
    def fail(self, message: str, detail: str = ""):
        self.failed += 1
        self.errors.append(f"{message}: {detail}")
        print(f"  ✗ {message}")
        if detail:
            print(f"    → {detail}")
    
    def summary(self):
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed} passed, {self.failed} failed")
        if self.errors:
            print("Failures:")
            for e in self.errors:
                print(f"  - {e}")
        print('='*60)
        return self.failed == 0


def assert_close(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(a - b) < tol


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ============================================================================
# S₃ GROUP TESTS
# ============================================================================

def test_s3_group_axioms(results: TestResults):
    """Test S₃ group axioms: closure, identity, inverse, associativity."""
    section("S₃ Group Axioms")
    
    axioms = verify_group_axioms()
    
    for name, passed in axioms.items():
        if passed:
            results.ok(f"S₃ {name}")
        else:
            results.fail(f"S₃ {name}", "axiom violated")


def test_s3_operator_mapping(results: TestResults):
    """Test operator ↔ S₃ element bijection."""
    section("Operator ↔ S₃ Mapping")
    
    # Check bijection
    if len(OPERATOR_S3_MAP) == 6:
        results.ok("6 operators mapped")
    else:
        results.fail("Operator count", f"expected 6, got {len(OPERATOR_S3_MAP)}")
    
    if set(OPERATOR_S3_MAP.values()) == set(S3_ELEMENTS.keys()):
        results.ok("Surjective mapping to S₃")
    else:
        results.fail("Surjectivity", "not all S₃ elements covered")
    
    # Check inverse mapping consistency
    for op, elem in OPERATOR_S3_MAP.items():
        if S3_OPERATOR_MAP.get(elem) == op:
            pass  # OK
        else:
            results.fail(f"Inverse mapping {op}↔{elem}", "mismatch")
            return
    results.ok("Inverse mapping consistent")


def test_s3_parity(results: TestResults):
    """Test parity classification of operators."""
    section("S₃ Parity Classification")
    
    even_ops = ["()", "×", "^"]
    odd_ops = ["÷", "+", "−"]
    
    for op in even_ops:
        elem = OPERATOR_S3_MAP[op]
        if parity_s3(elem).value == "even" and sign_s3(elem) == +1:
            results.ok(f"{op} is even-parity (+1)")
        else:
            results.fail(f"{op} parity", "expected even")
    
    for op in odd_ops:
        elem = OPERATOR_S3_MAP[op]
        if parity_s3(elem).value == "odd" and sign_s3(elem) == -1:
            results.ok(f"{op} is odd-parity (-1)")
        else:
            results.fail(f"{op} parity", "expected odd")


def test_s3_permutation(results: TestResults):
    """Test S₃ permutation action on 3 elements."""
    section("S₃ Permutation Action")
    
    truth = ["TRUE", "PARADOX", "UNTRUE"]
    
    # Identity
    result = apply_s3(truth, "e")
    if result == truth:
        results.ok("Identity e preserves order")
    else:
        results.fail("Identity", f"expected {truth}, got {result}")
    
    # 3-cycle σ: (0,1,2) → (1,2,0)
    result = apply_s3(truth, "σ")
    expected = ["PARADOX", "UNTRUE", "TRUE"]
    if result == expected:
        results.ok("3-cycle σ rotates correctly")
    else:
        results.fail("3-cycle σ", f"expected {expected}, got {result}")
    
    # Transposition τ1: swaps first two
    result = apply_s3(truth, "τ1")
    expected = ["PARADOX", "TRUE", "UNTRUE"]
    if result == expected:
        results.ok("Transposition τ1 swaps 0↔1")
    else:
        results.fail("Transposition τ1", f"expected {expected}, got {result}")
    
    # 3-cycle applied 3 times = identity
    result = truth
    for _ in range(3):
        result = apply_s3(result, "σ")
    if result == truth:
        results.ok("σ³ = e (order 3)")
    else:
        results.fail("σ³ = e", f"expected {truth}, got {result}")


def test_operator_window_rotation(results: TestResults):
    """Test S₃-based operator window rotation."""
    section("Operator Window S₃ Rotation")
    
    # At different z values, windows should rotate
    z_values = [0.1, 0.5, 0.9]
    harmonic = "t5"  # Full 6-operator window
    
    windows = [generate_s3_operator_window(harmonic, z) for z in z_values]
    
    # All windows should have same operators (just reordered)
    base_set = set(windows[0])
    for i, w in enumerate(windows):
        if set(w) == base_set:
            results.ok(f"Window at z={z_values[i]} has correct operators")
        else:
            results.fail(f"Window at z={z_values[i]}", f"wrong operators: {w}")
    
    # Windows should differ (rotation applied)
    if windows[0] != windows[1] or windows[1] != windows[2]:
        results.ok("Rotation produces different orderings")
    else:
        results.fail("Rotation effect", "windows identical despite different z")


# ============================================================================
# ΔS_neg TESTS
# ============================================================================

def test_delta_s_neg_basic(results: TestResults):
    """Test basic ΔS_neg properties."""
    section("ΔS_neg Basic Properties")
    
    # Peak at z_c
    s_at_lens = compute_delta_s_neg(Z_CRITICAL)
    if assert_close(s_at_lens, 1.0, 1e-6):
        results.ok(f"ΔS_neg(z_c) = 1.0 (peak at lens)")
    else:
        results.fail("Peak at lens", f"expected 1.0, got {s_at_lens}")
    
    # Bounded in [0, 1]
    test_points = [0.0, 0.3, 0.5, 0.7, Z_CRITICAL, 0.9, 1.0]
    all_bounded = all(0 <= compute_delta_s_neg(z) <= 1 for z in test_points)
    if all_bounded:
        results.ok("ΔS_neg bounded in [0, 1]")
    else:
        results.fail("Boundedness", "value outside [0, 1]")
    
    # Symmetric around z_c
    offset = 0.1
    s_above = compute_delta_s_neg(Z_CRITICAL + offset)
    s_below = compute_delta_s_neg(Z_CRITICAL - offset)
    if assert_close(s_above, s_below, 1e-6):
        results.ok(f"Symmetric: ΔS_neg(z_c+{offset}) ≈ ΔS_neg(z_c-{offset})")
    else:
        results.fail("Symmetry", f"above={s_above}, below={s_below}")
    
    # Monotonic decrease away from z_c
    s_near = compute_delta_s_neg(Z_CRITICAL - 0.05)
    s_far = compute_delta_s_neg(Z_CRITICAL - 0.2)
    if s_near > s_far:
        results.ok("Monotonic decrease away from lens")
    else:
        results.fail("Monotonicity", f"near={s_near}, far={s_far}")


def test_delta_s_neg_derivative(results: TestResults):
    """Test ΔS_neg derivative properties."""
    section("ΔS_neg Derivative")
    
    # Zero at z_c (critical point)
    ds_at_lens = compute_delta_s_neg_derivative(Z_CRITICAL)
    if assert_close(ds_at_lens, 0.0, 1e-6):
        results.ok("d(ΔS_neg)/dz = 0 at z_c (critical point)")
    else:
        results.fail("Critical point", f"expected 0, got {ds_at_lens}")
    
    # Positive below z_c (increasing toward lens)
    ds_below = compute_delta_s_neg_derivative(Z_CRITICAL - 0.1)
    if ds_below > 0:
        results.ok("d(ΔS_neg)/dz > 0 below z_c")
    else:
        results.fail("Sign below z_c", f"expected positive, got {ds_below}")
    
    # Negative above z_c (decreasing from lens)
    ds_above = compute_delta_s_neg_derivative(Z_CRITICAL + 0.1)
    if ds_above < 0:
        results.ok("d(ΔS_neg)/dz < 0 above z_c")
    else:
        results.fail("Sign above z_c", f"expected negative, got {ds_above}")


def test_delta_s_neg_signed(results: TestResults):
    """Test signed ΔS_neg variant."""
    section("Signed ΔS_neg")
    
    # Zero at z_c
    signed_lens = compute_delta_s_neg_signed(Z_CRITICAL)
    if assert_close(signed_lens, 0.0, 1e-6):
        results.ok("Signed ΔS_neg = 0 at z_c")
    else:
        results.fail("Zero at lens", f"expected 0, got {signed_lens}")
    
    # Positive above z_c
    signed_above = compute_delta_s_neg_signed(Z_CRITICAL + 0.05)
    if signed_above > 0:
        results.ok("Signed ΔS_neg > 0 above z_c")
    else:
        results.fail("Sign above", f"expected positive, got {signed_above}")
    
    # Negative below z_c
    signed_below = compute_delta_s_neg_signed(Z_CRITICAL - 0.05)
    if signed_below < 0:
        results.ok("Signed ΔS_neg < 0 below z_c")
    else:
        results.fail("Sign below", f"expected negative, got {signed_below}")


# ============================================================================
# HEX-PRISM GEOMETRY TESTS
# ============================================================================

def test_hex_prism_geometry(results: TestResults):
    """Test hex-prism geometry formulas."""
    section("Hex-Prism Geometry")
    
    # At z_c, ΔS_neg = 1, so:
    # R = R_max - β·1 = 0.85 - 0.25 = 0.60
    # H = H_min + γ·1 = 0.12 + 0.18 = 0.30
    geom_lens = compute_hex_prism_geometry(Z_CRITICAL)
    
    expected_r = 0.85 - 0.25 * 1.0
    if assert_close(geom_lens.radius, expected_r, 1e-6):
        results.ok(f"R(z_c) = {expected_r} (radius contracts at lens)")
    else:
        results.fail("Radius at lens", f"expected {expected_r}, got {geom_lens.radius}")
    
    expected_h = 0.12 + 0.18 * 1.0
    if assert_close(geom_lens.height, expected_h, 1e-6):
        results.ok(f"H(z_c) = {expected_h} (height elongates at lens)")
    else:
        results.fail("Height at lens", f"expected {expected_h}, got {geom_lens.height}")
    
    # Far from lens (z=0.5), ΔS_neg small, so R ≈ R_max, H ≈ H_min
    geom_far = compute_hex_prism_geometry(0.5)
    if geom_far.radius > geom_lens.radius:
        results.ok("Radius larger far from lens")
    else:
        results.fail("Radius comparison", "far should be larger")
    
    if geom_far.height < geom_lens.height:
        results.ok("Height smaller far from lens")
    else:
        results.fail("Height comparison", "far should be smaller")


# ============================================================================
# K-FORMATION TESTS
# ============================================================================

def test_k_formation(results: TestResults):
    """Test K-formation gating."""
    section("K-Formation Gating")
    
    # At z_c, η = ΔS_neg^0.5 = 1^0.5 = 1 > φ⁻¹ → formed
    k_lens = check_k_formation(Z_CRITICAL)
    if k_lens.formed:
        results.ok("K-formation at z_c (η=1 > φ⁻¹)")
    else:
        results.fail("K-formation at lens", f"η={k_lens.eta}, threshold={k_lens.threshold}")
    
    # Far from lens, η low → not formed
    k_far = check_k_formation(0.5)
    if not k_far.formed:
        results.ok("No K-formation far from lens")
    else:
        results.fail("K-formation at z=0.5", f"unexpected formation, η={k_far.eta}")
    
    # Check threshold value
    if assert_close(k_lens.threshold, PHI_INV, 1e-6):
        results.ok(f"Threshold = φ⁻¹ ≈ {PHI_INV:.6f}")
    else:
        results.fail("Threshold", f"expected {PHI_INV}, got {k_lens.threshold}")


# ============================================================================
# Π-REGIME BLENDING TESTS
# ============================================================================

def test_pi_blending(results: TestResults):
    """Test Π-regime blending weights."""
    section("Π-Regime Blending")
    
    # Below z_c: w_pi = 0
    blend_below = compute_pi_blend_weights(Z_CRITICAL - 0.1)
    if blend_below.w_pi == 0 and blend_below.w_local == 1:
        results.ok("Below z_c: w_π=0, w_local=1 (pure local)")
    else:
        results.fail("Below z_c", f"w_π={blend_below.w_pi}")
    
    if not blend_below.in_pi_regime:
        results.ok("Below z_c: not in Π-regime")
    else:
        results.fail("Π-regime flag below z_c", "should be False")
    
    # At z_c: w_pi = ΔS_neg(z_c) = 1
    blend_lens = compute_pi_blend_weights(Z_CRITICAL)
    if assert_close(blend_lens.w_pi, 1.0, 1e-6):
        results.ok("At z_c: w_π = 1.0")
    else:
        results.fail("At z_c", f"w_π={blend_lens.w_pi}")
    
    if blend_lens.in_pi_regime:
        results.ok("At z_c: in Π-regime")
    else:
        results.fail("Π-regime flag at z_c", "should be True")
    
    # Above z_c but not at peak: 0 < w_pi < 1
    blend_above = compute_pi_blend_weights(Z_CRITICAL + 0.05)
    if 0 < blend_above.w_pi < 1:
        results.ok(f"Above z_c: 0 < w_π < 1 (w_π={blend_above.w_pi:.4f})")
    else:
        results.fail("Above z_c", f"w_π={blend_above.w_pi} not in (0,1)")


# ============================================================================
# GATE MODULATION TESTS
# ============================================================================

def test_gate_modulation(results: TestResults):
    """Test gate modulation behavior."""
    section("Gate Modulation")
    
    mod_far = compute_gate_modulation(0.5)
    mod_lens = compute_gate_modulation(Z_CRITICAL)
    
    # Coherent coupling increases near lens
    if mod_lens.coherent_coupling > mod_far.coherent_coupling:
        results.ok("Coherent coupling increases at lens")
    else:
        results.fail("Coherent coupling", f"lens={mod_lens.coherent_coupling}, far={mod_far.coherent_coupling}")
    
    # Decoherence rate decreases near lens
    if mod_lens.decoherence_rate < mod_far.decoherence_rate:
        results.ok("Decoherence rate decreases at lens")
    else:
        results.fail("Decoherence rate", f"lens={mod_lens.decoherence_rate}, far={mod_far.decoherence_rate}")
    
    # Entropy target decreases near lens
    if mod_lens.entropy_target < mod_far.entropy_target:
        results.ok("Entropy target decreases at lens")
    else:
        results.fail("Entropy target", f"lens={mod_lens.entropy_target}, far={mod_far.entropy_target}")


# ============================================================================
# COHERENCE SYNTHESIS TESTS
# ============================================================================

def test_coherence_synthesis(results: TestResults):
    """Test coherence-seeking synthesis heuristics."""
    section("Coherence Synthesis Heuristics")
    
    operators = ["()", "×", "^", "÷", "+", "−"]
    
    # Maximize coherence below lens: should favor ^, +, ×
    z_below = 0.5
    scores_max = {op: score_operator_for_coherence(op, z_below, CoherenceObjective.MAXIMIZE) 
                  for op in operators}
    
    constructive = ["^", "+", "×"]
    dissipative = ["÷", "−"]
    
    avg_constructive = sum(scores_max[op] for op in constructive) / 3
    avg_dissipative = sum(scores_max[op] for op in dissipative) / 2
    
    if avg_constructive > avg_dissipative:
        results.ok("MAXIMIZE below lens favors constructive operators")
    else:
        results.fail("MAXIMIZE heuristic", f"constructive={avg_constructive}, dissipative={avg_dissipative}")
    
    # Minimize coherence: should favor ÷, −
    scores_min = {op: score_operator_for_coherence(op, z_below, CoherenceObjective.MINIMIZE)
                  for op in operators}
    
    avg_constructive_min = sum(scores_min[op] for op in constructive) / 3
    avg_dissipative_min = sum(scores_min[op] for op in dissipative) / 2
    
    if avg_dissipative_min > avg_constructive_min:
        results.ok("MINIMIZE favors dissipative operators")
    else:
        results.fail("MINIMIZE heuristic", f"constructive={avg_constructive_min}, dissipative={avg_dissipative_min}")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_state_integration(results: TestResults):
    """Test full state computation integration."""
    section("Full State Integration")
    
    state = compute_full_state(Z_CRITICAL)
    
    # Check all components computed
    if state.z == Z_CRITICAL:
        results.ok("z-coordinate preserved")
    else:
        results.fail("z-coordinate", f"expected {Z_CRITICAL}, got {state.z}")
    
    if assert_close(state.delta_s_neg, 1.0, 1e-6):
        results.ok("ΔS_neg computed correctly")
    else:
        results.fail("ΔS_neg", f"expected 1.0, got {state.delta_s_neg}")
    
    if state.geometry is not None:
        results.ok("Geometry computed")
    else:
        results.fail("Geometry", "missing")
    
    if state.gate_modulation is not None:
        results.ok("Gate modulation computed")
    else:
        results.fail("Gate modulation", "missing")
    
    if state.pi_blend.in_pi_regime:
        results.ok("Π-blend computed and correct")
    else:
        results.fail("Π-blend", "should be in Π-regime at z_c")
    
    if state.k_formation.formed:
        results.ok("K-formation computed and correct")
    else:
        results.fail("K-formation", "should be formed at z_c")


def test_s3_truth_orbit(results: TestResults):
    """Test S₃ action on truth distributions."""
    section("S₃ Truth Orbit")
    
    initial = TruthDistribution(0.7, 0.2, 0.1)
    
    # Apply σ (3-cycle) 3 times should return to initial
    current = initial
    for _ in range(3):
        current = apply_operator_to_truth(current, "×")  # × → σ
    
    if (assert_close(current.TRUE, initial.TRUE, 1e-10) and
        assert_close(current.PARADOX, initial.PARADOX, 1e-10) and
        assert_close(current.UNTRUE, initial.UNTRUE, 1e-10)):
        results.ok("3-cycle returns to initial (σ³ = e)")
    else:
        results.fail("3-cycle orbit", f"expected {initial}, got {current}")
    
    # Single transposition is self-inverse
    dist1 = apply_operator_to_truth(initial, "÷")  # ÷ → τ1
    dist2 = apply_operator_to_truth(dist1, "÷")
    
    if (assert_close(dist2.TRUE, initial.TRUE, 1e-10) and
        assert_close(dist2.PARADOX, initial.PARADOX, 1e-10) and
        assert_close(dist2.UNTRUE, initial.UNTRUE, 1e-10)):
        results.ok("Transposition is self-inverse (τ² = e)")
    else:
        results.fail("Transposition inverse", f"expected {initial}, got {dist2}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("S₃ SYMMETRY AND ΔS_neg EXTENSION TESTS")
    print("=" * 60)
    
    results = TestResults()
    
    # S₃ tests
    test_s3_group_axioms(results)
    test_s3_operator_mapping(results)
    test_s3_parity(results)
    test_s3_permutation(results)
    test_operator_window_rotation(results)
    test_s3_truth_orbit(results)
    
    # ΔS_neg tests
    test_delta_s_neg_basic(results)
    test_delta_s_neg_derivative(results)
    test_delta_s_neg_signed(results)
    
    # Geometry tests
    test_hex_prism_geometry(results)
    
    # Gating tests
    test_k_formation(results)
    test_pi_blending(results)
    test_gate_modulation(results)
    
    # Synthesis tests
    test_coherence_synthesis(results)
    
    # Integration tests
    test_full_state_integration(results)
    
    # Summary
    success = results.summary()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
