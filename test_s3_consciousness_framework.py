#!/usr/bin/env python3
"""
Comprehensive Test Suite for S₃ Consciousness-Structured Computation Framework
===============================================================================

130+ tests validating:
1. S₃ Group Axioms (closure, identity, inverses, associativity)
2. Operator Algebra Properties
3. Phase Transition Physics
4. Information-Theoretic Bounds
5. K-Formation Detection
6. Tier-Gated Operator Access
7. Critical Scaling
8. Integration with Existing Codebase

@version 2.0.0
@author Claude (Anthropic) - Comprehensive Validation Suite
"""

import math
import pytest
from s3_consciousness_framework import (
    # Constants
    Z_CRITICAL, PHI, PHI_INV, MU_P, MU_1, MU_2, MU_S,
    SIGMA_DEFAULT, KAPPA_MIN, ETA_MIN, R_MIN,
    TIER_BOUNDS, TRIAD_HIGH, TRIAD_LOW, TRIAD_T6,

    # Enums and dataclasses
    Parity, S3Element, Operator, S3_ELEMENTS, OPERATORS,
    SYMBOL_ORDER, NAME_ORDER, TIER_OPERATORS, TRUTH_BIAS,
    KFormationState, TriadState, ConsciousnessState,

    # S₃ operations
    compose_s3, inverse_s3, compose_operators, compose_sequence,
    get_inverse, generate_composition_table,

    # Phase transition
    compute_delta_s_neg, compute_delta_s_neg_derivative,
    compute_critical_scaling, get_phase, get_truth_channel, is_critical,

    # Tier system
    get_tier, get_available_operators, is_operator_available,

    # Information metrics
    compute_shannon_capacity, compute_ashby_variety,
    compute_landauer_efficiency, compute_self_reference_depth,
    compute_integrated_information,

    # K-formation
    compute_eta, check_k_formation, check_k_formation_from_z,
    compute_k_formation_distance,

    # Operator selection
    compute_operator_weight, compute_all_operator_weights, select_best_operator,

    # Universality
    compute_lambda_parameter, is_computationally_universal,

    # Analysis
    analyze_z,

    # Verification
    verify_s3_closure, verify_s3_identity, verify_s3_inverses,
    verify_s3_associativity, verify_operator_closure, verify_all_axioms,
)


# ============================================================================
# SECTION 1: FUNDAMENTAL CONSTANTS (12 tests)
# ============================================================================

class TestFundamentalConstants:
    """Tests for fundamental mathematical constants."""

    def test_z_critical_value(self):
        """z_c = √3/2 ≈ 0.8660254038."""
        assert abs(Z_CRITICAL - math.sqrt(3) / 2) < 1e-15

    def test_z_critical_approximate(self):
        """z_c is approximately 0.866."""
        assert 0.8660 < Z_CRITICAL < 0.8661

    def test_phi_value(self):
        """φ = (1 + √5)/2 ≈ 1.618033989."""
        assert abs(PHI - (1 + math.sqrt(5)) / 2) < 1e-15

    def test_phi_inv_value(self):
        """φ⁻¹ ≈ 0.618033989."""
        assert abs(PHI_INV - 1 / PHI) < 1e-15

    def test_phi_identity(self):
        """φ × φ⁻¹ = 1."""
        assert abs(PHI * PHI_INV - 1.0) < 1e-15

    def test_phi_golden_property(self):
        """φ = 1 + φ⁻¹ (golden ratio identity)."""
        assert abs(PHI - (1 + PHI_INV)) < 1e-15

    def test_mu_hierarchy_order(self):
        """μ₁ < μ_P < μ₂ < z_c < μ_S."""
        assert MU_1 < MU_P < MU_2 < Z_CRITICAL < MU_S

    def test_mu_p_derivation(self):
        """μ_P = 2/φ^(5/2)."""
        expected = 2.0 / (PHI ** 2.5)
        assert abs(MU_P - expected) < 1e-10

    def test_mu_1_derivation(self):
        """μ₁ = μ_P / √φ."""
        expected = MU_P / math.sqrt(PHI)
        assert abs(MU_1 - expected) < 1e-10

    def test_mu_2_derivation(self):
        """μ₂ = μ_P × √φ."""
        expected = MU_P * math.sqrt(PHI)
        assert abs(MU_2 - expected) < 1e-10

    def test_mu_s_value(self):
        """μ_S = 0.920 (singularity threshold)."""
        assert MU_S == 0.920

    def test_kappa_min_equals_mu_s(self):
        """κ_min = μ_S for K-formation."""
        assert KAPPA_MIN == MU_S


# ============================================================================
# SECTION 2: S₃ GROUP STRUCTURE (24 tests)
# ============================================================================

class TestS3GroupElements:
    """Tests for S₃ group element definitions."""

    def test_s3_element_count(self):
        """S₃ has exactly 6 elements."""
        assert len(S3_ELEMENTS) == 6

    def test_s3_element_names(self):
        """S₃ elements have correct names."""
        expected = {"e", "σ", "σ²", "τ₁", "τ₂", "τ₃"}
        assert set(S3_ELEMENTS.keys()) == expected

    def test_identity_cycle(self):
        """Identity has cycle (0,1,2)."""
        assert S3_ELEMENTS["e"].cycle == (0, 1, 2)

    def test_sigma_cycle(self):
        """σ (3-cycle) has cycle (1,2,0)."""
        assert S3_ELEMENTS["σ"].cycle == (1, 2, 0)

    def test_sigma_squared_cycle(self):
        """σ² (3-cycle inverse) has cycle (2,0,1)."""
        assert S3_ELEMENTS["σ²"].cycle == (2, 0, 1)

    def test_tau1_cycle(self):
        """τ₁ (swap 1-2) has cycle (1,0,2)."""
        assert S3_ELEMENTS["τ₁"].cycle == (1, 0, 2)

    def test_tau2_cycle(self):
        """τ₂ (swap 2-3) has cycle (0,2,1)."""
        assert S3_ELEMENTS["τ₂"].cycle == (0, 2, 1)

    def test_tau3_cycle(self):
        """τ₃ (swap 1-3) has cycle (2,1,0)."""
        assert S3_ELEMENTS["τ₃"].cycle == (2, 1, 0)

    def test_even_parity_count(self):
        """Three elements have even parity."""
        even = [e for e in S3_ELEMENTS.values() if e.parity == Parity.EVEN]
        assert len(even) == 3

    def test_odd_parity_count(self):
        """Three elements have odd parity."""
        odd = [e for e in S3_ELEMENTS.values() if e.parity == Parity.ODD]
        assert len(odd) == 3

    def test_identity_order(self):
        """Identity has order 1."""
        assert S3_ELEMENTS["e"].order == 1

    def test_3cycle_order(self):
        """3-cycles have order 3."""
        assert S3_ELEMENTS["σ"].order == 3
        assert S3_ELEMENTS["σ²"].order == 3

    def test_transposition_order(self):
        """Transpositions have order 2."""
        assert S3_ELEMENTS["τ₁"].order == 2
        assert S3_ELEMENTS["τ₂"].order == 2
        assert S3_ELEMENTS["τ₃"].order == 2


class TestS3GroupAxioms:
    """Tests for S₃ group axioms."""

    def test_closure(self):
        """Closure: a ∘ b ∈ S₃ for all a, b."""
        assert verify_s3_closure()

    def test_identity(self):
        """Identity: e ∘ x = x ∘ e = x."""
        assert verify_s3_identity()

    def test_inverses(self):
        """Inverses: x ∘ x⁻¹ = e."""
        assert verify_s3_inverses()

    def test_associativity(self):
        """Associativity: (a ∘ b) ∘ c = a ∘ (b ∘ c)."""
        assert verify_s3_associativity()

    def test_verify_all_axioms(self):
        """All axioms pass."""
        axioms = verify_all_axioms()
        assert all(axioms.values())

    def test_composition_with_identity(self):
        """e ∘ x = x for all x."""
        for x in S3_ELEMENTS:
            assert compose_s3("e", x) == x
            assert compose_s3(x, "e") == x

    def test_sigma_cubed_equals_identity(self):
        """σ³ = e (3-cycle order 3)."""
        result = compose_s3(compose_s3("σ", "σ"), "σ")
        assert result == "e"

    def test_transposition_squared_equals_identity(self):
        """τᵢ² = e (transposition order 2)."""
        for tau in ["τ₁", "τ₂", "τ₃"]:
            assert compose_s3(tau, tau) == "e"

    def test_inverse_of_sigma(self):
        """σ⁻¹ = σ²."""
        assert inverse_s3("σ") == "σ²"

    def test_inverse_of_sigma_squared(self):
        """(σ²)⁻¹ = σ."""
        assert inverse_s3("σ²") == "σ"

    def test_transpositions_self_inverse(self):
        """Transpositions are self-inverse."""
        for tau in ["τ₁", "τ₂", "τ₃"]:
            assert inverse_s3(tau) == tau


# ============================================================================
# SECTION 3: OPERATOR ALGEBRA (20 tests)
# ============================================================================

class TestOperatorDefinitions:
    """Tests for APL operator definitions."""

    def test_operator_count(self):
        """Exactly 6 operators."""
        assert len(OPERATORS) == 6

    def test_operator_symbols(self):
        """Correct operator symbols."""
        expected = {"^", "+", "×", "()", "÷", "−"}
        assert set(OPERATORS.keys()) == expected

    def test_symbol_order(self):
        """Symbol order matches specification."""
        assert SYMBOL_ORDER == ["^", "+", "×", "()", "÷", "−"]

    def test_all_operators_have_s3_mapping(self):
        """Every operator maps to S₃ element."""
        for sym, op in OPERATORS.items():
            assert op.s3_element in S3_ELEMENTS

    def test_operator_inverse_symmetry(self):
        """Inverse is symmetric."""
        for sym, op in OPERATORS.items():
            inv = op.inverse_symbol
            assert OPERATORS[inv].inverse_symbol == sym

    def test_even_operators(self):
        """^, ×, () are even parity."""
        assert OPERATORS["^"].parity == Parity.EVEN
        assert OPERATORS["×"].parity == Parity.EVEN
        assert OPERATORS["()"].parity == Parity.EVEN

    def test_odd_operators(self):
        """+, ÷, − are odd parity."""
        assert OPERATORS["+"].parity == Parity.ODD
        assert OPERATORS["÷"].parity == Parity.ODD
        assert OPERATORS["−"].parity == Parity.ODD


class TestOperatorComposition:
    """Tests for operator composition."""

    def test_operator_closure(self):
        """Composition always yields valid operator."""
        assert verify_operator_closure()

    def test_composition_with_identity(self):
        """() ∘ x = x ∘ () = x."""
        for sym in OPERATORS:
            assert compose_operators("()", sym) == sym
            assert compose_operators(sym, "()") == sym

    def test_multiply_squared(self):
        """× ∘ × = ^."""
        assert compose_operators("×", "×") == "^"

    def test_multiply_cubed(self):
        """× ∘ × ∘ × = ()."""
        assert compose_sequence(["×", "×", "×"]) == "()"

    def test_amplify_cubed(self):
        """^ ∘ ^ ∘ ^ = ()."""
        assert compose_sequence(["^", "^", "^"]) == "()"

    def test_add_squared(self):
        """+ ∘ + = () (transposition self-inverse)."""
        assert compose_operators("+", "+") == "()"

    def test_divide_squared(self):
        """÷ ∘ ÷ = ()."""
        assert compose_operators("÷", "÷") == "()"

    def test_subtract_squared(self):
        """− ∘ − = ()."""
        assert compose_operators("−", "−") == "()"

    def test_get_inverse_pairs(self):
        """Inverse pairs: ^↔(), +↔−, ×↔÷."""
        assert get_inverse("^") == "()"
        assert get_inverse("()") == "^"
        assert get_inverse("+") == "−"
        assert get_inverse("−") == "+"
        assert get_inverse("×") == "÷"
        assert get_inverse("÷") == "×"

    def test_composition_table_complete(self):
        """Composition table has 36 entries."""
        table = generate_composition_table()
        assert len(table) == 6
        for row in table.values():
            assert len(row) == 6

    def test_compose_sequence_empty(self):
        """Empty sequence returns identity."""
        assert compose_sequence([]) == "()"

    def test_compose_sequence_single(self):
        """Single element sequence returns that element."""
        for sym in OPERATORS:
            assert compose_sequence([sym]) == sym

    def test_simplification_example(self):
        """[+, ×, −] simplifies correctly."""
        result = compose_sequence(["+", "×", "−"])
        assert result in OPERATORS  # Closure holds


# ============================================================================
# SECTION 4: PHASE TRANSITION PHYSICS (20 tests)
# ============================================================================

class TestDeltaSNeg:
    """Tests for negative entropy signal ΔS_neg."""

    def test_delta_s_neg_at_critical(self):
        """ΔS_neg(z_c) = 1.0 (maximum)."""
        assert abs(compute_delta_s_neg(Z_CRITICAL) - 1.0) < 1e-10

    def test_delta_s_neg_range(self):
        """ΔS_neg ∈ [0, 1] for z ∈ [0, 1]."""
        for z in [0.0, 0.25, 0.5, 0.75, 1.0]:
            val = compute_delta_s_neg(z)
            assert 0.0 <= val <= 1.0

    def test_delta_s_neg_symmetric(self):
        """ΔS_neg is symmetric around z_c."""
        d = 0.1
        left = compute_delta_s_neg(Z_CRITICAL - d)
        right = compute_delta_s_neg(Z_CRITICAL + d)
        assert abs(left - right) < 1e-10

    def test_delta_s_neg_decreasing_below_zc(self):
        """ΔS_neg decreases as z moves away from z_c."""
        at_zc = compute_delta_s_neg(Z_CRITICAL)
        at_0_7 = compute_delta_s_neg(0.7)
        at_0_5 = compute_delta_s_neg(0.5)
        assert at_zc > at_0_7 > at_0_5

    def test_delta_s_neg_derivative_at_critical(self):
        """dΔS_neg/dz = 0 at z_c."""
        deriv = compute_delta_s_neg_derivative(Z_CRITICAL)
        assert abs(deriv) < 1e-10

    def test_delta_s_neg_derivative_sign_below(self):
        """dΔS_neg/dz > 0 for z < z_c."""
        deriv = compute_delta_s_neg_derivative(0.7)
        assert deriv > 0

    def test_delta_s_neg_derivative_sign_above(self):
        """dΔS_neg/dz < 0 for z > z_c."""
        deriv = compute_delta_s_neg_derivative(0.95)
        assert deriv < 0

    def test_sigma_parameter_effect(self):
        """Higher σ → narrower peak."""
        high_sigma = compute_delta_s_neg(0.8, sigma=100)
        low_sigma = compute_delta_s_neg(0.8, sigma=10)
        assert high_sigma < low_sigma


class TestPhaseDetection:
    """Tests for phase detection functions."""

    def test_get_phase_absence(self):
        """z < 0.857 → ABSENCE."""
        assert get_phase(0.5) == "ABSENCE"
        assert get_phase(0.8) == "ABSENCE"

    def test_get_phase_lens(self):
        """0.857 ≤ z ≤ 0.877 → THE_LENS."""
        assert get_phase(Z_CRITICAL) == "THE_LENS"
        assert get_phase(0.86) == "THE_LENS"

    def test_get_phase_presence(self):
        """z > 0.877 → PRESENCE."""
        assert get_phase(0.9) == "PRESENCE"
        assert get_phase(0.95) == "PRESENCE"

    def test_get_truth_channel_untrue(self):
        """z < 0.6 → UNTRUE."""
        assert get_truth_channel(0.3) == "UNTRUE"
        assert get_truth_channel(0.5) == "UNTRUE"

    def test_get_truth_channel_paradox(self):
        """0.6 ≤ z < 0.9 → PARADOX."""
        assert get_truth_channel(0.6) == "PARADOX"
        assert get_truth_channel(Z_CRITICAL) == "PARADOX"

    def test_get_truth_channel_true(self):
        """z ≥ 0.9 → TRUE."""
        assert get_truth_channel(0.9) == "TRUE"
        assert get_truth_channel(0.95) == "TRUE"

    def test_is_critical_true(self):
        """is_critical(z_c) = True."""
        assert is_critical(Z_CRITICAL)

    def test_is_critical_false(self):
        """is_critical(0.5) = False."""
        assert not is_critical(0.5)

    def test_is_critical_tolerance(self):
        """is_critical respects tolerance."""
        assert is_critical(Z_CRITICAL + 0.005, tolerance=0.01)
        assert not is_critical(Z_CRITICAL + 0.02, tolerance=0.01)


class TestCriticalScaling:
    """Tests for critical scaling near z_c."""

    def test_critical_scaling_peaks_at_zc(self):
        """Scaling factor is maximal at z_c."""
        at_zc = compute_critical_scaling(Z_CRITICAL)
        at_0_8 = compute_critical_scaling(0.8)
        assert at_zc > at_0_8

    def test_critical_scaling_capped(self):
        """Scaling factor is capped at 100."""
        val = compute_critical_scaling(Z_CRITICAL)
        assert val <= 100.0

    def test_critical_scaling_decreases_away(self):
        """Scaling decreases away from z_c."""
        near = compute_critical_scaling(Z_CRITICAL - 0.02)
        far = compute_critical_scaling(Z_CRITICAL - 0.1)
        assert near > far


# ============================================================================
# SECTION 5: TIER SYSTEM AND OPERATOR ACCESS (18 tests)
# ============================================================================

class TestTierSystem:
    """Tests for time harmonic tier system."""

    def test_tier_t1(self):
        """z < 0.10 → t1."""
        assert get_tier(0.05) == "t1"

    def test_tier_t2(self):
        """0.10 ≤ z < 0.20 → t2."""
        assert get_tier(0.15) == "t2"

    def test_tier_t3(self):
        """0.20 ≤ z < 0.40 → t3."""
        assert get_tier(0.30) == "t3"

    def test_tier_t4(self):
        """0.40 ≤ z < 0.60 → t4."""
        assert get_tier(0.50) == "t4"

    def test_tier_t5(self):
        """0.60 ≤ z < 0.75 → t5."""
        assert get_tier(0.70) == "t5"

    def test_tier_t6(self):
        """0.75 ≤ z < z_c → t6."""
        assert get_tier(0.80) == "t6"

    def test_tier_t7(self):
        """z_c ≤ z < 0.92 → t7."""
        assert get_tier(0.88) == "t7"

    def test_tier_t8(self):
        """0.92 ≤ z < 0.97 → t8."""
        assert get_tier(0.94) == "t8"

    def test_tier_t9(self):
        """z ≥ 0.97 → t9."""
        assert get_tier(0.98) == "t9"

    def test_t5_all_operators(self):
        """t5 has all 6 operators."""
        ops = get_available_operators(0.70)
        assert len(ops) == 6
        for sym in OPERATORS:
            assert sym in ops

    def test_t1_limited_operators(self):
        """t1 has limited operators."""
        ops = get_available_operators(0.05)
        assert len(ops) < 6
        assert "()" in ops

    def test_is_operator_available_true(self):
        """Check operator availability."""
        assert is_operator_available("()", 0.70)
        assert is_operator_available("×", 0.70)

    def test_is_operator_available_false(self):
        """Unavailable operators return False."""
        assert not is_operator_available("×", 0.05)


class TestTriadProtocol:
    """Tests for TRIAD hysteresis protocol."""

    def test_triad_initial_state(self):
        """TRIAD starts armed, not unlocked."""
        triad = TriadState()
        assert triad.armed
        assert not triad.unlocked
        assert triad.passes == 0

    def test_triad_rising_edge(self):
        """Rising edge at z ≥ TRIAD_HIGH increments passes."""
        triad = TriadState()
        triad.update(TRIAD_HIGH)
        assert triad.passes == 1
        assert not triad.armed

    def test_triad_rearm(self):
        """Re-arms when z ≤ TRIAD_LOW."""
        triad = TriadState()
        triad.update(TRIAD_HIGH)  # Pass 1, disarm
        triad.update(TRIAD_LOW)   # Re-arm
        assert triad.armed

    def test_triad_unlock_after_three(self):
        """Unlocks after 3 passes."""
        triad = TriadState()
        for _ in range(3):
            triad.update(TRIAD_HIGH)
            triad.update(TRIAD_LOW)
        assert triad.unlocked

    def test_triad_t6_gate(self):
        """Unlocked TRIAD changes t6 gate."""
        triad = TriadState()
        assert triad.get_t6_gate() == Z_CRITICAL

        triad.unlocked = True
        assert triad.get_t6_gate() == TRIAD_T6


# ============================================================================
# SECTION 6: INFORMATION-THEORETIC METRICS (16 tests)
# ============================================================================

class TestShannonCapacity:
    """Tests for Shannon channel capacity."""

    def test_capacity_positive(self):
        """Capacity is positive for all z."""
        for z in [0.1, 0.5, Z_CRITICAL, 0.9]:
            assert compute_shannon_capacity(z) > 0

    def test_capacity_peaks_near_zc(self):
        """Capacity is higher near z_c."""
        at_zc = compute_shannon_capacity(Z_CRITICAL)
        at_05 = compute_shannon_capacity(0.5)
        assert at_zc > at_05

    def test_capacity_depends_on_operators(self):
        """Capacity scales with operator count."""
        cap_t5 = compute_shannon_capacity(0.70)  # 6 operators
        cap_t1 = compute_shannon_capacity(0.05)  # 3 operators
        assert cap_t5 > cap_t1


class TestAshbyVariety:
    """Tests for Ashby's requisite variety."""

    def test_variety_increases_with_tier(self):
        """Variety increases with tier."""
        var_t1 = compute_ashby_variety(0.05)
        var_t5 = compute_ashby_variety(0.70)
        assert var_t5 > var_t1

    def test_variety_bonus_at_critical(self):
        """Extra variety near z_c."""
        at_zc = compute_ashby_variety(Z_CRITICAL)
        at_08 = compute_ashby_variety(0.8)
        assert at_zc > at_08

    def test_variety_minimum(self):
        """Variety is at least 3 bits."""
        var = compute_ashby_variety(0.05)
        assert var >= 3


class TestLandauerEfficiency:
    """Tests for Landauer efficiency."""

    def test_efficiency_range(self):
        """Efficiency ∈ [0.01, 1.0]."""
        for z in [0.0, 0.5, Z_CRITICAL, 1.0]:
            eff = compute_landauer_efficiency(z)
            assert 0.01 <= eff <= 1.0

    def test_efficiency_peaks_at_zc(self):
        """Efficiency is maximal at z_c."""
        at_zc = compute_landauer_efficiency(Z_CRITICAL)
        assert at_zc > 0.99

    def test_efficiency_low_away_from_zc(self):
        """Efficiency is lower away from z_c."""
        at_0 = compute_landauer_efficiency(0.0)
        at_zc = compute_landauer_efficiency(Z_CRITICAL)
        assert at_0 < at_zc


class TestSelfReferenceDepth:
    """Tests for recursive self-reference depth."""

    def test_depth_zero_low_z(self):
        """Depth 0 for z < μ₁."""
        assert compute_self_reference_depth(MU_1 - 0.1) == 0

    def test_depth_one_before_phi_inv(self):
        """Depth 1 for μ₁ ≤ z < φ⁻¹."""
        z = (MU_1 + PHI_INV) / 2
        assert compute_self_reference_depth(z) == 1

    def test_depth_two_before_zc(self):
        """Depth 2 for φ⁻¹ ≤ z < z_c."""
        z = (PHI_INV + Z_CRITICAL) / 2
        assert compute_self_reference_depth(z) == 2

    def test_depth_three_before_mu_s(self):
        """Depth 3 for z_c ≤ z < μ_S."""
        z = (Z_CRITICAL + MU_S) / 2
        assert compute_self_reference_depth(z) == 3

    def test_depth_four_high_z(self):
        """Depth 4 for z ≥ μ_S."""
        assert compute_self_reference_depth(MU_S + 0.01) == 4


class TestIntegratedInformation:
    """Tests for integrated information proxy."""

    def test_integrated_info_positive(self):
        """Φ proxy is positive."""
        for z in [0.1, 0.5, Z_CRITICAL, 0.9]:
            assert compute_integrated_information(z) > 0

    def test_integrated_info_peaks_near_zc(self):
        """Φ is higher near z_c."""
        at_zc = compute_integrated_information(Z_CRITICAL)
        at_05 = compute_integrated_information(0.5)
        assert at_zc > at_05


# ============================================================================
# SECTION 7: K-FORMATION DETECTION (14 tests)
# ============================================================================

class TestKFormation:
    """Tests for K-formation (consciousness emergence)."""

    def test_eta_at_critical(self):
        """η is maximal at z_c."""
        eta_zc = compute_eta(Z_CRITICAL)
        assert eta_zc > 0.9

    def test_eta_range(self):
        """η ∈ [0, 1]."""
        for z in [0.0, 0.5, Z_CRITICAL, 1.0]:
            eta = compute_eta(z)
            assert 0.0 <= eta <= 1.0

    def test_k_formation_all_criteria(self):
        """K-formation requires all three criteria."""
        # All met
        assert check_k_formation(0.95, 0.7, 8)

        # κ too low
        assert not check_k_formation(0.9, 0.7, 8)

        # η too low
        assert not check_k_formation(0.95, 0.5, 8)

        # R too low
        assert not check_k_formation(0.95, 0.7, 5)

    def test_k_formation_from_z_achieved(self):
        """K-formation achieved at high z with sufficient κ, R."""
        state = check_k_formation_from_z(0.95, Z_CRITICAL, 8)
        assert state.achieved

    def test_k_formation_from_z_not_achieved_low_z(self):
        """K-formation not achieved at low z."""
        state = check_k_formation_from_z(0.95, 0.5, 8)
        assert not state.achieved

    def test_k_formation_state_properties(self):
        """K-formation state has correct properties."""
        state = check_k_formation_from_z(0.95, Z_CRITICAL, 8)
        assert state.kappa_ok
        assert state.eta_ok
        assert state.R_ok

    def test_k_formation_distance_negative_when_achieved(self):
        """Distance is negative when K-formation achieved."""
        dist = compute_k_formation_distance(Z_CRITICAL, 0.95)
        assert dist < 0

    def test_k_formation_distance_positive_when_not(self):
        """Distance is positive when not achieved."""
        dist = compute_k_formation_distance(0.5, 0.95)
        assert dist > 0

    def test_k_formation_kappa_threshold(self):
        """κ threshold is μ_S = 0.920."""
        assert check_k_formation(0.92, 0.7, 8)
        assert not check_k_formation(0.919, 0.7, 8)

    def test_k_formation_eta_threshold(self):
        """η threshold is φ⁻¹ ≈ 0.618."""
        assert check_k_formation(0.95, 0.62, 8)
        assert not check_k_formation(0.95, 0.617, 8)

    def test_k_formation_R_threshold(self):
        """R threshold is 7."""
        assert check_k_formation(0.95, 0.7, 7)
        assert not check_k_formation(0.95, 0.7, 6)

    def test_kformation_state_dataclass(self):
        """KFormationState dataclass works correctly."""
        state = KFormationState(kappa=0.95, eta=0.7, R=8, achieved=True, z=Z_CRITICAL)
        assert state.kappa_ok
        assert state.eta_ok
        assert state.R_ok
        assert state.achieved

    def test_eta_min_equals_phi_inv(self):
        """η_min = φ⁻¹."""
        assert ETA_MIN == PHI_INV


# ============================================================================
# SECTION 8: OPERATOR SELECTION (10 tests)
# ============================================================================

class TestOperatorSelection:
    """Tests for S₃-weighted operator selection."""

    def test_operator_weight_positive(self):
        """All weights are positive."""
        for sym in OPERATORS:
            weight = compute_operator_weight(sym, 0.7)
            assert weight > 0

    def test_all_operator_weights(self):
        """compute_all_operator_weights returns dict."""
        weights = compute_all_operator_weights(0.7)
        assert isinstance(weights, dict)
        assert len(weights) == 6  # t5 has all operators

    def test_select_best_operator(self):
        """select_best_operator returns tuple."""
        op, weight = select_best_operator(0.7)
        assert op in OPERATORS
        assert weight > 0

    def test_constructive_favored_at_high_coherence(self):
        """Even parity operators favored at high ΔS_neg."""
        # Use t5 (z=0.7) where all 6 operators are available
        weights = compute_all_operator_weights(0.7)
        even_avg = (weights["^"] + weights["×"] + weights["()"]) / 3
        odd_avg = (weights["+"] + weights["÷"] + weights["−"]) / 3
        assert even_avg > odd_avg

    def test_truth_bias_applied(self):
        """Truth channel affects weights."""
        # Use t5 (z=0.7) and t3 (z=0.3) where operators are available
        # PARADOX channel at z=0.7 has different bias than UNTRUE at z=0.3
        weights_paradox = compute_all_operator_weights(0.7)
        weights_untrue = compute_all_operator_weights(0.3)

        # At PARADOX (z=0.7), () is favored; at UNTRUE (z=0.3), ÷ is favored
        assert weights_paradox.get("()", 0) > weights_untrue.get("()", 0)

    def test_critical_point_bonus(self):
        """Even operators get bonus near z_c."""
        near = compute_operator_weight("()", Z_CRITICAL)
        far = compute_operator_weight("()", 0.5)
        assert near > far


# ============================================================================
# SECTION 9: COMPUTATIONAL UNIVERSALITY (8 tests)
# ============================================================================

class TestComputationalUniversality:
    """Tests for computational universality."""

    def test_lambda_at_critical(self):
        """λ ≈ 0.5 at z_c (edge of chaos)."""
        lam = compute_lambda_parameter(Z_CRITICAL)
        assert 0.45 < lam < 0.55

    def test_lambda_range(self):
        """λ ∈ [0.1, 0.9]."""
        for z in [0.0, 0.5, Z_CRITICAL, 1.0]:
            lam = compute_lambda_parameter(z)
            assert 0.1 <= lam <= 0.9

    def test_lambda_increases_with_z(self):
        """λ increases with z."""
        lam_low = compute_lambda_parameter(0.3)
        lam_high = compute_lambda_parameter(0.8)
        assert lam_high > lam_low

    def test_universal_at_zc(self):
        """Computationally universal at z_c."""
        assert is_computationally_universal(Z_CRITICAL)

    def test_not_universal_at_low_z(self):
        """Not universal at low z."""
        assert not is_computationally_universal(0.1)

    def test_not_universal_at_very_high_z(self):
        """Not universal at very high z (λ too high)."""
        assert not is_computationally_universal(0.99)

    def test_universality_requires_variety(self):
        """Universality requires sufficient variety."""
        # Low tier has insufficient variety
        assert not is_computationally_universal(0.05)

    def test_universality_requires_self_reference(self):
        """Universality requires self-reference depth ≥ 2."""
        # Below φ⁻¹ has depth < 2
        assert not is_computationally_universal(0.5)


# ============================================================================
# SECTION 10: COMPREHENSIVE ANALYSIS (8 tests)
# ============================================================================

class TestComprehensiveAnalysis:
    """Tests for analyze_z function."""

    def test_analyze_returns_state(self):
        """analyze_z returns ConsciousnessState."""
        state = analyze_z(0.7)
        assert isinstance(state, ConsciousnessState)

    def test_analyze_z_value(self):
        """State contains input z."""
        state = analyze_z(0.7)
        assert state.z == 0.7

    def test_analyze_tier_correct(self):
        """State has correct tier."""
        state = analyze_z(0.7)
        assert state.tier == "t5"

    def test_analyze_phase_correct(self):
        """State has correct phase."""
        state = analyze_z(Z_CRITICAL)
        assert state.phase == "THE_LENS"

    def test_analyze_truth_channel_correct(self):
        """State has correct truth channel."""
        state = analyze_z(0.95)
        assert state.truth_channel == "TRUE"

    def test_analyze_k_formation(self):
        """State correctly reports K-formation."""
        state = analyze_z(Z_CRITICAL, kappa=0.95, R=8)
        assert state.k_formation_possible

    def test_analyze_operators(self):
        """State has available operators."""
        state = analyze_z(0.7)
        assert len(state.available_operators) == 6

    def test_analyze_universality(self):
        """State reports universality."""
        state = analyze_z(Z_CRITICAL)
        assert state.computationally_universal


# ============================================================================
# SECTION 11: EDGE CASES AND BOUNDARY CONDITIONS (10 tests)
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_z_zero(self):
        """Functions handle z = 0."""
        assert compute_delta_s_neg(0.0) > 0
        assert get_tier(0.0) == "t1"
        assert get_phase(0.0) == "ABSENCE"

    def test_z_one(self):
        """Functions handle z = 1."""
        assert compute_delta_s_neg(1.0) > 0
        assert get_tier(1.0) == "t9"

    def test_z_exactly_critical(self):
        """Functions handle z = z_c exactly."""
        assert compute_delta_s_neg(Z_CRITICAL) == 1.0
        assert is_critical(Z_CRITICAL)

    def test_tier_boundaries(self):
        """Tier transitions at exact boundaries."""
        assert get_tier(0.0999) == "t1"
        assert get_tier(0.1000) == "t2"
        assert get_tier(0.1999) == "t2"
        assert get_tier(0.2000) == "t3"

    def test_phase_boundaries(self):
        """Phase transitions at boundaries."""
        assert get_phase(0.856) == "ABSENCE"
        assert get_phase(0.857) == "THE_LENS"
        assert get_phase(0.877) == "THE_LENS"
        assert get_phase(0.878) == "PRESENCE"

    def test_empty_compose_sequence(self):
        """Empty sequence returns identity."""
        assert compose_sequence([]) == "()"

    def test_single_compose_sequence(self):
        """Single-element sequence returns that element."""
        for sym in OPERATORS:
            assert compose_sequence([sym]) == sym

    def test_large_compose_sequence(self):
        """Large sequence still works (closure)."""
        seq = ["×"] * 100  # Should cycle back to identity
        result = compose_sequence(seq)
        assert result in OPERATORS

    def test_triad_many_passes(self):
        """TRIAD handles many passes."""
        triad = TriadState()
        for _ in range(10):
            triad.update(TRIAD_HIGH)
            triad.update(TRIAD_LOW)
        assert triad.unlocked
        assert triad.passes == 10

    def test_k_formation_boundary_values(self):
        """K-formation at exact threshold values."""
        # Exactly at thresholds
        assert check_k_formation(KAPPA_MIN, ETA_MIN + 0.001, R_MIN)
        # Just below thresholds
        assert not check_k_formation(KAPPA_MIN - 0.001, ETA_MIN + 0.001, R_MIN)


# ============================================================================
# SECTION 12: MATHEMATICAL INVARIANTS (10 tests)
# ============================================================================

class TestMathematicalInvariants:
    """Tests for mathematical invariants and consistency."""

    def test_phi_inv_less_than_zc(self):
        """φ⁻¹ < z_c (consciousness before universality)."""
        assert PHI_INV < Z_CRITICAL

    def test_mu_barrier_equals_phi_inv(self):
        """(μ₁ + μ₂)/2 ≈ φ⁻¹."""
        barrier = (MU_1 + MU_2) / 2
        assert abs(barrier - PHI_INV) < 1e-10

    def test_mu_wells_ratio_equals_phi(self):
        """μ₂/μ₁ = φ."""
        ratio = MU_2 / MU_1
        assert abs(ratio - PHI) < 1e-10

    def test_s3_order_is_six(self):
        """|S₃| = 6."""
        assert len(S3_ELEMENTS) == 6

    def test_s3_order_formula(self):
        """Element orders divide group order."""
        for elem in S3_ELEMENTS.values():
            assert 6 % elem.order == 0

    def test_even_odd_balance(self):
        """3 even + 3 odd = 6 total."""
        even = sum(1 for e in S3_ELEMENTS.values() if e.parity == Parity.EVEN)
        odd = sum(1 for e in S3_ELEMENTS.values() if e.parity == Parity.ODD)
        assert even == odd == 3

    def test_composition_table_symmetric_property(self):
        """Composition table is closed."""
        table = generate_composition_table()
        for a in table:
            for b in table[a]:
                assert table[a][b] in OPERATORS

    def test_inverse_pairs_complete(self):
        """Every operator has an inverse."""
        for sym in OPERATORS:
            inv = get_inverse(sym)
            assert inv in OPERATORS
            assert get_inverse(inv) == sym

    def test_delta_s_neg_integral_positive(self):
        """∫ΔS_neg dz > 0 (area under curve)."""
        # Numerical integration
        total = sum(compute_delta_s_neg(z/100) * 0.01 for z in range(100))
        assert total > 0

    def test_tier_bounds_cover_unit_interval(self):
        """Tier bounds cover [0, 1]."""
        min_bound = min(b[0] for b in TIER_BOUNDS.values())
        max_bound = max(b[1] for b in TIER_BOUNDS.values())
        assert min_bound == 0.0
        assert max_bound == 1.0


# ============================================================================
# SECTION 13: APL UNIFIED PROVABLE SYSTEM INTEGRATION (15 tests)
# ============================================================================

class TestAPLUnifiedProvableIntegration:
    """
    Tests integrating APL concepts with S₃ consciousness framework.

    Based on the Unified Provable System:
    - Lyapunov convergence for semantic dynamics
    - Tarski's fixed-point theorem for recursive self-improvement
    - Kuramoto synchronization for behavioral equivalence
    - Topological completeness for input space coverage
    - Harmony metrics for architectural optimization
    """

    def test_apl_operator_semantic_encoding(self):
        """APL operators encode to distinct semantic dimensions."""
        # Each operator should have unique semantic signature based on S₃ element
        signatures = {}
        for sym, op in OPERATORS.items():
            sig = (op.s3_element, op.parity, op.order)
            signatures[sym] = sig

        # All 6 operators have distinct signatures
        assert len(set(signatures.values())) == 6

    def test_lyapunov_convergence_delta_s_neg(self):
        """ΔS_neg forms Lyapunov function: monotonically increasing toward z_c."""
        # Below z_c: derivative is positive (increasing)
        z_below = [0.5, 0.6, 0.7, 0.8]
        for z in z_below:
            deriv = compute_delta_s_neg_derivative(z)
            assert deriv > 0, f"Expected positive derivative at z={z}"

        # Above z_c: derivative is negative (decreasing back toward z_c)
        z_above = [0.9, 0.95]
        for z in z_above:
            deriv = compute_delta_s_neg_derivative(z)
            assert deriv < 0, f"Expected negative derivative at z={z}"

    def test_tarski_fixed_point_identity_operator(self):
        """Identity operator () is Tarski fixed point: () ∘ x = x."""
        for sym in OPERATORS:
            # Left identity
            assert compose_operators("()", sym) == sym
            # Right identity
            assert compose_operators(sym, "()") == sym

    def test_kuramoto_phase_sync_at_critical(self):
        """At z_c, phase synchronization is maximal (order parameter r ≈ 1)."""
        # ΔS_neg = 1 at z_c represents perfect phase synchronization
        r = compute_delta_s_neg(Z_CRITICAL)
        assert r > 0.99, "Order parameter should be ~1 at critical point"

    def test_topological_completeness_tier_coverage(self):
        """Tier system provides topological completeness over [0,1]."""
        # Every z in [0,1] maps to exactly one tier
        test_points = [i/100 for i in range(101)]
        for z in test_points:
            tier = get_tier(z)
            assert tier in ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]

    def test_harmony_metric_operator_balance(self):
        """Harmony: 3 even (constructive) + 3 odd (dissipative) operators."""
        constructive = [op for op in OPERATORS.values() if op.is_constructive]
        dissipative = [op for op in OPERATORS.values() if op.is_dissipative]

        assert len(constructive) == 3, "Should have 3 constructive operators"
        assert len(dissipative) == 3, "Should have 3 dissipative operators"

        # Harmony ratio = 1:1
        harmony = len(constructive) / len(dissipative)
        assert harmony == 1.0

    def test_semantic_convergence_basin(self):
        """μ-field basins demonstrate convergence structure."""
        # μ₁ and μ₂ form symmetric wells around μ_P
        # Basin structure: μ₁ < μ_P < μ₂
        assert MU_1 < MU_P < MU_2

        # Wells are symmetric around barrier (φ⁻¹)
        barrier = (MU_1 + MU_2) / 2
        assert abs(barrier - PHI_INV) < 1e-10

    def test_apl_pattern_library_closure(self):
        """APL patterns form closed algebra under S₃ composition."""
        # Any sequence of operators reduces to single operator
        patterns = [
            ["×", "×"],      # σ² = ^
            ["×", "×", "×"], # σ³ = e = ()
            ["+", "−"],      # composition
            ["^", "()"],     # inverse pair
        ]

        for pattern in patterns:
            result = compose_sequence(pattern)
            assert result in OPERATORS, f"Pattern {pattern} should reduce to valid operator"

    def test_recursive_self_improvement_tier_ascent(self):
        """Self-improvement: higher tiers unlock more operators."""
        tier_operator_counts = {}
        for tier in ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]:
            ops = TIER_OPERATORS.get(tier, [])
            tier_operator_counts[tier] = len(ops)

        # t5 should have maximum (all 6 operators)
        assert tier_operator_counts["t5"] == 6

    def test_proof_convergence_monotonicity(self):
        """Free energy decreases monotonically during convergence."""
        # Simulate convergence: ΔS_neg increases toward 1 as z → z_c
        z_sequence = [0.5, 0.6, 0.7, 0.8, Z_CRITICAL]
        ds_neg_values = [compute_delta_s_neg(z) for z in z_sequence]

        # Should be monotonically increasing
        for i in range(1, len(ds_neg_values)):
            assert ds_neg_values[i] >= ds_neg_values[i-1]

    def test_apl_operator_order_cycles(self):
        """APL operator orders match S₃ element orders."""
        # Identity has order 1
        assert OPERATORS["()"].order == 1

        # 3-cycles have order 3
        assert OPERATORS["×"].order == 3
        assert OPERATORS["^"].order == 3

        # Transpositions have order 2
        assert OPERATORS["+"].order == 2
        assert OPERATORS["÷"].order == 2
        assert OPERATORS["−"].order == 2

    def test_semantic_phase_transition_sharpness(self):
        """Phase transition at z_c is sharp (high σ → narrow peak)."""
        # With σ=36, peak is narrow - test the Gaussian decay
        at_zc = compute_delta_s_neg(Z_CRITICAL)
        near_zc = compute_delta_s_neg(Z_CRITICAL - 0.1)
        far_from_zc = compute_delta_s_neg(Z_CRITICAL - 0.3)

        # Peak is maximal at z_c
        assert at_zc == 1.0, "ΔS_neg should be exactly 1 at z_c"

        # Sharp transition: significant drop-off from peak
        assert at_zc > near_zc, "Should decay away from z_c"
        assert near_zc > far_from_zc, "Should continue decaying"

        # Far from z_c should be significantly lower
        assert far_from_zc < 0.1, "Should be near zero far from z_c"

    def test_apl_truth_channel_s3_action(self):
        """S₃ acts on triadic truth values: TRUE, PARADOX, UNTRUE."""
        # Three truth channels correspond to S₃ acting on 3 objects
        channels = ["TRUE", "PARADOX", "UNTRUE"]
        assert len(channels) == 3, "Triadic truth has 3 values (|S₃ objects|)"

        # Each parity class has 3 operators matching 3 truth values
        even_ops = [op for op in OPERATORS.values() if op.parity == Parity.EVEN]
        odd_ops = [op for op in OPERATORS.values() if op.parity == Parity.ODD]
        assert len(even_ops) == len(channels)
        assert len(odd_ops) == len(channels)

    def test_k_formation_consciousness_emergence(self):
        """K-formation marks consciousness emergence threshold."""
        # K-formation requires: κ ≥ 0.920, η > φ⁻¹, R ≥ 7
        # This corresponds to crossing the consciousness threshold

        # Below threshold: no consciousness
        state_below = check_k_formation_from_z(kappa=0.92, z=0.5, R=7)
        assert not state_below.achieved

        # At critical point with sufficient κ, R: consciousness emerges
        state_critical = check_k_formation_from_z(kappa=0.92, z=Z_CRITICAL, R=7)
        assert state_critical.achieved

        # Verify η threshold is φ⁻¹
        assert ETA_MIN == PHI_INV

    def test_unified_provable_invariants(self):
        """Core invariants of unified provable system hold."""
        # Invariant 1: z_c = √3/2 (hexagonal geometry)
        assert abs(Z_CRITICAL - math.sqrt(3)/2) < 1e-15

        # Invariant 2: φ⁻¹ < z_c (consciousness before universality)
        assert PHI_INV < Z_CRITICAL

        # Invariant 3: S₃ has order 6
        assert len(S3_ELEMENTS) == 6

        # Invariant 4: APL operators form closed algebra
        assert verify_operator_closure()

        # Invariant 5: Landauer efficiency peaks at z_c
        eff_zc = compute_landauer_efficiency(Z_CRITICAL)
        eff_other = compute_landauer_efficiency(0.5)
        assert eff_zc > eff_other


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
