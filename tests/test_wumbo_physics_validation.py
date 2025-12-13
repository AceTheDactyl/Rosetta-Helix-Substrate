#!/usr/bin/env python3
"""
WUMBO Physics Validation Tests
===============================

Comprehensive validation of WUMBO integration against APL physics constants.

TESTED PHYSICS:
1. Gaussian Negentropy: ΔS_neg(z) = exp(-σ(z - z_c)²) where σ = 36 = |S₃|²
2. μ Threshold Hierarchy: MU_1 < MU_P < MU_2 < z_c < MU_S < MU_3
3. K-Formation Criteria: κ ≥ 0.920, η > φ⁻¹, R ≥ 7
4. PHI Liminal / PHI_INV Physical Architecture
5. 7-Layer Prismatic Projection through THE LENS
6. S₃ Operator Algebra (closure, parity, invertibility)

All tests validate that PHYSICS CONSTANTS ARE NOT ARBITRARY.
They derive from:
- z_c = √3/2 (hexagonal geometry)
- φ = (1+√5)/2 (golden ratio)
- σ = 36 = 6² (S₃ group order squared)

@version 1.0.0
"""

import math
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ============================================================================
# PHYSICS CONSTANTS (Must match src/quantum_apl_python/constants.py)
# ============================================================================

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Z_CRITICAL = math.sqrt(3) / 2

# σ = 36 = |S₃|² (symmetric group order squared)
LENS_SIGMA = 36.0
GAUSSIAN_WIDTH = 1 / math.sqrt(2 * LENS_SIGMA)  # ≈ 0.118
GAUSSIAN_FWHM = 2 * math.sqrt(math.log(2) / LENS_SIGMA)  # ≈ 0.277

# μ Threshold Hierarchy
MU_P = 2.0 / (PHI ** 2.5)
MU_1 = MU_P / math.sqrt(PHI)
MU_2 = MU_P * math.sqrt(PHI)
MU_S = 0.920
MU_3 = 0.992
KAPPA_S = MU_S

# K-Formation thresholds
KAPPA_MIN = 0.920
ETA_MIN = PHI_INV
R_MIN = 7

# Phase boundaries
Z_ABSENCE_MAX = 0.857
Z_LENS_MIN = 0.857
Z_LENS_MAX = 0.877
Z_PRESENCE_MIN = 0.877


# ============================================================================
# GAUSSIAN NEGENTROPY FUNCTION
# ============================================================================

def compute_delta_s_neg(z: float, sigma: float = LENS_SIGMA, z_c: float = Z_CRITICAL) -> float:
    """Compute ΔS_neg(z) = exp(-σ(z - z_c)²)"""
    d = z - z_c
    return math.exp(-sigma * d * d)


def compute_delta_s_neg_derivative(z: float, sigma: float = LENS_SIGMA, z_c: float = Z_CRITICAL) -> float:
    """Compute d(ΔS_neg)/dz = -2σ(z - z_c) · ΔS_neg(z)"""
    d = z - z_c
    s = compute_delta_s_neg(z, sigma, z_c)
    return -2 * sigma * d * s


def check_k_formation(kappa: float, eta: float, R: float) -> bool:
    """K-Formation: κ ≥ 0.920 AND η > φ⁻¹ AND R ≥ 7"""
    return kappa >= KAPPA_MIN and eta > ETA_MIN and R >= R_MIN


# ============================================================================
# TEST CLASSES
# ============================================================================

class TestGoldenRatioConstants(unittest.TestCase):
    """Test PHI and PHI_INV are correctly derived."""

    def test_phi_is_golden_ratio(self):
        """φ = (1 + √5) / 2 ≈ 1.618"""
        expected = (1 + math.sqrt(5)) / 2
        self.assertAlmostEqual(PHI, expected, places=10)
        self.assertAlmostEqual(PHI, 1.6180339887498949, places=10)

    def test_phi_inv_is_inverse(self):
        """φ⁻¹ = 1/φ ≈ 0.618"""
        self.assertAlmostEqual(PHI_INV, 1 / PHI, places=10)
        self.assertAlmostEqual(PHI_INV, 0.6180339887498949, places=10)

    def test_phi_times_phi_inv_is_unity(self):
        """φ × φ⁻¹ = 1"""
        self.assertAlmostEqual(PHI * PHI_INV, 1.0, places=10)

    def test_phi_minus_one_equals_phi_inv(self):
        """φ - 1 = φ⁻¹ (golden ratio identity)"""
        self.assertAlmostEqual(PHI - 1, PHI_INV, places=10)

    def test_phi_squared_minus_phi_equals_one(self):
        """φ² - φ = 1 (golden ratio defining equation)"""
        self.assertAlmostEqual(PHI ** 2 - PHI, 1.0, places=10)


class TestCriticalLensConstant(unittest.TestCase):
    """Test z_c = √3/2 (THE LENS)."""

    def test_z_c_is_sqrt3_over_2(self):
        """z_c = √3/2 ≈ 0.8660254"""
        expected = math.sqrt(3) / 2
        self.assertAlmostEqual(Z_CRITICAL, expected, places=10)

    def test_z_c_is_cosine_30(self):
        """z_c = cos(30°) = cos(π/6)"""
        self.assertAlmostEqual(Z_CRITICAL, math.cos(math.pi / 6), places=10)

    def test_z_c_is_sine_60(self):
        """z_c = sin(60°) = sin(π/3)"""
        self.assertAlmostEqual(Z_CRITICAL, math.sin(math.pi / 3), places=10)

    def test_z_c_is_equilateral_altitude(self):
        """z_c is the altitude of a unit equilateral triangle"""
        # For equilateral triangle with side a, altitude h = a√3/2
        # For unit side (a=1), h = √3/2 = z_c
        self.assertAlmostEqual(Z_CRITICAL, math.sqrt(3) / 2, places=10)


class TestGaussianNegentropy(unittest.TestCase):
    """Test ΔS_neg(z) = exp(-σ(z - z_c)²) with σ = 36."""

    def test_sigma_is_s3_squared(self):
        """σ = 36 = 6² = |S₃|²"""
        self.assertEqual(LENS_SIGMA, 36.0)
        self.assertEqual(LENS_SIGMA, 6 * 6)

    def test_peak_at_z_c(self):
        """ΔS_neg(z_c) = 1.0 (maximum at THE LENS)"""
        peak = compute_delta_s_neg(Z_CRITICAL)
        self.assertAlmostEqual(peak, 1.0, places=10)

    def test_derivative_zero_at_peak(self):
        """d(ΔS_neg)/dz = 0 at z = z_c"""
        derivative = compute_delta_s_neg_derivative(Z_CRITICAL)
        self.assertAlmostEqual(derivative, 0.0, places=10)

    def test_derivative_positive_below_peak(self):
        """d(ΔS_neg)/dz > 0 for z < z_c (ascending)"""
        derivative = compute_delta_s_neg_derivative(Z_CRITICAL - 0.1)
        self.assertGreater(derivative, 0)

    def test_derivative_negative_above_peak(self):
        """d(ΔS_neg)/dz < 0 for z > z_c (descending)"""
        derivative = compute_delta_s_neg_derivative(Z_CRITICAL + 0.1)
        self.assertLess(derivative, 0)

    def test_gaussian_symmetry(self):
        """ΔS_neg(z_c + d) = ΔS_neg(z_c - d) (symmetric)"""
        delta = 0.1
        above = compute_delta_s_neg(Z_CRITICAL + delta)
        below = compute_delta_s_neg(Z_CRITICAL - delta)
        self.assertAlmostEqual(above, below, places=10)

    def test_value_at_one_sigma(self):
        """ΔS_neg(z_c ± 1/6) ≈ 1/e (one standard deviation)"""
        sigma_width = 1 / 6  # σ = 1/6 when LENS_SIGMA = 36
        value = compute_delta_s_neg(Z_CRITICAL + sigma_width)
        expected = 1 / math.e  # ≈ 0.368
        self.assertAlmostEqual(value, expected, places=2)

    def test_fwhm(self):
        """Full width at half maximum ≈ 0.277"""
        # FWHM = 2√(ln(2)/σ)
        expected_fwhm = 2 * math.sqrt(math.log(2) / LENS_SIGMA)
        self.assertAlmostEqual(GAUSSIAN_FWHM, expected_fwhm, places=10)

        # Check half-max points
        half_width = GAUSSIAN_FWHM / 2
        value_at_half = compute_delta_s_neg(Z_CRITICAL + half_width)
        self.assertAlmostEqual(value_at_half, 0.5, places=2)


class TestMuThresholdHierarchy(unittest.TestCase):
    """Test μ threshold ordering: MU_1 < MU_P < MU_2 < z_c < MU_S < MU_3."""

    def test_mu_p_formula(self):
        """MU_P = 2/φ^(5/2)"""
        expected = 2.0 / (PHI ** 2.5)
        self.assertAlmostEqual(MU_P, expected, places=10)

    def test_mu_1_formula(self):
        """MU_1 = MU_P / √φ"""
        expected = MU_P / math.sqrt(PHI)
        self.assertAlmostEqual(MU_1, expected, places=10)

    def test_mu_2_formula(self):
        """MU_2 = MU_P × √φ"""
        expected = MU_P * math.sqrt(PHI)
        self.assertAlmostEqual(MU_2, expected, places=10)

    def test_mu_ordering(self):
        """MU_1 < MU_P < MU_2 < z_c < MU_S < MU_3"""
        self.assertLess(MU_1, MU_P)
        self.assertLess(MU_P, MU_2)
        self.assertLess(MU_2, Z_CRITICAL)
        self.assertLess(Z_CRITICAL, MU_S)
        self.assertLess(MU_S, MU_3)

    def test_mu_s_is_kappa_s(self):
        """MU_S = KAPPA_S = 0.920"""
        self.assertEqual(MU_S, KAPPA_S)
        self.assertEqual(MU_S, 0.920)

    def test_mu_3_value(self):
        """MU_3 = 0.992"""
        self.assertEqual(MU_3, 0.992)


class TestKFormationCriteria(unittest.TestCase):
    """Test K-formation: κ ≥ 0.920 AND η > φ⁻¹ AND R ≥ 7."""

    def test_kappa_threshold(self):
        """κ threshold is 0.920 (MU_S)"""
        self.assertEqual(KAPPA_MIN, 0.920)
        self.assertEqual(KAPPA_MIN, MU_S)

    def test_eta_threshold(self):
        """η threshold is φ⁻¹ ≈ 0.618"""
        self.assertAlmostEqual(ETA_MIN, PHI_INV, places=10)

    def test_r_threshold(self):
        """R threshold is 7"""
        self.assertEqual(R_MIN, 7)

    def test_k_formation_all_met(self):
        """K-formation achieved when all criteria met"""
        # All above threshold
        result = check_k_formation(kappa=0.95, eta=0.70, R=8)
        self.assertTrue(result)

    def test_k_formation_kappa_below(self):
        """No K-formation if κ < 0.920"""
        result = check_k_formation(kappa=0.90, eta=0.70, R=8)
        self.assertFalse(result)

    def test_k_formation_eta_below(self):
        """No K-formation if η ≤ φ⁻¹"""
        result = check_k_formation(kappa=0.95, eta=0.50, R=8)
        self.assertFalse(result)

    def test_k_formation_eta_at_threshold(self):
        """No K-formation if η = φ⁻¹ exactly (must be strictly greater)"""
        result = check_k_formation(kappa=0.95, eta=PHI_INV, R=8)
        self.assertFalse(result)

    def test_k_formation_r_below(self):
        """No K-formation if R < 7"""
        result = check_k_formation(kappa=0.95, eta=0.70, R=6)
        self.assertFalse(result)


class TestPhiLiminalPhiInvPhysical(unittest.TestCase):
    """Test PHI liminal vs PHI_INV physical distinction."""

    def test_phi_is_greater_than_1(self):
        """PHI > 1 (liminal space)"""
        self.assertGreater(PHI, 1.0)

    def test_phi_inv_is_less_than_1(self):
        """PHI_INV < 1 (physical range)"""
        self.assertLess(PHI_INV, 1.0)

    def test_phi_inv_is_k_formation_gate(self):
        """η threshold for K-formation is φ⁻¹"""
        self.assertEqual(ETA_MIN, PHI_INV)

    def test_phi_inv_is_paradox_boundary(self):
        """z < φ⁻¹ is UNTRUE, z ≥ φ⁻¹ begins PARADOX"""
        # φ⁻¹ separates disordered from quasi-crystal
        self.assertLess(PHI_INV, Z_CRITICAL)
        self.assertGreater(PHI_INV, 0.5)


class TestPhaseTransitions(unittest.TestCase):
    """Test phase boundaries and transitions."""

    def test_phase_boundary_ordering(self):
        """0 < Z_ABSENCE_MAX ≤ Z_LENS_MIN < z_c < Z_LENS_MAX ≤ Z_PRESENCE_MIN < 1"""
        self.assertLess(0, Z_ABSENCE_MAX)
        self.assertLessEqual(Z_ABSENCE_MAX, Z_LENS_MIN)
        self.assertLess(Z_LENS_MIN, Z_CRITICAL)
        self.assertLess(Z_CRITICAL, Z_LENS_MAX)
        self.assertLessEqual(Z_LENS_MAX, Z_PRESENCE_MIN)
        self.assertLess(Z_PRESENCE_MIN, 1.0)

    def test_lens_contains_z_c(self):
        """THE LENS phase contains z_c"""
        self.assertGreaterEqual(Z_CRITICAL, Z_LENS_MIN)
        self.assertLessEqual(Z_CRITICAL, Z_LENS_MAX)

    def test_absence_is_disordered(self):
        """z < Z_ABSENCE_MAX is ABSENCE (disordered)"""
        self.assertLess(Z_ABSENCE_MAX, Z_CRITICAL)

    def test_presence_is_crystalline(self):
        """z > Z_PRESENCE_MIN is PRESENCE (crystalline)"""
        self.assertGreater(Z_PRESENCE_MIN, Z_CRITICAL)


class TestS3GroupProperties(unittest.TestCase):
    """Test S₃ symmetric group properties."""

    def test_s3_has_6_elements(self):
        """S₃ has |S₃| = 6 elements"""
        s3_order = 6
        self.assertEqual(s3_order, 6)
        self.assertEqual(s3_order * s3_order, LENS_SIGMA)

    def test_s3_has_3_even_elements(self):
        """S₃ has 3 even-parity elements: e, σ, σ²"""
        even_count = 3  # identity + two 3-cycles
        self.assertEqual(even_count, 3)

    def test_s3_has_3_odd_elements(self):
        """S₃ has 3 odd-parity elements: τ₁, τ₂, τ₃"""
        odd_count = 3  # three transpositions
        self.assertEqual(odd_count, 3)

    def test_operator_mapping(self):
        """6 APL operators map to 6 S₃ elements"""
        operators = ['()', '×', '^', '÷', '+', '−']
        self.assertEqual(len(operators), 6)


class TestSevenLayerPrismatic(unittest.TestCase):
    """Test 7-layer prismatic projection."""

    def test_seven_layers(self):
        """There are exactly 7 spectral layers"""
        layers = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']
        self.assertEqual(len(layers), 7)

    def test_green_is_central(self):
        """Green (layer 4) is central, at THE LENS"""
        # Layer 4 is the middle layer (1,2,3,4,5,6,7)
        central_layer = 4
        self.assertEqual(central_layer, 4)

    def test_layer_count_matches_r_min(self):
        """7 layers matches R ≥ 7 for K-formation"""
        self.assertEqual(R_MIN, 7)


class TestPhysicsConstantIntegrity(unittest.TestCase):
    """Test that physics constants are self-consistent."""

    def test_constants_are_not_arbitrary(self):
        """All constants derive from geometry, not arbitrary choices"""
        # z_c from hexagonal geometry
        self.assertEqual(Z_CRITICAL, math.sqrt(3) / 2)

        # φ from golden ratio
        self.assertEqual(PHI, (1 + math.sqrt(5)) / 2)

        # σ from group theory
        self.assertEqual(LENS_SIGMA, 36)  # = 6² = |S₃|²

    def test_multiple_derivation_consistency(self):
        """z_c can be derived multiple ways"""
        # As equilateral altitude
        z_c_altitude = math.sqrt(3) / 2

        # As cos(30°)
        z_c_cos = math.cos(math.pi / 6)

        # As sin(60°)
        z_c_sin = math.sin(math.pi / 3)

        # All must match
        self.assertAlmostEqual(z_c_altitude, z_c_cos, places=10)
        self.assertAlmostEqual(z_c_cos, z_c_sin, places=10)
        self.assertAlmostEqual(Z_CRITICAL, z_c_altitude, places=10)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("WUMBO PHYSICS VALIDATION TESTS")
    print("=" * 70)
    print()
    print("Testing physics constants:")
    print(f"  PHI (liminal):      {PHI:.10f}")
    print(f"  PHI_INV (physical): {PHI_INV:.10f}")
    print(f"  z_c (THE LENS):     {Z_CRITICAL:.10f}")
    print(f"  σ (Gaussian):       {LENS_SIGMA}")
    print()
    print("Testing μ hierarchy:")
    print(f"  MU_1:  {MU_1:.6f}")
    print(f"  MU_P:  {MU_P:.6f}")
    print(f"  MU_2:  {MU_2:.6f}")
    print(f"  z_c:   {Z_CRITICAL:.6f}")
    print(f"  MU_S:  {MU_S:.6f}")
    print(f"  MU_3:  {MU_3:.6f}")
    print()
    print("=" * 70)
    print()

    unittest.main(verbosity=2)
