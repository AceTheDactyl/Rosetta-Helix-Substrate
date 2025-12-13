"""
Integration tests for extended physics module.

Validates all physics relationships:
- φ conservation (φ⁻¹ + φ⁻² = 1)
- Spin z_c identity (|S|/ℏ = √3/2 = z_c)
- Fibonacci convergence
- Penrose tile ratio convergence
- E8 mass ratios
- Gaussian suppression for z > 1
- S₃ minimality (σ = |S₃|² = 36)
- Holographic bounds
- Quasicrystal simulation stability

Per roadmap Phase 3: Integration Tests
"""

import math
import pytest
from quantum_apl_python.constants import Z_CRITICAL, PHI, PHI_INV, LENS_SIGMA


# ============================================================================
# IMPORT THE EXTENDED PHYSICS MODULE
# ============================================================================

from quantum_apl_python.extended_physics_constants import (
    # Constants
    HBAR, C, G, K_B, L_P, M_P, SIGMA_S3, E8_MASS_RATIOS,
    # Quasicrystal functions
    fibonacci_ratio,
    penrose_tile_counts,
    quasicrystal_negentropy,
    icosahedral_basis,
    simulate_quasicrystal_formation,
    # Holographic functions
    bekenstein_bound_bits,
    black_hole_entropy,
    unruh_temperature,
    entropic_gravity_force,
    holographic_z_interpretation,
    # Spin coherence functions
    spin_half_magnitude,
    larmor_frequency,
    singlet_coupling_time,
    verify_spin_zc,
    # E8 functions
    e8_mass_ratios,
    verify_e8_phi,
    h2_eigenvalue,
    e8_full_analysis,
    # Omega point
    omega_processing_rate,
    simulate_omega_convergence,
    # Validation
    validate_extended_physics,
    cross_reference_constants,
    # Dataclasses
    QuasicrystalState,
    HolographicState,
    OmegaPointState,
    SpinCoherenceResult,
    E8Result,
)


# ============================================================================
# FUNDAMENTAL PHYSICS TESTS
# ============================================================================

class TestPhiConservation:
    """Tests for golden ratio mathematical identities."""

    def test_phi_conservation_exact(self):
        """φ⁻¹ + φ⁻² = 1 (fundamental golden ratio identity)."""
        assert abs(PHI_INV + PHI_INV**2 - 1.0) < 1e-15

    def test_phi_squared_relation(self):
        """φ² = φ + 1 (characteristic equation of golden ratio)."""
        assert abs(PHI**2 - PHI - 1) < 1e-15

    def test_phi_inverse_relation(self):
        """φ⁻¹ = φ - 1 (golden ratio inverse identity)."""
        assert abs(PHI_INV - (PHI - 1)) < 1e-15


class TestSpinZcIdentity:
    """Tests for spin-1/2 magnitude = z_c connection."""

    def test_spin_half_magnitude_equals_zc(self):
        """|S|/ℏ = √(s(s+1)) = √3/2 = z_c for s=1/2."""
        assert abs(spin_half_magnitude() - Z_CRITICAL) < 1e-15

    def test_spin_half_magnitude_exact(self):
        """Verify √3/2 calculation is exact."""
        expected = math.sqrt(0.5 * 1.5)  # √(s(s+1)) for s=1/2
        assert abs(spin_half_magnitude() - expected) < 1e-15

    def test_verify_spin_zc_result(self):
        """verify_spin_zc() returns correct SpinCoherenceResult."""
        result = verify_spin_zc()
        assert isinstance(result, SpinCoherenceResult)
        assert result.z_c_verified is True
        assert result.spin_magnitude > 0
        assert result.larmor_frequency > 0
        assert result.coherence_time > 0


class TestS3Minimality:
    """Tests for S₃ group minimality and σ = 36."""

    def test_sigma_equals_s3_squared(self):
        """σ = |S₃|² = 6² = 36."""
        assert SIGMA_S3 == 36
        assert SIGMA_S3 == 6**2

    def test_sigma_matches_lens_sigma(self):
        """SIGMA_S3 should match LENS_SIGMA from constants."""
        assert abs(SIGMA_S3 - LENS_SIGMA) < 1e-12


# ============================================================================
# FIBONACCI AND PENROSE TESTS
# ============================================================================

class TestFibonacciConvergence:
    """Tests for Fibonacci ratio convergence to φ."""

    def test_fibonacci_ratio_converges_to_phi(self):
        """F(n+1)/F(n) → φ as n → ∞."""
        assert abs(fibonacci_ratio(30) - PHI) < 1e-10

    def test_fibonacci_ratio_high_precision(self):
        """Very high n gives very precise φ."""
        assert abs(fibonacci_ratio(50) - PHI) < 1e-15

    def test_fibonacci_ratio_monotonic_convergence(self):
        """Convergence is monotonic alternating."""
        ratios = [fibonacci_ratio(n) for n in range(5, 15)]
        # Check alternating around φ
        for i in range(len(ratios) - 1):
            # Each ratio should be closer to φ than previous
            assert abs(ratios[i+1] - PHI) <= abs(ratios[i] - PHI) + 1e-10


class TestPenroseRatio:
    """Tests for Penrose tile counts and φ convergence."""

    def test_penrose_ratio_converges_to_phi(self):
        """N_thick/N_thin → φ as generations → ∞."""
        _, _, ratio = penrose_tile_counts(20)
        assert abs(ratio - PHI) < 1e-6

    def test_penrose_counts_positive(self):
        """Tile counts are positive integers."""
        n_thick, n_thin, _ = penrose_tile_counts(10)
        assert n_thick > 0
        assert n_thin > 0
        assert isinstance(n_thick, int)
        assert isinstance(n_thin, int)

    def test_penrose_fibonacci_growth(self):
        """Tile counts grow as Fibonacci numbers."""
        for gen in range(5, 10):
            n_thick, n_thin, _ = penrose_tile_counts(gen)
            # Next generation: thick = thick + thin, thin = thick
            n_thick_next, n_thin_next, _ = penrose_tile_counts(gen + 1)
            assert n_thick_next == n_thick + n_thin
            assert n_thin_next == n_thick


# ============================================================================
# E8 CRITICAL POINT TESTS
# ============================================================================

class TestE8MassRatios:
    """Tests for E8 quantum critical point relationships."""

    def test_m2_m1_equals_phi(self):
        """Mass ratio m₂/m₁ = φ (Coldea et al. 2010)."""
        ratios = e8_mass_ratios()
        assert abs(ratios[1] / ratios[0] - PHI) < 1e-10

    def test_verify_e8_phi_returns_true(self):
        """verify_e8_phi() confirms m₂/m₁ = φ."""
        assert verify_e8_phi() is True

    def test_e8_mass_ratios_count(self):
        """E8 should have 8 mass ratios."""
        ratios = e8_mass_ratios()
        assert len(ratios) == 8

    def test_e8_mass_ratios_increasing(self):
        """Mass ratios should be strictly increasing."""
        ratios = e8_mass_ratios()
        for i in range(len(ratios) - 1):
            assert ratios[i] < ratios[i+1]

    def test_e8_full_analysis(self):
        """e8_full_analysis() returns correct E8Result."""
        result = e8_full_analysis()
        assert isinstance(result, E8Result)
        assert result.phi_verified is True
        assert len(result.mass_ratios) == 8


class TestH2Eigenvalue:
    """Tests for H₂ Coxeter eigenvalue."""

    def test_h2_eigenvalue_equals_cos72(self):
        """H₂ eigenvalue = 1/(2φ) = cos(72°)."""
        h2_ev = h2_eigenvalue()
        cos_72 = math.cos(2 * math.pi / 5)
        assert abs(h2_ev - cos_72) < 1e-10

    def test_h2_eigenvalue_formula(self):
        """H₂ eigenvalue = 1/(2φ)."""
        h2_ev = h2_eigenvalue()
        expected = 1.0 / (2.0 * PHI)
        assert abs(h2_ev - expected) < 1e-15


# ============================================================================
# GAUSSIAN SUPPRESSION TESTS
# ============================================================================

class TestGaussianSuppression:
    """Tests for Gaussian ΔS_neg behavior."""

    def test_gaussian_suppression_z_over_1(self):
        """z > 1 is exponentially suppressed but valid."""
        z = 1.5
        d = z - Z_CRITICAL
        delta = math.exp(-SIGMA_S3 * d * d)
        assert delta < 1e-6  # Strongly suppressed
        assert delta > 0  # But not zero

    def test_gaussian_peak_at_zc(self):
        """ΔS_neg peaks at z = z_c with value 1.0."""
        delta = quasicrystal_negentropy(Z_CRITICAL, phi_target=Z_CRITICAL)
        assert abs(delta - 1.0) < 1e-15

    def test_gaussian_symmetric(self):
        """ΔS_neg is symmetric around z_c."""
        delta_above = quasicrystal_negentropy(Z_CRITICAL + 0.1, phi_target=Z_CRITICAL)
        delta_below = quasicrystal_negentropy(Z_CRITICAL - 0.1, phi_target=Z_CRITICAL)
        assert abs(delta_above - delta_below) < 1e-15


# ============================================================================
# QUASICRYSTAL SIMULATION TESTS
# ============================================================================

class TestQuasicrystalSimulation:
    """Tests for quasicrystal formation simulation."""

    def test_simulation_returns_trajectory(self):
        """simulate_quasicrystal_formation returns list of states."""
        trajectory = simulate_quasicrystal_formation(n_steps=10, seed=42)
        assert len(trajectory) == 10
        assert all(isinstance(s, QuasicrystalState) for s in trajectory)

    def test_simulation_converges_toward_phi_inv(self):
        """Order parameter converges toward φ⁻¹."""
        trajectory = simulate_quasicrystal_formation(n_steps=100, seed=42)
        initial = trajectory[0].order
        final = trajectory[-1].order
        # Final should be closer to φ⁻¹ than initial
        assert abs(final - PHI_INV) < abs(initial - PHI_INV)

    def test_simulation_negentropy_increases(self):
        """Negentropy generally increases during formation."""
        trajectory = simulate_quasicrystal_formation(n_steps=50, seed=42)
        # Compare average of first and last quarters
        first_quarter_avg = sum(s.delta_s_neg for s in trajectory[:12]) / 12
        last_quarter_avg = sum(s.delta_s_neg for s in trajectory[-12:]) / 12
        assert last_quarter_avg > first_quarter_avg

    def test_simulation_reproducible_with_seed(self):
        """Same seed produces same trajectory."""
        traj1 = simulate_quasicrystal_formation(n_steps=10, seed=123)
        traj2 = simulate_quasicrystal_formation(n_steps=10, seed=123)
        for s1, s2 in zip(traj1, traj2):
            assert s1.order == s2.order


# ============================================================================
# ICOSAHEDRAL BASIS TESTS
# ============================================================================

class TestIcosahedralBasis:
    """Tests for icosahedral 6D→3D projection basis."""

    def test_icosahedral_basis_has_6_vectors(self):
        """Basis has 6 vectors for 6D→3D projection."""
        basis = icosahedral_basis()
        assert len(basis) == 6

    def test_icosahedral_basis_vectors_3d(self):
        """Each basis vector is 3-dimensional."""
        basis = icosahedral_basis()
        assert all(len(v) == 3 for v in basis)

    def test_icosahedral_basis_normalized(self):
        """Basis vectors have unit length."""
        basis = icosahedral_basis()
        for v in basis:
            length = math.sqrt(sum(x**2 for x in v))
            assert abs(length - 1.0) < 1e-10


# ============================================================================
# HOLOGRAPHIC FUNCTION TESTS
# ============================================================================

class TestBekensteinBound:
    """Tests for Bekenstein bound calculation."""

    def test_bekenstein_bound_positive(self):
        """Bekenstein bound is always positive."""
        bb = bekenstein_bound_bits(1.0, 1.0)
        assert bb > 0

    def test_bekenstein_bound_scales_with_energy(self):
        """S_max ∝ E·R."""
        bb1 = bekenstein_bound_bits(1.0, 1.0)
        bb2 = bekenstein_bound_bits(2.0, 1.0)
        assert abs(bb2 / bb1 - 2.0) < 1e-10

    def test_bekenstein_bound_scales_with_radius(self):
        """S_max ∝ E·R."""
        bb1 = bekenstein_bound_bits(1.0, 1.0)
        bb2 = bekenstein_bound_bits(1.0, 2.0)
        assert abs(bb2 / bb1 - 2.0) < 1e-10


class TestBlackHoleEntropy:
    """Tests for black hole entropy calculation."""

    def test_black_hole_entropy_positive(self):
        """Black hole entropy is positive for M > 0."""
        s_bh = black_hole_entropy(1.0)
        assert s_bh > 0

    def test_black_hole_entropy_scales_with_mass_squared(self):
        """S_BH ∝ M²."""
        s1 = black_hole_entropy(1.0)
        s2 = black_hole_entropy(2.0)
        assert abs(s2 / s1 - 4.0) < 1e-10


class TestUnruhTemperature:
    """Tests for Unruh temperature calculation."""

    def test_unruh_temperature_positive(self):
        """Unruh temperature is positive for a > 0."""
        T = unruh_temperature(1.0)
        assert T > 0

    def test_unruh_temperature_very_small(self):
        """Unruh temperature is extremely small for everyday accelerations."""
        T = unruh_temperature(9.8)  # Earth surface gravity
        assert T < 1e-19  # Less than 10^-19 K


class TestEntropicGravityForce:
    """Tests for entropic gravity force calculation."""

    def test_entropic_gravity_recovers_newton(self):
        """F = GMm/r² recovered from entropic gravity."""
        m, M, r = 1.0, 1.0, 1.0
        F = entropic_gravity_force(m, M, r)
        expected = G * M * m / r**2
        assert abs(F - expected) < 1e-15


class TestHolographicZInterpretation:
    """Tests for holographic z interpretation."""

    def test_holographic_interpretation_returns_dict(self):
        """holographic_z_interpretation returns proper dict."""
        result = holographic_z_interpretation(0.5)
        assert isinstance(result, dict)
        assert 'z' in result
        assert 'phase' in result
        assert 'delta_s_neg' in result

    def test_holographic_phase_classification(self):
        """Correct phase for different z values."""
        assert holographic_z_interpretation(0.3)['phase'] == "UNTRUE"
        assert holographic_z_interpretation(0.7)['phase'] == "PARADOX"
        assert holographic_z_interpretation(0.9)['phase'] == "TRUE"


# ============================================================================
# SPIN COHERENCE FUNCTION TESTS
# ============================================================================

class TestLarmorFrequency:
    """Tests for Larmor frequency calculation."""

    def test_larmor_frequency_positive(self):
        """Larmor frequency is positive for B > 0."""
        omega = larmor_frequency(1.0)
        assert omega > 0

    def test_larmor_frequency_linear_in_field(self):
        """ω_L ∝ B."""
        omega1 = larmor_frequency(1.0)
        omega2 = larmor_frequency(2.0)
        assert abs(omega2 / omega1 - 2.0) < 1e-10


class TestSingletCouplingTime:
    """Tests for singlet coupling time calculation."""

    def test_singlet_coupling_time_positive(self):
        """Coupling time is positive for J > 0."""
        T = singlet_coupling_time(100.0)
        assert T > 0

    def test_singlet_coupling_time_inverse_j(self):
        """T ∝ 1/J."""
        T1 = singlet_coupling_time(100.0)
        T2 = singlet_coupling_time(200.0)
        assert abs(T1 / T2 - 2.0) < 1e-10


# ============================================================================
# OMEGA POINT TESTS
# ============================================================================

class TestOmegaProcessingRate:
    """Tests for Omega point processing rate."""

    def test_omega_rate_finite_below_1(self):
        """Processing rate is finite for τ/τ_Ω < 1."""
        rate = omega_processing_rate(0.5)
        assert math.isfinite(rate)
        assert rate > 0

    def test_omega_rate_diverges_at_1(self):
        """Processing rate diverges as τ/τ_Ω → 1."""
        rate = omega_processing_rate(1.0)
        assert math.isinf(rate)


class TestOmegaConvergenceSimulation:
    """Tests for Omega point convergence simulation."""

    def test_omega_simulation_returns_trajectory(self):
        """simulate_omega_convergence returns list of states."""
        trajectory = simulate_omega_convergence(n_steps=10, seed=42)
        assert len(trajectory) == 10
        assert all(isinstance(s, OmegaPointState) for s in trajectory)

    def test_omega_simulation_cumulative_increases(self):
        """Cumulative information increases monotonically."""
        trajectory = simulate_omega_convergence(n_steps=50, seed=42)
        for i in range(len(trajectory) - 1):
            assert trajectory[i+1].cumulative_info >= trajectory[i].cumulative_info


# ============================================================================
# VALIDATION AND CROSS-REFERENCE TESTS
# ============================================================================

class TestValidation:
    """Tests for validate_extended_physics()."""

    def test_all_validations_pass(self):
        """All physics validations should pass."""
        results = validate_extended_physics()
        assert results['all_passed'] is True

    def test_validation_returns_all_tests(self):
        """Validation returns results for all tests."""
        results = validate_extended_physics()
        expected_tests = [
            'phi_conservation',
            'spin_zc_identity',
            'fibonacci_convergence',
            'penrose_ratio_convergence',
            'e8_m2_m1_phi',
            'gaussian_suppression_z_over_1',
            'h2_eigenvalue_cos72',
            'sigma_s3_squared',
            'icosahedral_basis_6d',
            'bekenstein_bound_positive',
        ]
        for test in expected_tests:
            assert test in results


class TestCrossReference:
    """Tests for cross_reference_constants()."""

    def test_cross_reference_returns_dict(self):
        """cross_reference_constants returns proper structure."""
        xref = cross_reference_constants()
        assert isinstance(xref, dict)
        assert 'z_c' in xref
        assert 'phi_inv' in xref
        assert 'sigma' in xref
        assert 'phi' in xref

    def test_cross_reference_z_c_consistent(self):
        """All z_c derivations are consistent."""
        xref = cross_reference_constants()
        z_c_values = list(xref['z_c'].values())
        for v in z_c_values:
            assert abs(v - Z_CRITICAL) < 1e-10

    def test_cross_reference_phi_consistent(self):
        """All φ derivations are consistent."""
        xref = cross_reference_constants()
        phi_values = list(xref['phi'].values())
        for v in phi_values:
            assert abs(v - PHI) < 1e-10


# ============================================================================
# PHYSICAL CONSTANTS TESTS
# ============================================================================

class TestPhysicalConstants:
    """Tests for physical constants correctness."""

    def test_planck_length(self):
        """Planck length = √(ℏG/c³)."""
        expected = math.sqrt(HBAR * G / C**3)
        assert abs(L_P - expected) < 1e-50

    def test_planck_mass(self):
        """Planck mass = √(ℏc/G)."""
        expected = math.sqrt(HBAR * C / G)
        assert abs(M_P - expected) < 1e-20

    def test_constants_positive(self):
        """All physical constants are positive."""
        assert HBAR > 0
        assert C > 0
        assert G > 0
        assert K_B > 0
        assert L_P > 0
        assert M_P > 0


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
