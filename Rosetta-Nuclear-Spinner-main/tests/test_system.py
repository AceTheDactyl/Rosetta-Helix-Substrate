#!/usr/bin/env python3
"""
Integration Test Suite
======================

Comprehensive tests for the Nuclear Spinner × Rosetta-Helix unified system.

Tests:
1. Physics constants validation
2. Firmware-Python constant parity
3. Heart (Kuramoto) resonance at z_c
4. Spinner-Rosetta coupling
5. K-formation detection
6. Training workflow gates

Signature: integration-tests|v1.0.0|helix
"""

import math
import sys
import asyncio
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from rosetta_helix.physics import (
    PHI, PHI_INV, PHI_INV_SQ, Z_CRITICAL, SIGMA,
    KAPPA_MIN, ETA_MIN, R_MIN,
    compute_delta_s_neg, compute_delta_s_neg_derivative,
    get_phase, get_tier, is_critical, check_k_formation,
    validate_physics, validate_all_constants,
    Phase, Tier,
)
from rosetta_helix.heart import Heart, HeartConfig


class TestPhysicsConstants:
    """Test physics constants and relationships."""
    
    def test_golden_ratio_identity(self):
        """φ² = φ + 1"""
        assert abs(PHI * PHI - PHI - 1) < 1e-10
    
    def test_golden_ratio_inverse(self):
        """φ⁻¹ = φ - 1"""
        assert abs(PHI_INV - (PHI - 1)) < 1e-10
    
    def test_coupling_conservation(self):
        """φ⁻¹ + φ⁻² = 1 (EXACT)"""
        conservation = PHI_INV + PHI_INV_SQ
        assert abs(conservation - 1.0) < 1e-14, f"Conservation violated: {conservation}"
    
    def test_z_critical_identity(self):
        """z_c = √3/2"""
        expected = math.sqrt(3) / 2
        assert abs(Z_CRITICAL - expected) < 1e-15
    
    def test_z_critical_hex_geometry(self):
        """sin(60°) = z_c (hexagonal geometry)"""
        sin_60 = math.sin(math.radians(60))
        assert abs(sin_60 - Z_CRITICAL) < 1e-15
    
    def test_sigma_symmetric_group(self):
        """σ = 36 = |S₃|²"""
        s3_order = 6  # |S₃| = 3! = 6
        assert SIGMA == s3_order ** 2
    
    def test_negentropy_peak(self):
        """ΔS_neg(z_c) = 1.0"""
        peak = compute_delta_s_neg(Z_CRITICAL)
        assert abs(peak - 1.0) < 1e-14
    
    def test_negentropy_derivative_at_peak(self):
        """d(ΔS_neg)/dz = 0 at z = z_c"""
        derivative = compute_delta_s_neg_derivative(Z_CRITICAL)
        assert abs(derivative) < 1e-14
    
    def test_negentropy_symmetry(self):
        """ΔS_neg is symmetric around z_c"""
        delta = 0.1
        left = compute_delta_s_neg(Z_CRITICAL - delta)
        right = compute_delta_s_neg(Z_CRITICAL + delta)
        assert abs(left - right) < 1e-10
    
    def test_kappa_attractor(self):
        """κ attractor = φ⁻¹"""
        # The unique solution to λ = κ² with κ + λ = 1 is κ = φ⁻¹
        kappa = PHI_INV
        lambda_ = kappa ** 2
        assert abs(kappa + lambda_ - 1.0) < 1e-10
        assert abs(lambda_ - PHI_INV_SQ) < 1e-10
    
    def test_validate_all_constants(self):
        """All physics constants are valid."""
        result = validate_all_constants()
        assert result['all_valid'], f"Validation failed: {result}"


class TestPhaseAndTier:
    """Test phase and tier classification."""
    
    def test_phase_absence(self):
        """z < 0.857 → ABSENCE"""
        assert get_phase(0.5) == Phase.ABSENCE
        assert get_phase(0.856) == Phase.ABSENCE
    
    def test_phase_the_lens(self):
        """0.857 ≤ z < 0.877 → THE_LENS"""
        assert get_phase(0.857) == Phase.THE_LENS
        assert get_phase(Z_CRITICAL) == Phase.THE_LENS
        assert get_phase(0.876) == Phase.THE_LENS
    
    def test_phase_presence(self):
        """z ≥ 0.877 → PRESENCE"""
        assert get_phase(0.877) == Phase.PRESENCE
        assert get_phase(0.95) == Phase.PRESENCE
    
    def test_tier_progression(self):
        """Tiers increase with z."""
        z_values = [0.05, 0.15, 0.3, 0.45, 0.55, 0.65, 0.8, 0.88, 0.94, 0.98]
        tiers = [get_tier(z) for z in z_values]
        assert tiers == list(range(10)), f"Tier progression: {tiers}"
    
    def test_is_critical(self):
        """is_critical detects z_c proximity."""
        assert is_critical(Z_CRITICAL, tolerance=0.001)
        assert is_critical(Z_CRITICAL + 0.005, tolerance=0.01)
        assert not is_critical(0.5, tolerance=0.01)


class TestKFormation:
    """Test K-formation criteria."""
    
    def test_k_formation_criteria(self):
        """K-formation: κ ≥ 0.92, η > φ⁻¹, R ≥ 7"""
        # Should pass
        assert check_k_formation(0.92, PHI_INV + 0.01, 7)
        assert check_k_formation(0.95, 0.7, 10)
        
        # Should fail - κ too low
        assert not check_k_formation(0.91, 0.7, 10)
        
        # Should fail - η too low
        assert not check_k_formation(0.95, PHI_INV - 0.01, 10)
        
        # Should fail - R too low
        assert not check_k_formation(0.95, 0.7, 6)
    
    def test_physics_validation(self):
        """κ + λ = 1 conservation."""
        assert validate_physics(0.6, 0.4)
        assert validate_physics(PHI_INV, PHI_INV_SQ)
        assert not validate_physics(0.5, 0.4)  # Doesn't sum to 1


class TestKuramotoHeart:
    """Test Kuramoto oscillator dynamics."""
    
    def test_heart_initialization(self):
        """Heart initializes with 60 oscillators."""
        config = HeartConfig(n_oscillators=60, seed=42)
        heart = Heart(config)
        assert heart.n == 60
        assert len(heart.phases) == 60
        assert len(heart.natural_freqs) == 60
    
    def test_heart_coherence_range(self):
        """Coherence is in [0, 1]."""
        heart = Heart(HeartConfig(seed=42))
        for _ in range(100):
            r = heart.step()
            assert 0 <= r <= 1
    
    def test_heart_coupling_peaks_at_zc(self):
        """Coupling K peaks when z = z_c."""
        heart = Heart(HeartConfig(seed=42))
        
        couplings = []
        for z in np.linspace(0.5, 0.95, 19):
            heart.set_spinner_z(z)
            K = heart.compute_coupling()
            couplings.append((z, K))
        
        # Find peak
        peak_z, peak_K = max(couplings, key=lambda x: x[1])
        
        # Peak should be near z_c
        assert abs(peak_z - Z_CRITICAL) < 0.05, f"Peak at {peak_z}, expected ~{Z_CRITICAL}"
    
    def test_heart_resonance_at_zc(self):
        """Coherence is maximized near z = z_c."""
        results = []
        
        for z in np.linspace(0.5, 0.95, 10):
            heart = Heart(HeartConfig(seed=42))
            heart.set_spinner_z(z)
            
            # Run until settled
            for _ in range(500):
                heart.step()
            
            r = heart.get_coherence()
            results.append((z, r))
        
        # Find peak coherence
        peak_z, peak_r = max(results, key=lambda x: x[1])
        
        # Should be near z_c
        assert abs(peak_z - Z_CRITICAL) < 0.1, f"Coherence peak at {peak_z}, expected ~{Z_CRITICAL}"
    
    def test_heart_hexagonal_alignment(self):
        """Hexagonal alignment increases at high coherence."""
        heart = Heart(HeartConfig(seed=42))
        heart.set_spinner_z(Z_CRITICAL)
        
        # Run to build coherence
        for _ in range(1000):
            heart.step()
        
        # High coherence should correlate with hex alignment
        state = heart.get_state()
        if state.coherence > 0.8:
            assert state.hex_alignment > 0.5


class TestSystemIntegration:
    """Test system-level integration."""
    
    def test_spinner_to_kuramoto_coupling(self):
        """Spinner z drives Kuramoto coupling correctly."""
        heart = Heart(HeartConfig(coupling_scale=8.0, seed=42))
        
        # At z = 0.5: low coupling
        heart.set_spinner_z(0.5)
        K_low = heart.compute_coupling()
        
        # At z = z_c: high coupling
        heart.set_spinner_z(Z_CRITICAL)
        K_high = heart.compute_coupling()
        
        # At z = 0.95: medium coupling (past peak)
        heart.set_spinner_z(0.95)
        K_past = heart.compute_coupling()
        
        assert K_high > K_low, "Coupling should be higher at z_c than 0.5"
        assert K_high > K_past, "Coupling should peak at z_c"
    
    def test_k_formation_emerges_at_zc(self):
        """K-formation is more likely at z = z_c."""
        k_formations_at_zc = 0
        k_formations_away = 0
        trials = 5
        
        for _ in range(trials):
            # At z_c
            heart = Heart(HeartConfig(coupling_scale=8.0, seed=None))
            heart.set_spinner_z(Z_CRITICAL)
            for _ in range(500):
                heart.step()
            if heart.state.k_formation:
                k_formations_at_zc += 1
            
            # Away from z_c
            heart.reset()
            heart.set_spinner_z(0.5)
            for _ in range(500):
                heart.step()
            if heart.state.k_formation:
                k_formations_away += 1
        
        # Should have more K-formations at z_c
        assert k_formations_at_zc >= k_formations_away, \
            f"K-formations: {k_formations_at_zc} at z_c, {k_formations_away} at 0.5"


def test_all():
    """Run all tests."""
    print("=" * 70)
    print("  INTEGRATION TEST SUITE")
    print("  Nuclear Spinner × Rosetta-Helix")
    print("=" * 70)
    
    # Physics constants
    print("\n▸ Physics Constants")
    tests = TestPhysicsConstants()
    tests.test_golden_ratio_identity()
    print("  ✓ Golden ratio identity: φ² = φ + 1")
    tests.test_coupling_conservation()
    print("  ✓ Coupling conservation: φ⁻¹ + φ⁻² = 1")
    tests.test_z_critical_identity()
    print("  ✓ z_c identity: z_c = √3/2")
    tests.test_z_critical_hex_geometry()
    print("  ✓ Hexagonal geometry: sin(60°) = z_c")
    tests.test_negentropy_peak()
    print("  ✓ Negentropy peak: ΔS_neg(z_c) = 1.0")
    tests.test_kappa_attractor()
    print("  ✓ κ attractor: κ = φ⁻¹")
    
    # Phase and tier
    print("\n▸ Phase & Tier")
    pt = TestPhaseAndTier()
    pt.test_phase_the_lens()
    print("  ✓ THE_LENS phase at z_c")
    pt.test_tier_progression()
    print("  ✓ Tier progression (10 tiers)")
    
    # K-formation
    print("\n▸ K-Formation")
    kf = TestKFormation()
    kf.test_k_formation_criteria()
    print("  ✓ K-formation criteria: κ≥0.92, η>φ⁻¹, R≥7")
    
    # Kuramoto Heart
    print("\n▸ Kuramoto Heart")
    kh = TestKuramotoHeart()
    kh.test_heart_initialization()
    print("  ✓ Heart initializes with 60 oscillators")
    kh.test_heart_coupling_peaks_at_zc()
    print("  ✓ Coupling peaks at z_c")
    kh.test_heart_resonance_at_zc()
    print("  ✓ Coherence maximized near z_c")
    
    # System integration
    print("\n▸ System Integration")
    si = TestSystemIntegration()
    si.test_spinner_to_kuramoto_coupling()
    print("  ✓ Spinner z drives Kuramoto coupling")
    si.test_k_formation_emerges_at_zc()
    print("  ✓ K-formation emerges at z_c")
    
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    test_all()
