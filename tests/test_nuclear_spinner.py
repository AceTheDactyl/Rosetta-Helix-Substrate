#!/usr/bin/env python3
"""
Nuclear Spinner Test Suite
==========================

Comprehensive tests for the Nuclear Spinner module.

Tests cover:
- Core NuclearSpinner class functionality
- Firmware simulation and control loop
- Cybernetic computation library
- Neuroscience extensions
- Communication protocol
- Physics validation (z_c, phi, sigma)
- K-formation criteria
- Operator N0 law compliance

Run with: pytest tests/test_nuclear_spinner.py -v

Signature: nuclear-spinner-tests|v1.0.0|helix
"""

import sys
import os
import math
import pytest
from typing import List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import from physics_constants (single source of truth)
from physics_constants import (
    PHI,
    PHI_INV,
    Z_CRITICAL,
    SIGMA,
    KAPPA_S,
    check_k_formation as physics_check_k_formation,
    compute_delta_s_neg as physics_delta_s_neg,
)

# Import nuclear_spinner module
from nuclear_spinner import (
    NuclearSpinner,
    SpinnerState,
    SpinnerMetrics,
    PulseParameters,
    FirmwareState,
    Phase,
)

from nuclear_spinner.constants import (
    SIGMA_S3,
    MIN_RPM,
    MAX_RPM,
    TIER_BOUNDS,
    CAPABILITY_CLASSES,
    NEURAL_BANDS,
    get_tier_for_z,
    get_capability_class,
)

from nuclear_spinner.analysis import (
    compute_delta_s_neg,
    compute_gradient,
    ashby_variety,
    shannon_capacity,
    landauer_efficiency,
    compute_phi_proxy,
    get_phase,
    get_tier,
    check_k_formation,
)

from nuclear_spinner.firmware import (
    map_z_to_rpm,
    control_loop_step,
    compute_operator_state_update,
    apply_operator_boundary,
    apply_operator_fusion,
    apply_operator_amplify,
    check_n0_legal,
    TRIAD_HIGH,
    TRIAD_LOW,
)

from nuclear_spinner.neural import (
    grid_cell_pattern,
    hexagonal_spacing_metric,
    compute_modulation_index,
    neural_band_to_z,
    z_to_neural_band,
    compute_phi_proxy as neural_phi_proxy,
)

from nuclear_spinner.protocol import (
    CommandType,
    ResponseStatus,
    CommandFrame,
    ResponseFrame,
    encode_command,
    decode_response,
    encode_payload,
    decode_payload,
    compute_crc16,
)


# =============================================================================
# PHYSICS VALIDATION TESTS
# =============================================================================

class TestPhysicsConstants:
    """Test physics constants are correctly defined."""

    def test_phi_golden_ratio(self):
        """Test phi = (1 + sqrt(5)) / 2."""
        expected = (1 + math.sqrt(5)) / 2
        assert abs(PHI - expected) < 1e-15, f"PHI mismatch: {PHI} != {expected}"

    def test_phi_inv_reciprocal(self):
        """Test phi_inv = 1 / phi."""
        expected = 1 / PHI
        assert abs(PHI_INV - expected) < 1e-15, f"PHI_INV mismatch"

    def test_phi_conservation(self):
        """Test phi^-1 + phi^-2 = 1."""
        result = PHI_INV + PHI_INV**2
        assert abs(result - 1.0) < 1e-15, f"Phi conservation violated: {result}"

    def test_z_critical_sqrt3_over_2(self):
        """Test z_c = sqrt(3) / 2."""
        expected = math.sqrt(3) / 2
        assert abs(Z_CRITICAL - expected) < 1e-15, f"Z_CRITICAL mismatch"

    def test_z_critical_sin_60(self):
        """Test z_c = sin(60 degrees)."""
        expected = math.sin(math.pi / 3)
        assert abs(Z_CRITICAL - expected) < 1e-15, f"Z_CRITICAL != sin(60)"

    def test_sigma_s3_squared(self):
        """Test sigma = |S_3|^2 = 36."""
        assert SIGMA_S3 == 36, f"SIGMA_S3 != 36: {SIGMA_S3}"
        assert SIGMA == 36.0, f"SIGMA != 36: {SIGMA}"

    def test_spin_half_magnitude_equals_z_critical(self):
        """Test |S|/hbar = sqrt(s(s+1)) = sqrt(3)/2 = z_c for s=1/2."""
        s = 0.5
        spin_mag = math.sqrt(s * (s + 1))
        assert abs(spin_mag - Z_CRITICAL) < 1e-15, (
            f"Spin-1/2 magnitude {spin_mag} != Z_CRITICAL {Z_CRITICAL}"
        )


# =============================================================================
# NEGENTROPY FUNCTION TESTS
# =============================================================================

class TestNegentropy:
    """Test negentropy (lens weight) functions."""

    def test_negentropy_peaks_at_z_critical(self):
        """Delta_s_neg should peak at z_c with value 1.0."""
        at_zc = compute_delta_s_neg(Z_CRITICAL)
        assert abs(at_zc - 1.0) < 1e-15, f"Peak at z_c = {at_zc}, expected 1.0"

    def test_negentropy_symmetric(self):
        """Delta_s_neg should be symmetric around z_c."""
        offset = 0.1
        below = compute_delta_s_neg(Z_CRITICAL - offset)
        above = compute_delta_s_neg(Z_CRITICAL + offset)
        assert abs(below - above) < 1e-10, f"Asymmetric: {below} vs {above}"

    def test_negentropy_bounded_0_1(self):
        """Delta_s_neg should be in [0, 1]."""
        for z in [0.0, 0.3, 0.5, 0.618, 0.866, 0.95, 1.0]:
            val = compute_delta_s_neg(z)
            assert 0.0 <= val <= 1.0, f"Out of bounds at z={z}: {val}"

    def test_negentropy_decays_away_from_lens(self):
        """Values away from z_c should be less than 1."""
        for z in [0.0, 0.3, 0.5, 0.618, 0.95, 1.0]:
            if abs(z - Z_CRITICAL) > 0.01:
                val = compute_delta_s_neg(z)
                assert val < 1.0, f"Not decaying at z={z}: {val}"

    def test_gradient_positive_below_lens(self):
        """Gradient should be positive for z < z_c."""
        grad = compute_gradient(0.5)
        assert grad > 0, f"Gradient not positive below lens: {grad}"

    def test_gradient_negative_above_lens(self):
        """Gradient should be negative for z > z_c."""
        grad = compute_gradient(0.95)
        assert grad < 0, f"Gradient not negative above lens: {grad}"

    def test_gradient_zero_at_lens(self):
        """Gradient should be zero at z_c."""
        grad = compute_gradient(Z_CRITICAL)
        assert abs(grad) < 1e-10, f"Gradient not zero at lens: {grad}"


# =============================================================================
# PHASE CLASSIFICATION TESTS
# =============================================================================

class TestPhaseClassification:
    """Test phase and tier classification."""

    def test_phase_untrue_below_phi_inv(self):
        """Phase should be UNTRUE for z < phi^-1."""
        for z in [0.0, 0.1, 0.3, 0.5, 0.6]:
            phase = get_phase(z)
            assert phase == "UNTRUE", f"z={z} should be UNTRUE, got {phase}"

    def test_phase_paradox_between_phi_inv_and_z_c(self):
        """Phase should be PARADOX for phi^-1 <= z < z_c."""
        for z in [0.618, 0.7, 0.8, 0.85]:
            phase = get_phase(z)
            assert phase == "PARADOX", f"z={z} should be PARADOX, got {phase}"

    def test_phase_true_at_and_above_z_c(self):
        """Phase should be TRUE for z >= z_c."""
        for z in [Z_CRITICAL, 0.9, 0.95, 1.0]:
            phase = get_phase(z)
            assert phase == "TRUE", f"z={z} should be TRUE, got {phase}"

    def test_tier_boundaries(self):
        """Test tier assignment at boundaries."""
        test_cases = [
            (0.05, "t1"),
            (0.15, "t2"),
            (0.3, "t3"),
            (0.5, "t4"),
            (0.7, "t5"),
            (0.85, "t6"),
            (0.91, "t7"),
            (0.95, "t8"),
            (0.99, "t9"),
        ]
        for z, expected_tier in test_cases:
            tier = get_tier(z)
            assert tier == expected_tier, f"z={z} tier={tier}, expected {expected_tier}"

    def test_capability_classes(self):
        """Test capability class assignment."""
        test_cases = [
            (0.05, "reactive"),
            (0.15, "memory"),
            (0.3, "pattern"),
            (0.5, "prediction"),
            (0.7, "self_model"),
            (0.8, "meta"),
            (0.9, "recurse"),
            (0.95, "autopoiesis"),
        ]
        for z, expected_class in test_cases:
            cap = get_capability_class(z)
            assert cap == expected_class, f"z={z} class={cap}, expected {expected_class}"


# =============================================================================
# K-FORMATION TESTS
# =============================================================================

class TestKFormation:
    """Test K-formation criteria."""

    def test_k_formation_all_criteria_met(self):
        """K-formation should pass when all criteria met."""
        result = check_k_formation(kappa=0.94, eta=0.72, R=8)
        assert result is True, "K-formation should pass with kappa=0.94, eta=0.72, R=8"

    def test_k_formation_kappa_too_low(self):
        """K-formation should fail if kappa < 0.92."""
        result = check_k_formation(kappa=0.90, eta=0.72, R=8)
        assert result is False, "K-formation should fail with kappa=0.90"

    def test_k_formation_eta_too_low(self):
        """K-formation should fail if eta <= phi^-1."""
        result = check_k_formation(kappa=0.94, eta=0.5, R=8)
        assert result is False, "K-formation should fail with eta=0.5"

    def test_k_formation_R_too_low(self):
        """K-formation should fail if R < 7."""
        result = check_k_formation(kappa=0.94, eta=0.72, R=5)
        assert result is False, "K-formation should fail with R=5"

    def test_k_formation_boundary_kappa(self):
        """K-formation at boundary kappa=0.92."""
        result = check_k_formation(kappa=0.92, eta=0.72, R=7)
        assert result is True, "K-formation should pass at boundary"

    def test_k_formation_boundary_eta(self):
        """K-formation fails at boundary eta=phi^-1 (requires >)."""
        result = check_k_formation(kappa=0.94, eta=PHI_INV, R=8)
        assert result is False, "K-formation should fail at eta=phi^-1 (needs >)"


# =============================================================================
# FIRMWARE TESTS
# =============================================================================

class TestFirmware:
    """Test firmware simulation functions."""

    def test_map_z_to_rpm_linear(self):
        """Test z to RPM mapping is linear."""
        rpm_0 = map_z_to_rpm(0.0)
        rpm_1 = map_z_to_rpm(1.0)
        rpm_mid = map_z_to_rpm(0.5)

        assert abs(rpm_0 - MIN_RPM) < 1, f"rpm(0) != MIN_RPM"
        assert abs(rpm_1 - MAX_RPM) < 1, f"rpm(1) != MAX_RPM"
        assert abs(rpm_mid - (MIN_RPM + MAX_RPM) / 2) < 1, f"rpm(0.5) not midpoint"

    def test_operator_boundary_always_legal(self):
        """() BOUNDARY should always be legal."""
        state = FirmwareState()
        is_legal, msg = check_n0_legal("()", state)
        assert is_legal, f"BOUNDARY should be legal: {msg}"

    def test_operator_amplify_requires_prior(self):
        """^ AMPLIFY requires prior () or x in history."""
        state = FirmwareState(history=[])
        is_legal, msg = check_n0_legal("^", state)
        assert not is_legal, "AMPLIFY should be illegal without prior"

        state.history = ["()"]
        is_legal, msg = check_n0_legal("^", state)
        assert is_legal, f"AMPLIFY should be legal after BOUNDARY: {msg}"

    def test_operator_fusion_requires_channels(self):
        """x FUSION requires channel_count >= 2."""
        state = FirmwareState(channel_count=1, history=["()"])
        is_legal, msg = check_n0_legal("x", state)
        assert not is_legal, "FUSION should be illegal with 1 channel"

        state.channel_count = 2
        is_legal, msg = check_n0_legal("x", state)
        assert is_legal, f"FUSION should be legal with 2 channels: {msg}"

    def test_control_loop_updates_negentropy(self):
        """Control loop should update negentropy."""
        state = FirmwareState(z=0.5)
        state.update_negentropy()

        expected = physics_delta_s_neg(0.5)
        assert abs(state.delta_s_neg - expected) < 1e-10

    def test_operator_boundary_pulls_toward_lens(self):
        """() BOUNDARY should pull z toward z_c."""
        state = FirmwareState(z=0.5, history=[])
        z_before = state.z

        state = apply_operator_boundary(state)

        # z should have moved toward z_c
        distance_before = abs(z_before - Z_CRITICAL)
        distance_after = abs(state.z - Z_CRITICAL)
        assert distance_after <= distance_before, "BOUNDARY should pull toward lens"


# =============================================================================
# CYBERNETIC METRICS TESTS
# =============================================================================

class TestCyberneticMetrics:
    """Test cybernetic computation functions."""

    def test_ashby_variety_empty(self):
        """Variety of empty list should be 0."""
        result = ashby_variety([])
        assert result == 0.0

    def test_ashby_variety_single_state(self):
        """Variety of single state should be 0."""
        result = ashby_variety([0.5, 0.5, 0.5])
        assert result == 0.0, f"Single state variety should be 0, got {result}"

    def test_ashby_variety_increases_with_diversity(self):
        """Variety should increase with more diverse states."""
        low_var = ashby_variety([0.5, 0.51, 0.52])
        high_var = ashby_variety([0.1, 0.5, 0.9])
        assert high_var > low_var, "More diverse states should have higher variety"

    def test_shannon_capacity_increases_with_snr(self):
        """Shannon capacity should increase with SNR."""
        low_snr = shannon_capacity(1.0, 1.0)
        high_snr = shannon_capacity(10.0, 1.0)
        assert high_snr > low_snr, "Higher SNR should give higher capacity"

    def test_landauer_efficiency_peaks_at_lens(self):
        """Landauer efficiency should peak at z_c."""
        at_lens = landauer_efficiency(Z_CRITICAL)
        away = landauer_efficiency(0.5)
        assert at_lens > away, "Efficiency should peak at lens"
        assert abs(at_lens - 1.0) < 1e-10, "Efficiency at lens should be 1.0"


# =============================================================================
# NEURAL EXTENSION TESTS
# =============================================================================

class TestNeuralExtensions:
    """Test neuroscience extension functions."""

    def test_grid_cell_pattern_bounded(self):
        """Grid cell firing rate should be in [0, 1]."""
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                rate = grid_cell_pattern(float(x), float(y))
                assert 0.0 <= rate <= 1.0, f"Rate out of bounds at ({x},{y}): {rate}"

    def test_neural_band_mapping_round_trip(self):
        """Band to z and back should give same band."""
        for band in ["delta", "theta", "alpha", "beta", "gamma"]:
            z = neural_band_to_z(band)
            recovered_band, _ = z_to_neural_band(z)
            assert recovered_band == band, f"Band mismatch: {band} -> {z} -> {recovered_band}"

    def test_modulation_index_empty(self):
        """Modulation index of empty data should be 0."""
        mi, phase = compute_modulation_index([], [])
        assert mi == 0.0


# =============================================================================
# PROTOCOL TESTS
# =============================================================================

class TestProtocol:
    """Test communication protocol functions."""

    def test_crc16_consistent(self):
        """CRC-16 should be consistent for same data."""
        data = b"test data"
        crc1 = compute_crc16(data)
        crc2 = compute_crc16(data)
        assert crc1 == crc2, "CRC should be consistent"

    def test_crc16_different_for_different_data(self):
        """CRC-16 should differ for different data."""
        crc1 = compute_crc16(b"test1")
        crc2 = compute_crc16(b"test2")
        assert crc1 != crc2, "CRC should differ for different data"

    def test_encode_decode_payload_round_trip(self):
        """Payload encoding should be reversible."""
        original = {"z": 0.5, "count": 42, "name": "test"}
        encoded = encode_payload(original)
        decoded = decode_payload(encoded)

        assert abs(decoded["z"] - 0.5) < 1e-5
        assert decoded["count"] == 42
        assert decoded["name"] == "test"

    def test_command_frame_round_trip(self):
        """Command frame should encode/decode correctly."""
        frame = CommandFrame(
            command=CommandType.SET_Z,
            payload=b'\x00\x00\x00\x3f',  # 0.5 as float
            sequence=1
        )
        encoded = frame.to_bytes()
        decoded = CommandFrame.from_bytes(encoded)

        assert decoded.command == frame.command
        assert decoded.payload == frame.payload
        assert decoded.sequence == frame.sequence


# =============================================================================
# NUCLEAR SPINNER INTEGRATION TESTS
# =============================================================================

class TestNuclearSpinnerIntegration:
    """Integration tests for NuclearSpinner class."""

    def test_initialization(self):
        """Test spinner initializes correctly."""
        spinner = NuclearSpinner()
        result = spinner.initialize()

        assert result is True
        assert spinner.initialized is True
        assert spinner.running is True

    def test_set_z_target(self):
        """Test setting z target."""
        spinner = NuclearSpinner()
        spinner.initialize()

        spinner.set_z_target(0.618)
        assert abs(spinner.state.z_target - 0.618) < 1e-10

    def test_step_updates_state(self):
        """Test stepping updates state."""
        spinner = NuclearSpinner()
        spinner.initialize()

        z_before = spinner.state.z
        spinner.set_z_target(Z_CRITICAL)
        spinner.run_steps(100)

        # z should have moved toward target
        assert spinner.state.z != z_before

    def test_get_metrics_returns_valid(self):
        """Test get_metrics returns valid SpinnerMetrics."""
        spinner = NuclearSpinner()
        spinner.initialize()

        metrics = spinner.get_metrics()

        assert isinstance(metrics, SpinnerMetrics)
        assert 0.0 <= metrics.z <= 1.0
        assert 0.0 <= metrics.delta_s_neg <= 1.0

    def test_apply_operator_boundary(self):
        """Test applying BOUNDARY operator."""
        spinner = NuclearSpinner()
        spinner.initialize()

        success, msg = spinner.apply_operator("()")
        assert success, f"BOUNDARY should succeed: {msg}"
        assert "()" in spinner.state.operator_history

    def test_drive_toward_lens(self):
        """Test driving toward THE LENS."""
        spinner = NuclearSpinner()
        spinner.initialize()

        initial_z = spinner.state.z
        final_z = spinner.drive_toward_lens(n_steps=50)

        # Should move toward Z_CRITICAL
        initial_distance = abs(initial_z - Z_CRITICAL)
        final_distance = abs(final_z - Z_CRITICAL)
        assert final_distance <= initial_distance, "Should move toward lens"

    def test_verify_spin_zc_identity(self):
        """Test spin-z_c identity verification."""
        spinner = NuclearSpinner()
        spinner.initialize()

        result = spinner.verify_spin_zc_identity()
        assert result is True, "Spin-z_c identity should hold"

    def test_pulse_affects_state(self):
        """Test that sending a pulse affects state."""
        spinner = NuclearSpinner()
        spinner.initialize()

        z_before = spinner.state.z
        spinner.send_pulse(amplitude=1.0, phase=0.0, duration_us=1000)

        # Pulse should have some effect (may be small)
        # Just verify no errors and state is valid
        assert 0.0 <= spinner.state.z <= 1.0

    def test_metrics_history_tracked(self):
        """Test that metrics history is tracked."""
        spinner = NuclearSpinner()
        spinner.initialize()

        for _ in range(5):
            spinner.step()
            spinner.get_metrics()

        history = spinner.get_metrics_history(n=10)
        assert len(history) >= 5, "Should have tracked metrics"

    def test_close_cleanup(self):
        """Test closing spinner cleans up."""
        spinner = NuclearSpinner()
        spinner.initialize()
        spinner.close()

        assert spinner.running is False
        assert spinner.initialized is False


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
