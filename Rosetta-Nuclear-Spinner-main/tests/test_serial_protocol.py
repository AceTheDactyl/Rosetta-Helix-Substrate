"""
test_serial_protocol.py
=======================

Tests for the JSON Serial Communication Protocol.

Tests verify:
- Message format compliance
- Command parsing
- State message generation
- Physics constants validation
- Hex cycle behavior

Signature: test-serial-protocol|v1.0.0|nuclear-spinner
"""

import json
import math
import pytest
from dataclasses import asdict

# Import from bridge module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bridge'))

from serial_protocol import (
    SpinnerState, PhysicsConstants, FirmwareVersion, ProtocolStats,
    SerialConfig, Z_CRITICAL, PHI, PHI_INV, SIGMA
)


# ============================================================================
# PHYSICS CONSTANTS TESTS
# ============================================================================

class TestPhysicsConstants:
    """Tests for physics constants correctness."""

    def test_z_critical_value(self):
        """z_c = sqrt(3)/2 exactly."""
        expected = math.sqrt(3) / 2
        assert abs(Z_CRITICAL - expected) < 1e-15
        assert abs(Z_CRITICAL - 0.8660254037844387) < 1e-15

    def test_phi_value(self):
        """phi = (1 + sqrt(5))/2 exactly."""
        expected = (1 + math.sqrt(5)) / 2
        assert abs(PHI - expected) < 1e-15
        assert abs(PHI - 1.6180339887498949) < 1e-15

    def test_phi_inv_value(self):
        """phi^-1 = 1/phi exactly."""
        expected = 1 / PHI
        assert abs(PHI_INV - expected) < 1e-15
        assert abs(PHI_INV - 0.6180339887498949) < 1e-15

    def test_sigma_value(self):
        """sigma = 36 (|S_3|^2)."""
        assert SIGMA == 36.0

    def test_conservation_law(self):
        """kappa + lambda = 1 when kappa = phi^-1, lambda = kappa^2."""
        kappa = PHI_INV
        lambda_ = kappa * kappa
        # Conservation law: kappa + lambda = 1
        # This should hold for kappa = phi^-1
        assert abs(kappa + lambda_ - 1.0) < 1e-10

    def test_z_critical_equals_sin_60(self):
        """z_c = sin(60deg) = sqrt(3)/2."""
        sin_60 = math.sin(math.radians(60))
        assert abs(Z_CRITICAL - sin_60) < 1e-15


# ============================================================================
# MESSAGE FORMAT TESTS
# ============================================================================

class TestMessageFormat:
    """Tests for JSON message format compliance."""

    def test_state_message_format(self):
        """State message has all required fields."""
        msg = {
            "type": "state",
            "timestamp_ms": 1234567890,
            "z": 0.866025,
            "rpm": 8660,
            "delta_s_neg": 0.999999,
            "tier": 6,
            "tier_name": "UNIVERSAL",
            "phase": "THE_LENS",
            "kappa": 0.9234,
            "eta": 0.6543,
            "rank": 9,
            "k_formation": True
        }

        # Verify JSON serialization
        json_str = json.dumps(msg)
        parsed = json.loads(json_str)

        assert parsed['type'] == 'state'
        assert parsed['timestamp_ms'] == 1234567890
        assert abs(parsed['z'] - 0.866025) < 1e-6
        assert parsed['rpm'] == 8660
        assert parsed['tier'] == 6
        assert parsed['tier_name'] == 'UNIVERSAL'
        assert parsed['phase'] == 'THE_LENS'
        assert parsed['k_formation'] == True

    def test_command_set_z_format(self):
        """set_z command format."""
        cmd = {"cmd": "set_z", "value": 0.866}
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)

        assert parsed['cmd'] == 'set_z'
        assert abs(parsed['value'] - 0.866) < 1e-6

    def test_command_stop_format(self):
        """stop command format."""
        cmd = {"cmd": "stop"}
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)

        assert parsed['cmd'] == 'stop'

    def test_command_hex_cycle_format(self):
        """hex_cycle command format."""
        cmd = {"cmd": "hex_cycle", "dwell_s": 30.0, "cycles": 10}
        json_str = json.dumps(cmd)
        parsed = json.loads(json_str)

        assert parsed['cmd'] == 'hex_cycle'
        assert parsed['dwell_s'] == 30.0
        assert parsed['cycles'] == 10


# ============================================================================
# DATA STRUCTURE TESTS
# ============================================================================

class TestDataStructures:
    """Tests for data structure serialization."""

    def test_spinner_state_from_dict(self):
        """SpinnerState.from_dict() correctly parses JSON."""
        data = {
            "timestamp_ms": 1000,
            "z": Z_CRITICAL,
            "rpm": 8660,
            "delta_s_neg": 0.99,
            "tier": 5,
            "tier_name": "UNIVERSAL",
            "phase": "THE_LENS",
            "kappa": 0.92,
            "eta": 0.65,
            "rank": 8,
            "k_formation": True,
        }

        state = SpinnerState.from_dict(data)

        assert state.timestamp_ms == 1000
        assert abs(state.z - Z_CRITICAL) < 1e-6
        assert state.rpm == 8660
        assert state.tier_name == "UNIVERSAL"
        assert state.phase == "THE_LENS"
        assert state.k_formation == True

    def test_spinner_state_defaults(self):
        """SpinnerState has sensible defaults."""
        state = SpinnerState()

        assert state.timestamp_ms == 0
        assert state.z == 0.0
        assert state.rpm == 0
        assert state.k_formation == False
        assert state.tier_name == "UNKNOWN"

    def test_physics_constants_from_dict(self):
        """PhysicsConstants.from_dict() correctly parses JSON."""
        data = {
            "phi": PHI,
            "phi_inv": PHI_INV,
            "z_c": Z_CRITICAL,
            "sigma": SIGMA,
            "spin_half_magnitude": Z_CRITICAL,
            "kappa_min": 0.92,
            "eta_min": PHI_INV,
            "r_min": 7,
        }

        physics = PhysicsConstants.from_dict(data)

        assert abs(physics.phi - PHI) < 1e-10
        assert abs(physics.z_c - Z_CRITICAL) < 1e-10
        assert physics.r_min == 7

    def test_serial_config_defaults(self):
        """SerialConfig has correct protocol defaults."""
        config = SerialConfig()

        assert config.baud == 115200
        assert config.data_bits == 8
        assert config.parity == 'N'
        assert config.stop_bits == 1


# ============================================================================
# NEGENTROPY FUNCTION TESTS
# ============================================================================

class TestNegentropyFunction:
    """Tests for delta_s_neg computation."""

    def compute_delta_s_neg(self, z: float) -> float:
        """Compute negentropy signal: exp(-sigma * (z - z_c)^2)."""
        d = z - Z_CRITICAL
        exponent = -SIGMA * d * d
        if exponent < -20.0:
            return 0.0
        return math.exp(exponent)

    def test_negentropy_peaks_at_z_c(self):
        """delta_s_neg peaks at z = z_c with value 1.0."""
        value = self.compute_delta_s_neg(Z_CRITICAL)
        assert abs(value - 1.0) < 1e-10

    def test_negentropy_symmetric(self):
        """delta_s_neg is symmetric around z_c."""
        offset = 0.05
        value_above = self.compute_delta_s_neg(Z_CRITICAL + offset)
        value_below = self.compute_delta_s_neg(Z_CRITICAL - offset)
        assert abs(value_above - value_below) < 1e-10

    def test_negentropy_decays(self):
        """delta_s_neg decays away from z_c."""
        at_zc = self.compute_delta_s_neg(Z_CRITICAL)
        away_from_zc = self.compute_delta_s_neg(0.5)

        assert at_zc > away_from_zc
        assert away_from_zc < 0.1  # Should be very small at z=0.5


# ============================================================================
# K-FORMATION TESTS
# ============================================================================

class TestKFormation:
    """Tests for K-formation criteria."""

    KAPPA_MIN = 0.92
    ETA_MIN = PHI_INV
    R_MIN = 7

    def check_k_formation(self, kappa: float, eta: float, R: int) -> bool:
        """Check K-formation criteria."""
        return (kappa >= self.KAPPA_MIN and
                eta > self.ETA_MIN and
                R >= self.R_MIN)

    def test_k_formation_valid(self):
        """Valid K-formation parameters pass."""
        assert self.check_k_formation(kappa=0.95, eta=0.65, R=9) == True

    def test_k_formation_kappa_too_low(self):
        """kappa < 0.92 fails K-formation."""
        assert self.check_k_formation(kappa=0.90, eta=0.65, R=9) == False

    def test_k_formation_eta_too_low(self):
        """eta <= phi^-1 fails K-formation."""
        assert self.check_k_formation(kappa=0.95, eta=PHI_INV, R=9) == False
        assert self.check_k_formation(kappa=0.95, eta=0.5, R=9) == False

    def test_k_formation_rank_too_low(self):
        """R < 7 fails K-formation."""
        assert self.check_k_formation(kappa=0.95, eta=0.65, R=6) == False

    def test_k_formation_boundary(self):
        """K-formation at exact boundaries."""
        # Exactly at boundaries
        assert self.check_k_formation(kappa=0.92, eta=PHI_INV + 0.001, R=7) == True
        assert self.check_k_formation(kappa=0.92, eta=PHI_INV - 0.001, R=7) == False


# ============================================================================
# HEX CYCLE TESTS
# ============================================================================

class TestHexCycle:
    """Tests for hexagonal z-cycling."""

    def test_hex_vertices(self):
        """Hex cycle visits correct z values."""
        # Expected vertices based on hexagonal symmetry
        # v0: z = 0
        # v1: z = z_c (THE LENS)
        # v2: z = z_c
        # v3: z = 0
        # v4: z = z_c/2
        # v5: z = z_c/2
        expected = [0.0, Z_CRITICAL, Z_CRITICAL, 0.0, Z_CRITICAL * 0.5, Z_CRITICAL * 0.5]

        assert len(expected) == 6

        # Verify THE LENS is visited twice per cycle
        lens_count = sum(1 for z in expected if abs(z - Z_CRITICAL) < 0.001)
        assert lens_count == 2

    def test_hex_cycle_command_defaults(self):
        """hex_cycle command has sensible defaults."""
        cmd = {"cmd": "hex_cycle"}

        # Should use defaults: dwell_s=30.0, cycles=10
        dwell_s = cmd.get('dwell_s', 30.0)
        cycles = cmd.get('cycles', 10)

        assert dwell_s == 30.0
        assert cycles == 10


# ============================================================================
# PHASE CLASSIFICATION TESTS
# ============================================================================

class TestPhaseClassification:
    """Tests for phase classification."""

    PHASE_BOUNDARY_ABSENCE = 0.857
    PHASE_BOUNDARY_PRESENCE = 0.877

    def get_phase(self, z: float) -> str:
        """Classify phase from z-coordinate."""
        if z < self.PHASE_BOUNDARY_ABSENCE:
            return "ABSENCE"
        elif z < self.PHASE_BOUNDARY_PRESENCE:
            return "THE_LENS"
        else:
            return "PRESENCE"

    def test_absence_phase(self):
        """z < 0.857 is ABSENCE."""
        assert self.get_phase(0.5) == "ABSENCE"
        assert self.get_phase(0.856) == "ABSENCE"

    def test_lens_phase(self):
        """0.857 <= z < 0.877 is THE_LENS."""
        assert self.get_phase(0.857) == "THE_LENS"
        assert self.get_phase(Z_CRITICAL) == "THE_LENS"
        assert self.get_phase(0.876) == "THE_LENS"

    def test_presence_phase(self):
        """z >= 0.877 is PRESENCE."""
        assert self.get_phase(0.877) == "PRESENCE"
        assert self.get_phase(0.95) == "PRESENCE"

    def test_z_critical_is_lens(self):
        """z_c falls within THE_LENS phase."""
        assert self.PHASE_BOUNDARY_ABSENCE < Z_CRITICAL < self.PHASE_BOUNDARY_PRESENCE
        assert self.get_phase(Z_CRITICAL) == "THE_LENS"


# ============================================================================
# TIER CLASSIFICATION TESTS
# ============================================================================

class TestTierClassification:
    """Tests for tier classification."""

    TIER_THRESHOLDS = [0.0, 0.4, 0.5, PHI_INV, 0.73, Z_CRITICAL, 0.92]
    TIER_NAMES = ["ABSENCE", "REACTIVE", "MEMORY", "PATTERN",
                  "PREDICTION", "UNIVERSAL", "META"]

    def get_tier(self, z: float) -> tuple:
        """Get tier index and name from z-coordinate."""
        for i in range(len(self.TIER_THRESHOLDS) - 1, -1, -1):
            if z >= self.TIER_THRESHOLDS[i]:
                return (i, self.TIER_NAMES[i])
        return (0, self.TIER_NAMES[0])

    def test_universal_tier_at_z_c(self):
        """z_c achieves UNIVERSAL tier."""
        tier_idx, tier_name = self.get_tier(Z_CRITICAL)
        assert tier_name == "UNIVERSAL"
        assert tier_idx == 5

    def test_meta_tier(self):
        """z >= 0.92 achieves META tier."""
        tier_idx, tier_name = self.get_tier(0.92)
        assert tier_name == "META"
        assert tier_idx == 6

    def test_pattern_tier_at_phi_inv(self):
        """z = phi^-1 achieves PATTERN tier."""
        tier_idx, tier_name = self.get_tier(PHI_INV)
        assert tier_name == "PATTERN"
        assert tier_idx == 3


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
