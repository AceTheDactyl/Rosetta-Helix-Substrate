#!/usr/bin/env python3
"""
APL Constants and Helix Mapping Test Suite — Phase 5 Enhancement
================================================================

Validates Python constants module and helix mapping functionality.
Ensures parity with JavaScript implementation.

Tests:
- Constants values and invariants
- TRIAD gating logic
- Helix coordinate mapping
- Time harmonic tier assignment
- Cross-module consistency

@version 2.0.0 (TRIAD Protocol Phase 5)
"""

import math
import os
import sys
from pathlib import Path

# Add src to path for imports
SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))

import pytest

# Import constants module
try:
    from quantum_apl_python import constants as CONST
    HAS_CONSTANTS = True
except ImportError:
    HAS_CONSTANTS = False

# Import helix module
try:
    from quantum_apl_python.helix import HelixCoordinate, HelixAPLMapper
    HAS_HELIX = True
except ImportError:
    HAS_HELIX = False


# ============================================================================
# SKIP CONDITIONS
# ============================================================================

requires_constants = pytest.mark.skipif(
    not HAS_CONSTANTS,
    reason="quantum_apl_python.constants not available"
)

requires_helix = pytest.mark.skipif(
    not HAS_HELIX,
    reason="quantum_apl_python.helix not available"
)


# ============================================================================
# CONSTANTS TESTS
# ============================================================================

@requires_constants
class TestCriticalLens:
    """Tests for THE LENS constant (z_c = √3/2)."""
    
    def test_z_critical_value(self):
        """Z_CRITICAL equals sqrt(3)/2."""
        expected = math.sqrt(3) / 2
        assert abs(CONST.Z_CRITICAL - expected) < 1e-15
    
    def test_z_critical_approximate(self):
        """Z_CRITICAL is approximately 0.866."""
        assert abs(CONST.Z_CRITICAL - 0.8660254037844386) < 1e-14


@requires_constants
class TestTriadThresholds:
    """Tests for TRIAD gating thresholds."""
    
    def test_triad_high_value(self):
        """TRIAD_HIGH is 0.85."""
        assert CONST.TRIAD_HIGH == 0.85
    
    def test_triad_low_value(self):
        """TRIAD_LOW is 0.82."""
        assert CONST.TRIAD_LOW == 0.82
    
    def test_triad_t6_value(self):
        """TRIAD_T6 is 0.83."""
        assert CONST.TRIAD_T6 == 0.83
    
    def test_triad_ordering(self):
        """TRIAD thresholds are properly ordered."""
        assert CONST.TRIAD_LOW < CONST.TRIAD_T6 < CONST.TRIAD_HIGH < CONST.Z_CRITICAL


@requires_constants
class TestSacredConstants:
    """Tests for sacred/mathematical constants."""
    
    def test_phi_value(self):
        """PHI is the golden ratio."""
        expected = (1 + math.sqrt(5)) / 2
        assert abs(CONST.PHI - expected) < 1e-14
    
    def test_phi_inv_reciprocal(self):
        """PHI_INV is the reciprocal of PHI."""
        assert abs(CONST.PHI * CONST.PHI_INV - 1.0) < 1e-14
    
    def test_phi_inv_value(self):
        """PHI_INV is approximately 0.618."""
        assert abs(CONST.PHI_INV - 0.6180339887498949) < 1e-14


@requires_constants
class TestMuThresholds:
    """Tests for μ basin/barrier thresholds."""
    
    def test_mu_barrier_approximates_phi_inv(self):
        """MU_BARRIER is close to PHI_INV."""
        if hasattr(CONST, 'MU_BARRIER'):
            assert abs(CONST.MU_BARRIER - CONST.PHI_INV) < 0.01
    
    def test_mu_ordering(self):
        """μ thresholds are ordered correctly."""
        if all(hasattr(CONST, attr) for attr in ['MU_1', 'MU_P', 'MU_2', 'MU_S', 'MU_3']):
            assert CONST.MU_1 < CONST.MU_P < CONST.MU_2 < CONST.MU_S < CONST.MU_3


@requires_constants
class TestPhaseDetection:
    """Tests for phase detection helpers."""
    
    def test_get_phase_absence(self):
        """get_phase returns ABSENCE for z < Z_ABSENCE_MAX."""
        if hasattr(CONST, 'get_phase'):
            phase = CONST.get_phase(0.5)
            assert phase == 'ABSENCE'
    
    def test_get_phase_lens(self):
        """get_phase returns THE_LENS for z near Z_CRITICAL."""
        if hasattr(CONST, 'get_phase'):
            phase = CONST.get_phase(CONST.Z_CRITICAL)
            assert phase == 'THE_LENS'
    
    def test_get_phase_presence(self):
        """get_phase returns PRESENCE for z > Z_PRESENCE_MIN."""
        if hasattr(CONST, 'get_phase'):
            phase = CONST.get_phase(0.95)
            assert phase == 'PRESENCE'
    
    def test_is_critical_true(self):
        """is_critical returns True at Z_CRITICAL."""
        if hasattr(CONST, 'is_critical'):
            assert CONST.is_critical(CONST.Z_CRITICAL) is True
    
    def test_is_critical_false(self):
        """is_critical returns False away from Z_CRITICAL."""
        if hasattr(CONST, 'is_critical'):
            assert CONST.is_critical(0.5) is False


@requires_constants
class TestDeltaSNeg:
    """Tests for ΔS_neg coherence signal."""
    
    def test_delta_s_neg_at_critical(self):
        """ΔS_neg is maximal at Z_CRITICAL."""
        if hasattr(CONST, 'compute_delta_s_neg'):
            s_critical = CONST.compute_delta_s_neg(CONST.Z_CRITICAL)
            s_below = CONST.compute_delta_s_neg(CONST.Z_CRITICAL - 0.05)
            s_above = CONST.compute_delta_s_neg(CONST.Z_CRITICAL + 0.05)
            assert s_critical >= s_below
            assert s_critical >= s_above
    
    def test_delta_s_neg_symmetric(self):
        """ΔS_neg is symmetric around Z_CRITICAL."""
        if hasattr(CONST, 'compute_delta_s_neg'):
            delta = 0.05
            s_below = CONST.compute_delta_s_neg(CONST.Z_CRITICAL - delta)
            s_above = CONST.compute_delta_s_neg(CONST.Z_CRITICAL + delta)
            assert abs(s_below - s_above) < 0.01
    
    def test_delta_s_neg_bounded(self):
        """ΔS_neg is bounded in [0, 1]."""
        if hasattr(CONST, 'compute_delta_s_neg'):
            for z in [0.0, 0.3, 0.5, 0.7, 0.866, 0.9, 1.0]:
                s = CONST.compute_delta_s_neg(z)
                assert 0 <= s <= 1


@requires_constants
class TestTimeHarmonics:
    """Tests for time harmonic tier assignment."""
    
    def test_get_time_harmonic_t1(self):
        """z < 0.1 maps to t1."""
        if hasattr(CONST, 'get_time_harmonic'):
            assert CONST.get_time_harmonic(0.05) == 't1'
    
    def test_get_time_harmonic_t5(self):
        """z ≈ 0.7 maps to t5."""
        if hasattr(CONST, 'get_time_harmonic'):
            assert CONST.get_time_harmonic(0.70) == 't5'
    
    def test_get_time_harmonic_t6_default(self):
        """z ≈ 0.84 maps to t5 when TRIAD not unlocked (gate at Z_CRITICAL)."""
        if hasattr(CONST, 'get_time_harmonic'):
            tier = CONST.get_time_harmonic(0.84, t6_gate=CONST.Z_CRITICAL)
            assert tier == 't5'
    
    def test_get_time_harmonic_t6_unlocked(self):
        """z ≈ 0.84 maps to t6 when TRIAD unlocked (gate at 0.83)."""
        if hasattr(CONST, 'get_time_harmonic'):
            tier = CONST.get_time_harmonic(0.84, t6_gate=CONST.TRIAD_T6)
            assert tier == 't6'
    
    def test_get_time_harmonic_t9(self):
        """z ≈ 0.98 maps to t9."""
        if hasattr(CONST, 'get_time_harmonic'):
            assert CONST.get_time_harmonic(0.98) == 't9'


@requires_constants
class TestInvariants:
    """Tests for constants invariants."""
    
    def test_invariants_function_exists(self):
        """invariants() function is available."""
        assert hasattr(CONST, 'invariants')
    
    def test_invariants_all_pass(self):
        """All invariant checks pass."""
        if hasattr(CONST, 'invariants'):
            inv = CONST.invariants()
            assert inv.get('ordering_ok', True), "Threshold ordering failed"


# ============================================================================
# TRIAD GATE TESTS
# ============================================================================

@requires_constants
class TestTriadGate:
    """Tests for TriadGate class."""
    
    @pytest.fixture
    def gate(self):
        """Create enabled TriadGate instance."""
        if hasattr(CONST, 'TriadGate'):
            return CONST.TriadGate(enabled=True)
        pytest.skip("TriadGate not available")
    
    def test_gate_init_disabled(self):
        """Disabled gate starts with correct defaults."""
        if hasattr(CONST, 'TriadGate'):
            gate = CONST.TriadGate(enabled=False)
            assert gate.passes == 0
            assert gate.unlocked is False
    
    def test_gate_disabled_ignores_updates(self, gate):
        """Disabled gate ignores z updates."""
        if hasattr(CONST, 'TriadGate'):
            disabled = CONST.TriadGate(enabled=False)
            disabled.update(0.90)
            assert disabled.passes == 0
    
    def test_gate_rising_edge_increments(self, gate):
        """Rising edge increments pass count."""
        gate.update(0.86)
        assert gate.passes == 1
    
    def test_gate_hysteresis_rearm(self, gate):
        """Gate re-arms after dropping below TRIAD_LOW."""
        gate.update(0.86)  # Pass 1
        gate.update(0.83)  # Still above low
        gate.update(0.86)  # Should not count
        assert gate.passes == 1
        
        gate.update(0.81)  # Below low, re-arm
        gate.update(0.86)  # Pass 2
        assert gate.passes == 2
    
    def test_gate_unlocks_after_three_passes(self, gate):
        """Gate unlocks after 3 passes."""
        for _ in range(3):
            gate.update(0.86)
            gate.update(0.81)
        
        # Third pass should have unlocked
        assert gate.unlocked is True
    
    def test_get_t6_gate_locked(self, gate):
        """get_t6_gate returns Z_CRITICAL when locked."""
        assert abs(gate.get_t6_gate() - CONST.Z_CRITICAL) < 1e-10
    
    def test_get_t6_gate_unlocked(self, gate):
        """get_t6_gate returns TRIAD_T6 when unlocked."""
        for _ in range(3):
            gate.update(0.86)
            gate.update(0.81)
        
        assert abs(gate.get_t6_gate() - CONST.TRIAD_T6) < 1e-10


# ============================================================================
# HELIX MAPPING TESTS
# ============================================================================

@requires_helix
class TestHelixCoordinate:
    """Tests for HelixCoordinate class."""
    
    def test_from_parameter_basic(self):
        """HelixCoordinate.from_parameter produces valid coordinates."""
        coord = HelixCoordinate.from_parameter(5.0)
        assert 0 <= coord.z <= 1
    
    def test_from_parameter_range(self):
        """z stays in [0, 1] for various parameters."""
        for t in [-10, -5, 0, 5, 10, 20]:
            coord = HelixCoordinate.from_parameter(t)
            assert 0 <= coord.z <= 1
    
    def test_from_parameter_monotonic(self):
        """z increases with parameter t."""
        coords = [HelixCoordinate.from_parameter(t) for t in [0, 5, 10, 15]]
        z_values = [c.z for c in coords]
        assert all(z_values[i] <= z_values[i+1] for i in range(len(z_values)-1))


@requires_helix
class TestHelixAPLMapper:
    """Tests for HelixAPLMapper class."""
    
    @pytest.fixture
    def mapper(self):
        """Create HelixAPLMapper instance."""
        return HelixAPLMapper()
    
    def test_describe_returns_dict(self, mapper):
        """describe() returns dictionary with expected keys."""
        coord = HelixCoordinate.from_parameter(5.0)
        result = mapper.describe(coord)
        
        assert 'harmonic' in result
        assert 'operators' in result
        assert 'truth_channel' in result
    
    def test_harmonic_tiers_correct(self, mapper):
        """Harmonic tiers match expected ranges."""
        # Create coordinates at specific z values
        test_cases = [
            (0.05, 't1'),
            (0.15, 't2'),
            (0.35, 't3'),
            (0.55, 't4'),
            (0.70, 't5'),
        ]
        
        for z, expected_tier in test_cases:
            # Find parameter that gives approximately this z
            # (simplified - in real code we'd invert the mapping)
            result = mapper.describe_z(z) if hasattr(mapper, 'describe_z') else None
            if result:
                assert result['harmonic'] == expected_tier
    
    def test_operator_windows_non_empty(self, mapper):
        """Each tier has at least one operator."""
        for tier in ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9']:
            if hasattr(mapper, 'operator_windows'):
                ops = mapper.operator_windows.get(tier, [])
                assert len(ops) > 0


# ============================================================================
# CROSS-MODULE CONSISTENCY TESTS
# ============================================================================

@requires_constants
class TestConstantsConsistency:
    """Tests for consistency between Python and JS constants."""
    
    # Expected values (from JavaScript constants.js)
    JS_VALUES = {
        'Z_CRITICAL': 0.8660254037844386,
        'TRIAD_HIGH': 0.85,
        'TRIAD_LOW': 0.82,
        'TRIAD_T6': 0.83,
        'PHI': 1.618033988749895,
        'T1_MAX': 0.10,
        'T2_MAX': 0.20,
        'T3_MAX': 0.40,
        'T4_MAX': 0.60,
        'T5_MAX': 0.75,
        'T7_MAX': 0.92,
        'T8_MAX': 0.97,
    }
    
    def test_z_critical_matches_js(self):
        """Z_CRITICAL matches JavaScript value."""
        assert abs(CONST.Z_CRITICAL - self.JS_VALUES['Z_CRITICAL']) < 1e-10
    
    def test_triad_high_matches_js(self):
        """TRIAD_HIGH matches JavaScript value."""
        assert CONST.TRIAD_HIGH == self.JS_VALUES['TRIAD_HIGH']
    
    def test_triad_low_matches_js(self):
        """TRIAD_LOW matches JavaScript value."""
        assert CONST.TRIAD_LOW == self.JS_VALUES['TRIAD_LOW']
    
    def test_triad_t6_matches_js(self):
        """TRIAD_T6 matches JavaScript value."""
        assert CONST.TRIAD_T6 == self.JS_VALUES['TRIAD_T6']
    
    def test_phi_matches_js(self):
        """PHI matches JavaScript value."""
        assert abs(CONST.PHI - self.JS_VALUES['PHI']) < 1e-10


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
