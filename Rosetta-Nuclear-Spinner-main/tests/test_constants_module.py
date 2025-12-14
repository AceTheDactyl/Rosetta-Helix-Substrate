"""
Test suite for Quantum APL constants module.
Validates all constants, helper functions, and architectural constraints.

Per CONSTANTS_RESEARCH.md:
- Verifies Z_CRITICAL = √3/2 (THE LENS)
- Validates TRIAD separation from geometry
- Tests phase detection and time harmonics
- Verifies hex prism formulae (constants presence)
- Tests K-formation criteria and modeling parameters
"""

import math
from quantum_apl_python.constants import *  # noqa: F401,F403


def test_critical_lens_constant():
    expected = math.sqrt(3) / 2
    assert abs(Z_CRITICAL - expected) < 1e-12


def test_triad_constants():
    assert TRIAD_HIGH == 0.85
    assert TRIAD_LOW == 0.82
    assert TRIAD_T6 == 0.83
    assert TRIAD_LOW < TRIAD_T6 < TRIAD_HIGH
    assert TRIAD_T6 < Z_CRITICAL


def test_phase_boundaries_and_helpers():
    assert Z_ABSENCE_MAX == 0.857
    assert Z_LENS_MIN == 0.857
    assert Z_LENS_MAX == 0.877
    assert Z_PRESENCE_MIN == 0.877
    assert Z_LENS_MIN <= Z_CRITICAL <= Z_LENS_MAX
    assert get_phase(0.5) == 'ABSENCE'
    assert get_phase(Z_CRITICAL) == 'THE_LENS'
    assert get_phase(0.88) == 'PRESENCE'
    assert is_critical(Z_CRITICAL)
    assert is_in_lens(Z_CRITICAL)
    assert get_distance_to_critical(0.9) > 0


def test_time_harmonics():
    assert T1_MAX == 0.1 and T2_MAX == 0.2 and T3_MAX == 0.4
    assert T4_MAX == 0.6 and T5_MAX == 0.75 and T7_MAX == 0.92 and T8_MAX == 0.97
    assert get_time_harmonic(0.05) == 't1'
    assert get_time_harmonic(0.15) == 't2'
    assert get_time_harmonic(0.3) == 't3'
    assert get_time_harmonic(0.5) == 't4'
    assert get_time_harmonic(0.7) == 't5'
    assert get_time_harmonic(0.8) == 't6'
    assert get_time_harmonic(0.9) == 't7'
    assert get_time_harmonic(0.95) == 't8'
    assert get_time_harmonic(0.99) == 't9'
    # TRIAD gate override
    assert get_time_harmonic(0.84, t6_gate=TRIAD_T6) == 't7'
    assert get_time_harmonic(0.82, t6_gate=TRIAD_T6) == 't6'


def test_geometry_constants_present():
    # GEOM_SIGMA (SIGMA alias) defaults to LENS_SIGMA when unset
    assert abs(SIGMA - LENS_SIGMA) < 1e-12
    assert R_MAX == 0.85
    assert BETA == 0.25
    assert H_MIN == 0.12
    assert GAMMA == 0.18
    assert PHI_BASE == 0.0
    assert abs(ETA - math.pi / 12) < 1e-12


def test_k_formation_and_sacred_constants():
    phi_expected = (1 + math.sqrt(5)) / 2
    assert abs(PHI - phi_expected) < 1e-9
    assert abs(PHI_INV - 1 / PHI) < 1e-9
    assert KAPPA_S == 0.920
    assert abs(Q_KAPPA - 0.3514087324) < 1e-9
    assert abs(LAMBDA - 7.7160493827) < 1e-9
    assert KAPPA_MIN == KAPPA_S
    assert ETA_MIN == PHI_INV
    assert R_MIN == 7
    assert check_k_formation(0.94, 0.72, 8)
    assert not check_k_formation(0.90, 0.72, 8)
    assert not check_k_formation(0.94, 0.61, 8)
    assert not check_k_formation(0.94, 0.72, 6)


def test_pump_profiles_and_engine_params():
    assert PumpProfile.GENTLE['gain'] == 0.08 and PumpProfile.GENTLE['sigma'] == 0.16
    assert PumpProfile.BALANCED['gain'] == 0.12 and PumpProfile.BALANCED['sigma'] == 0.12
    assert PumpProfile.AGGRESSIVE['gain'] == 0.18 and PumpProfile.AGGRESSIVE['sigma'] == 0.10
    assert PUMP_DEFAULT_TARGET == Z_CRITICAL
    assert Z_BIAS_GAIN == 0.05 and Z_BIAS_SIGMA == 0.18
    assert abs(OMEGA - 2 * math.pi * 0.1) < 1e-12
    assert COUPLING_G == 0.05
    assert GAMMA_1 == 0.01 and GAMMA_2 == 0.02 and GAMMA_3 == 0.005 and GAMMA_4 == 0.015


def test_operator_weighting():
    assert OPERATOR_PREFERRED_WEIGHT == 1.3
    assert OPERATOR_DEFAULT_WEIGHT == 0.85
    for truth in ('TRUE', 'UNTRUE', 'PARADOX'):
        for op in ('()', '×', '^', '÷', '+', '-'):
            assert op in TRUTH_BIAS[truth]
