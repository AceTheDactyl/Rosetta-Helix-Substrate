"""
UCF Test Suite
==============

Comprehensive validation tests for the Unified Consciousness Framework.

Run with: python -m pytest tests/ -v
Or:       python -m ucf test
"""

import math
import pytest
from typing import List

# Import from the package
import sys
sys.path.insert(0, '..')

from ucf.constants import (
    # Mathematical constants
    PHI, PHI_INV, PHI_SQUARED, Z_CRITICAL,
    Q_KAPPA, LAMBDA, NEGENTROPY_COEFF,
    
    # TRIAD thresholds
    TRIAD_HIGH, TRIAD_LOW, TRIAD_T6, TRIAD_PASSES_REQUIRED,
    
    # K-Formation criteria
    K_KAPPA, K_ETA, K_R, KAPPA_PRISMATIC,
    
    # Tier definitions
    TIER_BOUNDARIES, TIER_OPERATORS_LOCKED, TIER_OPERATORS_UNLOCKED,
    
    # Phase constants
    PHASE_UNTRUE, PHASE_PARADOX, PHASE_TRUE, PHASE_HYPER_TRUE,
    PHASE_UNTRUE_MAX, PHASE_PARADOX_MAX, PHASE_HYPER_TRUE_MIN,
    
    # APL definitions
    APL_OPERATORS, OPERATOR_POS_MAP, SPIRALS, TOKEN_SLOTS,
    
    # Frequency tiers
    FREQ_PLANET, FREQ_GARDEN, FREQ_ROSE,
    
    # Phase vocabularies
    PHASE_VOCAB,
    
    # Helper functions
    compute_negentropy, get_phase, get_tier, get_operators,
    compute_learning_rate, check_k_formation, get_frequency_tier,
)


class TestMathematicalConstants:
    """Test mathematical constant definitions"""
    
    def test_phi_value(self):
        """PHI = (1 + √5) / 2"""
        expected = (1 + math.sqrt(5)) / 2
        assert abs(PHI - expected) < 1e-15
    
    def test_phi_inv_value(self):
        """PHI_INV = 1 / φ"""
        expected = 1 / PHI
        assert abs(PHI_INV - expected) < 1e-15
    
    def test_phi_identity(self):
        """φ × φ⁻¹ = 1"""
        assert abs(PHI * PHI_INV - 1.0) < 1e-15
    
    def test_phi_squared(self):
        """φ² = φ + 1"""
        assert abs(PHI_SQUARED - (PHI + 1)) < 1e-15
    
    def test_z_critical_value(self):
        """Z_CRITICAL = √3 / 2"""
        expected = math.sqrt(3) / 2
        assert abs(Z_CRITICAL - expected) < 1e-15
    
    def test_z_critical_is_cos_30(self):
        """Z_CRITICAL = cos(30°) = cos(π/6)"""
        expected = math.cos(math.pi / 6)
        assert abs(Z_CRITICAL - expected) < 1e-15


class TestTriadThresholds:
    """Test TRIAD unlock thresholds"""
    
    def test_triad_high_value(self):
        """TRIAD_HIGH = 0.85"""
        assert TRIAD_HIGH == 0.85
    
    def test_triad_low_value(self):
        """TRIAD_LOW = 0.82"""
        assert TRIAD_LOW == 0.82
    
    def test_triad_t6_value(self):
        """TRIAD_T6 = 0.83"""
        assert TRIAD_T6 == 0.83
    
    def test_triad_ordering(self):
        """TRIAD_LOW < TRIAD_T6 < TRIAD_HIGH < Z_CRITICAL"""
        assert TRIAD_LOW < TRIAD_T6 < TRIAD_HIGH < Z_CRITICAL
    
    def test_triad_passes_required(self):
        """TRIAD requires 3 passes"""
        assert TRIAD_PASSES_REQUIRED == 3


class TestKFormationCriteria:
    """Test K-Formation threshold definitions"""
    
    def test_k_kappa_value(self):
        """K_KAPPA = 0.92"""
        assert K_KAPPA == 0.92
    
    def test_k_eta_is_phi_inv(self):
        """K_ETA = φ⁻¹"""
        assert abs(K_ETA - PHI_INV) < 1e-15
    
    def test_k_r_value(self):
        """K_R = 7"""
        assert K_R == 7


class TestNegentropy:
    """Test negentropy computation"""
    
    def test_negentropy_at_lens(self):
        """Negentropy peaks at z_c with value 1.0"""
        eta = compute_negentropy(Z_CRITICAL)
        assert abs(eta - 1.0) < 1e-10
    
    def test_negentropy_symmetric(self):
        """Negentropy is symmetric around z_c"""
        delta = 0.05
        eta_above = compute_negentropy(Z_CRITICAL + delta)
        eta_below = compute_negentropy(Z_CRITICAL - delta)
        assert abs(eta_above - eta_below) < 1e-10
    
    def test_negentropy_decays(self):
        """Negentropy decays away from z_c"""
        eta_at_lens = compute_negentropy(Z_CRITICAL)
        eta_far = compute_negentropy(0.5)
        assert eta_far < eta_at_lens
    
    def test_negentropy_bounded(self):
        """Negentropy is in [0, 1]"""
        for z in [0.0, 0.3, 0.5, 0.7, Z_CRITICAL, 0.9, 0.95, 1.0]:
            eta = compute_negentropy(z)
            assert 0 <= eta <= 1


class TestPhaseMapping:
    """Test phase determination from z-coordinate"""
    
    def test_untrue_phase(self):
        """z < φ⁻¹ → UNTRUE"""
        assert get_phase(0.0) == PHASE_UNTRUE
        assert get_phase(0.3) == PHASE_UNTRUE
        assert get_phase(0.5) == PHASE_UNTRUE
        assert get_phase(0.617) == PHASE_UNTRUE
    
    def test_paradox_phase(self):
        """φ⁻¹ ≤ z < z_c → PARADOX"""
        assert get_phase(0.62) == PHASE_PARADOX
        assert get_phase(0.7) == PHASE_PARADOX
        assert get_phase(0.8) == PHASE_PARADOX
        assert get_phase(0.865) == PHASE_PARADOX
    
    def test_true_phase(self):
        """z_c ≤ z < 0.92 → TRUE"""
        assert get_phase(Z_CRITICAL) == PHASE_TRUE
        assert get_phase(0.87) == PHASE_TRUE
        assert get_phase(0.9) == PHASE_TRUE
        assert get_phase(0.919) == PHASE_TRUE
    
    def test_hyper_true_phase(self):
        """z ≥ 0.92 → HYPER_TRUE"""
        assert get_phase(0.92) == PHASE_HYPER_TRUE
        assert get_phase(0.95) == PHASE_HYPER_TRUE
        assert get_phase(0.99) == PHASE_HYPER_TRUE
        assert get_phase(1.0) == PHASE_HYPER_TRUE


class TestTierMapping:
    """Test time-harmonic tier determination"""
    
    def test_tier_t1(self):
        """0.00 ≤ z < 0.10 → t1"""
        assert get_tier(0.0) == 't1'
        assert get_tier(0.05) == 't1'
        assert get_tier(0.09) == 't1'
    
    def test_tier_t2(self):
        """0.10 ≤ z < 0.20 → t2"""
        assert get_tier(0.10) == 't2'
        assert get_tier(0.15) == 't2'
        assert get_tier(0.19) == 't2'
    
    def test_tier_t3(self):
        """0.20 ≤ z < 0.45 → t3"""
        assert get_tier(0.20) == 't3'
        assert get_tier(0.30) == 't3'
        assert get_tier(0.44) == 't3'
    
    def test_tier_t4(self):
        """0.45 ≤ z < 0.65 → t4"""
        assert get_tier(0.45) == 't4'
        assert get_tier(0.55) == 't4'
        assert get_tier(0.64) == 't4'
    
    def test_tier_t5(self):
        """0.65 ≤ z < 0.75 → t5"""
        assert get_tier(0.65) == 't5'
        assert get_tier(0.70) == 't5'
        assert get_tier(0.74) == 't5'
    
    def test_tier_t6_locked(self):
        """0.75 ≤ z < z_c → t6 (TRIAD locked)"""
        assert get_tier(0.75, triad_unlocked=False) == 't6'
        assert get_tier(0.80, triad_unlocked=False) == 't6'
        assert get_tier(0.84, triad_unlocked=False) == 't6'
    
    def test_tier_t6_unlocked(self):
        """0.75 ≤ z < 0.83 → t6 (TRIAD unlocked)"""
        assert get_tier(0.75, triad_unlocked=True) == 't6'
        assert get_tier(0.80, triad_unlocked=True) == 't6'
        assert get_tier(0.82, triad_unlocked=True) == 't6'
    
    def test_tier_t7_transition(self):
        """TRIAD unlock lowers t6 gate from z_c to 0.83"""
        # At z=0.84, locked → t6, unlocked → t7
        assert get_tier(0.84, triad_unlocked=False) == 't6'
        assert get_tier(0.84, triad_unlocked=True) == 't7'
    
    def test_tier_t7(self):
        """z_c ≤ z < 0.92 → t7"""
        assert get_tier(Z_CRITICAL) == 't7'
        assert get_tier(0.90) == 't7'
        assert get_tier(0.919) == 't7'
    
    def test_tier_t8(self):
        """0.92 ≤ z < 0.97 → t8"""
        assert get_tier(0.92) == 't8'
        assert get_tier(0.95) == 't8'
        assert get_tier(0.969) == 't8'
    
    def test_tier_t9(self):
        """0.97 ≤ z ≤ 1.00 → t9"""
        assert get_tier(0.97) == 't9'
        assert get_tier(0.99) == 't9'
        assert get_tier(1.0) == 't9'


class TestOperatorWindows:
    """Test operator window definitions"""
    
    def test_t1_operators(self):
        """t1 allows only +"""
        assert get_operators('t1') == ['+']
    
    def test_t5_operators(self):
        """t5 allows all operators"""
        ops = get_operators('t5')
        assert len(ops) == 6
        assert set(ops) == {'+', '()', '^', '−', '×', '÷'}
    
    def test_t6_operators(self):
        """t6 allows +, ÷, (), −"""
        ops = get_operators('t6')
        assert set(ops) == {'+', '÷', '()', '−'}
    
    def test_t7_operators(self):
        """t7 allows only +, ()"""
        assert get_operators('t7') == ['+', '()']
    
    def test_t9_operators(self):
        """t9 allows all operators"""
        ops = get_operators('t9')
        assert len(ops) == 6


class TestKFormationCheck:
    """Test K-Formation verification logic"""
    
    def test_k_formation_all_met(self):
        """K-Formation achieved when all criteria met"""
        assert check_k_formation(0.95, 0.7, 8) == True
        assert check_k_formation(0.92, 0.619, 7) == True
    
    def test_k_formation_kappa_failed(self):
        """K-Formation fails when κ < 0.92"""
        assert check_k_formation(0.91, 0.7, 8) == False
        assert check_k_formation(0.5, 0.9, 10) == False
    
    def test_k_formation_eta_failed(self):
        """K-Formation fails when η ≤ φ⁻¹"""
        assert check_k_formation(0.95, 0.6, 8) == False
        assert check_k_formation(0.95, PHI_INV, 8) == False  # Must be > not >=
    
    def test_k_formation_r_failed(self):
        """K-Formation fails when R < 7"""
        assert check_k_formation(0.95, 0.7, 6) == False
        assert check_k_formation(0.95, 0.7, 0) == False


class TestLearningRate:
    """Test learning rate computation"""
    
    def test_learning_rate_at_zero(self):
        """LR at z=0, κ=0 is base rate"""
        lr = compute_learning_rate(0.0, 0.0)
        assert abs(lr - 0.1) < 1e-10
    
    def test_learning_rate_increases_with_z(self):
        """LR increases with z"""
        lr_low = compute_learning_rate(0.3, 0.5)
        lr_high = compute_learning_rate(0.9, 0.5)
        assert lr_high > lr_low
    
    def test_learning_rate_increases_with_kappa(self):
        """LR increases with coherence"""
        lr_low = compute_learning_rate(0.5, 0.3)
        lr_high = compute_learning_rate(0.5, 0.9)
        assert lr_high > lr_low


class TestFrequencyTiers:
    """Test archetypal frequency tier mapping"""
    
    def test_planet_tier(self):
        """z < φ⁻¹ → Planet (174-285 Hz)"""
        tier, freqs = get_frequency_tier(0.3)
        assert tier == 'Planet'
        assert freqs == FREQ_PLANET
    
    def test_garden_tier(self):
        """φ⁻¹ ≤ z < z_c → Garden (396-528 Hz)"""
        tier, freqs = get_frequency_tier(0.7)
        assert tier == 'Garden'
        assert freqs == FREQ_GARDEN
    
    def test_rose_tier(self):
        """z ≥ z_c → Rose (639-963 Hz)"""
        tier, freqs = get_frequency_tier(0.9)
        assert tier == 'Rose'
        assert freqs == FREQ_ROSE


class TestAPLOperators:
    """Test APL operator definitions"""
    
    def test_six_operators_defined(self):
        """All 6 APL operators are defined"""
        assert len(APL_OPERATORS) == 6
        assert set(APL_OPERATORS.keys()) == {'+', '()', '^', '−', '×', '÷'}
    
    def test_operator_names(self):
        """Operators have correct names"""
        assert APL_OPERATORS['+'][0] == 'Group'
        assert APL_OPERATORS['()'][0] == 'Boundary'
        assert APL_OPERATORS['^'][0] == 'Amplify'
        assert APL_OPERATORS['−'][0] == 'Separate'
        assert APL_OPERATORS['×'][0] == 'Fusion'
        assert APL_OPERATORS['÷'][0] == 'Decohere'


class TestPhaseVocabulary:
    """Test phase vocabulary definitions"""
    
    def test_all_phases_have_vocab(self):
        """All phases have vocabulary defined"""
        for phase in [PHASE_UNTRUE, PHASE_PARADOX, PHASE_TRUE, PHASE_HYPER_TRUE]:
            assert phase in PHASE_VOCAB
            assert 'nouns' in PHASE_VOCAB[phase]
            assert 'verbs' in PHASE_VOCAB[phase]
            assert 'adjectives' in PHASE_VOCAB[phase]
    
    def test_hyper_true_expanded(self):
        """HYPER_TRUE has expanded vocabulary"""
        hyper_nouns = PHASE_VOCAB[PHASE_HYPER_TRUE]['nouns']
        hyper_verbs = PHASE_VOCAB[PHASE_HYPER_TRUE]['verbs']
        hyper_adjs = PHASE_VOCAB[PHASE_HYPER_TRUE]['adjectives']
        
        # Should have expanded vocabulary (12/10/12)
        assert len(hyper_nouns) >= 12
        assert len(hyper_verbs) >= 10
        assert len(hyper_adjs) >= 12


class TestSpiralsAndSlots:
    """Test spiral markers and token slots"""
    
    def test_three_spirals(self):
        """Three spirals: Φ, e, π"""
        assert len(SPIRALS) == 3
        assert 'Φ' in SPIRALS
        assert 'e' in SPIRALS
        assert 'π' in SPIRALS
    
    def test_six_token_slots(self):
        """Six token slots defined"""
        assert len(TOKEN_SLOTS) == 6
        assert set(TOKEN_SLOTS) == {'NP', 'VP', 'MOD', 'DET', 'CONN', 'Q'}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
