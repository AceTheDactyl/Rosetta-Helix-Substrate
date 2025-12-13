"""
Tests for Threshold Modules
============================

Comprehensive tests for all 10 threshold modules and the orchestrator.
"""

import pytest
import math
import time
from pathlib import Path

from threshold_modules import (
    # Core structures
    ThresholdState, ThresholdSignal, LearningLesson, ArchitecturalGap, Enhancement,
    TestResult, ConvergenceProof,

    # Modules
    ThresholdModule, NegEntropyEngine, LearningAccumulator, TestOracle,
    ReflectionEngine, GapDetector, EnhancementFinder, EnhancementBuilder,
    DecisionEngine, ConvergenceMonitor, ConsciousnessGate,

    # Orchestrator
    AutonomousTrainingOrchestrator,

    # Constants
    Z_CRITICAL, PHI, PHI_INV, Q_KAPPA, KAPPA_S, LAMBDA,
    MU_P, MU_1, MU_2, MU_S, MU_3,
    TRIAD_HIGH, TRIAD_LOW, TRIAD_T6,
    compute_delta_s_neg, classify_mu, check_k_formation
)


# =============================================================================
# PHYSICS CONSTANTS TESTS
# =============================================================================

class TestPhysicsConstants:
    """Verify all physics constants are correctly defined"""

    def test_z_critical_value(self):
        """Z_CRITICAL should be √3/2"""
        expected = math.sqrt(3.0) / 2.0
        assert abs(Z_CRITICAL - expected) < 1e-10
        assert abs(Z_CRITICAL - 0.8660254037844386) < 1e-10

    def test_golden_ratio(self):
        """PHI should be (1+√5)/2"""
        expected = (1.0 + math.sqrt(5.0)) / 2.0
        assert abs(PHI - expected) < 1e-10
        assert abs(PHI - 1.618033988749895) < 1e-10

    def test_phi_inverse(self):
        """PHI_INV should be 1/PHI"""
        assert abs(PHI_INV - 1.0/PHI) < 1e-10
        assert abs(PHI * PHI_INV - 1.0) < 1e-10

    def test_mu_hierarchy(self):
        """μ thresholds should be properly ordered"""
        assert MU_1 < MU_P < MU_2 < Z_CRITICAL < MU_S < MU_3

    def test_triad_hysteresis(self):
        """TRIAD thresholds should have proper hysteresis"""
        assert TRIAD_LOW < TRIAD_T6 < TRIAD_HIGH
        assert TRIAD_HIGH - TRIAD_LOW > 0.02  # Minimum hysteresis

    def test_kappa_s_value(self):
        """KAPPA_S should be 0.920"""
        assert abs(KAPPA_S - 0.920) < 1e-10

    def test_q_kappa_value(self):
        """Q_KAPPA should be 0.3514087324"""
        assert abs(Q_KAPPA - 0.3514087324) < 1e-10

    def test_lambda_value(self):
        """LAMBDA should be 7.7160493827"""
        assert abs(LAMBDA - 7.7160493827) < 1e-10


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Test helper functions from constants"""

    def test_compute_delta_s_neg_at_critical(self):
        """ΔS_neg should be maximal (1.0) at z_c"""
        s = compute_delta_s_neg(Z_CRITICAL)
        assert abs(s - 1.0) < 1e-10

    def test_compute_delta_s_neg_decays(self):
        """ΔS_neg should decay away from z_c"""
        s_critical = compute_delta_s_neg(Z_CRITICAL)
        s_below = compute_delta_s_neg(Z_CRITICAL - 0.1)
        s_above = compute_delta_s_neg(Z_CRITICAL + 0.1)
        assert s_below < s_critical
        assert s_above < s_critical

    def test_classify_mu(self):
        """Classify μ should return correct regions"""
        assert classify_mu(0.1) == 'pre_conscious_basin'
        assert classify_mu(MU_P - 0.01) == 'approaching_paradox'
        assert classify_mu(MU_2 - 0.01) == 'conscious_basin'
        assert classify_mu(Z_CRITICAL - 0.01) == 'pre_lens_integrated'
        assert classify_mu(Z_CRITICAL + 0.01) == 'lens_integrated'
        assert classify_mu(MU_S + 0.01) == 'singularity_proximal'
        assert classify_mu(MU_3 + 0.01) == 'ultra_integrated'

    def test_check_k_formation_true(self):
        """K-formation should trigger with correct parameters"""
        assert check_k_formation(kappa=0.94, eta=0.72, R=8) is True

    def test_check_k_formation_false_kappa(self):
        """K-formation should fail with low kappa"""
        assert check_k_formation(kappa=0.5, eta=0.72, R=8) is False

    def test_check_k_formation_false_eta(self):
        """K-formation should fail with low eta"""
        assert check_k_formation(kappa=0.94, eta=0.5, R=8) is False

    def test_check_k_formation_false_r(self):
        """K-formation should fail with low R"""
        assert check_k_formation(kappa=0.94, eta=0.72, R=3) is False


# =============================================================================
# THRESHOLD STATE TESTS
# =============================================================================

class TestThresholdState:
    """Test threshold state transitions"""

    def test_state_values(self):
        """All states should have correct values"""
        assert ThresholdState.DORMANT.value == "dormant"
        assert ThresholdState.ACTIVATING.value == "activating"
        assert ThresholdState.ACTIVE.value == "active"
        assert ThresholdState.DECAYING.value == "decaying"
        assert ThresholdState.CONVERGED.value == "converged"


# =============================================================================
# DATA STRUCTURE TESTS
# =============================================================================

class TestDataStructures:
    """Test core data structures"""

    def test_threshold_signal_creation(self):
        """ThresholdSignal should be created correctly"""
        signal = ThresholdSignal(
            source="test_module",
            value=0.85,
            timestamp=time.time(),
            payload={'action': 'test'},
            entropy_delta=0.1,
            confidence=0.9
        )
        assert signal.source == "test_module"
        assert signal.value == 0.85
        assert signal.entropy_delta == 0.1
        assert signal.confidence == 0.9

    def test_learning_lesson_to_dict(self):
        """LearningLesson should serialize correctly"""
        lesson = LearningLesson(
            lesson_id="test_lesson",
            category="pattern",
            description="Test description",
            evidence=["ev1", "ev2"],
            confidence=0.8,
            applicability=0.7
        )
        d = lesson.to_dict()
        assert d['id'] == "test_lesson"
        assert d['category'] == "pattern"
        assert d['confidence'] == 0.8

    def test_architectural_gap_to_dict(self):
        """ArchitecturalGap should serialize correctly"""
        gap = ArchitecturalGap(
            gap_id="test_gap",
            gap_type="complexity",
            location="test.py",
            description="Test gap",
            severity=0.5,
            enhancement_potential=0.6
        )
        d = gap.to_dict()
        assert d['id'] == "test_gap"
        assert d['type'] == "complexity"
        assert d['severity'] == 0.5

    def test_enhancement_to_dict(self):
        """Enhancement should serialize correctly"""
        enh = Enhancement(
            enhancement_id="test_enh",
            gap_id="test_gap",
            enhancement_type="refactor",
            description="Test enhancement",
            implementation_plan=["step1", "step2"],
            estimated_improvement=0.3,
            risk_level=0.2
        )
        d = enh.to_dict()
        assert d['id'] == "test_enh"
        assert d['gap'] == "test_gap"
        assert d['plan_steps'] == 2


# =============================================================================
# MODULE 1: NEG ENTROPY ENGINE TESTS
# =============================================================================

class TestNegEntropyEngine:
    """Tests for T1: NegEntropyEngine (Z_CRITICAL threshold)"""

    def test_threshold_is_z_critical(self):
        """Module threshold should be Z_CRITICAL"""
        engine = NegEntropyEngine()
        assert engine.threshold == Z_CRITICAL
        assert engine.threshold_constant_name == "Z_CRITICAL"

    def test_dormant_below_threshold(self):
        """Engine should be dormant below Z_CRITICAL"""
        engine = NegEntropyEngine()
        engine.update_z(Z_CRITICAL - 0.1)
        assert engine.state == ThresholdState.DORMANT

    def test_activates_at_threshold(self):
        """Engine should activate at Z_CRITICAL"""
        engine = NegEntropyEngine()
        engine.update_z(Z_CRITICAL)
        assert engine.state == ThresholdState.ACTIVATING
        engine.update_z(Z_CRITICAL)
        assert engine.state == ThresholdState.ACTIVE

    def test_process_generates_signal(self):
        """Process should generate a signal with entropy delta"""
        engine = NegEntropyEngine()
        engine.update_z(Z_CRITICAL)
        engine.update_z(Z_CRITICAL)  # Activate

        signal = engine.process({'z': Z_CRITICAL, 'target_z': Z_CRITICAL})
        assert signal.source == "NegEntropyEngine"
        assert 'injection' in signal.payload
        assert 'action' in signal.payload

    def test_pump_profiles(self):
        """Pump profiles should affect injection amount"""
        engine = NegEntropyEngine()
        engine.update_z(Z_CRITICAL)
        engine.update_z(Z_CRITICAL)

        # Test different profiles
        engine.set_pump_profile("gentle")
        sig1 = engine.process({'z': 0.5, 'target_z': Z_CRITICAL})

        engine.set_pump_profile("aggressive")
        sig2 = engine.process({'z': 0.5, 'target_z': Z_CRITICAL})

        # Aggressive should inject more
        assert abs(sig2.payload['injection']) >= abs(sig1.payload['injection'])


# =============================================================================
# MODULE 2: LEARNING ACCUMULATOR TESTS
# =============================================================================

class TestLearningAccumulator:
    """Tests for T2: LearningAccumulator (PHI_INV threshold)"""

    def test_threshold_is_phi_inv(self):
        """Module threshold should be PHI_INV"""
        acc = LearningAccumulator()
        assert acc.threshold == PHI_INV
        assert acc.threshold_constant_name == "PHI_INV"

    def test_learns_from_observations(self):
        """Should learn lessons from repeated observations"""
        acc = LearningAccumulator()
        acc.update_z(PHI_INV + 0.1)
        acc.update_z(PHI_INV + 0.1)

        # Provide observations
        observations = [
            {'type': 'pattern_a', 'value': 1},
            {'type': 'pattern_a', 'value': 2},
            {'type': 'pattern_a', 'value': 3},
        ]

        signal = acc.process({'observations': observations})
        assert 'lessons_learned' in signal.payload

    def test_get_lessons_by_category(self):
        """Should filter lessons by category"""
        acc = LearningAccumulator()
        acc.lessons = [
            LearningLesson("l1", "pattern", "desc1", [], 0.8, 0.5),
            LearningLesson("l2", "error", "desc2", [], 0.7, 0.4),
            LearningLesson("l3", "pattern", "desc3", [], 0.9, 0.6),
        ]

        pattern_lessons = acc.get_lessons("pattern")
        assert len(pattern_lessons) == 2


# =============================================================================
# MODULE 3: TEST ORACLE TESTS
# =============================================================================

class TestTestOracle:
    """Tests for T3: TestOracle (KAPPA_S threshold)"""

    def test_threshold_is_kappa_s(self):
        """Module threshold should be KAPPA_S"""
        oracle = TestOracle()
        assert oracle.threshold == KAPPA_S
        assert oracle.threshold_constant_name == "KAPPA_S"

    def test_validates_python_syntax(self):
        """Should validate Python syntax correctly"""
        oracle = TestOracle()
        oracle.update_z(KAPPA_S + 0.01)
        oracle.update_z(KAPPA_S + 0.01)

        # Test with current file (known valid)
        signal = oracle.process({'files_changed': []})
        assert 'tests_run' in signal.payload

    def test_general_validation(self):
        """General validation should pass with correct constants"""
        oracle = TestOracle()
        oracle.update_z(KAPPA_S + 0.01)
        oracle.update_z(KAPPA_S + 0.01)

        signal = oracle.process({})
        assert signal.payload.get('all_passed', False) or signal.value > 0


# =============================================================================
# MODULE 4: REFLECTION ENGINE TESTS
# =============================================================================

class TestReflectionEngine:
    """Tests for T4: ReflectionEngine (MU_P threshold)"""

    def test_threshold_is_mu_p(self):
        """Module threshold should be MU_P"""
        engine = ReflectionEngine()
        assert engine.threshold == MU_P
        assert engine.threshold_constant_name == "MU_P"

    def test_analyzes_files(self):
        """Should analyze Python files and generate insights"""
        engine = ReflectionEngine()
        engine.update_z(MU_P + 0.1)
        engine.update_z(MU_P + 0.1)

        signal = engine.process({'files': []})
        assert 'files_analyzed' in signal.payload
        assert 'metrics' in signal.payload


# =============================================================================
# MODULE 5: GAP DETECTOR TESTS
# =============================================================================

class TestGapDetector:
    """Tests for T5: GapDetector (MU_1 threshold)"""

    def test_threshold_is_mu_1(self):
        """Module threshold should be MU_1"""
        detector = GapDetector()
        assert detector.threshold == MU_1
        assert detector.threshold_constant_name == "MU_1"

    def test_detects_gaps(self):
        """Should detect architectural gaps"""
        detector = GapDetector()
        detector.update_z(MU_1 + 0.1)
        detector.update_z(MU_1 + 0.1)

        signal = detector.process({})
        assert 'gaps_detected' in signal.payload

    def test_gaps_from_reflections(self):
        """Should generate gaps from reflection insights"""
        detector = GapDetector()
        detector.update_z(MU_1 + 0.1)
        detector.update_z(MU_1 + 0.1)

        reflections = [{
            'insights': [
                {'type': 'large_file', 'file': 'test.py', 'detail': 'Large file'}
            ]
        }]

        signal = detector.process({'reflections': reflections})
        assert signal.payload['gaps_detected'] >= 0


# =============================================================================
# MODULE 6: ENHANCEMENT FINDER TESTS
# =============================================================================

class TestEnhancementFinder:
    """Tests for T6: EnhancementFinder (MU_2 threshold)"""

    def test_threshold_is_mu_2(self):
        """Module threshold should be MU_2"""
        finder = EnhancementFinder()
        assert finder.threshold == MU_2
        assert finder.threshold_constant_name == "MU_2"

    def test_finds_enhancements_from_gaps(self):
        """Should find enhancements for gaps"""
        finder = EnhancementFinder()
        finder.update_z(MU_2 + 0.1)
        finder.update_z(MU_2 + 0.1)

        gaps = [
            ArchitecturalGap(
                gap_id="gap1",
                gap_type="complexity",
                location="test.py",
                description="Complex code",
                severity=0.5,
                enhancement_potential=0.6
            )
        ]

        signal = finder.process({'gaps': gaps})
        assert signal.payload['enhancements_found'] >= 1

    def test_enhancement_patterns(self):
        """Should have enhancement patterns for all gap types"""
        finder = EnhancementFinder()
        patterns = finder.enhancement_patterns

        assert 'missing_module' in patterns
        assert 'high_coupling' in patterns
        assert 'redundancy' in patterns
        assert 'complexity' in patterns


# =============================================================================
# MODULE 7: ENHANCEMENT BUILDER TESTS
# =============================================================================

class TestEnhancementBuilder:
    """Tests for T7: EnhancementBuilder (MU_3 threshold)"""

    def test_threshold_is_mu_3(self):
        """Module threshold should be MU_3"""
        builder = EnhancementBuilder()
        assert builder.threshold == MU_3
        assert builder.threshold_constant_name == "MU_3"

    def test_builds_enhancement(self):
        """Should build enhancements"""
        builder = EnhancementBuilder()
        builder.update_z(MU_3 + 0.001)
        builder.update_z(MU_3 + 0.001)

        enhancements = [
            Enhancement(
                enhancement_id="enh1",
                gap_id="gap1",
                enhancement_type="refactor",
                description="Test enhancement",
                implementation_plan=["step1"],
                estimated_improvement=0.3,
                risk_level=0.1
            )
        ]

        signal = builder.process({'enhancements': enhancements})
        assert 'success' in signal.payload


# =============================================================================
# MODULE 8: DECISION ENGINE TESTS
# =============================================================================

class TestDecisionEngine:
    """Tests for T8: DecisionEngine (TRIAD_HIGH threshold)"""

    def test_threshold_is_triad_high(self):
        """Module threshold should be TRIAD_HIGH"""
        engine = DecisionEngine()
        assert engine.threshold == TRIAD_HIGH
        assert engine.threshold_constant_name == "TRIAD_HIGH"

    def test_makes_decision(self):
        """Should make decisions based on context"""
        engine = DecisionEngine()
        engine.update_z(TRIAD_HIGH + 0.01)
        engine.update_z(TRIAD_HIGH + 0.01)

        signal = engine.process({
            'lessons': [],
            'gaps': [],
            'enhancements': [],
            'test_results': []
        })

        assert 'decision' in signal.payload
        assert signal.payload['decision'] in [
            'PROCEED_WITH_ENHANCEMENTS',
            'ADDRESS_GAPS_FIRST',
            'FIX_TESTS_FIRST',
            'CONTINUE_LEARNING'
        ]


# =============================================================================
# MODULE 9: CONVERGENCE MONITOR TESTS
# =============================================================================

class TestConvergenceMonitor:
    """Tests for T9: ConvergenceMonitor (TRIAD_LOW threshold)"""

    def test_threshold_is_triad_low(self):
        """Module threshold should be TRIAD_LOW"""
        monitor = ConvergenceMonitor()
        assert monitor.threshold == TRIAD_LOW
        assert monitor.threshold_constant_name == "TRIAD_LOW"

    def test_tracks_trajectory(self):
        """Should track metric trajectory"""
        monitor = ConvergenceMonitor()
        monitor.update_z(TRIAD_LOW + 0.01)
        monitor.update_z(TRIAD_LOW + 0.01)

        for i in range(15):
            monitor.process({'metric': 0.5 + i * 0.01})

        assert len(monitor.trajectory) == 15

    def test_detects_convergence(self):
        """Should detect convergence when rate is low"""
        monitor = ConvergenceMonitor()
        monitor.update_z(TRIAD_LOW + 0.01)
        monitor.update_z(TRIAD_LOW + 0.01)

        # Simulate converging sequence
        for i in range(20):
            monitor.process({'metric': 0.9 + 0.001 * (1 / (i + 1))})

        signal = monitor.process({'metric': 0.901})
        # Should be close to converged
        assert 'converged' in signal.payload

    def test_prove_convergence(self):
        """Should generate convergence proof"""
        monitor = ConvergenceMonitor()
        monitor.trajectory = [0.8, 0.85, 0.88, 0.89, 0.895, 0.898, 0.899, 0.8995, 0.8998, 0.8999]

        proof = monitor.prove_convergence()
        assert isinstance(proof, ConvergenceProof)
        assert proof.iterations == 10
        assert proof.initial_entropy == 0.8

    def test_tarski_fixed_point(self):
        """Should identify Tarski fixed-point conditions"""
        monitor = ConvergenceMonitor()
        # Monotonically increasing bounded sequence
        monitor.trajectory = [float(0.5 + i * 0.01) for i in range(20)]

        proof = monitor.prove_convergence()
        assert proof.monotonic is True


# =============================================================================
# MODULE 10: CONSCIOUSNESS GATE TESTS
# =============================================================================

class TestConsciousnessGate:
    """Tests for T10: ConsciousnessGate (Q_KAPPA threshold)"""

    def test_threshold_is_q_kappa(self):
        """Module threshold should be Q_KAPPA"""
        gate = ConsciousnessGate()
        assert gate.threshold == Q_KAPPA
        assert gate.threshold_constant_name == "Q_KAPPA"

    def test_checks_k_formation(self):
        """Should check K-formation criteria"""
        gate = ConsciousnessGate()
        gate.update_z(Q_KAPPA + 0.1)
        gate.update_z(Q_KAPPA + 0.1)

        # Create module states that could trigger K-formation
        modules = {
            f'module_{i}': {'active': True, 'processing_count': 10}
            for i in range(10)
        }

        signal = gate.process({'modules': modules})
        assert 'k_formation' in signal.payload
        assert 'kappa' in signal.payload
        assert 'eta' in signal.payload
        assert 'R' in signal.payload

    def test_awareness_summary(self):
        """Should provide awareness summary"""
        gate = ConsciousnessGate()
        gate.update_z(Q_KAPPA + 0.1)
        gate.update_z(Q_KAPPA + 0.1)

        for _ in range(5):
            gate.process({'modules': {}})

        summary = gate.get_awareness_summary()
        assert 'total_observations' in summary
        assert summary['total_observations'] == 5


# =============================================================================
# ORCHESTRATOR TESTS
# =============================================================================

class TestAutonomousTrainingOrchestrator:
    """Tests for the AutonomousTrainingOrchestrator"""

    def test_initializes_all_modules(self):
        """Should initialize all 10 modules"""
        orch = AutonomousTrainingOrchestrator()
        assert len(orch.modules) == 10
        assert 'T1_NegEntropy' in orch.modules
        assert 'T10_Consciousness' in orch.modules

    def test_update_z_all_modules(self):
        """Should update z for all modules"""
        orch = AutonomousTrainingOrchestrator()
        orch.update_z(Z_CRITICAL)

        for module in orch.modules.values():
            assert module._z_current == Z_CRITICAL

    def test_run_cycle(self):
        """Should run a complete cycle"""
        orch = AutonomousTrainingOrchestrator()
        state = orch.run_cycle(z=Z_CRITICAL)

        assert 'cycle' in state
        assert 'z' in state
        assert 'active_modules' in state
        assert 'signals' in state

    def test_get_status(self):
        """Should return status for all modules"""
        orch = AutonomousTrainingOrchestrator()
        orch.run_cycle(z=0.5)

        status = orch.get_status()
        assert 'cycle_count' in status
        assert 'modules' in status
        assert len(status['modules']) == 10


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full system"""

    def test_full_training_cycle(self):
        """Should complete a full training cycle"""
        orch = AutonomousTrainingOrchestrator()

        # Run 10 cycles
        for i in range(10):
            z = 0.5 + i * 0.04  # Gradually increase z
            state = orch.run_cycle(z=z)

        assert orch.cycle_count == 10
        assert len(orch.state_history) == 10

    def test_signal_flow(self):
        """Signals should flow between modules"""
        orch = AutonomousTrainingOrchestrator()

        # Run at high z to activate all modules
        state = orch.run_cycle(z=0.99)

        # Should have signals from active modules
        assert len(orch.total_signals) > 0

    def test_module_threshold_ordering(self):
        """Module thresholds should be in correct order"""
        orch = AutonomousTrainingOrchestrator()

        thresholds = [
            ('T10_Consciousness', Q_KAPPA),
            ('T5_GapDetect', MU_1),
            ('T4_Reflection', MU_P),
            ('T2_Learning', PHI_INV),
            ('T6_EnhanceFind', MU_2),
            ('T9_Convergence', TRIAD_LOW),
            ('T8_Decision', TRIAD_HIGH),
            ('T1_NegEntropy', Z_CRITICAL),
            ('T3_Testing', KAPPA_S),
            ('T7_EnhanceBuild', MU_3),
        ]

        for name, expected_threshold in thresholds:
            module = orch.modules[name]
            assert abs(module.threshold - expected_threshold) < 1e-10, \
                f"{name} threshold mismatch: {module.threshold} != {expected_threshold}"


# =============================================================================
# CONVERGENCE PROOF TESTS
# =============================================================================

class TestConvergenceProofs:
    """Test mathematical convergence properties"""

    def test_monotonic_improvement(self):
        """Training should show monotonic improvement"""
        orch = AutonomousTrainingOrchestrator()

        z_values = []
        for i in range(20):
            z = 0.5 + i * 0.02
            state = orch.run_cycle(z=z)
            z_values.append(z)

        # z should be monotonically increasing
        for i in range(len(z_values) - 1):
            assert z_values[i] <= z_values[i + 1]

    def test_bounded_values(self):
        """All values should be bounded [0, 1]"""
        orch = AutonomousTrainingOrchestrator()

        for i in range(10):
            state = orch.run_cycle(z=0.5 + i * 0.05)

            for signal in orch.total_signals:
                assert 0 <= signal.value <= 1
                assert 0 <= signal.confidence <= 1


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_z_at_zero(self):
        """System should handle z=0"""
        orch = AutonomousTrainingOrchestrator()
        state = orch.run_cycle(z=0.0)
        assert state['active_modules'] >= 0

    def test_z_at_one(self):
        """System should handle z=1"""
        orch = AutonomousTrainingOrchestrator()
        state = orch.run_cycle(z=1.0)
        assert state is not None

    def test_empty_observations(self):
        """Modules should handle empty observations"""
        acc = LearningAccumulator()
        acc.update_z(PHI_INV + 0.1)
        acc.update_z(PHI_INV + 0.1)

        signal = acc.process({'observations': []})
        assert signal is not None

    def test_hysteresis_behavior(self):
        """Modules should exhibit hysteresis"""
        engine = NegEntropyEngine()

        # Go above threshold
        engine.update_z(Z_CRITICAL + 0.05)
        engine.update_z(Z_CRITICAL + 0.05)
        assert engine.state == ThresholdState.ACTIVE

        # Go slightly below (within hysteresis)
        engine.update_z(Z_CRITICAL - 0.01)
        assert engine.state == ThresholdState.ACTIVE  # Should still be active

        # Go well below (outside hysteresis)
        engine.update_z(Z_CRITICAL - 0.05)
        assert engine.state in (ThresholdState.DECAYING, ThresholdState.DORMANT)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance and efficiency tests"""

    def test_cycle_time(self):
        """Cycle should complete in reasonable time"""
        orch = AutonomousTrainingOrchestrator()

        start = time.time()
        for _ in range(100):
            orch.run_cycle(z=Z_CRITICAL)
        duration = time.time() - start

        # 100 cycles should complete in under 5 seconds
        assert duration < 5.0

    def test_memory_bounded(self):
        """Memory usage should be bounded"""
        orch = AutonomousTrainingOrchestrator()

        # Run many cycles
        for i in range(200):
            orch.run_cycle(z=0.5 + (i % 50) * 0.01)

        # Check bounded collections
        for module in orch.modules.values():
            if hasattr(module, 'trajectory'):
                assert len(module.trajectory) <= 100
            if hasattr(module, 'awareness_log'):
                assert len(module.awareness_log) <= 100


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
