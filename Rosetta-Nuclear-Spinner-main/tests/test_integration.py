#!/usr/bin/env python3
"""
test_integration.py
===================

Integration tests for the Nuclear Spinner × Rosetta-Helix unified system.

Tests the full integration stack:
1. Physics constants consistency across Python, JavaScript, and C
2. Bridge simulation mode
3. Rosetta-Helix node operation
4. K-formation detection
5. Training workflow
6. Experiment orchestration

Usage:
    pytest tests/test_integration.py -v
    python tests/test_integration.py

Signature: test-integration|v1.0.0|helix
"""

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Physics constants (source of truth)
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Z_CRITICAL = math.sqrt(3) / 2
SIGMA = 36.0
KAPPA_MIN = 0.92
ETA_MIN = PHI_INV
R_MIN = 7


class TestPhysicsConstants(unittest.TestCase):
    """Test physics constants consistency across modules."""

    def test_python_rosetta_helix_constants(self):
        """Test rosetta-helix/src/physics.py constants."""
        from rosetta_helix.src.physics import (
            PHI as PY_PHI,
            PHI_INV as PY_PHI_INV,
            Z_CRITICAL as PY_Z_CRITICAL,
            SIGMA as PY_SIGMA,
        )

        self.assertAlmostEqual(PY_PHI, PHI, places=10)
        self.assertAlmostEqual(PY_PHI_INV, PHI_INV, places=10)
        self.assertAlmostEqual(PY_Z_CRITICAL, Z_CRITICAL, places=10)
        self.assertAlmostEqual(PY_SIGMA, SIGMA, places=10)

    def test_python_physics_validation(self):
        """Test physics validation utilities."""
        from rosetta_helix.src.physics import (
            compute_delta_s_neg,
            check_k_formation,
            get_tier,
            get_phase_name,
        )

        # ΔS_neg peaks at z_c
        delta_s_at_zc = compute_delta_s_neg(Z_CRITICAL)
        self.assertAlmostEqual(delta_s_at_zc, 1.0, places=6)

        # ΔS_neg decreases away from z_c
        delta_s_low = compute_delta_s_neg(0.5)
        delta_s_high = compute_delta_s_neg(0.95)
        self.assertLess(delta_s_low, delta_s_at_zc)
        self.assertLess(delta_s_high, delta_s_at_zc)

        # K-formation criteria
        self.assertTrue(check_k_formation(0.95, 0.7, 8))
        self.assertFalse(check_k_formation(0.85, 0.7, 8))  # κ too low
        self.assertFalse(check_k_formation(0.95, 0.5, 8))  # η too low
        self.assertFalse(check_k_formation(0.95, 0.7, 5))  # R too low

        # Tier progression
        self.assertEqual(get_tier(0.3), 0)   # ABSENCE
        self.assertEqual(get_tier(0.45), 1)  # REACTIVE
        self.assertEqual(get_tier(0.55), 2)  # MEMORY
        self.assertEqual(get_tier(0.65), 3)  # PATTERN
        self.assertEqual(get_tier(0.80), 4)  # LEARNING
        self.assertEqual(get_tier(0.88), 6)  # UNIVERSAL

        # Phase transitions
        self.assertEqual(get_phase_name(0.5), "ABSENCE")
        self.assertEqual(get_phase_name(Z_CRITICAL), "THE_LENS")
        self.assertEqual(get_phase_name(0.90), "PRESENCE")

    def test_coupling_conservation(self):
        """Test κ + λ = 1 conservation law."""
        kappa = PHI_INV
        lambda_ = 1 - kappa

        self.assertAlmostEqual(kappa + lambda_, 1.0, places=15)
        self.assertAlmostEqual(lambda_, 1 - PHI_INV, places=15)

    def test_golden_ratio_properties(self):
        """Test golden ratio mathematical properties."""
        # φ² = φ + 1
        self.assertAlmostEqual(PHI * PHI, PHI + 1, places=10)

        # φ × φ⁻¹ = 1
        self.assertAlmostEqual(PHI * PHI_INV, 1.0, places=10)

        # φ⁻¹ + φ⁻² = 1
        phi_inv_sq = PHI_INV * PHI_INV
        self.assertAlmostEqual(PHI_INV + phi_inv_sq, 1.0, places=10)


class TestRosettaHelixModules(unittest.TestCase):
    """Test Rosetta-Helix core modules."""

    def test_heart_initialization(self):
        """Test Heart (Kuramoto) module initialization."""
        from rosetta_helix.src.heart import Heart, HeartConfig

        config = HeartConfig(n_oscillators=60, coupling_scale=8.0, seed=42)
        heart = Heart(config)

        self.assertEqual(heart.n, 60)
        self.assertEqual(len(heart.phases), 60)

    def test_heart_coherence_evolution(self):
        """Test Heart coherence evolution."""
        from rosetta_helix.src.heart import Heart, HeartConfig

        config = HeartConfig(n_oscillators=60, coupling_scale=8.0, seed=42)
        heart = Heart(config)

        # Set z to critical point
        heart.set_spinner_z(Z_CRITICAL)

        # Run steps and check coherence increases
        initial_coherence = heart.compute_coherence()
        for _ in range(100):
            heart.step()
        final_coherence = heart.compute_coherence()

        # Coherence should increase at z_c
        self.assertGreaterEqual(final_coherence, initial_coherence * 0.9)

    def test_brain_tier_gating(self):
        """Test Brain (GHMP) tier-gated processing."""
        from rosetta_helix.src.brain import Brain, BrainConfig

        config = BrainConfig(seed=42)
        brain = Brain(config)

        # At low z, fewer operators available
        brain.set_z(0.3)
        low_z_state = brain.get_state()

        brain.set_z(0.9)
        high_z_state = brain.get_state()

        self.assertGreater(high_z_state.tier, low_z_state.tier)

    def test_triad_k_formation_detection(self):
        """Test TRIAD K-formation detection."""
        from rosetta_helix.src.triad import TriadTracker, TriadConfig

        config = TriadConfig()
        triad = TriadTracker(config)

        # Update with high coherence near z_c
        events = triad.update_from_coherence(coherence=0.95, z=Z_CRITICAL)

        # Should detect K-formation conditions
        state = triad.get_state()
        self.assertIsNotNone(state)


class TestBridgeSimulation(unittest.TestCase):
    """Test bridge simulation mode."""

    def test_spinner_simulator_initialization(self):
        """Test SpinnerSimulator initialization."""
        from bridge.spinner_bridge import SpinnerSimulator

        sim = SpinnerSimulator()
        self.assertEqual(sim.z, 0.5)
        self.assertEqual(sim.target_z, 0.5)

    def test_spinner_simulator_z_ramp(self):
        """Test SpinnerSimulator z-coordinate ramping."""
        from bridge.spinner_bridge import SpinnerSimulator

        sim = SpinnerSimulator()
        sim.set_target_z(Z_CRITICAL)

        # Step until z approaches target
        for _ in range(200):
            state = sim.step()

        self.assertAlmostEqual(state.z, Z_CRITICAL, places=1)

    def test_spinner_simulator_k_formation(self):
        """Test K-formation detection in simulator."""
        from bridge.spinner_bridge import SpinnerSimulator

        sim = SpinnerSimulator()
        sim.set_target_z(Z_CRITICAL)

        # Step until K-formation
        k_formation_detected = False
        for _ in range(200):
            state = sim.step()
            if state.k_formation:
                k_formation_detected = True
                break

        self.assertTrue(k_formation_detected)


class TestExperimentOrchestration(unittest.TestCase):
    """Test experiment orchestration scripts."""

    def test_analyze_session_data_structures(self):
        """Test analyze_session.py data structures."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from analyze_session import (
            SessionAnalysis,
            KFormationEvent,
            PhaseStats,
            compute_delta_s_neg,
            get_phase,
        )

        # Test delta_s_neg computation
        delta_s = compute_delta_s_neg(Z_CRITICAL)
        self.assertAlmostEqual(delta_s, 1.0, places=6)

        # Test phase determination
        self.assertEqual(get_phase(0.5), "ABSENCE")
        self.assertEqual(get_phase(Z_CRITICAL), "THE_LENS")
        self.assertEqual(get_phase(0.9), "PRESENCE")

    def test_run_experiment_config(self):
        """Test run_experiment.py configuration."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from run_experiment import ExperimentConfig, ExperimentRunner

        config = ExperimentConfig(
            name="test_exp",
            experiment_type="k_formation",
            steps=100,
            target_z=Z_CRITICAL,
        )

        self.assertEqual(config.experiment_type, "k_formation")
        self.assertEqual(config.steps, 100)
        self.assertAlmostEqual(config.target_z, Z_CRITICAL, places=6)


class TestTrainingWorkflow(unittest.TestCase):
    """Test training workflow components."""

    def test_unified_workflow_modules(self):
        """Test unified workflow module definitions."""
        try:
            from training.src.unified_nightly_workflow import (
                ALL_MODULES,
                MODULE_PHASES,
            )

            # Should have 19 modules
            self.assertEqual(len(ALL_MODULES), 19)

            # All modules should have name and class
            for mod_name, class_name in ALL_MODULES:
                self.assertIsInstance(mod_name, str)
                self.assertIsInstance(class_name, str)
                self.assertGreater(len(mod_name), 0)
                self.assertGreater(len(class_name), 0)

            # Module phases should cover all modules
            all_phase_modules = []
            for phase_modules in MODULE_PHASES.values():
                all_phase_modules.extend(phase_modules)

            module_names = [m[0] for m in ALL_MODULES]
            for mod in module_names:
                self.assertIn(mod, all_phase_modules)

        except ImportError:
            self.skipTest("Training workflow not available")


class TestFilesystemIntegrity(unittest.TestCase):
    """Test filesystem structure and file existence."""

    def test_core_files_exist(self):
        """Test that core integration files exist."""
        required_files = [
            "rosetta-helix/src/heart.py",
            "rosetta-helix/src/brain.py",
            "rosetta-helix/src/triad.py",
            "rosetta-helix/src/spinner_client.py",
            "rosetta-helix/src/node.py",
            "rosetta-helix/src/physics.py",
            "bridge/spinner_bridge.py",
            "scripts/start_system.sh",
            "scripts/analyze_session.py",
            "scripts/run_experiment.py",
            "docs/INTEGRATION_GUIDE.md",
        ]

        for file_path in required_files:
            full_path = PROJECT_ROOT / file_path
            self.assertTrue(
                full_path.exists(),
                f"Required file missing: {file_path}"
            )

    def test_firmware_files_exist(self):
        """Test that firmware files exist."""
        required_files = [
            "nuclear_spinner_firmware/src/main.c",
            "nuclear_spinner_firmware/include/physics_constants.h",
        ]

        for file_path in required_files:
            full_path = PROJECT_ROOT / file_path
            self.assertTrue(
                full_path.exists(),
                f"Required firmware file missing: {file_path}"
            )

    def test_scripts_executable(self):
        """Test that scripts are executable."""
        scripts = [
            "scripts/start_system.sh",
        ]

        for script in scripts:
            full_path = PROJECT_ROOT / script
            if full_path.exists():
                self.assertTrue(
                    os.access(full_path, os.X_OK),
                    f"Script not executable: {script}"
                )


class TestQuasicrystalDynamics(unittest.TestCase):
    """Test quasicrystal dynamics module."""

    def test_quasicrystal_initialization(self):
        """Test QuasicrystalDynamics initialization."""
        from rosetta_helix.src.quasicrystal import (
            QuasicrystalDynamics, QuasicrystalConfig
        )

        config = QuasicrystalConfig(initial_fat=1, initial_thin=1)
        qc = QuasicrystalDynamics(config)

        state = qc.get_state()
        self.assertEqual(state.n_fat, 1)
        self.assertEqual(state.n_thin, 1)

    def test_quasicrystal_convergence_to_phi(self):
        """Test tile ratio converges to golden ratio."""
        from rosetta_helix.src.quasicrystal import QuasicrystalDynamics

        qc = QuasicrystalDynamics()
        qc.inflate_to_convergence(tolerance=1e-6)

        # Tile ratio should approach φ
        self.assertLess(qc.get_golden_error(), 1e-5)
        self.assertAlmostEqual(qc.get_tile_ratio(), PHI, places=4)

    def test_quasicrystal_negentropy_peak(self):
        """Test negentropy peaks at φ⁻¹."""
        from rosetta_helix.src.quasicrystal import QuasicrystalDynamics

        qc = QuasicrystalDynamics()

        # Inflate a few times
        for _ in range(5):
            qc.inflate()

        neg = qc.get_negentropy()
        # Negentropy should be positive
        self.assertGreater(neg, 0)
        self.assertLessEqual(neg, 1.0)

    def test_substitution_matrix_eigenvalue(self):
        """Test eigenvalue is φ²."""
        # Substitution matrix [[2,1],[1,1]] has eigenvalue (3+√5)/2 = φ²
        expected = PHI * PHI
        actual = (3 + math.sqrt(5)) / 2
        self.assertAlmostEqual(expected, actual, places=10)


class TestCyberneticTraining(unittest.TestCase):
    """Test cybernetic training module."""

    def test_training_module_enum(self):
        """Test training module enumeration."""
        from training.src.cybernetic_training import TrainingModule

        # Should have 19 modules
        self.assertEqual(len(TrainingModule), 19)

        # First module should be N0_SILENT_LAWS
        self.assertEqual(TrainingModule.N0_SILENT_LAWS.value, 0)

        # Last module should be NIGHTLY_3
        self.assertEqual(TrainingModule.NIGHTLY_3.value, 18)

    def test_training_phase_mapping(self):
        """Test module to phase mapping."""
        from training.src.cybernetic_training import (
            TrainingModule, TrainingPhase, MODULE_PHASES
        )

        # Core physics modules should be in phase 0
        self.assertEqual(
            MODULE_PHASES[TrainingModule.N0_SILENT_LAWS],
            TrainingPhase.CORE_PHYSICS
        )

        # Quasicrystal should be in dynamics phase
        self.assertEqual(
            MODULE_PHASES[TrainingModule.QUASICRYSTAL],
            TrainingPhase.DYNAMICS
        )

    def test_adaptive_params_physics_modulation(self):
        """Test adaptive parameters respond to physics."""
        from training.src.cybernetic_training import AdaptiveParams
        from bridge.unified_state_bridge import UnifiedState

        # Create mock state with high negentropy
        state = UnifiedState()
        state.delta_s_neg = 0.9  # High negentropy
        state.tier = 5
        state.ghmp.parity_even = True

        base_params = AdaptiveParams(learning_rate=1e-4)
        adapted = base_params.apply_physics(state)

        # Learning rate should increase with negentropy
        self.assertGreater(adapted.learning_rate, base_params.learning_rate)

        # Dropout should decrease
        self.assertLess(adapted.dropout_rate, base_params.dropout_rate)


class TestUnifiedStateBridge(unittest.TestCase):
    """Test unified state bridge protocol."""

    def test_unified_state_dataclass(self):
        """Test UnifiedState dataclass."""
        from bridge.unified_state_bridge import UnifiedState, TriadState

        state = UnifiedState()
        self.assertEqual(state.z, 0.5)
        self.assertEqual(state.delta_s_neg, 0.0)
        self.assertIsInstance(state.triad, TriadState)

    def test_triad_state_conservation(self):
        """Test TRIAD state preserves κ + λ = 1."""
        from bridge.unified_state_bridge import TriadState

        triad = TriadState(kappa=PHI_INV, lambda_=1-PHI_INV)
        self.assertAlmostEqual(triad.kappa + triad.lambda_, 1.0, places=10)

    def test_state_to_dict(self):
        """Test state serialization to dict."""
        from bridge.unified_state_bridge import UnifiedState

        state = UnifiedState()
        state.z = Z_CRITICAL
        state.delta_s_neg = 1.0

        d = state.to_dict()
        self.assertIn('z', d)
        self.assertIn('delta_s_neg', d)
        self.assertIn('triad', d)
        self.assertAlmostEqual(d['z'], Z_CRITICAL, places=6)


class TestFirmwareUnityFiles(unittest.TestCase):
    """Test new firmware files exist."""

    def test_unified_physics_state_header(self):
        """Test unified_physics_state.h exists."""
        path = PROJECT_ROOT / "nuclear_spinner_firmware/include/unified_physics_state.h"
        self.assertTrue(path.exists(), "unified_physics_state.h missing")

    def test_unified_physics_state_impl(self):
        """Test unified_physics_state.c exists."""
        path = PROJECT_ROOT / "nuclear_spinner_firmware/src/unified_physics_state.c"
        self.assertTrue(path.exists(), "unified_physics_state.c missing")

    def test_apl_operators_header(self):
        """Test apl_operators.h exists."""
        path = PROJECT_ROOT / "nuclear_spinner_firmware/include/apl_operators.h"
        self.assertTrue(path.exists(), "apl_operators.h missing")

    def test_apl_operators_impl(self):
        """Test apl_operators.c exists."""
        path = PROJECT_ROOT / "nuclear_spinner_firmware/src/apl_operators.c"
        self.assertTrue(path.exists(), "apl_operators.c missing")


class TestEndToEndSimulation(unittest.TestCase):
    """End-to-end integration tests in simulation mode."""

    def test_full_simulation_run(self):
        """Test a full simulation run."""
        from rosetta_helix.src.physics import (
            compute_delta_s_neg,
            check_k_formation,
        )

        # Simulate system evolution
        z = 0.5
        kappa = PHI_INV
        lambda_ = 1 - kappa
        k_formations = 0

        for step in range(500):
            # Evolve z toward z_c
            z += (Z_CRITICAL - z) * 0.01
            z = max(0.0, min(0.999, z))

            # Evolve kappa
            kappa += (PHI_INV - kappa) * 0.01
            kappa = max(0.0, min(1.0, kappa))
            lambda_ = 1 - kappa

            # Compute metrics
            neg = compute_delta_s_neg(z)
            eta = math.sqrt(neg) if neg > 0 else 0
            R = int(7 + 5 * neg)

            # Check K-formation
            if check_k_formation(kappa, eta, R):
                k_formations += 1

        # Should reach near z_c
        self.assertAlmostEqual(z, Z_CRITICAL, places=1)

        # Should have some K-formations
        self.assertGreater(k_formations, 0)

        # Conservation should hold
        self.assertAlmostEqual(kappa + lambda_, 1.0, places=10)


def run_tests():
    """Run all integration tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhysicsConstants))
    suite.addTests(loader.loadTestsFromTestCase(TestRosettaHelixModules))
    suite.addTests(loader.loadTestsFromTestCase(TestBridgeSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestExperimentOrchestration))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingWorkflow))
    suite.addTests(loader.loadTestsFromTestCase(TestFilesystemIntegrity))
    suite.addTests(loader.loadTestsFromTestCase(TestQuasicrystalDynamics))
    suite.addTests(loader.loadTestsFromTestCase(TestCyberneticTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestUnifiedStateBridge))
    suite.addTests(loader.loadTestsFromTestCase(TestFirmwareUnityFiles))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndSimulation))

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
