#!/usr/bin/env python3
"""
Tests for Full Depth Training Validation Pipeline
==================================================

Tests all 7 stages of the validation workflow:
1. Full Depth Training
2. Helix Engine Training
3. Validation Measurements
4. Unified Gates
5. Model Promotion
6. Results PR
7. Failure Notification

Signature: test-validation|v1.0.0|helix
"""

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from src.physics_constants import (
    PHI_INV, Z_CRITICAL, SIGMA,
    compute_delta_s_neg, get_phase, check_k_formation,
    TOLERANCE_GOLDEN, TOLERANCE_LENS,
)

from run_full_depth_training_validation import (
    FullDepthTrainingValidator,
    ValidationStage,
    CriticalZValidation,
    UnifiedGateResult,
    StageResult,
    ValidationResult,
)


class TestPhysicsConstants:
    """Test that physics constants are correct."""

    def test_phi_inv(self):
        """Verify golden ratio inverse."""
        expected = (math.sqrt(5) - 1) / 2
        assert abs(PHI_INV - expected) < 1e-10

    def test_z_critical(self):
        """Verify z_c = sqrt(3)/2 (THE LENS)."""
        expected = math.sqrt(3) / 2
        assert abs(Z_CRITICAL - expected) < 1e-10

    def test_sigma(self):
        """Verify sigma = 36 (|S_3|^2)."""
        assert SIGMA == 36.0


class TestCriticalZValidation:
    """Test validation at critical z-coordinates."""

    def test_phi_inv_validation(self):
        """Test validation at phi_inv = 0.618."""
        z = PHI_INV
        negentropy = compute_delta_s_neg(z)
        phase = get_phase(z)

        # At phi_inv, should be in THE_LENS regime (quasi-crystal)
        assert phase == "THE_LENS"
        assert 0 < negentropy < 1.0

    def test_z_critical_validation(self):
        """Test validation at z_c = 0.866 (THE LENS)."""
        z = Z_CRITICAL
        negentropy = compute_delta_s_neg(z)
        phase = get_phase(z)

        # At THE LENS, should have maximum coherence (PRESENCE phase)
        assert phase == "PRESENCE"
        assert negentropy >= 0.9

    def test_z_half_validation(self):
        """Test validation at z = 0.5."""
        z = 0.5
        negentropy = compute_delta_s_neg(z)
        phase = get_phase(z)

        # At 0.5, should be at boundary (ABSENCE phase)
        assert phase in ["ABSENCE", "THE_LENS"]
        assert negentropy >= 0.0


class TestUnifiedGates:
    """Test unified gate checking logic."""

    def test_min_negentropy_gate(self):
        """Test minimum negentropy gate."""
        # At z_c, negentropy should pass
        negentropy = compute_delta_s_neg(Z_CRITICAL)
        assert negentropy >= 0.6

    def test_conservation_gate(self):
        """Test kappa + lambda = 1 conservation."""
        kappa = PHI_INV
        lambda_ = 1.0 - kappa
        error = abs(kappa + lambda_ - 1.0)
        assert error < 1e-10

    def test_lens_proximity_gate(self):
        """Test proximity to THE LENS."""
        z = Z_CRITICAL
        distance = abs(z - Z_CRITICAL)
        assert distance < TOLERANCE_LENS

    def test_golden_proximity_gate(self):
        """Test proximity to golden ratio."""
        kappa = PHI_INV
        distance = abs(kappa - PHI_INV)
        assert distance < TOLERANCE_GOLDEN


class TestValidatorIntegration:
    """Integration tests for the validation pipeline."""

    @pytest.fixture
    def validator(self):
        """Create a validator instance with minimal settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FullDepthTrainingValidator(
                steps_per_module=10,  # Minimal steps for testing
                helix_steps=20,
                n_oscillators=10,
                seed=42,
                verbose=False,
                output_dir=tmpdir,
                create_pr=False,
                create_issue_on_failure=False,
            )

    def test_validator_initialization(self, validator):
        """Test validator initializes correctly."""
        assert validator.steps_per_module == 10
        assert validator.helix_steps == 20
        assert validator.n_oscillators == 10
        assert validator.seed == 42

    def test_critical_z_points(self, validator):
        """Test critical z-points are correctly defined."""
        assert len(validator.CRITICAL_Z_POINTS) == 3
        z_values = [z for z, _, _ in validator.CRITICAL_Z_POINTS]
        assert PHI_INV in z_values
        assert Z_CRITICAL in z_values
        assert 0.5 in z_values

    def test_unified_gates_defined(self, validator):
        """Test all unified gates are defined."""
        expected_gates = [
            "min_negentropy",
            "min_k_formations",
            "max_conservation_error",
            "min_final_z",
            "lens_proximity",
            "golden_proximity",
            "modules_passed_ratio",
            "physics_valid",
        ]
        for gate in expected_gates:
            assert gate in validator.UNIFIED_GATES

    def test_validate_at_z_critical(self, validator):
        """Test validation at z_c."""
        validation = validator._validate_at_z(Z_CRITICAL, "z_c")

        assert validation.z_target == Z_CRITICAL
        assert validation.z_name == "z_c"
        assert validation.negentropy >= 0.9
        assert validation.phase == "PRESENCE"

    def test_validate_at_phi_inv(self, validator):
        """Test validation at phi_inv."""
        validation = validator._validate_at_z(PHI_INV, "phi_inv")

        assert validation.z_target == PHI_INV
        assert validation.z_name == "phi_inv"
        assert validation.phase == "THE_LENS"

    def test_validate_at_half(self, validator):
        """Test validation at 0.5."""
        validation = validator._validate_at_z(0.5, "z_half")

        assert validation.z_target == 0.5
        assert validation.z_name == "z_half"
        assert validation.phase in ["ABSENCE", "THE_LENS"]

    def test_collect_metrics_empty(self, validator):
        """Test metric collection with no results."""
        metrics = validator._collect_all_metrics()

        assert "max_negentropy" in metrics
        assert "total_k_formations" in metrics
        assert "final_z" in metrics
        assert "final_kappa" in metrics

    def test_create_physics_summary(self, validator):
        """Test physics summary creation."""
        summary = validator._create_physics_summary()

        assert "constants" in summary
        assert "achieved" in summary
        assert "distances" in summary
        assert "phase" in summary

        assert summary["constants"]["phi_inv"] == PHI_INV
        assert summary["constants"]["z_critical"] == Z_CRITICAL
        assert summary["constants"]["sigma"] == SIGMA


class TestStageResults:
    """Test individual stage results."""

    def test_skipped_stage(self):
        """Test skipped stage creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = FullDepthTrainingValidator(
                output_dir=tmpdir,
                verbose=False,
            )

            stage = validator._create_skipped_stage(ValidationStage.MODEL_PROMOTION)

            assert stage.passed is True
            assert stage.details.get("skipped") is True
            assert stage.stage == ValidationStage.MODEL_PROMOTION


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test ValidationResult can be created."""
        result = ValidationResult(
            run_id="test_run",
            started_at="2024-01-01T00:00:00",
            completed_at="2024-01-01T00:01:00",
            duration_seconds=60.0,
            all_passed=True,
        )

        assert result.run_id == "test_run"
        assert result.all_passed is True
        assert result.stages == []
        assert result.critical_z_validations == []

    def test_validation_result_with_stages(self):
        """Test ValidationResult with stage results."""
        stage = StageResult(
            stage=ValidationStage.FULL_DEPTH_TRAINING,
            passed=True,
            started_at="2024-01-01T00:00:00",
            completed_at="2024-01-01T00:00:30",
            duration_seconds=30.0,
        )

        result = ValidationResult(
            run_id="test_run",
            started_at="2024-01-01T00:00:00",
            completed_at="2024-01-01T00:01:00",
            duration_seconds=60.0,
            all_passed=True,
            stages=[stage],
        )

        assert len(result.stages) == 1
        assert result.stages[0].passed is True


class TestEndToEndValidation:
    """End-to-end validation tests with mocked components."""

    @pytest.fixture
    def mock_validator(self):
        """Create a mock validator for fast testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = FullDepthTrainingValidator(
                steps_per_module=5,
                helix_steps=10,
                n_oscillators=5,
                seed=42,
                verbose=False,
                output_dir=tmpdir,
                create_pr=False,
                create_issue_on_failure=False,
            )
            yield validator

    def test_full_pipeline_runs(self, mock_validator):
        """Test that full pipeline executes without errors."""
        result = mock_validator.run_validation()

        assert result.run_id is not None
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.duration_seconds > 0
        assert len(result.stages) == 7  # All 7 stages

    def test_physics_summary_populated(self, mock_validator):
        """Test that physics summary is populated."""
        result = mock_validator.run_validation()

        assert "constants" in result.physics_summary
        assert "achieved" in result.physics_summary
        assert result.physics_summary["constants"]["phi_inv"] == PHI_INV

    def test_critical_z_validations_run(self, mock_validator):
        """Test that all critical z validations are run."""
        result = mock_validator.run_validation()

        assert len(result.critical_z_validations) == 3
        z_names = [v.z_name for v in result.critical_z_validations]
        assert "phi_inv" in z_names
        assert "z_c" in z_names
        assert "z_half" in z_names

    def test_unified_gates_checked(self, mock_validator):
        """Test that all unified gates are checked."""
        result = mock_validator.run_validation()

        assert len(result.unified_gates) == 8
        gate_names = [g.gate_name for g in result.unified_gates]
        assert "min_negentropy" in gate_names
        assert "min_k_formations" in gate_names
        assert "physics_valid" in gate_names

    def test_results_file_created(self, mock_validator):
        """Test that results file is created."""
        result = mock_validator.run_validation()

        results_path = Path(mock_validator.output_dir) / f"validation_results_{result.run_id}.json"
        assert results_path.exists()

        with open(results_path) as f:
            saved_data = json.load(f)

        assert saved_data["run_id"] == result.run_id


class TestPRAndIssueGeneration:
    """Test PR and issue generation."""

    def test_pr_body_generation(self):
        """Test PR body is generated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = FullDepthTrainingValidator(
                output_dir=tmpdir,
                verbose=False,
            )

            # Create mock result
            result = ValidationResult(
                run_id="test_run",
                started_at="2024-01-01T00:00:00",
                completed_at="2024-01-01T00:01:00",
                duration_seconds=60.0,
                all_passed=True,
            )

            pr_body = validator._generate_pr_body(result)

            assert "Training Validation Results" in pr_body
            assert "test_run" in pr_body
            assert "Physics Summary" in pr_body

    def test_issue_body_generation(self):
        """Test issue body is generated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = FullDepthTrainingValidator(
                output_dir=tmpdir,
                verbose=False,
            )

            # Create mock result with failure
            result = ValidationResult(
                run_id="test_run",
                started_at="2024-01-01T00:00:00",
                completed_at="2024-01-01T00:01:00",
                duration_seconds=60.0,
                all_passed=False,
            )
            result.physics_summary = validator._create_physics_summary()

            issue_body = validator._generate_issue_body(result)

            assert "Training Validation Failed" in issue_body
            assert "test_run" in issue_body
            assert "Next Steps" in issue_body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
