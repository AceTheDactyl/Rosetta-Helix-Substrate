"""
Tests for Helix Training Engine
"""

import os
import json
import tempfile
from pathlib import Path

import pytest

# Import engine components
from helix_engine.core.contract import (
    RunConfig,
    RunResult,
    RunStatus,
    ExitCode,
    EnvironmentSnapshot,
    GateResult,
    EngineContract,
)
from helix_engine.run.layout import RunLayout
from helix_engine.run.manager import RunManager
from helix_engine.reliability.guards import NaNGuard, InfGuard, PhysicsGuard, CompositeGuard
from helix_engine.reliability.determinism import DeterminismManager
from helix_engine.observability.logger import StructuredLogger, LogLevel
from helix_engine.observability.metrics import MetricsCollector, MetricsWriter
from helix_engine.config.schema import ConfigSchema, validate_config


class TestRunConfig:
    """Tests for RunConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RunConfig()
        assert config.seed == 42
        assert config.total_steps == 1000
        assert config.n_oscillators == 60
        assert config.use_wumbo is True
        assert config.use_helix is True
        assert config.use_substrate is True

    def test_run_id_generation(self):
        """Test that run_id is auto-generated."""
        config = RunConfig()
        assert config.run_id.startswith("run_")
        assert len(config.run_id) > 10

    def test_to_dict(self):
        """Test config serialization."""
        config = RunConfig(seed=123, total_steps=500)
        data = config.to_dict()
        assert data["seed"] == 123
        assert data["total_steps"] == 500

    def test_from_dict(self):
        """Test config deserialization."""
        data = {
            "seed": 456,
            "total_steps": 2000,
            "use_wumbo": False,
        }
        config = RunConfig.from_dict(data)
        assert config.seed == 456
        assert config.total_steps == 2000
        assert config.use_wumbo is False


class TestRunLayout:
    """Tests for RunLayout."""

    def test_layout_creation(self):
        """Test directory layout creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = RunLayout("test_run", tmpdir)
            layout.create()

            assert layout.run_dir.exists()
            assert layout.logs_dir.exists()
            assert layout.metrics_dir.exists()
            assert layout.checkpoints_dir.exists()
            assert layout.eval_dir.exists()

    def test_layout_paths(self):
        """Test path properties."""
        layout = RunLayout("test_run", "/tmp")

        assert str(layout.config_path).endswith("resolved_config.yaml")
        assert str(layout.env_path).endswith("env.json")
        assert str(layout.metrics_path).endswith("metrics.jsonl")
        assert str(layout.last_checkpoint_path).endswith("last.pt")


class TestRunManager:
    """Tests for RunManager."""

    def test_create_run(self):
        """Test run creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RunManager(tmpdir)
            config = RunConfig(output_dir=tmpdir)

            layout, result = manager.create_run(config)

            assert layout.exists()
            assert layout.config_path.exists()
            assert layout.env_path.exists()
            assert result.status == RunStatus.PENDING

    def test_start_run(self):
        """Test starting a run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RunManager(tmpdir)
            config = RunConfig(output_dir=tmpdir)

            layout, result = manager.create_run(config)
            result = manager.start_run(result)

            assert result.status == RunStatus.RUNNING
            assert result.started_at != ""

    def test_list_runs(self):
        """Test listing runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = RunManager(tmpdir)

            # Create a few runs
            for i in range(3):
                config = RunConfig(output_dir=tmpdir)
                manager.create_run(config)

            runs = manager.list_runs()
            assert len(runs) >= 3


class TestGuards:
    """Tests for training guards."""

    def test_nan_guard(self):
        """Test NaN detection."""
        guard = NaNGuard()

        # No violation
        assert guard.check(1, {"loss": 0.5}) is None

        # NaN violation
        violation = guard.check(2, {"loss": float("nan")})
        assert violation is not None
        assert "NaN" in violation.message

    def test_inf_guard(self):
        """Test Inf detection."""
        guard = InfGuard()

        # No violation
        assert guard.check(1, {"loss": 0.5}) is None

        # Inf violation
        violation = guard.check(2, {"loss": float("inf")})
        assert violation is not None
        assert "Inf" in violation.message

    def test_physics_guard(self):
        """Test physics constraint checking."""
        guard = PhysicsGuard()

        # Valid κ + λ = 1
        assert guard.check(1, {"kappa": 0.6, "lambda": 0.4}) is None

        # Invalid conservation
        violation = guard.check(2, {"kappa": 0.6, "lambda": 0.6})
        assert violation is not None
        assert "conservation" in violation.message.lower()

        # z out of bounds
        violation = guard.check(3, {"z": 1.5})
        assert violation is not None
        assert "bounds" in violation.message.lower()

    def test_composite_guard(self):
        """Test composite guard."""
        guard = CompositeGuard()

        # Valid metrics
        assert guard.check(1, {"z": 0.5, "kappa": 0.6, "lambda": 0.4}) is None

        # NaN triggers first
        violation = guard.check(2, {"z": float("nan")})
        assert violation is not None


class TestDeterminism:
    """Tests for determinism management."""

    def test_seed_setting(self):
        """Test that seeds are set correctly."""
        import random
        import numpy as np

        dm = DeterminismManager(seed=12345)
        dm.setup()

        # Should produce consistent random values
        val1 = random.random()
        arr1 = np.random.rand(3)

        dm.reset()

        val2 = random.random()
        arr2 = np.random.rand(3)

        assert val1 == val2
        assert (arr1 == arr2).all()


class TestMetrics:
    """Tests for metrics collection."""

    def test_collector(self):
        """Test metrics collector."""
        collector = MetricsCollector()

        collector.record(1, {"loss": 0.5, "z": 0.4})
        collector.record(2, {"loss": 0.4, "z": 0.5})
        collector.record(3, {"loss": 0.3, "z": 0.6})

        summary = collector.get_summary()
        assert summary["total_steps"] == 3
        assert "loss" in summary["metrics"]
        assert "z" in summary["metrics"]

    def test_writer(self):
        """Test metrics writer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = MetricsWriter(tmpdir)

            with writer:
                writer.write_step(1, {"loss": 0.5})
                writer.write_step(2, {"loss": 0.4})

            metrics = writer.read_metrics()
            assert len(metrics) == 2
            assert metrics[0]["loss"] == 0.5


class TestConfigSchema:
    """Tests for configuration schema."""

    def test_validate_valid_config(self):
        """Test validation of valid config."""
        config = {
            "seed": 42,
            "total_steps": 1000,
            "log_level": "INFO",
        }

        is_valid, errors = validate_config(config)
        assert is_valid
        assert len(errors) == 0

    def test_validate_invalid_type(self):
        """Test validation catches type errors."""
        config = {
            "seed": "not_an_int",
        }

        is_valid, errors = validate_config(config)
        assert not is_valid
        assert len(errors) > 0

    def test_validate_invalid_enum(self):
        """Test validation catches invalid enum values."""
        config = {
            "log_level": "INVALID",
        }

        is_valid, errors = validate_config(config)
        assert not is_valid


class TestEnvironmentSnapshot:
    """Tests for environment capture."""

    def test_capture(self):
        """Test environment snapshot capture."""
        env = EnvironmentSnapshot.capture()

        assert env.python_version != ""
        assert env.platform != ""
        assert env.cpu_count > 0
        assert env.captured_at != ""

    def test_to_dict(self):
        """Test environment serialization."""
        env = EnvironmentSnapshot.capture()
        data = env.to_dict()

        assert "python_version" in data
        assert "platform" in data
        assert "captured_at" in data


class TestGateResult:
    """Tests for gate results."""

    def test_gate_result_creation(self):
        """Test creating gate results."""
        gate = GateResult(
            name="min_negentropy",
            passed=True,
            expected=0.5,
            actual=0.7,
            message="OK",
        )

        assert gate.name == "min_negentropy"
        assert gate.passed is True
        assert gate.actual > gate.expected


class TestEngineContract:
    """Tests for engine contract validation."""

    def test_validate_run_directory(self):
        """Test run directory validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            run_dir.mkdir()

            # Missing files
            is_valid, missing = EngineContract.validate_run_directory(run_dir)
            assert not is_valid
            assert len(missing) > 0

            # Create required files
            (run_dir / "resolved_config.yaml").write_text("test: true")
            (run_dir / "env.json").write_text("{}")
            (run_dir / "metrics").mkdir()
            (run_dir / "metrics" / "metrics.jsonl").write_text("")
            (run_dir / "checkpoints").mkdir()
            (run_dir / "checkpoints" / "last.pt").write_text("")
            (run_dir / "report.json").write_text("{}")

            is_valid, missing = EngineContract.validate_run_directory(run_dir)
            assert is_valid
            assert len(missing) == 0


class TestExitCodes:
    """Tests for exit codes."""

    def test_exit_code_values(self):
        """Test exit code values are correct."""
        assert ExitCode.SUCCESS == 0
        assert ExitCode.CONFIG_ERROR == 1
        assert ExitCode.TRAINING_FAILED == 10
        assert ExitCode.NAN_DETECTED == 11
        assert ExitCode.INTERRUPTED == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
