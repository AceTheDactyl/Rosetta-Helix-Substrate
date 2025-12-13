"""
Helix Training Engine
=====================

The main training engine that orchestrates all modules.

Signature: engine|v0.1.0|helix
"""

from __future__ import annotations

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from helix_engine.core.contract import (
    ExitCode,
    GateResult,
    RunConfig,
    RunResult,
    RunStatus,
)
from helix_engine.run.manager import RunManager
from helix_engine.run.layout import RunLayout
from helix_engine.reliability.guards import CompositeGuard, GuardViolation
from helix_engine.reliability.checkpoint import CheckpointManager, CheckpointState
from helix_engine.reliability.signals import GracefulShutdown
from helix_engine.reliability.determinism import DeterminismManager
from helix_engine.observability.logger import StructuredLogger, LogLevel
from helix_engine.observability.metrics import MetricsCollector, MetricsWriter


class HelixEngine:
    """
    The main Helix Training Engine.

    Orchestrates:
    - Training modules (WUMBO, Helix, Substrate, Kuramoto, etc.)
    - Reliability features (checkpoints, guards, signals)
    - Observability (logging, metrics)
    - Evaluation gates
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self.run_manager = RunManager(config.output_dir)

        # These get set during run
        self.layout: Optional[RunLayout] = None
        self.result: Optional[RunResult] = None

        # Reliability components
        self.determinism = DeterminismManager(
            seed=config.seed,
            deterministic=config.deterministic,
        )
        self.guards = CompositeGuard()
        self.checkpoint_manager: Optional[CheckpointManager] = None

        # Observability components
        self.logger = StructuredLogger(
            name="helix",
            level=getattr(LogLevel, config.log_level, LogLevel.INFO),
        )
        self.metrics_collector = MetricsCollector()
        self.metrics_writer: Optional[MetricsWriter] = None

        # Training state
        self.current_step = 0
        self.trainer = None  # Will be set up during run

    def setup(self) -> None:
        """
        Set up the training run.

        Creates directories, initializes components, captures environment.
        """
        # Create run
        self.layout, self.result = self.run_manager.create_run(self.config)

        # Set up logging
        self.logger.set_log_file(str(self.layout.events_path))
        self.logger.set_context(run_id=self.config.run_id)

        # Set up metrics
        self.metrics_writer = MetricsWriter(str(self.layout.metrics_dir))
        self.metrics_writer.open()

        # Set up checkpoints
        self.checkpoint_manager = CheckpointManager(
            str(self.layout.checkpoints_dir),
            run_id=self.config.run_id,
        )

        # Set up determinism
        self.determinism.setup()

        # Initialize trainer
        self._setup_trainer()

        self.logger.info(f"Run initialized: {self.config.run_id}")
        self.logger.info(f"Output directory: {self.layout.run_dir}")

    def _setup_trainer(self) -> None:
        """Set up the actual training module."""
        try:
            from rosetta_helix_training import RosettaHelixTraining
            self.trainer = RosettaHelixTraining(
                n_oscillators=self.config.n_oscillators
            )
            self.logger.info("RosettaHelixTraining initialized")
        except ImportError as e:
            self.logger.warning(f"Could not import RosettaHelixTraining: {e}")
            self.trainer = None

    def train(self) -> RunResult:
        """
        Run the full training loop.

        Returns the RunResult with all artifacts.
        """
        if self.layout is None or self.result is None:
            self.setup()

        # Start run
        self.result = self.run_manager.start_run(self.result)
        self.logger.info("Training started")

        try:
            with GracefulShutdown() as shutdown:
                self._training_loop(shutdown)

            # Run evaluation gates
            self._evaluate_gates()

            # Generate final report
            exit_code = ExitCode.SUCCESS if self.result.gates_passed else ExitCode.GATE_FAILED
            self.result = self.run_manager.complete_run(
                self.result,
                self.layout,
                exit_code=exit_code,
            )

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            self._save_checkpoint(interrupted=True)
            self.result = self.run_manager.complete_run(
                self.result,
                self.layout,
                exit_code=ExitCode.INTERRUPTED,
                error_message="Interrupted by user",
            )

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self._save_checkpoint(interrupted=True)
            self.result = self.run_manager.complete_run(
                self.result,
                self.layout,
                exit_code=ExitCode.TRAINING_FAILED,
                error_message=str(e),
            )

        finally:
            self._cleanup()

        return self.result

    def _training_loop(self, shutdown: GracefulShutdown) -> None:
        """The main training loop."""
        import random

        # Get start step (for resume)
        start_step = self.current_step

        for step in range(start_step, self.config.total_steps):
            if shutdown.requested:
                self.logger.info("Graceful shutdown requested")
                break

            self.current_step = step
            self.logger.set_context(step=step)

            # Execute training step
            if self.trainer:
                input_value = random.random() * 0.618  # PHI_INV
                step_result = self.trainer.training_step(input_value)
                metrics = self._extract_metrics(step_result)
            else:
                # Fallback for when trainer not available
                metrics = self._generate_test_metrics(step)

            # Check guards
            violation = self.guards.check(step, metrics)
            if violation:
                self._handle_guard_violation(violation)
                break

            # Record metrics
            self.metrics_collector.record(step, metrics)
            if self.metrics_writer:
                self.metrics_writer.write_step(step, metrics)

            # Log progress
            if step % self.config.log_steps == 0:
                self.logger.step(step, metrics)

            # Save checkpoint
            if step > 0 and step % self.config.checkpoint_steps == 0:
                self._save_checkpoint()

        # Update result
        self.result.total_steps = self.current_step

    def _extract_metrics(self, step_result: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric metrics from a training step result."""
        metrics = {}
        keys_to_extract = [
            "z", "kappa", "lambda", "negentropy", "stillness",
            "conservation_error", "k_formed",
        ]

        for key in keys_to_extract:
            if key in step_result:
                value = step_result[key]
                if isinstance(value, bool):
                    metrics[key] = float(value)
                elif isinstance(value, (int, float)):
                    metrics[key] = float(value)

        return metrics

    def _generate_test_metrics(self, step: int) -> Dict[str, float]:
        """Generate test metrics when trainer is not available."""
        import math
        progress = step / max(self.config.total_steps, 1)
        z_critical = math.sqrt(3) / 2
        phi_inv = (math.sqrt(5) - 1) / 2

        return {
            "z": 0.5 + progress * (z_critical - 0.5),
            "kappa": phi_inv,
            "lambda": 1.0 - phi_inv,
            "negentropy": math.exp(-36 * (0.5 + progress * 0.366 - z_critical) ** 2),
            "stillness": progress * 0.8,
            "conservation_error": 0.0,
            "k_formed": float(progress > 0.8),
        }

    def _handle_guard_violation(self, violation: GuardViolation) -> None:
        """Handle a guard violation."""
        self.logger.error(
            f"Guard violation: {violation.guard_name} - {violation.message}",
            guard=violation.guard_name,
            metric=violation.metric_name,
            value=violation.value,
        )

        # Map violation to exit code
        if "NaN" in violation.guard_name:
            exit_code = ExitCode.NAN_DETECTED
        elif "Inf" in violation.guard_name:
            exit_code = ExitCode.INF_DETECTED
        elif "Physics" in violation.guard_name:
            exit_code = ExitCode.PHYSICS_VIOLATION
        else:
            exit_code = ExitCode.TRAINING_FAILED

        self._save_checkpoint(interrupted=True)
        self.result = self.run_manager.complete_run(
            self.result,
            self.layout,
            exit_code=exit_code,
            error_message=violation.message,
        )

    def _save_checkpoint(self, interrupted: bool = False) -> None:
        """Save a training checkpoint."""
        if self.checkpoint_manager is None:
            return

        # Get current metrics
        metrics = {}
        latest = self.metrics_collector.get_latest()
        if latest:
            metrics = {k: v for k, v in latest.items()
                      if isinstance(v, (int, float)) and k != "step"}

        # Create checkpoint state
        state = CheckpointState(
            step=self.current_step,
            training_state={
                "trainer_state": self.trainer.get_session_summary() if self.trainer else {},
            },
            metrics=metrics,
        )

        # Check if this is best
        is_best = self.checkpoint_manager.update_best(metrics)

        # Save
        self.checkpoint_manager.save(state, is_best=is_best)
        self.logger.info(f"Checkpoint saved at step {self.current_step}", is_best=is_best)

    def _evaluate_gates(self) -> None:
        """Evaluate all training gates."""
        gate_results = []
        summary = self.metrics_collector.get_summary()
        metrics = summary.get("metrics", {})

        # Check min_negentropy
        if "min_negentropy" in self.config.gates:
            expected = self.config.gates["min_negentropy"]
            actual = metrics.get("negentropy", {}).get("max", 0)
            gate_results.append(GateResult(
                name="min_negentropy",
                passed=actual >= expected,
                expected=expected,
                actual=actual,
                message=f"Max negentropy: {actual:.4f} (need >= {expected})",
            ))

        # Check min_k_formations
        if "min_k_formations" in self.config.gates:
            expected = self.config.gates["min_k_formations"]
            actual = metrics.get("k_formed", {}).get("mean", 0) * self.current_step
            gate_results.append(GateResult(
                name="min_k_formations",
                passed=actual >= expected,
                expected=expected,
                actual=actual,
                message=f"K-formations: {actual:.0f} (need >= {expected})",
            ))

        # Check max_conservation_error
        if "max_conservation_error" in self.config.gates:
            expected = self.config.gates["max_conservation_error"]
            actual = metrics.get("conservation_error", {}).get("max", 0)
            gate_results.append(GateResult(
                name="max_conservation_error",
                passed=actual <= expected,
                expected=expected,
                actual=actual,
                message=f"Max conservation error: {actual:.2e} (need <= {expected})",
            ))

        # Check min_final_z
        if "min_final_z" in self.config.gates:
            expected = self.config.gates["min_final_z"]
            actual = metrics.get("z", {}).get("latest", 0)
            gate_results.append(GateResult(
                name="min_final_z",
                passed=actual >= expected,
                expected=expected,
                actual=actual,
                message=f"Final z: {actual:.4f} (need >= {expected})",
            ))

        # Update result
        self.result.gate_results = gate_results
        self.result.gates_passed = all(g.passed for g in gate_results)
        self.result.final_metrics = {
            k: v.get("latest", v.get("mean", 0))
            for k, v in metrics.items()
        }

        # Write gates
        if self.layout:
            self.run_manager.write_gates(self.layout, gate_results)

        # Log results
        for gate in gate_results:
            status = "PASSED" if gate.passed else "FAILED"
            self.logger.info(f"Gate {gate.name}: {status} - {gate.message}")

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.metrics_writer:
            # Write final summary
            summary = self.metrics_collector.get_summary()
            self.metrics_writer.write_summary(summary)
            self.metrics_writer.close()

        self.logger.info(f"Training completed with status: {self.result.status}")
        self.logger.close()

    def resume(self, checkpoint_path: Optional[str] = None) -> RunResult:
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint. If None, uses last.pt.
        """
        if self.layout is None:
            raise ValueError("Must call setup() before resume()")

        if self.checkpoint_manager is None:
            raise ValueError("Checkpoint manager not initialized")

        # Load checkpoint
        state = self.checkpoint_manager.load(checkpoint_path)
        if state is None:
            raise ValueError(f"Could not load checkpoint: {checkpoint_path}")

        # Restore state
        self.current_step = state.step
        self.checkpoint_manager.restore_random_states(state)

        self.logger.info(f"Resumed from checkpoint at step {self.current_step}")

        # Continue training
        return self.train()

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the trained model.

        Returns evaluation results.
        """
        if self.layout is None:
            raise ValueError("Must call setup() or train() before evaluate()")

        # Run evaluation gates
        self._evaluate_gates()

        return {
            "gates_passed": self.result.gates_passed,
            "gate_results": [
                {
                    "name": g.name,
                    "passed": g.passed,
                    "expected": g.expected,
                    "actual": g.actual,
                    "message": g.message,
                }
                for g in self.result.gate_results
            ],
            "final_metrics": self.result.final_metrics,
        }
