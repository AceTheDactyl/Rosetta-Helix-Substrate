"""
Engine Contract
===============

Defines the non-negotiable outputs every training run must produce:
- Run ID (stable, unique)
- Config snapshot (exact resolved config used)
- Environment snapshot (python, torch, cuda, git commit, hardware)
- Structured metrics stream (metrics.jsonl)
- Checkpoints (best, last, plus optional periodic)
- Final report (report.json with pass/fail + reasons)
- Exit codes that mean something

Signature: engine-contract|v0.1.0|helix
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json


class ExitCode(IntEnum):
    """
    Standardized exit codes for the training engine.

    0 = success
    1-9 = user/config errors
    10-19 = training failures
    20-29 = resource errors
    30-39 = infrastructure errors
    """
    SUCCESS = 0

    # User/config errors (1-9)
    CONFIG_ERROR = 1
    CONFIG_NOT_FOUND = 2
    INVALID_MODULE = 3
    INVALID_CHECKPOINT = 4

    # Training failures (10-19)
    TRAINING_FAILED = 10
    NAN_DETECTED = 11
    INF_DETECTED = 12
    CONVERGENCE_FAILED = 13
    GATE_FAILED = 14
    PHYSICS_VIOLATION = 15

    # Resource errors (20-29)
    OUT_OF_MEMORY = 20
    DISK_FULL = 21

    # Infrastructure errors (30-39)
    INTERRUPTED = 30
    TIMEOUT = 31
    UNKNOWN_ERROR = 99


class RunStatus:
    """Run status constants."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class EnvironmentSnapshot:
    """
    Captures the exact environment for reproducibility.
    """
    python_version: str = ""
    platform: str = ""
    platform_version: str = ""
    cpu_count: int = 0
    hostname: str = ""

    # Package versions
    numpy_version: str = ""
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    cuda_available: bool = False

    # Git info
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: bool = False

    # Timestamp
    captured_at: str = ""

    @classmethod
    def capture(cls) -> "EnvironmentSnapshot":
        """Capture current environment."""
        snapshot = cls()

        # Python and platform
        snapshot.python_version = sys.version
        snapshot.platform = platform.system()
        snapshot.platform_version = platform.version()
        snapshot.cpu_count = os.cpu_count() or 0
        snapshot.hostname = platform.node()

        # Package versions
        try:
            import numpy
            snapshot.numpy_version = numpy.__version__
        except ImportError:
            pass

        try:
            import torch
            snapshot.torch_version = torch.__version__
            snapshot.cuda_available = torch.cuda.is_available()
            if snapshot.cuda_available:
                snapshot.cuda_version = torch.version.cuda
        except ImportError:
            pass

        # Git info
        try:
            snapshot.git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            snapshot.git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            # Check if dirty
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True
            )
            snapshot.git_dirty = bool(result.stdout.strip())
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        snapshot.captured_at = datetime.utcnow().isoformat()
        return snapshot

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "platform_version": self.platform_version,
            "cpu_count": self.cpu_count,
            "hostname": self.hostname,
            "numpy_version": self.numpy_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "cuda_available": self.cuda_available,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "captured_at": self.captured_at,
        }


@dataclass
class ModuleConfig:
    """Configuration for a single training module."""
    name: str
    enabled: bool = True
    steps: int = 100
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """
    Complete training configuration.

    This is the resolved configuration that gets saved with each run.
    """
    # Run identification
    run_id: str = ""
    run_name: str = ""
    tags: List[str] = field(default_factory=list)

    # Training settings
    seed: int = 42
    deterministic: bool = False

    # Module configuration
    modules: Dict[str, ModuleConfig] = field(default_factory=dict)

    # Module flags (convenience)
    use_wumbo: bool = True
    use_helix: bool = True
    use_substrate: bool = True
    use_kuramoto: bool = True
    use_feedback_loop: bool = True
    use_apl_engine: bool = True

    # Training hyperparameters
    total_steps: int = 1000
    warmup_steps: int = 100
    eval_steps: int = 100
    checkpoint_steps: int = 500
    log_steps: int = 10

    # Physics constants (immutable - from physics_constants.py)
    n_oscillators: int = 60

    # Paths
    output_dir: str = "runs"
    checkpoint_dir: str = ""

    # Evaluation gates
    gates: Dict[str, float] = field(default_factory=lambda: {
        "min_negentropy": 0.5,
        "min_k_formations": 1,
        "max_conservation_error": 1e-6,
        "min_final_z": 0.75,
    })

    # Logging
    log_level: str = "INFO"
    enable_tensorboard: bool = False
    enable_wandb: bool = False

    def __post_init__(self):
        """Generate run_id if not provided."""
        if not self.run_id:
            self.run_id = self._generate_run_id()
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(self.output_dir, self.run_id, "checkpoints")

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        # Include config hash for uniqueness
        config_str = f"{self.seed}_{self.total_steps}_{self.n_oscillators}"
        hash_suffix = hashlib.md5(config_str.encode()).hexdigest()[:6]
        return f"run_{timestamp}_{hash_suffix}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        modules_dict = {}
        for name, mod in self.modules.items():
            modules_dict[name] = {
                "name": mod.name,
                "enabled": mod.enabled,
                "steps": mod.steps,
                "params": mod.params,
            }

        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "tags": self.tags,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "modules": modules_dict,
            "use_wumbo": self.use_wumbo,
            "use_helix": self.use_helix,
            "use_substrate": self.use_substrate,
            "use_kuramoto": self.use_kuramoto,
            "use_feedback_loop": self.use_feedback_loop,
            "use_apl_engine": self.use_apl_engine,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "eval_steps": self.eval_steps,
            "checkpoint_steps": self.checkpoint_steps,
            "log_steps": self.log_steps,
            "n_oscillators": self.n_oscillators,
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "gates": self.gates,
            "log_level": self.log_level,
            "enable_tensorboard": self.enable_tensorboard,
            "enable_wandb": self.enable_wandb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        """Create from dictionary."""
        # Handle modules specially
        modules = {}
        for name, mod_data in data.get("modules", {}).items():
            modules[name] = ModuleConfig(
                name=mod_data["name"],
                enabled=mod_data.get("enabled", True),
                steps=mod_data.get("steps", 100),
                params=mod_data.get("params", {}),
            )

        return cls(
            run_id=data.get("run_id", ""),
            run_name=data.get("run_name", ""),
            tags=data.get("tags", []),
            seed=data.get("seed", 42),
            deterministic=data.get("deterministic", False),
            modules=modules,
            use_wumbo=data.get("use_wumbo", True),
            use_helix=data.get("use_helix", True),
            use_substrate=data.get("use_substrate", True),
            use_kuramoto=data.get("use_kuramoto", True),
            use_feedback_loop=data.get("use_feedback_loop", True),
            use_apl_engine=data.get("use_apl_engine", True),
            total_steps=data.get("total_steps", 1000),
            warmup_steps=data.get("warmup_steps", 100),
            eval_steps=data.get("eval_steps", 100),
            checkpoint_steps=data.get("checkpoint_steps", 500),
            log_steps=data.get("log_steps", 10),
            n_oscillators=data.get("n_oscillators", 60),
            output_dir=data.get("output_dir", "runs"),
            checkpoint_dir=data.get("checkpoint_dir", ""),
            gates=data.get("gates", {}),
            log_level=data.get("log_level", "INFO"),
            enable_tensorboard=data.get("enable_tensorboard", False),
            enable_wandb=data.get("enable_wandb", False),
        )


@dataclass
class GateResult:
    """Result of an evaluation gate check."""
    name: str
    passed: bool
    expected: float
    actual: float
    message: str = ""


@dataclass
class RunResult:
    """
    Complete result of a training run.

    Every training run produces this standardized output.
    """
    # Identification
    run_id: str
    config: RunConfig

    # Status
    status: str = RunStatus.PENDING
    exit_code: ExitCode = ExitCode.SUCCESS
    error_message: str = ""

    # Timing
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0

    # Environment
    environment: Optional[EnvironmentSnapshot] = None

    # Training metrics summary
    total_steps: int = 0
    final_metrics: Dict[str, float] = field(default_factory=dict)

    # Gate results
    gates_passed: bool = False
    gate_results: List[GateResult] = field(default_factory=list)

    # Paths to artifacts
    artifacts_dir: str = ""
    config_path: str = ""
    metrics_path: str = ""
    checkpoints: Dict[str, str] = field(default_factory=dict)
    report_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "status": self.status,
            "exit_code": int(self.exit_code),
            "exit_code_name": self.exit_code.name,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "environment": self.environment.to_dict() if self.environment else None,
            "total_steps": self.total_steps,
            "final_metrics": self.final_metrics,
            "gates_passed": self.gates_passed,
            "gate_results": [
                {
                    "name": g.name,
                    "passed": g.passed,
                    "expected": g.expected,
                    "actual": g.actual,
                    "message": g.message,
                }
                for g in self.gate_results
            ],
            "artifacts_dir": self.artifacts_dir,
            "config_path": self.config_path,
            "metrics_path": self.metrics_path,
            "checkpoints": self.checkpoints,
            "report_path": self.report_path,
        }

    def save_report(self, path: str) -> None:
        """Save result as JSON report."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        self.report_path = path


class EngineContract:
    """
    Defines what every training run MUST produce.

    This is the non-negotiable contract for the engine.
    """

    # Required output files
    REQUIRED_FILES = [
        "resolved_config.yaml",
        "env.json",
        "metrics/metrics.jsonl",
        "checkpoints/last.pt",
        "report.json",
    ]

    # Required report fields
    REQUIRED_REPORT_FIELDS = [
        "run_id",
        "status",
        "exit_code",
        "started_at",
        "completed_at",
        "total_steps",
        "gates_passed",
    ]

    @classmethod
    def validate_run_directory(cls, run_dir: Union[str, Path]) -> tuple[bool, List[str]]:
        """
        Validate that a run directory contains all required outputs.

        Returns (is_valid, list_of_missing_files)
        """
        run_path = Path(run_dir)
        missing = []

        for required_file in cls.REQUIRED_FILES:
            if not (run_path / required_file).exists():
                missing.append(required_file)

        return len(missing) == 0, missing

    @classmethod
    def validate_report(cls, report: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate that a report contains all required fields.

        Returns (is_valid, list_of_missing_fields)
        """
        missing = []

        for field in cls.REQUIRED_REPORT_FIELDS:
            if field not in report:
                missing.append(field)

        return len(missing) == 0, missing
