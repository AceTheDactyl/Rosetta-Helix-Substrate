"""
Helix Training Engine
======================

A productized training engine for Rosetta-Helix-Substrate.

Usage:
    from helix_engine import run_training

    result = run_training(config_path="configs/full.yaml")
    print(result.run_id, result.status, result.artifacts_dir)

CLI:
    helix train  --config configs/full.yaml
    helix eval   --run runs/<run_id>
    helix resume --run runs/<run_id>
    helix export --run runs/<run_id> --format onnx|torchscript|bundle
    helix nightly --config configs/nightly.yaml
"""

from helix_engine.core.contract import (
    RunConfig,
    RunResult,
    RunStatus,
    EngineContract,
    ExitCode,
)
from helix_engine.core.engine import HelixEngine
from helix_engine.core.api import run_training, evaluate_run, resume_run, export_model

__version__ = "0.1.0"
__all__ = [
    # Core types
    "RunConfig",
    "RunResult",
    "RunStatus",
    "EngineContract",
    "ExitCode",
    # Engine
    "HelixEngine",
    # Public API
    "run_training",
    "evaluate_run",
    "resume_run",
    "export_model",
]
