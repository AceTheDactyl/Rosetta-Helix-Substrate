"""
Public Python API
=================

High-level functions for using the Helix Training Engine.

Usage:
    from helix_engine import run_training, evaluate_run, resume_run, export_model

    result = run_training(config_path="configs/full.yaml")
    print(result.run_id, result.status, result.artifacts_dir)

Signature: api|v0.1.0|helix
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from helix_engine.core.contract import RunConfig, RunResult, ExitCode
from helix_engine.core.engine import HelixEngine
from helix_engine.run.manager import RunManager
from helix_engine.registry.model_registry import ModelRegistry, ModelEntry


def run_training(
    config_path: Optional[str] = None,
    config: Optional[Union[RunConfig, Dict[str, Any]]] = None,
    **kwargs
) -> RunResult:
    """
    Run a training session.

    Args:
        config_path: Path to YAML configuration file
        config: RunConfig object or dict (overrides config_path)
        **kwargs: Override specific config values

    Returns:
        RunResult with all training artifacts and metrics

    Example:
        result = run_training(config_path="configs/full.yaml")
        print(f"Run {result.run_id} completed with status {result.status}")
        print(f"Artifacts at: {result.artifacts_dir}")
    """
    # Build config
    if config is not None:
        if isinstance(config, dict):
            run_config = RunConfig.from_dict(config)
        else:
            run_config = config
    elif config_path is not None:
        run_config = load_config(config_path)
    else:
        run_config = RunConfig()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(run_config, key):
            setattr(run_config, key, value)

    # Run training
    engine = HelixEngine(run_config)
    return engine.train()


def evaluate_run(
    run_id: str,
    runs_dir: str = "runs",
) -> Dict[str, Any]:
    """
    Evaluate a completed training run.

    Args:
        run_id: ID of the run to evaluate
        runs_dir: Directory containing runs

    Returns:
        Evaluation results including gate checks

    Example:
        results = evaluate_run("run_20231201_120000_abc123")
        print(f"Gates passed: {results['gates_passed']}")
    """
    run_manager = RunManager(runs_dir)
    loaded = run_manager.load_run(run_id)

    if loaded is None:
        return {"error": f"Run not found: {run_id}"}

    layout, result = loaded
    engine = HelixEngine(result.config)
    engine.layout = layout
    engine.result = result

    return engine.evaluate()


def resume_run(
    run_id: str,
    checkpoint_path: Optional[str] = None,
    runs_dir: str = "runs",
) -> RunResult:
    """
    Resume a training run from checkpoint.

    Args:
        run_id: ID of the run to resume
        checkpoint_path: Optional path to specific checkpoint
        runs_dir: Directory containing runs

    Returns:
        RunResult after training completion

    Example:
        result = resume_run("run_20231201_120000_abc123")
    """
    run_manager = RunManager(runs_dir)
    loaded = run_manager.load_run(run_id)

    if loaded is None:
        raise ValueError(f"Run not found: {run_id}")

    layout, result = loaded
    engine = HelixEngine(result.config)
    engine.layout = layout
    engine.result = result

    return engine.resume(checkpoint_path)


def export_model(
    run_id: str,
    format: str = "bundle",
    output_path: Optional[str] = None,
    runs_dir: str = "runs",
) -> str:
    """
    Export a trained model.

    Args:
        run_id: ID of the run to export
        format: Export format (onnx, torchscript, bundle)
        output_path: Optional custom output path
        runs_dir: Directory containing runs

    Returns:
        Path to exported model

    Example:
        path = export_model("run_20231201_120000_abc123", format="onnx")
    """
    from helix_engine.run.layout import RunLayout
    import shutil

    layout = RunLayout(run_id, runs_dir)
    if not layout.exists():
        raise ValueError(f"Run not found: {run_id}")

    # Get checkpoint
    checkpoint = layout.get_latest_checkpoint()
    if checkpoint is None:
        raise ValueError(f"No checkpoint found for run: {run_id}")

    # Determine output path
    if output_path is None:
        output_path = str(layout.export_path(format))

    # Create exports directory
    layout.exports_dir.mkdir(exist_ok=True)

    if format == "bundle":
        # Create a bundle with config + checkpoint
        import json
        import zipfile

        with zipfile.ZipFile(output_path, "w") as zf:
            # Add checkpoint
            zf.write(checkpoint, "model.pt")

            # Add config
            if layout.config_path.exists():
                zf.write(layout.config_path, "config.yaml")

            # Add metadata
            metadata = {
                "run_id": run_id,
                "format": "helix_bundle",
                "version": "0.1.0",
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))

    elif format in ("pt", "torchscript"):
        # Just copy the checkpoint
        shutil.copy2(checkpoint, output_path)

    elif format == "onnx":
        # ONNX export would require torch model
        # For now, just note it's not implemented
        raise NotImplementedError("ONNX export requires model definition")

    else:
        raise ValueError(f"Unknown export format: {format}")

    return output_path


def promote_model(
    run_id: str,
    name: str,
    version: Optional[str] = None,
    tags: Optional[list] = None,
    description: str = "",
    runs_dir: str = "runs",
    registry_dir: str = "models",
) -> ModelEntry:
    """
    Promote a training run to a named model.

    Args:
        run_id: ID of the run to promote
        name: Model name
        version: Version string (auto-generated if not provided)
        tags: Optional list of tags
        description: Optional description
        runs_dir: Directory containing runs
        registry_dir: Directory for model registry

    Returns:
        ModelEntry for the promoted model

    Example:
        entry = promote_model("run_20231201_120000_abc123", "rosetta_v1")
        print(f"Promoted to {entry.name}:{entry.version}")
    """
    registry = ModelRegistry(registry_dir)
    return registry.promote(
        run_id=run_id,
        name=name,
        version=version,
        tags=tags,
        description=description,
        runs_dir=runs_dir,
    )


def load_config(config_path: str) -> RunConfig:
    """
    Load a RunConfig from a YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        RunConfig object
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return RunConfig.from_dict(data)


def list_runs(
    status: Optional[str] = None,
    runs_dir: str = "runs",
) -> list:
    """
    List all training runs.

    Args:
        status: Optional status filter
        runs_dir: Directory containing runs

    Returns:
        List of run information dicts
    """
    run_manager = RunManager(runs_dir)
    return run_manager.list_runs(status)


def get_run(
    run_id: str,
    runs_dir: str = "runs",
) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific run.

    Args:
        run_id: ID of the run
        runs_dir: Directory containing runs

    Returns:
        Run information dict or None if not found
    """
    run_manager = RunManager(runs_dir)
    loaded = run_manager.load_run(run_id)

    if loaded is None:
        return None

    layout, result = loaded
    return result.to_dict()
