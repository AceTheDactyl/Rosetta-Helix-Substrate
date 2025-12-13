"""
Model Registry
==============

Manages model artifacts and promotion:
- Store trained models with metadata
- Promote runs to named models
- Track model lineage

Signature: registry|v0.1.0|helix
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelEntry:
    """
    Entry in the model registry.
    """
    name: str
    version: str
    run_id: str

    # Metadata
    tags: List[str] = field(default_factory=list)
    description: str = ""
    created_at: str = ""

    # Metrics at time of promotion
    metrics: Dict[str, float] = field(default_factory=dict)
    gates_passed: bool = False

    # Artifact paths
    checkpoint_path: str = ""
    config_path: str = ""
    export_paths: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "run_id": self.run_id,
            "tags": self.tags,
            "description": self.description,
            "created_at": self.created_at,
            "metrics": self.metrics,
            "gates_passed": self.gates_passed,
            "checkpoint_path": self.checkpoint_path,
            "config_path": self.config_path,
            "export_paths": self.export_paths,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEntry":
        return cls(
            name=data["name"],
            version=data["version"],
            run_id=data["run_id"],
            tags=data.get("tags", []),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            metrics=data.get("metrics", {}),
            gates_passed=data.get("gates_passed", False),
            checkpoint_path=data.get("checkpoint_path", ""),
            config_path=data.get("config_path", ""),
            export_paths=data.get("export_paths", {}),
        )


class ModelRegistry:
    """
    Registry for managing trained models.
    """

    def __init__(self, registry_dir: str = "models"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.registry_dir / "index.json"
        self._index: Dict[str, List[ModelEntry]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load the registry index."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                data = json.load(f)
            for name, versions in data.items():
                self._index[name] = [ModelEntry.from_dict(v) for v in versions]

    def _save_index(self) -> None:
        """Save the registry index."""
        data = {
            name: [v.to_dict() for v in versions]
            for name, versions in self._index.items()
        }
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)

    def promote(
        self,
        run_id: str,
        name: str,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        runs_dir: str = "runs",
    ) -> ModelEntry:
        """
        Promote a run to a named model.

        Args:
            run_id: The run to promote
            name: Model name
            version: Version string (auto-generated if not provided)
            tags: Optional tags
            description: Optional description
            runs_dir: Directory containing runs

        Returns:
            The created ModelEntry
        """
        runs_path = Path(runs_dir)
        run_path = runs_path / run_id

        if not run_path.exists():
            raise ValueError(f"Run not found: {run_id}")

        # Auto-generate version if not provided
        if version is None:
            existing = self._index.get(name, [])
            version = f"v{len(existing) + 1}"

        # Create model directory
        model_dir = self.registry_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifacts
        checkpoint_src = run_path / "checkpoints" / "best.pt"
        if not checkpoint_src.exists():
            checkpoint_src = run_path / "checkpoints" / "last.pt"

        if checkpoint_src.exists():
            checkpoint_dst = model_dir / "model.pt"
            shutil.copy2(checkpoint_src, checkpoint_dst)
        else:
            checkpoint_dst = None

        config_src = run_path / "resolved_config.yaml"
        config_dst = model_dir / "config.yaml"
        if config_src.exists():
            shutil.copy2(config_src, config_dst)

        # Load report for metrics
        report_path = run_path / "report.json"
        metrics = {}
        gates_passed = False
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            metrics = report.get("final_metrics", {})
            gates_passed = report.get("gates_passed", False)

        # Create entry
        entry = ModelEntry(
            name=name,
            version=version,
            run_id=run_id,
            tags=tags or [],
            description=description,
            created_at=datetime.utcnow().isoformat(),
            metrics=metrics,
            gates_passed=gates_passed,
            checkpoint_path=str(checkpoint_dst) if checkpoint_dst else "",
            config_path=str(config_dst),
        )

        # Update index
        if name not in self._index:
            self._index[name] = []
        self._index[name].append(entry)
        self._save_index()

        return entry

    def get(self, name: str, version: Optional[str] = None) -> Optional[ModelEntry]:
        """
        Get a model by name and version.

        If version is None, returns the latest version.
        """
        if name not in self._index:
            return None

        versions = self._index[name]
        if not versions:
            return None

        if version is None:
            return versions[-1]

        for entry in versions:
            if entry.version == version:
                return entry

        return None

    def list_models(self) -> List[str]:
        """List all model names."""
        return list(self._index.keys())

    def list_versions(self, name: str) -> List[ModelEntry]:
        """List all versions of a model."""
        return self._index.get(name, [])

    def delete(self, name: str, version: Optional[str] = None) -> bool:
        """
        Delete a model or specific version.

        If version is None, deletes all versions.
        """
        if name not in self._index:
            return False

        if version is None:
            # Delete all versions
            model_dir = self.registry_dir / name
            if model_dir.exists():
                shutil.rmtree(model_dir)
            del self._index[name]
        else:
            # Delete specific version
            versions = self._index[name]
            new_versions = [v for v in versions if v.version != version]
            if len(new_versions) == len(versions):
                return False
            self._index[name] = new_versions

            version_dir = self.registry_dir / name / version
            if version_dir.exists():
                shutil.rmtree(version_dir)

        self._save_index()
        return True

    def search(
        self,
        tags: Optional[List[str]] = None,
        gates_passed: Optional[bool] = None,
    ) -> List[ModelEntry]:
        """
        Search models by criteria.
        """
        results = []

        for versions in self._index.values():
            for entry in versions:
                # Filter by tags
                if tags:
                    if not all(t in entry.tags for t in tags):
                        continue

                # Filter by gates_passed
                if gates_passed is not None:
                    if entry.gates_passed != gates_passed:
                        continue

                results.append(entry)

        return results
