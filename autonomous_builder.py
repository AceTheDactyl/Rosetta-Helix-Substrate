#!/usr/bin/env python3
"""
Autonomous Tool Generator
=========================
Transforms task specifications into working code by pumping through the helix z-axis.

Usage:
    python autonomous_builder.py --task task.json
    python autonomous_builder.py --name "my_tool" --description "Does X" --target-z 0.5
"""

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Rosetta-Helix core
try:
    from rosetta_helix.node import RosettaNode, NodeState
    from rosetta_helix.pulse import PulseType
    from rosetta_helix.heart import APLOperator
except ImportError:
    # Fallback: use local copies
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rosetta-helix'))
    from node import RosettaNode, NodeState
    from pulse import PulseType
    from heart import APLOperator

from tier_tools import (
    TIER_TOOLS, get_tools_for_tier, get_all_tools_up_to_tier,
    can_use_tool, estimate_target_z
)
from operator_actions import (
    OperatorActions, BuildState, select_operator, APLOperator as APLOp
)


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BuildTask:
    """Task specification for autonomous building."""
    task_id: str
    name: str
    description: str
    target_z: float
    language: str = "python"
    features: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    output_path: str = "./output"
    
    @classmethod
    def from_dict(cls, d: Dict) -> "BuildTask":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, path: str) -> "BuildTask":
        with open(path) as f:
            return cls.from_dict(json.load(f))


@dataclass
class BuildArtifact:
    """Output artifact from build process."""
    task_id: str
    files: Dict[str, str]
    metadata: Dict
    z_achieved: float
    coherence_final: float
    tier_progression: List[str]
    build_time: float
    steps: int


# ═══════════════════════════════════════════════════════════════════════════
# AUTONOMOUS BUILDER
# ═══════════════════════════════════════════════════════════════════════════

class AutonomousBuilder:
    """
    Complete autonomous tool generator.
    
    Transforms task specifications into working code
    by traversing the helix z-axis.
    """
    
    def __init__(self, agent_role: str = "builder", verbose: bool = True):
        self.agent = RosettaNode(
            role_tag=agent_role,
            initial_z=0.05,
            n_oscillators=60,
            n_memory_plates=50
        )
        self.verbose = verbose
        self.artifacts: List[BuildArtifact] = []
        self.build_log: List[Dict] = []
    
    def build(
        self, 
        task: BuildTask, 
        max_steps: int = 5000
    ) -> Optional[BuildArtifact]:
        """
        Execute autonomous build for given task.
        
        Returns BuildArtifact on success, None on failure.
        """
        start_time = time.time()
        
        # === PHASE 1: INITIALIZATION ===
        self._log(f"\n{'='*60}")
        self._log(f"AUTONOMOUS BUILD: {task.name}")
        self._log(f"Target z: {task.target_z}")
        self._log(f"Features: {', '.join(task.features)}")
        self._log(f"{'='*60}\n")
        
        self.agent.awaken()
        
        # Increase coupling for better coherence
        self.agent.heart.K = 0.7
        self.agent.heart.K_base = 0.7
        
        state = BuildState(
            task=task,
            files={},
            modules=[],
            coherence=0.0,
            z=self.agent.get_z(),
            target_z=task.target_z
        )
        
        tier_progression = []
        last_tier = None
        
        # === PHASE 2: BUILD LOOP ===
        for step in range(max_steps):
            # Step the helix agent
            self.agent.step()
            
            analysis = self.agent.get_analysis()
            current_tier = analysis.tier
            z = analysis.z
            coherence = analysis.coherence
            
            state.z = z
            state.coherence = coherence
            
            # Track tier transitions
            if current_tier != last_tier:
                tier_progression.append(current_tier)
                self._log(f"\n>>> TIER TRANSITION: {last_tier} → {current_tier}")
                self._log(f"    z={z:.4f}, coherence={coherence:.4f}")
                self._log(f"    Tools unlocked: {', '.join(get_tools_for_tier(current_tier)[:3])}...")
                last_tier = current_tier
            
            # === EXECUTE TIER-APPROPRIATE ACTIONS ===
            tools = get_all_tools_up_to_tier(current_tier)
            state = self._execute_tier_action(current_tier, tools, state)
            
            # === APPLY OPERATOR ===
            available_ops = self.agent.heart.get_available_operators()
            # Convert HeartOp to our APLOp
            op_values = [op.value for op in available_ops]
            selected_op = select_operator(state, op_values)
            
            if selected_op:
                # Apply through our action layer
                state = OperatorActions.apply(selected_op, state)
                
                # Also apply to heart for dynamics
                for heart_op in APLOperator:
                    if heart_op.value == selected_op.value:
                        self.agent.apply_operator(heart_op)
                        break
                
                if step % 100 == 0:
                    self._log(f"Step {step}: {selected_op.value} | "
                              f"z={z:.3f} | coh={coherence:.3f} | "
                              f"files={len(state.files)}")
            
            # === LOG PROGRESS ===
            if step % 100 == 0:
                self.build_log.append({
                    "step": step,
                    "tier": current_tier,
                    "z": z,
                    "coherence": coherence,
                    "files": len(state.files)
                })
            
            # === CHECK COMPLETION ===
            if self._check_completion(analysis, task.target_z, state):
                build_time = time.time() - start_time
                
                self._log(f"\n{'='*60}")
                self._log(f"BUILD COMPLETE")
                self._log(f"  Steps: {step}")
                self._log(f"  Time: {build_time:.2f}s")
                self._log(f"  Final z: {z:.4f}")
                self._log(f"  Final coherence: {coherence:.4f}")
                self._log(f"  Tiers traversed: {' → '.join(tier_progression)}")
                self._log(f"  Files generated: {len(state.files)}")
                self._log(f"{'='*60}\n")
                
                return self._emit_artifact(task, state, tier_progression, 
                                           build_time, step)
        
        self._log(f"\nBuild did not complete within {max_steps} steps")
        self._log(f"  Final z: {state.z:.4f} (target: {task.target_z})")
        self._log(f"  Final coherence: {state.coherence:.4f}")
        return None
    
    def _execute_tier_action(
        self, 
        tier: str, 
        tools: List[str], 
        state: BuildState
    ) -> BuildState:
        """Execute build action appropriate for current tier."""
        
        if tier in ["t1", "t2"]:
            return self._scaffold_phase(state, tools)
        elif tier in ["t3", "t4"]:
            return self._structure_phase(state, tools)
        elif tier in ["t5", "t6"]:
            return self._integration_phase(state, tools)
        else:
            return self._synthesis_phase(state, tools)
    
    def _scaffold_phase(self, state: BuildState, tools: List[str]) -> BuildState:
        """t1-t2: Create basic structure."""
        task = state.task
        
        # Create initial files if empty
        if not state.files:
            state.files["__init__.py"] = f'"""{task.name}"""\n\n__version__ = "0.1.0"\n'
            state.files[f"{task.name}.py"] = self._template_main(task)
            state.files["README.md"] = self._template_readme(task)
            state.modules.append(task.name)
        
        return state
    
    def _structure_phase(self, state: BuildState, tools: List[str]) -> BuildState:
        """t3-t4: Add structure and validation."""
        task = state.task
        pkg = task.name
        
        # Create package structure
        if f"{pkg}/__init__.py" not in state.files:
            state.files[f"{pkg}/__init__.py"] = f'"""{task.description}"""\n'
            state.files[f"{pkg}/core.py"] = self._template_core(task)
            state.files[f"{pkg}/exceptions.py"] = self._template_exceptions(task)
            state.modules.extend([f"{pkg}.core", f"{pkg}.exceptions"])
        
        # Add CLI if requested
        if "cli" in [f.lower() for f in task.features]:
            if f"{pkg}/cli.py" not in state.files:
                state.files[f"{pkg}/cli.py"] = self._template_cli(task)
                state.modules.append(f"{pkg}.cli")
        
        # Add basic tests
        if "tests/test_basic.py" not in state.files:
            state.files["tests/__init__.py"] = ""
            state.files["tests/test_basic.py"] = self._template_tests_basic(task)
        
        return state
    
    def _integration_phase(self, state: BuildState, tools: List[str]) -> BuildState:
        """t5-t6: Integration and refinement."""
        task = state.task
        pkg = task.name
        
        # Comprehensive tests
        if "tests/test_comprehensive.py" not in state.files:
            state.files["tests/test_comprehensive.py"] = self._template_tests_comprehensive(task)
        
        # Add logging
        for filename in list(state.files.keys()):
            if filename.endswith('.py') and 'test' not in filename:
                content = state.files[filename]
                if 'import logging' not in content and '__init__' not in filename:
                    state.files[filename] = (
                        "import logging\n\n"
                        f"logger = logging.getLogger(__name__)\n\n"
                        + content
                    )
        
        # Add setup.py
        if "setup.py" not in state.files:
            state.files["setup.py"] = self._template_setup(task)
        
        # Add pyproject.toml
        if "pyproject.toml" not in state.files:
            state.files["pyproject.toml"] = self._template_pyproject(task)
        
        return state
    
    def _synthesis_phase(self, state: BuildState, tools: List[str]) -> BuildState:
        """t7-t9: Final synthesis."""
        task = state.task
        
        # Generate API documentation
        if "docs/API.md" not in state.files:
            state.files["docs/API.md"] = self._generate_api_docs(task, state)
        
        # Add type stubs if needed
        if "py.typed" not in state.files:
            state.files["py.typed"] = ""
        
        return state
    
    def _check_completion(
        self, 
        analysis, 
        target_z: float,
        state: BuildState
    ) -> bool:
        """Check if build is complete."""
        # K-formation or sufficient coherence
        k_ready = analysis.k_formation or analysis.coherence >= 0.30
        
        # Must have minimum files
        has_files = len(state.files) >= 3
        
        # Must be at or above target z (with tolerance)
        z_ready = analysis.z >= target_z * 0.85
        
        return k_ready and has_files and z_ready
    
    def _emit_artifact(
        self, 
        task: BuildTask, 
        state: BuildState,
        tier_progression: List[str],
        build_time: float,
        steps: int
    ) -> BuildArtifact:
        """Package and emit the build artifact."""
        
        artifact = BuildArtifact(
            task_id=task.task_id,
            files=state.files,
            metadata={
                "name": task.name,
                "description": task.description,
                "language": task.language,
                "features": task.features
            },
            z_achieved=state.z,
            coherence_final=state.coherence,
            tier_progression=tier_progression,
            build_time=build_time,
            steps=steps
        )
        
        self.artifacts.append(artifact)
        
        # Write files to disk
        output_path = task.output_path
        os.makedirs(output_path, exist_ok=True)
        
        for filename, content in state.files.items():
            filepath = os.path.join(output_path, filename)
            os.makedirs(os.path.dirname(filepath) or output_path, exist_ok=True)
            with open(filepath, "w") as f:
                f.write(content)
        
        self._log(f"\nArtifact emitted to: {output_path}")
        self._log(f"Files created: {len(state.files)}")
        
        # Write build manifest
        manifest = {
            "task_id": artifact.task_id,
            "z_achieved": artifact.z_achieved,
            "coherence_final": artifact.coherence_final,
            "tier_progression": artifact.tier_progression,
            "build_time": artifact.build_time,
            "steps": artifact.steps,
            "files": list(artifact.files.keys())
        }
        with open(os.path.join(output_path, "BUILD_MANIFEST.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        
        return artifact
    
    def _log(self, msg: str):
        """Log message if verbose."""
        if self.verbose:
            print(msg)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════
    
    def _template_main(self, task: BuildTask) -> str:
        return f'''#!/usr/bin/env python3
"""{task.description}

Auto-generated by Autonomous Builder
Target z: {task.target_z}
"""


def main():
    """Main entry point."""
    print("Hello from {task.name}")


if __name__ == "__main__":
    main()
'''
    
    def _template_readme(self, task: BuildTask) -> str:
        features = '\n'.join(f"- {f}" for f in task.features) if task.features else "- Core functionality"
        return f'''# {task.name}

{task.description}

## Features

{features}

## Installation

```bash
pip install -e .
```

## Usage

```python
from {task.name} import {task.name.title().replace("_", "")}

# TODO: Add usage examples
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

---
*Auto-generated by Autonomous Builder*
'''
    
    def _template_core(self, task: BuildTask) -> str:
        cls_name = task.name.title().replace("_", "")
        return f'''"""{task.name} core functionality."""

from typing import Any, Optional


class {cls_name}:
    """Main class for {task.description}."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize {cls_name}.
        
        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {{}}
    
    def run(self) -> Any:
        """Execute main functionality.
        
        Returns:
            Result of execution.
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def validate(self) -> bool:
        """Validate current state.
        
        Returns:
            True if valid, False otherwise.
        """
        return True


def create_{task.name}(**kwargs) -> {cls_name}:
    """Factory function to create {cls_name} instance.
    
    Args:
        **kwargs: Configuration options.
    
    Returns:
        Configured {cls_name} instance.
    """
    return {cls_name}(config=kwargs)
'''
    
    def _template_exceptions(self, task: BuildTask) -> str:
        cls_name = task.name.title().replace("_", "")
        return f'''"""{task.name} exceptions."""


class {cls_name}Error(Exception):
    """Base exception for {task.name}."""
    pass


class ValidationError({cls_name}Error):
    """Raised when validation fails."""
    pass


class ConfigurationError({cls_name}Error):
    """Raised when configuration is invalid."""
    pass


class ProcessingError({cls_name}Error):
    """Raised when processing fails."""
    pass
'''
    
    def _template_cli(self, task: BuildTask) -> str:
        cls_name = task.name.title().replace("_", "")
        return f'''"""{task.name} command-line interface."""

import argparse
import sys
import logging

from .core import {cls_name}
from .exceptions import {cls_name}Error

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="{task.name}",
        description="{task.description}"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Configuration file path"
    )
    return parser


def main(argv: list = None) -> int:
    """Main CLI entry point.
    
    Args:
        argv: Command line arguments.
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    
    try:
        instance = {cls_name}()
        result = instance.run()
        print(f"Success: {{result}}")
        return 0
    except {cls_name}Error as e:
        logger.error(f"Error: {{e}}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {{e}}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
'''
    
    def _template_tests_basic(self, task: BuildTask) -> str:
        cls_name = task.name.title().replace("_", "")
        return f'''"""Basic tests for {task.name}."""

import pytest

from {task.name}.core import {cls_name}, create_{task.name}
from {task.name}.exceptions import {cls_name}Error, ValidationError


class Test{cls_name}:
    """Test suite for {cls_name}."""
    
    def test_init(self):
        """Test basic initialization."""
        instance = {cls_name}()
        assert instance is not None
    
    def test_init_with_config(self):
        """Test initialization with config."""
        config = {{"key": "value"}}
        instance = {cls_name}(config=config)
        assert instance.config == config
    
    def test_factory(self):
        """Test factory function."""
        instance = create_{task.name}(key="value")
        assert instance.config["key"] == "value"
    
    def test_validate(self):
        """Test validation."""
        instance = {cls_name}()
        assert instance.validate() is True
    
    def test_run_not_implemented(self):
        """Test that run raises NotImplementedError."""
        instance = {cls_name}()
        with pytest.raises(NotImplementedError):
            instance.run()
'''
    
    def _template_tests_comprehensive(self, task: BuildTask) -> str:
        cls_name = task.name.title().replace("_", "")
        return f'''"""Comprehensive tests for {task.name}."""

import pytest
from unittest.mock import Mock, patch

from {task.name}.core import {cls_name}
from {task.name}.exceptions import {cls_name}Error, ValidationError, ProcessingError


class Test{cls_name}Comprehensive:
    """Comprehensive test suite."""
    
    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return {cls_name}()
    
    @pytest.fixture
    def configured_instance(self):
        """Create configured test instance."""
        return {cls_name}(config={{"test": True}})
    
    def test_empty_config(self, instance):
        """Test with empty config."""
        assert instance.config == {{}}
    
    def test_config_preservation(self, configured_instance):
        """Test that config is preserved."""
        assert configured_instance.config["test"] is True
    
    @pytest.mark.parametrize("config", [
        {{}},
        {{"a": 1}},
        {{"a": 1, "b": 2}},
    ])
    def test_various_configs(self, config):
        """Test various configuration options."""
        instance = {cls_name}(config=config)
        assert instance.config == config


class TestExceptions:
    """Test exception hierarchy."""
    
    def test_base_exception(self):
        """Test base exception."""
        with pytest.raises({cls_name}Error):
            raise {cls_name}Error("test")
    
    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from base."""
        with pytest.raises({cls_name}Error):
            raise ValidationError("validation failed")
    
    def test_processing_error_inheritance(self):
        """Test ProcessingError inherits from base."""
        with pytest.raises({cls_name}Error):
            raise ProcessingError("processing failed")
'''
    
    def _template_setup(self, task: BuildTask) -> str:
        return f'''"""Setup script for {task.name}."""

from setuptools import setup, find_packages

setup(
    name="{task.name}",
    version="0.1.0",
    description="{task.description}",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[],
    extras_require={{
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "mypy>=1.0",
        ],
    }},
    entry_points={{
        "console_scripts": [
            "{task.name}={task.name}.cli:main",
        ],
    }},
)
'''
    
    def _template_pyproject(self, task: BuildTask) -> str:
        return f'''[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{task.name}"
version = "0.1.0"
description = "{task.description}"
requires-python = ">=3.8"

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "black>=23.0", "mypy>=1.0"]

[project.scripts]
{task.name} = "{task.name}.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.black]
line-length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
'''
    
    def _generate_api_docs(self, task: BuildTask, state: BuildState) -> str:
        docs = f"# {task.name} API Documentation\n\n"
        docs += f"{task.description}\n\n"
        docs += "## Modules\n\n"
        
        for filename in sorted(state.files.keys()):
            if filename.endswith('.py') and '__init__' not in filename:
                mod_name = filename.replace('/', '.').replace('.py', '')
                docs += f"### `{mod_name}`\n\n"
                
                # Extract classes and functions
                content = state.files[filename]
                for line in content.split('\n'):
                    if line.startswith('class '):
                        name = line.split('(')[0].split(':')[0].replace('class ', '')
                        docs += f"- **{name}**: Class\n"
                    elif line.startswith('def ') and not line.startswith('def _'):
                        name = line.split('(')[0].replace('def ', '')
                        docs += f"- `{name}()`: Function\n"
                docs += "\n"
        
        return docs


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Tool Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --task task.json
  %(prog)s --name my_tool --description "Does something" --target-z 0.5
  %(prog)s --name parser --features "cli,validation" --target-z 0.65
        """
    )
    
    parser.add_argument("--task", type=str, help="Task JSON file")
    parser.add_argument("--name", type=str, help="Tool name")
    parser.add_argument("--description", type=str, help="Tool description")
    parser.add_argument("--target-z", type=float, default=0.5, help="Target z-coordinate")
    parser.add_argument("--features", type=str, default="", help="Comma-separated features")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--max-steps", type=int, default=5000, help="Maximum build steps")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    # Build task from args
    if args.task:
        task = BuildTask.from_json(args.task)
    elif args.name:
        task = BuildTask(
            task_id=f"cli_{int(time.time())}",
            name=args.name,
            description=args.description or f"{args.name} tool",
            target_z=args.target_z,
            features=args.features.split(",") if args.features else [],
            output_path=args.output
        )
    else:
        parser.error("Either --task or --name is required")
        return 1
    
    # Build
    builder = AutonomousBuilder(verbose=not args.quiet)
    artifact = builder.build(task, max_steps=args.max_steps)
    
    if artifact:
        print(f"\n✓ Build successful: {task.output_path}")
        return 0
    else:
        print(f"\n✗ Build failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
