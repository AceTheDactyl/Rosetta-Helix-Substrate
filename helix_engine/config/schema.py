"""
Configuration Schema
====================

Defines the configuration schema for training runs.

Signature: schema|v0.1.0|helix
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# Schema definition
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        # Run identification
        "run_name": {"type": "string", "description": "Human-readable run name"},
        "tags": {"type": "array", "items": {"type": "string"}},

        # Training settings
        "seed": {"type": "integer", "default": 42},
        "deterministic": {"type": "boolean", "default": False},

        # Module flags
        "use_wumbo": {"type": "boolean", "default": True},
        "use_helix": {"type": "boolean", "default": True},
        "use_substrate": {"type": "boolean", "default": True},
        "use_kuramoto": {"type": "boolean", "default": True},
        "use_feedback_loop": {"type": "boolean", "default": True},
        "use_apl_engine": {"type": "boolean", "default": True},

        # Training hyperparameters
        "total_steps": {"type": "integer", "minimum": 1, "default": 1000},
        "warmup_steps": {"type": "integer", "minimum": 0, "default": 100},
        "eval_steps": {"type": "integer", "minimum": 1, "default": 100},
        "checkpoint_steps": {"type": "integer", "minimum": 1, "default": 500},
        "log_steps": {"type": "integer", "minimum": 1, "default": 10},

        # Physics (immutable)
        "n_oscillators": {"type": "integer", "minimum": 1, "default": 60},

        # Paths
        "output_dir": {"type": "string", "default": "runs"},

        # Evaluation gates
        "gates": {
            "type": "object",
            "properties": {
                "min_negentropy": {"type": "number", "minimum": 0, "maximum": 1},
                "min_k_formations": {"type": "integer", "minimum": 0},
                "max_conservation_error": {"type": "number", "minimum": 0},
                "min_final_z": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },

        # Module configurations
        "modules": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "enabled": {"type": "boolean", "default": True},
                    "steps": {"type": "integer", "minimum": 1},
                    "params": {"type": "object"},
                },
            },
        },

        # Logging
        "log_level": {
            "type": "string",
            "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
            "default": "INFO",
        },
        "enable_tensorboard": {"type": "boolean", "default": False},
        "enable_wandb": {"type": "boolean", "default": False},
    },
}


class ConfigSchema:
    """
    Configuration schema validator.
    """

    def __init__(self):
        self.schema = CONFIG_SCHEMA

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a configuration against the schema.

        Returns (is_valid, list_of_errors)
        """
        errors = []

        # Check required types
        for key, spec in self.schema.get("properties", {}).items():
            if key in config:
                value = config[key]
                expected_type = spec.get("type")

                if not self._check_type(value, expected_type):
                    errors.append(f"{key}: expected {expected_type}, got {type(value).__name__}")

                # Check minimum/maximum for numbers
                if expected_type in ("integer", "number"):
                    if "minimum" in spec and value < spec["minimum"]:
                        errors.append(f"{key}: value {value} below minimum {spec['minimum']}")
                    if "maximum" in spec and value > spec["maximum"]:
                        errors.append(f"{key}: value {value} above maximum {spec['maximum']}")

                # Check enum values
                if "enum" in spec and value not in spec["enum"]:
                    errors.append(f"{key}: value {value} not in {spec['enum']}")

        return len(errors) == 0, errors

    def _check_type(self, value: Any, expected: Optional[str]) -> bool:
        """Check if value matches expected type."""
        if expected is None:
            return True

        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected_type = type_map.get(expected)
        if expected_type is None:
            return True

        return isinstance(value, expected_type)

    def get_defaults(self) -> Dict[str, Any]:
        """Get default values from schema."""
        defaults = {}
        for key, spec in self.schema.get("properties", {}).items():
            if "default" in spec:
                defaults[key] = spec["default"]
        return defaults


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a configuration.

    Returns (is_valid, list_of_errors)
    """
    schema = ConfigSchema()
    return schema.validate(config)
