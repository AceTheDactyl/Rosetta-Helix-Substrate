"""
Configuration Loader
====================

Load and merge configuration files.

Signature: loader|v0.1.0|helix
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from helix_engine.config.schema import ConfigSchema


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Supports:
    - File inheritance via _base: field
    - Environment variable interpolation

    Args:
        path: Path to YAML config file

    Returns:
        Loaded configuration dict
    """
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    # Handle inheritance
    if "_base" in config:
        base_path = config.pop("_base")
        if not os.path.isabs(base_path):
            base_path = str(Path(path).parent / base_path)
        base_config = load_config(base_path)
        config = merge_configs(base_config, config)

    # Environment variable interpolation
    config = _interpolate_env(config)

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override taking precedence.

    Performs deep merge for nested dicts.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def _interpolate_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interpolate environment variables in config values.

    Supports ${VAR} and ${VAR:-default} syntax.
    """
    result = {}

    for key, value in config.items():
        if isinstance(value, str):
            result[key] = _interpolate_string(value)
        elif isinstance(value, dict):
            result[key] = _interpolate_env(value)
        elif isinstance(value, list):
            result[key] = [
                _interpolate_string(v) if isinstance(v, str) else v
                for v in value
            ]
        else:
            result[key] = value

    return result


def _interpolate_string(value: str) -> str:
    """Interpolate environment variables in a string."""
    import re

    def replace(match):
        var_spec = match.group(1)
        if ":-" in var_spec:
            var, default = var_spec.split(":-", 1)
            return os.environ.get(var, default)
        else:
            return os.environ.get(var_spec, match.group(0))

    return re.sub(r"\$\{([^}]+)\}", replace, value)


def get_default_config() -> Dict[str, Any]:
    """Get the default configuration."""
    schema = ConfigSchema()
    return schema.get_defaults()
