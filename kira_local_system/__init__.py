"""
KIRA local system (package wrapper)
==================================

Import-friendly package exposing the KIRA Flask app and engine while
reusing the canonical implementation in `kira-local-system/kira_server.py`.

Usage:
    from kira_local_system import create_app, get_engine, KIRAEngine
    app = create_app()
    eng = get_engine()  # lazy singleton from kira_server

Backwards-compatibility:
    from kira_local_system import kira_server  # underlying module
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

_MOD: ModuleType | None = None


def _load() -> ModuleType:
    """Load the underlying kira_server module by file path."""
    global _MOD
    if _MOD is not None:
        return _MOD
    repo = Path(__file__).resolve().parents[1]
    ks_path = repo / "kira-local-system" / "kira_server.py"
    spec = importlib.util.spec_from_file_location("kira_server", ks_path)
    if not spec or not spec.loader:
        raise RuntimeError("Unable to load kira_server module from kira-local-system")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    _MOD = mod
    return mod


def create_app() -> Any:
    """Return the KIRA Flask app instance from kira_server."""
    mod = _load()
    return getattr(mod, "app")


def get_engine() -> Any:
    """Return the KIRA engine singleton (lazy) from kira_server."""
    mod = _load()
    if hasattr(mod, "get_engine"):
        return mod.get_engine()
    # Fallback to global
    return getattr(mod, "engine", None)


def run(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    """Run the Flask development server."""
    app = create_app()
    app.run(host=host, port=port, debug=debug)


# Back-compat: expose the underlying module
kira_server = _load()

# Type/export convenience for users
KIRAEngine = getattr(kira_server, "KIRAEngine", object)

