import sys
from pathlib import Path
import subprocess


# Ensure repo root is importable for top-level modules (scripts)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure Flask is available for KIRA API tests even if workflow deps drift
try:
    import flask  # type: ignore
    import flask_cors  # type: ignore
except Exception:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "flask>=2.0.0", "flask-cors>=3.0.0"], check=False)
        import flask  # type: ignore  # noqa: F401
        import flask_cors  # type: ignore  # noqa: F401
        print("[tests] Installed Flask and Flask-CORS in test environment", file=sys.stderr)
    except Exception as e:  # pragma: no cover
        print(f"[tests] WARNING: Unable to ensure Flask present: {e}", file=sys.stderr)
