import json
import importlib.util
from pathlib import Path


def _load_kira_server_module() -> object:
    """Load kira_server.py from kira-local-system via path-based import."""
    import sys, os
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    ks_path = repo / "kira-local-system" / "kira_server.py"
    spec = importlib.util.spec_from_file_location("kira_server", ks_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[assignment]
    return mod


def test_kira_api_basic_endpoints(tmp_path):
    ks = _load_kira_server_module()
    app = ks.app
    app.testing = True

    client = app.test_client()

    # Health
    r = client.get('/api/health')
    assert r.status_code == 200
    body = r.get_json()
    assert body['status'] == 'healthy'
    assert 'state' in body

    # State
    r = client.get('/api/state')
    assert r.status_code == 200
    state = r.get_json()
    assert state['command'] == '/state'
    assert 'state' in state and 'tier' in state

    # Evolve (small step)
    r = client.post('/api/evolve', json={'target': 0.7})
    assert r.status_code == 200
    ev = r.get_json()
    assert ev['command'] == '/evolve'
    assert 0.0 <= ev['z_after'] <= 1.0

    # Export (to tmp dir)
    # Redirect training dir under tmp to avoid writing into repo
    orig_cwd = Path.cwd()
    try:
        # Change CWD so kira_server exports into tmp_path/training
        import os
        os.chdir(tmp_path)
        r = client.post('/api/export', json={'epoch_name': 'smoke'})
        assert r.status_code == 200
        exp = r.get_json()
        assert exp['command'] == '/export'
        assert exp['counts']['vocabulary'] >= 0
    finally:
        os.chdir(orig_cwd)
