import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, env=None):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    return result.returncode, result.stdout


def test_analyzer_reports_critical_gate_when_triad_off(tmp_path: Path):
    # Ensure TRIAD is off
    env = dict(os.environ)
    env.pop("QAPL_TRIAD_UNLOCK", None)
    env["QAPL_TRIAD_COMPLETIONS"] = "0"

    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "analyzer_smoke.json"

    # Generate a small unified run JSON
    rc, out1 = run_cmd(["qapl-run", "--steps", "3", "--mode", "unified", "--output", str(out)], env=env)
    assert rc == 0, f"qapl-run failed:\n{out1}"

    # Analyze and capture text
    rc, text = run_cmd(["qapl-analyze", str(out)], env=env)
    assert rc == 0, f"qapl-analyze failed:\n{text}"

    # Smoke assertion: t6 gate reported at CRITICAL with full-precision z_c
    assert "t6 gate: CRITICAL @ 0.8660254037844386" in text, (
        f"Analyzer did not report critical t6 gate with full precision:\n{text}"
    )
