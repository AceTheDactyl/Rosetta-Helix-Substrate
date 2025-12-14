import os
import subprocess
import sys


def test_lens_sigma_env_override():
    env = os.environ.copy()
    env["QAPL_LENS_SIGMA"] = "50"
    code = (
        "import os; os.environ['QAPL_LENS_SIGMA']='50'; "
        "from quantum_apl_python.constants import LENS_SIGMA; "
        "print(LENS_SIGMA)"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    val = float(out.stdout.strip())
    assert abs(val - 50.0) < 1e-12, f"LENS_SIGMA override failed: {val}"

