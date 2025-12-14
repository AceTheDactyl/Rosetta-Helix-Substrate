from src.quantum_apl_python.constants import Z_CRITICAL
from src.quantum_apl_python.engine import default_pump_target

def test_default_pump_target_py():
    assert abs(default_pump_target() - Z_CRITICAL) < 1e-15
    print('Python default pump target equals Z_CRITICAL')

