import os
from pathlib import Path
import pytest


def _has_matplotlib():
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not available")
@pytest.mark.timeout(3)
def test_analyzer_plot_headless(tmp_path: Path, monkeypatch):
    # Force headless backend before pyplot import
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from src.quantum_apl_python.analyzer import QuantumAnalyzer

    data = {"history": {
        "z": [0.80, 0.86, 0.87],
        "phi": [0.10, 0.20, 0.30],
        "entropy": [0.50, 0.40, 0.35],
    }}

    # overlays OFF
    monkeypatch.delenv("QAPL_ANALYZER_OVERLAYS", raising=False)
    a = QuantumAnalyzer(data)
    p1 = tmp_path / "plot_off.png"
    a.plot(save_path=p1)
    assert p1.exists() and p1.stat().st_size > 1024
    plt.close("all")

    # overlays ON
    monkeypatch.setenv("QAPL_ANALYZER_OVERLAYS", "1")
    a2 = QuantumAnalyzer(data)
    p2 = tmp_path / "plot_on.png"
    a2.plot(save_path=p2)
    assert p2.exists() and p2.stat().st_size > 1024
    plt.close("all")
