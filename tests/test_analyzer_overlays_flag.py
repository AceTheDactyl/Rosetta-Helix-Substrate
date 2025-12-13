import os

from src.quantum_apl_python.analyzer import _overlays_enabled


def test_overlays_off_by_default(monkeypatch):
    monkeypatch.delenv("QAPL_ANALYZER_OVERLAYS", raising=False)
    assert _overlays_enabled() is False


def test_overlays_on_when_flag_set(monkeypatch):
    monkeypatch.setenv("QAPL_ANALYZER_OVERLAYS", "1")
    assert _overlays_enabled() is True

