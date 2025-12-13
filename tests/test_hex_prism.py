from __future__ import annotations

import math

from quantum_apl_python.hex_prism import prism_params, lint_vertices, monotonicity_pairs


def test_vertices_radius_and_z_bounds():
    # Check a few representative z values
    for z in (0.41, 0.70, 0.87, 0.95):
        params = prism_params(z)
        issues = lint_vertices(params, tol=1e-9)
        assert not issues, f"geometry issues for z={z}: {issues}"


def test_monotonicity_when_delta_increases():
    # Choose z values that approach the critical point z_c to increase ΔS_neg
    zs = [0.10, 0.30, 0.50, 0.70, 0.80, 0.86, 0.90]
    issues = monotonicity_pairs(zs)
    assert not issues, f"monotonicity issues: {issues}"

