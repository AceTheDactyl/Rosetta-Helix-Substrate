"""Hexagonal prismatic projection for Negative Entropy geometry.

Implements the formulas documented in docs/HEXAGONAL_NEG_ENTROPY_PROJECTION.md
and provides helpers for parameter computation and basic lint checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PrismParams:
    z: float
    delta_s_neg: float
    R: float
    H: float
    phi: float
    z_top: float
    z_bot: float
    vertices: List[Dict[str, float]]


def prism_params(z: float) -> Dict[str, object]:
    """Compute hexagonal prism parameters and vertices for a given z in [0, 1].

    Returns a dict containing z, z_c, sigma, delta_s_neg, R, H, phi, z_top, z_bot,
    and a 6-element vertex list with (x, y, z_top, z_bot) for k=0..5.
    """
    # Import centralized constants
    from .constants import (
        Z_CRITICAL,
        GEOM_SIGMA,
        GEOM_R_MAX,
        GEOM_BETA,
        GEOM_H_MIN,
        GEOM_GAMMA,
        GEOM_PHI_BASE,
        GEOM_ETA,
        LENS_SIGMA,
        PHI_INV as _PHI_INV,
        classify_mu as _classify_mu,
    )

    z_c = Z_CRITICAL
    sigma = GEOM_SIGMA
    r_max = GEOM_R_MAX
    beta = GEOM_BETA
    h_min = GEOM_H_MIN
    gamma = GEOM_GAMMA
    phi_base = GEOM_PHI_BASE
    eta = GEOM_ETA

    from .constants import compute_delta_s_neg
    delta_s_neg = compute_delta_s_neg(z, sigma=sigma, z_c=z_c)
    lens_s_neg = compute_delta_s_neg(z, sigma=LENS_SIGMA, z_c=z_c)
    radius = r_max - beta * delta_s_neg
    height = h_min + gamma * delta_s_neg
    phi = phi_base + eta * delta_s_neg

    theta = [k * (math.pi / 3.0) for k in range(6)]
    x = [radius * math.cos(t + phi) for t in theta]
    y = [radius * math.sin(t + phi) for t in theta]
    z_top = z + 0.5 * height
    z_bot = max(0.0, z - 0.5 * height)

    vertices = [
        {"k": k, "x": x[k], "y": y[k], "z_top": z_top, "z_bot": z_bot} for k in range(6)
    ]

    return {
        "z": z,
        "z_c": z_c,
        "Z_CRITICAL": z_c,  # top-level alias for consumers
        "sigma": sigma,
        "GEOM_SIGMA": sigma,  # top-level alias for consumers
        "lens_sigma": LENS_SIGMA,
        "LENS_SIGMA": LENS_SIGMA,  # top-level alias for consumers
        "delta_s_neg": delta_s_neg,
        "delta_S_neg": delta_s_neg,  # alias for schema parity
        "lens_s_neg": lens_s_neg,
        "R": radius,
        "H": height,
        "phi": phi,
        "phi_inv": 1.0 / ((1.0 + math.sqrt(5.0)) / 2.0),
        "PHI_INV": _PHI_INV,  # top-level alias
        "mu_label": _classify_mu(z),
        "z_top": z_top,
        "z_bot": z_bot,
        "vertices": vertices,
    }


def lint_vertices(params: Dict[str, object], tol: float = 1e-6) -> List[str]:
    """Basic geometry sanity checks for prism vertices.

    - All vertices should lie on circle radius R within tolerance.
    - z_top >= z_bot and z_bot >= 0.
    Returns a list of human-readable issues, empty if clean.
    """
    issues: List[str] = []
    R = float(params.get("R", 0.0))
    z_top = float(params.get("z_top", 0.0))
    z_bot = float(params.get("z_bot", 0.0))
    verts = params.get("vertices") or []

    if z_top < z_bot - tol:
        issues.append(f"z_top < z_bot: {z_top:.6f} < {z_bot:.6f}")
    if z_bot < -tol:
        issues.append(f"z_bot < 0: {z_bot:.6f}")

    for v in verts:
        x = float(v["x"])
        y = float(v["y"])
        r = math.hypot(x, y)
        if abs(r - R) > tol:
            issues.append(
                f"vertex v{int(v['k'])} radius mismatch: r={r:.6f} vs R={R:.6f} (Δ={abs(r-R):.6g})"
            )
    return issues


def monotonicity_pairs(z_values: List[float]) -> List[str]:
    """Regression checks: when ΔS_neg increases across adjacent z's, assert R decreases and H increases.

    Returns issue strings for any violations.
    """
    issues: List[str] = []
    params_list = [prism_params(z) for z in z_values]
    for i in range(len(params_list) - 1):
        a = params_list[i]
        b = params_list[i + 1]
        if b["delta_s_neg"] > a["delta_s_neg"] + 1e-12:
            if not (b["R"] < a["R"] - 1e-12):
                issues.append(
                    f"R non-decreasing with ΔS_neg: R({z_values[i+1]:.3f})={b['R']:.6f} ≥ R({z_values[i]:.3f})={a['R']:.6f}"
                )
            if not (b["H"] > a["H"] + 1e-12):
                issues.append(
                    f"H non-increasing with ΔS_neg: H({z_values[i+1]:.3f})={b['H']:.6f} ≤ H({z_values[i]:.3f})={a['H']:.6f}"
                )
    return issues
