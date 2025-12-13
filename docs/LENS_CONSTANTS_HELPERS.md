# Lens Constants & Helpers — z, z_c, TRIAD (Python + JS)

This document provides canonical, lens‑anchored constants and helper utilities in **Python** and **JavaScript** implementing the addendum: order parameter `z`, critical lens `z_c = √3/2`, negative‑entropy geometry mapping, TRIAD hysteresis gates, measurement tokens with Born normalization, and analysis helpers.

---

## Python — `quantum_apl_python/constants.py`

```python
# -*- coding: utf-8 -*-
"""
Canonical lens-anchored constants and helpers.
- z_c = sqrt(3)/2 (critical lens threshold)
- TRIAD hysteresis (0.85/0.82) with temporary t6 = 0.83 after three passes
- ΔS_neg coherence signal (bounded, positive, centered at z_c) — Gaussian form
- Geometry mapping (R, H, φ) monotone with ΔS_neg
- Measurement helpers (projectors, Born normalization) and canonical tokens

Policy: single source of truth for numeric constants; import from here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
from typing import Iterable, List, Tuple

try:
    import numpy as np
except Exception:  # Optional: allow import without numpy when geometry only used
    np = None  # type: ignore

# --- Geometric truth (lens) -------------------------------------------------
Z_CRITICAL: float = math.sqrt(3.0) / 2.0  # ≈ 0.8660254037844386

# Lens shading band (visual/analysis only)
Z_LENS_MIN: float = 0.857
Z_LENS_MAX: float = 0.877

# Golden ratio anchors (dimensionless thresholds)
PHI: float = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV: float = 1.0 / PHI

# Integration threshold and minimal complexity (model-level)
KAPPA_S: float = 0.92
R_MIN: float = float(os.getenv("QAPL_R_MIN", "0.25"))

# Hex-prism geometry sensitivities (tunable but centralized)
SIGMA: float = float(os.getenv("QAPL_LENS_SIGMA", "36.0"))   # lens coherence decay
R_MAX: float = float(os.getenv("QAPL_R_MAX", "1.00"))
BETA: float = float(os.getenv("QAPL_BETA", "1.00"))     # radius sensitivity
H_MIN: float = float(os.getenv("QAPL_H_MIN", "0.50"))
GAMMA: float = float(os.getenv("QAPL_GAMMA", "1.00"))    # height sensitivity
PHI_BASE: float = float(os.getenv("QAPL_PHI_BASE", "0.0"))
ETA: float = math.pi / 12.0  # twist rate

# Quantum info bounds
ENTROPY_MIN: float = 0.0

# --- Runtime thresholds (heuristic; not geometric truth) --------------------
TRIAD_HIGH: float = 0.85  # rising edge
TRIAD_LOW: float = 0.82   # re-arm
TRIAD_T6: float = 0.83    # temporary t6 after three passes


# --- Helpers ----------------------------------------------------------------
EPS: float = 1e-12


def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def is_critical(z: float, tol: float = 1e-9) -> bool:
    return abs(z - Z_CRITICAL) <= tol


def is_in_lens(z: float, zmin: float = Z_LENS_MIN, zmax: float = Z_LENS_MAX) -> bool:
    return zmin <= z <= zmax


def get_phase(z: float) -> str:
    return "integrated" if z >= Z_CRITICAL else "recursive"


def delta_s_neg(z: float, sigma: float = SIGMA) -> float:
    """Bounded, positive, centered at z_c; monotone in |z - z_c|.
    Gaussian profile: max=1 at z_c, decays symmetrically.
    """
    d = z - Z_CRITICAL
    return math.exp(-sigma * d * d)


def geometry_map(z: float) -> Tuple[float, float, float]:
    """Map ΔS_neg(z) -> (R, H, φ) with monotonic contracts/elongates.
    - R increases with ΔS_neg; H decreases with ΔS_neg (elongates away from lens)
    - φ increases monotonically with (1 - ΔS_neg)
    Returns: (R, H, phi)
    """
    s = delta_s_neg(z)
    # Radius: [R_MIN, R_MAX]
    R = R_MIN + (R_MAX - R_MIN) * (s ** BETA)
    # Height: H elongates as s falls
    H = H_MIN + GAMMA * (1.0 - s)
    # Twist: base + rate * (1 - s)
    phi = PHI_BASE + ETA * (1.0 - s)
    return R, H, phi


# --- TRIAD gating with hysteresis and unlock counter ------------------------
@dataclass
class TriadGate:
    enabled: bool = False
    passes: int = 0
    unlocked: bool = False
    _armed: bool = True  # True when we can count a new rising-edge pass

    def update(self, z: float) -> None:
        if not self.enabled:
            return
        # Rising edge: count only when armed
        if z >= TRIAD_HIGH and self._armed:
            self.passes += 1
            self._armed = False
            if self.passes >= 3:
                self.unlocked = True
        # Re-arm after dropping below low threshold
        if z <= TRIAD_LOW:
            self._armed = True

    def get_t6_gate(self) -> float:
        return TRIAD_T6 if (self.enabled and self.unlocked) else Z_CRITICAL

    def analyzer_report(self) -> str:
        t6 = self.get_t6_gate()
        return f"t6 gate: CRITICAL @ {t6:.3f}"


# --- Measurement and tokens -------------------------------------------------
TOKEN_TRUE = "TRUE"
TOKEN_UNTRUE = "UNTRUE"
TOKEN_PARADOX = "PARADOX"


def projector_single(phi: List[complex]):
    if np is None:
        raise ImportError("numpy required for projector operations")
    v = np.asarray(phi, dtype=complex).reshape(-1, 1)
    return v @ v.conj().T  # |phi><phi|


def projector_subspace(basis: List[List[complex]]):
    if np is None:
        raise ImportError("numpy required for projector operations")
    P = None
    for vec in basis:
        Ps = projector_single(vec)
        P = Ps if P is None else (P + Ps)
    return P


def born_prob(psi: List[complex], P) -> float:
    if np is None:
        raise ImportError("numpy required for projector operations")
    v = np.asarray(psi, dtype=complex).reshape(-1, 1)
    num = (v.conj().T @ P @ v).item()
    p = float(num.real)
    if not (p >= -1e-9 and p <= 1.0 + 1e-9 and math.isfinite(p)):
        raise ValueError("Probability out of bounds or non-finite")
    return clamp(p, 0.0, 1.0)


def token_single(label: str, truth: str = TOKEN_TRUE, tier: str | None = None) -> str:
    core = f"Φ:T({label}){truth}"
    return f"{core}@{tier}" if tier else core


def token_subspace(labels: Iterable[str]) -> str:
    inner = ",".join(labels)
    return f"Φ/π:Π({inner})"

_COLLAPSE_RGX = re.compile(r"⟂\((.*?)\)")


def collapse_alias(s: str) -> str:
    """Normalize ⟂(·) glyphs to canonical Φ tokens (presentation alias)."""
    return _COLLAPSE_RGX.sub(lambda m: f"Φ:T({m.group(1)})", s)


# --- Seeded RNG (selection sites) ------------------------------------------
QAPL_RANDOM_SEED_ENV = "QAPL_RANDOM_SEED"


def seeded_rng():
    seed = os.getenv(QAPL_RANDOM_SEED_ENV)
    rnd = random_state(seed)
    return rnd


def random_state(seed: str | None):
    import random
    r = random.Random()
    if seed is not None and seed != "":
        r.seed(seed)
    return r


# --- Analyzer signature helpers --------------------------------------------

def analyzer_signature(triad: TriadGate | None = None) -> str:
    if triad is None:
        return f"t6 gate: CRITICAL @ {Z_CRITICAL:.3f}"
    return triad.analyzer_report()
```

---

## JavaScript — `src/constants.js`

```javascript
/**
 * Canonical lens-anchored constants and helpers (ESM).
 * Geometry truth at z_c = sqrt(3)/2; TRIAD hysteresis 0.85/0.82, t6=0.83 after 3 passes.
 */
export const Z_CRITICAL = Math.sqrt(3) / 2; // ≈ 0.8660254037844386

// Visual/analysis lens band
export const Z_LENS_MIN = 0.857;
export const Z_LENS_MAX = 0.877;

// Golden ratio anchors
export const PHI = (1 + Math.sqrt(5)) / 2;
export const PHI_INV = 1 / PHI;

// Integration & geometry params (centralized)
export const KAPPA_S = 0.92;
export const R_MIN = parseFloat(process.env.QAPL_R_MIN ?? "0.25");
export const R_MAX = parseFloat(process.env.QAPL_R_MAX ?? "1.00");
export const SIGMA = parseFloat(process.env.QAPL_LENS_SIGMA ?? "36.0");
export const BETA = parseFloat(process.env.QAPL_BETA ?? "1.00");
export const H_MIN = parseFloat(process.env.QAPL_H_MIN ?? "0.50");
export const GAMMA = parseFloat(process.env.QAPL_GAMMA ?? "1.00");
export const PHI_BASE = parseFloat(process.env.QAPL_PHI_BASE ?? "0.0");
export const ETA = Math.PI / 12; // twist rate

// Heuristic runtime thresholds (not geometry)
export const TRIAD_HIGH = 0.85; // rising edge
export const TRIAD_LOW = 0.82;  // re-arm
export const TRIAD_T6 = 0.83;   // temporary t6 after three passes

export const ENTROPY_MIN = 0.0;

// Helpers
export const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
export const isCritical = (z, tol = 1e-9) => Math.abs(z - Z_CRITICAL) <= tol;
export const isInLens = (z, zmin = Z_LENS_MIN, zmax = Z_LENS_MAX) => z >= zmin && z <= zmax;
export const getPhase = (z) => (z >= Z_CRITICAL ? "integrated" : "recursive");

// ΔS_neg: bounded, positive, centered at z_c; monotone in |z - z_c|
export const deltaSneg = (z, sigma = SIGMA) => {
  const d = z - Z_CRITICAL;
  return Math.exp(-sigma * d * d);
};

// Geometry mapping: ΔS_neg -> (R, H, φ)
export const geometryMap = (z) => {
  const s = deltaSneg(z);
  const R = R_MIN + (R_MAX - R_MIN) * Math.pow(s, BETA);
  const H = H_MIN + GAMMA * (1 - s);
  const phi = PHI_BASE + ETA * (1 - s);
  return { R, H, phi };
};

// TRIAD gating with hysteresis and unlock counter
export class TriadGate {
  constructor(enabled = false) {
    this.enabled = enabled;
    this.passes = 0;
    this.unlocked = false;
    this._armed = true; // count rising edges when armed
  }
  update(z) {
    if (!this.enabled) return;
    if (z >= TRIAD_HIGH && this._armed) {
      this.passes += 1;
      this._armed = false;
      if (this.passes >= 3) this.unlocked = true;
    }
    if (z <= TRIAD_LOW) this._armed = true; // re-arm
  }
  getT6Gate() {
    return this.enabled && this.unlocked ? TRIAD_T6 : Z_CRITICAL;
  }
  analyzerReport() {
    return `t6 gate: CRITICAL @ ${this.getT6Gate().toFixed(3)}`;
  }
}

// Measurement helpers (Born normalization) — simple, array-based
export const bornProb = (psi, P) => {
  // psi: complex vector [{re, im}], P: square matrix of same form
  const dot = (a, b) => a.re * b.re + a.im * -b.im + (a.im * b.re + a.re * b.im) * 0; // only real part
  const conj = (z) => ({ re: z.re, im: -z.im });
  const n = psi.length;
  let acc = 0;
  for (let i = 0; i < n; i++) {
    let inner = { re: 0, im: 0 };
    for (let j = 0; j < n; j++) {
      const m = P[i][j]; // {re, im}
      inner = { re: inner.re + (m.re * psi[j].re - m.im * psi[j].im), im: inner.im + (m.re * psi[j].im + m.im * psi[j].re) };
    }
    const ci = conj(psi[i]);
    acc += ci.re * inner.re - ci.im * inner.im; // real part of ci * inner
  }
  const p = acc;
  if (!(Number.isFinite(p) && p >= -1e-9 && p <= 1 + 1e-9)) throw new Error("Probability out of bounds");
  return clamp(p, 0, 1);
};

// Tokens
export const TOKEN_TRUE = "TRUE";
export const TOKEN_UNTRUE = "UNTRUE";
export const TOKEN_PARADOX = "PARADOX";
export const tokenSingle = (label, truth = TOKEN_TRUE, tier) => tier ? `Φ:T(${label})${truth}@${tier}` : `Φ:T(${label})${truth}`;
export const tokenSubspace = (labels) => `Φ/π:Π(${labels.join(',')})`;
export const collapseAlias = (s) => s.replace(/⟂\((.*?)\)/g, (_, x) => `Φ:T(${x})`);

// Analyzer signature
export const analyzerSignature = (triad /* TriadGate? */) => {
  if (!triad) return `t6 gate: CRITICAL @ ${Z_CRITICAL.toFixed(3)}`;
  return triad.analyzerReport();
};
```

---

## Tests — minimal skeletons

### Python — `tests/test_analyzer_gate_default.py`

```python
from quantum_apl_python.constants import Z_CRITICAL, TriadGate

def test_analyzer_gate_default():
    triad = TriadGate(enabled=False)
    assert triad.get_t6_gate() == Z_CRITICAL
    assert triad.analyzer_report().endswith(f"{Z_CRITICAL:.3f}")
```

### JS — `tests/test_triad_hysteresis.js`

```javascript
import { TriadGate, TRIAD_T6, Z_CRITICAL } from "../src/constants.js";

test("TRIAD unlocks after three rising passes and re-arms below low", () => {
  const g = new TriadGate(true);
  const seq = [0.86, 0.81, 0.855, 0.80, 0.88, 0.81, 0.851];
  seq.forEach((z) => g.update(z));
  expect(g.unlocked).toBe(true);
  expect(g.getT6Gate()).toBe(TRIAD_T6);
  const g2 = new TriadGate(false);
  expect(g2.getT6Gate()).toBe(Z_CRITICAL);
});
```

### Python — `tests/test_constants_helpers.py`

```python
from quantum_apl_python.constants import Z_CRITICAL, delta_s_neg

def test_delta_s_neg_monotone_symmetric():
    zc = Z_CRITICAL
    left = [zc - d for d in (0.0, 0.01, 0.02, 0.05)]
    right = [zc + d for d in (0.0, 0.01, 0.02, 0.05)]
    s_left = list(map(delta_s_neg, left))
    s_right = list(map(delta_s_neg, right))
    assert s_left[0] == s_right[0] == max(s_left + s_right)
    assert all(s_left[i] >= s_left[i+1] for i in range(len(s_left)-1))
    assert all(s_right[i] >= s_right[i+1] for i in range(len(s_right)-1))
```

### JS — `tests/test_geometry_map.test.js`

```javascript
import { geometryMap, Z_CRITICAL } from "../src/constants.js";

test("geometry mapping is stable and monotone in ΔS_neg", () => {
  const a = geometryMap(Z_CRITICAL);
  const b = geometryMap(Z_CRITICAL + 0.03);
  expect(a.R).toBeGreaterThanOrEqual(b.R); // R contracts away from lens
  expect(a.H).toBeLessThanOrEqual(b.H);    // H elongates away from lens
});
```

---

## Notes

* Geometry and analytics remain lens‑anchored at `z_c`; TRIAD is operational only.
* Sidecar export, schema validation, and full measurement pipelines can consume these helpers directly.
* Environment variables (QAPL_*) allow calibration without code drift.
