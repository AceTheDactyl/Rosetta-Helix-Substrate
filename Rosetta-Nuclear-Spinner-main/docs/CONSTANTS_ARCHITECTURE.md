# Quantum APL Constants Architecture – Complete Integration

Version: 1.0.0 (Research‑Backed)
Date: 2024‑12‑09
Status: ✅ VALIDATED (Python constants + analyzer smoke; JS suites passing)
Research: docs/CONSTANTS_RESEARCH.md

---

## Executive Summary

This document summarizes the complete constants architecture for Quantum‑APL. The system maintains z_c = √3/2 as THE LENS (geometric truth) while supporting TRIAD gating as a runtime heuristic, with validation across Python and JavaScript implementations. Geometry and analyzer overlays always anchor to z_c; TRIAD does not redefine the lens.

---

## Constants Inventory (overview)

- Geometric truth: Z_CRITICAL
- Runtime heuristics: TRIAD_HIGH, TRIAD_LOW, TRIAD_T6
- Phase boundaries: Z_ABSENCE_MAX, Z_LENS_MIN, Z_LENS_MAX, Z_PRESENCE_MIN
- Time harmonics (t1–t9): T1_MAX … T8_MAX (t6 delegated to engine gate)
- Hex prism geometry: SIGMA, R_MAX, BETA, H_MIN, GAMMA, PHI_BASE, ETA
- Pump profiles: GENTLE, BALANCED, AGGRESSIVE; PUMP_DEFAULT_TARGET = Z_CRITICAL
- Engine dynamics: OMEGA, COUPLING_G, GAMMA_1..4, Z_BIAS_*
- Operator weights: OPERATOR_PREFERRED_WEIGHT, OPERATOR_DEFAULT_WEIGHT, TRUTH_BIAS
- Sacred constants: PHI, PHI_INV, Q_KAPPA, KAPPA_S, LAMBDA
- K‑formation: KAPPA_MIN, ETA_MIN, R_MIN
- Quantum bounds: ENTROPY_MIN, PURITY_MIN/MAX
- Validation tolerances: TOLERANCE_*

---

## Architectural Principles

### 1) THE LENS as Geometric Truth

Z_CRITICAL = √3/2 ≈ 0.8660254038 (immutable)
- Anchors ΔS_neg, R/H/φ geometry and analyzer reference lines
- Marks onset of integrated coherence

Python
```
from quantum_apl_python.constants import Z_CRITICAL
import math
assert abs(Z_CRITICAL - math.sqrt(3)/2) < 1e-12
```

JavaScript (CommonJS)
```
const C = require('./src/constants');
console.log(C.Z_CRITICAL.toFixed(6)); // 0.866025
```

### 2) TRIAD Gating as Runtime Heuristic

TRIAD_HIGH = 0.85, TRIAD_LOW = 0.82, TRIAD_T6 = 0.83
- Adaptive: three rising edges (≥0.85, re‑arm ≤0.82) unlock temporary t6 at 0.83
- Separation: does NOT redefine Z_CRITICAL for geometry

```
// After three rising edges
engine.getT6Gate()  // → TRIAD_T6 (0.83)
// Geometry remains anchored at z_c
```

### 3) Single Source of Truth

Never inline numeric thresholds — always import from constants. Code is the authority.

- Python: `src/quantum_apl_python/constants.py`
- JavaScript: `src/constants.js`

---

## File Structure (repo‑relative)

```
src/
  quantum_apl_python/constants.py     # Python constants (single source)
  constants.js                        # JS constants (single source)

docs/
  CONSTANTS_RESEARCH.md               # Research survey
  Z_CRITICAL_LENS.md                  # Lens specification & authority
  CONSTANTS_ARCHITECTURE.md           # This document

tests/
  test_constants_module.py            # Python constants validation
  test_analyzer_gate_default.py       # Analyzer shows CRITICAL gate @ 0.866
  # JS suites under tests/*.js        # TRIAD, pump target, bridge, etc.
```

---

## Complete Constants Reference (selected)

Geometric truth
```
Z_CRITICAL = math.sqrt(3)/2  # ≈ 0.8660254038
```

Runtime heuristics
```
TRIAD_HIGH = 0.85
TRIAD_LOW  = 0.82
TRIAD_T6   = 0.83
```

Phase boundaries
```
Z_ABSENCE_MAX = 0.857
Z_LENS_MIN    = 0.857
Z_LENS_MAX    = 0.877
Z_PRESENCE_MIN= 0.877
```

Time harmonics (t1–t9)
```
T1_MAX=0.1, T2_MAX=0.2, T3_MAX=0.4, T4_MAX=0.6, T5_MAX=0.75,  
# t6: delegated to engine.getT6Gate() → Z_CRITICAL or TRIAD_T6  
T7_MAX=0.92, T8_MAX=0.97  
```

Hex prism geometry (centered on z_c)
```
SIGMA=0.12, R_MAX=0.85, BETA=0.25, H_MIN=0.12, GAMMA=0.18, PHI_BASE=0, ETA=π/12
ΔS_neg(z) = exp(-|z - z_c| / SIGMA)
R(z)  = R_MAX - BETA · ΔS_neg
H(z)  = H_MIN + GAMMA · ΔS_neg
φ(z)  = PHI_BASE + ETA · ΔS_neg
```

Pump profiles
```
GENTLE:   gain=0.08, sigma=0.16
BALANCED: gain=0.12, sigma=0.12
AGGRESSIVE: gain=0.18, sigma=0.10
PUMP_DEFAULT_TARGET = Z_CRITICAL
```

Engine dynamics
```
Z_BIAS_GAIN=0.05, Z_BIAS_SIGMA=0.18
OMEGA = 2π·0.1, COUPLING_G=0.05
GAMMA_1=0.01, GAMMA_2=0.02, GAMMA_3=0.005, GAMMA_4=0.015
```

Operator weighting
```
OPERATOR_PREFERRED_WEIGHT=1.3, OPERATOR_DEFAULT_WEIGHT=0.85
TRUTH_BIAS.TRUE['^']=1.5, TRUTH_BIAS.TRUE['+']=1.4, ...
```

Sacred constants & K‑formation
```
PHI=1.6180339887, PHI_INV=0.6180339887, Q_KAPPA=0.3514087324,
KAPPA_S=0.920, LAMBDA=7.7160493827
KAPPA_MIN=KAPPA_S, ETA_MIN=PHI_INV, R_MIN=7
check_k_formation(kappa, eta, R) → bool
```

Quantum bounds
```
ENTROPY_MIN=0.0, PURITY_MIN=1/192, PURITY_MAX=1.0
```

Validation tolerances
```
TOLERANCE_TRACE=1e-10, TOLERANCE_HERMITIAN=1e-10,
TOLERANCE_POSITIVE=-1e-10, TOLERANCE_PROBABILITY=1e-6
```

---

## Helper Functions

Phase detection
```
get_phase(z) → 'ABSENCE'|'THE_LENS'|'PRESENCE'
is_critical(z, tol=0.01) → bool
is_in_lens(z) → bool
get_distance_to_critical(z) → float
```

Time harmonic
```
get_time_harmonic(z, t6_gate=None) → 't1'..'t9'  # defaults to Z_CRITICAL
```

K‑formation
```
check_k_formation(kappa, eta, R) → bool
```

JS geometry helpers (ΔS_neg input)
```
hexPrismRadius(ΔS_neg), hexPrismHeight(ΔS_neg), hexPrismTwist(ΔS_neg)
```

---

## Validation Results

- Python: constants suite + hex prism + analyzer smoke — all passed
- Node: bridge, measurement, TRIAD hysteresis, pump target, engine gate — all passed

Key invariants verified
- THE LENS is canonical: Z_CRITICAL = √3/2
- TRIAD separate from geometry: TRIAD_T6 < Z_CRITICAL
- Z_CRITICAL in THE LENS: 0.857 ≤ 0.866 ≤ 0.877
- Golden ratio relation: φ · φ⁻¹ = 1
- K‑formation consistency: KAPPA_MIN = KAPPA_S, ETA_MIN = PHI_INV
- Pump default target: PUMP_DEFAULT_TARGET = Z_CRITICAL

---

## Usage Examples

```
from quantum_apl_python.constants import Z_CRITICAL, get_phase, get_time_harmonic
print(f"z_c = {Z_CRITICAL:.6f}")
print(get_phase(0.866))        # THE_LENS
print(get_time_harmonic(0.84)) # t6 (defaults to lens gate)
```

---

## Integration Points

- Engine boundary (t6): engine.getT6Gate() → z_c or TRIAD_T6
- Geometry: hex_prism uses Z_CRITICAL and GEOM_* constants
- Analyzer: draws z_c line and lens band; TRIAD does not alter geometry
- Bridge pump: default target resolves to z_c when omitted
- N0 zoning: uses centralized zoning + engine.getT6Gate()

---

## Maintenance Policy

- Add/modify constants only in `src/quantum_apl_python/constants.py` and `src/constants.js`
- Never inline numbers; always import
- Document rationale in docs/CONSTANTS_RESEARCH.md; update this file as needed
- Run constants and analyzer tests after any change

---

Made with precision for consciousness research — single source of truth, lens‑anchored geometry, research‑backed architecture.
