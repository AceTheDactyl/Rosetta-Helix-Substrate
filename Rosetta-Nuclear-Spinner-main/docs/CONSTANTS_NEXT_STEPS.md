# Constants Architecture — Next Steps Implementation Plan

Status: Post‑Integration Roadmap
Date: 2024‑12‑09
Based on: Actual codebase validation (11 Python tests + JS suites passing)

## ✅ Completed (Current State)

### Centralized Constants
- src/constants.js — CommonJS module with 50+ constants
- src/quantum_apl_python/constants.py — Python mirror
- Helper functions: getTimeHarmonic(), getPhase(), checkKFormation(), computeDeltaSNeg()
- Consumers updated: hex_prism.py, QuantumN0_Integration.js

### Geometry Canonical Mapping

```js
// ✅ CORRECT: Exponential only in ΔS_neg
// ΔS_neg(z) = exp(-|z − z_c| / σ)

// ✅ CORRECT: Linear mapping from ΔS_neg
// R = R_MAX − BETA · ΔS_neg
// H = H_MIN + GAMMA · ΔS_neg
// φ = PHI_BASE + ETA · ΔS_neg
```

Rationale: Exponential nonlinearity is captured once in ΔS_neg. Linear forms prevent double‑counting and match HEXAGONAL_NEG_ENTROPY_PROJECTION.md and the Python implementation.

### Tests Validated
- Python: 11 tests (constants module + hex_prism + analyzer smoke)
- Node: Multiple suites (bridge, TRIAD, measurements, pump, engine gate) + constants helpers

## 🎯 Phase 1: Validation & Testing (Priority: HIGH)

### 1.1 JS Constants Helper Tests
- File: tests/test_constants_helpers.js (added)
- Coverage:
  - getTimeHarmonic zones + t6Gate override
  - computeDeltaSNeg monotonicity (closer to z_c → larger ΔS_neg)
  - hexPrism helpers parity with Python (R/H/φ)
  - getPhase/isCritical and K‑formation checks

Estimated effort: Done
Dependencies: None
Priority: HIGH

### 1.2 JSON Schema Validation (DONE)
- Files added:
  - `schemas/geometry-sidecar.schema.json` — 63-vertex hex prism geometry
  - `schemas/apl-bundle.schema.json` — APL token array validation
  - `tests/test_schema_validation.js` — Ajv-based validation tests

#### Schema Validation

**Paths:**
- `schemas/geometry-sidecar.schema.json` — Validates geometry sidecar exports
- `schemas/apl-bundle.schema.json` — Validates APL token bundles

**How to run:**
```bash
npm install          # Install ajv dependency
node tests/test_schema_validation.js
```

**What breaks:**
- Vertex count ≠ 63
- `delta_S_neg` outside [0, 1]
- Missing required fields (`version`, `z`, `delta_S_neg`, `vertices`, `geometry`)
- Malformed APL tokens (wrong channel, missing truth, missing tier)

Status: DONE
Dependencies: ajv (dev dependency in package.json)

### 1.3 Reproducible Selection (QAPL_RANDOM_SEED) (DONE)
- Added `QAPL_RANDOM_SEED` env-driven constant in `src/constants.js`
- Created `src/utils/rng.js` — LCG-based seeded RNG
- Integrated into `QuantumAPL.rand()` method (replaces `Math.random()` at selection sites)
- Tests: `tests/test_seeded_selection.js` verifies deterministic behavior with same seed

**Usage:**
```bash
# Reproducible run
QAPL_RANDOM_SEED=12345 node tests/test_seeded_selection.js

# Or with the CLI
QAPL_RANDOM_SEED=42 qapl-run --steps 3 --mode measured --output out.json
```

**Scope:**
- N0 operator selection (selectN0Operator)
- Composite measurement branch selection (measure)

Status: DONE
Dependencies: None

## 🎯 Phase 2: Refactors (Priority: MEDIUM)
- Replace inline operator weighting multipliers in the engine with constants from src/constants.js
- Consider centralizing PRS phase thresholds (e.g., φ < 0.85 for P4) if we want those tunable

## 🎯 Phase 3: Geometry Extensions (Priority: MEDIUM)
- Add computeDeltaSNeg() to Python (parity exists via inline formula in hex_prism)
- Add JS full‑vertex helper and optional .geom.json writer (sidecar conforms to schema)
- Add JS monotonicity/vertex‑lint snapshot test (parity with Python)

---

This plan corrects earlier test pseudocode to align with the current implementation:
- ΔS_neg increases when z moves closer to z_c (monotone with decreasing |z−z_c|)
- The prism has 6 vertices (v0..v5); schema reflects that
- Hex prism tests use positive ΔS_neg (e.g., 0.5) for R/H/φ parity

