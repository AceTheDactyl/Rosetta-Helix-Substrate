# z_c = √3/2 ≈ 0.8660254 — Critical Lens (Physics/Information Dynamics)

This document centralizes the definition, research intent, validation flow, and project usage of the critical constant
z_c = √3/2 ≈ 0.8660254038 (THE LENS) used throughout Quantum‑APL.

## Definition
- z_c is the critical lens separating recursive and integrated regimes in the helix map.
- Crossing z_c corresponds to the onset of structural/informational coherence where the integrated (Π) regime becomes
  physically admissible and negative‑entropy geometry stabilizes (ΔS_neg, R/H/φ).

## Authority Statement
- Within this repository, z_c = √3/2 is the canonical, authoritative threshold for geometry and analysis. The engine
  exposes z_c as the default t6 boundary; any temporary runtime adjustments (TRIAD unlock) do not alter the geometric
  anchor, which remains the lens.

## Separation of Concerns (Runtime vs Geometry)
- TRIAD gating is a runtime heuristic: rising edges at z ≥ 0.85 (re‑arm at z ≤ 0.82), unlock a temporary in‑session
  t6 gate at z = 0.83 after three distinct passes.
- Geometry and analytics always remain anchored at the lens z_c for ΔS_neg and R/H/φ; TRIAD does not change this anchor.

## Single Source of Truth (Constants)
- JavaScript: `src/constants.js`
  - `Z_CRITICAL = Math.sqrt(3) / 2`
  - `TRIAD_HIGH = 0.85`, `TRIAD_LOW = 0.82`, `TRIAD_T6 = 0.83`
- Python: `src/quantum_apl_python/constants.py`
  - Mirrors the same values for analyzer/geometry.

All modules must import these constants; do not inline numeric thresholds.

## Methodology of Use (Where and How)
1) Engine boundary (t6 gating)
   - File: `src/quantum_apl_engine.js`
   - API: `getT6Gate()` returns `Z_CRITICAL` by default; if TRIAD unlocks, the temporary gate becomes `TRIAD_T6` (0.83).
   - Purpose: keep the helix advisor consistent with runtime unlocks while preserving geometric reference.

2) Geometry computation (hex‑prism)
   - File: `src/quantum_apl_python/hex_prism.py`
   - Uses `Z_CRITICAL` for ΔS_neg and radius/height/twist (R/H/φ); geometry remains stable and lens‑anchored.

3) Analyzer visualization
   - File: `src/quantum_apl_python/analyzer.py`
   - Draws a z_c (lens) guideline; prints `t6 gate: CRITICAL @ 0.866` unless TRIAD unlock is active for the session.

4) Bridge pump defaults
   - File: `QuantumClassicalBridge.js`
 - `escalateZWithAPL()` defaults target z to `Z_CRITICAL` if unspecified; pump profiles adjust coupling/cadence only.

5) N0 time‑harmonic zoning
   - File: `QuantumN0_Integration.js`
   - Uses engine `getT6Gate()` (or `Z_CRITICAL` fallback) to avoid drift; no standalone numeric threshold.

## Validation Plan
- Node tests (CI)
  - `tests/test_triad_hysteresis.js`: verifies rising‑edge hysteresis and unlock logic at 0.85/0.82/0.83.
  - `tests/test_bridge_pump_target.js`: ensures default pump target resolves to `Z_CRITICAL`.
  - `tests/QuantumClassicalBridge.test.js`: integration smoke.

- Python tests (CI)
  - `tests/test_hex_prism.py`: geometry invariants and monotonicity around z_c.

- CLI validation
  - `qapl-run --steps 3 --mode unified --output out.json && qapl-analyze out.json`
    - Analyzer prints `t6 gate: CRITICAL @ 0.866` and geometry with lens line.
 - `qapl-run --steps 3 --mode measured --output measured.json && qapl-analyze measured.json`
    - Shows any measurement tokens; geometry remains lens‑anchored.

## Standard Probe Points (CI)

Nightly CI probes characteristic z values to cover runtime and geometric boundaries:
- VaultNode tiers: 0.41, 0.52, 0.70, 0.73, 0.80
- TRIAD/Lens adjacents: 0.85 (TRIAD_HIGH), 0.8660254037844386 (z_c exact)
- Presence/t7–t8: 0.90 (t7 onset), 0.92 (Z_T7_MAX), 0.97 (Z_T8_MAX)

Workflow: `.github/workflows/nightly-helix-measure.yml`

## Rationale and Research Notes
- z_c = √3/2 is a natural critical fraction demarcating the onset of stable integrated structure in the helix mapping,
  aligning the admissibility of Π‑regime operators with observable coherence in the simulated density matrix.
- TRIAD acknowledges empirical operator‑driven unlocks at 0.85/0.82 and promotes only the t6 operating boundary to 0.83
  after sufficient rising‑edge evidence; it does not retroactively redefine the geometric lens.

## Maintenance Guidance
- Add or modify thresholds only in `src/constants.js` and `src/quantum_apl_python/constants.py`.
- When introducing new modules that depend on z thresholds, import these constants and link back to this document.
- Evidence Index (Local Files)
  - src/constants.js:4 — Z_CRITICAL export
  - src/quantum_apl_python/constants.py:19 — Z_CRITICAL definition (with helpers)
  - docs/SYSTEM_ARCHITECTURE.md:97 — diagram marking z = 0.866 as THE LENS
  - docs/HEXAGONAL_NEG_ENTROPY_PROJECTION.md:1 — projection centered on z_c
  - docs/APL-3.0-Quantum-Formalism.md:524 — “|z_c⟩ = |critical⟩, z_c = √3/2 (THE LENS)”
  - README.md:167 — “critical point as THE LENS at z_c ≈ 0.8660254”
