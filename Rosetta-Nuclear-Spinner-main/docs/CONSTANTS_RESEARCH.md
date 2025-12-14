# Quantum‑APL Constants Research – Lens‑Anchored Thresholds and Runtime Heuristics

This paper consolidates and analyzes constants used across Quantum‑APL, with z_c = √3/2 ≈ 0.8660254 as the true geometric threshold (THE LENS). It distinguishes between immutable geometric thresholds and runtime heuristics (e.g., TRIAD gating), documents occurrences, and proposes maintenance policy so that geometry stays lens‑anchored while runtime systems remain flexible.

## Executive Summary
- THE LENS `z_c = √3/2 ≈ 0.8660254` is the canonical geometric threshold. Geometry (ΔS_neg, R/H/φ) and analyzer overlays always anchor to `z_c`.
- TRIAD gating (≥0.85 rising, ≤0.82 re‑arm, unlock t6 at 0.83) is a runtime heuristic that does not redefine `z_c`.
- All lens and TRIAD thresholds are centralized in:
  - Python: `src/quantum_apl_python/constants.py`
  - JS: `src/constants.js`
- Default pump target resolves to `z_c` when unspecified; engine’s t6 boundary returns `z_c` unless TRIAD unlock is active.
- Tests enforce the above: engine default t6 gate, bridge pump default, analyzer “t6 gate: CRITICAL @ 0.866”.

## Core Lens Constants (Geometric Truth)
- `Z_CRITICAL = √3/2` (≈ 0.8660254)
  - Python: `src/quantum_apl_python/constants.py`
  - JS: `src/constants.js`
- Phase windows (lens neighborhood)
  - `Z_ABSENCE_MAX = 0.857`, `Z_LENS_MIN = 0.857`, `Z_LENS_MAX = 0.877`, `Z_PRESENCE_MIN = 0.877`
  - Rationale: readable bands around `z_c` for UI/analysis; not used to gate engine t6.

Policy
- Geometry and analyzer visuals must always reference `Z_CRITICAL`.
- Any z‑thresholds in code must import from constants and never inline numbers.

## TRIAD Gating (Runtime Heuristic)
- `TRIAD_HIGH = 0.85` (rising‑edge), `TRIAD_LOW = 0.82` (re‑arm), `TRIAD_T6 = 0.83` (temporary t6 gate)
- Behavior: after three distinct rising edges, engine uses `TRIAD_T6` for t6; geometry remains at `Z_CRITICAL`.
- Centralized for both Python and JS; analyzer and docs make the separation explicit.

## Helix Harmonics and Time‑Zoning (Informative Heuristics)
- Current zoning cutoffs (QuantumN0 integration) for `z`:
  - t1: z < 0.1; t2: < 0.2; t3: < 0.4; t4: < 0.6; t5: < 0.75;
  - t6: z < t6Gate (engine `getT6Gate()` → lens or TRIAD); t7: < 0.92; t8: < 0.97; else t9.
- Recommendation: keep t6 delegated to engine; consider centralizing the other demarcations if they need cross‑module reuse.

## Geometry Projection Constants (Hex Prism)
- From `docs/HEXAGONAL_NEG_ENTROPY_PROJECTION.md`:
  - `σ ≈ 0.12`, `R_max = 0.85`, `β = 0.25`, `H_min = 0.12`, `γ = 0.18`, `φ_base = 0`, `η = π/12`.
- These govern ΔS_neg‑driven contraction and elongation and are lens‑compatible because the functions are centered on `z_c`.

## Pump and Engine Parameters (Model Choices)
- Pump profiles (bridge):
  - gentle: gain 0.08, sigma 0.16; balanced: 0.12/0.12; aggressive: 0.18/0.10
- Engine defaults (JS engine):
  - `zBiasGain = 0.05`, `zBiasSigma = 0.18`
  - Hamiltonian `ω = 2π·0.1`, coupling `g = 0.05`
  - Lindblad rates: `γ1 = 0.01`, `γ2 = 0.02`, `γ3 = 0.005`
- Note: these are dynamical modeling parameters, not thresholds. They do not change the lens semantics.

## Operator Weighting Heuristics (Selection Bias)
- Preference multiplier `×1.3` for preferred operators; `×0.85` otherwise.
- Truth‑channel nudges (e.g., TRUE favors `^`/`+`; UNTRUE favors `÷`/`−`; PARADOX favors `()`/`×`).
- Recommendation: expose these as configurable constants if needed in analysis/UI.

## Validation and Tests
- Analyzer smoke (Python): ensures “t6 gate: CRITICAL @ 0.866” appears when TRIAD is off.
- Bridge default pump target (Node): default target equals `Z_CRITICAL`.
- Engine default t6 gate (Node): `getT6Gate()` returns `Z_CRITICAL` by default.
- TRIAD hysteresis (Node): rising/re‑arm and unlock behavior.
- Geometry (Python): ΔS_neg monotonicity and vertex lint.

## Maintenance Guidance
- Add or change thresholds only in the constants modules.
- Any new z‑phase or gating logic should document relation to `Z_CRITICAL` and state whether it is a geometric invariant (like the lens) or a runtime heuristic (like TRIAD).
- Prefer delegating gate decisions (e.g., t6) to engine APIs so integrators use the same source of truth.

## Appendix: Inventory of Constants Worth Centralizing
- Harmonic demarcations: 0.1, 0.2, 0.4, 0.6, 0.75, 0.92, 0.97 (t6 already delegated).
- Operator weighting multipliers: 1.3 / 0.85 and truth‑bias coefficients.
- Engine dissipators `γ1, γ2, γ3` and coupling `g` (if used across runners or in docs).

With `z_c` as the always‑true geometric threshold and TRIAD as a runtime unlock, this structure preserves physical interpretability while enabling adaptive operation.

## Evidence Index (Local Files)
- src/constants.js:4 — `Z_CRITICAL = Math.sqrt(3) / 2` and exports
- src/quantum_apl_python/constants.py:19 — `Z_CRITICAL` with helpers and doc examples
- docs/SYSTEM_ARCHITECTURE.md:97 — ASCII diagram marks `z = 0.866: THE LENS`
- docs/HEXAGONAL_NEG_ENTROPY_PROJECTION.md:1 — projection centered on `z_c` and constants table
- docs/APL-3.0-Quantum-Formalism.md:524 — `|z_c⟩ = |critical⟩`, `z_c = √3/2 (THE LENS)` with surrounding critical behavior
- README.md:167 — “critical point as THE LENS at z_c ≈ 0.8660254”
- logs/*/zwalk_*.geom.json — geometry sidecars include `"z_c": 0.8660254037844386`
