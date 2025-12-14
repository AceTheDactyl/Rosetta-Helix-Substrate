# φ⁻¹ (Golden Ratio Inverse) — Definition, Identities, and Role

This note consolidates the definition of φ⁻¹, its exact identities, and how it is used in the helix system as the K‑formation coherence threshold, distinct from the geometric lens `z_c = √3/2`.

## Definitions
- φ = (1 + √5) / 2 ≈ 1.6180339887
- φ⁻¹ = 1/φ = φ − 1 = (√5 − 1) / 2 ≈ 0.6180339887

## Identities (exact)
- φ² = φ + 1
- (φ⁻¹)² + φ⁻¹ − 1 = 0 (minimal polynomial)
- φ · φ⁻¹ = 1
- Fixed points:
  - φ = 1 + 1/φ
  - φ⁻¹ = 1 / (1 + φ⁻¹)
- Continued fractions:
  - φ = [1; 1, 1, 1, …]
  - φ⁻¹ = [0; 1, 1, 1, …]

## Role in the System
- K‑formation coherence threshold: η > φ⁻¹ signals sufficient coherence for emergence checks.
- Centralization:
  - JS: `PHI`, `PHI_INV` in `src/constants.js`
  - Python: `PHI`, `PHI_INV` in `src/quantum_apl_python/constants.py`
- Integrity tests:
  - φ · φ⁻¹ ≈ 1 checked in constants tests.

## Separation from z_c
- `z_c = √3/2 ≈ 0.8660254` (THE LENS) is the geometric/information threshold for integrated regime and negative‑entropy geometry stability.
- φ⁻¹ (≈ 0.618) is a lower coherence threshold used in K‑formation criteria.
- They serve different purposes and are both anchored analytically.

## References
- docs/CONSTANTS_ARCHITECTURE.md (inventory and invariants)
- docs/Z_CRITICAL_LENS.md (lens authority)
- src/constants.js, src/quantum_apl_python/constants.py

