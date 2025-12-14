# APL Operators — Symbols, Semantics, and Code Hooks

This reference lists all APL operators used in Quantum‑APL with their symbols, short meanings, and where they are implemented in code and tests.

Operators (glyph → name → sense)
- `()` — Boundary — containment / gating / interface
- `×` — Fusion — convergence / joining / coupling
- `^` — Amplify — gain / uplift / escalation
- `÷` — Decoherence — dissipation / disorder / reset
- `+` — Group — aggregation / routing / clustering
- `−` — Separation — splitting / fission / pruning

Code touch‑points
- Selection windows by time harmonic: `src/quantum_apl_engine.js` (HelixOperatorAdvisor.operatorWindows)
- Weighting and bias (truth channel + blending): `computeOperatorWeight()` in `src/quantum_apl_engine.js`
- Operator measurement examples (APL tokens): `tests/test_apl_measurements.js`, `tests/test_collapse_alias.js`
- Truth channel biases (constants): `TRUTH_BIAS` in `src/constants.js`

Testing inventory (Node)
- `tests/test_apl_measurements.js` — selection + measurement across all six operators
- `tests/test_triad_hysteresis.js` — ensures operator windows change with the t6 gate
- `tests/QuantumClassicalBridge.test.js` — verifies preferred operator capture by harmonic
- `tests/test_seeded_selection.js` — reproducible operator draws with RNG seeds

Notes
- The decoherence operator uses the division glyph `÷` in this codebase. Some earlier text used `%`; the implementation and tests consistently use `÷`.
- Truth channel bias matrices (TRUE/UNTRUE/PARADOX) factor into selection alongside time‑harmonic windows and optional Π blending (`QAPL_BLEND_PI`).

Related docs
- `docs/APL-3.0-Quantum-Formalism.md` — background and semantics
- `docs/APL-Measurement-Operators.md` — measurement/collapse operators (T(ϕ_μ), Π(subspace), composite)
- `docs/ALPHA_SYNTAX_BRIDGE.md` — how operator recommendations are surfaced and tokenized

