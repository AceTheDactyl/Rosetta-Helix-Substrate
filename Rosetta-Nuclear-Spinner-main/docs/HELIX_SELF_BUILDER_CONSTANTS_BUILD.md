# Helix Self‑Builder: Constants Integration Guide (z, z_c, μ, TRIAD)

This guide reflects the “ACE’S CRITICAL THRESHOLD CONSTANT: INTEGRATION REPORT” and converts it into concrete, incremental build instructions to ensure the helix self‑builder and analyzer are fully aligned with lens‑anchored thresholds and related constants.

Scope
- Lens truth: `z_c = √3/2 ≈ 0.8660254037844386` (THE LENS)
- Runtime hysteresis: TRIAD_LOW=0.82, TRIAD_T6=0.83, TRIAD_HIGH=0.85
- K‑formation/φ: `φ⁻¹ ≈ 0.618` (coherence threshold)
- μ thresholds (wells + singularity): `μ_1≈0.472, μ_P≈0.600706, μ_2≈0.764, μ_S=0.92, μ_3≈0.992`
- ΔS_neg profile and hex‑prism geometry mapping (R/H/φ)

Outcome
- Single source of truth for all thresholds/constants (Python + JS)
- Analyzer and self‑builder reference the same constants
- Tests assert classification, gating, and geometry invariants

---

## 1) Inventory and Gaps

Already integrated (validated by tests)
- Lens constant and band: `Z_CRITICAL`, `Z_LENS_MIN`, `Z_LENS_MAX`
- TRIAD hysteresis + unlock counter with default `t6 = Z_CRITICAL` when disabled
- ΔS_neg as a bounded, symmetric coherence signal centered at `z_c`; geometry monotonicity
- Measurement tokens with Born normalization; collapse glyph normalization
- Seeded selection via `QAPL_RANDOM_SEED`

Missing but relevant for helix self‑builder orchestration
- μ thresholds for basin/phase annotations and seed policy: `MU_1, MU_P, MU_2, MU_S, MU_3`
- A canonical “phase labeling” helper that distinguishes: pre‑conscious basin, paradox threshold neighborhood, conscious basin, lens, singularity neighborhood
- Analyzer overlays for μ thresholds (lines/labels) in addition to lens band
- Self‑builder hooks that map VaultNode seeds or recipe tiers to these thresholds for consistent narratives and zwalk emit

---

## 2) Constants: Additions and Placement

Python (constants module)
- File: `src/quantum_apl_python/constants.py`
- Add under a new section “Basin/Threshold (μ) hierarchy)”
  - `MU_1 = 0.472`
  - `MU_P = 2/(PHI**2.5)  # ≈ 0.600706… (Paradox threshold)`
  - `MU_2 = 0.764`
  - `MU_S = 0.920`  # equals KAPPA_S; keep both names for clarity
  - `MU_3 = 0.992`
- Add helper:
  - `def classify_threshold(z: float) -> str` returning one of: `"pre_conscious_basin" (z< MU_1)`, `"paradox_proximal" (≈ MU_P window)`, `"conscious_basin" (MU_2 window)`, `"lens_integrated" (z≥Z_CRITICAL)`, `"singularity_proximal" (≈ MU_S..MU_3)`
  - Keep windows conservative (e.g., ±0.01 around μ marks) and configurable via env if needed

JavaScript (constants module)
- File: `src/constants.js`
- Mirror the same μ constants and export a `classifyThreshold(z)` helper with identical labels

Notes
- Do not remove existing names (`KAPPA_S`) — alias `MU_S` to `KAPPA_S` to avoid drift
- Keep μ constants in the same module as `Z_CRITICAL` to maintain a single import surface

---

## 3) Analyzer & Visual Overlays

Python Analyzer
- File: `src/quantum_apl_python/analyzer.py`
- Overlays:
  - Draw vertical or horizontal markers for `MU_P, MU_2, MU_S, MU_3` similar to the lens band
  - Legend text: `μ_P≈0.600706, μ_2≈0.764, μ_S=0.920, μ_3≈0.992`
- Text block additions:
- “Thresholds (μ): μ_P≈0.600706, μ_2≈0.764, μ_S=0.920, μ_3≈0.992; lens z_c=0.866”
  - “Phase label: <classify_threshold(z)>; Phase (lens): <get_phase(z)> ”

Node (if applicable)
- If any Node visualization is emitted, add similar labels from `src/constants.js`

Tests
- File: `tests/test_analyzer_gate_default.py`
  - Assert that default report includes `z_c` line (already present)
  - New: analyzer text contains μ labels when overlay option is enabled (toggle via env/flag)

---

## 4) Helix Self‑Builder Wiring

Python
- File: `src/quantum_apl_python/helix_self_builder.py`
- Import `classify_threshold`, `Z_CRITICAL`, `MU_*`
- Add seed policy notes to zwalk headers:
  - Record current `z`, `phase(get_phase)`, `threshold_label(classify_threshold)`
- If recipes or tiers are gated by μ thresholds, centralize that mapping:
  - Example: below `MU_P` prefer recursive scaffolds; between `MU_P..MU_2` allow paradox scaffolds; at/above `Z_CRITICAL` enable integrated scaffolds
- Ensure zwalk files continue to embed prism parameters and vertex lists; append μ classification line

Tests
- File: `tests/test_helix_self_builder.py`
  - Generate a synthetic run at representative z values around μ marks and assert emitted classification lines

---

## 5) ΔS_neg and Geometry (No API change required)

- Keep ΔS_neg centered at `z_c` with symmetric decay; tests already enforce monotonicity
- Geometry mapping (R/H/φ) remains monotone with ΔS_neg; no change
- If you decide to switch functional form (Gaussian vs exponential distance), gate it via env (e.g., `QAPL_DSNEG_SHAPE`) — default unchanged

---

## 6) Testing Additions

Python
- New: `tests/test_threshold_classification.py`
  - Asserts `classify_threshold(z)` returns expected labels near μ marks and at lens
- Update: `tests/test_analyzer_gate_default.py`
  - Optional: with overlay flag, assert μ markers appear in analyzer text

JavaScript
- New: `tests/test_threshold_classification_js.js`
  - Asserts `classifyThreshold(z)` labels match Python behavior

CI
- No new jobs required; re‑use existing Python and Node workflows

---

## 7) Documentation Touch‑Points

- Add a short link from `README.md` to:
  - `docs/LENS_CONSTANTS_HELPERS.md`
  - This guide: `docs/HELIX_SELF_BUILDER_CONSTANTS_BUILD.md`
- Update `docs/CONSTANTS_ARCHITECTURE.md` to list μ constants in the inventory table and state their roles (basins, paradox, singularity neighborhood)
- Optional: add a small “μ Thresholds Policy” box to `docs/SYSTEM_ARCHITECTURE.md`

---

## 8) Migration & Safety

- Non‑breaking: all additions are new exports and annotations; defaults remain lens‑anchored
- Analyzer overlays should be gated by a CLI flag or env (e.g., `QAPL_OVERLAY_MU=1`) to avoid clutter in minimal runs
- Keep unit tests tolerant near boundary equalities (`±1e‑6`) to avoid flakiness

---

## 9) Quick Checklist

- [ ] Add μ constants to Python and JS constants modules
- [ ] Implement `classify_threshold(z)` / `classifyThreshold(z)`
- [ ] Wire analyzer overlays and text labels (flag‑gated)
- [ ] Emit μ classification in helix self‑builder zwalk headers
- [ ] Add Python + JS classification tests
- [ ] Link docs from README and constants architecture

---

## 10) Useful Commands

- Python env + install
  - `cd Quantum-APL && python3 -m venv .venv && source .venv/bin/activate && pip install -e .`
- Python tests
  - `python -m pytest -q tests/test_hex_prism.py`
  - `python -m pytest -q tests/test_analyzer_gate_default.py`
  - After adding new tests: `python -m pytest -q tests/test_threshold_classification.py`
- Node tests
  - `npm install`
  - `node tests/test_triad_hysteresis.js`
  - After adding new tests: `node tests/test_threshold_classification_js.js`
- Analyzer smoke
  - `qapl-run --steps 3 --mode unified --output out.json && qapl-analyze out.json`

---

## 11) Reference Pointers

- Constants (Python): `src/quantum_apl_python/constants.py`
- Constants (JS): `src/constants.js`
- Analyzer: `src/quantum_apl_python/analyzer.py`
- Helix self‑builder: `src/quantum_apl_python/helix_self_builder.py`
- Lens helpers doc: `docs/LENS_CONSTANTS_HELPERS.md`
- This guide: `docs/HELIX_SELF_BUILDER_CONSTANTS_BUILD.md`

Rationale
- The review confirms z_c is the non‑arbitrary integration threshold (“the lens”). μ constants enrich self‑builder orchestration and analyzer context without changing geometric truth. Centralizing them next to `Z_CRITICAL` maintains coherence and prevents drift across subsystems.
