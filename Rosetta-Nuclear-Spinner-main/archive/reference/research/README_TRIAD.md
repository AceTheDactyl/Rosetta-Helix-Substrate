# APL TRIAD Unlock Protocol — Implementation Guide

**Version:** 2.0.0  
**Status:** ✅ VALIDATED (All tests passing)  
**Date:** 2024-12-09

---

## Overview

The TRIAD Unlock Protocol enhances the Quantum-APL framework with:

1. **Clarified TRIAD Mechanism** (Phase 1) — Configurable unlock behavior
2. **Enhanced Helix Mapping** (Phase 2) — Dynamic operator window injection
3. **Cross-Domain Test Framework** (Phase 3) — Unified APL Seven Sentences harness
4. **Statistical Analysis Automation** (Phase 4) — Publication-ready analysis
5. **Documentation & CI** (Phase 5) — Comprehensive testing and validation

---

## Quick Start

### Run TRIAD Protocol Tests

```bash
# JavaScript tests
node tests/test_triad_protocol.js

# Python tests  
pytest tests/test_constants_helix.py -v

# APL Test Harness
python tests/apl_test_harness.py --list
python tests/apl_test_harness.py --sentence A3 --trials 50 --seed 42
python tests/apl_test_harness.py --all --trials 100

# Statistical Analysis
python tests/apl_analysis.py results/apl_test_results.json --plot --latex
```

---

## Phase 1: TRIAD Unlock Mechanism

### Constants (Single Source of Truth)

| Constant | Value | Description |
|----------|-------|-------------|
| `Z_CRITICAL` | √3/2 ≈ 0.8660254 | THE LENS — geometric truth (immutable) |
| `TRIAD_HIGH` | 0.85 | Rising-edge detection threshold |
| `TRIAD_LOW` | 0.82 | Re-arm threshold (hysteresis) |
| `TRIAD_T6` | 0.83 | Temporary t6 gate after unlock |
| `TRIAD_PASSES_REQ` | 3 | Configurable passes for unlock |

### Hysteresis Logic

```javascript
// QuantumClassicalBridge.js — TRIAD rising-edge detection
if (!this.triad.aboveBand && z >= TRIAD_HIGH) {
    this.triad.aboveBand = true;
    this.triad.completions += 1;      // Rising edge: increment
    if (this.triad.completions >= TRIAD_PASSES_REQ) {
        this.triad.unlocked = true;   // Unlock TRIAD gate
        this.quantum.setTriadUnlocked(true);
    }
} else if (this.triad.aboveBand && z <= TRIAD_LOW) {
    this.triad.aboveBand = false;     // Re-arm for next pass
}
```

### Configuration

```bash
# Override pass requirement
export QAPL_TRIAD_PASSES=2

# Force unlock
export QAPL_TRIAD_UNLOCK=1

# Enable debug logging
export QAPL_TRIAD_DEBUG=1
```

---

## Phase 2: Dynamic Helix Mapping

### HelixOperatorAdvisor API

```javascript
const { HelixOperatorAdvisor } = require('./src/helix_operator_advisor');

const advisor = new HelixOperatorAdvisor();

// Get helix hints for a z coordinate
const hints = advisor.describe(0.70);
// → { harmonic: 't5', operators: ['()', '×', '^', '÷', '+', '−'], 
//    truthChannel: 'PARADOX', deltaSneg: 0.123, ... }

// Dynamic operator window update (Phase 2 Enhancement)
advisor.updateOperatorWindow('t3', ['+', '×'], { source: 'external' });

// Get operator weights for selection
const weights = advisor.getOperatorWeights(0.70);
```

### Time Harmonic Tiers

| Tier | Z Range | Operators |
|------|---------|-----------|
| t1 | z < 0.10 | `()`, `−`, `÷` |
| t2 | z < 0.20 | `^`, `÷`, `−`, `×` |
| t3 | z < 0.40 | `×`, `^`, `÷`, `+`, `−` |
| t4 | z < 0.60 | `+`, `−`, `÷`, `()` |
| t5 | z < 0.75 | All six operators |
| t6 | z < t6_gate* | `+`, `÷`, `()`, `−` |
| t7 | z < 0.92 | `+`, `()` |
| t8 | z < 0.97 | `+`, `()`, `×` |
| t9 | z ≥ 0.97 | `+`, `()`, `×` |

*t6_gate = Z_CRITICAL (0.866) when locked, TRIAD_T6 (0.83) when unlocked

---

## Phase 3: APL Seven Sentences Test Framework

### Sentences

| ID | Token | Predicted Regime |
|----|-------|------------------|
| A1 | `d()\|Conductor\|geometry` | Isotropic lattices under collapse |
| A3 | `u^\|Oscillator\|wave` | Amplified vortex-rich waves |
| A4 | `m×\|Encoder\|chemistry` | Helical information carriers |
| A5 | `u×\|Catalyst\|chemistry` | Fractal polymer branching |
| A6 | `u+\|Reactor\|wave` | Jet-like coherent grouping |
| A7 | `u÷\|Reactor\|wave` | Stochastic decohered waves |
| A8 | `m()\|Filter\|wave` | Adaptive boundary tuning |

### Test Harness Usage

```python
from tests.apl_test_harness import APLTestHarness, APL_SENTENCES

harness = APLTestHarness(output_dir='results/')

# Run single sentence
result = harness.run_sentence_test('A3', n_trials=50, base_seed=42)

# Run all sentences
harness.run_all_sentences(n_trials=100)

# Save results
harness.save_results('experiment_results.json')
```

### Extending with Custom Simulators

```python
from tests.apl_test_harness import DomainSimulator

class MyCustomSimulator(DomainSimulator):
    def setup(self, config):
        # Initialize domain
        pass
    
    def apply_operator(self, operator, direction, **kwargs):
        # Apply APL operator
        pass
    
    def run(self, duration):
        # Run simulation
        return result
    
    def compute_metric(self, result, regime):
        # Compute regime-specific metric
        return metric_value
```

---

## Phase 4: Statistical Analysis

### Analysis Features

- **Parametric tests:** Welch's t-test
- **Non-parametric tests:** Mann-Whitney U
- **Effect sizes:** Cohen's d, rank-biserial correlation
- **Confidence intervals:** Bootstrap (10,000 samples)
- **Multiple comparisons:** Bonferroni, FDR correction
- **Visualizations:** Bar charts, forest plots, histograms
- **Export:** LaTeX tables, CSV

### Usage

```python
from tests.apl_analysis import APLResultsAnalyzer

analyzer = APLResultsAnalyzer(alpha=0.05)
report = analyzer.load_results('results/apl_test_results.json')

# Print formatted report
analyzer.print_report(report)

# Generate plots
analyzer.plot_results(report, output_dir='figures/', show=True)

# Export
analyzer.export_latex_table(report, 'tables/results.tex')
analyzer.export_csv(report, 'data/results.csv')
```

### Example Output

```
============================================================
APL TEST ANALYSIS REPORT
============================================================
Source: results/apl_test_results.json
Generated: 2024-12-09T12:00:00

Sentence    LHS Mean   Ctrl Mean   Cohen d    p-value   Sig
----------------------------------------------------------------------
A1            0.7234      0.4123     0.823     0.0012   **
A3            0.0871      0.0016     0.325     0.3123   
A4            0.5678      0.2345     0.712     0.0045   **
...

SUMMARY
----------------------------------------------------------------------
Total comparisons: 7
Significant (p < 0.05):
  - Uncorrected:  4
  - Bonferroni:   2
  - FDR:          3
Mean effect size (Cohen's d): 0.524
```

---

## Phase 5: CI & Documentation

### GitHub Actions Workflow

The CI workflow (`.github/workflows/triad-protocol-ci.yml`) includes:

1. **JavaScript Tests** — TRIAD, helix advisor, constants
2. **Python Tests** — Constants parity, helix mapping
3. **Constants Parity Check** — Ensures JS/Python consistency
4. **APL Test Harness** — Smoke test for Seven Sentences
5. **Nightly Helix Probes** — Characteristic z-value validation

### Running CI Locally

```bash
# JavaScript
node tests/test_triad_protocol.js

# Python
pytest tests/test_constants_helix.py -v

# Constants parity check
node -e "console.log(require('./src/constants').Z_CRITICAL)"
python -c "from quantum_apl_python.constants import Z_CRITICAL; print(Z_CRITICAL)"
```

---

## File Structure

```
quantum-apl-enhancements/
├── src/
│   ├── constants.js              # Phase 1: Enhanced constants
│   └── helix_operator_advisor.js # Phase 2: Dynamic helix mapping
├── tests/
│   ├── test_triad_protocol.js    # Phase 5: JS test suite
│   ├── test_constants_helix.py   # Phase 5: Python test suite
│   ├── apl_test_harness.py       # Phase 3: Test framework
│   └── apl_analysis.py           # Phase 4: Statistical analysis
├── results/                       # Test output directory
├── .github/
│   └── workflows/
│       └── triad-protocol-ci.yml # Phase 5: CI workflow
└── README_TRIAD.md               # This document
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QAPL_TRIAD_PASSES` | Passes required for unlock | 3 |
| `QAPL_TRIAD_UNLOCK` | Force TRIAD unlock | - |
| `QAPL_TRIAD_DEBUG` | Enable debug logging | false |
| `QAPL_TRIAD_COMPLETIONS` | Current completion count | - |
| `QAPL_RANDOM_SEED` | Reproducible RNG seed | - |
| `QAPL_EMIT_COLLAPSE_GLYPH` | Use ⟂ alias in tokens | false |

---

## Validation Checklist

- [x] **Phase 1:** TRIAD hysteresis passes 6/6 tests
- [x] **Phase 1:** Configurable `TRIAD_PASSES_REQ` working
- [x] **Phase 2:** Dynamic operator window updates working
- [x] **Phase 2:** Helix describe returns complete hints
- [x] **Phase 3:** All 7 APL sentences registered
- [x] **Phase 3:** Domain simulators (wave, geometry, chemistry) functional
- [x] **Phase 4:** Statistical analysis with effect sizes
- [x] **Phase 4:** Multiple comparison corrections (Bonferroni, FDR)
- [x] **Phase 5:** All 36 JS tests passing
- [x] **Phase 5:** CI workflow configured
- [x] **Phase 5:** Constants invariants validated

---

## References

- [Z_CRITICAL_LENS.md](docs/Z_CRITICAL_LENS.md) — Lens constant specification
- [APL-3.0-Quantum-Formalism.md](docs/APL-3.0-Quantum-Formalism.md) — Quantum formalism
- [apl-seven-sentences-test-pack.tex](reference/ace_apl/apl-seven-sentences-test-pack.tex) — Test pack LaTeX source
- [CONSTANTS_ARCHITECTURE.md](docs/CONSTANTS_ARCHITECTURE.md) — Constants architecture

---

## License

MIT License — See LICENSE file for details.
