# Alpha-Physical Language (APL) — Seven Sentences Test Pack

A minimal operator grammar for describing physical system behaviors across geometry, waves, chemistry, and biology.

## Overview

**APL (Alpha-Physical Language)** is an experimental framework that uses compact "sentences" to predict physical regimes across diverse domains. This repository contains a test pack designed to allow independent teams to validate whether APL's core operator language has genuine physical content.

The test pack translates 7 compact APL sentences into **falsifiable, cross-domain hypotheses** that can be probed with standard models:
- Navier–Stokes (fluid dynamics)
- Wave equations (electromagnetics, acoustics)
- Reaction–diffusion (chemistry)
- Phase-field models (interfaces, materials)
- Polymer growth / aggregation

## Core Concept

APL describes physical systems using:

### Fields ("Spirals")
- **Φ** — Structure field (geometry, lattice, boundaries)
- **e** — Energy field (waves, thermodynamics, flows)
- **π** — Emergence field (information, chemistry, biology)

### Universal Operations
- `()` — Boundary / containment
- `×` — Fusion / convergence / joining
- `^` — Amplify / gain
- `%` — Decohere / noise / reset
- `+` — Group / aggregation / routing
- `–` — Separate / splitting / fission

### Operator States (UMOL: Universal Modulation Operator Law)
- **u** (𝒰) — Expansion / forward projection
- **d** (𝒟) — Collapse / backward integration
- **m** (CLT) — Modulation / coherence lock

An APL sentence has the form:
```
[Direction][Op] | [Machine] | [Domain] → [Regime/Behavior]
```

For example: `u^|Oscillator|wave` reads as "Forward amplification in an oscillatory machine in a wave domain."

## The Seven Test Sentences

Each sentence is a **testable hypothesis** predicting that specific operator-machine-domain combinations statistically favor particular physical regimes:

| # | Sentence | Predicted Regime | Domain |
|---|----------|------------------|--------|
| **A3** | `u^|Oscillator|wave` | Closed vortex / recirculation | Wave dynamics |
| **A7** | `u%|Reactor|wave` | Turbulent decoherence | Flow/wave systems |
| **A1** | `d()|Conductor|geometry` | Isotropic lattice / sphere | Geometry/interfaces |
| **A4** | `m×|Encoder|chemistry` | Helical encoding | Chemistry/polymers |
| **A5** | `u×|Catalyst|chemistry` | Branching networks | Chemistry/growth |
| **A6** | `u+|Reactor|wave` | Focusing jet / beam | Fluid/plasma/wave |
| **A8** | `m()|Filter|wave` & `d×|Catalyst|chemistry` | Adaptive filter / selectivity | Wave & chemistry |

### Interpretation Rule

For all sentences:
```
LHS → RHS
```
means:

> If a system is built to match the **left-hand side** (LHS) structure and driving, then the **right-hand side** (RHS) regime should appear **more often, more strongly, or at lower thresholds** than in controls that break the LHS structure, with all else as equal as possible.

**Evidence FOR APL:** Clear, reproducible overrepresentation of the RHS regime under LHS conditions vs. controls.

**Evidence AGAINST APL:** No such bias, or controls produce the RHS regime equally or more often.

## Repository Structure

```
APL/
├── README.md                           # This file
├── apl-operators-manual.tex            # Complete operator reference (LaTeX)
├── apl-seven-sentences-test-pack.tex   # Complete test protocol (LaTeX)
├── COMPILE_INSTRUCTIONS.md             # LaTeX compilation guide
└── docs/                               # Documentation and compiled outputs
    ├── index.html                      # HTML version of operator's manual
    ├── apl-operators-manual.pdf        # PDF version (auto-compiled)
    └── apl-seven-sentences-test-pack.pdf  # Test pack PDF
```

## Testing Strategy

For each sentence, the recommended approach:

1. **Choose a standard model** appropriate to the domain
   - Geometry: phase-field, Cahn–Hilliard, curvature flow
   - Flows/waves: Navier–Stokes, lattice Boltzmann, wave equation
   - Chemistry: reaction–diffusion, polymerization, DLA, kinetic Monte Carlo

2. **Implement the LHS conditions**
   - `u^`: Add gain/amplification at resonant modes
   - `u%`: Add explicit stochastic/decohering forcing
   - `d()`: Allow boundaries to relax/collapse under isotropic energy
   - `m()`: Modulate boundaries in response to passing modes
   - `u×`/`d×`: Implement forward-biased or collapse–fusion catalysts
   - `u+`: Add grouping/convergent geometry or fields

3. **Design matched controls**
   - Remove or invert the key operator
   - Keep everything else comparable

4. **Define regime metrics** (A1–A8)
   - A1: Sphericity, surface/volume ratio, packing isotropy
   - A3: Vortex count/lifetime, closed streamline fraction
   - A4: Helical order parameters, information capacity
   - A5: Fractal dimension, branching degree
   - A6: Jet opening angle, centerline coherence
   - A7: Spectral width, RMS fluctuations, Lyapunov exponents
   - A8: Adaptive sharpening, retuning capability

5. **Sweep parameters** and compare
   - Drive strength, noise amplitude, surface tension, catalytic bias, etc.
   - Run multiple realizations
   - Check whether LHS conditions robustly bias metrics toward target regime

## Preliminary Results

The test pack includes two toy numerical checks:

- **A1 (Isotropic cluster):** 2D point collapse under isotropic central force → circular, angle-isotropic cluster ✓
- **A5 (Branching networks):** 2D diffusion-limited aggregation → fractal branching structure (D ≈ 1.2) ✓

These are minimal sandbox experiments consistent with APL predictions. Full testing requires domain-appropriate models across all seven sentences.

## Requirements

**To use this test pack, you need:**
- Basic familiarity with physics, chemistry, and data analysis
- NO prior knowledge of CET or other APL meta-structures
- Access to standard simulation tools (Python/NumPy, MATLAB, COMSOL, OpenFOAM, etc.)

**Assumptions:**
- Independent testing by teams without invested stake in APL
- Falsifiable approach: both positive and negative results are valuable

## Documentation

### APL Operator's Manual

A comprehensive reference guide for APL operators, syntax, and usage patterns is available in multiple formats:

- **HTML Version:** Open `docs/index.html` in your browser for an interactive, responsive manual
- **PDF Version:** Automatically compiled from LaTeX source via GitHub Actions
- **GitHub Pages:** Available online if GitHub Pages is enabled for this repository

The manual includes:
- Complete operator reference with symbols and meanings
- Field definitions (the three "spirals": Φ, e, π)
- Operator state modulation (UMOL)
- Machine contexts and domains
- Syntax rules and sentence structure
- Usage patterns and examples
- Quick reference tables

### Compilation

To compile the LaTeX documents locally:
```bash
pdflatex -interaction=nonstopmode apl-operators-manual.tex
pdflatex -interaction=nonstopmode apl-seven-sentences-test-pack.tex
```

See `COMPILE_INSTRUCTIONS.md` for detailed instructions.

## Getting Started

1. **Read the operator's manual:**
   - Open `docs/index.html` in your browser, or
   - Compile the PDF: `pdflatex apl-operators-manual.tex`

2. **Read the test pack:**
   ```bash
   pdflatex apl-seven-sentences-test-pack.tex
   ```
   Or view the `.tex` source directly.

3. **Choose a sentence to test** (recommend starting with A1, A3, or A5 as they have clear metrics)

4. **Implement the test protocol** using your preferred simulation framework

5. **Report results** — both confirmations and refutations help refine or reject APL

## Philosophy

APL is designed to be **falsifiable**. The language should stand or fall on whether these seven sentences predict robust, cross-domain biases in real physical and chemical systems — nothing more and nothing less.

If the predicted regimes are NOT overrepresented under the specified conditions, that is strong evidence against APL's validity.

## Contributing

This is an open scientific test. Contributions of:
- Test implementations
- Simulation results (positive or negative)
- Refined control designs
- Additional metrics
- Critical analysis

...are all welcome and valuable.

## License

This work is provided for scientific testing and educational purposes.

## Citation

If you use this test pack in your research, please reference:
```
APL Seven Sentences Test Pack v1.0
Alpha-Physical Language Testing Framework
```

## Contact

For questions, results, or collaboration inquiries, please open an issue in this repository.
