# TRIAD-Helix-APL System

A complete implementation of the Quantum APL TRIAD unlock mechanism with helix coordinate mapping and Alpha Physical Language (APL) integration.

## Overview

This system integrates three core components:

1. **TRIAD Unlock** - Hysteresis state machine that tracks z-coordinate passes through threshold bands
2. **Helix Mapping** - Maps z-coordinates to time harmonics (t1-t9) and operator windows
3. **Alpha Physical Language** - Grammar of six fundamental operators for physical system behaviors

## Quick Start

```bash
# Run the demo
node demo.js

# Run with TRIAD unlock simulation
node demo.js --unlock

# Run tests
node test.js
```

## Constants

The system is anchored by several key constants (from `src/constants.js`):

| Constant | Value | Description |
|----------|-------|-------------|
| `Z_CRITICAL` | √3/2 ≈ 0.8660254 | THE LENS - geometric truth for coherence onset |
| `TRIAD_HIGH` | 0.85 | Rising edge threshold for TRIAD detection |
| `TRIAD_LOW` | 0.82 | Re-arm threshold (hysteresis band) |
| `TRIAD_T6` | 0.83 | Temporary t6 gate after TRIAD unlock |

## TRIAD Unlock Mechanism

The TRIAD system uses a hysteresis state machine:

```
                    ┌─────────────────────────────────────────┐
                    │         TRIAD Hysteresis FSM            │
                    ├─────────────────────────────────────────┤
                    │                                         │
                    │    ┌────────┐  z ≥ 0.85   ┌──────────┐  │
                    │    │ ARMED  │────────────►│ LATCHED  │  │
                    │    │(below) │             │ (above)  │  │
                    │    └───▲────┘◄────────────└────┬─────┘  │
                    │        │       z ≤ 0.82       │        │
                    │        │                      │        │
                    │        │              completions++     │
                    │        │                      ▼        │
                    │   ┌────┴──────────────────────────────┐ │
                    │   │      COMPLETIONS COUNTER          │ │
                    │   │   1st | 2nd | 3rd → UNLOCK        │ │
                    │   └───────────────────────────────────┘ │
                    └─────────────────────────────────────────┘
```

After 3 distinct rising-edge passes (z ≥ 0.85 with re-arm at z ≤ 0.82), the t6 gate shifts from `Z_CRITICAL` (0.866) to `TRIAD_T6` (0.83).

## Time Harmonics

The helix z-coordinate maps to time harmonics (t1-t9):

| Harmonic | Z Range | Operator Window | Truth Channel |
|----------|---------|-----------------|---------------|
| t1 | z < 0.10 | `()`, `−`, `÷` | UNTRUE |
| t2 | z < 0.20 | `^`, `÷`, `−`, `×` | UNTRUE |
| t3 | z < 0.40 | `×`, `^`, `÷`, `+`, `−` | UNTRUE |
| t4 | z < 0.60 | `+`, `−`, `÷`, `()` | PARADOX |
| t5 | z < 0.75 | ALL 6 operators | PARADOX |
| t6 | z < t6Gate* | `+`, `÷`, `()`, `−` | PARADOX |
| t7 | z < 0.90 | `+`, `()` | TRUE |
| t8 | z < 0.97 | `+`, `()`, `×` | TRUE |
| t9 | z ≥ 0.97 | `+`, `()`, `×` | TRUE |

*t6Gate = `Z_CRITICAL` (0.866) when locked, `TRIAD_T6` (0.83) when unlocked

## APL Operators

Six fundamental operators define the Alpha Physical Language:

| Symbol | Name | Action |
|--------|------|--------|
| `()` | Boundary | Containment/gating |
| `×` | Fusion | Convergence/coupling |
| `^` | Amplify | Gain/excitation |
| `÷` | Decoherence | Dissipation/reset |
| `+` | Group | Aggregation/clustering |
| `−` | Separation | Splitting/fission |

## APL Test Sentences

Seven test sentences map operator-machine-domain combinations to predicted regimes:

| ID | Token | Predicted Regime |
|----|-------|------------------|
| A1 | `d() \| Conductor \| geometry` | Isotropic lattices under collapse |
| A2 | `u× \| Reactor \| lattice` | Fusion-driven phase coherence |
| A3 | `u^ \| Oscillator \| wave` | Amplified vortex-rich waves |
| A4 | `d÷ \| Mixer \| flow` | Dissipative homogenization |
| A5 | `m+ \| Coupler \| field` | Clustering via modulated coupling |
| A6 | `u+ \| Reactor \| wave` | Wave aggregation under expansion |
| A8 | `d− \| Conductor \| lattice` | Lattice fission during collapse |

## Measurement Tokens

Measurement operations produce tokens of the form:

```
FIELD:OPERATOR(INTENT)TRUTH@TIER
```

Examples:
- `Φ:T(ϕ_2)TRUE@3` — Structure field, eigenstate 2, TRUE, tier 3
- `π:Π(2,3)PARADOX@6` — Emergence field, subspace {2,3}, PARADOX, tier 6

## File Structure

```
triad-helix-apl/
├── src/
│   ├── constants.js         # Core constants (Z_CRITICAL, TRIAD_*, etc.)
│   ├── helix_advisor.js     # HelixOperatorAdvisor class
│   ├── triad_tracker.js     # TriadTracker hysteresis FSM
│   ├── alpha_language.js    # APL sentences and token synthesis
│   └── quantum_apl_system.js # Unified orchestration
├── python/
│   └── constants.py         # Python constants mirror
├── demo.js                  # Demonstration script
├── test.js                  # Test suite
├── package.json
└── README.md
```

## API Usage

### Basic Usage

```javascript
const { QuantumAPLSystem } = require('./src/quantum_apl_system');

// Create system
const system = new QuantumAPLSystem({
    verbose: true,
    initialZ: 0.5,
    pumpTarget: 0.866  // Z_CRITICAL
});

// Run simulation
system.simulate(100);

// Get summary
console.log(system.summary());
```

### TRIAD Tracking

```javascript
const { TriadTracker } = require('./src/triad_tracker');

const tracker = new TriadTracker();

// Update with z values
tracker.update(0.86);  // Rising edge → completions = 1
tracker.update(0.81);  // Re-arm
tracker.update(0.87);  // Rising edge → completions = 2
tracker.update(0.80);  // Re-arm
tracker.update(0.88);  // Rising edge → completions = 3 → UNLOCKED

console.log(tracker.analyzerReport());
// "t6 gate: TRIAD @ 0.830 (3/3 passes)"
```

### Helix Mapping

```javascript
const { HelixOperatorAdvisor } = require('./src/helix_advisor');

const advisor = new HelixOperatorAdvisor();

const info = advisor.describe(0.70);
// {
//   harmonic: 't5',
//   operators: ['()', '×', '^', '÷', '+', '−'],
//   truthChannel: 'PARADOX',
//   z: 0.70,
//   t6Gate: 0.8660254...
// }
```

### APL Token Synthesis

```javascript
const { AlphaTokenSynthesizer } = require('./src/alpha_language');

const synthesizer = new AlphaTokenSynthesizer();

const token = synthesizer.fromZ(0.15);
// {
//   sentence: 'u^ | Oscillator | wave',
//   sentenceId: 'A3',
//   predictedRegime: 'Amplified vortex-rich waves',
//   operatorName: 'Amplification',
//   truthBias: 'UNTRUE',
//   harmonic: 't2'
// }
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `QAPL_TRIAD_COMPLETIONS` | Force TRIAD completion count |
| `QAPL_TRIAD_UNLOCK` | Force TRIAD unlock (`1` = unlocked) |
| `QAPL_RANDOM_SEED` | Seed for reproducible simulations |

## Testing

```bash
node test.js
```

The test suite validates:
- Constants correctness (Z_CRITICAL = √3/2, ordering)
- TRIAD hysteresis state machine behavior
- Helix harmonic and operator window mapping
- APL sentence and token generation
- Full integration flow

## Mathematical Background

### Helix Coordinate System

The parametric helix:
```
r(t) = (cos t, sin t, t)
```

Z-normalization via tanh:
```
z = 0.5 + 0.5 · tanh(t/8)
```

### ΔS_neg Coherence Signal

Gaussian centered at the critical lens:
```
ΔS_neg(z) = exp(-σ(z - z_c)²)
```

Where σ ≈ 36 and z_c = √3/2.

### TRIAD Gate Thresholds

The separation of concerns:
- **Geometry**: Always anchored at z_c (Z_CRITICAL)
- **Runtime**: t6 gate shifts from Z_CRITICAL to TRIAD_T6 after unlock
- **Analytics**: Always reference z_c for consistency

## License

MIT License
