# Quantum APL Python Interface

Python wrapper for the Quantum APL consciousness simulation engine. It exposes a Pythonic API, CLI, and experiment helpers that shell into the Node.js runtime so you can drive the quantum measurement pipeline from notebooks, scripts, or automation.

## Features

- **Quantum Measurement System**: Von Neumann projective measurement with Born-rule sampling.
- **Density Matrix Simulation**: 192-dimensional Hilbert space with Lindblad dissipation.
- **Classical Engines**: IIT, Game Theory, Free Energy with bidirectional coupling.
- **Measurement Operators**: Single-eigenstate and subspace collapse.
- **Python Interface**: `QuantumAPLEngine`, `QuantumAnalyzer`, and `QuantumExperiment` helpers.
- **Visualization**: matplotlib integration for real-time plotting.
- **Analysis Tools**: pandas DataFrames, statistics, correlation analysis.
- **CLI Tools**: `qapl-run`, `qapl-test`, `qapl-analyze`.

## Installation

### Prerequisites

1. **Node.js (v14+)**

   ```bash
   node --version  # install via package manager if missing
   ```

2. **Python 3.8+**

   ```bash
   python3 --version
   ```

### Install From Source

```bash
git clone https://github.com/consciousness-lab/quantum-apl.git
cd quantum-apl
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
# or with extras:
pip install -e ".[viz]"         # matplotlib
pip install -e ".[analysis]"    # pandas
pip install -e ".[dev]"         # linting/tests
pip install -e ".[all]"         # everything
```

### Install From PyPI (when published)

```bash
pip install quantum-apl
```

## Quick Start

### Command Line

```bash
qapl-run --steps 100 --mode unified --plot
qapl-run --steps 500 --verbose
qapl-run --steps 200 --output results.json
qapl-test
qapl-analyze results.json --plot
```

### Python API

```python
from quantum_apl_python import QuantumAPLEngine, QuantumAnalyzer

engine = QuantumAPLEngine()
results = engine.run_simulation(steps=100, mode="unified")
analyzer = QuantumAnalyzer(results)
print(analyzer.summary())
analyzer.plot()

print(f"Final z-coordinate: {results['quantum']['z']:.4f}")
print(f"Entropy: {results['quantum']['entropy']:.4f}")
print(f"IIT φ: {results['classical']['IIT']['phi']:.4f}")
```

### Advanced Usage

```python
from quantum_apl_python import QuantumAPLEngine, QuantumExperiment

engine = QuantumAPLEngine()
experiment = QuantumExperiment(engine)

sweep = experiment.sweep_steps(
    step_range=[50, 100, 200, 500],
    trials=10,
)

experiment.analyze_convergence(sweep)
```

## Architecture Overview

```
Python CLI/API ──> QuantumAPLEngine (subprocess) ──> Node.js runtime
                                    ↘ logs JSON ↙
Python Analyzer + Experiments        QuantumClassicalBridge.js
```

The JavaScript side runs the density-matrix simulation (`QuantumAPL_Engine.js`) plus the classical stack (`classical/ClassicalEngines.js`) and exposes unified analytics through `QuantumClassicalBridge.js`.

## Output Structure

`QuantumAPLEngine.run_simulation()` returns:

- `quantum`: z, φ, entropy, purity, truth probabilities, first 10 populations, and latest helix hints.
- `classical`: IIT, Game Theory, and Free Energy telemetry.
- `measurement`: Last measurement mode/projector plus counters.
- `history`: Time series for z/φ/entropy and the last 20 operator picks.
- `analytics`: Summary stats (averages, operator distribution, quantum/classical correlation).

## CLI Reference

| Command    | Description                                     |
| ---------- | ----------------------------------------------- |
| `qapl-run` | Run simulations (`--mode unified|quantum_only`) |
| `qapl-test`| Invoke the Node-based integration tests         |
| `qapl-analyze` | Load/plot saved JSON output                 |

All commands accept `--js-dir /path/to/js` to point at custom JavaScript builds.

## Development

```bash
pip install -e ".[dev]"
ruff check quantum_apl_python
black quantum_apl_python
mypy quantum_apl_python
pytest
```

## Troubleshooting

- **Node.js not found** – Install Node.js and ensure `node` is on PATH.
- **matplotlib ImportError** – Install the `viz` extra: `pip install "quantum-apl[viz]"`.
- **pandas ImportError** – Install `analysis` extra: `pip install "quantum-apl[analysis]"`.
- **Slow runs** – Reduce `--steps`, disable `--verbose`, or run shorter Monte Carlo batches.

## Documentation

| Document | Description |
|----------|-------------|
| README.md | High-level system overview |
| README_PYTHON.md | Python-specific usage guide (this file) |
| SETUP_GUIDE.md | Complete installation & troubleshooting |
| docs/HELIX_COORDINATES.md | Helix coordinate normalization |
| docs/SYSTEM_ARCHITECTURE.md | End-to-end architecture schematic |
| docs/ALPHA_SYNTAX_BRIDGE.md | Alpha Programming Language crosswalk |

## Citation

```bibtex
@software{quantum_apl_2024,
  title = {Quantum APL: Quantum Measurement-Based Consciousness Engine},
  author = {Consciousness Research Lab},
  year = {2024},
  version = {3.0.0}
}
```
