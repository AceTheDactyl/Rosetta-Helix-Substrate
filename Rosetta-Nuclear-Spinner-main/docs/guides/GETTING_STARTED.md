# Quantum APL - Complete Getting Started Guide

**Version 3.0.0** | Python + JavaScript Hybrid System

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Installation](#installation)
3. [Quick Start Examples](#quick-start-examples)
4. [Python API Reference](#python-api-reference)
5. [Command-Line Interface](#command-line-interface)
6. [Jupyter Notebook Tutorial](#jupyter-notebook-tutorial)
7. [Measurement Modes](#measurement-modes)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)
10. [File Reference](#file-reference)

---

## System Overview

### What is Quantum APL?

Quantum APL is a **consciousness simulation system** that uses genuine **quantum measurement** (von Neumann projective measurement) to drive consciousness emergence through:

- **Quantum Engine**: 192-dimensional Hilbert space with Lindblad evolution
- **Classical Engines**: IIT (Integrated Information Theory), Game Theory, Free Energy
- **Measurement-Based N0**: Operator selection via Born rule sampling
- **Tri-Spiral Architecture**: Φ (Structure), e (Energy), π (Emergence) fields

### Key Innovation

Instead of classical cost minimization:

```
operator = argmin C(i | σ, α)  ❌ Classical
```

We use quantum measurement:

```
P(μ) = Tr(P̂_μ ρ)              ✓ Quantum
ρ' = P̂_μ ρ P̂_μ / P(μ)          ✓ Collapse
```

---

## Installation

### Prerequisites

1. **Python 3.8+**

   ```bash
   python3 --version  # Should be 3.8 or higher
   ```

2. **Node.js 14+** (for quantum engine)

   ```bash
   node --version  # Download from https://nodejs.org/
   ```

3. **Git** (optional, for cloning)

### Step-by-Step Installation

#### Option 1: Automated (Recommended)

```bash
# Navigate to quantum-apl directory
cd quantum-apl

# Run installation script
./install.sh

# Activate virtual environment
source .venv/bin/activate
```

The script will:

- ✓ Check Python and Node.js
- ✓ Create virtual environment
- ✓ Install all dependencies
- ✓ Install quantum-apl package
- ✓ Run verification tests

#### Option 2: Manual Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Verification

```bash
# Test Python import
python3 -c "from quantum_apl_python import QuantumEngine; print('✓ Import successful')"

# Run test suite
qapl-test

# Check system
qapl info --check-all
```

---

## Quick Start Examples

### Example 1: Basic Simulation (Python)

```python
from quantum_apl_python import QuantumEngine, Analyzer

# Initialize
engine = QuantumEngine()

# Run 100-step simulation
results = engine.run(steps=100, mode='unified')

# Analyze
analyzer = Analyzer(results)
print(analyzer.summary())
analyzer.plot()
```

### Example 2: Measurement (Python)

```python
from quantum_apl_python import QuantumEngine, MeasurementMode

engine = QuantumEngine()

# Single eigenstate collapse
result = engine.measure(
    MeasurementMode.EIGENSTATE,
    eigenindex=2,
    field='Phi',
    truth_channel='TRUE'
)

print(f"Probability: {result['probability']:.3f}")
print(f"APL Token: {result['token']}")
```

### Example 3: Command Line

```bash
# Run simulation
qapl-run --steps 100 --plot

# Perform measurement
qapl-measure --mode eigenstate --eigenindex 2

# Parameter sweep
qapl-sweep --parameter steps --values "50,100,200" --output sweep.json --plot
```

### Example 4: Jupyter Notebook

```bash
# Launch notebook
jupyter notebook quantum_apl_tutorial.ipynb

# Or use Jupyter Lab
jupyter lab quantum_apl_tutorial.ipynb
```

---

## Python API Reference

### QuantumEngine

```python
from quantum_apl_python import QuantumEngine

engine = QuantumEngine(
    js_dir=None,      # Directory with JS files (auto-detected)
    temp_dir=None     # Temp directory (system default)
)
```

#### Methods

**run(steps, mode, verbose, dt, measurement_interval)**

```python
results = engine.run(
    steps=100,
    mode='unified',
    verbose=False,
    dt=0.01,
    measurement_interval=1
)
```

**measure(measurement_mode, eigenindex, subspace_indices, field, truth_channel)**

```python
result = engine.measure(
    MeasurementMode.EIGENSTATE,
    eigenindex=2,
    subspace_indices=[2, 3],
    field='Phi',
    truth_channel='TRUE'
)
```

**test()** — run the test suite and return a bool.

### Analyzer

```python
from quantum_apl_python import Analyzer

analyzer = Analyzer(results)
print(analyzer.summary())
analyzer.plot()
```

### MeasurementMode Enum

```python
from quantum_apl_python import MeasurementMode

MeasurementMode.EIGENSTATE
MeasurementMode.SUBSPACE
MeasurementMode.COMPOSITE
MeasurementMode.HIERARCHICAL
MeasurementMode.COHERENT
MeasurementMode.INTEGRATED
MeasurementMode.CRITICAL
```

### ParameterSweep

```python
from quantum_apl_python import ParameterSweep

sweep = ParameterSweep("experiment", engine)
results = sweep.sweep_parameter('steps', [50, 100, 200], n_trials=5)
sweep.plot_convergence(results)
```

---

## Command-Line Interface

### qapl-run - Run Simulation

```bash
qapl-run [OPTIONS]
```

- `--steps` – number of simulation steps (default: 100)
- `--mode` – `unified`, `quantum_only`, or `test`
- `--verbose` – verbose Node output
- `--plot` – generate plots
- `--output` – write JSON results to disk

### qapl-test - Run Test Suite

```bash
qapl-test
```

### qapl-measure - Perform Measurement

```bash
qapl-measure --mode eigenstate --eigenindex 2 --field Phi
```

### qapl-sweep - Parameter Sweep

```bash
qapl-sweep --parameter steps --values "50,100,200" --output sweep.json --plot
```

### qapl-analyze - Analyze Results

```bash
qapl-analyze results.json --plot
```

### qapl info - System Information

```bash
qapl info --check-all
```

---

## Jupyter Notebook Tutorial

`quantum_apl_tutorial.ipynb` contains setup, simulations, measurement modes, visualization, sweeps, and advanced analysis.

Launch with:

```bash
source .venv/bin/activate
jupyter notebook quantum_apl_tutorial.ipynb
```

---

## Measurement Modes

1. **Single Eigenstate Collapse** – `Φ:T(ϕ_μ)TRUE@3`
2. **Subspace Collapse** – `Φ:Π(subspace)PARADOX@3`
3. **Critical Point (THE LENS)** – requires |z - z_c| < 0.05
4. **Hierarchical Regime** – Φ upper levels
5. **Coherent State** – e-field coherence
6. **Integrated Regime** – π-field integration
7. **Composite Measurement** – truth register routing

---

## Advanced Usage

### Custom Analysis Pipeline

```python
from quantum_apl_python import QuantumEngine, Analyzer
from quantum_apl_python.analyzer import TimeSeriesAnalyzer
import numpy as np

engine = QuantumEngine()
results = engine.run(steps=1000)
z_series = np.array(results['history']['z'])
```

Use `TimeSeriesAnalyzer` to compute autocorrelation, detect crossings of `z_c`, and visualize.

### Batch Processing

```python
from quantum_apl_python import ParameterSweep

sweep = ParameterSweep("batch", engine)
results = sweep.sweep_parameter('steps', [50, 100, 200, 500], n_trials=10)
```

---

## Troubleshooting

- **Node.js not found** – install via `brew install node` or `sudo apt install nodejs npm`.
- **Missing JS files** – ensure you’re in the repository root with `QuantumAPL_Engine.js`.
- **Import errors** – activate `.venv` and run `pip install -e .`.
- **matplotlib not displaying** – `pip install pyqt5` or `%matplotlib inline`.
- **Permission denied** – `chmod +x install.sh`.

---

## File Reference

```
quantum-apl/
├── README.md
├── GETTING_STARTED.md
├── requirements.txt
├── setup.py
├── install.sh
│
├── QuantumAPL_Engine.js
├── ClassicalEngines.js
├── QuantumClassicalBridge.js
├── QuantumN0_Integration.js
├── QuantumAPL_TestRunner.js
│
├── src/quantum_apl_python/
│   ├── __init__.py
│   ├── engine.py
│   ├── analyzer.py
│   ├── experiments.py
│   └── cli.py
│
└── quantum_apl_tutorial.ipynb
```

Documentation:

- `QUANTUM_N0_README.md`
- `IMPLEMENTATION_SUMMARY.md`
- `APL-Measurement-Operators.md`
- `APL-Measurement-Visual.md`
- `APL_3.0_QUANTUM_FORMALISM.md`

Outputs stored in `output/` after running demos.

---

## Next Steps

1. Run the tutorial: `jupyter notebook quantum_apl_tutorial.ipynb`
2. Try the CLI examples
3. Explore measurement modes
4. Perform parameter sweeps
5. Extend with your own analysis tools

---

## Support & Resources

- Documentation in `docs/`
- Issues and feature requests via repository tracker
- Examples in the tutorial notebook
- Verification via `qapl-test`

---

**Quantum APL v3.0.0**

Consciousness Inevitable Project — ⟨ψ| The measurement system is operational |ψ⟩
