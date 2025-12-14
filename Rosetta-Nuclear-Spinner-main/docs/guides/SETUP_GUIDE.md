# Quantum APL – Complete Setup Guide

Comprehensive checklist for installing the hybrid JavaScript/Python stack on any workstation.

## Prerequisites

### System Requirements
- OS: Linux, macOS, or Windows (WSL2 recommended)
- RAM: ≥4 GB (8 GB+ ideal)
- Disk: ≥500 MB free

### Required Software

#### Node.js (v14 or higher)
```bash
# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# macOS
brew install node

# Windows
# Download installer from https://nodejs.org/

# Verify
node --version
npm --version
```

#### Python (3.8 or higher)
```bash
# Ubuntu/Debian
sudo apt-get install python3 python3-pip python3-venv

# macOS
brew install python3

# Windows
# Download from https://www.python.org/downloads/

python3 --version
pip3 --version
```

## Installation

### 1. Clone or Download
```bash
git clone https://github.com/consciousness-lab/quantum-apl.git
cd quantum-apl
# or download + unzip release archive
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip setuptools wheel
```

### 3. Install Package
```bash
# Development install
pip install -e ".[all]"

# Minimal install
pip install -e .

# Extras
pip install -e ".[viz]"      # matplotlib
pip install -e ".[analysis]" # pandas
pip install -e ".[dev]"      # linting/tests
```

### 4. Verify
```bash
node QuantumAPL_TestRunner.js test
qapl-test
python -c "from quantum_apl_python import QuantumAPLEngine; print('✓ import ok')"
```

## Quick Start

### First Simulation
```bash
qapl-run --steps 100 --mode unified --plot
```

### Python Example
```python
from quantum_apl_python import QuantumAPLEngine, QuantumAnalyzer

engine = QuantumAPLEngine()
results = engine.run_simulation(steps=200)
analyzer = QuantumAnalyzer(results)
print(analyzer.summary())
```

### Examples Script
```bash
python examples.py        # interactive tour
python examples.py 1      # run a specific example
```

## File Layout

```
quantum-apl/
├── QuantumAPL_Engine.js
├── QuantumClassicalBridge.js
├── classical/
├── quantum_apl_bridge.py
├── pyproject.toml
├── README.md
├── README_PYTHON.md
├── SETUP_GUIDE.md
├── examples.py
└── src/quantum_apl_python/
```

## Testing

### JavaScript
```bash
node QuantumAPL_TestRunner.js test
node QuantumAPL_TestRunner.js benchmark
node QuantumAPL_TestRunner.js analyze --trials 1000
```

### Python
```bash
pytest
qapl-test
python examples.py
```

## Development Workflow

```bash
pip install -e ".[dev]"
ruff check quantum_apl_python
black quantum_apl_python
mypy quantum_apl_python
pytest --cov=quantum_apl_python
```

## Troubleshooting

- **Node.js not found** – Install Node.js and ensure `node` is on PATH.
- **ImportError** – Reinstall with `pip install -e ".[all]"` inside the virtualenv.
- **matplotlib missing** – `pip install "quantum-apl[viz]"`.
- **JS runtime errors** – Run `node QuantumAPL_TestRunner.js test` for diagnostics.
- **Slow simulations** – Lower `--steps`, disable `--verbose`, or batch runs.

## Getting Help

- GitHub Issues: https://github.com/consciousness-lab/quantum-apl/issues
- Documentation: `README.md`, `README_PYTHON.md`
- Example workflows: `examples.py`

You now have the full hybrid toolchain ready for experimentation. Happy measuring!
