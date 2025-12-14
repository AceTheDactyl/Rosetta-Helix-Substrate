# Nuclear Spinner × Rosetta-Helix Integration Guide

**Signature:** `integration-guide|v1.0.0|helix`

This document provides comprehensive guidance for integrating and operating the Nuclear Spinner hardware with the Rosetta-Helix software framework as a unified monorepo.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Components](#components)
5. [Communication Protocol](#communication-protocol)
6. [Deployment](#deployment)
7. [Experiments](#experiments)
8. [Troubleshooting](#troubleshooting)
9. [Physics Reference](#physics-reference)

---

## System Overview

The integrated system couples:
- **Nuclear Spinner**: Physical hardware (STM32H743ZI MCU controlling rotor, RF pulses, and sensors)
- **Rosetta-Helix**: Software framework (Kuramoto oscillators, GHMP processing, TRIAD dynamics)

The coupling mechanism: **z-coordinate** from the spinner drives the Kuramoto coupling strength, which affects coherence and triggers K-formation events.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     INTEGRATED SYSTEM ARCHITECTURE                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   HARDWARE (Firmware)              BRIDGE              SOFTWARE          │
│   ┌─────────────────┐         ┌───────────┐       ┌─────────────────┐   │
│   │ Nuclear Spinner │ Serial  │  spinner  │  WS   │  Rosetta-Helix  │   │
│   │   STM32H743ZI   │───────► │  _bridge  │───────│     Node        │   │
│   │   0-10k RPM     │ JSON    │   .py     │ JSON  │                 │   │
│   └─────────────────┘         └───────────┘       └─────────────────┘   │
│          ▼                                               ▼               │
│   z-coordinate ─────────────────────────────────► Kuramoto Coupling     │
│                                                          ▼               │
│                                                    K-Formation           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### Directory Structure

```
Rosetta-Nuclear-Spinner/
├── rosetta-helix/           # Core Rosetta-Helix framework
│   └── src/
│       ├── heart.py         # Kuramoto 60-oscillator system
│       ├── brain.py         # GHMP tier-gated processing
│       ├── triad.py         # Triadic threshold dynamics
│       ├── spinner_client.py # WebSocket client for bridge
│       ├── node.py          # Main integrated node
│       └── physics.py       # Shared physics constants
│
├── bridge/                  # Hardware-software bridge
│   └── spinner_bridge.py    # WebSocket server + serial relay
│
├── nuclear_spinner_firmware/ # Embedded firmware
│   ├── src/                 # C source files
│   │   ├── main.c           # Main entry point
│   │   ├── rotor_control.c  # Motor control
│   │   ├── pulse_control.c  # RF pulse generation
│   │   └── comm_protocol.c  # JSON serial protocol
│   ├── include/             # Header files
│   │   └── physics_constants.h
│   └── training_modules.h/c # 19 training modules
│
├── training/                # Training system
│   └── src/
│       └── unified_nightly_workflow.py
│
├── src/                     # JavaScript APL engine
│   ├── constants.js         # Physics constants
│   ├── quantum_apl_engine.js
│   └── quantum_apl_python/  # Python APL package
│
├── scripts/                 # Operational scripts
│   ├── start_system.sh      # Full stack startup
│   ├── analyze_session.py   # Session analysis
│   └── run_experiment.py    # Experiment orchestration
│
├── tests/                   # Test suite
│   ├── test_*.py            # Python tests
│   └── test_*.js            # JavaScript tests
│
├── docs/                    # Documentation
│   ├── INTEGRATION_GUIDE.md # This file
│   ├── PHYSICS_GROUNDING.md
│   └── ...
│
└── .github/workflows/       # CI/CD
    └── unified-nightly-training.yml
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.11+
python3 --version

# Install dependencies
pip install websockets pyserial numpy

# For firmware (optional)
arm-none-eabi-gcc --version
```

### Start the System

```bash
# Start full integrated system
./scripts/start_system.sh

# Or in simulation mode (no hardware)
./scripts/start_system.sh --simulate

# Bridge only (for debugging)
./scripts/start_system.sh --bridge-only
```

### Run an Experiment

```bash
# K-formation experiment
python scripts/run_experiment.py k_formation --steps 5000

# Z-sweep with output
python scripts/run_experiment.py z_sweep --output results/

# Attractor convergence
python scripts/run_experiment.py attractor --tolerance 0.001
```

### Analyze Results

```bash
python scripts/analyze_session.py results/k_formation_*/
```

---

## Components

### Rosetta-Helix Core (rosetta-helix/src/)

| Module | Purpose |
|--------|---------|
| `heart.py` | 60 Kuramoto oscillators with hexagonal geometry |
| `brain.py` | GHMP tier-gated operator processing |
| `triad.py` | Triadic threshold (κ, λ, η) dynamics |
| `spinner_client.py` | WebSocket client for hardware bridge |
| `node.py` | Main orchestrator integrating all modules |
| `physics.py` | Shared physics constants and utilities |

### Bridge Service (bridge/)

| Module | Purpose |
|--------|---------|
| `spinner_bridge.py` | WebSocket server + serial relay |

The bridge:
- Connects to firmware via serial (115200 baud)
- Serves WebSocket on port 8765
- Broadcasts spinner state at 100 Hz
- Relays commands from software to hardware
- Falls back to simulation if no hardware

### Firmware (nuclear_spinner_firmware/)

| File | Purpose |
|------|---------|
| `main.c` | Entry point, main control loop |
| `rotor_control.c` | Motor speed control (0-10k RPM) |
| `pulse_control.c` | RF pulse generation |
| `threshold_logic.c` | z threshold detection |
| `comm_protocol.c` | JSON serial protocol |
| `physics_constants.h` | Physics constants |
| `training_modules.h/c` | 19 training modules |

---

## Communication Protocol

### Serial (Firmware ↔ Bridge)

**Format:** JSON at 115200 baud

**Firmware → Bridge (State at 100 Hz):**
```json
{
  "type": "state",
  "timestamp_ms": 123456,
  "z": 0.8660,
  "rpm": 8600,
  "delta_s_neg": 0.9999,
  "tier": 5,
  "tier_name": "UNIVERSAL",
  "phase": "THE_LENS",
  "kappa": 0.618,
  "eta": 0.85,
  "rank": 12,
  "k_formation": true,
  "k_formation_duration_ms": 1500
}
```

**Bridge → Firmware (Commands):**
```json
{"cmd": "set_z", "value": 0.866}
{"cmd": "set_rpm", "value": 8660}
{"cmd": "stop"}
{"cmd": "hex_cycle", "dwell_s": 30, "cycles": 10}
```

### WebSocket (Bridge ↔ Rosetta-Helix)

**Format:** Same JSON as serial

**Bridge:** `ws://localhost:8765`

---

## Deployment

### 1. Firmware Flash (if hardware present)

```bash
cd nuclear_spinner_firmware
make clean && make
make flash  # Requires ST-Link
```

### 2. Start Bridge

```bash
python -m bridge.spinner_bridge [--port /dev/ttyACM0] [--simulate]
```

### 3. Start Rosetta-Helix Node

```bash
python -m rosetta_helix.src.node
```

### 4. Full Stack (Recommended)

```bash
./scripts/start_system.sh
```

---

## Experiments

### Available Experiments

| Experiment | Description |
|------------|-------------|
| `z_sweep` | Sweep z from 0.3 → z_c → 0.95 |
| `k_formation` | Achieve and sustain K-formation |
| `hex_cycle` | Hexagonal cycling (60 positions) |
| `attractor` | Verify κ → φ⁻¹ convergence |
| `tier_climb` | Progressive tier ascent |
| `phase_map` | Map phase transitions |
| `stress` | Rapid z oscillations |

### Running Experiments

```bash
# Basic usage
python scripts/run_experiment.py k_formation

# With options
python scripts/run_experiment.py z_sweep \
  --steps 10000 \
  --output results/ \
  --target-z 0.866
```

### Analysis

```bash
python scripts/analyze_session.py results/experiment_dir/

# Output options
python scripts/analyze_session.py results/ --output analysis.json
```

---

## Troubleshooting

### Bridge Won't Connect

1. Check serial port: `ls /dev/ttyACM*`
2. Check permissions: `sudo usermod -a -G dialout $USER`
3. Run in simulation: `--simulate`

### WebSocket Connection Failed

1. Check bridge is running: `ps aux | grep spinner_bridge`
2. Check port: `netstat -tlnp | grep 8765`
3. Check firewall: `sudo ufw allow 8765`

### No K-Formations

1. Verify z is at z_c: `z ≈ 0.866`
2. Check κ threshold: `κ ≥ 0.92`
3. Increase simulation time

### Physics Validation Errors

1. Check conservation: `κ + λ = 1`
2. Check bounds: `0 ≤ z ≤ 1`, `0 ≤ κ ≤ 1`
3. Review physics constants match across languages

---

## Physics Reference

### Fundamental Constants

| Constant | Symbol | Value | Definition |
|----------|--------|-------|------------|
| Golden ratio | φ | 1.618034 | (1 + √5) / 2 |
| Golden ratio inverse | φ⁻¹ | 0.618034 | 1 / φ |
| Critical z | z_c | 0.866025 | √3 / 2 |
| Gaussian width | σ | 36 | |S₃|² |

### Negentropy Signal

```
ΔS_neg(z) = exp(-σ(z - z_c)²)
```

Peaks at z = z_c with value 1.0.

### K-Formation Criteria

```
K-formation iff:
  κ ≥ 0.92
  η > φ⁻¹ (≈ 0.618)
  R ≥ 7
```

### Phases

| Phase | z Range | Description |
|-------|---------|-------------|
| ABSENCE | z < 0.857 | Pre-critical |
| THE_LENS | 0.857 ≤ z ≤ 0.877 | Critical window |
| PRESENCE | z > 0.877 | Post-critical |

### Tiers

| Tier | z Threshold | Description |
|------|-------------|-------------|
| ABSENCE | < 0.40 | Pre-operational |
| REACTIVE | 0.40 | Basic reactions |
| MEMORY | 0.50 | State retention |
| PATTERN | φ⁻¹ (0.618) | Pattern recognition |
| PREDICTION | 0.73 | Predictive modeling |
| UNIVERSAL | z_c (0.866) | Full universality |
| META | 0.92 | Meta-cognitive recursion |

---

## Related Documentation

- [PHYSICS_GROUNDING.md](PHYSICS_GROUNDING.md) - Physics foundations
- [Z_CRITICAL_LENS.md](Z_CRITICAL_LENS.md) - The z_c critical point
- [HEXAGONAL_NEG_ENTROPY_PROJECTION.md](HEXAGONAL_NEG_ENTROPY_PROJECTION.md) - Geometry
- [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) - Architecture details

---

**Δ|integration-guide|z_c=0.866|Ω**
