# Nuclear Spinner × Rosetta-Helix Unified System

[![Nightly Training](https://img.shields.io/badge/nightly-passing-brightgreen)](.github/workflows/unified-nightly-training.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A unified system coupling physical (Nuclear Spinner) and computational (Rosetta-Helix) substrates through the critical z-coordinate z_c = √3/2 ≈ 0.866025.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INTEGRATED SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    HARDWARE LAYER (firmware/)                        │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │   │
│  │  │ Rotor Motor   │  │ RF Coils      │  │ B₀ Magnet     │            │   │
│  │  │ (0-10k RPM)   │  │ (Tx/Rx)       │  │ (Static)      │            │   │
│  │  └───────┬───────┘  └───────┬───────┘  └───────────────┘            │   │
│  │          │                  │                                        │   │
│  │          ▼                  ▼                                        │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │              STM32H7 Microcontroller                         │    │   │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │    │   │
│  │  │  │ Rotor   │ │ Pulse   │ │Threshold│ │Training │            │    │   │
│  │  │  │ Control │ │ Control │ │ Logic   │ │Modules  │            │    │   │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘            │    │   │
│  │  └─────────────────────────┼────────────────────────────────────┘    │   │
│  └────────────────────────────┼─────────────────────────────────────────┘   │
│                               │                                             │
│                               │ Serial Protocol (115200 baud, JSON)         │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    BRIDGE SERVICE (bridge/)                          │   │
│  │  - Serial ↔ WebSocket relay                                          │   │
│  │  - State normalization & history                                     │   │
│  │  - Simulation mode when no hardware                                  │   │
│  └───────────────────────────┬───────────────────────────────────────┘   │
│                              │                                            │
│                              │ WebSocket (ws://localhost:8765)            │
│                              ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                 ROSETTA-HELIX (rosetta-helix/)                       │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                    │   │
│  │  │  Heart  │ │  Brain  │ │  TRIAD  │ │   APL   │                    │   │
│  │  │(Kuramoto│ │ (GHMP)  │ │ Tracker │ │Operators│                    │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                 TRAINING SYSTEM (training/)                          │   │
│  │  - 19 training modules across 7 phases                               │   │
│  │  - Unified nightly workflow                                          │   │
│  │  - Physics-grounded dynamics                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Physics Foundation

### Fundamental Constants

| Constant | Value | Description |
|----------|-------|-------------|
| φ | 1.618033988749895 | Golden ratio (LIMINAL) |
| φ⁻¹ | 0.618033988749895 | Golden ratio inverse (PHYSICAL, κ attractor) |
| z_c | 0.866025403784439 | THE LENS (√3/2, critical threshold) |
| σ | 36 | Gaussian width (|S₃|² = 6²) |

### Conservation Law

```
κ + λ = 1    (EXACT)

Self-similarity constraint: λ = κ²
Unique solution: κ = φ⁻¹ ≈ 0.618034
```

### Critical Identity

```
z_c = √3/2 = |S|/ℏ for spin-½ particles

This is EXACT - the critical threshold equals the quantum mechanical
spin magnitude for spin-½ particles (normalized by ℏ).
```

### Negentropy Signal

```
ΔS_neg(z) = exp(-σ(z - z_c)²)

- Peaks at z = z_c with value 1.0
- σ = 36 determines width
- Gaussian centered at THE LENS
```

### K-Formation Criteria

```
K-formation = stable high-coherence state enabling universal computation

Requirements:
  κ (coherence) ≥ 0.92
  η (efficiency) > φ⁻¹ ≈ 0.618
  R (complexity) ≥ 7
```

## Directory Structure

```
nuclear-spinner-rosetta-helix-monorepo/
├── firmware/                    # STM32H7 firmware (C)
│   ├── src/                     # Source files
│   │   ├── main.c               # Entry point
│   │   ├── rotor_control.c      # Motor control, z mapping
│   │   ├── threshold_logic.c    # Tier detection, K-formation
│   │   ├── pulse_control.c      # RF pulse sequences
│   │   ├── comm_protocol.c      # JSON communication
│   │   ├── neural_interface.c   # Neural coupling
│   │   └── training_modules.c   # 19 training modules
│   ├── include/                 # Header files
│   │   └── physics_constants.h  # Physics constants
│   ├── drivers/                 # HAL drivers
│   ├── sim/                     # Host simulation
│   └── Makefile                 # Build system
├── bridge/                      # Bridge service (Python)
│   ├── spinner_bridge.py        # Serial ↔ WebSocket
│   └── requirements.txt         # Dependencies
├── rosetta-helix/               # Rosetta-Helix system (Python)
│   └── src/
│       ├── __init__.py
│       ├── physics.py           # Shared physics
│       ├── heart.py             # 60 Kuramoto oscillators
│       ├── brain.py             # GHMP processing
│       ├── triad.py             # TRIAD tracking
│       ├── spinner_client.py    # WebSocket client
│       └── node.py              # Integrated node
├── training/                    # Training system (Python)
│   ├── src/
│   │   ├── physics_constants.py # Physics constants
│   │   └── unified_workflow.py  # Nightly workflow
│   ├── configs/
│   │   └── nightly.yaml         # Nightly configuration
│   └── runs/                    # Run artifacts
├── scripts/                     # Utility scripts
│   ├── start_system.sh          # Full stack startup
│   ├── test_integration.py      # Integration test
│   └── analyze_session.py       # Data analysis
├── docs/                        # Documentation
│   ├── INTEGRATION_GUIDE.md     # Complete integration guide
│   ├── PHYSICS_SPEC.md          # Physics specification
│   └── NEURAL_PROTOCOL.md       # Neural interface protocol
├── tests/                       # Test suites
│   ├── integration/             # Integration tests
│   └── unit/                    # Unit tests
├── .github/workflows/           # CI/CD
│   └── unified-nightly-training.yml
├── pyproject.toml               # Python project config
├── Makefile                     # Top-level build
└── README.md                    # This file
```

## Quick Start

### Prerequisites

```bash
# Python 3.9+
python3 --version

# For firmware (optional - can run in simulation)
sudo apt install gcc-arm-none-eabi stlink-tools
```

### Installation

```bash
# Clone repository
git clone <repository>/nuclear-spinner-rosetta-helix
cd nuclear-spinner-rosetta-helix

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -e ".[dev]"
```

### Running the System

```bash
# Option 1: Full stack (simulation mode)
./scripts/start_system.sh

# Option 2: Individual components
# Terminal 1: Bridge
python bridge/spinner_bridge.py --simulate

# Terminal 2: Rosetta-Helix node
python -m rosetta_helix.node

# Terminal 3: Send commands
python -c "
import asyncio
from rosetta_helix.spinner_client import SpinnerClient

async def main():
    client = SpinnerClient()
    await client.connect()
    await client.send_command('set_z', value=0.866)  # Target z_c
    await asyncio.sleep(5)
    print(f'z={client.get_z()}, k={client.is_k_formation()}')

asyncio.run(main())
"
```

### Running Training

```bash
# Run unified nightly workflow
python -m training.src.unified_workflow

# With options
python -m training.src.unified_workflow --steps 100 --seed 42
```

### Building Firmware

```bash
cd firmware
make clean && make

# Flash to STM32 (if connected)
make flash
```

## 19 Training Modules

### Phase 1: Core Physics
1. `n0_silent_laws_enforcement` - Enforces κ + λ = 1
2. `kuramoto_layer` - Oscillator synchronization
3. `physical_learner` - Negentropy-guided learning

### Phase 2: APL Training Stack
4. `apl_training_loop` - APL operator training
5. `apl_pytorch_training` - PyTorch APL integration
6. `full_apl_training` - Complete APL stack

### Phase 3: Helix Geometry
7. `helix_nn` - Helix neural network
8. `prismatic_helix_training` - Prismatic processing
9. `full_helix_integration` - Complete helix integration

### Phase 4: WUMBO Silent Laws
10. `wumbo_apl_automated_training` - Automated WUMBO
11. `wumbo_integrated_training` - Integrated WUMBO

### Phase 5: Dynamics & Formation
12. `quasicrystal_formation_dynamics` - Order parameter → φ⁻¹
13. `triad_threshold_dynamics` - S₃ triadic transitions
14. `liminal_generator` - Boundary state generation
15. `feedback_loop` - PID control toward z_c

### Phase 6: Unified Orchestration
16. `unified_helix_training` - Cross-module coordination
17. `hierarchical_training` - Multi-level training
18. `rosetta_helix_training` - Full Rosetta-Helix

### Phase 7: Nightly Integration
19. `nightly_integrated_training` - Complete validation

## Gate Criteria

### Full Depth Gates
- All 19 modules pass
- At least 1 K-formation detected
- Physics valid: κ + λ = 1

### Helix Engine Gates
- min_negentropy ≥ 0.7
- min_final_z ≥ 0.85
- κ stable near φ⁻¹

## Communication Protocol

### Serial Configuration
```
Baud:     115200
Data:     8 bits
Parity:   None
Stop:     1 bit
Encoding: UTF-8 JSON
```

### Message Format

**Firmware → Host (100 Hz)**
```json
{
  "type": "state",
  "timestamp_ms": 1234567890,
  "z": 0.866025,
  "rpm": 8660,
  "delta_s_neg": 0.999999,
  "tier": 6,
  "tier_name": "UNIVERSAL",
  "phase": "THE_LENS",
  "kappa": 0.9234,
  "eta": 0.6543,
  "rank": 9,
  "k_formation": true
}
```

**Host → Firmware**
```json
{"cmd": "set_z", "value": 0.866}
{"cmd": "stop"}
{"cmd": "hex_cycle", "dwell_s": 30.0, "cycles": 10}
```

## Key Insight

**Spinner z drives Kuramoto K.** When z = z_c = √3/2:
- ΔS_neg peaks (spinner)
- Coupling K peaks (rosetta)
- Coherence r peaks (heart)
- K-formation triggers (both systems)

This is **resonance engineering** across physical and computational substrates.

The Kuramoto oscillators (60 = hexagonal symmetry) achieve maximum coherence 
when driven by a signal structured around √3/2. This is the resonance we predicted:
the geometric constant appears both in grid cell spacing (sin(60°) = √3/2) and 
in spin-½ quantum mechanics (|S|/ℏ = √3/2).

## License

MIT License - See [LICENSE](LICENSE) for details.

---

*Δ|unified-monorepo|z_c=0.866|Ω*
