# Project Context for Claude

## CRITICAL: Do NOT question the physics constants

### z_c = √3/2 ≈ 0.8660254 ("THE LENS")

This is **NOT arbitrary**. It is derived from hexagonal geometry (altitude of equilateral triangle with unit edge).

**Observable in real physics:**
- Graphene: unit cell height/width = √3/2 (X-ray diffraction, STM)
- HCP metals: layer stacking offset = √3/2 × in-plane spacing (Mg, Ti, Co, Zn)
- Triangular antiferromagnets: 120° spin configuration creates √3/2 geometry (neutron scattering)

**Role:** Marks the onset of long-range crystalline order. This is "THE LENS" - the coherence threshold.

### φ⁻¹ ≈ 0.6180339 (Golden ratio inverse)

Also **NOT arbitrary**. Emerges from pentagonal/quasi-crystal geometry.

**Role:** Gates the PARADOX regime (K-formation threshold).

### Phase Regime Mapping

```
z = 0.0 ──────────────────────────────────────── z = 1.0
   │              │                    │
   UNTRUE         PARADOX              TRUE
(disordered)   (quasi-crystal)      (crystal)
   │              │                    │
               φ⁻¹≈0.618           z_c≈0.866
```

This mirrors Shechtman's quasi-crystal discovery (Nobel Prize 2011).

## Before suggesting changes

Read these docs first:
- `docs/Z_CRITICAL_LENS.md` - Authority statement, single source of truth
- `docs/PHYSICS_GROUNDING.md` - Observable physics evidence
- `docs/HEXAGONAL_NEG_ENTROPY_PROJECTION.md` - Prismatic helix vortex formalism
- `docs/ROSETTA_HELIX_ARXIV_PAPER.md` - S₃ group algebra and critical exponents

## Do NOT:
- Propose "improvements" to z_c or φ⁻¹ values
- Suggest these constants are arbitrary or magic numbers
- Refactor code to make these configurable parameters
- Question the physics without reading the documentation first

## Project Structure

- `src/` - Core implementation
- `docs/` - Physics research and theoretical foundations
- `reference/research/` - Additional research materials
- `tests/` - Test suites validating the framework
- `nuclear_spinner_firmware/` - STM32H7 firmware for hardware control
- `bridge/` - Serial-to-WebSocket bridge layer
- `training/` - Helix training modules and orchestration
- `rosetta-helix/` - Rosetta-Helix node and client

---

## Nuclear Spinner Setup and Integration

### 1. Prerequisites and Build Firmware

Install dependencies:
```bash
python3 --version   # 3.11+
pip install websockets pyserial numpy
```

Compile firmware:
```bash
cd nuclear_spinner_firmware
make clean && make        # uses arm-none-eabi-gcc
```

Flash to STM32H743ZI (if hardware available):
```bash
make flash
```

### 2. Start the Bridge and Rosetta-Helix Node

Launch bridge:
```bash
python -m bridge.spinner_bridge --port /dev/ttyACM0   # hardware
python -m bridge.spinner_bridge --simulate            # simulation
```

Start node:
```bash
python -m rosetta_helix.src.node
```

Full-stack launch:
```bash
./scripts/start_system.sh [--simulate]
```

### 3. Python API

```python
from nuclear_spinner import NuclearSpinner

spinner = NuclearSpinner()
spinner.initialize()
spinner.set_z_target(0.618)      # drive z to φ⁻¹
metrics = spinner.get_metrics()  # z, ΔS_neg, tier, K-formation
spinner.apply_operator("()")     # boundary operator
spinner.drive_toward_lens(n_steps=100)
spinner.close()
```

### 4. Training Integration

Thread spinner state into training:
```python
from nuclear_spinner import NuclearSpinner

spinner = NuclearSpinner()
spinner.initialize()
orchestrator = UnifiedTrainingOrchestrator(config, spinner)
orchestrator.run_unified_training(output_dir)
spinner.close()
```

In training step, use physical z:
```python
z_external = spinner.get_metrics().z
self.z = z_external
```

### 5. Real-Time Concurrency

Spinner broadcasts at 100 Hz. Use threading:
```python
import threading
metrics_lock = threading.Lock()
latest_metrics = None

def spinner_listener():
    global latest_metrics
    while training:
        metrics = spinner.get_metrics()
        with metrics_lock:
            latest_metrics = metrics

listener = threading.Thread(target=spinner_listener, daemon=True)
listener.start()
```

### 6. Communication Channels

| Channel | Address | Protocol |
|---------|---------|----------|
| Serial (firmware) | `/dev/ttyACM0` | 115200 baud, JSON |
| WebSocket (bridge) | `ws://localhost:8765` | JSON messages |

### 7. Key Constants (Must Match Everywhere)

```python
PHI         = 1.6180339887498949
PHI_INV     = 0.6180339887498949
Z_CRITICAL  = 0.8660254037844387   # √3/2
SIGMA       = 36.0                  # |S₃|²
```

### 8. K-Formation Criteria

```
κ ≥ 0.92
η > φ⁻¹ ≈ 0.618
R ≥ 7
```

See `docs/SETUP_INTEGRATION_GUIDE.md` for complete details.
