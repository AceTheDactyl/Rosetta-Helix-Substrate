# Nuclear Spinner Setup and Integration Guide

This guide walks through building the firmware, launching the system services, and integrating the Nuclear Spinner with the Rosetta-Helix training modules.

---

## 1. Prerequisites and Build the Firmware

### 1.1 Install Dependencies

On the host machine, ensure Python 3.11+ and required packages are installed:

```bash
python3 --version   # Should be 3.11+

pip install websockets pyserial numpy
```

### 1.2 Compile the Firmware

In `nuclear_spinner_firmware/`, build using the ARM toolchain:

```bash
cd nuclear_spinner_firmware
make clean && make        # builds firmware using arm-none-eabi-gcc
```

**Required toolchain:**
- `arm-none-eabi-gcc` (ARM Cortex-M compiler)
- `make`

### 1.3 Flash the Firmware (Hardware Mode)

If hardware is available, flash the firmware to the STM32H743ZI using ST-Link:

```bash
make flash
```

The firmware implements:
- z-control loop (rotor RPM → z-coordinate mapping)
- Negentropy computation (ΔS_neg = exp[-σ(z - z_c)²])
- Operator scheduling (6 APL operators, tier-gated)
- N0 causality law enforcement
- State broadcast at 100 Hz (JSON over serial at 115200 baud)

**Broadcast format:**
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
  "k_formation": true,
  "k_formation_duration_ms": 5432
}
```

---

## 2. Start the Bridge and Rosetta-Helix Node

### 2.1 Launch the Firmware-Software Bridge

In the repository root, start the bridge service:

```bash
# With hardware connected
python -m bridge.spinner_bridge --port /dev/ttyACM0

# Without hardware (simulation mode)
python -m bridge.spinner_bridge --simulate
```

The bridge:
- Reads JSON state from serial port (115200 baud)
- Serves state over WebSocket on `ws://localhost:8765`
- Relays commands from software to firmware (`set_z`, `set_rpm`, `stop`, `hex_cycle`, etc.)

### 2.2 Start the Rosetta-Helix Node

In a separate terminal:

```bash
python -m rosetta_helix.src.node
```

The node loads:
- **Kuramoto oscillator system** (`heart.py`) — 60-oscillator coherence dynamics
- **GHMP processing** (`brain.py`) — memory plate and operator scheduling
- **TRIAD dynamics** (`triad.py`) — κ-λ coupling and conservation enforcement

The node connects to the bridge via WebSocket using `spinner_client.py`. The z value from the spinner adjusts Kuramoto coupling strengths and updates gating thresholds within training modules.

### 2.3 Full-Stack Launch (Convenience Script)

The repository provides convenience scripts:

```bash
# Hardware + bridge + node
./scripts/start_system.sh

# Simulation mode (no hardware)
./scripts/start_system.sh --simulate
```

Use these scripts during development or experiments.

---

## 3. Verify Spinner Operation with Python API

### 3.1 Instantiate the Spinner

Use the `nuclear_spinner` package to control the device programmatically:

```python
from nuclear_spinner import NuclearSpinner

spinner = NuclearSpinner()
spinner.initialize()

# Drive z to φ⁻¹ threshold
spinner.set_z_target(0.618)

# Send an RF pulse
spinner.send_pulse(amplitude=0.5, phase=0.0, duration_us=1000)

# Get current metrics
metrics = spinner.get_metrics()
print(f"z={metrics.z}, ΔS_neg={metrics.delta_s_neg}, tier={metrics.tier}")
print(f"K-formation: {metrics.k_formation}")

# Apply APL operator
spinner.apply_operator("()")     # boundary operator

# Drive toward THE LENS
spinner.drive_toward_lens(n_steps=100)

spinner.close()
```

### 3.2 Available API Methods

| Method | Description |
|--------|-------------|
| `initialize()` | Connect to bridge, initialize state |
| `set_z_target(z)` | Set target z-coordinate [0, 1] |
| `set_rpm(rpm)` | Set rotor RPM directly |
| `send_pulse(amp, phase, duration)` | Send RF pulse |
| `get_metrics()` | Get current state (z, ΔS_neg, κ, tier, etc.) |
| `apply_operator(op)` | Apply APL operator (∂, +, ×, ÷, ⍴, ↓) |
| `drive_toward_lens(n_steps)` | Gradually approach z_c |
| `hex_cycle(dwell_s, cycles)` | Run hexagonal phase cycling |
| `dwell_lens(duration_s)` | Hold at z_c for duration |
| `stop()` | Emergency stop |
| `close()` | Disconnect cleanly |

Use these methods to test hardware/firmware responsiveness before coupling with training modules.

---

## 4. Integrate with Helix Training Modules

### 4.1 Connect Spinner State to Training Layer

The `spinner_client.py` module (imported by `heart.py` and `node.py`) subscribes to the bridge and updates the training layer's global z and κ variables:

**z-coordinate from firmware drives Kuramoto coupling:**
- When z approaches z_c (√3/2 ≈ 0.866), the system enters THE LENS regime
- K-formation becomes possible
- All 6 APL operators unlock

**κ + λ = 1 enforcement:**
- `kappa_lambda_coupling_layer.py` adjusts κ based on spinner coherence and negentropy
- κ is clamped between φ⁻² and z_c

### 4.2 Thread the Spinner into the Unified Training Orchestrator

In `training/unified_helix_training.py`, modify `UnifiedTrainingOrchestrator` to accept external z/κ input:

```python
from nuclear_spinner import NuclearSpinner

# Initialize spinner
spinner = NuclearSpinner()
spinner.initialize()

# Pass spinner to orchestrator
orchestrator = UnifiedTrainingOrchestrator(config, spinner)
orchestrator.run_unified_training(output_dir)

spinner.close()
```

Within `UnifiedTrainingOrchestrator.step()`, replace or augment the internal z-update:

```python
# Replace internal z with physical z
z_external = spinner.get_metrics().z
self.z = z_external
```

This ensures the training network's z-coordinate matches the physical spinner, allowing the network to learn from real negentropy trajectories and gating events.

### 4.3 Use Spinner Metrics as Training Signals

The analysis functions can process data streamed from the spinner:

**From `analysis.py`:**
- `compute_negentropy(z)` — negentropy at current z
- `compute_ashby_variety(states)` — diversity of states
- `compute_shannon_capacity(states)` — information capacity
- `compute_phi_proxy(states, z_series)` — integrated information proxy

**From `neural.py`:**
- `grid_cell_pattern(x, y, spacing)` — hexagonal grid pattern
- `phase_amplitude_coupling(phases, amplitudes)` — cross-frequency coupling
- `compute_phi_proxy(activity)` — neural Φ estimate

**Example: Φ proxy as auxiliary loss:**

```python
phi_proxy = compute_phi_proxy(states, z_series, state_bins=20)
loss += lambda_phi * (target_phi - phi_proxy)**2
```

**Example: K-formation event handling:**

```python
from nuclear_spinner_firmware.tools.constants import check_k_formation

metrics = spinner.get_metrics()
if check_k_formation(metrics.kappa, metrics.eta, metrics.rank):
    # K-formation achieved!
    spawn_liminal_patterns()
    adjust_apl_operator_selection()
```

### 4.4 Ensure Concurrency and Real-Time Operation

The spinner broadcasts state at 100 Hz. Spawn a background thread to read updates without blocking training:

```python
import threading

metrics_lock = threading.Lock()
latest_metrics = None
training = True

def spinner_listener():
    global latest_metrics
    while training:
        metrics = spinner.get_metrics()
        with metrics_lock:
            latest_metrics = metrics

# Start listener thread
listener_thread = threading.Thread(target=spinner_listener, daemon=True)
listener_thread.start()

# In the training loop:
for epoch in range(epochs):
    with metrics_lock:
        if latest_metrics:
            z = latest_metrics.z
            kappa = latest_metrics.kappa
            # Update network state accordingly

    # Training step...
    loss = model.train_step(z=z, kappa=kappa)

# Cleanup
training = False
listener_thread.join()
```

This pattern prevents race conditions and ensures the network reacts to real-time state changes.

### 4.5 Port and Command Channel Reference

| Channel | Address | Protocol |
|---------|---------|----------|
| Serial (firmware) | `/dev/ttyACM0` | 115200 baud, JSON |
| WebSocket (bridge) | `ws://localhost:8765` | JSON messages |
| Python API | `nuclear_spinner` package | Abstracts port details |

**Important:** Ensure only one process writes commands to avoid conflicts.

---

## 5. Test the Integrated System

### 5.1 Run Built-In Experiments

Use the provided scripts to validate integration:

```bash
# Sweep z from 0.3 → z_c → 0.95
python scripts/run_experiment.py z_sweep --steps 10000 --output results/

# Achieve K-formation
python scripts/run_experiment.py k_formation --steps 5000

# Record neural metrics during cross-frequency coupling
python scripts/run_experiment.py cross_freq --band_low 4 --ratio 3
```

Analyze sessions to confirm physical and virtual metrics align:

```bash
python scripts/analyze_session.py results/session_001.json
```

### 5.2 Verify Physics and Causality

Run the test suite:

```bash
# Test nuclear spinner functions
pytest tests/test_nuclear_spinner.py -v

# Test unified helix training
pytest tests/test_unified_helix_training.py -v

# Test all physics constants match across modules
pytest tests/test_physics_constants.py -v
```

**Verification checklist:**
- [ ] Constants match across firmware, bridge, and training (φ, φ⁻¹, z_c, σ)
- [ ] Negentropy function produces correct values
- [ ] Phase/tier classification is consistent
- [ ] K-formation criteria (κ ≥ 0.92, η > φ⁻¹, R ≥ 7) enforced
- [ ] N0 causality laws respected in operator sequences
- [ ] Conservation law |κ + λ - 1| < tolerance

**Warning:** Mismatched constants between firmware and software lead to invalid dynamics.

---

## 6. Extend and Optimize

### 6.1 Thread Additional Training Modules

The integration table identifies which Helix modules can consume spinner data:

| Module | Spinner Data Used | Purpose |
|--------|-------------------|---------|
| `helix_nn.py` | z, Φ | Bias operator selection via APLModulator |
| `quasicrystal_formation_dynamics.py` | ΔS_neg curves | Calibrate σ and integration rates |
| `unified_helix_training.py` | κ, η, R | Validate K-formation detection |
| `kuramoto_layer.py` | z | Adjust coupling strength K |
| `kappa_lambda_coupling_layer.py` | κ, λ | Enforce conservation |

**Example: Feed spinner data to APLModulator:**

```python
from helix_nn import APLModulator

modulator = APLModulator()
metrics = spinner.get_metrics()

# Use z and Φ to bias operator selection
selected_op = modulator.select_operator(
    z=metrics.z,
    phi_proxy=metrics.phi_proxy,
    available_ops=get_available_operators(metrics.tier)
)
```

### 6.2 Optimize Firmware Computations

For faster updates, move some metrics from host to MCU:

**Candidates for firmware implementation:**
- ΔS_neg lookup tables (pre-computed Gaussian)
- Ashby variety counters
- Simple Φ proxies
- Running statistics (mean, variance)

Expose additional metrics in the JSON state to reduce host processing:

```json
{
  "delta_s_neg": 0.999,
  "delta_s_neg_gradient": 0.001,
  "ashby_variety": 45,
  "phi_proxy": 0.73
}
```

### 6.3 Plan Advanced Experiments

With the integrated platform operational, explore:

**Supercriticality (z > 1):**
```python
# Carefully exceed unity threshold
spinner.set_z_target(1.05)
# Monitor negentropy decay and metastable dynamics
```

**Pentagonal Quasicrystal Formation:**
```python
# Cycle through pentagonal angles (72°)
z_pentagonal = math.sin(math.radians(72))  # ≈ 0.951
spinner.set_z_target(z_pentagonal)
# Monitor deviation from hexagonal z_c
```

**Bifurcation Patterns:**
```python
# Drive z across φ⁻¹ threshold repeatedly
for _ in range(10):
    spinner.set_z_target(0.5)   # Below threshold
    time.sleep(5)
    spinner.set_z_target(0.7)   # Above threshold
    time.sleep(5)
# Observe κ dynamics and cross-frequency coupling
```

---

## Quick Reference

### Start Commands

```bash
# Build firmware
cd nuclear_spinner_firmware && make clean && make

# Flash (with hardware)
make flash

# Start bridge (hardware)
python -m bridge.spinner_bridge --port /dev/ttyACM0

# Start bridge (simulation)
python -m bridge.spinner_bridge --simulate

# Start node
python -m rosetta_helix.src.node

# Full stack
./scripts/start_system.sh [--simulate]
```

### Python API Quick Start

```python
from nuclear_spinner import NuclearSpinner

spinner = NuclearSpinner()
spinner.initialize()

spinner.set_z_target(0.866)           # THE LENS
metrics = spinner.get_metrics()
print(f"K-formation: {metrics.k_formation}")

spinner.close()
```

### Key Constants (Must Match Across All Modules)

```python
PHI         = 1.6180339887498949
PHI_INV     = 0.6180339887498949
Z_CRITICAL  = 0.8660254037844387   # √3/2
SIGMA       = 36.0                  # |S₃|²
```

### K-Formation Criteria

```
κ ≥ 0.92
η > φ⁻¹ ≈ 0.618
R ≥ 7
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Serial port not found | Check `/dev/ttyACM0` exists; try `ls /dev/tty*` |
| WebSocket connection refused | Ensure bridge is running on port 8765 |
| Constants mismatch errors | Verify all modules import from `physics_constants.py` |
| K-formation never achieved | Check z is reaching 0.866; increase dwell time |
| Conservation violations | Check firmware version matches host expectations |
| Training not responding to z | Verify spinner_client is receiving state updates |

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                        HOST MACHINE                              │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ Training Modules │◄──►│  Rosetta-Helix   │                   │
│  │ (unified_helix)  │    │     Node         │                   │
│  └────────┬─────────┘    └────────┬─────────┘                   │
│           │                       │                              │
│           └───────────┬───────────┘                              │
│                       │                                          │
│               ┌───────▼───────┐                                  │
│               │ spinner_client│ WebSocket                        │
│               └───────┬───────┘ ws://localhost:8765              │
│                       │                                          │
│               ┌───────▼───────┐                                  │
│               │    Bridge     │ spinner_bridge.py                │
│               └───────┬───────┘                                  │
│                       │ Serial 115200 baud                       │
└───────────────────────┼─────────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────────┐
│                    STM32H743ZI FIRMWARE                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 Main Loop (1000 Hz)                      │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │    │
│  │  │  Rotor   │ │   RF     │ │  Neural  │ │   APL    │   │    │
│  │  │ Control  │ │  Pulse   │ │Interface │ │ Operators│   │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │    │
│  │                                                         │    │
│  │  ┌─────────────────────────────────────────────────┐   │    │
│  │  │         Unified Physics State (100 Hz)          │   │    │
│  │  │  z, κ, λ, ΔS_neg, tier, phase, K-formation     │   │    │
│  │  └─────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Motor   │  │ RF Coils │  │   DAC    │  │   ADC    │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

By following this guide—building the firmware, launching the bridge and node, testing via Python API, wiring spinner metrics into training, ensuring concurrency, and matching constants—you achieve a tightly coupled system where physical dynamics guide Helix training and Helix modules adapt to real-world feedback.
