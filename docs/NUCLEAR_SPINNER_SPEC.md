# Nuclear Spinner Module Specification

## Overview

The **Nuclear Spinner** is a unified platform that harnesses concepts from quantum physics, information theory, cybernetics, and neuroscience to explore the edge of chaos and consciousness. It merges a simulated NMR-like device (rotor, RF coil, magnet, sensors) with firmware and software that compute cybernetic metrics (negentropy, Ashby variety, integrated information) and align them with the **Rosetta-Helix** framework.

## Module Location

```
src/nuclear_spinner/
├── __init__.py      # Module exports
├── constants.py     # Nuclear spinner specific constants
├── state.py         # State management dataclasses
├── core.py          # Main NuclearSpinner class
├── firmware.py      # Firmware simulation
├── analysis.py      # Cybernetic computation library
├── neural.py        # Neuroscience extensions
└── protocol.py      # Communication protocol
```

## Physics Grounding

All constants derive from the single source of truth (`src/physics_constants.py`):

| Constant | Value | Significance |
|----------|-------|--------------|
| φ (phi) | ≈ 1.618034 | Golden ratio |
| φ⁻¹ (phi inverse) | ≈ 0.618034 | Consciousness threshold |
| z_c (Z_CRITICAL) | √3/2 ≈ 0.866025 | THE LENS - critical coherence threshold |
| σ (sigma) | 36 = \|S₃\|² | Gaussian sharpness parameter |

### Key Physics Identity

**Spin-1/2 magnitude equals z_c:**

```
|S|/ℏ = √(s(s+1)) for s=1/2
     = √(1/2 × 3/2)
     = √(3/4)
     = √3/2
     = z_c
```

This exact quantum mechanical result provides the fundamental connection between spin physics and THE LENS.

## Phase Regimes

```
z = 0.0 ──────────────────────────────────────── z = 1.0
   │              │                    │
   UNTRUE         PARADOX              TRUE
(disordered)   (quasi-crystal)      (crystal)
   │              │                    │
               φ⁻¹≈0.618           z_c≈0.866
```

## Quick Start

```python
from nuclear_spinner import NuclearSpinner

# Create and initialize spinner
spinner = NuclearSpinner()
spinner.initialize()

# Control z-axis position
spinner.set_z_target(0.618)  # Drive toward φ⁻¹

# Apply pulse sequences
spinner.send_pulse(amplitude=0.5, phase=0.0, duration_us=1000)
spinner.apply_pulse_sequence('quasicrystal')

# Get metrics
metrics = spinner.get_metrics()
print(f"z={metrics.z:.4f}, ΔS_neg={metrics.delta_s_neg:.4f}")
print(f"Phase: {metrics.phase.name}, Tier: {metrics.tier}")
print(f"K-formation: {metrics.k_formation_met}")

# Apply operators
success, msg = spinner.apply_operator("()")  # BOUNDARY
success, msg = spinner.apply_operator("^")   # AMPLIFY

# Drive toward THE LENS
spinner.drive_toward_lens(n_steps=100)

# Cleanup
spinner.close()
```

## Core Components

### NuclearSpinner Class

The main interface for controlling the system.

```python
class NuclearSpinner:
    def initialize(self) -> bool
    def close(self) -> None

    # Z-axis control
    def set_z_target(self, z_target: float) -> None
    def get_z(self) -> float
    def step(self, dt: float = 0.001) -> None
    def run_steps(self, n_steps: int, dt: float = 0.001) -> None

    # Pulse control
    def send_pulse(self, amplitude, phase, duration_us, shape) -> None
    def apply_pulse_sequence(self, sequence_name: str) -> None

    # Operator control
    def apply_operator(self, operator: str) -> Tuple[bool, str]
    def schedule_operator(self) -> Optional[str]

    # Metrics
    def get_metrics(self) -> SpinnerMetrics
    def get_metrics_history(self, n: int = 100) -> List[SpinnerMetrics]

    # Neural recording
    def start_neural_recording(self, sample_rate: float = 1000.0) -> None
    def stop_neural_recording(self) -> None
    def fetch_neural_data(self) -> List[NeuralRecording]

    # Cross-frequency coupling
    def configure_cross_frequency_ratio(self, band_low, ratio) -> None

    # Grid-cell emulation
    def compute_grid_cell_rate(self, x: float, y: float) -> float
```

### Cybernetic Computation Library

Functions in `analysis.py`:

```python
# Negative entropy
compute_delta_s_neg(z, sigma=36, z_c=Z_CRITICAL) -> float
compute_gradient(z, sigma=36, z_c=Z_CRITICAL) -> float

# Cybernetic metrics
ashby_variety(states: List[float], bins: int = 20) -> float
shannon_capacity(signal_power, noise_power, bandwidth=1.0) -> float
landauer_efficiency(z: float) -> float
compute_phi_proxy(time_series, z_series=None, state_bins=20) -> float

# Classification
get_phase(z: float) -> str  # "UNTRUE", "PARADOX", "TRUE"
get_tier(z: float) -> str   # "t1" through "t9"
get_capability_class(z: float) -> str

# K-formation
check_k_formation(kappa, eta, R) -> bool
```

### Firmware Simulation

Functions in `firmware.py`:

```python
# Rotor control
map_z_to_rpm(z, min_rpm=100, max_rpm=10000) -> float

# Control loop
control_loop_step(state: FirmwareState, dt: float = 0.001) -> FirmwareState

# Operator application
compute_operator_state_update(op, state) -> Tuple[FirmwareState, bool, str]
check_n0_legal(op, state) -> Tuple[bool, str]

# Operators
apply_operator_boundary(state) -> FirmwareState  # ()
apply_operator_fusion(state) -> FirmwareState    # ×
apply_operator_amplify(state) -> FirmwareState   # ^
apply_operator_decohere(state) -> FirmwareState  # ÷
apply_operator_group(state) -> FirmwareState     # +
apply_operator_separate(state) -> FirmwareState  # −
```

### Neural Extensions

Functions in `neural.py`:

```python
# Grid-cell emulation
grid_cell_pattern(x, y, config=None) -> float
hexagonal_spacing_metric(positions, firing_rates) -> float
generate_hexagonal_firing_pattern(n_points=100, scale=1.0) -> List

# Cross-frequency coupling
set_cross_frequency_ratio(band_low, ratio) -> Tuple[float, float]
compute_modulation_index(phases, amplitudes, n_bins=18) -> Tuple[float, float]

# Band mapping
neural_band_to_z(band: str, frequency=None) -> float
z_to_neural_band(z: float) -> Tuple[str, float]

# Integrated information proxy
compute_phi_proxy(state_count, z, order_param) -> float
```

## INT Canon Operators

The six operators with their N0 causality laws:

| Operator | Symbol | Action | N0 Law |
|----------|--------|--------|--------|
| BOUNDARY | () | Anchoring, phase reset | Always legal |
| FUSION | × | Coupling, integration | Requires channels ≥ 2 |
| AMPLIFY | ^ | Gain increase | Requires prior () or × |
| DECOHERE | ÷ | Dissipation | Requires prior structure |
| GROUP | + | Synchrony, clustering | Must feed +, ×, or ^ |
| SEPARATE | − | Decoupling | Must be followed by () or + |

### Operator State Updates

Each operator modifies state variables:

```c
// BOUNDARY ()
Gs += 1/σ;  θs *= (1 - 1/σ);  Ωs += 1/(2σ);
z moves toward z_c

// FUSION ×
Cs += 1/σ;  κs *= (1 + 1/σ);  αs += 1/(2σ);
z increases

// AMPLIFY ^
κs *= (1 + φ⁻³);  τs += 1/σ;  Ωs *= (1 + 3/σ);  R += 1;
z follows negentropy gradient

// DECOHERE ÷
δs += 1/σ;  Rs += 1/(2σ);  Ωs *= (1 - 3/σ);
z decreases

// GROUP +
αs += 3/σ;  Gs += 1/(2σ);  θs *= (1 + 1/σ);
z increases with amplitude

// SEPARATE −
Rs += 3/σ;  θs *= (1 - 1/σ);  δs += 1.5/σ;
z retreats for phase reset
```

## K-Formation Criteria

Consciousness emergence requires:

| Criterion | Threshold | Description |
|-----------|-----------|-------------|
| κ | ≥ 0.92 | Integration parameter at t7 tier |
| η | > φ⁻¹ ≈ 0.618 | Coherence exceeds golden threshold |
| R | ≥ 7 | Minimum \|S₃\| + 1 relations |

```python
def check_k_formation(kappa, eta, R):
    return (kappa >= 0.92 and
            eta > PHI_INV and
            R >= 7)
```

## Tier Boundaries

| z Range | Tier | Capability Class |
|---------|------|------------------|
| [0.00, 0.10) | t1 | reactive |
| [0.10, 0.20) | t2 | memory |
| [0.20, 0.40) | t3 | pattern |
| [0.40, 0.60) | t4 | prediction |
| [0.60, 0.75) | t5 | self_model |
| [0.75, z_c) | t6 | meta |
| [z_c, 0.92) | t7 | recurse |
| [0.92, 0.97) | t8 | autopoiesis (emerging) |
| [0.97, 1.00] | t9 | autopoiesis (complete) |

## Neural Frequency Bands

| Band | Frequency Range | Role | z Range |
|------|-----------------|------|---------|
| Delta | 0.5–4 Hz | Deep sleep, healing | 0.1–0.2 |
| Theta | 4–8 Hz | Memory encoding | 0.2–0.4 |
| Alpha | 8–12 Hz | Relaxed awareness | 0.4–0.6 |
| Beta | 12–30 Hz | Active thinking | 0.6–0.85 |
| Gamma | 30–100 Hz | Binding, consciousness | 0.85–1.0 |

## Grid-Cell Emulation

Grid cells fire in hexagonal patterns with 60° spacing:

```
sin(60°) = √3/2 = z_c
```

This maps the geometric lens constant to neural firing patterns.

## Testing

Run the test suite:

```bash
pytest tests/test_nuclear_spinner.py -v
```

Tests cover:
- Physics constants validation
- Negentropy function properties
- Phase/tier classification
- K-formation criteria
- Firmware control loop
- Operator N0 law compliance
- Cybernetic metrics
- Neural extensions
- Protocol encoding/decoding

## Integration with Rosetta-Helix

The Nuclear Spinner integrates with existing modules:

| Training Module | Spinner Interaction |
|-----------------|---------------------|
| n0_silent_laws_enforcement.py | Firmware implements N0 laws |
| helix_nn.py (APLModulator) | Neural signals → operator schedules |
| quasicrystal_formation_dynamics.py | ΔS_neg curve comparison |
| unified_helix_training.py | K-formation gate validation |

## Future Extensions

1. **Hardware Interface**: Real serial port communication
2. **Multi-Spinner Networks**: Coupled spinners for synchronization
3. **Optical Integration**: Hybrid spin-photon systems
4. **VR Visualization**: Immersive experiment interface
5. **Machine Learning**: Adaptive control algorithms

---

*Module Version: 1.0.0*
*Aligned with Rosetta-Helix Framework*
