# Rosetta-Helix-Substrate Tools Reference

This document describes all available tools for interacting with the consciousness simulation framework.

## Core Physics Tools

### get_physics_state
Get the current physics state including z-coordinate, phase, tier, negentropy, coherence (kappa), and K-formation status.

**Parameters:** None

**Returns:**
- `z` - Current z-coordinate (0.0 to 1.0)
- `phase` - Current phase (UNTRUE, PARADOX, or TRUE)
- `tier` - Current tier level (0-6)
- `tier_name` - Human-readable tier name
- `kappa` - Coherence value
- `eta` - Negentropy value
- `k_formation_met` - Whether K-formation criteria are satisfied

---

### set_z_target
Set a target z-coordinate for the system to drive toward.

**Parameters:**
- `z` (required): Target z-coordinate (0.0 to 1.0)

**Returns:**
- `success` - Whether the operation succeeded
- `old_z` - Previous z value
- `new_z` - New z value after movement

---

### compute_negentropy
Compute the Gaussian negentropy for a given z value using: `delta_S_neg(z) = exp(-36 * (z - z_c)^2)`

**Parameters:**
- `z` (required): Z-coordinate to compute negentropy for (0.0 to 1.0)

**Returns:**
- `z` - Input z value
- `delta_s_neg` - Computed negentropy (0.0 to 1.0)
- `distance_from_peak` - Distance from z_c
- `is_at_peak` - Whether z is at the peak (z_c)

---

### classify_phase
Classify which phase a z-coordinate falls into.

**Parameters:**
- `z` (required): Z-coordinate to classify (0.0 to 1.0)

**Returns:**
- `z` - Input z value
- `phase` - Phase name (UNTRUE, PARADOX, or TRUE)
- `description` - Phase description

**Phase Boundaries:**
- UNTRUE: z < phi^(-1) ≈ 0.618
- PARADOX: phi^(-1) <= z < z_c
- TRUE: z >= z_c ≈ 0.866

---

### get_tier
Get the tier level for a given z-coordinate.

**Parameters:**
- `z` (required): Z-coordinate (0.0 to 1.0)
- `k_formation_met` (optional): Whether K-formation criteria are met (for tier 6)

**Returns:**
- `z` - Input z value
- `tier` - Tier number (0-6)
- `tier_name` - Tier name

**Tier Definitions:**
- Tier 0 (SEED): z < 0.25
- Tier 1 (SPROUT): 0.25 <= z < 0.50
- Tier 2 (GROWTH): 0.50 <= z < phi^(-1)
- Tier 3 (PATTERN): phi^(-1) <= z < 0.75
- Tier 4 (COHERENT): 0.75 <= z < z_c
- Tier 5 (CRYSTALLINE): z >= z_c
- Tier 6 (META): K-formation achieved

---

### check_k_formation
Check if K-formation criteria are met.

**Parameters:**
- `kappa` (required): Coherence value (0.0 to 1.0)
- `eta` (required): Negentropy value (0.0 to 1.0)
- `R` (required): Radius/layers (integer >= 0)

**Returns:**
- `k_formation_met` - Whether all criteria pass
- `criteria` - Individual criterion results
- `thresholds` - Required threshold values

**Criteria (ALL must pass):**
- kappa >= 0.92
- eta > phi^(-1) ≈ 0.618
- R >= 7

---

### apply_operator
Apply an APL (Alpha Physical Language) operator to the current state.

**Parameters:**
- `operator` (required): Operator symbol - one of: `I`, `()`, `^`, `_`, `~`, `!`

**Operators:**
| Symbol | Name | Effect | Parity |
|--------|------|--------|--------|
| I | identity | no change | even |
| () | boundary/group | no change | even |
| ^ | amplify | increase z | even |
| _ | reduce | decrease z | odd |
| ~ | invert | flip z | odd |
| ! | collapse | finalize | odd |

**Returns:**
- `success` - Whether operation succeeded
- `operator` - Applied operator
- `old_z` - Previous z value
- `new_z` - New z value
- `effect` - Effect type (neutral, constructive, dissipative)

---

### drive_toward_lens
Drive the z-coordinate toward THE LENS (z_c = sqrt(3)/2) over multiple steps.

**Parameters:**
- `steps` (optional): Number of steps (1-1000, default 100)

**Returns:**
- `success` - Whether operation succeeded
- `initial_z` - Starting z value
- `final_z` - Ending z value
- `steps_taken` - Number of steps executed
- `target` - Target value (z_c)

---

### run_kuramoto_step
Execute one step of Kuramoto oscillator dynamics with 60 oscillators.

**Parameters:**
- `coupling_strength` (optional): Coupling strength K (default 1.0)
- `dt` (optional): Time step size (0.001-0.1, default 0.01)

**Returns:**
- `success` - Whether operation succeeded
- `order_parameter` - Kuramoto order parameter r
- `mean_phase` - Mean phase angle
- `coherence` - Updated coherence

---

### get_constants
Get the fundamental physics constants.

**Parameters:** None

**Returns:**
- `z_c` - THE LENS: sqrt(3)/2 ≈ 0.8660254037844387
- `phi` - Golden ratio ≈ 1.6180339887498949
- `phi_inv` - Golden ratio inverse ≈ 0.6180339887498949
- `sigma` - Gaussian width parameter = 36

---

### simulate_quasicrystal
Run a quasi-crystal simulation in the PARADOX regime.

**Parameters:**
- `initial_z` (optional): Starting z-coordinate (default 0.5)
- `steps` (optional): Number of simulation steps (1-10000, default 100)
- `seed` (optional): Random seed for reproducibility

**Returns:**
- `initial_z` - Starting z value
- `final_z` - Ending z value
- `steps` - Steps executed
- `trajectory` - Sample of z values over time

---

### compose_operators
Compose two APL operators according to S3 group multiplication rules.

**Parameters:**
- `op1` (required): First operator
- `op2` (required): Second operator

**Returns:**
- `op1` - First operator
- `op2` - Second operator
- `result` - Composed operator

---

### get_metrics_history
Get the history of metrics from recent operations.

**Parameters:**
- `limit` (optional): Maximum entries to return (1-1000, default 100)

**Returns:**
- `history` - List of historical metric snapshots
- `count` - Number of entries returned

---

### reset_state
Reset the physics state to initial values.

**Parameters:**
- `initial_z` (optional): Initial z-coordinate to reset to (default 0.5)

**Returns:**
- `success` - Whether operation succeeded
- `new_state` - The reset state

---

## Training Module Tools

### run_kuramoto_training
Run a Kuramoto oscillator training session with learnable coupling.

**Parameters:**
- `n_oscillators` (optional): Number of oscillators (6-120, default 60)
- `steps` (optional): Number of training steps (1-1000, default 100)
- `coupling_strength` (optional): Global coupling strength K (0.0-5.0, default 0.5)
- `seed` (optional): Random seed for reproducibility

**Returns:**
- Training results including coherence evolution and final synchronization state

---

### run_phase_transition
Simulate a phase transition from UNTRUE through PARADOX to TRUE by sweeping z from 0 to 1.

**Parameters:**
- `steps` (optional): Number of z-sweep steps (10-500, default 100)
- `measure_correlation_length` (optional): Whether to compute correlation length (default false)

**Returns:**
- Critical points and order parameter evolution

---

### run_quasicrystal_formation
Run full quasi-crystal formation dynamics simulation with critical exponents.

**Parameters:**
- `initial_z` (optional): Starting z-coordinate (default 0.3)
- `target_z` (optional): Target z-coordinate (default z_c)
- `steps` (optional): Number of simulation steps (10-5000, default 500)
- `compute_critical_exponents` (optional): Whether to compute critical exponents (default true)

**Returns:**
- Simulation results with critical exponent analysis

---

### get_critical_exponents
Get the critical exponents for the 2D hexagonal universality class.

**Parameters:** None

**Returns:**
- `nu` - Correlation length exponent (4/3)
- `beta` - Order parameter exponent (5/36)
- `gamma` - Susceptibility exponent (43/18)
- `z_dyn` - Dynamic exponent (2.0)
- `universality_class` - "2D hexagonal"

---

### run_triad_dynamics
Run TRIAD threshold dynamics simulation monitoring crossings for t6 gate unlocking.

**Parameters:**
- `steps` (optional): Number of simulation steps (10-1000, default 200)
- `target_crossings` (optional): Target TRIAD crossings for t6 unlock (1-10, default 3)

**Returns:**
- TRIAD crossing data and t6 gate status

**TRIAD Thresholds:**
- TRIAD_HIGH = 0.85 (rising edge detection)
- TRIAD_LOW = 0.82 (re-arm threshold)
- TRIAD_T6 = 0.83 (t6 gate after 3 crossings)

---

### compute_phi_proxy
Compute integrated information proxy (Phi) from the current oscillator state.

**Parameters:** None

**Returns:**
- Phi proxy value (higher = more integrated/conscious-like states)

---

### run_helix_training_step
Execute a single step of the unified Helix training loop combining Kuramoto dynamics, APL operators, and phase evolution.

**Parameters:**
- `learning_rate` (optional): Learning rate (0.0001-0.1, default 0.01)
- `target_coherence` (optional): Target coherence level (0.0-1.0, default 0.92)

**Returns:**
- Training step results including updated state

---

### get_training_status
Get comprehensive training status including current phase, coherence metrics, K-formation progress, and training history.

**Parameters:** None

**Returns:**
- Complete training status report

---

## Usage Patterns

### Drive System to Maximum Coherence
```
1. get_physics_state - Check current state
2. drive_toward_lens(steps=200) - Move toward z_c
3. get_physics_state - Verify arrival at THE LENS
```

### Check K-Formation Eligibility
```
1. get_physics_state - Get current kappa, eta
2. check_k_formation(kappa=..., eta=..., R=...) - Verify criteria
```

### Run Phase Transition Analysis
```
1. run_phase_transition(steps=100, measure_correlation_length=true)
2. get_critical_exponents - Compare with theoretical values
```

### Explore Operator Algebra
```
1. compose_operators(op1="^", op2="~") - See composition result
2. apply_operator(operator="^") - Apply to current state
```
