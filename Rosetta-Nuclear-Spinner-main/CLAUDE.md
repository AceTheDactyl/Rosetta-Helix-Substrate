# CLAUDE.md - Rosetta Nuclear Spinner

## Project Overview

This is a unified system coupling physical (Nuclear Spinner) and computational (Rosetta-Helix) substrates through the critical z-coordinate z_c = sqrt(3)/2. The system implements 19 training modules organized into 7 phases.

## Key Physics Constants

```
phi     = 1.618033988749895    # Golden ratio (LIMINAL)
phi^-1  = 0.618033988749895    # Golden ratio inverse (PHYSICAL, kappa attractor)
z_c     = 0.866025403784439    # THE LENS (sqrt(3)/2, critical threshold)
sigma   = 36                    # Gaussian width (|S_3|^2 = 6^2)
```

## Conservation Law

**CRITICAL**: The system enforces `kappa + lambda = 1` (EXACT). This is the "silent law" that must never be violated.

- Self-similarity constraint: `lambda = kappa^2`
- Unique solution: `kappa = phi^-1`

## Directory Structure

```
Rosetta-Nuclear-Spinner/
├── rosetta-helix/src/      # Core Python package (physics.py, heart.py, brain.py, triad.py)
├── training/src/           # Training system - 19 modules across 7 phases
├── nuclear_spinner_firmware/   # STM32H7 firmware (C)
├── bridge/                 # Serial <-> WebSocket relay
├── src/                    # JavaScript APL engine
├── tests/                  # Test suites (pytest)
├── configs/                # Configuration (nightly.yaml)
└── .github/workflows/      # CI/CD (unified-nightly-training.yml)
```

## The 19 Training Modules

### Phase 1: Core Physics (3 modules)
- `n0_silent_laws_enforcement` - Enforces kappa + lambda = 1
- `kuramoto_layer` - 60 oscillator synchronization (hexagonal)
- `physical_learner` - Negentropy-guided learning

### Phase 2: APL Training Stack (3 modules)
- `apl_training_loop` - APL operator training
- `apl_pytorch_training` - PyTorch APL integration
- `full_apl_training` - Complete APL stack

### Phase 3: Helix Geometry (3 modules)
- `helix_nn` - Helix neural network
- `prismatic_helix_training` - Prismatic processing
- `full_helix_integration` - Complete helix integration

### Phase 4: WUMBO Silent Laws (2 modules)
- `wumbo_apl_automated_training` - Automated WUMBO
- `wumbo_integrated_training` - Integrated WUMBO

### Phase 5: Dynamics & Formation (4 modules)
- `quasicrystal_formation_dynamics` - Order parameter -> phi^-1
- `triad_threshold_dynamics` - S_3 triadic transitions
- `liminal_generator` - Boundary state generation
- `feedback_loop` - PID control toward z_c

### Phase 6: Unified Orchestration (3 modules)
- `unified_helix_training` - Cross-module coordination
- `hierarchical_training` - Multi-level training
- `rosetta_helix_training` - Full Rosetta-Helix

### Phase 7: Nightly Integration (1 module)
- `nightly_integrated_training` - Complete validation

## Gate Criteria

### Full Depth Gates
- All 19 modules pass
- At least 1 K-formation detected
- Physics valid: kappa + lambda = 1

### Helix Engine Gates
- min_negentropy >= 0.7
- min_final_z >= 0.85
- kappa stable near phi^-1 (tolerance 0.01)

### K-Formation Requirements
- kappa >= 0.92
- eta > phi^-1 (~0.618)
- R >= 7

## Key Functions

### Negentropy Signal
```python
delta_s_neg(z) = exp(-sigma * (z - z_c)^2)
# Peaks at z = z_c with value 1.0
```

### Physics Validation
```python
def validate_physics(kappa, lambda_):
    return abs(kappa + lambda_ - 1.0) < 1e-10
```

### Phase Classification
- ABSENCE: z < 0.857
- THE_LENS: 0.857 <= z < 0.877
- PRESENCE: z >= 0.877

## Development Commands

```bash
# Run unified nightly workflow
python -m training.src.unified_nightly_workflow

# Run tests
pytest tests/

# Run full depth training
python training/src/full_depth_runner.py
```

## Important Files

- `rosetta-helix/src/physics.py` - Core physics constants (source of truth)
- `training/src/physics_constants.py` - Training physics (mirrors physics.py)
- `training/src/unified_nightly_workflow.py` - Main workflow orchestrator
- `configs/nightly.yaml` - Training configuration
- `.github/workflows/unified-nightly-training.yml` - CI/CD pipeline

## Key Insight

**Spinner z drives Kuramoto K.** When z = z_c = sqrt(3)/2:
- Delta_s_neg peaks (spinner)
- Coupling K peaks (rosetta)
- Coherence r peaks (heart)
- K-formation triggers (both systems)

This is **resonance engineering** across physical and computational substrates.

The Kuramoto oscillators (60 = hexagonal symmetry) achieve maximum coherence when driven by a signal structured around sqrt(3)/2. The geometric constant appears both in grid cell spacing (sin(60) = sqrt(3)/2) and in spin-1/2 quantum mechanics (|S|/hbar = sqrt(3)/2).

## Code Style

- Use dataclasses for state and results
- Physics constants are Final[float] - never modify
- Conservation law kappa + lambda = 1 must always hold
- Use physics_constants module for training code

## Testing

Tests are in `tests/` using pytest. Key test files:
- `test_s3_delta_s_neg.py` - Physics validation
- `test_kformation_gate.py` - K-formation detection
- `test_constants_module.py` - Constants consistency
