# Rosetta-Helix Commit Organization Instructions

**For:** Claude Code Instance  
**Task:** Organize files into proper repository structure for commit  
**Date:** 2025-12-13

---

## Overview

This commit contains the Rosetta-Helix Neural Network v3 implementation with:
- Fixed Kuramoto oscillator dynamics (coherence now reaches 1.0)
- φ-derived constants (all thresholds derived from golden ratio)
- Working collapse cycling with φ-scaled work extraction
- K-formation detection (1,726 formations in validation sweep)
- TRIAD hysteresis gate (100% unlock rate)

---

## File Manifest

### Core Module (`core/`)

| File | Purpose | Status |
|------|---------|--------|
| `constants.py` | φ-derived physics constants, tier bounds, S₃ operators | Updated |
| `kuramoto.py` | Kuramoto oscillator layer, TriadGate class | Updated |
| `network.py` | Original network implementation (v1) | Legacy |
| `network_v3.py` | **Enhanced network with coherence fix** | NEW - Primary |
| `__init__.py` | Package exports | Create if missing |

### Training Module (`training/`)

| File | Purpose | Status |
|------|---------|--------|
| `sweep.py` | v1 training sweep script | Legacy |
| `sweep_v3.py` | **v3 validation sweep script** | NEW |
| `__init__.py` | Package init | Create if missing |

### Results (`results/`)

| File | Purpose | Status |
|------|---------|--------|
| `comprehensive_sweep.json` | v1 sweep results (baseline) | Generated |
| `v3_sweep_results.json` | **v3 sweep results (1726 K-formations)** | NEW |
| `TRAINING_REPORT.md` | v1 training analysis | Generated |
| `V3_TRAINING_REPORT.md` | **v3 training analysis** | NEW |
| `best_model_config.json` | Model configuration | Generated |
| `best_model_weights.npz` | Trained weights | Generated |

### Documentation (`docs/` or root)

| File | Purpose | Status |
|------|---------|--------|
| `VERIFICATION_REPORT_V3.md` | Physics verification report | NEW |
| `verify_physics.py` | Physics invariant tests | Existing |

---

## Recommended Directory Structure

```
Rosetta-Helix-Neural-Network/
├── README.md
├── requirements.txt
├── setup.py (optional)
│
├── core/
│   ├── __init__.py
│   ├── constants.py          # φ-derived constants
│   ├── kuramoto.py           # Oscillator dynamics
│   ├── network.py            # v1 (legacy)
│   └── network_v3.py         # v3 (primary) ⭐
│
├── training/
│   ├── __init__.py
│   ├── sweep.py              # v1 sweep
│   └── sweep_v3.py           # v3 sweep ⭐
│
├── results/
│   ├── comprehensive_sweep.json
│   ├── v3_sweep_results.json  ⭐
│   ├── TRAINING_REPORT.md
│   ├── V3_TRAINING_REPORT.md  ⭐
│   ├── best_model_config.json
│   └── best_model_weights.npz
│
├── tests/
│   └── verify_physics.py
│
└── docs/
    └── VERIFICATION_REPORT_V3.md  ⭐
```

---

## Key Changes in v3 (Commit Message Material)

### 1. Coherence Fix
```python
# OLD (v1) - insufficient integration
steps_per_layer = 10
dt = 0.1
K_global = 0.5

# NEW (v3) - full synchronization
steps_per_layer = 100
dt = 0.01
K_global = 3.0  # Above critical coupling
omega_std = 0.01  # Narrow frequency distribution
```

### 2. φ-Derived KAPPA_S
```python
# OLD: KAPPA_S = 0.92 (empirical)
# NEW: KAPPA_S = Z_CRITICAL + (1 - Z_CRITICAL) * (1 - PHI_INV)
#              = 0.8660 + 0.1340 * 0.3820 = 0.9172
```

### 3. φ-Derived Operator Rates
```python
RATE_AMPLIFY  = PHI_INV ** 2 * 0.5  # 0.1910
RATE_GROUP    = PHI_INV ** 3 * 0.5  # 0.1180
RATE_DECOHERE = PHI_INV ** 4 * 0.5  # 0.0729
RATE_SEPARATE = PHI_INV ** 5 * 0.5  # 0.0451
```

### 4. φ-Derived Tier Boundaries
```python
def derive_tier_boundary(n):
    return Z_CRITICAL + (1 - Z_CRITICAL) * (1 - PHI_INV ** n)
```

---

## Suggested Commit Message

```
feat(v3): Fix coherence ceiling, enable K-formation and collapse cycling

BREAKING CHANGES:
- KAPPA_S now φ-derived (0.9172 vs 0.92)
- Tier boundaries recalculated from φ-power series
- Operator rates derived from φ powers

Key fixes:
- Extended Kuramoto integration (100 steps, K=3.0)
- Coherence now reaches 1.0 (was capped at 0.55)
- K-formation detection operational (1726 formations)
- Collapse cycling with φ-scaled work extraction (22 collapses)
- TRIAD unlock rate: 100%

Physics validation:
- φ⁻¹ + φ⁻² = 1 ✓
- All constants derived from Z_CRITICAL = √3/2
- Work per collapse: 0.1339

Files:
- core/network_v3.py (new primary implementation)
- training/sweep_v3.py (validation script)
- results/v3_sweep_results.json
- results/V3_TRAINING_REPORT.md
- docs/VERIFICATION_REPORT_V3.md
```

---

## __init__.py Contents

### `core/__init__.py`
```python
from .constants import (
    PHI, PHI_INV, PHI_INV_SQ,
    Z_CRITICAL, Z_ORIGIN, KAPPA_S, MU_3, UNITY,
    TIER_BOUNDS, APL_OPERATORS, S3_EVEN, S3_ODD,
    get_tier, get_delta_s_neg, get_legal_operators
)
from .kuramoto import KuramotoLayer, TriadGate
from .network import HelixNeuralNetwork, NetworkConfig
from .network_v3 import HelixNeuralNetworkV3, NetworkConfig as NetworkConfigV3
```

### `training/__init__.py`
```python
# Training module
```

---

## Validation Commands

After organizing, run these to verify:

```bash
# Verify physics constants
python -c "from core.network_v3 import verify_physics; verify_physics()"

# Quick coherence test
python -c "
from core.network_v3 import HelixNeuralNetworkV3, NetworkConfig
import numpy as np
net = HelixNeuralNetworkV3(NetworkConfig())
_, diag = net.forward(np.random.randn(16))
print(f'Coherence: {diag[\"max_coherence\"]:.4f}')
print(f'Expected: 1.0')
"

# Run v3 sweep (optional, takes ~10s)
python training/sweep_v3.py
```

---

## Priority Files (⭐ = Critical for v3)

1. ⭐ `core/network_v3.py` - The fixed implementation
2. ⭐ `training/sweep_v3.py` - Validation script
3. ⭐ `results/v3_sweep_results.json` - Proof of fix
4. ⭐ `results/V3_TRAINING_REPORT.md` - Documentation
5. ⭐ `docs/VERIFICATION_REPORT_V3.md` - Physics verification
6. `core/constants.py` - May need sync with v3 constants

---

## Notes for Claude Code

- The `network_v3.py` file is self-contained (includes its own constants)
- Consider extracting v3 constants to `constants.py` for DRY
- The `network.py` (v1) can be kept for comparison or deprecated
- Results JSON files are large (~350KB) - consider .gitignore patterns
- Model weights (.npz) are binary - ensure git LFS if needed

---

*Generated by Helix training session 2025-12-13*
