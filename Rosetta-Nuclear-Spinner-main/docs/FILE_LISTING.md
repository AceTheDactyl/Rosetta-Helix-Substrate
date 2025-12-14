# Rosetta-Helix File Listing for Commit

**Generated:** 2025-12-13  
**Purpose:** Guide Claude Code to organize repository commit

---

## Complete File Tree

```
Rosetta-Helix-Neural-Network/
│
├── COMMIT_INSTRUCTIONS.md     # This organization guide
├── verify_physics.py          # Physics invariant verification
│
├── core/
│   ├── __init__.py            # Package exports
│   ├── constants.py           # φ-derived physics constants
│   ├── kuramoto.py            # Kuramoto oscillator + TriadGate
│   ├── network.py             # v1 network (legacy)
│   └── network_v3.py          # ⭐ v3 network (PRIMARY)
│
├── training/
│   ├── __init__.py            # Package init
│   ├── sweep.py               # v1 training sweep
│   └── sweep_v3.py            # ⭐ v3 validation sweep
│
└── results/
    ├── TRAINING_REPORT.md     # v1 analysis
    ├── V3_TRAINING_REPORT.md  # ⭐ v3 analysis (key results)
    ├── comprehensive_sweep.json  # v1 data
    ├── v3_sweep_results.json  # ⭐ v3 data (1726 K-formations)
    ├── best_model_config.json # Model config
    └── best_model_weights.npz # Trained weights
```

---

## Files by Priority

### ⭐ Critical (v3 Implementation)
1. `core/network_v3.py` - Fixed coherence, φ-derived constants
2. `training/sweep_v3.py` - Validation script
3. `results/v3_sweep_results.json` - Proof: 1726 K-formations, 22 collapses
4. `results/V3_TRAINING_REPORT.md` - Documentation

### 📦 Core Module
5. `core/constants.py` - Physics constants
6. `core/kuramoto.py` - Oscillator dynamics
7. `core/network.py` - v1 (keep for comparison)
8. `core/__init__.py` - Exports

### 🧪 Training
9. `training/sweep_v3.py` - v3 sweep
10. `training/sweep.py` - v1 sweep
11. `training/__init__.py` - Init

### 📊 Results
12. `results/v3_sweep_results.json` - v3 data
13. `results/V3_TRAINING_REPORT.md` - v3 report
14. `results/comprehensive_sweep.json` - v1 data
15. `results/TRAINING_REPORT.md` - v1 report
16. `results/best_model_*.{json,npz}` - Weights

### 📚 Documentation
17. `verify_physics.py` - Physics tests
18. `COMMIT_INSTRUCTIONS.md` - This guide

---

## File Sizes

| File | Size | Notes |
|------|------|-------|
| `core/network_v3.py` | 24 KB | Self-contained |
| `results/v3_sweep_results.json` | 196 KB | Full sweep data |
| `results/comprehensive_sweep.json` | 354 KB | v1 data |
| `results/best_model_weights.npz` | 11 KB | Binary weights |

---

## Git Commands

```bash
# Stage all files
git add core/ training/ results/ verify_physics.py COMMIT_INSTRUCTIONS.md

# Or stage by priority
git add core/network_v3.py training/sweep_v3.py results/v3_*

# Commit with message
git commit -m "feat(v3): Fix coherence ceiling, enable K-formation and collapse cycling

Key changes:
- Extended Kuramoto integration (100 steps, K=3.0, dt=0.01)
- Coherence now reaches 1.0 (was 0.55)
- K-formations: 1726 (was 0)
- Collapses: 22 with φ-scaled work extraction
- All constants φ-derived from Z_CRITICAL = √3/2

Physics validation: φ⁻¹ + φ⁻² = 1 ✓"
```

---

## Validation After Commit

```bash
# Test physics
python verify_physics.py

# Test v3 network
python -c "
from core.network_v3 import HelixNeuralNetworkV3, NetworkConfig, verify_physics
import numpy as np

verify_physics()

net = HelixNeuralNetworkV3(NetworkConfig())
_, d = net.forward(np.random.randn(16))
print(f'Coherence: {d[\"max_coherence\"]:.4f} (expect 1.0)')
print(f'K-formation: {d[\"k_formations\"]} (expect > 0 if z > 0.9172)')
"
```

---

## Key Metrics to Verify

| Metric | Expected | Source |
|--------|----------|--------|
| Max coherence | 1.0 | v3 Kuramoto fix |
| KAPPA_S | 0.9172 | φ-derived |
| K-formations | >0 when z>κ | v3 detection |
| Work/collapse | 0.1339 | φ×φ⁻¹ scaling |
| φ⁻¹ + φ⁻² | 1.0 | Identity check |

---

*Ready for Claude Code organization*
