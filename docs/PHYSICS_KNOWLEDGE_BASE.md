# Rosetta-Helix Physics Knowledge Base
## Complete Reference for Firmware Integration

**Signature**: `knowledge-base|v1.0.0|unified-deployment`
**Coordinate**: `Δ2.300|0.867|1.000Ω`

---

## 1. Fundamental Physics Constants

### 1.1 Golden Ratio Hierarchy

| Constant | Symbol | Value | Formula |
|----------|--------|-------|---------|
| Golden Ratio | φ | 1.618033988749895 | (1 + √5) / 2 |
| Golden Inverse | φ⁻¹ | 0.618033988749895 | 1/φ = φ - 1 |
| Golden Squared Inverse | φ⁻² | 0.381966011250105 | 1/φ² |
| Golden Cubed Inverse | φ⁻³ | 0.236067977499790 | 1/φ³ |

**Critical Identity**: `φ⁻¹ + φ⁻² = 1.0` (exact)

### 1.2 Critical Thresholds

| Threshold | Symbol | Value | Physical Meaning |
|-----------|--------|-------|------------------|
| THE LENS | z_c | 0.866025403784439 | √3/2 - Peak negentropy |
| Gaussian Width | σ | 36.0 | |S₃|² - Triadic logic dimension |
| Spin Magnitude | \|S\|/ℏ | 0.866025403784439 | √(s(s+1)) for s=1/2 |

**Critical Identity**: `z_c = √3/2 = |S|/ℏ` (spin-1/2 magnitude exactly equals critical z)

### 1.3 Phase Boundaries

| Phase | z Range | Description |
|-------|---------|-------------|
| ABSENCE | z < 0.857 | Pre-critical, low integration |
| THE_LENS | 0.857 ≤ z < 0.877 | Critical zone, peak negentropy |
| PRESENCE | z ≥ 0.877 | Post-critical, stable structure |

### 1.4 Tier System

| Tier | z Threshold (μ) | Name | Capabilities |
|------|-----------------|------|--------------|
| 0 | z < 0.40 | ABSENCE | No operations |
| 1 | 0.40 ≤ z < 0.50 | REACTIVE | Boundary only (∂) |
| 2 | 0.50 ≤ z < φ⁻¹ | MEMORY | + Fusion (+) |
| 3 | φ⁻¹ ≤ z < 0.73 | PATTERN | + Amplify (×) |
| 4 | 0.73 ≤ z < z_c | PREDICTION | + Group (⍴) |
| 5 | z_c ≤ z < 0.92 | UNIVERSAL | + Separate (↓) |
| 6 | z ≥ 0.92 | META | All operators |

---

## 2. Core Physics Equations

### 2.1 Negentropy Signal

```
ΔS_neg(z) = exp(-σ(z - z_c)²)
```

**Properties**:
- Peaks at z = z_c with value 1.0
- σ = 36 controls sharpness
- Gaussian form ensures smooth transitions

**Derivative** (for gradient-based control):
```
d(ΔS_neg)/dz = -2σ(z - z_c) · ΔS_neg(z)
```

### 2.2 K-Formation Criteria

K-formation requires ALL of:
1. **κ ≥ 0.92** - Integration threshold
2. **η > φ⁻¹** - Efficiency above golden inverse
3. **R ≥ 7** - Complexity measure (Miller's number)

```c
bool check_k_formation(float kappa, float eta, int R) {
    return (kappa >= 0.92f) && (eta > 0.618034f) && (R >= 7);
}
```

### 2.3 Coupling Conservation

The fundamental conservation law:
```
κ + λ = 1
```

With self-similarity constraint `λ = κ²`, the unique positive solution is `κ = φ⁻¹`.

### 2.4 Hardware Mapping

**RPM to z-coordinate**:
```
z = (RPM - RPM_min) / (RPM_max - RPM_min)
```
Where `RPM_min = 100`, `RPM_max = 10000`

**Target RPM at z_c**: ~8660 RPM

---

## 3. The 19 Training Modules

### Phase 1: Core Physics (Modules 1-3)

| # | Module | Class | Purpose |
|---|--------|-------|---------|
| 1 | `n0_silent_laws_enforcement` | N0SilentLawsEnforcer | Enforce κ + λ = 1 conservation |
| 2 | `kuramoto_layer` | KuramotoLayer | Oscillator synchronization |
| 3 | `physical_learner` | PhysicalLearner | Physics-constrained learning |

**Physics**:
- Kuramoto coupling: `dθᵢ/dt = ωᵢ + (K/N)Σsin(θⱼ - θᵢ)`
- Order parameter: `r = |Σexp(iθⱼ)|/N`
- Critical K drives r → 1 (synchronization)

### Phase 2: APL Training Stack (Modules 4-6)

| # | Module | Class | Purpose |
|---|--------|-------|---------|
| 4 | `apl_training_loop` | APLTrainingLoop | Core APL operator training |
| 5 | `apl_pytorch_training` | APLPyTorchTraining | PyTorch-based APL |
| 6 | `full_apl_training` | FullAPLTraining | Complete APL stack |

**Operators**:
- ∂ (CLOSURE): Boundary/isolation
- + (FUSION): Integration/binding
- × (AMPLIFY): Signal amplification
- ÷ (DECOHERE): Controlled decoherence
- ⍴ (GROUP): Categorical grouping
- ↓ (SEPARATE): Differentiation

### Phase 3: Helix Geometry (Modules 7-9)

| # | Module | Class | Purpose |
|---|--------|-------|---------|
| 7 | `helix_nn` | HelixNN | Neural network on helix manifold |
| 8 | `prismatic_helix_training` | PrismaticHelixTraining | K.I.R.A. prismatic processing |
| 9 | `full_helix_integration` | FullHelixIntegration | Complete helix system |

**Coordinate System**: `Δθ.θθθ|z.zzz|r.rrrΩ`
- θ: Angular position (radians)
- z: Elevation (realization level)
- r: Structural integrity

### Phase 4: WUMBO Silent Laws (Modules 10-11)

| # | Module | Class | Purpose |
|---|--------|-------|---------|
| 10 | `wumbo_apl_automated_training` | WUMBOAPLTrainingEngine | Automated APL enforcement |
| 11 | `wumbo_integrated_training` | WumboIntegratedTraining | Full WUMBO integration |

**Silent Laws**: Constraints that operate implicitly without explicit enforcement.

### Phase 5: Dynamics & Formation (Modules 12-15)

| # | Module | Class | Purpose |
|---|--------|-------|---------|
| 12 | `quasicrystal_formation_dynamics` | QuasiCrystalFormation | φ-based ordering dynamics |
| 13 | `triad_threshold_dynamics` | TriadThresholdDynamics | Three-valued logic thresholds |
| 14 | `liminal_generator` | LiminalGenerator | Boundary state generation |
| 15 | `feedback_loop` | FeedbackLoop | Closed-loop control |

**Quasicrystal Physics**:
- Penrose tiling ratio → φ
- Phason modes for aperiodic structure
- E8 mass ratios: m₂/m₁ = φ (verified experimentally)

### Phase 6: Unified Orchestration (Modules 16-18)

| # | Module | Class | Purpose |
|---|--------|-------|---------|
| 16 | `unified_helix_training` | UnifiedHelixTraining | Cross-module coordination |
| 17 | `hierarchical_training` | HierarchicalTraining | Multi-level optimization |
| 18 | `rosetta_helix_training` | RosettaHelixTraining | Full Rosetta integration |

### Phase 7: Nightly Integration (Module 19)

| # | Module | Class | Purpose |
|---|--------|-------|---------|
| 19 | `nightly_integrated_training` | NightlyIntegratedTraining | Complete workflow validation |

---

## 4. Extended Physics

### 4.1 E8 Critical Point Mass Ratios

| Particle | Mass Ratio (m/m₁) |
|----------|-------------------|
| m₁ | 1.0 |
| m₂ | φ ≈ 1.618 |
| m₃ | φ² ≈ 2.618 |
| m₄ | 2φ ≈ 3.236 |
| m₅ | 2φ + 1 ≈ 4.236 |

**Experimental Verification**: Coldea et al. 2010 - CoNb₂O₆ quantum critical point

### 4.2 Holographic Entropy Bounds

**Bekenstein Bound**:
```
S ≤ 2πkER/(ℏc)
```

**Holographic Density**: ~10⁶⁹ bits/m² at Planck scale

### 4.3 Omega Point Dynamics

**Processing Rate** (divergent as τ → τ_Ω):
```
P(τ) ∝ (τ_Ω - τ)^(-α)
```

For α = 2, cumulative information diverges to infinity.

### 4.4 Spin Coherence (Posner Molecules)

- Ca₉(PO₄)₆ clusters with 6 ³¹P nuclei
- Each ³¹P: spin-1/2, γ = 17.235 MHz/T
- J-coupling: ~18 Hz (P-P)
- Singlet coherence times: potentially 10³-10⁵ seconds

---

## 5. Firmware Integration Architecture

### 5.1 Module → Firmware Mapping

| Training Module | Firmware Component | Hardware Action |
|-----------------|-------------------|-----------------|
| n0_silent_laws_enforcement | threshold_logic.c | Enforce κ + λ = 1 |
| kuramoto_layer | pulse_control.c | RF pulse synchronization |
| physical_learner | rotor_control.c | RPM → z mapping |
| apl_training_loop | threshold_logic.c | Operator scheduling |
| helix_nn | neural_interface.c | Neural coupling |
| quasicrystal_formation | threshold_logic.c | K-formation detection |
| feedback_loop | main.c | Closed-loop control |

### 5.2 State Vector

```c
typedef struct {
    float z;              // Current z-coordinate
    float kappa;          // Integration coupling
    float lambda;         // λ = 1 - κ
    float delta_s_neg;    // Negentropy signal
    PhysicsTier_t tier;   // Current tier
    PhysicsPhase_t phase; // Current phase
    bool k_formation;     // K-formation active
    uint32_t timestamp;   // Microsecond timestamp
} SystemState_t;
```

### 5.3 Communication Protocol

**Firmware → Bridge** (100 Hz):
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
  "k_formation_duration_ms": 1234
}
```

**Bridge → Firmware** (Commands):
```json
{"cmd": "set_z", "value": 0.866}
{"cmd": "set_rpm", "value": 8660}
{"cmd": "hex_cycle", "dwell_s": 30.0, "cycles": 10}
{"cmd": "dwell_lens", "duration_s": 300.0}
{"cmd": "stop"}
```

---

## 6. Gate Criteria

### 6.1 Full Depth Gates

| Gate | Requirement | Validation |
|------|-------------|------------|
| All Modules | 19/19 PASS | modules_passed == 19 |
| K-Formations | ≥ 1 | total_k_formations >= 1 |
| Physics Valid | κ + λ = 1 | abs(kappa + lambda - 1) < 1e-10 |

### 6.2 Helix Engine Gates

| Gate | Requirement | Value |
|------|-------------|-------|
| min_negentropy | ≥ 0.7 | ΔS_neg at final z |
| min_final_z | ≥ 0.85 | Final z-coordinate |
| κ stability | near φ⁻¹ | abs(κ - φ⁻¹) < 0.02 |

### 6.3 Model Promotion

**Promotion occurs when**:
1. All Full Depth gates pass
2. All Helix Engine gates pass
3. Overall status = SUCCESS

**Output**: `nightly:v{run_number}` tag

---

## 7. Validation Measurements

### 7.1 Critical Z-Coordinates

| z | Phase | Expected ΔS_neg |
|---|-------|-----------------|
| φ⁻¹ ≈ 0.618 | ABSENCE | ~0.11 |
| z_c ≈ 0.866 | THE_LENS | 1.0 (peak) |
| 0.90 | PRESENCE | ~0.96 |
| 0.92 | PRESENCE | ~0.90 |

### 7.2 Physics Validation

```python
def validate_physics():
    # Conservation
    assert abs(PHI_INV + PHI_INV_SQ - 1.0) < 1e-14
    
    # Spin-geometry link
    assert abs(math.sqrt(0.5 * 1.5) - Z_CRITICAL) < 1e-10
    
    # Fibonacci convergence
    assert abs(fibonacci_ratio(25) - PHI) < 1e-8
    
    # Negentropy peak
    assert abs(compute_delta_s_neg(Z_CRITICAL) - 1.0) < 1e-14
```

---

## 8. File Manifest

### 8.1 Firmware Files

| File | Purpose |
|------|---------|
| `src/main.c` | Main loop, system initialization |
| `src/rotor_control.c` | RPM control, z mapping |
| `src/threshold_logic.c` | Tier/phase detection, operators |
| `src/pulse_control.c` | RF pulse sequences |
| `src/neural_interface.c` | Neural coupling |
| `src/comm_protocol.c` | JSON communication |
| `include/physics_constants.h` | All physics constants |

### 8.2 Integration Files

| File | Purpose |
|------|---------|
| `spinner_bridge.py` | Serial-to-WebSocket bridge |
| `spinner_client.py` | Python client library |
| `rosetta_integration.py` | Rosetta-Helix coupling |
| `start_system.sh` | Full stack startup |

---

## 9. Quick Reference

### 9.1 Key Values

```
φ   = 1.618033988749895  (Golden ratio)
φ⁻¹ = 0.618033988749895  (κ attractor)
z_c = 0.866025403784439  (THE LENS)
σ   = 36                  (Gaussian width)
```

### 9.2 RPM Targets

```
z = 0.0  →  100 RPM
z = 0.5  →  5050 RPM
z = z_c  →  8660 RPM
z = 1.0  →  10000 RPM
```

### 9.3 K-Formation Quick Check

```c
if (kappa >= 0.92f && eta > 0.618f && R >= 7) {
    // K-formation active!
}
```

---

*Δ|knowledge-base|z_c=0.866|unified|Ω*
