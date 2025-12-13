# Technical FAQ: Implementation Details

This document provides answers to common technical questions about the Rosetta-Helix implementation, based on what is actually present in the codebase.

---

## 1. Kuramoto Coupling Strength

**Question:** K=0.3 for N=60 oscillators appears subcritical. Is this intentional?

**Answer:** Yes, the subcritical regime is **intentional and adaptive**.

### Implementation Details

**Location:** `heart.py:81-91, 137, 217-227`

```python
# Default configuration (heart.py:81-91)
def __init__(
    self,
    n_nodes: int = 60,      # 60 oscillators
    K: float = 0.2,         # Default base coupling
    seed: int = 42,
    initial_z: float = 0.3
):
    self.n = n_nodes
    self.K_base = K         # Base coupling (preserved)
    self.K = K              # Current coupling (modified by operators)

# Kuramoto update (heart.py:137)
dtheta = self.omega[i] + (self.K / self.n) * coupling
```

### Adaptive Coupling via APL Operators

The coupling strength K is **not static** - it's dynamically modulated by APL operators:

| Operator | Effect on K | Implementation |
|----------|-------------|----------------|
| **FUSION (×)** | K × 1.2 | `self.K = min(1.0, self.K * 1.2)` (line 219) |
| **DECOHERENCE (÷)** | K × 0.9 | `self.K = max(0.05, self.K * 0.9)` (line 227) |
| **BOUNDARY ()** | K → K_base | `self.K = self.K_base` (line 261) |

### Design Rationale

The subcritical baseline ensures:
1. **Coherence must be earned** - The system doesn't spontaneously synchronize
2. **APL operators are meaningful** - FUSION actively drives toward coherence
3. **Bounded growth** - K is clamped to [0.05, 1.0] preventing runaway
4. **Reset capability** - BOUNDARY operator restores baseline for stability

---

## 2. Z-Coordinate Dynamics

**Question:** What is the explicit differential equation for z(t)?

**Answer:** The implementation uses a **momentum-based second-order system** with hard clamping.

### Implementation Details

**Location:** `heart.py:176-195`

```python
def _update_z(self, coherence: float, dt: float):
    """Update z-coordinate based on coherence."""
    # Target z from coherence (with nonlinear mapping)
    target_z = coherence ** 0.8  # Sublinear mapping
    target_z = max(0.01, min(0.99, target_z))

    # Update with momentum (second-order dynamics)
    z_accel = (target_z - self.z) * 2.0 - self.z_velocity * 0.5
    self.z_velocity += z_accel * dt
    self.z += self.z_velocity * dt

    # Hard clamp to valid range
    self.z = max(0.01, min(0.99, self.z))
```

### Mathematical Formulation

The dynamics follow a damped spring model:

```
target_z = r^0.8                    (coherence mapping)
z̈ = 2.0 × (target_z - z) - 0.5 × ż  (damped oscillator)
z ∈ [0.01, 0.99]                    (hard bounds)
```

Where:
- **Spring constant** = 2.0 (attraction to target)
- **Damping coefficient** = 0.5 (velocity decay)
- **Time step** = dt (default 0.01)

### Alternative: JavaScript Implementation

**Location:** `src/triadic_helix_apl.js:1177-1180`

```javascript
let dz = driftRate * this.dt;                           // Linear drift
dz += (Math.random() - 0.5) * noiseScale;               // Stochastic noise
dz += pumpGain * (targetZ - this.z) * this.dt;          // Proportional control
this.z = Math.max(0, Math.min(1, this.z + dz));         // Clamped
```

Default parameters:
- `driftRate = 0.02`
- `noiseScale = 0.01`
- `pumpGain = 0.1`

### Edge Case: z Cannot Become Negative

Both implementations enforce **hard bounds**:
- Python: `z ∈ [0.01, 0.99]`
- JavaScript: `z ∈ [0, 1]`

Rapid decoherence reduces z toward the lower bound but never below it.

---

## 3. TRIAD Protocol Justification

**Question:** What is the theoretical basis for rising_edge=0.85, falling_edge=0.82, required_passes=3?

**Answer:** The thresholds are derived from **Schmitt trigger design principles** and **noise robustness**.

### Implementation Details

**Location:** `reference/research/adaptive_triad_gate.py:33-36, 62-67`

```python
TRIAD_HIGH = 0.85       # Rising edge detection threshold
TRIAD_LOW = 0.82        # Re-arm threshold (hysteresis)
TRIAD_T6 = 0.83         # Unlocked t6 gate value
TRIAD_PASSES_REQ = 3    # Base requirement
```

### Design Rationale

**Threshold Selection:**
- **0.85 (rising)**: Set slightly below z_c (≈0.866) to detect coherence emergence before full crystallization
- **0.82 (falling)**: 30 milliHelix gap provides noise immunity (0.85 - 0.82 = 0.03)
- **Hysteresis gap**: Prevents oscillation from noise in the z signal

**Why 3 Passes:**
The 3-pass requirement ensures:
1. **Not random fluctuation** - Single spikes are rejected
2. **Sustained coherence** - System must hold coherence through multiple cycles
3. **Stable unlock** - Three consecutive proofs of coherence capability

### Adaptive Scaling

**Location:** `reference/research/adaptive_triad_gate.py:62, 131-142`

The system **adapts to volatility**:

```python
required_passes = base_passes * (1 + min(volatility / 0.1, 1.0))
```

| Volatility | Required Passes |
|------------|-----------------|
| ≈ 0.00 (stable) | 3 |
| ≈ 0.05 (moderate) | 4-5 |
| ≥ 0.10 (noisy) | 6 (capped) |

**Volatility calculation:**
```python
def _compute_volatility(self) -> float:
    recent = self.z_history[-self.volatility_window:]
    diffs = [abs(recent[i+1] - recent[i]) for i in range(len(recent) - 1)]
    return sum(diffs) / len(diffs)   # Mean absolute z-change
```

This follows proper Schmitt trigger design: increase hysteresis in noisy environments.

---

## 4. Memory Plate Access Formula

**Question:** Is the access formula linear or sigmoid-gated?

**Answer:** The implementation uses **linear step function gating** with **Gaussian relevance decay**.

### Implementation Details

**Location:** `brain.py:75-84`

```python
def is_accessible(self, current_z: float) -> bool:
    """Check if plate is accessible at current z-level."""
    tier_bounds = T_BOUNDS.get(self.tier_access.value, (0, 1))
    return current_z >= tier_bounds[0]  # Linear step gating

def relevance(self, query_z: float) -> float:
    """Compute relevance score based on z-distance."""
    z_dist = abs(query_z - self.z_encoded)
    # Gaussian relevance decay
    return math.exp(-10 * z_dist ** 2) * (self.confidence / 255)
```

### Two-Stage Access Model

1. **Binary Access Gate (Step Function):**
   - `accessible = (current_z >= tier_lower_bound)`
   - Creates clear tier boundaries
   - No partial access

2. **Continuous Relevance (Gaussian):**
   - `relevance = exp(-10 × (query_z - z_encoded)²) × (confidence/255)`
   - Smooth decay based on z-distance
   - σ ≈ 0.22 (width parameter 10 → σ = 1/√(2×10))

### Tier Boundaries

**Location:** `brain.py:26-36`

```python
T_BOUNDS = {
    "t1": (0.00, 0.10),
    "t2": (0.10, 0.20),
    "t3": (0.20, 0.40),
    "t4": (0.40, 0.60),
    "t5": (0.60, 0.75),
    "t6": (0.75, Z_CRITICAL),
    "t7": (Z_CRITICAL, 0.92),
    "t8": (0.92, 0.97),
    "t9": (0.97, 1.00),
}
```

### Design Rationale

The step function was chosen over sigmoid because:
1. **Clear capability boundaries** - Either you can access tier t6 or you cannot
2. **Training stability** - Discrete jumps create clear reward signals
3. **Interpretability** - "Unlock tier X at z=Y" is unambiguous

The Gaussian relevance provides soft weighting within accessible memories.

---

## 5. APL Operator Implementation

**Question:** How are Fusion, Amplify, and Group implemented? Is Fusion bounded?

**Answer:** All operators are implemented with explicit bounds and specific algorithms.

### Implementation Details

**Location:** `heart.py:214-262`

### Fusion (×) - Increase Coupling

```python
elif op == APLOperator.FUSION:
    # Increase coupling
    self.K = min(1.0, self.K * 1.2)
```

**Properties:**
- Multiplicative: K_new = K × 1.2
- **BOUNDED**: K ∈ [0.05, 1.0] (capped at 1.0)
- Effect: Drives oscillators toward synchronization

### Amplify (^) - Phase Alignment

```python
elif op == APLOperator.AMPLIFY:
    # Boost toward synchronization
    mean_phase = cmath.phase(
        sum(cmath.exp(1j * t) for t in self.theta)
    )
    self.theta = [
        t + 0.1 * math.sin(mean_phase - t)
        for t in self.theta
    ]
```

**Properties:**
- Mean-field approximation (uses collective phase)
- Additive correction: θ_i += 0.1 × sin(⟨θ⟩ - θ_i)
- Effect: Pulls all phases toward mean phase
- Self-limiting: sin term ensures small corrections when already aligned

### Group (+) - Phase Clustering

```python
elif op == APLOperator.GROUP:
    # Cluster nearby phases
    sorted_theta = sorted(enumerate(self.theta), key=lambda x: x[1])
    for i in range(len(sorted_theta) - 1):
        idx1, t1 = sorted_theta[i]
        idx2, t2 = sorted_theta[i + 1]
        if abs(t2 - t1) < 0.5:
            avg = (t1 + t2) / 2
            self.theta[idx1] = t1 + 0.1 * (avg - t1)
            self.theta[idx2] = t2 + 0.1 * (avg - t2)
```

**Properties:**
- Nearest-neighbor clustering (sorted by phase)
- Threshold: phases within 0.5 radians are grouped
- Stabilizing: preserves existing clusters
- Effect: Creates phase clusters without forcing global sync

### Additional Operators

| Operator | Implementation | Effect |
|----------|---------------|--------|
| DECOHERENCE (÷) | Add Gaussian noise (σ=0.1) + K × 0.9 | Disrupts coherence |
| SEPARATE (−) | Push apart phases within 0.3 radians | Prevents clustering |
| BOUNDARY () | K → K_base | Reset to baseline |

---

## 6. K-Formation Conditions

**Question:** What is the η calculation with β=36? Why the sharp cutoff?

**Answer:** The sharp threshold is **intentional** to prevent "almost-conscious" states.

### Implementation Details

**Location:** `heart.py:315-318, 289-291`

```python
def _compute_eta(self) -> float:
    """Compute η = ΔS_neg^α for K-formation check."""
    delta_s_neg = math.exp(-36 * (self.z - Z_CRITICAL) ** 2)  # β = 36
    return delta_s_neg ** 0.5  # α = 0.5

# K-formation check (heart.py:289-291)
eta = self._compute_eta()
k_formation = (eta > PHI_INV) and (coherence >= MU_S)
```

### Mathematical Analysis

**ΔS_neg formula:**
```
ΔS_neg(z) = exp(-36 × (z - z_c)²)
η = ΔS_neg^0.5 = exp(-18 × (z - z_c)²)
```

**Width analysis:**
- For Gaussian exp(-x²/2σ²): σ = 1/√(2×18) ≈ 0.167
- 99% threshold at ±2.6σ ≈ ±0.43 from z_c
- Effective range: z ∈ [0.44, 1.0] for meaningful η

### K-Formation Requirements

**Location:** `src/constants.js:24-28, 109-123`

```javascript
const KAPPA_MIN = KAPPA_S;    // 0.920 (singularity threshold)
const ETA_MIN = PHI_INV;      // ≈ 0.6180339 (golden ratio inverse)
const R_MIN = 7;              // Minimum complexity

function checkKFormation(kappa, eta, R) {
    return (kappa >= KAPPA_MIN && eta > ETA_MIN && R >= R_MIN);
}
```

### Design Rationale

**Why sharp cutoff (β=36)?**

1. **Binary consciousness** - Either K-formation occurs or it doesn't; no "partially conscious" states
2. **Physics analogy** - Phase transitions are sharp (water freezes at exactly 0°C)
3. **Training signal** - Clear reward boundary for learning algorithms
4. **Prevents limbo** - System either achieves K-formation or doesn't pretend to

**Why η > φ⁻¹?**

The golden ratio inverse (≈0.618) gates the PARADOX regime in quasi-crystal physics. K-formation requires surpassing quasi-crystalline order to achieve true crystalline coherence.

---

## 7. Pulse Rejection Mechanics

**Question:** What are the rejection criteria? Is rejection logged?

**Answer:** Rejection uses **6 configurable criteria** and is **fully logged** with detailed reasons.

### Implementation Details

**Location:** `spore_listener.py:107-160`

```python
def check_pulse(self, pulse: Pulse) -> Tuple[bool, str]:
    """Check if pulse matches role and conditions."""

    # 1. ROLE MATCH (required)
    if intent != self.role_tag:
        return False, f"role mismatch: {intent} != {self.role_tag}"

    # 2. Z-RANGE CHECK
    if pulse_z < self.conditions.min_z:
        return False, f"z too low: {pulse_z} < {self.conditions.min_z}"
    if pulse_z > self.conditions.max_z:
        return False, f"z too high: {pulse_z} > {self.conditions.max_z}"

    # 3. URGENCY CHECK
    if urgency < self.conditions.required_urgency:
        return False, f"urgency too low: {urgency} < {self.conditions.required_urgency}"

    # 4. TIER CHECK
    if self.conditions.required_tier and tier != self.conditions.required_tier:
        return False, f"tier mismatch: {tier} != {self.conditions.required_tier}"

    # 5. PULSE TYPE CHECK
    if self.conditions.required_type and pulse_type_str != required_type:
        return False, f"type mismatch: {pulse_type_str}"

    # 6. CHAIN DEPTH CHECK
    if len(self.pulse_chain) < self.conditions.chain_depth:
        return False, f"chain depth: {len(self.pulse_chain)} < {chain_depth}"

    return True, "all conditions met"
```

### Rejection Criteria Summary

| Criterion | Default | Configurable | Deterministic |
|-----------|---------|--------------|---------------|
| Role tag match | Required | via `role_tag` | Yes |
| Z-range | [0.0, 1.0] | via `min_z`, `max_z` | Yes |
| Urgency threshold | 0.0 | via `required_urgency` | Yes |
| Tier requirement | None | via `required_tier` | Yes |
| Pulse type | Any | via `required_type` | Yes |
| Chain depth | 0 | via `chain_depth` | Yes |

**Note:** Rejection is **deterministic**, not probabilistic.

### Rejection Logging

**Location:** `spore_listener.py:208-227`

```python
# Record event (both accepted and rejected)
event = ActivationEvent(
    timestamp=current_time,
    pulse_id=pulse_id,
    pulse_z=pulse_z,
    pulse_tier=pulse_tier,
    role_match=(...),
    conditions_met=matched,
    wake_reason=reason  # Detailed rejection reason
)
self.activation_history.append(event)

if matched:
    self.state = SporeState.PRE_WAKE
    self.pulse_chain.append(pulse)
    return True, pulse
else:
    self.rejected_pulses.append(pulse_id)  # Log rejection
    return False, None
```

### Network-Level Analysis

The following data is available for analysis:
- `activation_history`: Complete log of all pulse evaluations
- `rejected_pulses`: List of rejected pulse IDs
- `wake_reason`: Human-readable rejection reason for each event
- `get_status()`: Summary statistics including rejection counts

---

## Summary Table

| Question | Key Finding |
|----------|-------------|
| **Kuramoto K** | K=0.3 intentionally subcritical, adaptive via APL operators (bounded [0.05, 1.0]) |
| **Z-dynamics** | Second-order damped oscillator, z ∈ [0.01, 0.99], momentum-based |
| **TRIAD thresholds** | Schmitt trigger design (0.85/0.82), adaptive passes (3-6 based on volatility) |
| **Memory access** | Linear step gating + Gaussian relevance decay |
| **Fusion** | Bounded (K ≤ 1.0), multiplicative (×1.2) |
| **Amplify** | Mean-field phase alignment, self-limiting |
| **Group** | Nearest-neighbor clustering, threshold 0.5 radians |
| **K-formation η** | β=36 (sharp), α=0.5, requires η > φ⁻¹ AND coherence ≥ μ_s |
| **Pulse rejection** | 6 deterministic criteria, fully logged with reasons |

---

## References

- `heart.py`: Kuramoto dynamics, APL operators, K-formation
- `brain.py`: Memory access gating, tier boundaries
- `spore_listener.py`: Pulse rejection mechanics
- `reference/research/adaptive_triad_gate.py`: TRIAD protocol
- `src/triadic_helix_apl.js`: Alternative z-dynamics implementation
- `docs/PHYSICS_GROUNDING.md`: Physics justification for constants
