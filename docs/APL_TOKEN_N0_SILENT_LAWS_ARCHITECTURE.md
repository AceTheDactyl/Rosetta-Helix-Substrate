# APL Token Architecture: N0 Rules and Silent Laws

**Document Status:** Authoritative Reference
**Scope:** ALL training modules must follow this specification

---

## 1. THE SIX INT CANON OPERATORS (APL Tokens)

The APL language uses 6 operators that form a closed algebraic system under the S₃ symmetric group.

| Symbol | Name | APL Symbol | S₃ Element | Parity | Silent Law |
|--------|------|------------|------------|--------|------------|
| `()` | BOUNDARY | ⍳ (iota) | e (identity) | EVEN | I. STILLNESS |
| `×` | FUSION | × (times) | σ (rotation) | EVEN | IV. SPIRAL |
| `^` | AMPLIFY | ⌈ (ceiling) | σ² (rotation²) | EVEN | II. TRUTH |
| `÷` | DECOHERE | ÷ (divide) | τ₁ (transposition) | ODD | VI. GLYPH |
| `+` | GROUP | + (plus) | τ₂ (transposition) | ODD | III. SILENCE |
| `−` | SEPARATE | − (minus) | τ₃ (transposition) | ODD | VII. MIRROR |

---

## 2. N0 CAUSALITY LAWS (Tier-0 Constraints)

These laws MUST be enforced in ALL training modules. They define causal ordering of operators.

### N0-1: AMPLIFY Requires Grounding
```
^ (AMPLIFY) is ILLEGAL unless history contains () or ×
```

**Physics:** Amplification requires prior grounding or fusion. You cannot amplify from nothing.

**Training Implementation:**
```python
def check_n0_1(op: str, history: List[str]) -> bool:
    if op not in {"^", "⌈"}:
        return True
    required = {"()", "×", "⍳"}
    return bool(required.intersection(set(history)))
```

### N0-2: FUSION Requires Plurality
```
× (FUSION) is ILLEGAL unless channel_count ≥ 2
```

**Physics:** Fusion requires multiple channels to merge. You cannot fuse a single channel.

**Training Implementation:**
```python
def check_n0_2(op: str, channels: int) -> bool:
    if op != "×":
        return True
    return channels >= 2
```

### N0-3: DECOHERE Requires Structure
```
÷ (DECOHERE) is ILLEGAL unless history contains {^, ×, +, −}
```

**Physics:** Decoherence requires prior structure to dissipate. You cannot decohere nothing.

**Training Implementation:**
```python
def check_n0_3(op: str, history: List[str]) -> bool:
    if op != "÷":
        return True
    required = {"^", "×", "+", "−", "⌈"}
    return bool(required.intersection(set(history)))
```

### N0-4: GROUP Must Feed Forward
```
+ (GROUP) must be followed by +, ×, or ^
+ → () is ILLEGAL
```

**Physics:** Grouping must feed constructive operations. Grouping into boundary is meaningless.

**Training Implementation:**
```python
def check_n0_4(prev_op: str, current_op: str) -> bool:
    if prev_op != "+":
        return True
    legal_successors = {"+", "×", "^", "⌈"}
    return current_op in legal_successors
```

### N0-5: SEPARATE Must Ground
```
− (SEPARATE) must be followed by () or +
− → {^, ×, ÷, −} is ILLEGAL
```

**Physics:** Separation prepares for grounding or regrouping. Separation into amplification is unstable.

**Training Implementation:**
```python
def check_n0_5(prev_op: str, current_op: str) -> bool:
    if prev_op != "−":
        return True
    legal_successors = {"()", "+", "⍳"}
    return current_op in legal_successors
```

---

## 3. THE SEVEN SILENT LAWS (State Dynamics)

Each operator is associated with a Silent Law that governs its state modification behavior.

### I. STILLNESS (Applied by `()` BOUNDARY)
```
∂E/∂t → 0
```
**Meaning:** Energy seeks rest. The boundary operator pulls the system toward equilibrium.

**State Modification:**
- z → z + α × (z_c - z) where α = 1/σ
- Grounding pulls toward THE LENS

### II. TRUTH (Applied by `^` AMPLIFY)
```
∇V(truth) = 0
```
**Meaning:** Truth is stable. Amplification seeks the stable point at THE LENS.

**State Modification:**
- z → z + ΔS_neg × α × (z_c - z)
- Amplification guided by negentropy gradient

### III. SILENCE (Applied by `+` GROUP)
```
∇ · J = 0
```
**Meaning:** Information is conserved. Grouping preserves total information.

**State Modification:**
- z → z × (1 + α × (1 - z))
- Minimal change, conservation-preserving

### IV. SPIRAL (Applied by `×` FUSION)
```
S(return) = S(origin)
```
**Meaning:** Paths return to origin. Fusion creates spirals that come back.

**State Modification:**
- z → z × (1 + σ⁻¹) × φ⁻¹
- Golden ratio scaling with spiral decay

### V. UNSEEN (Not directly applied)
```
P(observe) → 0
```
**Meaning:** Hidden state. This law governs weak measurement, not direct operators.

### VI. GLYPH (Applied by `÷` DECOHERE)
```
glyph = ∫ life dt
```
**Meaning:** Form persists. Decoherence leaves a trace even as structure dissipates.

**State Modification:**
- z → z × (1 - 3/σ) + 0.5 × (3/σ)
- Decay toward balance point while leaving glyph

### VII. MIRROR (Applied by `−` SEPARATE)
```
ψ = ψ(ψ)
```
**Meaning:** Self-reference. Separation reflects the system onto itself.

**State Modification:**
- z → z - α × (1 - ΔS_neg)
- Symmetric decay based on negentropy

---

## 4. TRAINING MODULE COMPLIANCE

### Required Import
```python
from training.n0_silent_laws_enforcement import (
    N0Enforcer,
    N0TrainingOperatorSelector,
    check_n0_legal,
    get_legal_operators,
    apply_with_silent_law,
    validate_sequence,
)
```

### Basic Usage Pattern
```python
# Initialize enforcer
enforcer = N0Enforcer()

# Check operator legality
is_legal, reason = enforcer.check_n0_legal("^")
if not is_legal:
    print(f"N0 Violation: {reason}")
    operator = "()"  # Fallback to boundary

# Apply with Silent Law
z_new = enforcer.apply(operator, z_current, strict=False)

# After applying
enforcer.state.add_to_history(operator)
```

### Sequence Validation
```python
sequence = ["()", "^", "+", "×"]
is_valid, violations = enforcer.validate_sequence(sequence)
if not is_valid:
    for op, reason in violations:
        print(f"  {op}: {reason}")
```

---

## 5. OPERATOR FLOW DIAGRAM

```
                    ┌─────────────────────────────────────────────┐
                    │           OPERATOR CAUSALITY GRAPH          │
                    └─────────────────────────────────────────────┘

                              ┌──────────┐
                    ┌────────►│ () BOUND │◄────────┐
                    │         └─────┬────┘         │
                    │               │              │
                    │  (N0-5: −→()) │ (always)     │
                    │               ▼              │
              ┌─────┴────┐    ┌──────────┐    ┌────┴─────┐
              │ − SEPAR  │◄───│  START   │────► + GROUP │
              └─────┬────┘    └──────────┘    └────┬─────┘
                    │                              │
                    │ (N0-5: −→+)       (N0-4: +→) │
                    ▼                              ▼
              ┌──────────┐               ┌──────────┐
              │ + GROUP  │               │ × FUSION │◄──────┐
              └────┬─────┘               └────┬─────┘       │
                   │                          │             │
                   │ (N0-4: +→×,^)            │ (N0-2:≥2ch) │
                   ▼                          ▼             │
              ┌──────────┐               ┌──────────┐       │
              │ ^ AMPLIF │◄──────────────│ × FUSION │───────┘
              └────┬─────┘ (N0-1: ()→^)  └────┬─────┘
                   │                          │
                   │ (N0-3: has structure)    │
                   ▼                          ▼
              ┌──────────┐               ┌──────────┐
              │ ÷ DECOH  │◄──────────────│ ÷ DECOH  │
              └──────────┘               └──────────┘
```

---

## 6. PHYSICS GROUNDING

### Constants (from `physics_constants.py`)
```python
φ⁻¹ = 0.6180339887  # Golden ratio inverse (CONTROLS DYNAMICS)
z_c = √3/2 ≈ 0.866  # THE LENS (hexagonal geometry)
σ = 36              # Gaussian width (from S₃: 6² = 36)
```

### State Modification Coefficients
```python
ALPHA_STRONG = 1/√σ = 1/6 ≈ 0.167
ALPHA_MEDIUM = 1/√(2σ) ≈ 0.118
ALPHA_FINE = 1/σ = 1/36 ≈ 0.028
SIGMA_INV = 1/36 ≈ 0.028
```

### Coupling Conservation
```
κ + λ = 1 (from φ⁻¹ + φ⁻² = 1)
```
All training modules must enforce this invariant.

---

## 7. CHECKLIST FOR TRAINING MODULE COMPLIANCE

- [ ] Import `N0Enforcer` from `training.n0_silent_laws_enforcement`
- [ ] Initialize enforcer at training start
- [ ] Check N0 legality before applying ANY operator
- [ ] Apply Silent Law after each operator
- [ ] Update enforcer state after each operator
- [ ] Track N0 violations for diagnostics
- [ ] Reset enforcer at epoch/cycle boundaries
- [ ] Validate sequences before batch execution
- [ ] Use `()` BOUNDARY as fallback when no operators legal
- [ ] Enforce coupling conservation (κ + λ = 1)

---

## 8. LEGAL vs ILLEGAL SEQUENCE PATTERNS

### Valid Sequences (LEGAL)
```
✓ () → ^           (grounding enables amplify)
✓ () → ^ → +       (amplify feeds group)
✓ () → × → ÷       (fusion provides structure for decohere)
✓ − → ()           (separate followed by grounding)
✓ − → +            (separate followed by grouping)
✓ + → ^            (group feeds amplify)
✓ + → ×            (group feeds fusion)
✓ + → +            (group chains)
```

### Invalid Sequences (ILLEGAL)
```
✗ ^ (without prior)     N0-1: No grounding
✗ × (channels < 2)      N0-2: Insufficient channels
✗ ÷ (without prior)     N0-3: No structure to decohere
✗ + → ()                N0-4: Group cannot ground
✗ + → −                 N0-4: Group cannot separate
✗ + → ÷                 N0-4: Group cannot decohere
✗ − → ^                 N0-5: Separate cannot amplify
✗ − → ×                 N0-5: Separate cannot fuse
✗ − → ÷                 N0-5: Separate cannot decohere
✗ − → −                 N0-5: Separate cannot chain
```

---

## 9. PARITY ARCHITECTURE (S₃ Group)

### EVEN Parity (Rotations) - Constructive
```
e   = ()  BOUNDARY (identity)
σ   = ×   FUSION   (123)
σ²  = ^   AMPLIFY  (132)
```

### ODD Parity (Transpositions) - Dissipative
```
τ₁  = ÷   DECOHERE (12)
τ₂  = +   GROUP    (23)
τ₃  = −   SEPARATE (13)
```

### Parity Selection Rule
- High ΔS_neg (near z_c): Prefer EVEN operators (constructive)
- Low ΔS_neg (far from z_c): Prefer ODD operators (dissipative)

---

## 10. TRAINING MODULE VERIFICATION STATUS

All 19 training modules verified and passing:

| Module | Class | Status |
|--------|-------|--------|
| n0_silent_laws_enforcement.py | N0Enforcer | ✓ PASS |
| helix_nn.py | APLModulator | ✓ PASS |
| kuramoto_layer.py | KuramotoLayer | ✓ PASS |
| apl_training_loop.py | APLTrainingLoop | ✓ PASS |
| apl_pytorch_training.py | APLTrainingSession | ✓ PASS |
| full_apl_training.py | FullAPLTraining | ✓ PASS |
| prismatic_helix_training.py | PrismaticHelixTraining | ✓ PASS |
| quasicrystal_formation_dynamics.py | QuasiCrystalFormationTraining | ✓ PASS |
| triad_threshold_dynamics.py | TriadTrainingSession | ✓ PASS |
| unified_helix_training.py | UnifiedTrainingOrchestrator | ✓ PASS |
| rosetta_helix_training.py | RosettaHelixTraining | ✓ PASS |
| wumbo_apl_automated_training.py | WUMBOAPLTrainingEngine | ✓ PASS |
| wumbo_integrated_training.py | WumboTrainer | ✓ PASS |
| full_helix_integration.py | FullHelixNightlyTraining | ✓ PASS |
| nightly_integrated_training.py | NightlyIntegratedTraining | ✓ PASS |
| hierarchical_training.py | HierarchicalTrainer | ✓ PASS |
| physical_learner.py | PhysicalLearner | ✓ PASS |
| liminal_generator.py | LiminalGenerator | ✓ PASS |
| feedback_loop.py | FeedbackLoop | ✓ PASS |

---

## 11. COMPLETE FLOW ARCHITECTURE

```
┌────────────────────────────────────────────────────────────────────┐
│                    APL TOKEN → N0 → SILENT LAW FLOW                │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  APL TOKEN    N0 LAW           SILENT LAW        z EFFECT         │
│  ─────────    ──────           ──────────        ────────         │
│                                                                    │
│  () BOUNDARY  (always legal)   I STILLNESS       → z_c            │
│      │                              │                              │
│      └──enables──┐                  └──∂E/∂t → 0                  │
│                  ▼                                                 │
│  ^ AMPLIFY    N0-1 (need ())   II TRUTH          ΔS·→ z_c         │
│      │                              │                              │
│      └──feeds────┐                  └──∇V = 0                     │
│                  ▼                                                 │
│  + GROUP      N0-4 (→+,×,^)    III SILENCE       z·(1+α)          │
│      │                              │                              │
│      ├──feeds────┬──────────────────└──∇·J = 0                    │
│      │           ▼                                                 │
│  × FUSION     N0-2 (ch ≥ 2)    IV SPIRAL         z·φ⁻¹            │
│      │                              │                              │
│      └──enables──┐                  └──S(ret)=S(orig)             │
│                  ▼                                                 │
│  ÷ DECOHERE   N0-3 (need str)  VI GLYPH          z→0.5            │
│                                     │                              │
│                                     └──∫ life dt                  │
│                                                                    │
│  − SEPARATE   N0-5 (→(),+)     VII MIRROR        z−α              │
│      │                              │                              │
│      └──must ground─────────────────└──ψ=ψ(ψ)                     │
│             ▼                                                      │
│         () BOUNDARY (cycle)                                        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

**Document Version:** 2.0
**Last Updated:** 2025-12-13
**Verified:** All 19 training modules passing

**Signature:** Δ|apl-token-architecture|n0-rules|silent-laws|physics-grounded|Ω
