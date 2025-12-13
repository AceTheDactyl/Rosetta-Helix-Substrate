# Rosetta Helix Development Build Specification

## CRITICAL PHYSICS INVARIANTS

**READ THIS FIRST. VIOLATION = SYSTEM BREAK.**

### The Core Rule
```
PHI (1.618)     = LIMINAL ONLY. Never drives dynamics. Never becomes physical.
PHI_INV (0.618) = ALWAYS controls physical dynamics.
```

### Why This Matters
If PHI_INV "flips" at unity (z >= 1.0) and PHI becomes the dominant ratio:
- Infinite entropy expansion
- NegEntropyEngine breaks
- System diverges catastrophically

### The Solution: Instant Collapse
```
At z >= 0.9999:
  1. INSTANT collapse (not gradual decay)
  2. Extract work via weak value: work = (z - Z_CRITICAL) * PHI * PHI_INV
  3. Reset to origin: z = Z_CRITICAL * PHI_INV (~0.535)
  4. PHI contributed via weak value at collapse, never during evolution
```

### What PHI Can Do
- Contribute to work extraction via weak values AT COLLAPSE
- Exist in superposition (liminal state)
- Appear in weak value formulas: `work = ... * PHI * PHI_INV`

### What PHI Must NEVER Do
- Drive dynamics: `dz = PHI * ...` **WRONG**
- Be a target: `target = PHI` **WRONG**
- Cap values: `min(PHI, x)` **WRONG**
- Decay PHI_INV: `PHI_INV -= decay` **CATASTROPHIC**

---

## PHASE 1: Core Physics Engine

**Goal:** Establish the foundational physics that all tools use.

### Files to Create
1. `core/constants.py` - Physics constants with documentation
2. `core/collapse_engine.py` - The instant collapse mechanism
3. `core/liminal_state.py` - PHI superposition handling

### Key Implementation

```python
# core/constants.py
PHI = (1.0 + math.sqrt(5.0)) / 2.0      # 1.618 - LIMINAL ONLY
PHI_INV = 1.0 / PHI                      # 0.618 - ALWAYS PHYSICAL
Z_CRITICAL = math.sqrt(3.0) / 2.0        # 0.866 - Hexagonal geometry
KAPPA_S = 0.920                          # Superposition entry
MU_3 = 0.992                             # Ultra-integration threshold
UNITY = 0.9999                           # Collapse trigger (NOT 1.0, NOT PHI)

# NEVER CHANGE THESE RELATIONSHIPS
```

```python
# core/collapse_engine.py
class CollapseEngine:
    def evolve(self, z: float, work_input: float) -> tuple[float, float]:
        """
        Evolve z-coordinate. PHI_INV ALWAYS controls.
        Returns (new_z, work_extracted)
        """
        # PHI_INV drives evolution
        dz = work_input * PHI_INV
        z_new = z + dz

        # INSTANT COLLAPSE at unity
        if z_new >= UNITY:
            work = (z_new - Z_CRITICAL) * PHI * PHI_INV  # PHI in extraction only
            z_new = Z_CRITICAL * PHI_INV  # Reset to origin
            return z_new, work

        return min(UNITY - 0.0001, z_new), 0.0
```

### Verification Test
```python
def test_phi_never_drives():
    """PHI must never appear in dz calculations"""
    engine = CollapseEngine()
    for _ in range(1000):
        z, work = engine.evolve(z, 0.1)
        assert z < 1.0, "z must never exceed unity"
        assert z < PHI, "z must never approach PHI"
```

---

## PHASE 2: Tool Generation Framework

**Goal:** Meta-tools that produce child tools using collapse physics.

### Architecture
```
MetaTool (uses CollapseEngine)
    │
    ├── pumps work into mini-collapse
    ├── at collapse: extracts work
    └── work converts to ChildTool
```

### Files to Create
1. `tools/meta_tool.py` - Tool that produces tools
2. `tools/child_tool.py` - Produced tool with capabilities
3. `tools/tool_types.py` - Enums for tool categories

### Key Implementation

```python
# tools/meta_tool.py
class MetaTool:
    def __init__(self):
        self.collapse = CollapseEngine()
        self.z = 0.5
        self.work_accumulated = 0.0

    def pump(self, work: float) -> Optional[ChildTool]:
        """Pump work, potentially produce child tool at collapse"""
        self.z, extracted = self.collapse.evolve(self.z, work * PHI_INV)

        if extracted > 0:
            # Collapse happened - produce tool
            tool = ChildTool(work_invested=extracted)
            return tool

        self.work_accumulated += work * PHI_INV
        return None
```

---

## PHASE 3: Training Loop

**Goal:** Exponential learning through feedback cycles.

### Architecture
```
Physical (PHI_INV) ──feedback──> MetaMeta ──spawn──> Liminal (PHI)
       ↑                                                   │
       └──────────── weak measurement ─────────────────────┘
```

### Key Rules
1. Physical learners use `dominant_ratio = PHI_INV`
2. Liminal patterns stay `in_superposition = True` always
3. Cross-level coupling caps at 0.9 (NEVER PHI)
4. If coupling >= 1.0: instant collapse to Z_CRITICAL

### Files to Create
1. `training/feedback_loop.py` - The exponential training cycle
2. `training/physical_learner.py` - PHI_INV controlled learner
3. `training/liminal_generator.py` - PHI superposition patterns

---

## PHASE 4: Hierarchical Training

**Goal:** Multi-level lesson extraction and weight updates.

### Key Rules
```python
# Cross-level coupling - CORRECT
delta = learning_rate * ratio * PHI_INV  # PHI_INV controls
coupling = min(0.9, coupling + delta)     # Cap at 0.9, NOT PHI

if coupling >= 1.0:
    coupling = Z_CRITICAL  # INSTANT collapse, not gradual decay
```

### What NOT to Do
```python
# WRONG - PHI_INV decay
if coupling > 1.0:
    decay = excess * PHI_INV  # NO! Don't weaken PHI_INV
    coupling -= decay

# WRONG - Cap at PHI
coupling = min(PHI, coupling)  # NO! Never approach PHI
```

---

## PHASE 5: Unified Engine

**Goal:** Integrate all systems into coherent whole.

### Components
1. NegEntropyEngine (T1) - stays ACTIVE always
2. CollapseEngine - instant collapse at unity
3. MetaToolGenerator - produces tools from work
4. TrainingLoop - exponential learning

### Key State
```python
class EngineState:
    z_current: float = 0.5
    supercritical_mode: bool = True
    neg_entropy_active: bool = True  # NEVER turns off

    # Caps and limits
    z_max = 0.9999  # NEVER exceed unity
    coupling_max = 0.9  # NEVER approach PHI
```

---

## PHASE 6: Dev Tools (7-Layer Prismatic)

**Goal:** Specialized tools across spectral layers.

### Layers
| Layer | Color | Function | PHI Weight | PHI_INV Weight |
|-------|-------|----------|------------|----------------|
| 1 | RED | Analyzers | 0.8 | 1.2 |
| 2 | ORANGE | Learners | 0.9 | 1.1 |
| 3 | YELLOW | Generators | 1.0 | 1.0 |
| 4 | GREEN | Reflectors | 1.0 | 1.0 |
| 5 | BLUE | Builders | 1.1 | 0.9 |
| 6 | INDIGO | Deciders | 1.2 | 0.8 |
| 7 | VIOLET | Probers | 1.3 | 0.7 |

**Note:** PHI weight affects work EXTRACTION at collapse, not dynamics.

---

## ANTI-PATTERNS (DO NOT DO)

### 1. PHI Driving Dynamics
```python
# WRONG
expansion = PHI * weight * (target - z)
dz += PHI * factor
```

### 2. Targeting PHI
```python
# WRONG
target = PHI  # 1.618
if z < PHI:
    z += expansion
```

### 3. PHI_INV Decay
```python
# WRONG - This weakens the safety mechanism
if coupling > 1.0:
    phi_inv_contribution -= decay
```

### 4. Gradual Supercritical Decay
```python
# WRONG - Allows PHI to become dominant over time
if z > 1.0:
    entropy_return = excess * factor
    z -= entropy_return  # Gradual, not instant
```

### 5. Caps at PHI
```python
# WRONG
z = min(PHI, z)
coupling = min(PHI, coupling)
```

---

## CORRECT PATTERNS

### 1. PHI_INV Drives All Dynamics
```python
dz = work * PHI_INV
expansion = PHI_INV * weight * (MU_3 - z)
coupling_delta = learning_rate * PHI_INV
```

### 2. Instant Collapse at Unity
```python
if z >= 0.9999:
    work = (z - Z_CRITICAL) * PHI * PHI_INV
    z = Z_CRITICAL * PHI_INV  # ~0.535
```

### 3. PHI Only in Weak Value Extraction
```python
# At collapse only:
work_extracted = accumulated * PHI * PHI_INV
if in_superposition:
    work *= PHI  # Liminal boost at extraction
```

### 4. Safe Caps
```python
z = min(0.9999, z)  # Never unity
coupling = min(0.9, coupling)  # Never approach PHI
```

---

## BUILD ORDER

1. **Phase 1:** Core physics (constants, collapse engine)
2. **Phase 2:** Tool framework (meta-tool, child tool)
3. **Phase 3:** Training loop (feedback, learners)
4. **Phase 4:** Hierarchical training (multi-level)
5. **Phase 5:** Unified engine (integration)
6. **Phase 6:** Dev tools (prismatic layers)

Each phase must pass physics verification before proceeding:
```python
def verify_physics():
    assert z_max < 1.0
    assert coupling_max < PHI
    assert "PHI *" not in dynamics_code
    assert phi_inv_never_decays()
```

---

## SUMMARY

```
PHI = liminal, superposition, weak values at collapse
PHI_INV = physical, dynamics, always dominant

At unity: INSTANT collapse, not gradual decay
Caps: 0.9999 for z, 0.9 for coupling (NEVER PHI)

If unsure: use PHI_INV
If tempted to use PHI in dynamics: DON'T
```
