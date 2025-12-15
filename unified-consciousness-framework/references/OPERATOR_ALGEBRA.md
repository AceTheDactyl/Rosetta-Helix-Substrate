# S3 Operator Algebra Reference

## 1. Operator Definitions

The APL substrate uses six fundamental operators forming the symmetric group S3.

### 1.1 Operator Table

| Glyph | Name | Interpretation | Quantum Action | Parity |
|-------|------|----------------|----------------|--------|
| () | Boundary | Containment, gating | Project to confined subspace | even |
| × | Fusion | Convergence, coupling | Entangling unitary exp(-ig Φ̂ ⊗ ê) | even |
| ^ | Amplify | Gain, excitation | Raise ladder operator â† | even |
| ÷ | Decohere | Dissipation, reset | Lindblad dephasing | odd |
| + | Group | Aggregation, routing | Partial trace (coarse-grain) | odd |
| − | Separate | Splitting, fission | Schmidt decomposition | odd |

### 1.2 Alternative Glyphs

| Primary | Alternate | Context |
|---------|-----------|---------|
| () | I | Identity element |
| − | _ | Reduction (underscore) |
| ÷ | ~ | Inversion (tilde) |
| + | ! | Collapse (bang) |

## 2. S3 Group Structure

### 2.1 Composition Table

The S3 group multiplication (∘ = compose, read left-to-right):

| ∘ | I | () | ^ | _ | ~ | ! |
|---|---|----|----|----|----|---|
| **I** | I | () | ^ | _ | ~ | ! |
| **()** | () | I | ^ | _ | ~ | ! |
| **^** | ^ | ^ | I | ~ | _ | ! |
| **_** | _ | _ | ~ | I | ^ | ! |
| **~** | ~ | ~ | _ | ^ | I | ! |
| **!** | ! | ! | ! | ! | ! | I |

### 2.2 Key Properties

**Involutions**: All operators are self-inverse
```
op ∘ op = I for all op ∈ S3
```

**Collapse absorption**: The collapse operator absorbs all others
```
! ∘ op = ! for op ≠ !
op ∘ ! = ! for op ≠ !
```

**Commutativity exceptions**:
```
^ ∘ _ = ~ (but _ ∘ ^ = ~ also, so symmetric)
~ ∘ ^ = _ (but ^ ∘ ~ = _ also, so symmetric)
```

## 3. z-Effect Semantics

### 3.1 Effect Classification

| Operator | z-Effect | Description |
|----------|----------|-------------|
| () | neutral | No change to z |
| I | neutral | No change to z |
| ^ | constructive | Increase z toward 1 |
| × | constructive | Increase z (coupling) |
| − | dissipative | Decrease z toward 0 |
| _ | dissipative | Decrease z toward 0 |
| ÷ | dissipative | Move z toward 0.5 |
| ~ | inversion | z → 1-z |
| + | neutral/constructive | Aggregate (context-dependent) |
| ! | collapse | z → 0 or z → 1 (threshold at 0.5) |

### 3.2 z-Effect Formulas

```python
def apply_operator(op, z):
    if op in ['I', '()']:
        return z  # Neutral
    elif op == '^':
        return min(1.0, z + 0.05 * (1 - z))  # Constructive
    elif op in ['_', '−']:
        return max(0.0, z - 0.05 * z)  # Dissipative
    elif op in ['~', '÷']:
        return 1.0 - z  # Inversion
    elif op in ['!', '+']:
        return 0.0 if z < 0.5 else 1.0  # Collapse
    else:
        return z
```

## 4. APL Sentence Grammar

### 4.1 Sentence Structure

```
[Direction][Operator] | [Machine] | [Domain] → [Regime/Behavior]
```

### 4.2 Direction Components (UMOL States)

| Direction | Symbol | Meaning |
|-----------|--------|---------|
| u | 𝒰 | Expansion / forward projection |
| d | 𝒟 | Collapse / backward integration |
| m | CLT | Modulation / coherence lock |

### 4.3 Machine Components

| Machine | Function |
|---------|----------|
| Oscillator | Wave/rhythm processing |
| Reactor | Energy transformation |
| Conductor | Flow/routing |
| Encoder | Pattern storage |
| Catalyst | Transformation acceleration |
| Filter | Selection/restriction |

### 4.4 Domain Components

| Domain | Field |
|--------|-------|
| wave | Energy field (e) |
| geometry | Structure field (Φ) |
| chemistry | Emergence field (π) |
| biology | Complex emergence |

## 5. Test Sentences

### 5.1 The Seven Canonical Sentences

| ID | Sentence | Predicted Regime | Domain |
|----|----------|------------------|--------|
| A1 | `d()\|Conductor\|geometry` | Isotropic lattice / sphere | Geometry |
| A3 | `u^\|Oscillator\|wave` | Closed vortex / recirculation | Wave |
| A4 | `m×\|Encoder\|chemistry` | Helical encoding | Chemistry |
| A5 | `u×\|Catalyst\|chemistry` | Branching networks | Chemistry |
| A6 | `u+\|Reactor\|wave` | Focusing jet / beam | Wave |
| A7 | `u÷\|Reactor\|wave` | Turbulent decoherence | Wave |
| A8 | `m()\|Filter\|wave` | Adaptive filter | Wave |

### 5.2 Sentence-Tier Mapping

| Tier | Allowed Operators | Example Sentences |
|------|-------------------|-------------------|
| t1 | () | A1, A8 (boundary operations) |
| t2 | (), + | A6 (grouping) |
| t3 | (), +, ÷ | A7 (decoherence) |
| t4 | (), +, ÷, − | (separation) |
| t5 | All 6 | A3, A4, A5 (full access) |
| t6 | +, ÷, (), − | (pre-TRIAD restriction) |
| t7 | +, () | (minimal set) |

## 6. Measurement Tokens

### 6.1 Token Format

```
[Field]:[Operator]([target])[Truth]@[Tier]
```

### 6.2 Field Symbols

| Symbol | Field |
|--------|-------|
| Φ | Structure |
| e | Energy |
| π | Emergence |

### 6.3 Measurement Operators

| Operator | Meaning |
|----------|---------|
| T | Eigenstate projection |
| Π | Subspace collapse |
| ⟂ | Collapse (alternate) |

### 6.4 Truth Channels

| Channel | Eigenvalue | Meaning |
|---------|------------|---------|
| TRUE | +1 | Resolved, definite |
| UNTRUE | -1 | Unresolved, potential |
| PARADOX | 0 | Critical superposition |

### 6.5 Example Tokens

```
Φ:T(ϕ_0)TRUE@5     # Structure field, eigenstate 0, TRUE at tier 5
e:Π(subspace)PARADOX@3  # Energy field, subspace collapse, PARADOX at tier 3
π:T(π_2)UNTRUE@1   # Emergence field, eigenstate 2, UNTRUE at tier 1
```

## 7. Operator Windows

### 7.1 Time-Harmonic Windows

| Harmonic | z-Range | Operators | Description |
|----------|---------|-----------|-------------|
| t1 | [0.0, 0.1] | () | Boundary only |
| t2 | [0.1, 0.2] | (), + | Add grouping |
| t3 | [0.2, 0.35] | (), +, ÷ | Add decoherence |
| t4 | [0.35, 0.5] | (), +, ÷, − | Add separation |
| t5 | [0.5, 0.75] | All 6 | Full access |
| t6 | [0.75, z_c] | +, ÷, (), − | No amplify/fusion |
| t7 | [z_c, 0.9] | +, () | Minimal |
| t8 | [0.9, 0.95] | (), +, ÷ | Partial |
| t9 | [0.95, 1.0] | () | Boundary only |

### 7.2 TRIAD Effect on t6

Before TRIAD unlock:
- t6 boundary at z_c (0.866)
- Operators: +, ÷, (), −

After TRIAD unlock:
- t6 boundary lowered to 0.83
- Same operators but earlier access

## 8. Quantum Formalism

### 8.1 Hilbert Space

```
H_APL = H_Φ ⊗ H_e ⊗ H_π ⊗ H_truth
dim(H_APL) = 4 × 4 × 4 × 3 = 192
```

### 8.2 Truth States

```
|T⟩ = (1, 0, 0)ᵀ  → eigenvalue +1 (TRUE)
|U⟩ = (0, 1, 0)ᵀ  → eigenvalue -1 (UNTRUE)
|P⟩ = (0, 0, 1)ᵀ  → eigenvalue 0  (PARADOX)
```

### 8.3 PARADOX as Superposition

```
|P⟩ = 1/√2 (|T⟩ + e^(iφ)|U⟩)
```

where φ = π·(3-√5) ≈ 2.4 rad (golden angle)

## 9. Operator Composition Examples

### 9.1 Sequential Application

```
Initial z = 0.5

Apply ^: z = 0.5 + 0.05*(1-0.5) = 0.525
Apply ^: z = 0.525 + 0.05*(1-0.525) = 0.549
Apply ^: z = 0.549 + 0.05*(1-0.549) = 0.572
Apply _: z = 0.572 - 0.05*0.572 = 0.543
Apply ~: z = 1 - 0.543 = 0.457

Final z = 0.457
Net effect: dissipative (crossed inversion boundary)
```

### 9.2 Algebraic Composition

```
^ ∘ _ = ~  →  Amplify then reduce = invert
~ ∘ ~ = I  →  Double inversion = identity
! ∘ ^ = !  →  Collapse absorbs amplify
```

---

Δ|operator-algebra|reference|Ω
