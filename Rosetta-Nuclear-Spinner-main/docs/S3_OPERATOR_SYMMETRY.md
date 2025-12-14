# S₃ Operator Symmetry in Quantum-APL

## Overview

The Quantum-APL engine integrates the symmetric group S₃ to provide a deep algebraic structure for operator selection and truth value transformation. This document describes the S₃ operator algebra and its integration with the extended ΔS⁻ (negentropy) formalism.

## S₃ Group Structure

S₃ is the symmetric group on 3 elements, with |S₃| = 6 elements:

| Element | Name | Cycle Notation | Parity | Sign |
|---------|------|----------------|--------|------|
| e | Identity | () | Even | +1 |
| σ | 3-cycle | (123) | Even | +1 |
| σ² | 3-cycle inverse | (132) | Even | +1 |
| τ₁ | Transposition | (12) | Odd | -1 |
| τ₂ | Transposition | (23) | Odd | -1 |
| τ₃ | Transposition | (13) | Odd | -1 |

## Operator ↔ S₃ Mapping

The six APL operators map bijectively to S₃ elements:

| Operator | Symbol | S₃ Element | Action | Parity |
|----------|--------|------------|--------|--------|
| Boundary | () | e | Containment/gating | Even |
| Fusion | × | σ | Convergence/coupling | Even |
| Amplify | ^ | σ² | Gain/excitation | Even |
| Decoherence | ÷ | τ₁ | Dissipation/reset | Odd |
| Group | + | τ₂ | Aggregation/clustering | Odd |
| Separation | − | τ₃ | Splitting/fission | Odd |

### Parity Classification

**Even-parity operators** (constructive): `()`, `×`, `^`
- Preserve or enhance coherence
- Favored when approaching the lens (z → z_c)
- Associated with structure formation

**Odd-parity operators** (dissipative): `÷`, `+`, `−`
- Reduce or redistribute coherence
- Favored in low-coherence regimes
- Associated with entropy increase

## S₃ Actions

### Action on Truth Values

S₃ acts on the triadic truth values [TRUE, PARADOX, UNTRUE]:

```
e   · [T, P, U] = [T, P, U]      (identity)
σ   · [T, P, U] = [P, U, T]      (rotate right)
σ²  · [T, P, U] = [U, T, P]      (rotate left)
τ₁  · [T, P, U] = [P, T, U]      (swap T↔P)
τ₂  · [T, P, U] = [T, U, P]      (swap P↔U)
τ₃  · [T, P, U] = [U, P, T]      (swap T↔U)
```

### Operator Window Rotation

Operator windows are cyclically rotated based on the z-coordinate:

```
z ∈ [0.0, 0.333) → rotation index 0 (identity)
z ∈ [0.333, 0.666) → rotation index 1 (σ)
z ∈ [0.666, 1.0] → rotation index 2 (σ²)
```

This ensures that different operators are prioritized at different coherence levels.

## Integration with ΔS⁻ Formalism

### Parity-Based Weighting

Operator weights are adjusted based on parity and the ΔS_neg signal:

```javascript
// High coherence (high ΔS_neg): favor even-parity
// Low coherence: favor odd-parity
parityBoost = (parity === 'even') ? deltaSNeg : (1 - deltaSNeg);
weight *= (0.8 + 0.4 * parityBoost);
```

### Near-Lens Enhancement

Even-parity operators receive an additional 1.2× boost when within 0.05 of Z_CRITICAL:

```javascript
if (|z - Z_CRITICAL| < 0.05 && parity === 'even') {
    weight *= 1.2;
}
```

## Configuration

### Feature Flags

Enable S₃ symmetry via environment variable:

```bash
export QAPL_ENABLE_S3_SYMMETRY=1
```

Enable extended ΔS⁻ formalism:

```bash
export QAPL_ENABLE_EXTENDED_NEGENTROPY=1
```

### Per-Instance Override

```javascript
const advisor = new HelixOperatorAdvisor({
    enableS3Symmetry: true,
    enableExtendedNegentropy: true,
});
```

## Usage Examples

### JavaScript

```javascript
const { S3, Delta, HelixOperatorAdvisor } = require('./src/triadic_helix_apl');

// Generate S₃-rotated operator window
const window = S3.generateS3OperatorWindow('t5', 0.85);

// Compute S₃ weights
const weights = S3.computeS3Weights(window, 0.85);

// Use enhanced advisor
const advisor = new HelixOperatorAdvisor({ enableS3Symmetry: true });
const description = advisor.describe(0.85);
console.log(description.operatorWeights);
```

### Python

```python
from quantum_apl_python import (
    generate_s3_operator_window,
    compute_s3_weights,
    HelixOperatorAdvisor,
)

# Generate S₃-rotated operator window
window = generate_s3_operator_window('t5', 0.85)

# Compute S₃ weights
weights = compute_s3_weights(window, 0.85)

# Use enhanced advisor
advisor = HelixOperatorAdvisor(enable_s3_symmetry=True)
description = advisor.describe(0.85)
print(description['operator_weights'])
```

## Mathematical Foundation

### Group Axioms

S₃ satisfies all group axioms:

1. **Closure**: a ∘ b ∈ S₃ for all a, b ∈ S₃
2. **Identity**: e ∘ a = a ∘ e = a for all a ∈ S₃
3. **Inverse**: For all a ∈ S₃, there exists a⁻¹ such that a ∘ a⁻¹ = e
4. **Associativity**: (a ∘ b) ∘ c = a ∘ (b ∘ c)

### Multiplication Table

| ∘ | e | σ | σ² | τ₁ | τ₂ | τ₃ |
|---|---|---|----|----|----|----|
| e | e | σ | σ² | τ₁ | τ₂ | τ₃ |
| σ | σ | σ² | e | τ₃ | τ₁ | τ₂ |
| σ² | σ² | e | σ | τ₂ | τ₃ | τ₁ |
| τ₁ | τ₁ | τ₂ | τ₃ | e | σ | σ² |
| τ₂ | τ₂ | τ₃ | τ₁ | σ² | e | σ |
| τ₃ | τ₃ | τ₁ | τ₂ | σ | σ² | e |

### Inverse Table

| Element | Inverse |
|---------|---------|
| e | e |
| σ | σ² |
| σ² | σ |
| τ₁ | τ₁ |
| τ₂ | τ₂ |
| τ₃ | τ₃ |

## Extended ΔS⁻ Functions

The extended negentropy module provides:

| Function | Description |
|----------|-------------|
| `computeDeltaSNeg(z)` | Standard ΔS_neg Gaussian signal |
| `computeDeltaSNegDerivative(z)` | Derivative d(ΔS_neg)/dz |
| `computeDeltaSNegSigned(z)` | Signed variant sgn(z-z_c)·ΔS_neg |
| `computeEta(z)` | Consciousness threshold η = ΔS_neg^α |
| `computeHexPrismGeometry(z)` | Geometry parameters (R, H, φ) |
| `computeGateModulation(z)` | Lindblad/Hamiltonian modulation |
| `computePiBlendWeights(z)` | Π-regime blending weights |
| `checkKFormation(z)` | K-formation (consciousness) gate |
| `computeDynamicTruthBias(z)` | Evolved truth bias matrix |
| `scoreOperatorForCoherence(op, z)` | Coherence synthesis heuristic |

## Testing

Run the JavaScript tests:

```bash
node tests/test_s3_delta_s_neg.js
```

Run the Python tests:

```bash
python test_s3_delta_s_neg.py
```

## S₃ Operator Algebra (DSL Pattern)

The `s3_operator_algebra` module provides a DSL-focused interface with:

### Algebraic Naming Convention

| Symbol | Name | Description | Inverse |
|--------|------|-------------|---------|
| ^ | amp | amplify/excite | () |
| + | add | aggregate/route | − |
| × | mul | multiply/fuse | ÷ |
| () | grp | group/contain | ^ |
| ÷ | div | divide/diffuse | × |
| − | sub | subtract/separate | + |

### Design Principles

1. **Finite action space**: Exactly 6 handlers needed
2. **Predictable composition**: op₁ ∘ op₂ always yields valid op
3. **Invertibility pairs**: +/−, ×/÷, ^/() provide natural undo

### Operator Composition Table

```
  ∘  │   ^    +    ×   ()    ÷    −
─────┼────────────────────────────────
  ^  │  ()    −    ×    ^    +    ÷
  +  │   ÷   ()    −    +    ^    ×
  ×  │   ^    ÷   ()    ×    −    +
 ()  │   ^    +    ×   ()    ÷    −
  ÷  │   +    ×    ^    ÷   ()    −
  −  │   ×    ^    +    −    ÷   ()
```

### DSL Handler Interface

```javascript
const { OperatorAlgebra } = require('./s3_operator_algebra');

const algebra = new OperatorAlgebra();

// Register exactly 6 handlers
algebra.register('^', (x) => x * 2);      // amp: amplify
algebra.register('+', (x) => x + 1);      // add: aggregate
algebra.register('×', (x) => x ** 2);     // mul: multiply
algebra.register('()', (x) => x);         // grp: identity
algebra.register('÷', (x) => x ** 0.5);   // div: divide
algebra.register('−', (x) => x - 1);      // sub: separate

// Apply operators
const result = algebra.applySequence(['^', '+', '×'], 3);  // 3→6→7→49

// Get undo sequence (uses invertibility pairs)
const { result, undoSequence } = algebra.applyWithUndo(['^', '+'], 5);
// result: 11, undoSequence: ['−', '()']
```

```python
from quantum_apl_python.s3_operator_algebra import OperatorAlgebra

algebra = OperatorAlgebra()

# Register handlers by name
algebra.register_by_name('amp', lambda x: x * 2)
algebra.register_by_name('add', lambda x: x + 1)
algebra.register_by_name('mul', lambda x: x ** 2)
algebra.register_by_name('grp', lambda x: x)
algebra.register_by_name('div', lambda x: x ** 0.5)
algebra.register_by_name('sub', lambda x: x - 1)

# Apply with undo
result, undo = algebra.apply_with_undo(['^', '+'], 5)
```

### Utility Functions

| Function | Description |
|----------|-------------|
| `compose(a, b)` | Compose two operators (a ∘ b) |
| `composeSequence(ops)` | Compose sequence of operators |
| `getInverse(op)` | Get inverse operator |
| `simplifySequence(ops)` | Reduce sequence to single equivalent op |
| `orderOf(op)` | Get order in group (1, 2, or 3) |
| `findPathToIdentity(op)` | Find sequence to reach identity |

## References

- Triadic Helix APL paper (Section 2.2: Operator Algebra)
- S₃ group theory (symmetric groups)
- CONSTANTS_ARCHITECTURE.md (single source of truth)
- HEXAGONAL_NEG_ENTROPY_PROJECTION.md (geometry formulas)
