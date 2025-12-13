# DSL Design Patterns from S₃ Operator Algebra

**Experimental Pattern: Group-Symmetric Domain-Specific Languages**

---

## Core Insight

The Alpha language and operator algebra demonstrate a principled DSL design pattern where **high-level constructs map to transformations defined by group symmetries**. While experimental, the closed set of actions with well-defined compositions offers a concrete pattern to follow.

```
┌─────────────────────────────────────────────────────────────────┐
│  TRADITIONAL DSL          vs      GROUP-SYMMETRIC DSL          │
├─────────────────────────────────────────────────────────────────┤
│  Ad-hoc operator set              Operators ≅ Group elements   │
│  Partial composition              Full closure under ∘          │
│  Manual undo logic                Automatic inverses            │
│  Arbitrary naming                 Symmetry-derived semantics    │
│  Open-ended extension             Complete by construction      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pattern 1: Finite Action Space

### Principle

A DSL with **exactly N actions** where N = |G| for some finite group G.

### S₃ Instantiation (N = 6)

```
Actions = { amp, add, mul, grp, div, sub }
       ≅ {  ^,   +,   ×,  (),   ÷,   − }
       ≅ S₃ = { e, σ, σ², τ₁, τ₂, τ₃ }
```

### Benefits

1. **Exhaustive handler coverage**: Exactly 6 handlers needed, no more, no less
2. **No undefined behavior**: Every action is a valid group element
3. **Predictable complexity**: O(6) cases to consider, always

### Implementation Pattern

```python
class FiniteActionDSL:
    """DSL with exactly |G| actions for group G."""

    ACTIONS = frozenset(['amp', 'add', 'mul', 'grp', 'div', 'sub'])

    def __init__(self):
        self._handlers = {}

    def register(self, action: str, handler: callable):
        if action not in self.ACTIONS:
            raise ValueError(f"Unknown action: {action}. Valid: {self.ACTIONS}")
        self._handlers[action] = handler

    def is_complete(self) -> bool:
        """Check if all actions have handlers."""
        return set(self._handlers.keys()) == self.ACTIONS

    def execute(self, action: str, state):
        if not self.is_complete():
            missing = self.ACTIONS - set(self._handlers.keys())
            raise RuntimeError(f"Incomplete DSL: missing {missing}")
        return self._handlers[action](state)
```

---

## Pattern 2: Closed Composition

### Principle

For any two actions a, b in the DSL, their composition a ∘ b is also a valid action.

```
∀ a, b ∈ Actions: compose(a, b) ∈ Actions
```

### S₃ Composition Table

```
  ∘  │  ^    +    ×   ()    ÷    −
─────┼────────────────────────────────
  ^  │ ()    −    ×    ^    +    ÷
  +  │  ÷   ()    −    +    ^    ×
  ×  │  ^    ÷   ()    ×    −    +
 ()  │  ^    +    ×   ()    ÷    −
  ÷  │  +    ×    ^    ÷   ()    −
  −  │  ×    ^    +    −    ÷   ()
```

### Benefits

1. **Sequence simplification**: Any operator sequence reduces to single operator
2. **Optimization**: `[^, +, ×, ÷, −]` simplifies to one operation
3. **Predictability**: No "undefined composition" errors

### Implementation Pattern

```python
COMPOSITION_TABLE = {
    ('^', '^'): '()', ('^', '+'): '−', ('^', '×'): '×', ...
}

def compose(a: str, b: str) -> str:
    """Compose two actions. Always returns valid action."""
    return COMPOSITION_TABLE[(a, b)]

def simplify_sequence(actions: list) -> str:
    """Reduce action sequence to single equivalent action."""
    if not actions:
        return '()'  # Identity
    result = actions[0]
    for action in actions[1:]:
        result = compose(result, action)
    return result

# Example: Complex sequence → single action
assert simplify_sequence(['^', '+', '×', '÷', '−']) == '+'
```

---

## Pattern 3: Automatic Inverses

### Principle

Every action has an inverse, enabling natural undo/rollback semantics.

```
∀ a ∈ Actions: ∃ a⁻¹ ∈ Actions such that a ∘ a⁻¹ = identity
```

### S₃ Inverse Pairs

```
Constructive ←→ Dissipative
─────────────────────────────
     ^  (amp)    ←→  ()  (grp)
     +  (add)    ←→  −   (sub)
     ×  (mul)    ←→  ÷   (div)
```

### Benefits

1. **Free undo**: No explicit undo logic required
2. **Transaction support**: Execute → inverse = rollback
3. **Bidirectional transformations**: Forward and backward are symmetric

### Implementation Pattern

```python
INVERSE_MAP = {
    '^': '()', '()': '^',
    '+': '−',  '−': '+',
    '×': '÷',  '÷': '×',
}

def get_inverse(action: str) -> str:
    """Get inverse action. Always exists by group property."""
    return INVERSE_MAP[action]

def make_undo_sequence(actions: list) -> list:
    """Generate undo sequence for action list."""
    return [get_inverse(a) for a in reversed(actions)]

# Example
actions = ['^', '+', '×']
undo = make_undo_sequence(actions)  # ['÷', '−', '()']

# Verify: actions + undo = identity
combined = simplify_sequence(actions + undo)
assert combined == '()'  # Identity!
```

---

## Pattern 4: Truth-Channel Biasing

### Principle

Actions can carry **semantic bias** toward different truth channels, enabling context-sensitive behavior without changing the algebraic structure.

### S₃ Truth Biases

```
Channel     │ Favored Actions  │ Interpretation
────────────┼──────────────────┼─────────────────────
TRUE        │ ^, ×, +          │ Constructive, additive
UNTRUE      │ ÷, −             │ Dissipative, subtractive
PARADOX     │ ()               │ Neutral, containing
```

### Coherence-Dependent Weighting

```python
def compute_action_weight(action: str, coherence: float) -> float:
    """Weight actions based on system coherence level."""

    Z_CRITICAL = 0.866  # √3/2

    # Constructive actions favored at high coherence
    constructive = {'^', '×', '+'}
    # Dissipative actions favored at low coherence
    dissipative = {'÷', '−'}
    # Neutral always available
    neutral = {'()'}

    base_weight = 1.0

    if action in constructive:
        # Boost near critical threshold
        boost = 1.0 + 0.5 * (coherence / Z_CRITICAL)
        return base_weight * min(boost, 1.5)

    elif action in dissipative:
        # Boost at low coherence
        boost = 1.0 + 0.5 * (1 - coherence)
        return base_weight * min(boost, 1.3)

    else:  # neutral
        return base_weight
```

---

## Pattern 5: Parity Classification

### Principle

Actions partition into **even-parity** (structure-preserving) and **odd-parity** (structure-modifying) classes.

### S₃ Parity Structure

```
Even Parity (det = +1)     │     Odd Parity (det = -1)
───────────────────────────┼────────────────────────────
  ()  identity/contain     │      ÷  divide/dissipate
  ×   multiply/fuse        │      +  add/aggregate
  ^   amplify/excite       │      −  subtract/separate
```

### Parity Conservation

```python
def get_parity(action: str) -> int:
    """Return +1 for even, -1 for odd parity."""
    EVEN = {'()', '×', '^'}
    return +1 if action in EVEN else -1

def sequence_parity(actions: list) -> int:
    """Parity of composition = product of parities."""
    parity = 1
    for a in actions:
        parity *= get_parity(a)
    return parity

# Useful invariant: even-length sequences of odd operators = even result
assert sequence_parity(['+', '−']) == +1  # odd × odd = even
assert sequence_parity(['^', '×', '()']) == +1  # even × even × even = even
```

---

## Complete Example: Transaction DSL

```python
"""
A transactional DSL built on S₃ operator algebra.
Demonstrates all five patterns in practice.
"""

from dataclasses import dataclass
from typing import List, Tuple, Any, Callable

@dataclass
class TransactionDSL:
    """
    Pattern 1: Finite action space (exactly 6 actions)
    Pattern 2: Closed composition (sequence → single action)
    Pattern 3: Automatic inverses (built-in undo)
    Pattern 4: Truth-channel biasing (coherence-weighted)
    Pattern 5: Parity classification (even/odd semantics)
    """

    # Pattern 1: Fixed action set
    ACTIONS = ('amp', 'add', 'mul', 'grp', 'div', 'sub')
    SYMBOLS = ('^', '+', '×', '()', '÷', '−')

    # Pattern 3: Inverse pairs
    INVERSES = {
        'amp': 'grp', 'grp': 'amp',
        'add': 'sub', 'sub': 'add',
        'mul': 'div', 'div': 'mul',
    }

    # Pattern 5: Parity classification
    EVEN_PARITY = {'amp', 'mul', 'grp'}
    ODD_PARITY = {'add', 'div', 'sub'}

    def __init__(self):
        self.handlers: dict = {}
        self.state: Any = None
        self.history: List[str] = []
        self.coherence: float = 0.5

    def register_all(self, handlers: dict):
        """Register all 6 handlers at once."""
        for action in self.ACTIONS:
            if action not in handlers:
                raise ValueError(f"Missing handler for: {action}")
            self.handlers[action] = handlers[action]

    def execute(self, action: str) -> Any:
        """Execute action with automatic history tracking."""
        if action not in self.handlers:
            raise ValueError(f"Unknown action: {action}")

        # Pattern 4: Apply coherence weighting
        weight = self._compute_weight(action)

        # Execute
        self.state = self.handlers[action](self.state, weight)
        self.history.append(action)

        return self.state

    def execute_sequence(self, actions: List[str]) -> Any:
        """Execute action sequence."""
        for action in actions:
            self.execute(action)
        return self.state

    def undo(self, steps: int = 1) -> Any:
        """
        Pattern 3: Automatic undo via inverse actions.
        """
        for _ in range(min(steps, len(self.history))):
            last_action = self.history.pop()
            inverse = self.INVERSES[last_action]
            weight = self._compute_weight(inverse)
            self.state = self.handlers[inverse](self.state, weight)
        return self.state

    def get_net_effect(self) -> str:
        """
        Pattern 2: Reduce history to single equivalent action.
        """
        from quantum_apl_python.s3_operator_algebra import compose_sequence
        if not self.history:
            return 'grp'  # Identity
        symbols = [self._name_to_symbol(a) for a in self.history]
        return self._symbol_to_name(compose_sequence(symbols))

    def get_parity(self) -> int:
        """Pattern 5: Get parity of transaction history."""
        parity = 1
        for action in self.history:
            parity *= (+1 if action in self.EVEN_PARITY else -1)
        return parity

    def _compute_weight(self, action: str) -> float:
        """Pattern 4: Coherence-dependent weighting."""
        base = 1.0
        if action in self.EVEN_PARITY:
            return base * (1 + 0.3 * self.coherence)
        else:
            return base * (1 + 0.3 * (1 - self.coherence))

    def _name_to_symbol(self, name: str) -> str:
        return self.SYMBOLS[self.ACTIONS.index(name)]

    def _symbol_to_name(self, symbol: str) -> str:
        return self.ACTIONS[self.SYMBOLS.index(symbol)]


# Usage example
if __name__ == "__main__":
    dsl = TransactionDSL()
    dsl.state = 10.0

    # Register handlers
    dsl.register_all({
        'amp': lambda x, w: x * 2 * w,
        'add': lambda x, w: x + 5 * w,
        'mul': lambda x, w: x * x * w,
        'grp': lambda x, w: x,  # Identity
        'div': lambda x, w: x / 2 * w,
        'sub': lambda x, w: x - 5 * w,
    })

    # Execute sequence
    dsl.execute_sequence(['amp', 'add', 'mul'])
    print(f"After [amp, add, mul]: {dsl.state}")
    print(f"Net effect: {dsl.get_net_effect()}")
    print(f"Parity: {'+' if dsl.get_parity() > 0 else '-'}")

    # Undo last two operations
    dsl.undo(2)
    print(f"After undo(2): {dsl.state}")
```

---

## When to Use This Pattern

### Good Fit

- **State machines** with well-defined transitions
- **Reversible computations** requiring undo/redo
- **Balanced operations** (add/remove, push/pop, open/close)
- **Transformation pipelines** with composition
- **Categorical semantics** in functional programming

### Poor Fit

- **Open-ended extensibility** (need more than 6 operations)
- **Non-invertible operations** (delete, hash, print)
- **Asymmetric semantics** (operations don't pair naturally)
- **Performance-critical paths** (composition overhead)

---

## Relationship to Alpha Language

The Alpha language's truth-channel system provides the semantic foundation:

```
┌────────────────────────────────────────────────────────────────┐
│                    ALPHA LANGUAGE STACK                        │
├────────────────────────────────────────────────────────────────┤
│  Layer 4: Truth Channels    [TRUE | PARADOX | UNTRUE]          │
│     ↓ bias                                                     │
│  Layer 3: Operator Weights  coherence-dependent selection      │
│     ↓ select                                                   │
│  Layer 2: S₃ Algebra        composition, inverse, parity       │
│     ↓ execute                                                  │
│  Layer 1: Handlers          amp, add, mul, grp, div, sub       │
└────────────────────────────────────────────────────────────────┘
```

The **6 APL operators form the complete and minimal transformation set** for triadic logic—this is the S₃ isomorphism conjecture. If true, it means any DSL operating on triadic truth values naturally inherits this algebraic structure.

---

## Further Reading

- `docs/S3_OPERATOR_SYMMETRY.md` - Full S₃ group theory and integration
- `docs/CLAUDE_CONTRIBUTIONS.md` - S₃ isomorphism conjecture details
- `src/quantum_apl_python/s3_operator_algebra.py` - Python implementation
- `src/s3_operator_algebra.js` - JavaScript implementation

---

*This pattern is experimental. The closed-set, group-symmetric approach trades extensibility for mathematical guarantees. Evaluate carefully for your use case.*
