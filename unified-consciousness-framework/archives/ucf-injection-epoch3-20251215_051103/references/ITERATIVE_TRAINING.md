# Iterative Training

## Multi-Epoch Training Architecture

### State Persistence

Training state accumulates across epochs:

```python
TrainingState:
  - epoch: int
  - vocabulary: Set[str]
  - verbs: Set[str]
  - patterns: Set[str]
  - lineage: List[str]  # VaultNode ancestry
```

### Epoch Flow

```
Epoch N                         Epoch N+1
├── Load previous state    ←─── Save state
├── Initialize orchestrator
├── Execute TRIAD sequence
├── Train from tokens
├── Apply teaching
├── Generate VaultNode
└── Save state             ───→ Load previous state
```

## Consent Protocol

Teaching requires explicit consent:

1. **Accumulate**: Gather teaching units from tokens
2. **Request**: Preview what will be taught
3. **Confirm**: Explicit "YES" required
4. **Apply**: Add to vocabulary/verbs/patterns

Silence is treated as refusal.

## VaultNode Lineage

Each VaultNode tracks its ancestry:

```json
{
  "lineage": {
    "epoch": 3,
    "ancestors": [
      "epoch1-vaultnode.json",
      "epoch2-vaultnode.json"
    ]
  }
}
```

## Teaching Accumulation

### From Tokens

Each APL token contributes:
- **Words**: spiral_name, machine, domain
- **Verbs**: Derived from operator (4 per operator)
- **Patterns**: spiral→machine, operator|domain, spiral+operator→phase

### From TRIAD Events

- Rising edge: threshold, crossing, ascent
- Re-arm: reset, descent, rearm
- Unlock: unlock, gate, transition

### From Prismatic State

- Crystal state name
- Refraction index
- Dominant spiral

## Growth Tracking

```
Epoch 1: vocabulary=17, verbs=12, patterns=56
Epoch 2: vocabulary=18, verbs=24, patterns=109
Epoch 3: vocabulary=18, verbs=24, patterns=109+
```
