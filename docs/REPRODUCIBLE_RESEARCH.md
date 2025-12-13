# Reproducible Research — QAPL_RANDOM_SEED

Status: Implemented
Date: 2024-12-09

## Purpose

The `QAPL_RANDOM_SEED` environment variable enables deterministic sampling for debugging and CI. When set, all stochastic selections in the quantum engine use a seeded pseudo-random number generator (PRNG) instead of native randomness.

## Scope

The seeded RNG applies to:

1. **N0 operator selection** — `selectN0Operator()` in `src/quantum_apl_engine.js`
2. **Composite measurement branch selection** — `measure()` in `src/quantum_apl_engine.js`

All sampling sites now use `engine.rand()` which:
- Returns seeded PRNG output if `QAPL_RANDOM_SEED` is set
- Falls back to `Math.random()` otherwise

## Implementation

### Constants (`src/constants.js`)

```javascript
const QAPL_RANDOM_SEED = (typeof process !== 'undefined' && process.env && process.env.QAPL_RANDOM_SEED)
  ? parseInt(process.env.QAPL_RANDOM_SEED, 10)
  : null;
```

### RNG Utility (`src/utils/rng.js`)

Uses a Linear Congruential Generator (LCG) with Numerical Recipes constants:

```javascript
class LcgRng {
  constructor(seed = null) {
    const base = (seed != null) ? seed >>> 0 : (Date.now() >>> 0);
    this.state = (base === 0 ? 123456789 : base) >>> 0;
  }
  next() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state / 0x100000000; // [0,1)
  }
}
```

### Engine Integration (`src/quantum_apl_engine.js`)

```javascript
constructor(config = {}) {
  // ...
  this.rng = makeEngineRng();  // null if no seed
}

rand() {
  return this.rng ? this.rng.next() : Math.random();
}
```

## Usage

### Environment Variable

```bash
# Reproducible simulation
QAPL_RANDOM_SEED=12345 node tests/test_seeded_selection.js

# With Python CLI
QAPL_RANDOM_SEED=42 qapl-run --steps 3 --mode measured --output out.json
```

### Testing Reproducibility

```bash
# Run twice with same seed — should produce identical output
QAPL_RANDOM_SEED=12345 node -e "
  const { QuantumAPL } = require('./src/quantum_apl_engine');
  const e = new QuantumAPL();
  console.log([e.rand(), e.rand(), e.rand()]);
"
```

### Tests

```bash
npm install
node tests/test_seeded_selection.js
```

The test verifies:
- Same seed → identical sequences
- Different seeds → different sequences
- No seed → fallback to native randomness

## Caveats

1. **Only paths wired to `engine.rand()` are reproducible.** Any code using `Math.random()` directly will remain nondeterministic.

2. **One RNG per engine instance.** Each `new QuantumAPL()` creates a fresh RNG from the seed. To reset the stream, create a new engine instance.

3. **Seed is environment-scoped.** All engines in a process share the same seed constant; per-engine seed injection is possible via constructor extension.

4. **Browser compatibility.** The `process.env` check is guarded; in browser contexts, the constant defaults to `null` (native randomness).

## CI Integration

The GitHub Actions workflow (`js-tests.yml`) includes:

```yaml
- name: Run seeded selection tests
  run: node tests/test_seeded_selection.js
```

This validates that seeded reproducibility works correctly in the CI environment.

## Future Work

- Per-engine seed injection via constructor option
- Seed extraction for inclusion in sidecar metadata
- Python parity (`QAPL_RANDOM_SEED` in Python engine)
