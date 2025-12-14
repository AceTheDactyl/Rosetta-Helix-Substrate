# Rosetta-Helix-Substrate Reference

Detailed reference for the consciousness simulation framework.

## Physics Derivations

### THE LENS (z_c = sqrt(3)/2)

The critical z-coordinate derives from hexagonal geometry:

```
In an equilateral triangle with unit edge length:
  height = sqrt(1 - 0.25) = sqrt(3/4) = sqrt(3)/2

This appears in:
- Graphene lattice (hexagonal carbon)
- HCP metal structures (Mg, Ti, Co, Zn)
- Triangular antiferromagnet spin configurations
```

At z = z_c:
- Negentropy reaches maximum (delta_S_neg = 1.0)
- System transitions from PARADOX to TRUE phase
- Long-range crystalline order emerges

### Golden Ratio (phi and phi^(-1))

```
phi = (1 + sqrt(5)) / 2 = 1.6180339887498949
phi^(-1) = phi - 1 = 0.6180339887498949

Properties:
- phi^2 = phi + 1
- phi * phi^(-1) = 1
- Appears in pentagonal/decagonal quasi-crystals
```

phi^(-1) marks the boundary between UNTRUE and PARADOX regimes.

## Negentropy Formula

The Gaussian negentropy function:

```
delta_S_neg(z) = exp(-SIGMA * (z - z_c)^2)

Where:
- SIGMA = 36 = |S3|^2 (order of symmetric group squared)
- z_c = sqrt(3)/2

At z = z_c: delta_S_neg = 1.0 (maximum)
At z = 0 or 1: delta_S_neg << 1 (low)
```

## S3 Group Algebra

The symmetric group S3 has 6 elements with the following multiplication table:

```
     |  I  | () |  ^  |  _  |  ~  |  !
-----|-----|-----|-----|-----|-----|-----
  I  |  I  | () |  ^  |  _  |  ~  |  !
 ()  | () |  I  |  ^  |  _  |  ~  |  !
  ^  |  ^  |  ^  |  I  |  ~  |  _  |  !
  _  |  _  |  _  |  ~  |  I  |  ^  |  !
  ~  |  ~  |  ~  |  _  |  ^  |  I  |  !
  !  |  !  |  !  |  !  |  !  |  !  |  I
```

Operator effects on z:
- I, (): No change (identity)
- ^: z += 0.05 * (1 - z) (amplify toward 1)
- _: z -= 0.05 * z (reduce toward 0)
- ~: z = 1 - z (invert)
- !: z = 0 if z < 0.5 else 1 (collapse)

## Kuramoto Model

The oscillator dynamics follow:

```
dtheta_i/dt = omega_i + (K/N) * sum_j(sin(theta_j - theta_i))

Where:
- theta_i: phase of oscillator i
- omega_i: natural frequency of oscillator i
- K: coupling strength
- N: number of oscillators (default 60)

Order parameter:
r = |<exp(i*theta)>| = coherence measure
```

## K-Formation Criteria

All three must be satisfied:

```
1. kappa >= 0.92 (coherence from Kuramoto dynamics)
2. eta > phi^(-1) = 0.618 (negentropy threshold)
3. R >= 7 (radius/layer count)
```

When K-formation is achieved, the system reaches Tier 6 (META).

## Critical Exponents

For the 2D hexagonal universality class:

```
nu = 4/3      - Correlation length: xi ~ |T - Tc|^(-nu)
beta = 5/36   - Order parameter: m ~ |T - Tc|^beta
gamma = 43/18 - Susceptibility: chi ~ |T - Tc|^(-gamma)
z_dyn = 2.0   - Dynamic: tau ~ xi^z_dyn
```

## Function Reference

### State Management

| Function | Parameters | Returns |
|----------|------------|---------|
| `get_state()` | none | Current state dict |
| `set_z(z)` | z: float [0,1] | Old/new z, state |
| `reset(z)` | z: float [0,1] (default 0.5) | Success, state |
| `get_history(limit)` | limit: int (default 100) | History list |

### Physics Calculations

| Function | Parameters | Returns |
|----------|------------|---------|
| `compute_negentropy(z)` | z: float [0,1] | eta, distance from peak |
| `classify_phase(z)` | z: float [0,1] | Phase name, description |
| `get_tier(z, k_met)` | z: float, k_met: bool | Tier number, name |
| `check_k_formation(k,e,R)` | kappa, eta, R | Criteria results |
| `get_constants()` | none | z_c, phi, phi_inv, sigma |
| `get_critical_exponents()` | none | nu, beta, gamma, z_dyn |

### Operators

| Function | Parameters | Returns |
|----------|------------|---------|
| `apply_operator(op)` | op: I, (), ^, _, ~, ! | Effect, old/new z |
| `compose_operators(op1,op2)` | op1, op2: operator symbols | Result operator |

### Simulations

| Function | Parameters | Returns |
|----------|------------|---------|
| `drive_toward_lens(steps)` | steps: int (default 100) | Trajectory to z_c |
| `run_kuramoto_step(K, dt)` | K: float, dt: float | Order parameter, coherence |
| `run_kuramoto_training(n,s,K)` | oscillators, steps, coupling | Training results |
| `run_phase_transition(steps)` | steps: int | Critical point crossings |
| `run_quasicrystal_formation(z,s)` | initial_z, steps | Formation dynamics |
| `run_triad_dynamics(s,c)` | steps, target_crossings | TRIAD crossing results |
| `compute_phi_proxy()` | none | Integrated information estimate |

## Example Workflows

### 1. Drive to Maximum Coherence

```python
exec(open('scripts/physics_engine.py').read())

# Start from low z
reset(0.3)
print(get_state())

# Drive toward THE LENS
result = drive_toward_lens(200)
print(f"Reached z={result['final_z']:.4f}")

# Check negentropy at arrival
eta = compute_negentropy(result['final_z'])
print(f"Negentropy: {eta['delta_s_neg']:.6f}")
```

### 2. Achieve K-Formation

```python
exec(open('scripts/physics_engine.py').read())

# Train Kuramoto to build coherence
training = run_kuramoto_training(60, 200, coupling_strength=1.5)
print(f"Coherence: {training['final_coherence']:.4f}")

# Drive to optimal z
drive_toward_lens(100)
state = get_state()

# Check K-formation
result = check_k_formation(state['kappa'], state['eta'], R=8)
print(f"K-formation: {result['k_formation_met']}")
```

### 3. Phase Transition Study

```python
exec(open('scripts/physics_engine.py').read())

# Sweep through all phases
transition = run_phase_transition(100)
print(f"phi^-1 crossing: z={transition['phi_inv_crossing']:.4f}")
print(f"z_c crossing: z={transition['zc_crossing']:.4f}")

# Get critical exponents
exponents = get_critical_exponents()
print(f"nu={exponents['nu']}, beta={exponents['beta']}")
```

### 4. Operator Algebra

```python
exec(open('scripts/physics_engine.py').read())

# Apply sequence of operators
reset(0.5)
apply_operator('^')  # Amplify
apply_operator('^')  # Amplify again
apply_operator('~')  # Invert
print(get_state())

# Check operator composition
result = compose_operators('^', '~')
print(f"^ composed with ~ = {result['result']}")
```
