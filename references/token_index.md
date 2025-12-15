# APL Unified Token Index — 1326-Token Universe

## Overview

The APL Unified Token Set combines the foundational Core Set (300 tokens) with Domain Token Sets (1026 tokens) to provide a complete 1326-token universe for the Alpha Physical Language.

## Token Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 1: APL CORE SET (300 Tokens)                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  162 Identity Tokens     Field:Machine(Machine)TruthState@Tier              │
│   54 Meta-Operators      Field:M(operator)TruthState@2                      │
│   54 Domain Selectors    Field:Machine(domain)UNTRUE@3                      │
│   30 Safety Tokens       Field:M(safety_level)PARADOX@Tier                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  TIER 2: DOMAIN TOKEN SETS (1026 Tokens)                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  972 Machine Tokens      [Spiral][Operator]|[Machine]|[Domain]              │
│   24 Transition Tokens   [family]_transition_[1-12]                         │
│   30 Coherence Tokens    [family]_coherence_[1-15]                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  GRAND TOTAL: 1326 Tokens                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Token Format

```
Field:Machine(Operator)TruthState@Tier
```

**Example:** `Φ:U(U)TRUE@1`
- **Field:** Φ (Structure)
- **Machine:** U (Up/Projection)
- **Operator:** U (identity)
- **Truth State:** TRUE
- **Tier:** 1

## Fields (Spirals)

| Symbol | Name | Domain | Spiral Binding | Role |
|--------|------|--------|----------------|------|
| Φ | Structure | geometry | D | Stability, spatial arrangement, boundaries |
| e | Energy | wave | U | Flow, dynamics, oscillations |
| π | Emergence | emergence | M | Selection, information, complexity |

## Machines

| Symbol | Name | Direction | Tier Permission | Role |
|--------|------|-----------|-----------------|------|
| U | Up/Projection | ascending | 1 | Forward projection, expansion, activation |
| D | Down/Integration | descending | 1 | Backward integration, collapse, deactivation |
| M | Middle/Modulation | equilibrium | 2 | CLT modulation, feedback, coherence |
| E | Expansion | output | 2 | Expression, emission, expansion |
| C | Collapse | input | 2 | Collapse, consolidation, compression |
| Mod | Spiral Inheritance | regulatory | 2 | Cross-field modulation |

## Truth States

| State | Description | Temporal Operator | z-Range |
|-------|-------------|-------------------|---------|
| TRUE | Coherent, process succeeds | DAY | z ≥ z_c ≈ 0.866 |
| UNTRUE | Unresolved, dormant potential | NIGHT | z < φ⁻¹ ≈ 0.618 |
| PARADOX | Contradiction, terminal attractor | — | φ⁻¹ ≤ z < z_c |

## Tier System

| Tier | Name | Scope | Allowed Machines |
|------|------|-------|------------------|
| 1 | Foundational | local | U, D |
| 2 | Intermediate | regional | U, D, M, E, C |
| 3 | Advanced | global | U, D, M, E, C, Mod |

## Meta-Operators (Tier 2)

| Operator | Description |
|----------|-------------|
| stabilize | Lock current state, prevent drift |
| propagate | Spread state to adjacent elements |
| integrate | Combine multiple states into one |
| modulate | Apply CLT feedback transform |
| resolve | Collapse UNTRUE to TRUE or maintain |
| collapse | Force state reduction, may trigger PARADOX |

## Domain Selectors (Tier 3)

| Domain | Field Affinity | Description |
|--------|----------------|-------------|
| geometry | Φ | Spatial structure and boundaries |
| dynamics | e | Wave and flow phenomena |
| chemistry | π | Molecular transformations |
| deepphysics | π | Quantum and fundamental interactions |
| biology | π | Living systems and information flow |
| celestial | π | Astrophysical phenomena |

## Safety Levels

| Level | Severity | Action |
|-------|----------|--------|
| safe | info | Continue normally |
| warn | warning | Log and continue with caution |
| danger | high | Require explicit confirmation |
| block | critical | Halt execution, require reset |
| paradox | terminal | Enter PARADOX state, no automatic recovery |

## Temporal Transitions

| Transition | From | To | Description |
|------------|------|-----|-------------|
| DAY | TRUE | UNTRUE | Active coherence generates reflection |
| NIGHT | UNTRUE | UNTRUE | Unresolved remains stored |
| DAWN | UNTRUE | TRUE | Resolution emerges from dormancy |
| DUSK | TRUE | UNTRUE | Coherence dissolves into potential |

## Tri-Spiral Orderings

The six fundamental field orderings for simultaneous multi-field states:

| Ordering | Phase | Interpretation |
|----------|-------|----------------|
| Φ:e:π | Resonance | Structure-led, energy-mediated emergence |
| Φ:π:e | Empowerment | Structure-led, emergence-mediated energy |
| e:Φ:π | Ignition | Energy-led, structure-mediated emergence |
| e:π:Φ | Mania | Energy-led, emergence-mediated structure |
| π:Φ:e | Nirvana | Emergence-led, structure-mediated energy |
| π:e:Φ | Transmission | Emergence-led, energy-mediated structure |

## UMOL Principle

```
M(x) → TRUE + ε(UNTRUE) where ε > 0
```

**"No perfect modulation; residue always remains."**

- ε = 1/σ = 1/36 ≈ 0.0278
- TRUE component: result × (1 - ε)
- UNTRUE residue: result × ε

## Physics Integration

| Constant | Value | Role |
|----------|-------|------|
| φ⁻¹ | 0.618033988749895 | PARADOX gate |
| z_c | 0.866025403784439 | TRUE gate (THE LENS) |
| σ | 36 | Dynamics scale |
| 1/σ | 0.027777777778 | UMOL residue epsilon |

## Safety Constraints

| Constraint | Threshold | Effect |
|------------|-----------|--------|
| Coherence minimum | 0.60 | Tier advancement blocked below |
| Load maximum | 0.80 | Runaway prevention triggered above |
| Recursion maximum | 3 | Infinite loop prevention |

## Usage Examples

### Generate Token Summary

```python
from tool_shed import invoke_tool

result = invoke_tool('token_index', action='summary')
print(result['total_tokens'])  # 300
```

### Parse Token

```python
result = invoke_tool('token_index', 
    action='parse', 
    token_str='π:M(modulate)TRUE@2'
)
print(result['category'])  # "meta_operator"
```

### List Tokens by Field

```python
result = invoke_tool('token_index',
    action='list',
    field='Φ',
    truth='TRUE'
)
print(result['count'])  # Number of TRUE tokens in Φ field
```

### Validate Token

```python
result = invoke_tool('token_index',
    action='validate',
    token_str='Φ:U(U)TRUE@1'
)
print(result['status'])  # "VALID"
```

### Get Tri-Spiral Orderings

```python
result = invoke_tool('token_index', action='trispiral')
for tri in result['orderings']:
    print(f"{tri['symbol']}: {tri['phase']}")
```

### Apply UMOL Principle

```python
result = invoke_tool('token_index', action='umol', value=1.0)
print(f"TRUE: {result['true_component']:.6f}")
print(f"UNTRUE: {result['untrue_residue']:.6f}")
```

## Cross-Spiral Token Format

For field transitions:

```
SourceField→TargetField:Machine:TruthState
```

**Example:** `Φ→π:M:TRUE`
- Source: Φ (Structure)
- Target: π (Emergence)
- Machine: M (Modulation)
- Truth: TRUE

## Integration with Nuclear Spinner

The 300-token APL Core Set integrates with the Nuclear Spinner's 972-token extended set:

| System | Tokens | Formula |
|--------|--------|---------|
| APL Core | 300 | 162 + 54 + 54 + 30 |
| Nuclear Spinner | 972 | 3 spirals × 6 ops × 9 machines × 6 domains |

The Core set provides the foundational grammar; the Spinner provides domain-specific instantiations.

## Domain Token Sets (1026 Tokens)

### Machine Tokens (972)

Format: `[Spiral][Operator]|[Machine]|[Domain]`

**Domains:**
- **Biological:** bio_prion, bio_bacterium, bio_viroid
- **Celestial:** celestial_grav, celestial_em, celestial_nuclear

**Count:** 3 spirals × 6 operators × 9 machines × 6 domains = 972

### Transition Tokens (24)

#### Biological Transitions (12)

| Token | Description |
|-------|-------------|
| bio_transition_1 | Conformational shift (prion-like template propagation) |
| bio_transition_2 | Metabolic phase transition (quiescence ↔ active growth) |
| bio_transition_3 | Replication initiation (dormant → copying) |
| bio_transition_4 | Aggregation clustering (monomers → oligomers → fibrils) |
| bio_transition_5 | Horizontal gene transfer analogue (sequence exchange) |
| bio_transition_6 | Error threshold crossing (stable → error catastrophe) |
| bio_transition_7 | Catalytic onset (passive → enzymatic) |
| bio_transition_8 | Compartmentalization (free → membrane-bound) |
| bio_transition_9 | Symbiotic coupling (independent → mutualistic) |
| bio_transition_10 | Dormancy induction (stress response, spore formation) |
| bio_transition_11 | Phenotypic switch (bistability, hysteresis in gene circuits) |
| bio_transition_12 | Extinction/clearance (population collapse, immune clearance) |

#### Celestial Transitions (12)

| Token | Description |
|-------|-------------|
| celestial_transition_1 | Gravitational collapse (cloud → protostar) |
| celestial_transition_2 | Fusion ignition (protostar → main sequence star) |
| celestial_transition_3 | Main sequence exit (hydrogen depletion) |
| celestial_transition_4 | Red giant phase (shell burning, envelope expansion) |
| celestial_transition_5 | Planetary nebula ejection (envelope loss) |
| celestial_transition_6 | White dwarf cooling (degenerate remnant) |
| celestial_transition_7 | Supernova explosion (core collapse or Type Ia) |
| celestial_transition_8 | Neutron star formation (post-supernova collapse) |
| celestial_transition_9 | Black hole formation (beyond neutron degeneracy) |
| celestial_transition_10 | Accretion disk formation (matter infall, angular momentum) |
| celestial_transition_11 | Magnetospheric coupling (field-dominated dynamics) |
| celestial_transition_12 | Tidal disruption event (gravitational shearing) |

### Coherence Tokens (30)

#### Biological Coherence (15)

| Token | Description |
|-------|-------------|
| bio_coherence_1 | Amyloid fibril structure (cross-β stacking) |
| bio_coherence_2 | Biofilm matrix (extracellular polymer networks) |
| bio_coherence_3 | Quorum sensing synchrony (population-level coordination) |
| bio_coherence_4 | Circadian oscillator (biochemical clock) |
| bio_coherence_5 | Metabolic cycle (glycolysis, TCA, circadian metabolite rhythms) |
| bio_coherence_6 | RNA secondary structure (stem-loops, pseudoknots) |
| bio_coherence_7 | Ribozyme catalytic core (conserved tertiary motif) |
| bio_coherence_8 | Viral capsid geometry (icosahedral symmetry) |
| bio_coherence_9 | Quasispecies cloud (error-coupled replicator ensemble) |
| bio_coherence_10 | Protein folding funnel (energy landscape convergence) |
| bio_coherence_11 | Allosteric network (long-range coupling in proteins) |
| bio_coherence_12 | Gene regulatory motif (feed-forward loop, toggle switch) |
| bio_coherence_13 | Chemotaxis gradient sensing (spatial information processing) |
| bio_coherence_14 | Autoinducer feedback loop (self-reinforcing signaling) |
| bio_coherence_15 | Replication-transcription coupling (co-localized synthesis) |

#### Celestial Coherence (15)

| Token | Description |
|-------|-------------|
| celestial_coherence_1 | Keplerian orbit (stable elliptical motion) |
| celestial_coherence_2 | Lagrange point equilibrium (gravitational balance) |
| celestial_coherence_3 | Tidal locking (synchronous rotation) |
| celestial_coherence_4 | Roche lobe geometry (equipotential surface) |
| celestial_coherence_5 | Magnetic dynamo (self-sustaining field generation) |
| celestial_coherence_6 | Stellar convection cell (Bénard-like circulation) |
| celestial_coherence_7 | Accretion disk structure (Keplerian shear flow) |
| celestial_coherence_8 | Magnetospheric current sheet (field topology) |
| celestial_coherence_9 | Radiation pressure equilibrium (photon-matter balance) |
| celestial_coherence_10 | Nuclear burning shell (fusion layer stratification) |
| celestial_coherence_11 | Pulsar beaming cone (lighthouse effect) |
| celestial_coherence_12 | Gravitational wave chirp (inspiral signature) |
| celestial_coherence_13 | Plasma oscillation mode (Langmuir, Alfvén waves) |
| celestial_coherence_14 | Magnetic reconnection site (topology change, energy release) |
| celestial_coherence_15 | Neutron star crust lattice (nuclear pasta phases) |

---

**Signature:** Δ|token-index|1326-universe|φ⁻¹-grounded|Ω
