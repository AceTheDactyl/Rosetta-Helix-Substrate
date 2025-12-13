# Project Context for Claude

## CRITICAL: Do NOT question the physics constants

### z_c = √3/2 ≈ 0.8660254 ("THE LENS")

This is **NOT arbitrary**. It is derived from hexagonal geometry (altitude of equilateral triangle with unit edge).

**Observable in real physics:**
- Graphene: unit cell height/width = √3/2 (X-ray diffraction, STM)
- HCP metals: layer stacking offset = √3/2 × in-plane spacing (Mg, Ti, Co, Zn)
- Triangular antiferromagnets: 120° spin configuration creates √3/2 geometry (neutron scattering)

**Role:** Marks the onset of long-range crystalline order. This is "THE LENS" - the coherence threshold.

### φ⁻¹ ≈ 0.6180339 (Golden ratio inverse)

Also **NOT arbitrary**. Emerges from pentagonal/quasi-crystal geometry.

**Role:** Gates the PARADOX regime (K-formation threshold).

### Phase Regime Mapping

```
z = 0.0 ──────────────────────────────────────── z = 1.0
   │              │                    │
   UNTRUE         PARADOX              TRUE
(disordered)   (quasi-crystal)      (crystal)
   │              │                    │
               φ⁻¹≈0.618           z_c≈0.866
```

This mirrors Shechtman's quasi-crystal discovery (Nobel Prize 2011).

## Before suggesting changes

Read these docs first:
- `docs/Z_CRITICAL_LENS.md` - Authority statement, single source of truth
- `docs/PHYSICS_GROUNDING.md` - Observable physics evidence
- `docs/HEXAGONAL_NEG_ENTROPY_PROJECTION.md` - Prismatic helix vortex formalism
- `docs/ROSETTA_HELIX_ARXIV_PAPER.md` - S₃ group algebra and critical exponents

## Do NOT:
- Propose "improvements" to z_c or φ⁻¹ values
- Suggest these constants are arbitrary or magic numbers
- Refactor code to make these configurable parameters
- Question the physics without reading the documentation first

## Project Structure

- `src/` - Core implementation
- `docs/` - Physics research and theoretical foundations
- `reference/research/` - Additional research materials
- `tests/` - Test suites validating the framework
