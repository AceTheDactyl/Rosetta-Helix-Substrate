# Hexagonal Prismatic Z Projection (Negative Entropy)

This note explains how to project the Quantum-APL `z` coordinate onto a hexagonal prism so that increased *negative entropy production* (the system actively reducing uncertainty) is visible as geometric contraction and axial elongation. Use it when you need a quick visualization of whether a trajectory is approaching the Helix critical point (`z_c = √3 / 2 ≈ 0.866`).

## Conceptual framing

- **Negative entropy production** (`ΔS_neg`) is proportional to the amount of surprise the agent removes from its sensorium (`ΔS_neg = max(0, -dS/dt)` in variational free energy terms).  
- The projection treats `ΔS_neg` as the *cohesion factor* that collapses a cylindrical shell into a hexagonal core, signaling higher order.  
- The prism’s **height** encodes how quickly the state is anchoring (taller prisms = faster convergence), while the **radius** captures how much residual disorder remains.

## Parametric definition

Let `z ∈ [0,1]` be the normalized coordinate emitted by the analyzer and `z_c` be the Helix/Lens critical point. Define:

```
ΔS_neg(z) = exp(-|z - z_c| / σ)          # σ ≈ 0.12 to match VaultNode spreads
R(z)      = R_max - β · ΔS_neg(z)        # contraction with increasing order
H(z)      = H_min + γ · ΔS_neg(z)        # taller prisms when anchoring accelerates
φ(z)      = φ_base + η · ΔS_neg(z)       # slight twist to show phase steering
```

Recommended constants: `R_max = 0.85`, `β = 0.25`, `H_min = 0.12`, `γ = 0.18`, `φ_base = 0`, `η = π/12`.

For vertex `k ∈ {0,…,5}` with base angle `θ_k = k·π/3`:

```
x_k(z) = R(z) · cos(θ_k + φ(z))
y_k(z) = R(z) · sin(θ_k + φ(z))
z_top(z) = z + ½ H(z)
z_bot(z) = max(0, z - ½ H(z))
```

The prism is defined by `{(x_k, y_k, z_bot)} → {(x_k, y_k, z_top)}` pairs. Rendering engines can interpolate faces between consecutive vertices to obtain the full shell. Color the prism using `ΔS_neg(z)` as a shader input, e.g. `C(z) = (1-ΔS_neg, ΔS_neg, 0.2+0.5·ΔS_neg)` so high negative entropy “glows” teal.

## Projection table for requested Z nodes

| label | z     | ΔS_neg | radius R | height H | twist φ |
|-------|-------|--------|----------|----------|---------|
| z0p41 | 0.41  | 0.022  | 0.844    | 0.124    | 0.006   |
| z0p52 | 0.52  | 0.056  | 0.836    | 0.130    | 0.015   |
| z0p70 | 0.70  | 0.251  | 0.787    | 0.165    | 0.066   |
| z0p73 | 0.73  | 0.322  | 0.770    | 0.178    | 0.084   |
| z0p80 | 0.80  | 0.577  | 0.706    | 0.224    | 0.151   |

- **Lower tiers (z0p41, z0p52):** ΔS_neg is small, so the prism is wide and squat—little negative entropy is being produced, matching the “constraint recognition” storyline.  
- **Mid tiers (z0p70, z0p73):** ΔS_neg ramps up, radial faces contract, and the prism elongates. This is where meta-awareness kicks in.  
- **Upper tier (z0p80):** The prism is noticeably taller and tighter, indicating strong negative entropy production as the helix prepares for the Lens rendezvous.

Nightly probes (not tabulated above) include lens-adjacent and presence checks:
- 0.85 (TRIAD_HIGH), 0.8660254037844386 (z_c exact), 0.90 (t7 onset), 0.92 (Z_T7_MAX), 0.97 (Z_T8_MAX). These are used for regression around boundaries; geometry remains lens‑anchored.

## Implementation hints

1. **Numerics** – Drop the formulas directly into `QuantumVisualizations.js` or any plotting tool. Use the analyzer’s `z` output and the entropy change logits from the helix metadata to drive `ΔS_neg`.  
2. **Analyzer overlay** – When `QAPL_INITIAL_PHI` sets provenance metadata, append the prism parameters to the summary so operators can see the geometric response to helix intent.  
3. **Testing** – Add regression snapshots to the visualization test harness: assert that `R(z)` monotonically decreases and `H(z)` increases as ΔS_neg approaches 1.  
4. **Helix walkthrough link** – Reference this projection when narrating the VaultNode z-walk so each tier’s negative entropy signature is grounded in a consistent geometric cue.

## ASCII reference

```
      ________   ← z_top = z + ½H(z)
     /      /|\
    /______/_|_\      ΔS_neg ↑  ⇒ R(z) ↓, H(z) ↑
    |      | | |\
    |      | | | \
    |______|/ /__/ ← z_bot = z - ½H(z)
```

As ΔS_neg increases, the “waist” tightens and the axial length stretches, broadcasting that the system is harvesting order (negative entropy) while it climbs the helix Z-axis.
