# APL Measurement Operators (Derived from Collapse Visual)

The measurement flow from the diagram yields two canonical operator modes:

## 1. Single-Eigenstate Collapse
- **Projector:** `P_μ = |ϕ_μ⟩⟨ϕ_μ|`
- **Rule:** `|ψ⟩ → |ϕ_μ⟩` (phase ignored)
- **Born probability:** `Pr(μ) = ⟨ψ|P_μ|ψ⟩`

### Operator token
```
Φ:T(ϕ_μ)TRUE@Tier
```
Maps the structure field into eigenstate `|ϕ_μ⟩` when the TRUE channel fires.

## 2. Subspace (Degenerate) Collapse
- **Projector:** `P = Σ_{k∈subspace} |ϕ_k⟩⟨ϕ_k|`
- **Rule:** `|ψ⟩ → P|ψ⟩ / √⟨ψ|P|ψ⟩`
- **Born probability:** `⟨ψ|P|ψ⟩`

### Operator tokens
```
Φ:Π(subspace)PARADOX@Tier
π:Π(subspace)UNTRUE@Tier
```
Both apply the same projector but route to different truth channels; the normalization automatically enforces physical probabilities.

## Composite Measurement Operator
```
M_meas = Σ_μ |ϕ_μ⟩⟨ϕ_μ| ⊗ |T_μ⟩⟨T_μ|
```
- Collapses the system and records the outcome in the truth register `|T_μ⟩`.
- Compatible with the visualization’s branching nodes (single eigenstate vs degenerate subspace).

## Usage Guidance
1. Choose `P_μ` when the operator grammar targets a specific eigenmode token (e.g., `Φ:T(lattice)`).
2. Choose `P` when the sentence references an entire regime (e.g., “hierarchical states”) rather than one eigenvector.
3. Always normalize using the Born factor `⟨ψ|P|ψ⟩` before feeding the result back into the classical channels (bridge ensures this).
