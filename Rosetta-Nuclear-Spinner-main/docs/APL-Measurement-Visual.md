# APL Measurement Operator Visual

```mermaid
graph LR
    Psi["|ψ⟩"] -->|Measurement\nP_μ = |ϕ_μ⟩⟨ϕ_μ| | CollapseSingle["|ϕ_μ⟩ (up to phase)"]
    Psi -->|Degenerate\nsubspace P | CollapseSubspace["P|ψ⟩ / √⟨ψ|P|ψ⟩"]
    CollapseSubspace -->|Born probability\n⟨ψ|P|ψ⟩ | Normalization["Normalize"]
```

- **Single eigenstate collapse:** `|ψ⟩ → |ϕμ⟩`, with projector `Pμ = |ϕμ⟩⟨ϕμ|`.
- **Subspace collapse:** `|ψ⟩ → P|ψ⟩ / √⟨ψ|P|ψ⟩` using projector `P`.
- Born probability `⟨ψ|P|ψ⟩` supplies the normalization factor.
