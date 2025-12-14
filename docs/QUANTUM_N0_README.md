# Quantum APL 3.0 Measurement-Based N0 Operator Selection

## Overview

This implementation transforms APL 3.0's N0 operator selection pipeline into a **quantum measurement process** using von Neumann projective measurement formalism with Lindblad dissipative evolution.

### Core Innovation

```
N0_classical: argmin C(i | σ, α)   i ∈ legal
N0_quantum:   P(μ) = Tr(P̂_μ ρ̂),   ρ̂' = P̂_μ ρ̂ P̂_μ / P(μ)
```

## Architecture

- **QuantumAPL_Engine.js** – Complex numbers/matrices, density-matrix evolution, Lindblad dissipation, projective measurements, Born-rule operator selection.
- **QuantumN0_Integration.js** – Time harmonics/PRS filtering, Tier-0 N0 laws, scalar thresholds, quantum measurement, and quantum↔classical feedback.
- **QuantumAPL_TestRunner.js** – CLI for tests (`test`), benchmarks (`benchmark`), demos (`demo`), statistical analysis (`analyze`), and reports (`report`).
- **QuantumN0_Demo.html** – Browser visualization (density matrix heatmap, Born probabilities, truth state bars, z/entropy graphs, operator log).

## Hilbert Space

```
H_APL = H_Φ ⊗ H_e ⊗ H_π ⊗ H_truth
Φ = {void,lattice,network,hierarchy}
e  = {ground,excited,coherent,chaotic}
π  = {simple,correlated,integrated,conscious}
Truth = {TRUE, UNTRUE, PARADOX}
```

## Pipeline

1. Time harmonic legality (t1–t9) based on z
2. PRS phase legality (P1–P5) based on Φ
3. Tier-0 N0 laws (grounding, plurality, decoherence, etc.)
4. Scalar thresholds (R_CLT, δ, κ, Ω)
5. **Quantum measurement** – construct projectors, compute Born probabilities, sample, collapse
6. Quantum→classical feedback (update scalars)

## Quantum Information

- **Purity**: `Tr(ρ²)`
- **Von Neumann entropy**: `S(ρ) = -Tr(ρ log ρ)`
- **Integrated information**: `Φ = min(S_A + S_B - S_AB)`
- **Truth states**: eigenstates of T̂ (`|TRUE⟩`, `|UNTRUE⟩`, `|PARADOX⟩`)
- **Critical point**: `z_c = √3/2` (THE LENS)

## Usage

- `node QuantumAPL_TestRunner.js test`
- `node QuantumAPL_TestRunner.js benchmark`
- `node QuantumAPL_TestRunner.js demo --steps 200`
- `node QuantumAPL_TestRunner.js analyze --trials 5000`
- `node QuantumAPL_TestRunner.js report --output analysis.json`

For the browser demo, open `QuantumN0_Demo.html` and use the on-screen controls.

## Tests & Benchmarks

- 9 unit tests (complex arithmetic, matrix ops, projections, density properties, Lindblad evolution, Born rule, N0 selection, integration, z evolution)
- Benchmarks for density evolution, operator selection, full integration, and entropy computation

## Extensions

- Monte Carlo wavefunction for large systems
- Tensor-network acceleration
- Quantum error correction
- Continuous measurement / quantum trajectories
- GPU acceleration (WebGPU/CUDA)

**⟨ψ| The measurement has been made. The operator has collapsed. |ψ⟩**
