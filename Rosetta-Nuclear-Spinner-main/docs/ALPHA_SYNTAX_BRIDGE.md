# Alpha Programming Language Bridge

This repository now consumes the upstream Alpha Programming Language (APL) test pack
that lives at `/home/acead/Aces-Brain-Thpughts/APL`.  That directory contains the
canonical LaTeX sources (`apl-operators-manual.tex`, `apl-seven-sentences-test-pack.tex`)
and the HTML quick reference under `docs/index.html`.  During the workspace sweep we
located every helix/Z reference (see `rg -i "helix equation"` and `rg -i "z coordinate"`
outputs) so the Quantum-APL runtime can stay aligned with the VaultNode/Triad logs that
encode state as `(θ, z, r)` via the parametric helix `r(t) = (cos t, sin t, t)`.

## Imported Assets

The `Quantum-APL` folder already shipped PDF/TeX copies of the operator manual, so the
new integration focuses on structured data:

1. `src/quantum_apl_python/alpha_language.py` mirrors the operator table, field definitions,
   and the seven validation sentences from the Alpha test pack.
2. `src/quantum_apl_python/analyzer.py` now asks the helix mapper for a normalized `z`
   coordinate and synthesizes a legal Alpha sentence (token) that matches the
   recommended operator bundle.  Each simulation summary prints the sentence ID,
   operator semantics, and predicted regime beside the helix harmonic + truth channel.

These additions ensure that Quantum-APL operators/tokens are generated directly from
the same grammar used elsewhere in the workspace, rather than relying on bespoke
strings or forcing `TRUE`-biased measurements.

## How the Helix Equation Steers Tokens

1. A helix coordinate is built from `r(t) = (cos t, sin t, t)` and mapped into
   `z ∈ [0,1]` using the smooth tanh normalization described in `src/quantum_apl_python/helix.py`.
2. `HelixAPLMapper` locates the harmonic window (`t1`…`t9`) and the associated operator
   set for that window.  These windows were derived from the references flagged during
   the `rg -i "helix equation"` sweep (Triad logs, witness checklists, etc.).
3. `AlphaTokenSynthesizer` filters the Seven Sentence test pack for any entry whose
   operator matches the harmonic window and, if provided, a domain/machine hint.
4. The winning sentence becomes the surfaced token (for example, `u^ | Oscillator | wave`
   when the helix harmonic calls for amplification and the truth bias sits near z ≈ 0.7).

The net effect is that the Z coordinate—sourced from the canonical helix equation—now
drives both the physics-space recommendations and the Alpha Programming Language syntax.

## Future Data Sources

- If additional APL grammars appear elsewhere (for example in `/home/acead/TRIAD083 Multi Agent System`
  witness decks), add them to this registry rather than inventing new enums.
- The LaTeX manuals inside `Aces-Brain-Thpughts/APL` contain richer charts for UMOL
  states; port those tables into `alpha_language.py` if we need automated validation.
- When the JavaScript engine starts consuming helix-aware operator weights, pipe the
  `AlphaTokenSynthesizer` output to those scripts so CLI experiments and runtime
  behavior stay synchronized.

## Runtime Integration Status

- `QuantumAPL_Engine.js` and its CommonJS twin under `src/` now instantiate a
  `HelixOperatorAdvisor` that mirrors the same harmonic windows documented above.
- Every N0 operator selection call fetches the latest helix hints (via the z
  coordinate measured from the density matrix) and scales the projector weights
  so that only the operators endorsed by the Alpha grammar are favored. Truth
  channels modulate the gain, which prevents the runtime from biasing toward
  `TRUE` collapses when the helix recommends `UNTRUE`/`PARADOX` measurement modes.
- The selection API returns the helix hints alongside the operator probabilities
  so downstream logging/bridges can confirm which harmonic drove a given token.

### CLI Translator

For quick linting or to feed APL sentences into automation, use:

```bash
python -m quantum_apl_python.translator --text "Φ:M(stabilize)PARADOX@2" --pretty
```

This CLI (implemented in `src/quantum_apl_python/translator.py`) parses Quantum‑APL lines and emits structured JSON (`subject`, `operator`, `intent`, `truth`, `tier`). Add it to validation pipelines when new helix playbooks or operator configs are introduced.

Need to prove a whole Z-solve program threads through the helix properly? Chain the translator output into the helix self-building runner:

```bash
python -m quantum_apl_python.helix_self_builder \
  --tokens docs/examples/z_solve.apl \
  --output reference/helix_bridge/HELIX_Z_WALKTHROUGH.md
```

The runner maps each operator to the nearest VaultNode tier (z0p41 → z0p80) and injects provenance + chant snippets from the metadata so operators know **why** each elevation matters.
