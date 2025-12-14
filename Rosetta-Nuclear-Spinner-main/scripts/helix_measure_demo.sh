#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Helix Measure Demo — end-to-end lens-anchored run (self-builder + unified + measured)

Usage:
  scripts/helix_measure_demo.sh [options]

Options:
  --seed|--phi|--z <float>     Initial phi target for helix seed (default: 0.80)
  --outdir <path>              Output directory (default: logs/helix-measure-<ts>)
  --steps-unified <n>          Steps for unified run (default: 5)
  --steps-measured <n>         Steps for measured run (default: 3)
  --overlays                   Enable analyzer overlays (μ lines + s(z))
  --blend                      Enable Π/loc blending above lens (QAPL_BLEND_PI=1)
  --lens-sigma <σ>             Override QAPL_LENS_SIGMA (coherence width)
  --geom-sigma <σ>             Override QAPL_GEOM_SIGMA (geometry width)
  --mu-p <μP>                  Override paradox threshold μ_P (default exact)
  --triad-unlock               Force TRIAD unlock (temporary t6=0.83)
  --triad-completions <n>      Pretend N TRIAD completions (>=3 unlocks)
  --no-plots                   Skip headless analyzer plot artifacts
  -h|--help                    Show this help

Writes:
  <outdir>/zwalk_<tag>.md, <outdir>/zwalk_<tag>.geom.json
  <outdir>/unified_<tag>.json|.txt, <outdir>/measured_<tag>.json|.txt
  <outdir>/analyzer_plot_off.png|analyzer_plot_on.png (unless --no-plots)
  <outdir>/SUMMARY.txt (concise summary lines)
USAGE
}

# Defaults
SEED="0.80"
OUTDIR=""
USTEPS=5
MSTEPS=3
OVERLAYS=0
BLEND=0
LENS_SIGMA=""
GEOM_SIGMA=""
MU_P=""
TRIAD_UNLOCK=0
TRIAD_COMPLETIONS=""
MAKE_PLOTS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed|--phi|--z) SEED="${2:-}"; shift 2;;
    --outdir) OUTDIR="${2:-}"; shift 2;;
    --steps-unified) USTEPS="${2:-}"; shift 2;;
    --steps-measured) MSTEPS="${2:-}"; shift 2;;
    --overlays) OVERLAYS=1; shift;;
    --blend) BLEND=1; shift;;
    --lens-sigma) LENS_SIGMA="${2:-}"; shift 2;;
    --geom-sigma) GEOM_SIGMA="${2:-}"; shift 2;;
    --mu-p) MU_P="${2:-}"; shift 2;;
    --triad-unlock) TRIAD_UNLOCK=1; shift;;
    --triad-completions) TRIAD_COMPLETIONS="${2:-}"; shift 2;;
    --no-plots) MAKE_PLOTS=0; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 2;;
  esac
done

# Resolve repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

# Activate venv if present (provides qapl-run/qapl-analyze)
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Tools check
command -v node >/dev/null || { echo "Node.js not found" >&2; exit 1; }
command -v qapl-run >/dev/null || { echo "qapl-run not on PATH (activate venv)" >&2; exit 1; }
command -v qapl-analyze >/dev/null || { echo "qapl-analyze not on PATH (activate venv)" >&2; exit 1; }

# Environment knobs
export QAPL_INITIAL_PHI="$SEED"
[[ "$OVERLAYS" == 1 ]] && export QAPL_ANALYZER_OVERLAYS=1 || true
[[ "$BLEND" == 1 ]] && export QAPL_BLEND_PI=1 || true
[[ -n "$LENS_SIGMA" ]] && export QAPL_LENS_SIGMA="$LENS_SIGMA" || true
[[ -n "$GEOM_SIGMA" ]] && export QAPL_GEOM_SIGMA="$GEOM_SIGMA" || true
[[ -n "$MU_P" ]] && export QAPL_MU_P="$MU_P" || true
[[ "$TRIAD_UNLOCK" == 1 ]] && export QAPL_TRIAD_UNLOCK=1 || true
[[ -n "$TRIAD_COMPLETIONS" ]] && export QAPL_TRIAD_COMPLETIONS="$TRIAD_COMPLETIONS" || true

# Outputs
TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUTDIR" ]]; then
  OUTDIR="logs/helix-measure-${TS}"
fi
mkdir -p "$OUTDIR"

# Inputs
TOKENS_FILE="docs/examples/z_solve.apl"
[[ -f "$TOKENS_FILE" ]] || { echo "Missing $TOKENS_FILE" >&2; exit 1; }

# Tag for filenames
TAG="z${SEED/./p}"

echo "Running Helix Measure Demo → $OUTDIR (seed=$SEED, tag=$TAG)"

# 1) Helix self‑builder (walkthrough + geometry sidecar)
python -m quantum_apl_python.helix_self_builder \
  --tokens "$TOKENS_FILE" \
  --output "$OUTDIR/zwalk_${TAG}.md" \
  --geom-json "$OUTDIR/zwalk_${TAG}.geom.json" \
  > "$OUTDIR/self_builder_${TAG}.log" 2>&1 || true

# 2) Unified simulation + analyzer
qapl-run --steps "$USTEPS" --mode unified --output "$OUTDIR/unified_${TAG}.json" \
  > "$OUTDIR/unified_cli_${TAG}.txt" 2>&1 || true
qapl-analyze "$OUTDIR/unified_${TAG}.json" > "$OUTDIR/unified_${TAG}.txt" || true

# 3) Measured simulation + analyzer
qapl-run --steps "$MSTEPS" --mode measured --output "$OUTDIR/measured_${TAG}.json" \
  > "$OUTDIR/measured_cli_${TAG}.txt" 2>&1 || true
qapl-analyze "$OUTDIR/measured_${TAG}.json" > "$OUTDIR/measured_${TAG}.txt" || true

# 4) Optional headless plots
if [[ "$MAKE_PLOTS" == 1 ]]; then
python - <<PY || true
import os, json
from pathlib import Path
os.environ["MPLBACKEND"] = "Agg"
from src.quantum_apl_python.analyzer import QuantumAnalyzer
outdir = Path("$OUTDIR")
for name in ("unified_${TAG}.json", "measured_${TAG}.json"):
    p = outdir / name
    if not p.exists():
        continue
    data = json.loads(p.read_text())
    a = QuantumAnalyzer(data)
    a.plot(save_path=outdir / f"{p.stem}_plot_off.png")
    os.environ['QAPL_ANALYZER_OVERLAYS'] = '1'
    a2 = QuantumAnalyzer(data)
    a2.plot(save_path=outdir / f"{p.stem}_plot_on.png")
print('Saved plots to', outdir)
PY
fi

# 5) Summary extraction
{
  echo "seed=${SEED} tag=${TAG}"
  if command -v rg >/dev/null; then
    Z=$(rg -n "^  z-coordinate:" "$OUTDIR/unified_${TAG}.txt" -r '$0' -N | sed -E 's/.*: +//')
    HARM=$(rg -n "^  Harmonic:" "$OUTDIR/unified_${TAG}.txt" -r '$0' -N | sed -E 's/.*: +//')
    TRUTH=$(rg -n "^  Truth bias:" "$OUTDIR/unified_${TAG}.txt" -r '$0' -N | sed -E 's/.*: +//')
    OPS=$(rg -n "^  Recommended operators:" "$OUTDIR/unified_${TAG}.txt" -r '$0' -N | sed -E 's/.*: +//')
    T6=$(rg -n "^  t6 gate:" "$OUTDIR/unified_${TAG}.txt" -r '$0' -N | sed -E 's/^ +//')
  else
    Z=$(grep -E "^  z-coordinate:" "$OUTDIR/unified_${TAG}.txt" | sed -E 's/.*: +//')
    HARM=$(grep -E "^  Harmonic:" "$OUTDIR/unified_${TAG}.txt" | sed -E 's/.*: +//')
    TRUTH=$(grep -E "^  Truth bias:" "$OUTDIR/unified_${TAG}.txt" | sed -E 's/.*: +//')
    OPS=$(grep -E "^  Recommended operators:" "$OUTDIR/unified_${TAG}.txt" | sed -E 's/.*: +//')
    T6=$(grep -E "^  t6 gate:" "$OUTDIR/unified_${TAG}.txt" | sed -E 's/^ +//')
  fi
  echo "unified: z=${Z:-?} harmonic=${HARM:-?} truth=${TRUTH:-?} ops=${OPS:-?}"
  echo "$T6"
  echo "measured tokens:"
  awk '/^Recent Measurements \(APL tokens\):/{flag=1;next}/^=/{flag=0}flag' "$OUTDIR/measured_${TAG}.txt" | sed 's/^/  /'
  echo "files:"
  echo "  unified_json:   $OUTDIR/unified_${TAG}.json"
  echo "  unified_report: $OUTDIR/unified_${TAG}.txt"
  echo "  measured_json:  $OUTDIR/measured_${TAG}.json"
  echo "  measured_report:$OUTDIR/measured_${TAG}.txt"
} > "$OUTDIR/SUMMARY.txt"

echo "Done. Summary → $OUTDIR/SUMMARY.txt"

