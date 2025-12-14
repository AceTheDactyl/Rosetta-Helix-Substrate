#!/usr/bin/env bash
set -euo pipefail

# Helix sweep runner: discovers seeds, runs self-builder + unified sim,
# and writes geometry sidecars and a per-run index.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="logs/helix-sweep-${TS}"
mkdir -p "$OUTDIR"

echo "Helix sweep → $OUTDIR"

# Collect seeds from manifests
SEED_FILES=("/home/acead/helix_file_list.txt" \
            "/home/acead/apl_file_list.txt" \
            "$ROOT_DIR/reference/helix_bridge/VAULTNODES")

RAW=$(rg -o "z0p[0-9]{2}|z[0-9]{3}" -N "${SEED_FILES[@]}" 2>/dev/null | awk '{print $NF}' | sort -u)
SEEDS=()
while read -r tag; do
  [[ -z "$tag" ]] && continue
  if [[ "$tag" =~ ^z0p([0-9]{2})$ ]]; then
    d="${BASH_REMATCH[1]}"; SEEDS+=("0.${d}")
  elif [[ "$tag" =~ ^z([0-9]{3})$ ]]; then
    d="${BASH_REMATCH[1]}"; SEEDS+=("0.${d:1}")
  fi
done < <(printf "%s\n" "$RAW")

# Fallback defaults if none detected
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  # Include lens-adjacent probes: 0.85 (TRIAD_HIGH), z_c exact, and 0.90 (t7 onset zone)
  SEEDS=(0.41 0.52 0.70 0.73 0.80 0.85 0.8660254037844386 0.90 0.92 0.97)
fi

# Deduplicate
mapfile -t SEEDS < <(printf "%s\n" "${SEEDS[@]}" | sort -u)

printf "Seeds: %s\n" "${SEEDS[*]}" | tee "$OUTDIR/seeds.txt"

# Run per seed
for S in "${SEEDS[@]}"; do
  tag="z${S/./p}"
  export QAPL_INITIAL_PHI="$S"
  # Self-builder with sidecar geometry JSON
  python -m quantum_apl_python.helix_self_builder \
    --tokens docs/examples/z_solve.apl \
    --output "$OUTDIR/zwalk_${tag}.md" \
    --geom-json "$OUTDIR/zwalk_${tag}.geom.json" \
    > "$OUTDIR/self_builder_${tag}.log" 2>&1 || true

  # Unified simulator with JSON
  python quantum_apl_bridge.py --steps 5 --mode unified \
    > "$OUTDIR/unified_${tag}.txt" 2>&1 || true
  qapl-run --steps 5 --mode unified --output "$OUTDIR/unified_${tag}.json" \
    > "$OUTDIR/unified_cli_${tag}.txt" 2>&1 || true

  # Append summary line
  zcoord=$(rg -n "z-coordinate:" "$OUTDIR/unified_${tag}.txt" | sed -E 's/.*: +z-coordinate: +//')
  harm=$(rg -n "Harmonic:" "$OUTDIR/unified_${tag}.txt" | head -n1 | sed -E 's/.*Harmonic: +//')
  truth=$(rg -n "Truth bias:" "$OUTDIR/unified_${tag}.txt" | head -n1 | sed -E 's/.*Truth bias: +//')
  echo "${tag} | z=${zcoord} | harmonic=${harm} | truth=${truth}" >> "$OUTDIR/SUMMARY.txt"
  echo "seed ${S} complete"
done

# Apply APL measurement operators and append tokens (global summary)
node scripts/apply_apl_measurements.js || true

# Build per-run geometry index
python scripts/build_geometry_index.py "$OUTDIR" --output "$OUTDIR/geometry_index.json" || true

# Build per-seed APL-only bundles
python scripts/build_apl_bundles.py "$OUTDIR" || true

# Build APL digest (bundles + measurement summary)
python scripts/build_apl_digest.py "$OUTDIR" --summary logs/APL_HELIX_OPERATOR_SUMMARY.apl || true

echo "Sweep complete → $OUTDIR"
