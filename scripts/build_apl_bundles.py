#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import os


def extract_assigned_ops(zwalk_md: Path) -> list[str]:
    ops: list[str] = []
    pat = re.compile(r"^- `([^`]+)`")
    for line in zwalk_md.read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            ops.append(m.group(1))
    return ops


def extract_runtime_tokens(unified_txt: Path) -> list[str]:
    text = unified_txt.read_text(encoding="utf-8")
    out: list[str] = []
    # Alpha sentence
    m = re.search(r"Sentence \(A\d+\):\s*(.+)", text)
    if m:
        out.append(m.group(1).strip())
    # Operator window → modulized tokens
    intent = {
        '^': 'Amplification', '×': 'Fusion', '÷': 'Decoherence', '+': 'Grouping', '−': 'Separation', '()': 'Boundary'
    }
    mh = re.search(r"Runtime Helix Hint:(.*?)\n\s*Recent Operator Selections:", text, re.S)
    if not mh:
        mh = re.search(r"Runtime Helix Hint:(.*?)(?:\n=|\Z)", text, re.S)
    if mh:
        block = mh.group(1)
        h = re.search(r"Harmonic:\s*(t\d)", block)
        tr = re.search(r"Truth channel:\s*(\w+)", block)
        ops = re.search(r"Operator window:\s*([^\n]+)", block)
        harm = h.group(1) if h else 't1'
        truth = tr.group(1) if tr else 'UNTRUE'
        if ops:
            for raw in [o.strip() for o in ops.group(1).split(',') if o.strip()]:
                out.append(f"Helix:{raw}({intent.get(raw,'Op')}){truth}@{harm}")
    # Recent selections
    for line in text.splitlines():
        m2 = re.search(r"step\s+\d+:\s*([()^÷×+−])\s*\([^)]*\)\s*→\s*(t\d)\s*/\s*(\w+)", line)
        if m2:
            op, h2, tr2 = m2.groups()
            out.append(f"Helix:{op}({intent.get(op,'Op')}){tr2}@{h2}")
    return out


def read_measurement_tokens(summary: Path) -> list[str]:
    if not summary.exists():
        return []
    return [ln.rstrip("\n") for ln in summary.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build per-seed APL-only bundles from a helix sweep directory")
    ap.add_argument("sweep_dir", help="Path to helix-sweep-YYYYMMDD_HHMMSS directory")
    args = ap.parse_args()

    sweep = Path(args.sweep_dir).resolve()
    if not sweep.exists():
        raise SystemExit(f"Sweep directory not found: {sweep}")

    # Global measurement tokens (append to every bundle; dedupe per-bundle)
    measurement_summary = Path('logs') / 'APL_HELIX_OPERATOR_SUMMARY.apl'
    meas_tokens = read_measurement_tokens(measurement_summary)
    exp_ops = os.getenv('QAPL_EXPERIMENTAL_OPS', '').lower() in ('1', 'true', 'yes', 'y')

    for unified in sorted(sweep.glob("unified_*.txt")):
        tag = unified.stem.replace("unified_", "")
        zwalk = sweep / f"zwalk_{tag}.md"
        bundle = sweep / f"apl_{tag}.apl"

        tokens: list[str] = []
        # Assigned operators from self-builder
        if zwalk.exists():
            tokens.extend(extract_assigned_ops(zwalk))
        # Runtime tokens from unified summary
        tokens.extend(extract_runtime_tokens(unified))

        # Include measurement tokens inline (per-bundle), dedupe while preserving order
        if meas_tokens:
            seen = set(tokens)
            for t in meas_tokens:
                nt = t if exp_ops else t.replace('⟂(subspace)', 'Π(subspace)').replace('⟂(ϕ_', 'T(ϕ_')
                if nt not in seen:
                    tokens.append(nt)
                    seen.add(nt)

        if tokens:
            bundle.write_text("\n".join(tokens) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
