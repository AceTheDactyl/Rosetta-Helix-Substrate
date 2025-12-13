#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from collections import OrderedDict
import os


def read_lines(p: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a canonical APL digest from per-seed bundles + measurement summary")
    ap.add_argument("sweep_dir", help="helix-sweep directory (uses apl_*.apl files)")
    ap.add_argument("--summary", default="logs/APL_HELIX_OPERATOR_SUMMARY.apl", help="Global APL summary to include")
    ap.add_argument("--output", help="Path to write digest (default: <sweep_dir>/apl_digest.apl)")
    args = ap.parse_args()

    sweep = Path(args.sweep_dir).resolve()
    if not sweep.exists():
        raise SystemExit(f"Sweep directory not found: {sweep}")

    tokens: "OrderedDict[str, None]" = OrderedDict()

    # 1) Per-seed bundles first (preserve seed ordering by filename)
    exp_ops = os.getenv('QAPL_EXPERIMENTAL_OPS', '').lower() in ('1', 'true', 'yes', 'y')

    for bundle in sorted(sweep.glob("apl_*.apl")):
        for ln in read_lines(bundle):
            norm = ln if exp_ops else ln.replace('⟂(subspace)', 'Π(subspace)').replace('⟂(ϕ_', 'T(ϕ_')
            tokens.setdefault(norm, None)

    # 2) Global measurement summary last
    summary = Path(args.summary).resolve()
    if summary.exists():
        for ln in read_lines(summary):
            norm = ln if exp_ops else ln.replace('⟂(subspace)', 'Π(subspace)').replace('⟂(ϕ_', 'T(ϕ_')
            tokens.setdefault(norm, None)

    out_path = Path(args.output).resolve() if args.output else (sweep / "apl_digest.apl")
    out_path.write_text("\n".join(tokens.keys()) + "\n", encoding="utf-8")
    print(f"Wrote digest: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
