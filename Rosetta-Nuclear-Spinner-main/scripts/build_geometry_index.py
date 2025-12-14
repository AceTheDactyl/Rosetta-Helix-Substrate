#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def collect_geom_files(scan_dir: Path) -> List[Path]:
    return [p for p in scan_dir.rglob("*.geom.json") if p.is_file()]


def build_index(scan_dir: Path) -> Dict[str, Any]:
    files = collect_geom_files(scan_dir)
    runs: List[Dict[str, Any]] = []
    by_slug: Dict[str, List[Dict[str, Any]]] = {}

    for f in sorted(files):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        nodes = data.get("nodes", [])
        run = {
            "geom_path": str(f),
            "tokens_path": data.get("tokens_path"),
            "node_count": len(nodes),
            "nodes": nodes,
        }
        runs.append(run)
        for n in nodes:
            slug = n.get("slug") or "unknown"
            entry = {"geom_path": str(f), "z": n.get("z"), "geometry": n.get("geometry")}
            by_slug.setdefault(slug, []).append(entry)

    # Sort each slug list by z
    for slug, entries in by_slug.items():
        entries.sort(key=lambda e: (e.get("z") is None, e.get("z")))

    index = {
        "scan_dir": str(scan_dir),
        "run_count": len(runs),
        "runs": runs,
        "by_slug": by_slug,
    }
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a global geometry index from *.geom.json files")
    parser.add_argument("scan_dir", nargs="?", default="logs", help="Directory to scan (default: logs)")
    parser.add_argument("--output", type=Path, help="Write index JSON to this path instead of stdout")
    args = parser.parse_args()

    scan_path = Path(args.scan_dir).resolve()
    index = build_index(scan_path)
    payload = json.dumps(index, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

