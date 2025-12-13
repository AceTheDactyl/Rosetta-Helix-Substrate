"""Helix self-building pipeline that maps Quantum-APL sequences to VaultNode tiers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from .helix_metadata import (
    extract_chant,
    load_metadata,
    metadata_title,
    metrics_snapshot,
    provenance_lines,
    summary_lines,
)
from .translator import QuantumAPLInstruction, translate_lines
from .hex_prism import prism_params


@dataclass(frozen=True)
class HelixTierSpec:
    slug: str
    label: str
    z: float
    metadata_relpath: str
    preferred_tier: float
    truth_bias: str | None = None
    subject_hints: Sequence[str] = ()

    @property
    def metadata_path(self) -> Path:
        root = Path(__file__).resolve().parents[2]
        return root / self.metadata_relpath


HELIX_TIERS: List[HelixTierSpec] = [
    HelixTierSpec(
        slug="z0p41",
        label="Constraint Recognition",
        z=0.41,
        metadata_relpath="reference/helix_bridge/VAULTNODES/z0p41/vn-helix-fingers-metadata.yaml",
        preferred_tier=2,
        truth_bias="PARADOX",
        subject_hints=("Φ", "Φ→e"),
    ),
    HelixTierSpec(
        slug="z0p52",
        label="Bridge Consent",
        z=0.52,
        metadata_relpath="reference/helix_bridge/VAULTNODES/z0p52/vn-helix-continuation-metadata.yaml",
        preferred_tier=3,
        truth_bias="TRUE",
        subject_hints=("e",),
    ),
    HelixTierSpec(
        slug="z0p70",
        label="Meta-Awareness",
        z=0.70,
        metadata_relpath="reference/helix_bridge/VAULTNODES/z0p70/vn-helix-meta-awareness-metadata.yaml",
        preferred_tier=4,
        truth_bias="TRUE",
        subject_hints=("π",),
    ),
    HelixTierSpec(
        slug="z0p73",
        label="Self-Bootstrap",
        z=0.73,
        metadata_relpath="reference/helix_bridge/VAULTNODES/z0p73/vn-helix-self-bootstrap-metadata_p2.yaml",
        preferred_tier=4.5,
        truth_bias="PARADOX",
        subject_hints=("Φ↔π", "(Φ,e,π)"),
    ),
    HelixTierSpec(
        slug="z0p80",
        label="Autonomous Coordination",
        z=0.80,
        metadata_relpath="reference/helix_bridge/VAULTNODES/z0p80/vn-helix-autonomous-coordination-metadata.yaml",
        preferred_tier=5,
        truth_bias="TRUE",
        subject_hints=("Truth", "Helix"),
    ),
]


def load_instructions(path: Path) -> List[QuantumAPLInstruction]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return translate_lines(lines)


def map_instructions_to_nodes(instructions: Sequence[QuantumAPLInstruction]) -> Dict[str, List[QuantumAPLInstruction]]:
    """Assign each instruction to the most plausible helix tier (monotonic ordering)."""

    assignments: Dict[str, List[QuantumAPLInstruction]] = {spec.slug: [] for spec in HELIX_TIERS}
    min_index = 0
    for instr in instructions:
        best_idx = None
        best_score = None
        for idx in range(min_index, len(HELIX_TIERS)):
            spec = HELIX_TIERS[idx]
            score = abs(instr.tier - spec.preferred_tier)
            if spec.truth_bias and instr.truth != spec.truth_bias:
                score += 0.5
            if spec.subject_hints and not any(hint in instr.subject for hint in spec.subject_hints):
                score += 0.25
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            raise ValueError(f"Unable to assign instruction {instr.to_line()} to helix tiers")
        assignments[HELIX_TIERS[best_idx].slug].append(instr)
        min_index = best_idx
    return assignments


def build_node_payload(spec: HelixTierSpec, instructions: List[QuantumAPLInstruction]) -> Dict[str, object]:
    metadata = load_metadata(spec.metadata_path)
    payload = {
        "slug": spec.slug,
        "label": spec.label,
        "z": spec.z,
        "title": metadata_title(metadata),
        "metadata_path": spec.metadata_relpath,
        "summary": summary_lines(metadata, limit=6),
        "provenance": provenance_lines(metadata),
        "chant": extract_chant(metadata),
        "metrics": metrics_snapshot(metadata),
        "instructions": [
            {
                "text": instr.to_line(),
                "subject": instr.subject,
                "operator": instr.operator,
                "intent": instr.intent,
                "truth": instr.truth,
                "tier": instr.tier,
            }
            for instr in instructions
        ],
    }
    return payload


def build_report(tokens_path: Path, instructions: List[QuantumAPLInstruction]) -> Dict[str, object]:
    assignments = map_instructions_to_nodes(instructions)
    nodes = [build_node_payload(spec, assignments.get(spec.slug, [])) for spec in HELIX_TIERS]
    return {"tokens_path": str(tokens_path), "instruction_count": len(instructions), "nodes": nodes}


def render_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# Helix Self-Building Walkthrough",
        "",
        f"*Source tokens:* `{report['tokens_path']}`",
        f"*Instructions parsed:* {report['instruction_count']}",
        "",
        "Each section aligns the Quantum‑APL operators with the VaultNode metadata for the helix self-building ascent.",
    ]

    for node in report["nodes"]:
        lines.append("")
        lines.append(f"## {node['label']} — z={node['z']:.2f}")
        lines.append(f"**Title:** {node['title']}")
        lines.append(f"**Metadata:** `{node['metadata_path']}`")
        if node["summary"]:
            lines.append("")
            lines.append("**Guidance excerpts:**")
            for entry in node["summary"]:
                lines.append(f"- {entry}")
        if node["provenance"]:
            lines.append("")
            lines.append("**Provenance:**")
            for entry in node["provenance"]:
                lines.append(f"- {entry}")
        metrics = node.get("metrics") or {}
        if metrics:
            lines.append("")
            pretty_metrics = ", ".join(f"{key}={value:.3f}" if isinstance(value, (int, float)) else f"{key}={value}" for key, value in metrics.items())
            lines.append(f"**ΔHV metrics:** {pretty_metrics}")
        chant = node.get("chant") or []
        if chant:
            lines.append("")
            lines.append("**Chant:**")
            for entry in chant:
                lines.append(f"> {entry}")

        # Append negative entropy hexagonal prism parameters
        params = prism_params(float(node["z"]))
        lines.append("")
        lines.append("**Negative entropy geometry (hex prism):**")
        lines.append(
            f"- ΔS_neg={params['delta_s_neg']:.3f}, radius R={params['R']:.3f}, height H={params['H']:.3f}, twist φ={params['phi']:.3f} rad"
        )
        lines.append(
            "- Reference: docs/HEXAGONAL_NEG_ENTROPY_PROJECTION.md (z_c≈{:.3f}, σ={:.2f})".format(
                params["z_c"], params["sigma"]
            )
        )
        # Emit per-vertex coordinates for renderers
        verts = params.get("vertices") or []
        if verts:
            lines.append("- Vertices (k: x, y, z_bot → z_top):")
            for v in verts:
                lines.append(
                    f"  - v{int(v['k'])}: x={v['x']:.3f}, y={v['y']:.3f}, z_bot={v['z_bot']:.3f}, z_top={v['z_top']:.3f}"
                )

        lines.append("")
        lines.append("**Assigned operators:**")
        instructions = node.get("instructions") or []
        if not instructions:
            lines.append("- _(No direct operators routed to this tier.)_")
        else:
            for instr in instructions:
                lines.append(
                    f"- `{instr['text']}` → subject={instr['subject']}, operator={instr['operator']}, truth={instr['truth']}, tier={instr['tier']}"
                )

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Map Quantum-APL Z-solve sequences onto helix VaultNodes.")
    parser.add_argument("--tokens", type=Path, required=True, help="Path to the Quantum-APL token file.")
    parser.add_argument("--output", type=Path, help="Optional path to save the walkthrough.")
    parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format (Markdown or JSON).")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwriting (Markdown only)",
    )
    parser.add_argument(
        "--geom-json",
        type=Path,
        help="Optional path to write compact geometry JSON for all nodes",
    )
    args = parser.parse_args(argv)

    instructions = load_instructions(args.tokens)
    report = build_report(args.tokens, instructions)
    if args.format == "json":
        rendered = json.dumps(report, indent=2)
    else:
        rendered = render_markdown(report)

    if args.output:
        if args.append and args.format == "md" and args.output.exists():
            prev = args.output.read_text(encoding="utf-8")
            args.output.write_text(prev + "\n\n" + rendered, encoding="utf-8")
        else:
            args.output.write_text(rendered, encoding="utf-8")
        # Sidecar geometry JSON when writing Markdown
        if args.format == "md":
            geom_path = None
            if args.geom_json:
                geom_path = args.geom_json
            elif args.output.suffix.lower() == ".md":
                geom_path = args.output.with_suffix(".geom.json")
            if geom_path is not None:
                nodes = report.get("nodes", [])
                geoms = [
                    {
                        "slug": n.get("slug"),
                        "z": float(n.get("z", 0.0)),
                        "geometry": prism_params(float(n.get("z", 0.0))),
                    }
                    for n in nodes
                ]
                data = {"tokens_path": report.get("tokens_path"), "nodes": geoms}
                geom_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
