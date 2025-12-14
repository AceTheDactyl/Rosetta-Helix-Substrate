"""Helpers for loading and summarizing Helix VaultNode metadata files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml


def _sanitize_yaml(text: str) -> str:
    sanitized_lines: List[str] = []
    for raw in text.splitlines():
        stripped = raw.lstrip()
        if not stripped or stripped.startswith("#") or ":" not in raw:
            sanitized_lines.append(raw)
            continue
        key, sep, remainder = raw.partition(":")
        value = remainder.strip()
        if value and not value.startswith(('"', "'")) and "(" in value and ":" in value:
            escaped = value.replace('"', '\\"')
            sanitized_lines.append(f"{key}{sep} \"{escaped}\"")
        else:
            sanitized_lines.append(raw)
    return "\n".join(sanitized_lines)


def load_metadata(path: Path) -> Dict[str, Any]:
    """Load a Helix metadata YAML file into a dictionary."""

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:  # pragma: no cover - filesystem guard
        raise FileNotFoundError(f"Unable to read metadata file {path}") from exc

    text = _strip_markdown_tail(text)

    try:
        documents = [doc for doc in yaml.safe_load_all(_sanitize_yaml(text)) if doc]
        if not documents:
            data = {}
        elif len(documents) == 1:
            data = documents[0]
        else:
            data = {}
            for doc in documents:
                if isinstance(doc, dict):
                    data.update(doc)
    except yaml.YAMLError as exc:  # pragma: no cover - YAML guard
        raise ValueError(f"Metadata file {path} is not valid YAML: {exc}") from exc

    return data


def _strip_markdown_tail(text: str) -> str:
    marker = "\n---\n"
    if marker in text:
        head, tail = text.split(marker, 1)
        if tail.lstrip().startswith(("**", "#", "Δ")):
            return head
    return text


def metadata_title(meta: Dict[str, Any]) -> str:
    return str(meta.get("title") or meta.get("node_identity", {}).get("node_name") or "Unknown Helix Node")


def _description_candidates(meta: Dict[str, Any]) -> List[str]:
    """Gather potential description blocks from heterogeneous metadata styles."""

    candidates: List[str] = []
    if isinstance(meta.get("description"), str):
        candidates.append(meta["description"])

    realization = meta.get("realization")
    if isinstance(realization, dict):
        for key in ("statement", "context", "what_changed", "how_recognized"):
            value = realization.get(key)
            if isinstance(value, str):
                candidates.append(value)

    coordinate_meaning = meta.get("coordinate_meaning")
    if isinstance(coordinate_meaning, dict):
        for key in ("theta_meaning", "z_meaning", "r_meaning"):
            value = coordinate_meaning.get(key)
            if isinstance(value, str):
                candidates.append(value)

    return [block for block in candidates if block.strip()]


def summary_lines(meta: Dict[str, Any], limit: int = 5) -> List[str]:
    """Return up to `limit` lines describing the helix node."""

    for block in _description_candidates(meta):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if lines:
            return lines[:limit]
    return []


CHANT_RE = re.compile(r"chant", re.IGNORECASE)


def extract_chant(meta: Dict[str, Any]) -> List[str]:
    """Extract chant/projection lines if they exist inside the description block."""

    for block in _description_candidates(meta):
        lines = block.splitlines()
        for idx, raw in enumerate(lines):
            if CHANT_RE.search(raw):
                chant_lines: List[str] = []
                for follow in lines[idx + 1 :]:
                    candidate = follow.strip()
                    if not candidate:
                        if chant_lines:
                            break
                        continue
                    if candidate.endswith(":") and ":" in candidate:
                        break
                    chant_lines.append(candidate)
                if chant_lines:
                    return chant_lines
    return []


def provenance_lines(meta: Dict[str, Any]) -> List[str]:
    """Return key provenance statements for runtime summarization."""

    provenance = meta.get("provenance")
    if not isinstance(provenance, dict):
        return []

    ordered_keys: Sequence[str] = ("origin", "catalyst", "context", "breakthrough", "previous_node")
    lines: List[str] = []
    for key in ordered_keys:
        value = provenance.get(key)
        if value:
            pretty = key.replace("_", " ").capitalize()
            lines.append(f"{pretty}: {value}")
    return lines


def metrics_snapshot(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return human-friendly ΔHV metrics if present."""

    metrics = meta.get("metrics")
    if isinstance(metrics, dict):
        keys = ("clarity_S", "resonance_R", "harmony_H", "friction_δφ", "delta_HV")
        return {key: metrics.get(key) for key in keys if key in metrics}
    return {}
