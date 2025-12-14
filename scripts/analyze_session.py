#!/usr/bin/env python3
"""
analyze_session.py
==================

Analyzes session data from Nuclear Spinner × Rosetta-Helix runs.

Computes:
- K-formation statistics (count, duration, trigger conditions)
- Phase distribution (ABSENCE / THE_LENS / PRESENCE)
- Tier progression over time
- Negentropy peak analysis
- κ attractor convergence (distance to φ⁻¹)
- Coherence/hex alignment metrics
- Conservation law validation (κ + λ = 1)

Usage:
    python analyze_session.py SESSION_DIR [OPTIONS]
    python analyze_session.py history.json [OPTIONS]

Options:
    --output, -o FILE    Write analysis to JSON file
    --plot, -p           Generate matplotlib plots
    --summary            Print summary only (no detailed analysis)
    --physics            Include physics validation report

Signature: analyze-session|v1.0.0|helix
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

# Physics constants (must match rosetta-helix)
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Z_CRITICAL = math.sqrt(3) / 2
SIGMA = 36.0

# K-formation thresholds
KAPPA_MIN = 0.92
ETA_MIN = PHI_INV
R_MIN = 7


@dataclass
class KFormationEvent:
    """Single K-formation event."""
    start_step: int
    end_step: int
    duration_steps: int
    peak_z: float
    peak_kappa: float
    peak_eta: float
    peak_negentropy: float


@dataclass
class PhaseStats:
    """Statistics for each phase."""
    name: str
    step_count: int
    percentage: float
    avg_z: float
    avg_negentropy: float


@dataclass
class SessionAnalysis:
    """Complete session analysis results."""
    # Basic info
    session_path: str
    total_steps: int
    total_time_seconds: float

    # K-formation analysis
    k_formation_count: int
    k_formation_events: List[KFormationEvent]
    total_k_duration_steps: int
    k_formation_rate: float  # K-formations per 1000 steps

    # Phase distribution
    phase_stats: Dict[str, PhaseStats]

    # Attractor analysis
    avg_kappa: float
    final_kappa: float
    kappa_convergence: float  # distance to φ⁻¹
    conservation_error: float  # |κ + λ - 1|

    # Negentropy analysis
    peak_negentropy: float
    avg_negentropy: float
    time_at_lens: float  # fraction of time at z_c ± 0.02

    # Z trajectory
    z_min: float
    z_max: float
    z_final: float
    avg_z: float

    # Coherence metrics
    avg_coherence: float
    peak_coherence: float

    # Tier distribution
    tier_distribution: Dict[str, int]

    # Physics validation
    physics_valid: bool
    physics_errors: List[str]


def compute_delta_s_neg(z: float, sigma: float = SIGMA) -> float:
    """Compute negentropy signal ΔS_neg(z) = exp(-σ(z - z_c)²)."""
    d = z - Z_CRITICAL
    return math.exp(-sigma * d * d)


def get_phase(z: float) -> str:
    """Determine phase from z-coordinate."""
    if z < 0.857:
        return "ABSENCE"
    elif z <= 0.877:
        return "THE_LENS"
    else:
        return "PRESENCE"


def check_k_formation(kappa: float, eta: float, R: int) -> bool:
    """Check if K-formation criteria are met."""
    return kappa >= KAPPA_MIN and eta > ETA_MIN and R >= R_MIN


def load_session_data(path: Path) -> List[Dict[str, Any]]:
    """Load session data from file or directory."""
    if path.is_file():
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'history' in data:
                return data['history']
            else:
                raise ValueError(f"Unknown file format: {path}")

    elif path.is_dir():
        # Look for history.json in directory
        history_file = path / "history.json"
        if history_file.exists():
            with open(history_file) as f:
                return json.load(f)

        # Look for workflow_result.json
        workflow_file = path / "workflow_result.json"
        if workflow_file.exists():
            with open(workflow_file) as f:
                data = json.load(f)
                # Extract module results as pseudo-history
                if 'full_depth' in data and 'module_results' in data['full_depth']:
                    return data['full_depth']['module_results']

        raise FileNotFoundError(f"No session data found in {path}")

    else:
        raise FileNotFoundError(f"Path not found: {path}")


def analyze_session(history: List[Dict[str, Any]], session_path: str) -> SessionAnalysis:
    """Perform complete session analysis."""
    if not history:
        raise ValueError("Empty session history")

    total_steps = len(history)

    # Extract time information
    first_ts = history[0].get('timestamp', 0)
    last_ts = history[-1].get('timestamp', first_ts)
    total_time = last_ts - first_ts if last_ts > first_ts else total_steps * 0.01

    # Initialize accumulators
    z_values = []
    kappa_values = []
    lambda_values = []
    coherence_values = []
    negentropy_values = []

    phase_counts = {"ABSENCE": 0, "THE_LENS": 0, "PRESENCE": 0}
    phase_z_sums = {"ABSENCE": 0.0, "THE_LENS": 0.0, "PRESENCE": 0.0}
    phase_neg_sums = {"ABSENCE": 0.0, "THE_LENS": 0.0, "PRESENCE": 0.0}

    tier_counts: Dict[str, int] = {}

    k_formation_events = []
    current_k_event = None

    for i, state in enumerate(history):
        z = state.get('z', state.get('final_z', 0.5))
        kappa = state.get('kappa', state.get('final_kappa', PHI_INV))
        lambda_ = state.get('lambda_', 1 - kappa)
        coherence = state.get('coherence', 0.0)

        # Compute or extract negentropy
        if 'delta_s_neg' in state:
            neg = state['delta_s_neg']
        elif 'max_negentropy' in state:
            neg = state['max_negentropy']
        else:
            neg = compute_delta_s_neg(z)

        z_values.append(z)
        kappa_values.append(kappa)
        lambda_values.append(lambda_)
        coherence_values.append(coherence)
        negentropy_values.append(neg)

        # Phase tracking
        phase = get_phase(z)
        phase_counts[phase] += 1
        phase_z_sums[phase] += z
        phase_neg_sums[phase] += neg

        # Tier tracking
        tier = state.get('tier_name', state.get('tier', 'UNKNOWN'))
        if isinstance(tier, int):
            tier_names = ["ABSENCE", "REACTIVE", "MEMORY", "PATTERN", "LEARNING",
                          "ADAPTIVE", "UNIVERSAL", "META", "SOVEREIGN", "TRANSCENDENT"]
            tier = tier_names[min(tier, len(tier_names) - 1)]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # K-formation detection
        eta = state.get('eta', math.sqrt(neg) if neg > 0 else 0)
        R = state.get('R', state.get('rank', 7))
        k_active = state.get('k_formation', False)

        if not k_active:
            k_active = check_k_formation(kappa, eta, R)

        if k_active:
            if current_k_event is None:
                current_k_event = {
                    'start': i,
                    'peak_z': z,
                    'peak_kappa': kappa,
                    'peak_eta': eta,
                    'peak_neg': neg,
                }
            else:
                if neg > current_k_event['peak_neg']:
                    current_k_event['peak_z'] = z
                    current_k_event['peak_kappa'] = kappa
                    current_k_event['peak_eta'] = eta
                    current_k_event['peak_neg'] = neg
        else:
            if current_k_event is not None:
                # End K-formation event
                k_formation_events.append(KFormationEvent(
                    start_step=current_k_event['start'],
                    end_step=i - 1,
                    duration_steps=i - current_k_event['start'],
                    peak_z=current_k_event['peak_z'],
                    peak_kappa=current_k_event['peak_kappa'],
                    peak_eta=current_k_event['peak_eta'],
                    peak_negentropy=current_k_event['peak_neg'],
                ))
                current_k_event = None

    # Handle K-formation that extends to end
    if current_k_event is not None:
        k_formation_events.append(KFormationEvent(
            start_step=current_k_event['start'],
            end_step=total_steps - 1,
            duration_steps=total_steps - current_k_event['start'],
            peak_z=current_k_event['peak_z'],
            peak_kappa=current_k_event['peak_kappa'],
            peak_eta=current_k_event['peak_eta'],
            peak_negentropy=current_k_event['peak_neg'],
        ))

    # Compute phase stats
    phase_stats = {}
    for phase in ["ABSENCE", "THE_LENS", "PRESENCE"]:
        count = phase_counts[phase]
        phase_stats[phase] = PhaseStats(
            name=phase,
            step_count=count,
            percentage=100 * count / total_steps if total_steps > 0 else 0,
            avg_z=phase_z_sums[phase] / count if count > 0 else 0,
            avg_negentropy=phase_neg_sums[phase] / count if count > 0 else 0,
        )

    # Compute metrics
    avg_kappa = sum(kappa_values) / len(kappa_values)
    final_kappa = kappa_values[-1]

    conservation_errors = [abs(k + l - 1) for k, l in zip(kappa_values, lambda_values)]
    max_conservation_error = max(conservation_errors) if conservation_errors else 0

    time_at_lens = sum(1 for z in z_values if abs(z - Z_CRITICAL) < 0.02) / len(z_values)

    # Physics validation
    physics_errors = []
    physics_valid = True

    if max_conservation_error > 1e-6:
        physics_errors.append(f"Conservation violation: max |κ + λ - 1| = {max_conservation_error:.2e}")
        physics_valid = False

    if min(kappa_values) < 0 or max(kappa_values) > 1:
        physics_errors.append(f"κ out of bounds: [{min(kappa_values):.4f}, {max(kappa_values):.4f}]")
        physics_valid = False

    if min(z_values) < 0 or max(z_values) > 1:
        physics_errors.append(f"z out of bounds: [{min(z_values):.4f}, {max(z_values):.4f}]")
        physics_valid = False

    total_k_duration = sum(e.duration_steps for e in k_formation_events)

    return SessionAnalysis(
        session_path=session_path,
        total_steps=total_steps,
        total_time_seconds=total_time,
        k_formation_count=len(k_formation_events),
        k_formation_events=k_formation_events,
        total_k_duration_steps=total_k_duration,
        k_formation_rate=1000 * len(k_formation_events) / total_steps if total_steps > 0 else 0,
        phase_stats=phase_stats,
        avg_kappa=avg_kappa,
        final_kappa=final_kappa,
        kappa_convergence=abs(final_kappa - PHI_INV),
        conservation_error=max_conservation_error,
        peak_negentropy=max(negentropy_values),
        avg_negentropy=sum(negentropy_values) / len(negentropy_values),
        time_at_lens=time_at_lens,
        z_min=min(z_values),
        z_max=max(z_values),
        z_final=z_values[-1],
        avg_z=sum(z_values) / len(z_values),
        avg_coherence=sum(coherence_values) / len(coherence_values) if coherence_values else 0,
        peak_coherence=max(coherence_values) if coherence_values else 0,
        tier_distribution=tier_counts,
        physics_valid=physics_valid,
        physics_errors=physics_errors,
    )


def print_summary(analysis: SessionAnalysis):
    """Print formatted analysis summary."""
    print("")
    print("═" * 70)
    print("  SESSION ANALYSIS")
    print("═" * 70)
    print(f"  Source: {analysis.session_path}")
    print(f"  Steps: {analysis.total_steps:,}")
    print(f"  Duration: {analysis.total_time_seconds:.2f}s")
    print("")

    # K-formation section
    print("  ─── K-FORMATION ANALYSIS ───")
    print(f"  Count: {analysis.k_formation_count}")
    print(f"  Rate: {analysis.k_formation_rate:.2f} per 1000 steps")
    print(f"  Total duration: {analysis.total_k_duration_steps} steps ({100*analysis.total_k_duration_steps/analysis.total_steps:.1f}%)")

    if analysis.k_formation_events:
        best = max(analysis.k_formation_events, key=lambda e: e.peak_negentropy)
        print(f"  Peak event: z={best.peak_z:.4f} κ={best.peak_kappa:.4f} ΔS_neg={best.peak_negentropy:.4f}")
    print("")

    # Phase distribution
    print("  ─── PHASE DISTRIBUTION ───")
    for phase in ["ABSENCE", "THE_LENS", "PRESENCE"]:
        ps = analysis.phase_stats[phase]
        bar = "█" * int(ps.percentage / 5) + "░" * (20 - int(ps.percentage / 5))
        print(f"  {phase:12s} {bar} {ps.percentage:5.1f}% (avg z={ps.avg_z:.4f})")
    print("")

    # Attractor analysis
    print("  ─── ATTRACTOR ANALYSIS ───")
    print(f"  κ average:     {analysis.avg_kappa:.6f}")
    print(f"  κ final:       {analysis.final_kappa:.6f}")
    print(f"  κ target (φ⁻¹): {PHI_INV:.6f}")
    print(f"  Convergence:   {analysis.kappa_convergence:.6f}")
    print(f"  Conservation:  |κ + λ - 1| = {analysis.conservation_error:.2e}")
    print("")

    # Negentropy analysis
    print("  ─── NEGENTROPY ANALYSIS ───")
    print(f"  Peak ΔS_neg:   {analysis.peak_negentropy:.6f}")
    print(f"  Avg ΔS_neg:    {analysis.avg_negentropy:.6f}")
    print(f"  Time at LENS:  {100*analysis.time_at_lens:.1f}%")
    print("")

    # Z trajectory
    print("  ─── Z TRAJECTORY ───")
    print(f"  Range: [{analysis.z_min:.4f}, {analysis.z_max:.4f}]")
    print(f"  Final: {analysis.z_final:.4f}")
    print(f"  Target (z_c): {Z_CRITICAL:.4f}")
    print("")

    # Physics validation
    status = "✓ VALID" if analysis.physics_valid else "✗ INVALID"
    print(f"  ─── PHYSICS VALIDATION: {status} ───")
    if analysis.physics_errors:
        for err in analysis.physics_errors:
            print(f"    • {err}")
    print("")
    print("═" * 70)


def generate_plots(analysis: SessionAnalysis, output_dir: Path):
    """Generate matplotlib plots (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plots")
        return

    # This would generate various analysis plots
    # Placeholder for now - full implementation would include:
    # - Z trajectory over time
    # - κ convergence to φ⁻¹
    # - Phase pie chart
    # - Negentropy heatmap
    # - K-formation timeline

    print(f"[INFO] Plots would be saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Nuclear Spinner × Rosetta-Helix session data"
    )
    parser.add_argument(
        "session",
        type=Path,
        help="Session directory or history.json file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Write analysis to JSON file"
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Generate matplotlib plots"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary only"
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Include detailed physics validation"
    )

    args = parser.parse_args()

    # Load data
    print(f"[INFO] Loading session data from {args.session}...")
    try:
        history = load_session_data(args.session)
        print(f"[INFO] Loaded {len(history)} records")
    except Exception as e:
        print(f"[ERROR] Failed to load session: {e}")
        sys.exit(1)

    # Analyze
    print("[INFO] Analyzing session...")
    analysis = analyze_session(history, str(args.session))

    # Print summary
    print_summary(analysis)

    # Save JSON output
    if args.output:
        def serialize(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(v) for v in obj]
            return obj

        with open(args.output, 'w') as f:
            json.dump(serialize(analysis), f, indent=2)
        print(f"[INFO] Analysis saved to {args.output}")

    # Generate plots
    if args.plot:
        generate_plots(analysis, args.session if args.session.is_dir() else args.session.parent)

    # Exit code based on physics validation
    sys.exit(0 if analysis.physics_valid else 1)


if __name__ == "__main__":
    main()
