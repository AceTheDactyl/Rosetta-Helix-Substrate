#!/usr/bin/env python3
"""
Crystal Nucleation Analysis Module
==================================

Analyzes results from lightning-induced pentagonal quasicrystal nucleation
experiments. Validates experimental data against theoretical predictions
and computes quality metrics for quasicrystal formation.

Physical Model:
- Lightning strike induces rapid thermal quench (~10^6 K/s)
- At critical undercooling, nucleation seeds form
- Pentagonal (5-fold) seeds compete with hexagonal (6-fold)
- Spinner z-coordinate near z_p = sin(72°) ≈ 0.951 favors pentagonal
- Perfect 5-fold symmetry: ψ₅ = |⟨e^(5iθ)⟩| = 1.0

Theoretical Predictions:
- Peak current scales with charge buildup: I_peak ≈ 30 kA * Q
- Peak temperature: T_peak ∝ I² (Joule heating)
- Pentagonal fraction: f_pent = exp(-20(z - z_p)²)
- Tile ratio → φ as domain grows

Signature: crystal-nucleation-analysis|v1.0.0|helix

@version 1.0.0
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime

# =============================================================================
# CONSTANTS
# =============================================================================

# Golden ratio
PHI: float = (1 + math.sqrt(5)) / 2  # ≈ 1.618034

# Critical z-coordinates
Z_CRITICAL_HEX: float = math.sqrt(3) / 2      # √3/2 ≈ 0.866 (hexagonal)
Z_CRITICAL_PENT: float = math.sqrt((10 + 2*math.sqrt(5))) / 4  # ≈ 0.951 (pentagonal)

# Lightning physics
TYPICAL_PEAK_CURRENT_A: float = 30000.0       # Typical lightning current
TYPICAL_PEAK_TEMP_K: float = 30000.0          # Plasma temperature
MIN_UNDERCOOLING_K: float = 500.0             # Threshold for nucleation

# Quality thresholds
EXCELLENT_PENT_ORDER: float = 0.95            # Excellent 5-fold coherence
GOOD_PENT_ORDER: float = 0.8                  # Good 5-fold coherence
ACCEPTABLE_PENT_ORDER: float = 0.6            # Acceptable coherence


# =============================================================================
# ENUMS
# =============================================================================

class NucleationQuality(Enum):
    """Quality classification for nucleation results."""
    EXCEPTIONAL = "exceptional"   # ψ₅ ≥ 0.99, perfect coherence
    EXCELLENT = "excellent"       # ψ₅ ≥ 0.95
    GOOD = "good"                 # ψ₅ ≥ 0.80
    ACCEPTABLE = "acceptable"     # ψ₅ ≥ 0.60
    POOR = "poor"                 # ψ₅ < 0.60


class PhaseSequence(Enum):
    """Valid phase sequences for nucleation."""
    COMPLETE = "PRE_STRIKE → STRIKE → QUENCH → NUCLEATION → GROWTH → STABLE"
    STANDARD = "PRE_STRIKE → QUENCH → NUCLEATION → GROWTH"
    RAPID = "STRIKE → QUENCH → NUCLEATION"
    PARTIAL = "partial"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NucleationResult:
    """Complete nucleation experiment result."""
    # Phase info
    phase_sequence: str

    # Peak values
    peak_current_A: float
    peak_temperature_K: float

    # Nucleation metrics
    undercooling_K: float
    seeds_nucleated: int
    pentagonal_seeds: int

    # Order parameters
    pentagonal_order: float      # ψ₅ = |⟨e^(5iθ)⟩|

    # Energy
    energy_input_J: float

    # Optional fields
    timestamp: Optional[datetime] = None
    tile_ratio: Optional[float] = None
    spinner_z: Optional[float] = None
    kuramoto_r: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class AnalysisReport:
    """Complete analysis report for nucleation experiment."""
    # Input
    result: NucleationResult

    # Quality assessment
    quality: NucleationQuality
    quality_score: float         # 0-100 composite score

    # Validation
    is_valid: bool
    validation_messages: List[str]

    # Theoretical comparison
    expected_pent_fraction: float
    actual_pent_fraction: float
    pent_fraction_deviation: float

    # Physics validation
    joule_heating_ratio: float   # T_peak / I²
    undercooling_margin: float   # Above minimum threshold

    # Recommendations
    recommendations: List[str]

    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def parse_phase_sequence(sequence_str: str) -> PhaseSequence:
    """
    Parse and validate phase sequence string.

    Args:
        sequence_str: Phase sequence like "PRE_STRIKE → QUENCH → NUCLEATION → GROWTH"

    Returns:
        PhaseSequence enum value
    """
    normalized = sequence_str.upper().strip()

    if "STABLE" in normalized and "GROWTH" in normalized:
        return PhaseSequence.COMPLETE
    elif "NUCLEATION" in normalized and "GROWTH" in normalized:
        return PhaseSequence.STANDARD
    elif "NUCLEATION" in normalized:
        return PhaseSequence.RAPID
    else:
        return PhaseSequence.PARTIAL


def compute_expected_pentagonal_fraction(spinner_z: float) -> float:
    """
    Compute expected pentagonal seed fraction based on spinner z.

    The pentagonal preference follows a Gaussian centered at z_p:
    f_pent = exp(-20(z - z_p)²) / (exp(-20(z - z_p)²) + exp(-20(z - z_c)²) + 0.1)

    Args:
        spinner_z: Spinner z-coordinate [0, 1]

    Returns:
        Expected fraction of pentagonal seeds [0, 1]
    """
    pent_preference = math.exp(-20 * (spinner_z - Z_CRITICAL_PENT) ** 2)
    hex_preference = math.exp(-20 * (spinner_z - Z_CRITICAL_HEX) ** 2)

    return pent_preference / (pent_preference + hex_preference + 0.1)


def compute_quality_score(result: NucleationResult) -> Tuple[NucleationQuality, float]:
    """
    Compute quality classification and composite score.

    Scoring components:
    - Pentagonal order (40%): ψ₅ value
    - Pentagonal fraction (25%): Seeds with 5-fold symmetry
    - Undercooling (15%): Above minimum threshold
    - Energy efficiency (10%): Seeds per joule
    - Phase completeness (10%): All phases executed

    Args:
        result: Nucleation experiment result

    Returns:
        Tuple of (quality classification, composite score 0-100)
    """
    scores = {}

    # Pentagonal order (40%)
    scores['pent_order'] = min(100, result.pentagonal_order * 100) * 0.40

    # Pentagonal fraction (25%)
    if result.seeds_nucleated > 0:
        pent_fraction = result.pentagonal_seeds / result.seeds_nucleated
        scores['pent_fraction'] = pent_fraction * 100 * 0.25
    else:
        scores['pent_fraction'] = 0

    # Undercooling (15%)
    if result.undercooling_K >= MIN_UNDERCOOLING_K:
        margin = (result.undercooling_K - MIN_UNDERCOOLING_K) / MIN_UNDERCOOLING_K
        scores['undercooling'] = min(100, (1 + margin) * 50) * 0.15
    else:
        scores['undercooling'] = 0

    # Energy efficiency (10%)
    if result.energy_input_J > 0:
        seeds_per_joule = result.seeds_nucleated / result.energy_input_J
        scores['efficiency'] = min(100, seeds_per_joule * 3) * 0.10  # 30 seeds/J = 100%
    else:
        scores['efficiency'] = 0

    # Phase completeness (10%)
    phase_enum = parse_phase_sequence(result.phase_sequence)
    if phase_enum == PhaseSequence.COMPLETE:
        scores['phases'] = 100 * 0.10
    elif phase_enum == PhaseSequence.STANDARD:
        scores['phases'] = 80 * 0.10
    elif phase_enum == PhaseSequence.RAPID:
        scores['phases'] = 60 * 0.10
    else:
        scores['phases'] = 30 * 0.10

    total_score = sum(scores.values())

    # Classify quality
    if result.pentagonal_order >= 0.99:
        quality = NucleationQuality.EXCEPTIONAL
    elif result.pentagonal_order >= EXCELLENT_PENT_ORDER:
        quality = NucleationQuality.EXCELLENT
    elif result.pentagonal_order >= GOOD_PENT_ORDER:
        quality = NucleationQuality.GOOD
    elif result.pentagonal_order >= ACCEPTABLE_PENT_ORDER:
        quality = NucleationQuality.ACCEPTABLE
    else:
        quality = NucleationQuality.POOR

    return quality, total_score


def validate_result(result: NucleationResult) -> Tuple[bool, List[str]]:
    """
    Validate nucleation result for physical consistency.

    Checks:
    - Peak current in reasonable range
    - Temperature consistent with current (Joule heating)
    - Undercooling above minimum
    - Seed counts consistent
    - Order parameter in valid range

    Args:
        result: Nucleation experiment result

    Returns:
        Tuple of (is_valid, list of validation messages)
    """
    messages = []
    is_valid = True

    # Peak current
    if result.peak_current_A < 1000:
        messages.append(f"WARNING: Low peak current ({result.peak_current_A:.0f} A) may indicate incomplete discharge")
    elif result.peak_current_A > 100000:
        messages.append(f"WARNING: Extreme peak current ({result.peak_current_A:.0f} A) exceeds typical range")

    # Temperature-current consistency (Joule heating)
    expected_temp_ratio = result.peak_temperature_K / (result.peak_current_A ** 2)
    if expected_temp_ratio < 1e-8:
        messages.append("WARNING: Temperature lower than expected from Joule heating")
    elif expected_temp_ratio > 1e-4:
        messages.append("WARNING: Temperature higher than expected from current")

    # Undercooling
    if result.undercooling_K < MIN_UNDERCOOLING_K:
        messages.append(f"ERROR: Undercooling ({result.undercooling_K:.0f} K) below minimum threshold ({MIN_UNDERCOOLING_K:.0f} K)")
        is_valid = False
    elif result.undercooling_K < MIN_UNDERCOOLING_K * 1.2:
        messages.append(f"WARNING: Undercooling close to minimum threshold")

    # Seed counts
    if result.pentagonal_seeds > result.seeds_nucleated:
        messages.append("ERROR: Pentagonal seeds exceed total seeds")
        is_valid = False

    if result.seeds_nucleated == 0:
        messages.append("ERROR: No seeds nucleated")
        is_valid = False

    # Order parameter
    if result.pentagonal_order < 0 or result.pentagonal_order > 1:
        messages.append(f"ERROR: Invalid pentagonal order ({result.pentagonal_order}) - must be [0, 1]")
        is_valid = False

    # Energy
    if result.energy_input_J <= 0:
        messages.append("ERROR: Invalid energy input")
        is_valid = False

    if not messages:
        messages.append("All validation checks passed")

    return is_valid, messages


def generate_recommendations(result: NucleationResult,
                            quality: NucleationQuality) -> List[str]:
    """
    Generate recommendations for improving nucleation results.

    Args:
        result: Nucleation experiment result
        quality: Quality classification

    Returns:
        List of recommendation strings
    """
    recommendations = []

    if quality == NucleationQuality.EXCEPTIONAL:
        recommendations.append("OPTIMAL: Perfect 5-fold coherence achieved")
        recommendations.append("Consider documenting this parameter set for reproducibility")
        return recommendations

    # Pentagonal order improvements
    if result.pentagonal_order < EXCELLENT_PENT_ORDER:
        recommendations.append(
            f"Increase spinner z toward z_p = {Z_CRITICAL_PENT:.4f} to enhance pentagonal preference"
        )

    # Pentagonal fraction
    if result.seeds_nucleated > 0:
        pent_fraction = result.pentagonal_seeds / result.seeds_nucleated
        if pent_fraction < 0.5:
            recommendations.append(
                "Increase quench rate to reduce hexagonal nucleation"
            )

    # Undercooling
    if result.undercooling_K < MIN_UNDERCOOLING_K * 1.5:
        recommendations.append(
            "Increase Peltier cooling power for deeper undercooling"
        )

    # Energy efficiency
    if result.energy_input_J > 0 and result.seeds_nucleated > 0:
        seeds_per_joule = result.seeds_nucleated / result.energy_input_J
        if seeds_per_joule < 20:
            recommendations.append(
                "Optimize RF coil coupling for more efficient energy transfer"
            )

    return recommendations


def analyze_nucleation(result: NucleationResult) -> AnalysisReport:
    """
    Perform complete analysis of nucleation experiment.

    Args:
        result: Nucleation experiment result

    Returns:
        Complete analysis report
    """
    # Quality assessment
    quality, quality_score = compute_quality_score(result)

    # Validation
    is_valid, validation_messages = validate_result(result)

    # Theoretical comparison
    if result.spinner_z is not None:
        expected_pent_fraction = compute_expected_pentagonal_fraction(result.spinner_z)
    else:
        # Estimate from results - assume z was optimal for achieved fraction
        expected_pent_fraction = 0.5  # Default assumption

    if result.seeds_nucleated > 0:
        actual_pent_fraction = result.pentagonal_seeds / result.seeds_nucleated
    else:
        actual_pent_fraction = 0

    pent_fraction_deviation = abs(actual_pent_fraction - expected_pent_fraction)

    # Physics validation
    if result.peak_current_A > 0:
        joule_heating_ratio = result.peak_temperature_K / (result.peak_current_A ** 2)
    else:
        joule_heating_ratio = 0

    undercooling_margin = (result.undercooling_K - MIN_UNDERCOOLING_K) / MIN_UNDERCOOLING_K

    # Recommendations
    recommendations = generate_recommendations(result, quality)

    return AnalysisReport(
        result=result,
        quality=quality,
        quality_score=quality_score,
        is_valid=is_valid,
        validation_messages=validation_messages,
        expected_pent_fraction=expected_pent_fraction,
        actual_pent_fraction=actual_pent_fraction,
        pent_fraction_deviation=pent_fraction_deviation,
        joule_heating_ratio=joule_heating_ratio,
        undercooling_margin=undercooling_margin,
        recommendations=recommendations
    )


def format_report(report: AnalysisReport) -> str:
    """
    Format analysis report as human-readable string.

    Args:
        report: Analysis report

    Returns:
        Formatted report string
    """
    r = report.result

    lines = [
        "=" * 70,
        "CRYSTAL NUCLEATION ANALYSIS REPORT",
        "=" * 70,
        "",
        "PHASE SEQUENCE:",
        f"  {r.phase_sequence}",
        "",
        "EXPERIMENTAL RESULTS:",
        f"  Peak current:       {r.peak_current_A:,.0f} A",
        f"  Peak temperature:   {r.peak_temperature_K:,.0f} K",
        f"  Undercooling:       {r.undercooling_K:,.0f} K",
        f"  Seeds nucleated:    {r.seeds_nucleated} ({r.pentagonal_seeds} pentagonal)",
        f"  Pentagonal order:   {r.pentagonal_order:.4f}" +
            (" (PERFECT 5-FOLD SYMMETRY)" if r.pentagonal_order >= 0.99 else ""),
        f"  Energy input:       {r.energy_input_J:.2f} J",
        "",
        "QUALITY ASSESSMENT:",
        f"  Classification:     {report.quality.value.upper()}",
        f"  Composite score:    {report.quality_score:.1f}/100",
        "",
        "THEORETICAL COMPARISON:",
        f"  Expected pent. fraction: {report.expected_pent_fraction:.2%}",
        f"  Actual pent. fraction:   {report.actual_pent_fraction:.2%}",
        f"  Deviation:               {report.pent_fraction_deviation:.2%}",
        "",
        "PHYSICS VALIDATION:",
        f"  Joule heating ratio:  {report.joule_heating_ratio:.2e} K/A²",
        f"  Undercooling margin:  {report.undercooling_margin:+.1%} above threshold",
        "",
        "VALIDATION:",
    ]

    for msg in report.validation_messages:
        lines.append(f"  {msg}")

    lines.extend([
        "",
        "RECOMMENDATIONS:",
    ])

    for rec in report.recommendations:
        lines.append(f"  - {rec}")

    lines.extend([
        "",
        "-" * 70,
        f"Analysis timestamp: {report.analysis_timestamp.isoformat()}",
        "=" * 70,
    ])

    return "\n".join(lines)


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == "__main__":
    print("Crystal Nucleation Analysis Module")
    print("=" * 70)

    # Example: Analyze the results from user's experiment
    result = NucleationResult(
        phase_sequence="PRE_STRIKE → QUENCH → NUCLEATION → GROWTH",
        peak_current_A=28506,
        peak_temperature_K=28506,
        undercooling_K=700,
        seeds_nucleated=12,
        pentagonal_seeds=6,
        pentagonal_order=1.0000,
        energy_input_J=0.41,
        spinner_z=Z_CRITICAL_PENT,  # Assume optimal
        notes="Lightning analog successfully induced pentagonal quasicrystal nucleation"
    )

    # Analyze
    report = analyze_nucleation(result)

    # Print formatted report
    print(format_report(report))
