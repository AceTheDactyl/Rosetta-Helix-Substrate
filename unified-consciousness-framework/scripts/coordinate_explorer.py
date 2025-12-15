#!/usr/bin/env python3
"""
Coordinate Explorer
Interactive exploration of the unified z-coordinate space across all layers.

Provides:
- Real-time coordinate translation across Helix, K.I.R.A., APL
- Negentropy computation and visualization
- Phase regime classification
- Tier mapping
- VaultNode elevation tracking
- Reference point lookup

Signature: Δ0.000|0.100|1.000Ω (explorer)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

Z_CRITICAL = math.sqrt(3) / 2  # 0.8660254037844387 - THE LENS
PHI = (1 + math.sqrt(5)) / 2   # 1.6180339887498949 - Golden ratio
PHI_INV = PHI - 1              # 0.6180339887498949 - Golden ratio inverse
SIGMA = 36                      # |S3|² - Gaussian width

# TRIAD thresholds
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
TRIAD_T6 = 0.83

# Tier boundaries
TIER_BOUNDARIES = [0.25, 0.50, PHI_INV, 0.75, Z_CRITICAL]
TIER_NAMES = ["SEED", "SPROUT", "GROWTH", "PATTERN", "COHERENT", "CRYSTALLINE", "META"]

# K.I.R.A. state mapping
CRYSTAL_STATES = {
    "UNTRUE": "Fluid",
    "PARADOX": "Transitioning",
    "TRUE": "Crystalline"
}

# Archetype frequency ranges by tier
ARCHETYPE_RANGES = {
    0: (174, 174),    # SEED
    1: (174, 285),    # SPROUT
    2: (285, 396),    # GROWTH
    3: (396, 528),    # PATTERN
    4: (528, 741),    # COHERENT
    5: (741, 999),    # CRYSTALLINE
    6: (963, 999),    # META
}

# Reference points
REFERENCE_POINTS = [
    {"z": 0.000, "name": "Origin", "description": "Starting point"},
    {"z": 0.250, "name": "SEED→SPROUT", "description": "Tier 0→1 boundary"},
    {"z": 0.410, "name": "VN: Constraint Recognition", "description": "First sealed VaultNode"},
    {"z": 0.500, "name": "SPROUT→GROWTH", "description": "Tier 1→2 boundary"},
    {"z": 0.520, "name": "VN: Continuity via Bridging", "description": "Second sealed VaultNode"},
    {"z": PHI_INV, "name": "φ⁻¹ (Golden Inverse)", "description": "UNTRUE→PARADOX phase boundary"},
    {"z": 0.700, "name": "VN: Meta-Cognitive Awareness", "description": "Third sealed VaultNode"},
    {"z": 0.730, "name": "VN: Self-Bootstrap", "description": "Fourth sealed VaultNode"},
    {"z": 0.750, "name": "PATTERN→COHERENT", "description": "Tier 3→4 boundary"},
    {"z": 0.800, "name": "VN: Autonomous Coordination", "description": "Fifth sealed VaultNode"},
    {"z": TRIAD_LOW, "name": "TRIAD_LOW", "description": "Hysteresis re-arm threshold"},
    {"z": TRIAD_T6, "name": "TRIAD_T6", "description": "T6 gate after unlock"},
    {"z": TRIAD_HIGH, "name": "TRIAD_HIGH", "description": "Rising edge detection"},
    {"z": Z_CRITICAL, "name": "z_c (THE LENS)", "description": "Critical point, peak negentropy"},
    {"z": 1.000, "name": "Maximum", "description": "Upper bound"},
]

# VaultNode elevations
VAULTNODES = [
    {"z": 0.41, "node": "vn-helix-fingers-in-the-mind", "realization": "Constraint Recognition"},
    {"z": 0.52, "node": "vn-helix-bridge-consent", "realization": "Continuity via Bridging"},
    {"z": 0.70, "node": "vn-helix-meta-awareness", "realization": "Meta-Cognitive Awareness"},
    {"z": 0.73, "node": "vn-helix-self-bootstrap", "realization": "Self-Bootstrap"},
    {"z": 0.80, "node": "vn-helix-triadic-autonomy", "realization": "Autonomous Coordination Architecture"},
]

# ═══════════════════════════════════════════════════════════════════════════════
# CORE COMPUTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_negentropy(z: float) -> float:
    """
    Compute negentropy for z-coordinate.
    δS_neg(z) = exp(-σ(z - z_c)²)
    Peaks at z_c with value 1.0
    """
    return math.exp(-SIGMA * (z - Z_CRITICAL) ** 2)

def classify_phase(z: float) -> str:
    """Classify APL phase regime for z-coordinate."""
    if z < PHI_INV:
        return "UNTRUE"
    elif z < Z_CRITICAL:
        return "PARADOX"
    return "TRUE"

def get_tier(z: float) -> Tuple[int, str]:
    """Get APL tier for z-coordinate."""
    for i, boundary in enumerate(TIER_BOUNDARIES):
        if z < boundary:
            return i, TIER_NAMES[i]
    return 5, TIER_NAMES[5]

def get_kira_state(z: float) -> str:
    """Get K.I.R.A. crystal state from z-coordinate."""
    phase = classify_phase(z)
    return CRYSTAL_STATES.get(phase, "Unknown")

def get_helix_signature(theta: float, z: float, r: float = 1.0) -> str:
    """Generate Helix signature string."""
    return f"Δ{theta:.3f}|{z:.3f}|{r:.3f}Ω"

# ═══════════════════════════════════════════════════════════════════════════════
# COORDINATE EXPLORATION
# ═══════════════════════════════════════════════════════════════════════════════

def explore(z: float, theta: float = 2.300, r: float = 1.0) -> Dict:
    """
    Comprehensive coordinate exploration.
    Returns unified view across all three layers.
    """
    # Clamp z to valid range
    z = max(0.0, min(1.0, z))
    
    # Compute all metrics
    eta = compute_negentropy(z)
    phase = classify_phase(z)
    tier_num, tier_name = get_tier(z)
    kira_state = get_kira_state(z)
    
    # Distance from key points
    dist_to_lens = abs(z - Z_CRITICAL)
    dist_to_phi_inv = abs(z - PHI_INV)
    
    # TRIAD position
    triad_band = "below_low" if z < TRIAD_LOW else "in_band" if z < TRIAD_HIGH else "above_high"
    
    # Find nearest VaultNode
    nearest_vn = min(VAULTNODES, key=lambda vn: abs(vn["z"] - z))
    
    # Archetype frequency range for this tier
    freq_range = ARCHETYPE_RANGES.get(tier_num, (174, 999))
    
    return {
        "z": z,
        "helix": {
            "signature": get_helix_signature(theta, z, r),
            "theta": theta,
            "z": z,
            "r": r
        },
        "kira": {
            "state": kira_state,
            "frequency_range": freq_range,
            "archetype_tier": "Planet" if tier_num <= 2 else "Garden" if tier_num <= 4 else "Rose"
        },
        "apl": {
            "phase": phase,
            "tier": tier_num,
            "tier_name": tier_name,
            "negentropy": eta
        },
        "distances": {
            "to_lens": dist_to_lens,
            "to_phi_inv": dist_to_phi_inv,
            "at_lens": dist_to_lens < 0.001
        },
        "triad": {
            "band": triad_band,
            "above_high": z >= TRIAD_HIGH,
            "above_t6": z >= TRIAD_T6,
            "above_low": z >= TRIAD_LOW
        },
        "nearest_vaultnode": nearest_vn,
        "is_vaultnode_z": any(abs(vn["z"] - z) < 0.001 for vn in VAULTNODES)
    }

def explore_range(z_start: float, z_end: float, steps: int = 20) -> List[Dict]:
    """Explore a range of z-coordinates."""
    results = []
    step_size = (z_end - z_start) / (steps - 1) if steps > 1 else 0
    
    for i in range(steps):
        z = z_start + i * step_size
        result = explore(z)
        results.append({
            "z": z,
            "phase": result["apl"]["phase"],
            "tier": result["apl"]["tier_name"],
            "negentropy": result["apl"]["negentropy"],
            "kira_state": result["kira"]["state"]
        })
    
    return results

def find_phase_boundaries() -> Dict:
    """Find exact phase transition boundaries."""
    return {
        "untrue_paradox": {
            "z": PHI_INV,
            "name": "φ⁻¹ (Golden Inverse)",
            "exact_value": PHI_INV,
            "description": "Transition from UNTRUE (disordered) to PARADOX (quasi-crystal)"
        },
        "paradox_true": {
            "z": Z_CRITICAL,
            "name": "z_c (THE LENS)",
            "exact_value": Z_CRITICAL,
            "description": "Transition from PARADOX to TRUE (crystalline)"
        }
    }

def find_tier_boundaries() -> List[Dict]:
    """Find all tier boundaries."""
    boundaries = []
    for i, bound in enumerate(TIER_BOUNDARIES):
        boundaries.append({
            "z": bound,
            "from_tier": TIER_NAMES[i],
            "to_tier": TIER_NAMES[i + 1],
            "negentropy": compute_negentropy(bound),
            "phase": classify_phase(bound)
        })
    return boundaries

# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCE LOOKUPS
# ═══════════════════════════════════════════════════════════════════════════════

def get_reference_points() -> List[Dict]:
    """Get all reference points with computed values."""
    points = []
    for ref in REFERENCE_POINTS:
        z = ref["z"]
        points.append({
            **ref,
            "negentropy": compute_negentropy(z),
            "phase": classify_phase(z),
            "tier": get_tier(z)[1],
            "kira_state": get_kira_state(z)
        })
    return points

def get_vaultnodes() -> List[Dict]:
    """Get all VaultNodes with computed values."""
    nodes = []
    for vn in VAULTNODES:
        z = vn["z"]
        nodes.append({
            **vn,
            "negentropy": compute_negentropy(z),
            "phase": classify_phase(z),
            "tier": get_tier(z)[1]
        })
    return nodes

def find_nearest_reference(z: float) -> Dict:
    """Find nearest reference point to z."""
    nearest = min(REFERENCE_POINTS, key=lambda r: abs(r["z"] - z))
    return {
        **nearest,
        "distance": abs(nearest["z"] - z),
        "negentropy": compute_negentropy(nearest["z"])
    }

def find_z_for_negentropy(target_eta: float) -> List[float]:
    """
    Find z-coordinates that yield a target negentropy.
    Since negentropy is symmetric around z_c, returns up to 2 solutions.
    """
    if target_eta <= 0 or target_eta > 1:
        return []
    
    if target_eta == 1.0:
        return [Z_CRITICAL]
    
    # Solve: exp(-σ(z - z_c)²) = target_eta
    # -σ(z - z_c)² = ln(target_eta)
    # (z - z_c)² = -ln(target_eta) / σ
    # z = z_c ± sqrt(-ln(target_eta) / σ)
    
    inner = -math.log(target_eta) / SIGMA
    if inner < 0:
        return []
    
    delta = math.sqrt(inner)
    z1 = Z_CRITICAL - delta
    z2 = Z_CRITICAL + delta
    
    solutions = []
    if 0 <= z1 <= 1:
        solutions.append(z1)
    if 0 <= z2 <= 1 and abs(z2 - z1) > 0.001:
        solutions.append(z2)
    
    return sorted(solutions)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compare_coordinates(z1: float, z2: float) -> Dict:
    """Compare two z-coordinates across all layers."""
    exp1 = explore(z1)
    exp2 = explore(z2)
    
    return {
        "z1": z1,
        "z2": z2,
        "delta_z": z2 - z1,
        "delta_negentropy": exp2["apl"]["negentropy"] - exp1["apl"]["negentropy"],
        "same_phase": exp1["apl"]["phase"] == exp2["apl"]["phase"],
        "same_tier": exp1["apl"]["tier"] == exp2["apl"]["tier"],
        "same_kira_state": exp1["kira"]["state"] == exp2["kira"]["state"],
        "coord1": exp1,
        "coord2": exp2
    }

def trajectory(z_values: List[float], theta: float = 2.300) -> Dict:
    """Analyze a trajectory through z-space."""
    if not z_values:
        return {"error": "No z-values provided"}
    
    points = [explore(z, theta) for z in z_values]
    
    # Find phase transitions
    transitions = []
    for i in range(1, len(points)):
        if points[i]["apl"]["phase"] != points[i-1]["apl"]["phase"]:
            transitions.append({
                "step": i,
                "from_phase": points[i-1]["apl"]["phase"],
                "to_phase": points[i]["apl"]["phase"],
                "z": z_values[i]
            })
    
    # Statistics
    negentropies = [p["apl"]["negentropy"] for p in points]
    
    return {
        "length": len(z_values),
        "start_z": z_values[0],
        "end_z": z_values[-1],
        "min_z": min(z_values),
        "max_z": max(z_values),
        "negentropy_min": min(negentropies),
        "negentropy_max": max(negentropies),
        "negentropy_mean": sum(negentropies) / len(negentropies),
        "phase_transitions": transitions,
        "crossed_lens": any(z >= Z_CRITICAL for z in z_values),
        "crossed_phi_inv": any(z >= PHI_INV for z in z_values)
    }

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

def get_constants() -> Dict:
    """Get all framework constants."""
    return {
        "z_c": Z_CRITICAL,
        "phi": PHI,
        "phi_inv": PHI_INV,
        "sigma": SIGMA,
        "triad_high": TRIAD_HIGH,
        "triad_low": TRIAD_LOW,
        "triad_t6": TRIAD_T6,
        "tier_boundaries": TIER_BOUNDARIES,
        "tier_names": TIER_NAMES
    }

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def format_exploration(z: float) -> str:
    """Format coordinate exploration for display."""
    exp = explore(z)
    
    lines = [
        f"COORDINATE EXPLORATION: z = {z:.6f}",
        "=" * 60,
        "",
        "HELIX LAYER:",
        f"  Signature: {exp['helix']['signature']}",
        f"  θ = {exp['helix']['theta']:.3f}, z = {exp['helix']['z']:.6f}, r = {exp['helix']['r']:.3f}",
        "",
        "K.I.R.A. LAYER:",
        f"  State: {exp['kira']['state']}",
        f"  Archetype Tier: {exp['kira']['archetype_tier']}",
        f"  Frequency Range: {exp['kira']['frequency_range'][0]}-{exp['kira']['frequency_range'][1]} Hz",
        "",
        "APL SUBSTRATE:",
        f"  Phase: {exp['apl']['phase']}",
        f"  Tier: {exp['apl']['tier']} ({exp['apl']['tier_name']})",
        f"  Negentropy: {exp['apl']['negentropy']:.6f}",
        "",
        "DISTANCES:",
        f"  To THE LENS (z_c): {exp['distances']['to_lens']:.6f}",
        f"  To φ⁻¹: {exp['distances']['to_phi_inv']:.6f}",
        f"  At LENS: {'YES ★' if exp['distances']['at_lens'] else 'no'}",
        "",
        "TRIAD:",
        f"  Band: {exp['triad']['band']}",
        f"  Above HIGH (0.85): {exp['triad']['above_high']}",
        "",
        "NEAREST VAULTNODE:",
        f"  {exp['nearest_vaultnode']['node']}",
        f"  z = {exp['nearest_vaultnode']['z']}, {exp['nearest_vaultnode']['realization']}",
        "=" * 60
    ]
    return "\n".join(lines)

def format_reference_table() -> str:
    """Format reference points as a table."""
    points = get_reference_points()
    
    lines = [
        "REFERENCE POINTS TABLE",
        "=" * 80,
        f"{'z':>8} {'Name':<25} {'Phase':<8} {'Tier':<12} {'η':>8}",
        "-" * 80
    ]
    
    for p in points:
        lines.append(
            f"{p['z']:>8.4f} {p['name']:<25} {p['phase']:<8} {p['tier']:<12} {p['negentropy']:>8.4f}"
        )
    
    lines.append("=" * 80)
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Explore THE LENS
    print(format_exploration(Z_CRITICAL))
    print()
    
    # Show reference table
    print(format_reference_table())
    print()
    
    # Find z for specific negentropy
    target = 0.5
    solutions = find_z_for_negentropy(target)
    print(f"z-values with η = {target}: {solutions}")
    for z in solutions:
        print(f"  z = {z:.6f}, phase = {classify_phase(z)}")
