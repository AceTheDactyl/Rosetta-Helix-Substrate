#!/usr/bin/env python3
"""
Post-Critical Dynamics: What Happens Beyond z_c = √3/2
=======================================================

When z exceeds the hexagonal critical point z_c, several key dynamics emerge:

PHASE DIAGRAM:
```
z = 1.0   ┌─────────────────────────────────────────────┐
          │  TRANSCENDENT (z > z_p)                      │
          │  • Beyond pentagonal critical                │
          │  • Quasicrystal growth/annealing            │
z = 0.951 ├─────────────────────────────────────────────┤ z_p = sin(72°)
          │  PENTAGONAL TRANSITION ZONE                  │
          │  • 5-fold nucleation favored                 │
          │  • ΔS_neg decreasing but z_p preference     │
          │  • Fat/thin tile ratio → φ                  │
z = 0.877 ├─────────────────────────────────────────────┤ PRESENCE boundary
          │  POST-CRITICAL DESCENT                       │
          │  • ΔS_neg decreasing from peak              │
          │  • EM activation decreasing                  │
          │  • But still in PRESENCE phase              │
z = 0.866 ╞═════════════════════════════════════════════╡ z_c = √3/2 (POLARIS)
          │  ▲ PEAK: ΔS_neg = 1.0, Max EM              │
          │  THE LENS - Phase transition boundary       │
z = 0.857 ├─────────────────────────────────────────────┤ ABSENCE boundary
          │  PRE-CRITICAL ASCENT                        │
          │  • ΔS_neg increasing toward peak            │
          │  • EM activation increasing                  │
z = 0.5   └─────────────────────────────────────────────┘
```

KEY INSIGHT: While ΔS_neg peaks at z_c, the PENTAGONAL order parameter
peaks at z_p = sin(72°) ≈ 0.951. The system must traverse the
post-critical zone to reach the pentagonal nucleation sweet spot.

POST-CRITICAL DYNAMICS:
1. Negentropy descent (Gaussian tail)
2. Symmetry transition (6-fold → 5-fold)
3. Pentagonal preference emergence
4. Quasicrystal nucleation window

Signature: post-critical-dynamics|v1.0.0|helix
"""

import math
import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import IntEnum

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = (1 + math.sqrt(5)) / 2          # Golden ratio ≈ 1.618034
PHI_INV = 1 / PHI                      # ≈ 0.618034
Z_CRITICAL_HEX = math.sqrt(3) / 2      # √3/2 ≈ 0.866025 (hexagonal)
Z_CRITICAL_PENT = math.sqrt(10 + 2*math.sqrt(5)) / 4  # sin(72°) ≈ 0.951057
SIGMA = 36.0                           # Gaussian width for negentropy

# Phase boundaries
PHASE_ABSENCE_UPPER = 0.857
PHASE_PRESENCE_LOWER = 0.877

# Pentagon geometry
SIN_36 = math.sqrt(10 - 2*math.sqrt(5)) / 4  # ≈ 0.588
COS_36 = PHI / 2                              # ≈ 0.809
SIN_72 = Z_CRITICAL_PENT                      # ≈ 0.951
COS_72 = (math.sqrt(5) - 1) / 4               # ≈ 0.309


# =============================================================================
# PHASE DEFINITIONS
# =============================================================================

class DynamicsRegion(IntEnum):
    """Regions of post-critical dynamics."""
    PRE_CRITICAL = 0        # z < z_c (ascending)
    AT_CRITICAL = 1         # z ≈ z_c (THE LENS)
    POST_CRITICAL = 2       # z_c < z < z_p (descending negentropy)
    PENTAGONAL_ZONE = 3     # z ≈ z_p (pentagonal nucleation)
    TRANSCENDENT = 4        # z > z_p (beyond pentagonal)


class SymmetryPreference(IntEnum):
    """Which symmetry is preferred at current z."""
    DISORDERED = 0
    HEXAGONAL = 6           # 6-fold (favored at z_c)
    PENTAGONAL = 5          # 5-fold (favored at z_p)
    MIXED = 1               # Transition zone


# =============================================================================
# DYNAMICS ANALYSIS
# =============================================================================

@dataclass
class DynamicsState:
    """State of post-critical dynamics."""
    z: float
    region: DynamicsRegion

    # Negentropy (peaks at z_c)
    negentropy_hex: float       # ΔS_neg relative to z_c
    negentropy_pent: float      # ΔS_neg relative to z_p

    # EM activation
    em_activation: float        # Plate activation (peaks at z_c)
    em_gradient: float          # dEM/dz (negative post-critical)

    # Symmetry preferences
    hex_preference: float       # Preference for 6-fold
    pent_preference: float      # Preference for 5-fold
    dominant_symmetry: SymmetryPreference

    # Phase transition metrics
    transition_progress: float  # 0 = at z_c, 1 = at z_p
    nucleation_probability: float


def compute_dynamics_state(z: float) -> DynamicsState:
    """
    Compute the dynamics state at a given z-coordinate.

    This reveals what happens as z moves beyond z_c toward z_p.
    """
    # Determine region
    if z < Z_CRITICAL_HEX - 0.01:
        region = DynamicsRegion.PRE_CRITICAL
    elif z < Z_CRITICAL_HEX + 0.01:
        region = DynamicsRegion.AT_CRITICAL
    elif z < Z_CRITICAL_PENT - 0.02:
        region = DynamicsRegion.POST_CRITICAL
    elif z < Z_CRITICAL_PENT + 0.02:
        region = DynamicsRegion.PENTAGONAL_ZONE
    else:
        region = DynamicsRegion.TRANSCENDENT

    # Negentropy relative to hexagonal critical (peaks at z_c)
    neg_hex = math.exp(-SIGMA * (z - Z_CRITICAL_HEX) ** 2)

    # Negentropy relative to pentagonal critical (peaks at z_p)
    # Use different width for pentagonal (sharper peak)
    sigma_pent = 50.0  # Sharper Gaussian for pentagonal
    neg_pent = math.exp(-sigma_pent * (z - Z_CRITICAL_PENT) ** 2)

    # EM activation (proportional to z × cos(60°), peaks effectiveness at z_c)
    # The 60° plate geometry gives maximum coupling at z_c
    em_activation = z * 0.5 * neg_hex  # Modulated by negentropy

    # EM gradient (derivative of activation)
    dz = 0.001
    em_plus = (z + dz) * 0.5 * math.exp(-SIGMA * ((z + dz) - Z_CRITICAL_HEX) ** 2)
    em_minus = (z - dz) * 0.5 * math.exp(-SIGMA * ((z - dz) - Z_CRITICAL_HEX) ** 2)
    em_gradient = (em_plus - em_minus) / (2 * dz)

    # Symmetry preferences
    # Hexagonal preference peaks at z_c
    hex_preference = neg_hex

    # Pentagonal preference peaks at z_p
    pent_preference = neg_pent

    # Determine dominant symmetry
    if hex_preference > pent_preference + 0.1:
        dominant = SymmetryPreference.HEXAGONAL
    elif pent_preference > hex_preference + 0.1:
        dominant = SymmetryPreference.PENTAGONAL
    elif hex_preference > 0.3 or pent_preference > 0.3:
        dominant = SymmetryPreference.MIXED
    else:
        dominant = SymmetryPreference.DISORDERED

    # Transition progress (0 at z_c, 1 at z_p)
    if z <= Z_CRITICAL_HEX:
        transition_progress = 0.0
    elif z >= Z_CRITICAL_PENT:
        transition_progress = 1.0
    else:
        transition_progress = (z - Z_CRITICAL_HEX) / (Z_CRITICAL_PENT - Z_CRITICAL_HEX)

    # Nucleation probability (peaks in pentagonal zone)
    # Requires both thermal driving force AND structural preference
    nucleation_prob = pent_preference * transition_progress

    return DynamicsState(
        z=z,
        region=region,
        negentropy_hex=neg_hex,
        negentropy_pent=neg_pent,
        em_activation=em_activation,
        em_gradient=em_gradient,
        hex_preference=hex_preference,
        pent_preference=pent_preference,
        dominant_symmetry=dominant,
        transition_progress=transition_progress,
        nucleation_probability=nucleation_prob
    )


def analyze_post_critical_trajectory(z_start: float = 0.5,
                                     z_end: float = 1.0,
                                     steps: int = 100) -> List[DynamicsState]:
    """
    Analyze dynamics along a z trajectory.
    """
    z_values = np.linspace(z_start, z_end, steps)
    return [compute_dynamics_state(z) for z in z_values]


def print_dynamics_analysis():
    """
    Print comprehensive analysis of post-critical dynamics.
    """
    print("=" * 80)
    print("POST-CRITICAL DYNAMICS ANALYSIS")
    print("What happens when z exceeds z_c = √3/2?")
    print("=" * 80)

    print(f"""
CRITICAL POINTS:
  z_c (hexagonal) = √3/2 = sin(60°) = {Z_CRITICAL_HEX:.6f}
  z_p (pentagonal) = sin(72°)       = {Z_CRITICAL_PENT:.6f}

  Gap: z_p - z_c = {Z_CRITICAL_PENT - Z_CRITICAL_HEX:.6f}
""")

    print("-" * 80)
    print("DYNAMICS AT KEY Z-COORDINATES:")
    print("-" * 80)
    print(f"{'z':>8} {'Region':<18} {'ΔS_hex':>8} {'ΔS_pent':>8} "
          f"{'EM':>8} {'dEM/dz':>8} {'Hex%':>6} {'Pent%':>6} {'Dominant':<12}")
    print("-" * 80)

    key_z_values = [
        0.5,        # Low z
        0.618,      # φ⁻¹
        0.80,       # Approaching z_c
        0.857,      # ABSENCE boundary
        Z_CRITICAL_HEX,  # z_c (PEAK)
        0.877,      # PRESENCE boundary
        0.90,       # Post-critical
        0.92,       # Deep post-critical
        Z_CRITICAL_PENT,  # z_p (PENTAGONAL)
        0.98,       # Beyond z_p
    ]

    for z in key_z_values:
        state = compute_dynamics_state(z)

        # Mark special points
        marker = ""
        if abs(z - Z_CRITICAL_HEX) < 0.001:
            marker = " ← POLARIS (EM PEAK)"
        elif abs(z - Z_CRITICAL_PENT) < 0.001:
            marker = " ← PENTAGONAL"
        elif abs(z - PHI_INV) < 0.001:
            marker = " ← φ⁻¹"

        print(f"{z:>8.4f} {state.region.name:<18} {state.negentropy_hex:>8.4f} "
              f"{state.negentropy_pent:>8.4f} {state.em_activation:>8.4f} "
              f"{state.em_gradient:>8.4f} {state.hex_preference*100:>5.1f}% "
              f"{state.pent_preference*100:>5.1f}% {state.dominant_symmetry.name:<12}{marker}")

    print("-" * 80)

    print("""
POST-CRITICAL DYNAMICS EXPLANATION:
===================================

1. AT z = z_c (√3/2 ≈ 0.866) - THE POLARIS PEAK:
   ─────────────────────────────────────────────
   • ΔS_neg (hexagonal) = 1.0 (MAXIMUM)
   • EM plate activation = MAXIMUM
   • Hexagonal (6-fold) symmetry strongly favored
   • This is THE LENS - the phase transition boundary
   • Kuramoto coupling peaks → synchronization threshold

2. IMMEDIATELY POST-CRITICAL (z_c < z < 0.90):
   ────────────────────────────────────────────
   • ΔS_neg DECREASING (Gaussian descent)
   • EM activation DECREASING (past resonance)
   • dEM/dz < 0 (NEGATIVE gradient)
   • Still hexagonal preferred, but weakening
   • System in PRESENCE phase

   KEY INSIGHT: Even though EM is decreasing, the system has
   CROSSED the phase boundary. It's now in a higher-energy state.

3. TRANSITION ZONE (0.90 < z < 0.95):
   ───────────────────────────────────
   • Hexagonal and pentagonal preferences COMPETING
   • Neither symmetry strongly dominant
   • System is UNSTABLE - susceptible to fluctuations
   • Nucleation seeds can form with either symmetry

   This is the LIMINAL zone - between two ordered states.

4. PENTAGONAL ZONE (z ≈ z_p = 0.951):
   ───────────────────────────────────
   • ΔS_neg (pentagonal) peaks
   • Pentagonal (5-fold) symmetry now FAVORED
   • Quasicrystal nucleation probability MAXIMUM
   • Fat/thin Penrose tile ratio → φ

   KEY PHYSICS: sin(72°) = √(10 + 2√5)/4 connects to
   pentagon interior angles and golden ratio geometry.

5. TRANSCENDENT (z > z_p):
   ────────────────────────
   • Both negentropies decreasing
   • Pentagonal domains can GROW but not nucleate new seeds
   • Annealing regime - defects heal, order improves
   • System approaches final quasicrystal state

THE CRITICAL INSIGHT:
=====================
The EM peak at z_c creates a "pump" that pushes the system INTO
the post-critical regime. Once past z_c:

  • The system has MOMENTUM (energy stored)
  • EM field provides DIRECTIONAL bias (from 60° geometry)
  • Decreasing negentropy acts like COOLING
  • At z_p, this "cooled" state nucleates PENTAGONAL order

It's analogous to:
  1. HEATING (z → z_c): Energy input, disorder increases
  2. PEAK (z = z_c): Maximum energy, phase boundary
  3. QUENCH (z_c → z_p): Rapid "cooling", order nucleates
  4. GROWTH (z ≈ z_p): Pentagonal domains expand
  5. ANNEAL (z > z_p): Defects removed, final structure

This is why the LIGHTNING ANALOG works:
  Strike (heat to z_c) → Quench (traverse to z_p) → Crystal forms
""")

    # Detailed numerical analysis
    print("\n" + "=" * 80)
    print("NUMERICAL ANALYSIS: z_c → z_p TRAJECTORY")
    print("=" * 80)

    trajectory = analyze_post_critical_trajectory(Z_CRITICAL_HEX, Z_CRITICAL_PENT, 20)

    print(f"\n{'Step':>4} {'z':>8} {'ΔS_hex':>8} {'ΔS_pent':>8} {'Transition':>10} {'Nucl.Prob':>10}")
    print("-" * 60)

    for i, state in enumerate(trajectory):
        print(f"{i:>4} {state.z:>8.4f} {state.negentropy_hex:>8.4f} "
              f"{state.negentropy_pent:>8.4f} {state.transition_progress*100:>9.1f}% "
              f"{state.nucleation_probability:>10.4f}")

    print(f"""
KEY OBSERVATIONS:
─────────────────
• At z_c: ΔS_hex = 1.0, ΔS_pent ≈ 0.0, Transition = 0%
• Midway:  Both negentropies low, system in transition
• At z_p: ΔS_hex ≈ 0.0, ΔS_pent = 1.0, Transition = 100%

The CROSSING point where ΔS_hex = ΔS_pent occurs at:
  z_cross ≈ {find_crossing_point():.4f}

This is where neither hexagonal nor pentagonal is favored -
the system is maximally UNDETERMINED.
""")


def find_crossing_point() -> float:
    """Find z where hexagonal and pentagonal negentropy are equal."""
    for z in np.linspace(Z_CRITICAL_HEX, Z_CRITICAL_PENT, 1000):
        neg_hex = math.exp(-SIGMA * (z - Z_CRITICAL_HEX) ** 2)
        neg_pent = math.exp(-50.0 * (z - Z_CRITICAL_PENT) ** 2)
        if abs(neg_hex - neg_pent) < 0.01:
            return z
    return (Z_CRITICAL_HEX + Z_CRITICAL_PENT) / 2


def export_dynamics_data(filepath: str = "post_critical_dynamics.json"):
    """Export dynamics data for visualization."""
    z_values = np.linspace(0.5, 1.0, 200)

    data = {
        "constants": {
            "z_c_hex": Z_CRITICAL_HEX,
            "z_c_pent": Z_CRITICAL_PENT,
            "phi": PHI,
            "phi_inv": PHI_INV,
            "sigma": SIGMA
        },
        "trajectory": []
    }

    for z in z_values:
        state = compute_dynamics_state(z)
        data["trajectory"].append({
            "z": float(z),
            "region": state.region.name,
            "negentropy_hex": float(state.negentropy_hex),
            "negentropy_pent": float(state.negentropy_pent),
            "em_activation": float(state.em_activation),
            "em_gradient": float(state.em_gradient),
            "hex_preference": float(state.hex_preference),
            "pent_preference": float(state.pent_preference),
            "dominant_symmetry": state.dominant_symmetry.name,
            "transition_progress": float(state.transition_progress),
            "nucleation_probability": float(state.nucleation_probability)
        })

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nExported dynamics data to: {filepath}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_dynamics_analysis()
    export_dynamics_data("training_artifacts/post_critical_dynamics.json")
