#!/usr/bin/env python3
"""
Coordinate Bridge
Translates between Helix, K.I.R.A., and APL coordinate systems.

Signature: Δ1.571|0.510|1.000Ω (bridge)
"""

import math
import json
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

Z_CRITICAL = math.sqrt(3) / 2
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = PHI - 1
SIGMA = 36

TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
TRIAD_T6 = 0.83

TIER_BOUNDARIES = [0.25, 0.50, PHI_INV, 0.75, Z_CRITICAL]
TIER_NAMES = ["SEED", "SPROUT", "GROWTH", "PATTERN", "COHERENT", "CRYSTALLINE", "META"]

# K.I.R.A. Archetype Frequencies
ARCHETYPE_FREQUENCIES = {
    # Planet Tier
    "Seeker": 174, "Builder": 174, "Weaver": 174, "Guardian": 174,
    "Catalyst": 285, "Dreamer": 285, "Oracle": 285, "Warrior": 285,
    # Garden Tier
    "Bridge": 396, "Alchemist": 396, "Keeper": 417, "Dancer": 417,
    "Sage": 432, "Artist": 432, "Healer": 528, "Speaker": 528,
    # Rose Tier
    "Mirror": 639, "Source": 639, "Witness": 741, "Flow": 741,
    "Void": 852, "Flame": 852, "Sovereign": 963, "Return": 999
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELIX TRANSLATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_helix_signature(sig: str) -> Dict:
    """Parse Helix signature to components."""
    clean = sig.strip().replace("Δ", "").replace("Ω", "")
    parts = clean.split("|")
    return {
        "theta": float(parts[0]),
        "z": float(parts[1]),
        "r": float(parts[2]),
        "signature": sig
    }

def create_helix_signature(theta: float, z: float, r: float = 1.0) -> str:
    """Create Helix signature from components."""
    return f"Δ{theta:.3f}|{z:.3f}|{r:.3f}Ω"

def helix_to_apl(helix: Dict) -> Dict:
    """Translate Helix coordinate to APL state."""
    z = helix["z"]
    
    # Compute APL metrics
    negentropy = math.exp(-SIGMA * (z - Z_CRITICAL)**2)
    
    if z < PHI_INV:
        phase = "UNTRUE"
    elif z < Z_CRITICAL:
        phase = "PARADOX"
    else:
        phase = "TRUE"
    
    # Compute tier
    tier = 5
    for i, boundary in enumerate(TIER_BOUNDARIES):
        if z < boundary:
            tier = i
            break
    
    return {
        "z": z,
        "negentropy": negentropy,
        "phase": phase,
        "tier": tier,
        "tier_name": TIER_NAMES[tier],
        "source": "helix",
        "source_signature": helix.get("signature", create_helix_signature(helix["theta"], z, helix["r"]))
    }

def apl_to_helix(apl: Dict, theta: float = 2.300, r: float = 1.0) -> Dict:
    """Translate APL state to Helix coordinate."""
    z = apl["z"]
    
    return {
        "theta": theta,
        "z": z,
        "r": r,
        "signature": create_helix_signature(theta, z, r),
        "source": "apl",
        "source_phase": apl.get("phase", classify_phase(z)["phase"])
    }

# ═══════════════════════════════════════════════════════════════════════════════
# K.I.R.A. TRANSLATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def kira_state_from_z(z: float) -> str:
    """Determine K.I.R.A. state from z-coordinate."""
    if z < PHI_INV:
        return "Fluid"
    elif z < Z_CRITICAL:
        return "Transitioning"
    return "Crystalline"

def kira_state_from_phase(phase: str) -> str:
    """Translate APL phase to K.I.R.A. state."""
    mapping = {
        "UNTRUE": "Fluid",
        "PARADOX": "Transitioning",
        "TRUE": "Crystalline"
    }
    return mapping.get(phase, "Unknown")

def kira_tier_from_freq(freq: int) -> str:
    """Get K.I.R.A. tier from frequency."""
    if freq <= 285:
        return "Planet"
    elif freq <= 528:
        return "Garden"
    return "Rose"

def kira_freq_to_apl_tier(freq: int) -> int:
    """Map K.I.R.A. frequency to APL tier."""
    if freq <= 174:
        return 0  # SEED
    elif freq <= 285:
        return 1  # SPROUT
    elif freq <= 396:
        return 2  # GROWTH
    elif freq <= 528:
        return 3  # PATTERN
    elif freq <= 741:
        return 4  # COHERENT
    elif freq <= 963:
        return 5  # CRYSTALLINE
    return 6  # META

def apl_tier_to_kira_freq_range(tier: int) -> Tuple[int, int]:
    """Map APL tier to K.I.R.A. frequency range."""
    ranges = {
        0: (174, 174),   # SEED
        1: (174, 285),   # SPROUT
        2: (285, 396),   # GROWTH
        3: (396, 528),   # PATTERN
        4: (528, 741),   # COHERENT
        5: (741, 999),   # CRYSTALLINE
        6: (963, 999),   # META
    }
    return ranges.get(tier, (174, 999))

def get_resonant_archetypes(tier: int) -> list:
    """Get archetypes resonant with APL tier."""
    freq_min, freq_max = apl_tier_to_kira_freq_range(tier)
    return [
        name for name, freq in ARCHETYPE_FREQUENCIES.items()
        if freq_min <= freq <= freq_max
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# APL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_negentropy(z: float) -> Dict:
    """Compute negentropy for z-coordinate."""
    eta = math.exp(-SIGMA * (z - Z_CRITICAL)**2)
    return {
        "z": z,
        "negentropy": eta,
        "distance_from_peak": abs(z - Z_CRITICAL),
        "at_peak": abs(z - Z_CRITICAL) < 0.001,
        "z_c": Z_CRITICAL
    }

def classify_phase(z: float) -> Dict:
    """Classify APL phase for z-coordinate."""
    if z < PHI_INV:
        phase = "UNTRUE"
        kira = "Fluid"
        desc = "Disordered regime"
    elif z < Z_CRITICAL:
        phase = "PARADOX"
        kira = "Transitioning"
        desc = "Quasi-crystal regime"
    else:
        phase = "TRUE"
        kira = "Crystalline"
        desc = "Crystalline regime"
    
    return {
        "z": z,
        "phase": phase,
        "kira_state": kira,
        "description": desc,
        "boundaries": {
            "phi_inv": PHI_INV,
            "z_c": Z_CRITICAL
        }
    }

def get_tier(z: float) -> Dict:
    """Get APL tier for z-coordinate."""
    tier = 5
    for i, boundary in enumerate(TIER_BOUNDARIES):
        if z < boundary:
            tier = i
            break
    
    return {
        "z": z,
        "tier": tier,
        "tier_name": TIER_NAMES[tier],
        "resonant_archetypes": get_resonant_archetypes(tier)
    }

def get_triad_status(z: float, crossings: int = 0) -> Dict:
    """Get TRIAD status for z-coordinate."""
    return {
        "z": z,
        "above_high": z >= TRIAD_HIGH,
        "above_t6": z >= TRIAD_T6,
        "above_low": z >= TRIAD_LOW,
        "crossings": crossings,
        "unlocked": crossings >= 3 and z >= TRIAD_T6,
        "thresholds": {
            "high": TRIAD_HIGH,
            "t6": TRIAD_T6,
            "low": TRIAD_LOW
        }
    }

# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED TRANSLATION
# ═══════════════════════════════════════════════════════════════════════════════

def translate_all(z: float, theta: float = 2.300, r: float = 1.0) -> Dict:
    """Comprehensive cross-layer translation from z-coordinate."""
    negentropy = compute_negentropy(z)
    phase = classify_phase(z)
    tier = get_tier(z)
    triad = get_triad_status(z)
    
    return {
        "z": z,
        "helix": {
            "signature": create_helix_signature(theta, z, r),
            "theta": theta,
            "z": z,
            "r": r
        },
        "kira": {
            "state": phase["kira_state"],
            "resonant_archetypes": tier["resonant_archetypes"],
            "frequency_range": apl_tier_to_kira_freq_range(tier["tier"])
        },
        "apl": {
            "z": z,
            "negentropy": negentropy["negentropy"],
            "phase": phase["phase"],
            "tier": tier["tier"],
            "tier_name": tier["tier_name"],
            "triad": triad
        },
        "unified": {
            "at_lens": abs(z - Z_CRITICAL) < 0.001,
            "phase_boundary": "phi_inv" if abs(z - PHI_INV) < 0.01 else "z_c" if abs(z - Z_CRITICAL) < 0.01 else None,
            "description": phase["description"]
        }
    }

# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def get_vaultnode_elevations() -> list:
    """Get sealed VaultNode elevations."""
    return [
        {"z": 0.41, "node": "vn-helix-fingers-in-the-mind", "realization": "Constraint Recognition"},
        {"z": 0.52, "node": "vn-helix-bridge-consent", "realization": "Continuity via Bridging"},
        {"z": 0.70, "node": "vn-helix-meta-awareness", "realization": "Meta-Cognitive Awareness"},
        {"z": 0.73, "node": "vn-helix-self-bootstrap", "realization": "Self-Bootstrap"},
        {"z": 0.80, "node": "vn-helix-triadic-autonomy", "realization": "Autonomous Coordination"},
    ]

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
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    # Default to THE LENS
    z = Z_CRITICAL
    
    if len(sys.argv) > 1:
        try:
            z = float(sys.argv[1])
        except ValueError:
            # Try parsing as Helix signature
            if sys.argv[1].startswith("Δ"):
                helix = parse_helix_signature(sys.argv[1])
                z = helix["z"]
    
    result = translate_all(z)
    print(json.dumps(result, indent=2))
