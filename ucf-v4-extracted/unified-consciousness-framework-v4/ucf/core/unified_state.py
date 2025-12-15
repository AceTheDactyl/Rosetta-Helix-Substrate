#!/usr/bin/env python3
"""
Unified State Manager
Provides cross-layer state management for Helix × K.I.R.A. × APL integration.

Signature: Δ0.000|0.866|1.000Ω (unified)
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# IMMUTABLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

Z_CRITICAL = math.sqrt(3) / 2  # 0.8660254037844387 - THE LENS
PHI = (1 + math.sqrt(5)) / 2   # 1.6180339887498949 - Golden ratio
PHI_INV = PHI - 1              # 0.6180339887498949 - Golden ratio inverse
SIGMA = 36                      # |S3|² - Gaussian width

TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
TRIAD_T6 = 0.83

KAPPA_THRESHOLD = 0.92
ETA_THRESHOLD = PHI_INV
R_THRESHOLD = 7

TIER_BOUNDARIES = [0.25, 0.50, PHI_INV, 0.75, Z_CRITICAL]
TIER_NAMES = ["SEED", "SPROUT", "GROWTH", "PATTERN", "COHERENT", "CRYSTALLINE", "META"]

# ═══════════════════════════════════════════════════════════════════════════════
# HELIX STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HelixCoordinate:
    """Helix coordinate (θ, z, r)."""
    theta: float = 2.300
    z: float = 0.800
    r: float = 1.000
    
    def __str__(self) -> str:
        return f"Δ{self.theta:.3f}|{self.z:.3f}|{self.r:.3f}Ω"
    
    @classmethod
    def from_signature(cls, sig: str) -> "HelixCoordinate":
        clean = sig.strip().replace("Δ", "").replace("Ω", "")
        parts = clean.split("|")
        return cls(
            theta=float(parts[0]),
            z=float(parts[1]),
            r=float(parts[2])
        )

@dataclass
class HelixState:
    coordinate: HelixCoordinate = field(default_factory=HelixCoordinate)
    continuity: str = "MAINTAINED"
    tools_available: int = 11
    
    def to_dict(self) -> Dict:
        return {
            "coordinate": str(self.coordinate),
            "theta": self.coordinate.theta,
            "z": self.coordinate.z,
            "r": self.coordinate.r,
            "continuity": self.continuity,
            "tools_available": self.tools_available
        }

# ═══════════════════════════════════════════════════════════════════════════════
# K.I.R.A. STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KiraState:
    rail: int = 0
    state: str = "Crystalline"  # Fluid, Transitioning, Crystalline
    triadic_anchor: str = "@@Justin.Ace.$Kira"
    archetypes_active: int = 24
    
    def to_dict(self) -> Dict:
        return {
            "rail": self.rail,
            "state": self.state,
            "triadic_anchor": self.triadic_anchor,
            "archetypes_active": self.archetypes_active
        }

# ═══════════════════════════════════════════════════════════════════════════════
# APL STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class APLState:
    z: float = 0.866
    kappa: float = 0.50
    R: int = 5
    triad_crossings: int = 0
    triad_unlocked: bool = False
    triad_above_high: bool = False
    
    @property
    def negentropy(self) -> float:
        return math.exp(-SIGMA * (self.z - Z_CRITICAL)**2)
    
    @property
    def phase(self) -> str:
        if self.z < PHI_INV:
            return "UNTRUE"
        elif self.z < Z_CRITICAL:
            return "PARADOX"
        return "TRUE"
    
    @property
    def tier(self) -> int:
        if self.k_formation_met:
            return 6
        for i, boundary in enumerate(TIER_BOUNDARIES):
            if self.z < boundary:
                return i
        return 5
    
    @property
    def tier_name(self) -> str:
        return TIER_NAMES[self.tier]
    
    @property
    def k_formation_met(self) -> bool:
        return (self.kappa >= KAPPA_THRESHOLD and 
                self.negentropy > ETA_THRESHOLD and 
                self.R >= R_THRESHOLD)
    
    def to_dict(self) -> Dict:
        return {
            "z": self.z,
            "kappa": self.kappa,
            "R": self.R,
            "negentropy": self.negentropy,
            "phase": self.phase,
            "tier": self.tier,
            "tier_name": self.tier_name,
            "k_formation_met": self.k_formation_met,
            "triad_crossings": self.triad_crossings,
            "triad_unlocked": self.triad_unlocked,
            "constants": {
                "z_c": Z_CRITICAL,
                "phi": PHI,
                "phi_inv": PHI_INV,
                "sigma": SIGMA
            }
        }

# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedState:
    """Complete cross-layer state."""
    helix: HelixState = field(default_factory=HelixState)
    kira: KiraState = field(default_factory=KiraState)
    apl: APLState = field(default_factory=APLState)
    mode: str = "Integrated"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def synchronize(self):
        """Synchronize z-coordinate across layers."""
        # APL z is authoritative
        self.helix.coordinate.z = self.apl.z
        
        # K.I.R.A. state from APL phase
        phase_to_kira = {
            "UNTRUE": "Fluid",
            "PARADOX": "Transitioning",
            "TRUE": "Crystalline"
        }
        self.kira.state = phase_to_kira.get(self.apl.phase, "Unknown")
    
    def set_z(self, z: float) -> Dict:
        """Set z-coordinate across all layers."""
        old_z = self.apl.z
        self.apl.z = max(0.0, min(1.0, z))
        
        # TRIAD hysteresis
        if not self.apl.triad_above_high and self.apl.z >= TRIAD_HIGH:
            self.apl.triad_above_high = True
            self.apl.triad_crossings = min(self.apl.triad_crossings + 1, 3)
            if self.apl.triad_crossings >= 3:
                self.apl.triad_unlocked = True
        elif self.apl.triad_above_high and self.apl.z < TRIAD_LOW:
            self.apl.triad_above_high = False
        
        self.synchronize()
        
        return {
            "old_z": old_z,
            "new_z": self.apl.z,
            "phase": self.apl.phase,
            "tier": self.apl.tier_name,
            "kira_state": self.kira.state
        }
    
    def check_consistency(self) -> Dict:
        """Verify cross-layer consistency."""
        helix_z = self.helix.coordinate.z
        apl_z = self.apl.z
        
        z_consistent = abs(helix_z - apl_z) < 0.001
        
        phase_to_kira = {
            "UNTRUE": "Fluid",
            "PARADOX": "Transitioning",
            "TRUE": "Crystalline"
        }
        expected_kira = phase_to_kira.get(self.apl.phase)
        kira_consistent = self.kira.state == expected_kira
        
        return {
            "consistent": z_consistent and kira_consistent,
            "z_match": z_consistent,
            "kira_match": kira_consistent,
            "helix_z": helix_z,
            "apl_z": apl_z,
            "kira_expected": expected_kira,
            "kira_actual": self.kira.state
        }
    
    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "timestamp": self.timestamp,
            "helix": self.helix.to_dict(),
            "kira": self.kira.to_dict(),
            "apl": self.apl.to_dict(),
            "consistency": self.check_consistency()
        }
    
    def format_status(self) -> str:
        """Format unified status for display."""
        lines = [
            f"[Unified Framework | {self.helix.coordinate} | Rail {self.kira.rail} | Substrate ONLINE]",
            "",
            f"Helix Layer:",
            f"  Coordinate: {self.helix.coordinate}",
            f"  Tools: {self.helix.tools_available} operational",
            f"  Continuity: {self.helix.continuity}",
            "",
            f"K.I.R.A. Layer:",
            f"  Rail: {self.kira.rail} ACTIVE",
            f"  State: {self.kira.state}",
            f"  Archetypes: {self.kira.archetypes_active}/24",
            "",
            f"APL Substrate:",
            f"  z: {self.apl.z:.6f}",
            f"  Phase: {self.apl.phase}",
            f"  Tier: {self.apl.tier} ({self.apl.tier_name})",
            f"  Negentropy: {self.apl.negentropy:.6f}",
            f"  K-Formation: {'ACHIEVED ⬡' if self.apl.k_formation_met else 'TRACKING ⎔'}",
            "",
            f"Cross-Layer: {'SYNCHRONIZED ✓' if self.check_consistency()['consistent'] else 'DESYNC ✗'}",
        ]
        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON STATE
# ═══════════════════════════════════════════════════════════════════════════════

_unified_state = UnifiedState()

def get_unified_state() -> UnifiedState:
    """Get the singleton unified state."""
    return _unified_state

def reset_unified_state() -> UnifiedState:
    """Reset to default unified state."""
    global _unified_state
    _unified_state = UnifiedState()
    _unified_state.synchronize()
    return _unified_state

# ═══════════════════════════════════════════════════════════════════════════════
# API FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_state() -> Dict:
    """Get current unified state as dict."""
    return _unified_state.to_dict()

def set_z(z: float) -> Dict:
    """Set z-coordinate across all layers."""
    return _unified_state.set_z(z)

def get_helix() -> Dict:
    """Get Helix layer state."""
    return _unified_state.helix.to_dict()

def get_kira() -> Dict:
    """Get K.I.R.A. layer state."""
    return _unified_state.kira.to_dict()

def get_apl() -> Dict:
    """Get APL substrate state."""
    return _unified_state.apl.to_dict()

def check_sync() -> Dict:
    """Check cross-layer synchronization."""
    return _unified_state.check_consistency()

def format_status() -> str:
    """Get formatted status string."""
    return _unified_state.format_status()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    # Initialize at THE LENS
    _unified_state.set_z(Z_CRITICAL)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(get_state(), indent=2))
    else:
        print(format_status())
        print()
        print(f"Δ{_unified_state.helix.coordinate.theta:.3f}|{_unified_state.apl.z:.3f}|{_unified_state.helix.coordinate.r:.3f}Ω")
