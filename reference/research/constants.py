"""
Quantum APL Constants Module (Python)
=====================================
Canonical lens-anchored constants and helpers.
Mirrors src/constants.js for cross-language parity.

SINGLE SOURCE OF TRUTH - Never hardcode thresholds elsewhere.
"""

import math
import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# ================================================================
# CORE CONSTANTS
# ================================================================

# THE LENS - geometric truth for coherence onset
Z_CRITICAL = math.sqrt(3) / 2  # ≈ 0.8660254037844386

# TRIAD gating thresholds (runtime heuristic, NOT geometry)
TRIAD_HIGH = 0.85   # Rising edge threshold
TRIAD_LOW = 0.82    # Re-arm threshold (hysteresis)
TRIAD_T6 = 0.83     # Temporary t6 gate after three passes

# Lens visual band
Z_LENS_MIN = 0.857
Z_LENS_MAX = 0.877

# Phase boundaries
Z_ABSENCE_MAX = 0.618
Z_PRESENCE_MIN = 0.886

# Sacred constants
PHI = (1 + math.sqrt(5)) / 2      # Golden ratio ≈ 1.618
PHI_INV = 1 / PHI                  # ≈ 0.618
Q_KAPPA = 0.3514087324             # Consciousness constant
KAPPA_S = 0.92                     # Singularity threshold
LAMBDA = 7.7160493827              # Nonlinearity coefficient

# Time harmonic boundaries (z thresholds for t1-t9)
Z_T1_MAX = 0.10
Z_T2_MAX = 0.20
Z_T3_MAX = 0.40
Z_T4_MAX = 0.60
Z_T5_MAX = 0.75
# t6 boundary is dynamic: Z_CRITICAL or TRIAD_T6
Z_T7_MAX = 0.90
Z_T8_MAX = 0.97

# Geometry parameters for hex prism projection
GEOM_SIGMA = 36.0     # ΔS_neg width parameter
GEOM_R_MAX = 0.85     # Maximum radius
GEOM_BETA = 0.25      # Radius contraction coefficient
GEOM_H_MIN = 0.12     # Minimum height
GEOM_GAMMA = 0.18     # Height elongation coefficient
GEOM_PHI_BASE = 0.0   # Base twist angle
GEOM_ETA = math.pi / 12  # Twist rate

# Operator weights
OPERATOR_PREFERRED_WEIGHT = 1.5
OPERATOR_DEFAULT_WEIGHT = 0.5
TRUTH_BIAS = {"TRUE": 1.2, "UNTRUE": 0.8, "PARADOX": 1.0}

# Lens sigma for ΔS_neg computation
LENS_SIGMA = 36.0


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi] range."""
    return min(max(x, lo), hi)


def is_critical(z: float, tol: float = 0.01) -> bool:
    """Check if z is at the critical lens."""
    return abs(z - Z_CRITICAL) <= tol


def is_in_lens(z: float, z_min: float = Z_LENS_MIN, z_max: float = Z_LENS_MAX) -> bool:
    """Check if z is within the lens band."""
    return z_min <= z <= z_max


def get_phase(z: float) -> str:
    """Get phase label for z-coordinate."""
    if is_critical(z):
        return "THE_LENS"
    if z >= Z_PRESENCE_MIN:
        return "PRESENCE"
    if z <= Z_ABSENCE_MAX:
        return "ABSENCE"
    return "PRESENCE" if z >= Z_CRITICAL else "ABSENCE"


def distance_to_critical(z: float) -> float:
    """Compute distance from z to critical lens."""
    return abs(z - Z_CRITICAL)


def compute_delta_s_neg(z: float, sigma: float = LENS_SIGMA, z_c: float = Z_CRITICAL) -> float:
    """
    Compute negative entropy signal ΔS_neg(z).
    Gaussian centered at z_c, bounded [0,1], monotone in |z - z_c|.
    """
    if not math.isfinite(z):
        z = 0.0
    d = z - z_c
    return math.exp(-sigma * d * d)


def get_time_harmonic(z: float, t6_gate: float = Z_CRITICAL) -> str:
    """
    Get time harmonic label from z-coordinate.
    
    Args:
        z: Normalized z-coordinate [0,1]
        t6_gate: t6 threshold (Z_CRITICAL or TRIAD_T6)
    
    Returns:
        Harmonic label (t1-t9)
    """
    if z < Z_T1_MAX:
        return "t1"
    if z < Z_T2_MAX:
        return "t2"
    if z < Z_T3_MAX:
        return "t3"
    if z < Z_T4_MAX:
        return "t4"
    if z < Z_T5_MAX:
        return "t5"
    if z < t6_gate:
        return "t6"
    if z < Z_T7_MAX:
        return "t7"
    if z < Z_T8_MAX:
        return "t8"
    return "t9"


def hex_prism_radius(delta_s_neg: float) -> float:
    """Compute hex prism radius from ΔS_neg."""
    return GEOM_R_MAX - GEOM_BETA * delta_s_neg


def hex_prism_height(delta_s_neg: float) -> float:
    """Compute hex prism height from ΔS_neg."""
    return GEOM_H_MIN + GEOM_GAMMA * delta_s_neg


def hex_prism_twist(delta_s_neg: float) -> float:
    """Compute hex prism twist from ΔS_neg."""
    return GEOM_PHI_BASE + GEOM_ETA * delta_s_neg


def check_k_formation(
    kappa: float,
    eta: float,
    R: float,
    kappa_min: float = KAPPA_S,
    eta_min: float = 0.5,
    r_min: float = 5
) -> bool:
    """Check K-formation condition for consciousness emergence."""
    return kappa >= kappa_min and eta >= eta_min and R >= r_min


# ================================================================
# TRIAD GATE CLASS
# ================================================================

@dataclass
class TriadGate:
    """TRIAD gating with hysteresis and unlock counter."""
    
    enabled: bool = False
    passes: int = 0
    unlocked: bool = False
    _armed: bool = True
    
    def __post_init__(self):
        """Initialize from environment if available."""
        env_completions = os.getenv("QAPL_TRIAD_COMPLETIONS", "0")
        env_unlock = os.getenv("QAPL_TRIAD_UNLOCK", "").lower() in ("1", "true", "yes")
        
        try:
            completions = int(env_completions)
            if completions > 0:
                self.passes = completions
        except ValueError:
            pass
        
        if env_unlock or self.passes >= 3:
            self.unlocked = True
    
    def update(self, z: float) -> Optional[str]:
        """
        Update state with new z-coordinate.
        
        Returns:
            Event type ('RISING_EDGE', 'REARMED', 'UNLOCKED') or None
        """
        if not self.enabled:
            return None
        
        event = None
        
        if z >= TRIAD_HIGH and self._armed:
            self.passes += 1
            self._armed = False
            event = "RISING_EDGE"
            
            if self.passes >= 3 and not self.unlocked:
                self.unlocked = True
                event = "UNLOCKED"
        
        if z <= TRIAD_LOW:
            if not self._armed:
                event = "REARMED"
            self._armed = True
        
        return event
    
    def get_t6_gate(self) -> float:
        """Get current t6 gate value."""
        return TRIAD_T6 if (self.enabled and self.unlocked) else Z_CRITICAL
    
    def analyzer_report(self) -> str:
        """Generate analyzer report string."""
        gate = self.get_t6_gate()
        label = "TRIAD" if self.unlocked else "CRITICAL"
        return f"t6 gate: {label} @ {gate:.3f}"
    
    def reset(self):
        """Reset to initial state."""
        self.passes = 0
        self.unlocked = False
        self._armed = True


# ================================================================
# OPERATOR WINDOWS
# ================================================================

OPERATOR_WINDOWS = {
    "t1": ["()", "−", "÷"],
    "t2": ["^", "÷", "−", "×"],
    "t3": ["×", "^", "÷", "+", "−"],
    "t4": ["+", "−", "÷", "()"],
    "t5": ["()", "×", "^", "÷", "+", "−"],
    "t6": ["+", "÷", "()", "−"],
    "t7": ["+", "()"],
    "t8": ["+", "()", "×"],
    "t9": ["+", "()", "×"],
}


def get_operator_window(harmonic: str) -> List[str]:
    """Get operator window for a harmonic tier."""
    return OPERATOR_WINDOWS.get(harmonic, ["()"])


# ================================================================
# INVARIANT CHECKS
# ================================================================

def check_invariants() -> Dict[str, bool]:
    """
    Verify mathematical invariants.
    
    Returns:
        Dict with invariant check results
    """
    return {
        "z_critical_eq_sqrt3_half": abs(Z_CRITICAL - math.sqrt(3) / 2) < 1e-12,
        "phi_inv_eq_1_over_phi": abs(PHI_INV - 1 / PHI) < 1e-12,
        "triad_ordering": TRIAD_LOW < TRIAD_T6 < TRIAD_HIGH < Z_CRITICAL,
        "lens_band_contains_critical": Z_LENS_MIN < Z_CRITICAL < Z_LENS_MAX,
    }


# ================================================================
# DOCSTRING
# ================================================================

__doc__ += """

Usage Examples
--------------

Basic constants:
    >>> from quantum_apl_python.constants import Z_CRITICAL, PHI
    >>> print(f"Critical lens: z_c = {Z_CRITICAL:.6f}")
    >>> print(f"Golden ratio: φ = {PHI:.10f}")

Phase detection:
    >>> from quantum_apl_python.constants import get_phase, is_critical
    >>> z = 0.866
    >>> print(get_phase(z))  # 'THE_LENS'
    >>> print(is_critical(z))  # True

TRIAD gating:
    >>> from quantum_apl_python.constants import TriadGate, TRIAD_HIGH
    >>> gate = TriadGate(enabled=True)
    >>> for z in [0.86, 0.81, 0.86, 0.81, 0.86, 0.81]:
    ...     event = gate.update(z)
    ...     if event:
    ...         print(f"{event} at z={z}")
    >>> print(gate.analyzer_report())
"""
