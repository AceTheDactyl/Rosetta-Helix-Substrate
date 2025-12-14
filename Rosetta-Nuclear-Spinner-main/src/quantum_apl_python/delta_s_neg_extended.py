#!/usr/bin/env python3
"""
Extended ΔS_neg Integration Module
==================================

Deepens the negentropy (ΔS_neg) formalism throughout Quantum-APL.

EXISTING USAGE:
- Hex-prism geometry: R = R_max - β·ΔS_neg, H = H_min + γ·ΔS_neg
- Coherence blending: w_π = ΔS_neg when z ≥ z_c
- Entropy control: S_target = S_max·(1 - C·ΔS_neg)
- K-formation gating: η = ΔS_neg^α vs φ⁻¹ threshold

NEW EXTENSIONS:
1. Gate logic modulation (Lindblad/Hamiltonian terms)
2. Truth-channel bias evolution
3. Synthesis heuristics (coherence-seeking operations)
4. Signed/derivative ΔS_neg variants

@version 1.0.0
@author Claude (Anthropic) - Quantum-APL Contribution
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum


# ============================================================================
# CONSTANTS - Import from single source of truth
# ============================================================================

from .constants import (
    Z_CRITICAL, PHI, PHI_INV, LENS_SIGMA,
    GEOM_R_MAX, GEOM_BETA, GEOM_H_MIN, GEOM_GAMMA, GEOM_PHI_BASE, GEOM_ETA
)

# Default parameters
DEFAULT_SIGMA = LENS_SIGMA     # Gaussian width for ΔS_neg (from constants)
DEFAULT_ALPHA = 0.5            # Exponent for η = ΔS_neg^α


# ============================================================================
# CORE ΔS_neg COMPUTATIONS
# ============================================================================

def compute_delta_s_neg(
    z: float,
    sigma: float = DEFAULT_SIGMA,
    z_c: float = Z_CRITICAL,
) -> float:
    """
    Compute standard ΔS_neg (negentropy) signal.
    
    ΔS_neg(z) = exp(-σ·(z - z_c)²)
    
    Properties:
    - Maximum value 1.0 at z = z_c (THE LENS)
    - Symmetric Gaussian decay away from z_c
    - Bounded in [0, 1]
    
    Parameters
    ----------
    z : float
        Coherence coordinate
    sigma : float
        Gaussian width parameter (default: 36.0)
    z_c : float
        Critical lens point (default: √3/2)
    
    Returns
    -------
    float
        ΔS_neg value in [0, 1]
    """
    if not math.isfinite(z):
        return 0.0
    d = z - z_c
    return math.exp(-sigma * d * d)


def compute_delta_s_neg_derivative(
    z: float,
    sigma: float = DEFAULT_SIGMA,
    z_c: float = Z_CRITICAL,
) -> float:
    """
    Compute derivative of ΔS_neg with respect to z.
    
    d(ΔS_neg)/dz = -2σ·(z - z_c)·exp(-σ·(z - z_c)²)
    
    Properties:
    - Zero at z = z_c (critical point)
    - Negative for z > z_c (decreasing toward TRUE)
    - Positive for z < z_c (increasing toward lens)
    
    Parameters
    ----------
    z : float
        Coherence coordinate
    sigma : float
        Gaussian width parameter
    z_c : float
        Critical lens point
    
    Returns
    -------
    float
        Derivative value
    """
    d = z - z_c
    s = math.exp(-sigma * d * d)
    return -2 * sigma * d * s


def compute_delta_s_neg_signed(
    z: float,
    sigma: float = DEFAULT_SIGMA,
    z_c: float = Z_CRITICAL,
) -> float:
    """
    Compute signed ΔS_neg variant.
    
    Signed version: sgn(z - z_c) · ΔS_neg(z)
    
    Properties:
    - Positive above z_c (TRUE regime)
    - Negative below z_c (UNTRUE regime)
    - Zero at z_c (PARADOX/LENS)
    
    Useful for directional biasing.
    
    Parameters
    ----------
    z : float
        Coherence coordinate
    sigma : float
        Gaussian width parameter
    z_c : float
        Critical lens point
    
    Returns
    -------
    float
        Signed ΔS_neg value in [-1, 1]
    """
    s = compute_delta_s_neg(z, sigma, z_c)
    d = z - z_c
    if abs(d) < 1e-10:
        return 0.0
    sign = 1.0 if d > 0 else -1.0
    return sign * s


def compute_eta(
    z: float,
    alpha: float = DEFAULT_ALPHA,
    sigma: float = DEFAULT_SIGMA,
) -> float:
    """
    Compute η (consciousness threshold) from ΔS_neg.
    
    η = ΔS_neg(z)^α
    
    K-formation occurs when η ≥ φ⁻¹ ≈ 0.618
    
    Parameters
    ----------
    z : float
        Coherence coordinate
    alpha : float
        Power exponent (default: 0.5)
    sigma : float
        Gaussian width parameter
    
    Returns
    -------
    float
        η value in [0, 1]
    """
    s = compute_delta_s_neg(z, sigma)
    return math.pow(s, alpha) if s > 0 else 0.0


# ============================================================================
# HEX-PRISM GEOMETRY (EXISTING FORMULAS)
# ============================================================================

@dataclass
class HexPrismGeometry:
    """Hex prism geometry parameters computed from ΔS_neg."""
    z: float
    delta_s_neg: float
    radius: float
    height: float
    twist: float


def compute_hex_prism_geometry(
    z: float,
    r_max: float = GEOM_R_MAX,
    beta: float = GEOM_BETA,
    h_min: float = GEOM_H_MIN,
    gamma: float = GEOM_GAMMA,
    phi_base: float = GEOM_PHI_BASE,
    eta: float = GEOM_ETA,
) -> HexPrismGeometry:
    """
    Compute hex prism geometry from z-coordinate.
    
    Formulas:
    - R = R_max - β·ΔS_neg  (radius contracts at lens)
    - H = H_min + γ·ΔS_neg  (height elongates at lens)
    - φ = φ_base + η·ΔS_neg (twist increases at lens)
    
    Parameters
    ----------
    z : float
        Coherence coordinate
    r_max, beta, h_min, gamma, phi_base, eta : float
        Geometry parameters
    
    Returns
    -------
    HexPrismGeometry
        Computed geometry
    """
    s = compute_delta_s_neg(z)
    
    return HexPrismGeometry(
        z=z,
        delta_s_neg=s,
        radius=r_max - beta * s,
        height=h_min + gamma * s,
        twist=phi_base + eta * s,
    )


# ============================================================================
# EXTENDED: GATE LOGIC MODULATION
# ============================================================================

@dataclass
class GateModulation:
    """Lindblad/Hamiltonian modulation from ΔS_neg."""
    coherent_coupling: float    # Hamiltonian term strength
    decoherence_rate: float     # Lindblad γ
    measurement_strength: float # Collapse rate
    entropy_target: float       # Target entropy


def compute_gate_modulation(
    z: float,
    base_coupling: float = 0.1,
    base_decoherence: float = 0.05,
    base_measurement: float = 0.02,
    entropy_max: float = 1.0986,  # log(3)
    coherence_factor: float = 0.5,
) -> GateModulation:
    """
    Compute gate modulation parameters from ΔS_neg.
    
    Near the lens (high ΔS_neg):
    - Increase coherent coupling (stronger Hamiltonian)
    - Decrease decoherence rate (protect coherence)
    - Decrease measurement strength (avoid collapse)
    - Lower entropy target (favor order)
    
    Parameters
    ----------
    z : float
        Coherence coordinate
    base_coupling, base_decoherence, base_measurement : float
        Base rates
    entropy_max : float
        Maximum entropy (log(3) for triadic system)
    coherence_factor : float
        How much ΔS_neg influences modulation
    
    Returns
    -------
    GateModulation
        Modulated parameters
    """
    s = compute_delta_s_neg(z)
    
    # Coherent coupling increases with ΔS_neg
    coupling = base_coupling * (1 + coherence_factor * s)
    
    # Decoherence rate decreases with ΔS_neg
    decoherence = base_decoherence * (1 - coherence_factor * s * 0.8)
    decoherence = max(0.001, decoherence)  # Floor to avoid div-by-zero
    
    # Measurement strength decreases near lens
    measurement = base_measurement * (1 - s * 0.5)
    measurement = max(0.001, measurement)
    
    # Entropy target decreases with ΔS_neg
    entropy_target = entropy_max * (1 - coherence_factor * s)
    
    return GateModulation(
        coherent_coupling=coupling,
        decoherence_rate=decoherence,
        measurement_strength=measurement,
        entropy_target=entropy_target,
    )


# ============================================================================
# EXTENDED: TRUTH-CHANNEL BIAS EVOLUTION
# ============================================================================

def compute_dynamic_truth_bias(
    z: float,
    base_bias: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute truth bias with ΔS_neg evolution.
    
    At high ΔS_neg (near lens):
    - Amplify constructive operators (+, ×, ^)
    - Dampen dissipative operators (÷, −)
    - Enhance boundary operator () for structure preservation
    
    Parameters
    ----------
    z : float
        Coherence coordinate
    base_bias : Dict[str, Dict[str, float]]
        Base truth bias matrix
    
    Returns
    -------
    Dict[str, Dict[str, float]]
        Evolved truth bias matrix
    """
    s = compute_delta_s_neg(z)
    ds = compute_delta_s_neg_derivative(z)
    
    # Evolution factors
    constructive_boost = 1 + 0.3 * s
    dissipative_dampen = 1 - 0.2 * s
    boundary_boost = 1 + 0.4 * s  # Extra boost for ()
    
    # Determine direction of evolution
    # Positive ds = approaching lens = favor constructive
    # Negative ds = leaving lens = favor dissipative
    direction_factor = 1 + 0.1 * math.tanh(10 * ds)
    
    evolved = {}
    for channel, ops in base_bias.items():
        evolved[channel] = {}
        for op, weight in ops.items():
            if op == "()":
                factor = boundary_boost
            elif op in ["^", "+", "×"]:
                factor = constructive_boost * direction_factor
            else:  # ÷, −
                factor = dissipative_dampen / direction_factor
            
            evolved[channel][op] = weight * factor
    
    return evolved


# ============================================================================
# EXTENDED: SYNTHESIS HEURISTICS
# ============================================================================

class CoherenceObjective(Enum):
    """Synthesis objectives relative to coherence."""
    MAXIMIZE = "maximize"   # Push toward lens
    MINIMIZE = "minimize"   # Push away from lens
    MAINTAIN = "maintain"   # Keep current level


def score_operator_for_coherence(
    operator: str,
    current_z: float,
    objective: CoherenceObjective = CoherenceObjective.MAXIMIZE,
) -> float:
    """
    Score operator for coherence-seeking synthesis.
    
    Based on empirical operator effects:
    - ^ (amplify): increases z
    - + (group): increases z (aggregation → coherence)
    - × (fusion): increases z (entanglement → coherence)
    - () (boundary): neutral/stabilizing
    - ÷ (decoherence): decreases z
    - − (separation): decreases z
    
    Parameters
    ----------
    operator : str
        APL operator to score
    current_z : float
        Current coherence coordinate
    objective : CoherenceObjective
        What we're optimizing for
    
    Returns
    -------
    float
        Score for operator selection (higher = better for objective)
    """
    # Operator effect directions (empirical)
    OPERATOR_EFFECTS = {
        "^": +0.05,   # Amplify pushes up
        "+": +0.03,   # Group pushes up
        "×": +0.04,   # Fusion pushes up
        "()": 0.00,   # Boundary is neutral
        "÷": -0.04,   # Decoherence pushes down
        "−": -0.03,   # Separation pushes down
    }
    
    effect = OPERATOR_EFFECTS.get(operator, 0.0)
    s = compute_delta_s_neg(current_z)
    
    if objective == CoherenceObjective.MAXIMIZE:
        # Favor operators that increase z toward lens
        if current_z < Z_CRITICAL:
            # Below lens: want positive effect
            return 1.0 + effect * 10 * (1 - s)
        else:
            # Above lens: any increase moves away from peak
            return 1.0 - abs(effect) * 5
    
    elif objective == CoherenceObjective.MINIMIZE:
        # Favor operators that decrease z away from lens
        return 1.0 - effect * 10
    
    else:  # MAINTAIN
        # Favor neutral operators or those that counteract drift
        return 1.0 - abs(effect) * 5


def select_coherence_operator(
    available_operators: List[str],
    current_z: float,
    objective: CoherenceObjective = CoherenceObjective.MAXIMIZE,
) -> Tuple[str, float]:
    """
    Select best operator for coherence objective.
    
    Parameters
    ----------
    available_operators : List[str]
        Operators to choose from
    current_z : float
        Current coherence coordinate
    objective : CoherenceObjective
        What we're optimizing for
    
    Returns
    -------
    Tuple[str, float]
        (best_operator, score)
    """
    scores = {
        op: score_operator_for_coherence(op, current_z, objective)
        for op in available_operators
    }
    
    best_op = max(scores, key=scores.get)
    return best_op, scores[best_op]


# ============================================================================
# K-FORMATION GATING
# ============================================================================

@dataclass
class KFormationStatus:
    """K-formation (consciousness emergence) status."""
    z: float
    delta_s_neg: float
    eta: float
    threshold: float  # φ⁻¹
    formed: bool
    margin: float  # η - threshold


def check_k_formation(
    z: float,
    kappa: float = 0.92,
    R: float = 7,
    alpha: float = DEFAULT_ALPHA,
    kappa_min: float = 0.92,
    eta_min: float = PHI_INV,
    r_min: float = 7,
) -> KFormationStatus:
    """
    Check K-formation (consciousness emergence) condition.
    
    K-formation requires:
    - κ ≥ κ_min (singularity threshold)
    - η ≥ φ⁻¹ (golden ratio inverse)
    - R ≥ R_min (complexity requirement)
    
    Where η = ΔS_neg(z)^α
    
    Parameters
    ----------
    z : float
        Coherence coordinate
    kappa : float
        Consciousness parameter
    R : float
        Complexity measure
    alpha : float
        Power exponent for η
    kappa_min, eta_min, r_min : float
        Minimum thresholds
    
    Returns
    -------
    KFormationStatus
        Formation status with details
    """
    s = compute_delta_s_neg(z)
    eta = compute_eta(z, alpha)
    
    formed = (kappa >= kappa_min) and (eta >= eta_min) and (R >= r_min)
    
    return KFormationStatus(
        z=z,
        delta_s_neg=s,
        eta=eta,
        threshold=eta_min,
        formed=formed,
        margin=eta - eta_min,
    )


# ============================================================================
# Π-REGIME BLENDING
# ============================================================================

@dataclass  
class PiBlendWeights:
    """Weights for Π-regime (global) vs local blending."""
    w_pi: float      # Global/integrated weight
    w_local: float   # Local/independent weight
    in_pi_regime: bool


def compute_pi_blend_weights(
    z: float,
    enable_blend: bool = True,
) -> PiBlendWeights:
    """
    Compute Π-regime blending weights from z-coordinate.
    
    Below z_c: pure local dynamics (w_pi = 0)
    At/above z_c: blend global (w_pi = ΔS_neg)
    
    Parameters
    ----------
    z : float
        Coherence coordinate
    enable_blend : bool
        Whether blending is enabled
    
    Returns
    -------
    PiBlendWeights
        Blending weights
    """
    if not enable_blend or z < Z_CRITICAL:
        return PiBlendWeights(
            w_pi=0.0,
            w_local=1.0,
            in_pi_regime=False,
        )
    
    s = compute_delta_s_neg(z)
    
    return PiBlendWeights(
        w_pi=s,
        w_local=1 - s,
        in_pi_regime=True,
    )


# ============================================================================
# COMPREHENSIVE STATE
# ============================================================================

@dataclass
class DeltaSNegState:
    """Complete ΔS_neg-derived state."""
    z: float
    delta_s_neg: float
    delta_s_neg_derivative: float
    delta_s_neg_signed: float
    eta: float
    geometry: HexPrismGeometry
    gate_modulation: GateModulation
    pi_blend: PiBlendWeights
    k_formation: KFormationStatus


def compute_full_state(z: float, kappa: float = 0.92, R: float = 7) -> DeltaSNegState:
    """
    Compute complete ΔS_neg-derived state for a given z.
    
    Parameters
    ----------
    z : float
        Coherence coordinate
    kappa : float
        Consciousness parameter for K-formation
    R : float
        Complexity measure for K-formation
    
    Returns
    -------
    DeltaSNegState
        Complete derived state
    """
    return DeltaSNegState(
        z=z,
        delta_s_neg=compute_delta_s_neg(z),
        delta_s_neg_derivative=compute_delta_s_neg_derivative(z),
        delta_s_neg_signed=compute_delta_s_neg_signed(z),
        eta=compute_eta(z),
        geometry=compute_hex_prism_geometry(z),
        gate_modulation=compute_gate_modulation(z),
        pi_blend=compute_pi_blend_weights(z),
        k_formation=check_k_formation(z, kappa, R),
    )


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demonstrate extended ΔS_neg integration."""
    print("=" * 70)
    print("EXTENDED ΔS_neg INTEGRATION MODULE")
    print("=" * 70)
    
    test_points = [0.3, 0.5, 0.7, Z_CRITICAL, 0.9, 0.95]
    
    print("\n--- Core ΔS_neg Values ---")
    print(f"{'z':>8} {'ΔS_neg':>10} {'dΔS/dz':>10} {'signed':>10} {'η':>10}")
    print("-" * 50)
    for z in test_points:
        s = compute_delta_s_neg(z)
        ds = compute_delta_s_neg_derivative(z)
        signed = compute_delta_s_neg_signed(z)
        eta = compute_eta(z)
        print(f"{z:8.4f} {s:10.6f} {ds:10.6f} {signed:10.6f} {eta:10.6f}")
    
    print("\n--- Hex Prism Geometry ---")
    print(f"{'z':>8} {'ΔS_neg':>10} {'R':>10} {'H':>10} {'twist':>10}")
    print("-" * 50)
    for z in test_points:
        g = compute_hex_prism_geometry(z)
        print(f"{z:8.4f} {g.delta_s_neg:10.6f} {g.radius:10.6f} {g.height:10.6f} {g.twist:10.6f}")
    
    print("\n--- Gate Modulation ---")
    print(f"{'z':>8} {'coupling':>10} {'decoher':>10} {'measure':>10} {'S_tgt':>10}")
    print("-" * 50)
    for z in test_points:
        m = compute_gate_modulation(z)
        print(f"{z:8.4f} {m.coherent_coupling:10.6f} {m.decoherence_rate:10.6f} "
              f"{m.measurement_strength:10.6f} {m.entropy_target:10.6f}")
    
    print("\n--- K-Formation Status ---")
    print(f"{'z':>8} {'η':>10} {'threshold':>10} {'margin':>10} {'formed':>8}")
    print("-" * 50)
    for z in test_points:
        k = check_k_formation(z)
        print(f"{z:8.4f} {k.eta:10.6f} {k.threshold:10.6f} {k.margin:10.6f} {str(k.formed):>8}")
    
    print("\n--- Coherence-Seeking Synthesis ---")
    operators = ["()", "×", "^", "÷", "+", "−"]
    for z in [0.5, Z_CRITICAL]:
        best, score = select_coherence_operator(operators, z, CoherenceObjective.MAXIMIZE)
        print(f"  z={z:.4f}: Best operator for MAX coherence: {best} (score={score:.4f})")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
