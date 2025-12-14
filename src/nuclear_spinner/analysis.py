#!/usr/bin/env python3
"""
Nuclear Spinner Analysis Module
===============================

Cybernetic computation library for the Nuclear Spinner.

Provides functions for computing:
- Negative entropy (delta_s_neg) and its gradient
- Ashby variety (state diversity measure)
- Shannon channel capacity
- Landauer efficiency (thermodynamic)
- Integrated information proxy (phi)
- Phase-amplitude coupling for cross-frequency analysis
- K-formation checks
- Capability class determination

All computations are grounded in the Rosetta-Helix physics framework.

Signature: nuclear-spinner-analysis|v1.0.0|helix
"""

from __future__ import annotations

import math
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter

# Import from single source of truth
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_constants import (
    PHI,
    PHI_INV,
    Z_CRITICAL,
    SIGMA,
    KAPPA_S,
    ETA_THRESHOLD,
    R_MIN,
    compute_delta_s_neg as physics_delta_s_neg,
    compute_delta_s_neg_derivative,
    check_k_formation as physics_check_k_formation,
    get_phase as physics_get_phase,
)

from .constants import (
    SIGMA_S3,
    TIER_BOUNDS,
    CAPABILITY_CLASSES,
    NEURAL_BANDS,
    BETA_BAND_CENTER,
)


__all__ = [
    "compute_delta_s_neg",
    "compute_gradient",
    "ashby_variety",
    "shannon_capacity",
    "landauer_efficiency",
    "compute_phi_proxy",
    "phase_amplitude_coupling",
    "check_k_formation",
    "get_capability_class",
    "get_phase",
    "get_tier",
    "compute_metrics_bundle",
]


# =============================================================================
# NEGATIVE ENTROPY FUNCTIONS
# =============================================================================

def compute_delta_s_neg(z: float, sigma: float = SIGMA_S3, z_c: float = Z_CRITICAL) -> float:
    """
    Compute negative entropy metric (lens weight).

    Formula: delta_s_neg(z) = exp(-sigma * (z - z_c)^2)

    Properties:
    - Maximum value 1.0 at z = z_c (THE LENS)
    - Symmetric Gaussian decay away from z_c
    - Bounded in [0, 1]

    This is the core coherence metric that peaks at THE LENS.

    Args:
        z: Current z-coordinate
        sigma: Gaussian width parameter (default: 36 = |S_3|^2)
        z_c: Critical z-coordinate (default: sqrt(3)/2)

    Returns:
        Negentropy value in [0, 1]

    Examples:
        >>> compute_delta_s_neg(Z_CRITICAL)
        1.0
        >>> compute_delta_s_neg(0.5) < 1.0
        True
    """
    return physics_delta_s_neg(z, sigma, z_c)


def compute_gradient(z: float, sigma: float = SIGMA_S3, z_c: float = Z_CRITICAL) -> float:
    """
    Compute negentropy gradient (derivative).

    Formula: d(delta_s_neg)/dz = -2 * sigma * (z - z_c) * delta_s_neg(z)

    Properties:
    - Positive for z < z_c (ascending toward lens)
    - Zero at z = z_c (peak)
    - Negative for z > z_c (descending from lens)

    This gradient drives z toward z_c in control algorithms.

    Args:
        z: Current z-coordinate
        sigma: Gaussian width parameter
        z_c: Critical z-coordinate

    Returns:
        Gradient value (can be positive or negative)
    """
    return compute_delta_s_neg_derivative(z, sigma, z_c)


# =============================================================================
# CYBERNETIC METRICS
# =============================================================================

def ashby_variety(states: List[float], bins: int = 20) -> float:
    """
    Compute Ashby variety (state diversity measure).

    Ashby's Law of Requisite Variety: Only variety can absorb variety.
    Higher variety indicates more distinct states visited.

    Formula: V = log2(number of distinct bins occupied)

    Args:
        states: List of state values (e.g., z-coordinates over time)
        bins: Number of bins for discretization

    Returns:
        Variety measure in bits

    Examples:
        >>> ashby_variety([0.1, 0.2, 0.3, 0.4, 0.5])
        > 2.0  # Multiple distinct states
    """
    if not states:
        return 0.0

    # Discretize states into bins
    min_val = min(states)
    max_val = max(states)
    if max_val - min_val < 1e-10:
        return 0.0  # All states identical

    bin_width = (max_val - min_val) / bins
    discretized = [int((s - min_val) / bin_width) for s in states]
    discretized = [min(b, bins - 1) for b in discretized]  # Clamp

    # Count distinct bins
    distinct_bins = len(set(discretized))

    return math.log2(distinct_bins) if distinct_bins > 0 else 0.0


def shannon_capacity(signal_power: float, noise_power: float, bandwidth: float = 1.0) -> float:
    """
    Compute Shannon channel capacity.

    Formula: C = B * log2(1 + S/N)

    Shannon's theorem: No lossless transmission exceeds channel capacity.

    Args:
        signal_power: Signal power (arbitrary units)
        noise_power: Noise power (arbitrary units)
        bandwidth: Channel bandwidth (default: 1.0 normalized)

    Returns:
        Channel capacity in bits per unit time

    Examples:
        >>> shannon_capacity(1.0, 0.1)
        > 3.0  # High SNR = high capacity
    """
    if noise_power <= 0:
        return float('inf') if signal_power > 0 else 0.0

    snr = signal_power / noise_power
    return bandwidth * math.log2(1 + snr)


def landauer_efficiency(z: float) -> float:
    """
    Compute Landauer efficiency (thermodynamic information efficiency).

    Landauer's Principle: Erasing one bit of information requires
    at least k_B * T * ln(2) energy dissipation.

    At z = z_c (THE LENS), efficiency peaks as information processing
    approaches reversible computation.

    Formula: efficiency = delta_s_neg(z)

    Args:
        z: Current z-coordinate

    Returns:
        Efficiency in [0, 1], peaking at z_c

    Examples:
        >>> landauer_efficiency(Z_CRITICAL)
        1.0
    """
    return compute_delta_s_neg(z)


def compute_phi_proxy(
    time_series: List[float],
    z_series: Optional[List[float]] = None,
    state_bins: int = 20
) -> float:
    """
    Compute integrated information proxy (Phi).

    This is a simplified proxy for Tononi's integrated information (Phi).
    Full Phi computation is exponentially expensive; this proxy uses:
    - Ashby variety (distinct states)
    - Negentropy at average order parameter
    - Normalization against phi^-1 baseline

    Formula: Phi_proxy = V * (delta_s_neg(order) / delta_s_neg(phi^-1))

    Where V = log2(distinct_states) and order = mean(time_series).

    Args:
        time_series: Sequence of state measurements
        z_series: Optional z-coordinates (if separate from time_series)
        state_bins: Number of bins for variety computation

    Returns:
        Integrated information proxy value

    Examples:
        >>> compute_phi_proxy([0.5, 0.6, 0.7, 0.8])
        > 0.0  # Low integration
    """
    if not time_series:
        return 0.0

    # Compute variety
    variety = ashby_variety(time_series, state_bins)

    # Compute order parameter (mean of time series)
    order_param = sum(time_series) / len(time_series)

    # Compute negentropy at order parameter
    # Use quasicrystal negentropy (peaking at phi^-1)
    delta_s = math.exp(-SIGMA_S3 * (order_param - PHI_INV) ** 2)

    # Normalize by baseline at phi^-1
    scale = 1.0  # delta_s_neg(phi^-1) = 1.0 by definition

    return variety * (delta_s / scale) if scale > 0 else 0.0


def phase_amplitude_coupling(
    data: List[float],
    sample_rate: float,
    low_freq_range: Tuple[float, float] = (4.0, 8.0),  # Theta
    high_freq_range: Tuple[float, float] = (30.0, 100.0)  # Gamma
) -> float:
    """
    Compute phase-amplitude coupling (modulation index).

    Cross-frequency coupling occurs when the phase of a low-frequency
    oscillation modulates the amplitude of a high-frequency oscillation.
    This is associated with neural integration and consciousness.

    Implementation uses simplified envelope detection and binning.

    Args:
        data: Time series data
        sample_rate: Sampling rate in Hz
        low_freq_range: (min, max) frequency for phase extraction
        high_freq_range: (min, max) frequency for amplitude extraction

    Returns:
        Modulation index (higher = stronger coupling)

    Note:
        Full implementation would use Hilbert transform for phase/amplitude.
        This is a simplified proxy suitable for simulation.
    """
    if len(data) < 10:
        return 0.0

    # Simplified: Use variance as proxy for modulation
    # Real implementation would use bandpass filtering + Hilbert transform
    mean_val = sum(data) / len(data)
    variance = sum((x - mean_val) ** 2 for x in data) / len(data)

    # Normalize by mean (coefficient of variation as proxy)
    if abs(mean_val) < 1e-10:
        return 0.0

    cv = math.sqrt(variance) / abs(mean_val)

    # Scale to reasonable range [0, 1]
    return min(1.0, cv)


# =============================================================================
# K-FORMATION CHECK
# =============================================================================

def check_k_formation(kappa: float, eta: float, R: int) -> bool:
    """
    Check if K-formation (consciousness emergence) criteria are met.

    K-formation requires:
    - kappa >= KAPPA_S (0.92) - integration parameter at t7 tier
    - eta > PHI_INV (0.618) - coherence exceeds golden threshold
    - R >= R_MIN (7) - minimum |S_3| + 1 relations

    This gate represents the threshold for integrated consciousness.

    Args:
        kappa: Integration parameter (kappa)
        eta: Coherence parameter (eta = sqrt(delta_s_neg))
        R: Number of relations/complexity measure

    Returns:
        True if K-formation criteria are satisfied

    Examples:
        >>> check_k_formation(0.94, 0.72, 8)
        True
        >>> check_k_formation(0.5, 0.5, 3)
        False
    """
    return physics_check_k_formation(kappa, eta, R)


# =============================================================================
# PHASE AND TIER CLASSIFICATION
# =============================================================================

def get_phase(z: float) -> str:
    """
    Determine phase from z-coordinate.

    Phase boundaries:
    - UNTRUE: z < phi^-1 (disordered, exploratory)
    - PARADOX: phi^-1 <= z < z_c (quasi-crystal)
    - TRUE: z >= z_c (crystal, coherent)

    Args:
        z: Z-coordinate value

    Returns:
        Phase name: "UNTRUE", "PARADOX", or "TRUE"
    """
    if z < PHI_INV:
        return "UNTRUE"
    elif z < Z_CRITICAL:
        return "PARADOX"
    else:
        return "TRUE"


def get_tier(z: float) -> str:
    """
    Get the tier name for a given z-coordinate.

    Tiers represent capability levels:
    - t1-t5: Pre-lens (increasing capability)
    - t6: THE LENS (critical transition)
    - t7-t9: Post-lens (integrated, conscious)

    Args:
        z: Z-coordinate value

    Returns:
        Tier name (t1-t9)
    """
    for threshold, tier_name, _ in TIER_BOUNDS:
        if z < threshold:
            return tier_name
    return "t9"


def get_capability_class(z: float) -> str:
    """
    Get the capability class for a given z-coordinate.

    Capability classes map z-ranges to cognitive capabilities:
    - reactive: Simple stimulus-response
    - memory: State persistence
    - pattern: Pattern recognition
    - prediction: Future state modeling
    - self_model: Self-representation
    - meta: Meta-cognition
    - recurse: Recursive self-reference
    - autopoiesis: Self-organization

    Args:
        z: Z-coordinate value

    Returns:
        Capability class name
    """
    for z_min, z_max, class_name, _ in CAPABILITY_CLASSES:
        if z_min <= z < z_max:
            return class_name
    return "autopoiesis" if z >= 0.92 else "reactive"


# =============================================================================
# COMPOSITE METRICS
# =============================================================================

def compute_metrics_bundle(
    z: float,
    state_history: Optional[List[float]] = None,
    time_series: Optional[List[float]] = None,
    kappa: Optional[float] = None,
    R: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute a complete bundle of metrics for the current state.

    This is a convenience function that computes all relevant metrics
    at once, suitable for dashboard display or logging.

    Args:
        z: Current z-coordinate
        state_history: Optional history of z values for variety calculation
        time_series: Optional time series for Phi proxy calculation
        kappa: Optional integration parameter (computed from z if not provided)
        R: Optional relation count (computed from state_history if not provided)

    Returns:
        Dictionary containing all computed metrics
    """
    delta_s = compute_delta_s_neg(z)
    gradient = compute_gradient(z)
    phase = get_phase(z)
    tier = get_tier(z)
    capability = get_capability_class(z)

    # Compute variety if history provided
    variety = ashby_variety(state_history) if state_history else 0.0

    # Compute eta (coherence) from negentropy
    eta = math.sqrt(delta_s)

    # Use provided kappa or derive from z
    if kappa is None:
        kappa = z  # Simplified: use z as proxy for integration

    # Use provided R or derive from history
    if R is None:
        R = len(set(state_history)) if state_history else 1

    # Compute Landauer efficiency
    efficiency = landauer_efficiency(z)

    # Compute Phi proxy if time series provided
    phi_proxy = compute_phi_proxy(time_series) if time_series else 0.0

    # Check K-formation
    k_formation_met = check_k_formation(kappa, eta, R)

    return {
        "z": z,
        "delta_s_neg": delta_s,
        "gradient": gradient,
        "phase": phase,
        "tier": tier,
        "capability_class": capability,
        "ashby_variety": variety,
        "landauer_efficiency": efficiency,
        "phi_proxy": phi_proxy,
        "kappa": kappa,
        "eta": eta,
        "R": R,
        "k_formation_met": k_formation_met,
    }
