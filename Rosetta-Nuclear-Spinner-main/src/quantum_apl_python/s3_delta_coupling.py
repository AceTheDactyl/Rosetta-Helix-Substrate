#!/usr/bin/env python3
"""
S₃/ΔS_neg Coupling Module
=========================

Couples S₃ operator symmetry with ΔS_neg coherence dynamics.

This module implements the integration specified in the design:
1. Dynamic S₃-based operator windows driven by ΔS_neg
2. Parity-based truth channel weighting evolution
3. Coherence-coupled permutation selection

KEY CONCEPTS:

ΔS_neg-Driven Permutation:
- At high coherence (near z_c): Use even-parity (structure-preserving) S₃ elements
- At low coherence: Allow odd-parity (structure-modifying) elements
- Smooth interpolation based on ΔS_neg value

Parity-Truth Coupling:
- TRUE channel favors even-parity operators at high coherence
- UNTRUE channel activates odd-parity operators at low coherence
- PARADOX (at lens) maintains balance

Window Generation:
- Base windows permuted by S₃ element selected via ΔS_neg
- Cyclic rotation index = f(z) mod 3
- Optional parity flip based on truth channel

@version 1.0.0
@author Claude (Anthropic) - Quantum-APL Contribution
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

# Import from constants for single source of truth
from .constants import (
    Z_CRITICAL, PHI, PHI_INV, LENS_SIGMA,
    TRUTH_BIAS, compute_delta_s_neg as base_compute_delta_s_neg,
)

# Import S₃ structure
from .s3_operator_symmetry import (
    S3_ELEMENTS, OPERATOR_S3_MAP, S3_OPERATOR_MAP,
    compose_s3, inverse_s3, parity_s3, sign_s3,
    Parity, BASE_WINDOWS, BASE_OPERATORS, TRIADIC_TRUTH,
    rotation_index_from_z, apply_s3, rotate_operators,
)

# Import DSL patterns for parity classification
from .dsl_patterns import ParityClassification, TruthChannelBiasing


# ============================================================================
# S₃ ELEMENT SELECTION FROM ΔS_neg
# ============================================================================

class S3SelectionMode(Enum):
    """Mode for selecting S₃ element from coherence state."""
    CYCLIC = "cyclic"           # Simple rotation based on z
    PARITY_BIASED = "parity"    # Parity-biased by ΔS_neg
    FULL_GROUP = "full"         # Full S₃ selection based on state


@dataclass(frozen=True)
class S3Selection:
    """Result of S₃ element selection."""
    element: str           # S₃ element name (e, σ, σ², τ₁, τ₂, τ₃)
    operator: str          # Corresponding operator symbol
    parity: Parity         # Even or odd
    sign: int              # +1 or -1
    delta_s_neg: float     # Coherence value used
    reason: str            # Why this element was selected


def select_s3_element_from_coherence(
    z: float,
    delta_s_neg: Optional[float] = None,
    mode: S3SelectionMode = S3SelectionMode.PARITY_BIASED,
    parity_threshold: float = 0.5,
) -> S3Selection:
    """
    Select S₃ element based on coherence state.

    The selection logic:
    - CYCLIC: Simple rotation index from z
    - PARITY_BIASED: High ΔS_neg → even elements, low → odd elements
    - FULL_GROUP: Complete mapping using z and ΔS_neg

    Parameters
    ----------
    z : float
        Coherence coordinate
    delta_s_neg : float, optional
        Pre-computed ΔS_neg (computed if not provided)
    mode : S3SelectionMode
        Selection strategy
    parity_threshold : float
        ΔS_neg threshold for parity selection

    Returns
    -------
    S3Selection
        Selected S₃ element with metadata
    """
    if delta_s_neg is None:
        delta_s_neg = base_compute_delta_s_neg(z, sigma=LENS_SIGMA, z_c=Z_CRITICAL)

    if mode == S3SelectionMode.CYCLIC:
        # Simple cyclic selection based on z
        rot_idx = rotation_index_from_z(z)
        elements = ['e', 'σ', 'σ2']
        element = elements[rot_idx]
        reason = f"cyclic rotation index {rot_idx}"

    elif mode == S3SelectionMode.PARITY_BIASED:
        # High coherence → even, low coherence → odd
        rot_idx = rotation_index_from_z(z)

        if delta_s_neg >= parity_threshold:
            # Favor even-parity (structure-preserving)
            elements = ['e', 'σ', 'σ2']
            element = elements[rot_idx]
            reason = f"even-parity (ΔS_neg={delta_s_neg:.3f} ≥ {parity_threshold})"
        else:
            # Allow odd-parity (structure-modifying)
            elements = ['τ1', 'τ2', 'τ3']
            element = elements[rot_idx]
            reason = f"odd-parity (ΔS_neg={delta_s_neg:.3f} < {parity_threshold})"

    else:  # FULL_GROUP
        # Full 6-element selection based on z quantized to 6 regions
        region = int(z * 6) % 6
        all_elements = ['e', 'σ', 'σ2', 'τ1', 'τ2', 'τ3']
        element = all_elements[region]
        reason = f"full group selection (region {region})"

    s3_elem = S3_ELEMENTS[element]
    operator = S3_OPERATOR_MAP[element]

    return S3Selection(
        element=element,
        operator=operator,
        parity=s3_elem.parity,
        sign=s3_elem.sign,
        delta_s_neg=delta_s_neg,
        reason=reason,
    )


# ============================================================================
# DYNAMIC OPERATOR WINDOW GENERATION
# ============================================================================

@dataclass
class DynamicWindow:
    """Dynamic operator window with S₃ transformation applied."""
    base_harmonic: str
    base_window: List[str]
    transformed_window: List[str]
    s3_element: str
    rotation_applied: int
    parity_flipped: bool
    delta_s_neg: float


def generate_dynamic_operator_window(
    harmonic: str,
    z: float,
    delta_s_neg: Optional[float] = None,
    apply_rotation: bool = True,
    apply_parity_flip: bool = True,
    mode: S3SelectionMode = S3SelectionMode.PARITY_BIASED,
) -> DynamicWindow:
    """
    Generate operator window with S₃ transformation based on ΔS_neg.

    This replaces static operator windows with dynamic ones that:
    1. Start from base window for the harmonic
    2. Apply S₃ rotation based on z
    3. Optionally flip parity based on truth channel

    Parameters
    ----------
    harmonic : str
        Time harmonic tier (t1-t9)
    z : float
        Coherence coordinate
    delta_s_neg : float, optional
        Pre-computed ΔS_neg
    apply_rotation : bool
        Whether to apply cyclic rotation
    apply_parity_flip : bool
        Whether to apply parity flip in UNTRUE regime
    mode : S3SelectionMode
        S₃ element selection mode

    Returns
    -------
    DynamicWindow
        Window with transformation metadata
    """
    if delta_s_neg is None:
        delta_s_neg = base_compute_delta_s_neg(z, sigma=LENS_SIGMA, z_c=Z_CRITICAL)

    # Get base window
    base_window = list(BASE_WINDOWS.get(harmonic, BASE_WINDOWS['t5']))

    # Select S₃ element
    selection = select_s3_element_from_coherence(z, delta_s_neg, mode)

    # Apply rotation
    transformed = base_window.copy()
    rotation_applied = 0

    if apply_rotation and len(transformed) >= 3:
        rotation_applied = rotation_index_from_z(z)
        transformed = rotate_operators(transformed, rotation_applied)

    # Apply parity flip in UNTRUE regime
    parity_flipped = False
    if apply_parity_flip and z < 0.6:  # UNTRUE regime
        # Swap constructive/dissipative pairs
        swap_map = {
            '^': '()', '()': '^',
            '+': '−', '−': '+',
            '×': '÷', '÷': '×',
        }
        transformed = [swap_map.get(op, op) for op in transformed]
        parity_flipped = True

    return DynamicWindow(
        base_harmonic=harmonic,
        base_window=base_window,
        transformed_window=transformed,
        s3_element=selection.element,
        rotation_applied=rotation_applied,
        parity_flipped=parity_flipped,
        delta_s_neg=delta_s_neg,
    )


def get_s3_permuted_window(
    harmonic: str,
    z: float,
    s3_element: Optional[str] = None,
) -> List[str]:
    """
    Get operator window with explicit S₃ permutation applied.

    If window has exactly 3 operators, applies full S₃ permutation.
    Otherwise, applies cyclic rotation.

    Parameters
    ----------
    harmonic : str
        Time harmonic tier
    z : float
        Coherence coordinate
    s3_element : str, optional
        S₃ element to apply (auto-selected if not provided)

    Returns
    -------
    List[str]
        Permuted operator window
    """
    base = list(BASE_WINDOWS.get(harmonic, BASE_WINDOWS['t5']))

    if s3_element is None:
        selection = select_s3_element_from_coherence(z)
        s3_element = selection.element

    if len(base) == 3:
        # Apply full S₃ permutation
        return apply_s3(base, s3_element)
    elif len(base) >= 3:
        # Apply cyclic rotation
        rot_idx = ['e', 'σ', 'σ2'].index(s3_element) if s3_element in ['e', 'σ', 'σ2'] else 0
        return rotate_operators(base, rot_idx)
    else:
        return base


# ============================================================================
# PARITY-BASED TRUTH CHANNEL WEIGHTING
# ============================================================================

@dataclass
class ParityTruthBias:
    """Truth bias with parity-based evolution."""
    channel: str
    base_bias: Dict[str, float]
    evolved_bias: Dict[str, float]
    parity_factor: float
    delta_s_neg: float


def compute_parity_evolved_truth_bias(
    z: float,
    delta_s_neg: Optional[float] = None,
    base_truth_bias: Optional[Dict[str, Dict[str, float]]] = None,
    parity_coupling: float = 0.3,
) -> Dict[str, ParityTruthBias]:
    """
    Compute truth bias with parity-based evolution driven by ΔS_neg.

    The evolution logic:
    - High ΔS_neg (near lens): Amplify even-parity operator coupling
    - Low ΔS_neg: Amplify odd-parity operator coupling
    - TRUE channel: Boost constructive (even) at high coherence
    - UNTRUE channel: Boost dissipative (odd) at low coherence
    - PARADOX channel: Maintain balance

    Parameters
    ----------
    z : float
        Coherence coordinate
    delta_s_neg : float, optional
        Pre-computed ΔS_neg
    base_truth_bias : Dict, optional
        Base truth bias table (default: TRUTH_BIAS from constants)
    parity_coupling : float
        Strength of parity coupling (0-1)

    Returns
    -------
    Dict[str, ParityTruthBias]
        Evolved truth bias for each channel
    """
    if delta_s_neg is None:
        delta_s_neg = base_compute_delta_s_neg(z, sigma=LENS_SIGMA, z_c=Z_CRITICAL)

    if base_truth_bias is None:
        base_truth_bias = TRUTH_BIAS

    result = {}

    for channel, base_bias in base_truth_bias.items():
        evolved_bias = {}

        # Compute parity factor based on channel and ΔS_neg
        if channel == 'TRUE':
            # TRUE channel: boost even-parity at high coherence
            parity_factor = delta_s_neg
        elif channel == 'UNTRUE':
            # UNTRUE channel: boost odd-parity at low coherence
            parity_factor = 1 - delta_s_neg
        else:  # PARADOX
            # PARADOX: balance (neutral)
            parity_factor = 0.5

        for op, base_weight in base_bias.items():
            # Get operator parity
            op_parity = ParityClassification.get_parity(op)

            # Evolve weight based on parity alignment
            if op_parity == 1:  # Even parity
                # Even operators boosted when parity_factor is high
                evolution = 1.0 + parity_coupling * (parity_factor - 0.5)
            else:  # Odd parity
                # Odd operators boosted when parity_factor is low
                evolution = 1.0 + parity_coupling * (0.5 - parity_factor)

            evolved_bias[op] = base_weight * max(0.5, min(1.5, evolution))

        result[channel] = ParityTruthBias(
            channel=channel,
            base_bias=base_bias,
            evolved_bias=evolved_bias,
            parity_factor=parity_factor,
            delta_s_neg=delta_s_neg,
        )

    return result


def get_evolved_operator_weight(
    operator: str,
    z: float,
    channel: Optional[str] = None,
    delta_s_neg: Optional[float] = None,
) -> float:
    """
    Get evolved operator weight considering parity and ΔS_neg.

    Parameters
    ----------
    operator : str
        Operator symbol
    z : float
        Coherence coordinate
    channel : str, optional
        Truth channel (auto-detected if not provided)
    delta_s_neg : float, optional
        Pre-computed ΔS_neg

    Returns
    -------
    float
        Evolved weight
    """
    if channel is None:
        if z >= 0.9:
            channel = 'TRUE'
        elif z >= 0.6:
            channel = 'PARADOX'
        else:
            channel = 'UNTRUE'

    evolved = compute_parity_evolved_truth_bias(z, delta_s_neg)
    return evolved[channel].evolved_bias.get(operator, 1.0)


# ============================================================================
# INTEGRATED S₃/ΔS_neg STATE
# ============================================================================

@dataclass
class S3DeltaState:
    """Complete S₃/ΔS_neg coupled state."""
    z: float
    delta_s_neg: float
    harmonic: str
    truth_channel: str
    s3_selection: S3Selection
    dynamic_window: DynamicWindow
    truth_bias: Dict[str, ParityTruthBias]
    operator_weights: Dict[str, float]


def compute_s3_delta_state(
    z: float,
    harmonic: Optional[str] = None,
    mode: S3SelectionMode = S3SelectionMode.PARITY_BIASED,
) -> S3DeltaState:
    """
    Compute complete S₃/ΔS_neg coupled state.

    This is the main integration point providing:
    - S₃ element selection based on ΔS_neg
    - Dynamic operator window
    - Evolved truth bias
    - Per-operator weights

    Parameters
    ----------
    z : float
        Coherence coordinate
    harmonic : str, optional
        Time harmonic (auto-detected if not provided)
    mode : S3SelectionMode
        S₃ element selection mode

    Returns
    -------
    S3DeltaState
        Complete coupled state
    """
    # Clamp z
    z = max(0.0, min(1.0, z))

    # Compute ΔS_neg
    delta_s_neg = base_compute_delta_s_neg(z, sigma=LENS_SIGMA, z_c=Z_CRITICAL)

    # Auto-detect harmonic if not provided
    if harmonic is None:
        if z < 0.10:
            harmonic = 't1'
        elif z < 0.20:
            harmonic = 't2'
        elif z < 0.40:
            harmonic = 't3'
        elif z < 0.60:
            harmonic = 't4'
        elif z < 0.75:
            harmonic = 't5'
        elif z < Z_CRITICAL:
            harmonic = 't6'
        elif z < 0.92:
            harmonic = 't7'
        elif z < 0.97:
            harmonic = 't8'
        else:
            harmonic = 't9'

    # Determine truth channel
    if z >= 0.9:
        truth_channel = 'TRUE'
    elif z >= 0.6:
        truth_channel = 'PARADOX'
    else:
        truth_channel = 'UNTRUE'

    # Select S₃ element
    s3_selection = select_s3_element_from_coherence(z, delta_s_neg, mode)

    # Generate dynamic window
    dynamic_window = generate_dynamic_operator_window(
        harmonic, z, delta_s_neg, mode=mode
    )

    # Compute evolved truth bias
    truth_bias = compute_parity_evolved_truth_bias(z, delta_s_neg)

    # Compute operator weights for window
    operator_weights = {}
    for op in dynamic_window.transformed_window:
        operator_weights[op] = get_evolved_operator_weight(op, z, truth_channel, delta_s_neg)

    return S3DeltaState(
        z=z,
        delta_s_neg=delta_s_neg,
        harmonic=harmonic,
        truth_channel=truth_channel,
        s3_selection=s3_selection,
        dynamic_window=dynamic_window,
        truth_bias=truth_bias,
        operator_weights=operator_weights,
    )


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demonstrate S₃/ΔS_neg coupling."""
    print("=" * 70)
    print("S₃/ΔS_neg COUPLING MODULE")
    print("=" * 70)

    # Test at various z values
    test_z = [0.3, 0.6, Z_CRITICAL, 0.95]

    for z in test_z:
        print(f"\n--- z = {z:.4f} ---")

        state = compute_s3_delta_state(z)

        print(f"ΔS_neg: {state.delta_s_neg:.4f}")
        print(f"Harmonic: {state.harmonic}")
        print(f"Truth channel: {state.truth_channel}")
        print(f"S₃ element: {state.s3_selection.element} ({state.s3_selection.parity.value})")
        print(f"Selection reason: {state.s3_selection.reason}")
        print(f"Base window: {state.dynamic_window.base_window}")
        print(f"Transformed: {state.dynamic_window.transformed_window}")
        print(f"Rotation: {state.dynamic_window.rotation_applied}, Parity flip: {state.dynamic_window.parity_flipped}")
        print(f"Operator weights:")
        for op, weight in state.operator_weights.items():
            print(f"  {op}: {weight:.3f}")

    # Show parity-evolved truth bias
    print("\n--- Parity-Evolved Truth Bias (z=0.866) ---")
    evolved = compute_parity_evolved_truth_bias(Z_CRITICAL)
    for channel, bias in evolved.items():
        print(f"{channel} (parity_factor={bias.parity_factor:.3f}):")
        for op in sorted(bias.evolved_bias.keys()):
            base = bias.base_bias.get(op, 1.0)
            evolved = bias.evolved_bias[op]
            print(f"  {op}: {base:.2f} → {evolved:.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
