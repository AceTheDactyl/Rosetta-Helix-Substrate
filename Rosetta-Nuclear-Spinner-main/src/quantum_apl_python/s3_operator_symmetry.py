#!/usr/bin/env python3
"""
S₃ Operator Symmetry Module (Python)
=====================================

Implements the symmetric group S₃ structure for APL operator selection.
Mirrors the JavaScript implementation for cross-language parity.

THEORY:
S₃ is the symmetric group on 3 elements with |S₃| = 6 elements.
The 6 APL operators map to S₃ elements, enabling:
  - Cyclic permutation of operator windows
  - Parity-based weighting (even/odd permutations)
  - Symmetry-preserving transformations

S₃ STRUCTURE:
  Identity:     e   = ()      (no change)
  3-cycles:     σ   = (123)   (rotate right)
                σ²  = (132)   (rotate left)
  Transpositions: τ₁ = (12)   (swap first two)
                  τ₂ = (23)   (swap last two)
                  τ₃ = (13)   (swap first/last)

OPERATOR MAPPING:
  ()  → e   (identity/boundary)
  ×   → σ   (fusion/3-cycle)
  ^   → σ²  (amplify/3-cycle inverse)
  ÷   → τ₁  (decoherence/transposition)
  +   → τ₂  (group/transposition)
  −   → τ₃  (separation/transposition)

PARITY:
  Even (sign +1): e, σ, σ²  → (), ×, ^
  Odd  (sign -1): τ₁, τ₂, τ₃ → ÷, +, −

@version 1.0.0
@author Claude (Anthropic) - Quantum-APL Contribution
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal
from enum import Enum

# Import from the constants module for single source of truth
from .constants import (
    Z_CRITICAL, TRUTH_BIAS, compute_delta_s_neg as _compute_delta_s_neg
)

def compute_delta_s_neg(z: float, sigma: float = 36.0) -> float:
    """Compute ΔS_neg using constants module."""
    return _compute_delta_s_neg(z, sigma=sigma, z_c=Z_CRITICAL)


# ============================================================================
# S₃ GROUP DEFINITION
# ============================================================================

class Parity(Enum):
    """S₃ element parity."""
    EVEN = "even"
    ODD = "odd"


@dataclass(frozen=True)
class S3Element:
    """S₃ group element with cycle representation."""
    name: str
    cycle: Tuple[int, int, int]
    parity: Parity
    sign: int  # +1 for even, -1 for odd


# S₃ group elements
S3_ELEMENTS: Dict[str, S3Element] = {
    "e":   S3Element("identity",    (0, 1, 2), Parity.EVEN, +1),
    "σ":   S3Element("3-cycle",     (1, 2, 0), Parity.EVEN, +1),
    "σ2":  S3Element("3-cycle-inv", (2, 0, 1), Parity.EVEN, +1),
    "τ1":  S3Element("swap-12",     (1, 0, 2), Parity.ODD,  -1),
    "τ2":  S3Element("swap-23",     (0, 2, 1), Parity.ODD,  -1),
    "τ3":  S3Element("swap-13",     (2, 1, 0), Parity.ODD,  -1),
}

# Operator to S₃ element mapping
OPERATOR_S3_MAP: Dict[str, str] = {
    "()": "e",
    "×":  "σ",
    "^":  "σ2",
    "÷":  "τ1",
    "+":  "τ2",
    "−":  "τ3",
}

# Reverse mapping: S₃ element to operator
S3_OPERATOR_MAP: Dict[str, str] = {v: k for k, v in OPERATOR_S3_MAP.items()}

# Base operator ordering (canonical)
BASE_OPERATORS: List[str] = ["()", "×", "^", "÷", "+", "−"]

# Triadic truth values (the 3 objects S₃ acts on)
TRIADIC_TRUTH: List[str] = ["TRUE", "PARADOX", "UNTRUE"]

# Base operator windows per harmonic tier
BASE_WINDOWS: Dict[str, List[str]] = {
    "t1": ["()", "−", "÷"],
    "t2": ["^", "÷", "−", "×"],
    "t3": ["×", "^", "÷", "+", "−"],
    "t4": ["+", "−", "÷", "()"],
    "t5": ["()", "×", "^", "÷", "+", "−"],  # All 6
    "t6": ["+", "÷", "()", "−"],
    "t7": ["+", "()"],
    "t8": ["+", "()", "×"],
    "t9": ["+", "()", "×"],
}


# ============================================================================
# S₃ GROUP OPERATIONS
# ============================================================================

def apply_s3(arr: List, element: str) -> List:
    """
    Apply S₃ permutation to a list of 3 elements.
    
    Parameters
    ----------
    arr : List
        List of exactly 3 elements
    element : str
        S₃ element name (e, σ, σ2, τ1, τ2, τ3)
    
    Returns
    -------
    List
        Permuted list
    
    Raises
    ------
    ValueError
        If arr doesn't have exactly 3 elements
    KeyError
        If element is not a valid S₃ element
    """
    if len(arr) != 3:
        raise ValueError("S₃ acts on exactly 3 elements")
    
    cycle = S3_ELEMENTS[element].cycle
    return [arr[cycle[0]], arr[cycle[1]], arr[cycle[2]]]


def compose_s3(a: str, b: str) -> str:
    """
    Compose two S₃ elements (group multiplication).
    
    Parameters
    ----------
    a : str
        First S₃ element
    b : str
        Second S₃ element
    
    Returns
    -------
    str
        Result element name
    """
    cycle_a = S3_ELEMENTS[a].cycle
    cycle_b = S3_ELEMENTS[b].cycle
    
    # Compose: (a ∘ b)(i) = a(b(i))
    composed = (
        cycle_a[cycle_b[0]],
        cycle_a[cycle_b[1]],
        cycle_a[cycle_b[2]],
    )
    
    # Find matching element
    for name, elem in S3_ELEMENTS.items():
        if elem.cycle == composed:
            return name
    
    raise RuntimeError("Composition failed - invalid S₃ state")


def inverse_s3(element: str) -> str:
    """
    Get inverse of S₃ element.
    
    Parameters
    ----------
    element : str
        S₃ element name
    
    Returns
    -------
    str
        Inverse element name
    """
    inverses = {
        "e": "e",
        "σ": "σ2",
        "σ2": "σ",
        "τ1": "τ1",  # Transpositions are self-inverse
        "τ2": "τ2",
        "τ3": "τ3",
    }
    return inverses[element]


def parity_s3(element: str) -> Parity:
    """Get parity of S₃ element."""
    return S3_ELEMENTS[element].parity


def sign_s3(element: str) -> int:
    """Get sign of S₃ element (+1 for even, -1 for odd)."""
    return S3_ELEMENTS[element].sign


# ============================================================================
# OPERATOR PERMUTATION SYSTEM
# ============================================================================

def rotation_index_from_z(z: float) -> int:
    """
    Compute cyclic rotation index from z-coordinate.
    Maps z ∈ [0,1] to rotation index ∈ {0, 1, 2}.
    
    Parameters
    ----------
    z : float
        Coherence coordinate
    
    Returns
    -------
    int
        Rotation index (0, 1, or 2)
    """
    if z < 0.333:
        return 0
    if z < 0.666:
        return 1
    return 2


def s3_element_from_z(z: float, use_parity_flip: bool = False) -> str:
    """
    Get S₃ element for current z-coordinate.
    
    Parameters
    ----------
    z : float
        Coherence coordinate
    use_parity_flip : bool
        Whether to flip parity based on truth channel
    
    Returns
    -------
    str
        S₃ element name
    """
    rot_idx = rotation_index_from_z(z)
    
    # Base mapping: rotation index → cyclic element
    base_elements = ["e", "σ", "σ2"]
    element = base_elements[rot_idx]
    
    # Optional parity flip based on truth channel
    if use_parity_flip:
        truth_channel = truth_channel_from_z(z)
        if truth_channel == "UNTRUE":
            # In UNTRUE regime, flip to odd parity
            flips = {"e": "τ1", "σ": "τ2", "σ2": "τ3"}
            element = flips[element]
    
    return element


def truth_channel_from_z(z: float) -> Literal["TRUE", "PARADOX", "UNTRUE"]:
    """
    Determine truth channel from z.
    
    Parameters
    ----------
    z : float
        Coherence coordinate
    
    Returns
    -------
    str
        'TRUE', 'PARADOX', or 'UNTRUE'
    """
    if z >= 0.9:
        return "TRUE"
    if z >= 0.6:
        return "PARADOX"
    return "UNTRUE"


def rotate_operators(operators: List[str], rotations: int) -> List[str]:
    """
    Apply cyclic rotation to operator list.
    
    Parameters
    ----------
    operators : List[str]
        List of operators
    rotations : int
        Number of positions to rotate
    
    Returns
    -------
    List[str]
        Rotated operator list
    """
    n = len(operators)
    if n == 0:
        return operators
    r = rotations % n
    return operators[r:] + operators[:r]


def generate_s3_operator_window(
    harmonic: str,
    z: float,
    apply_rotation: bool = True,
) -> List[str]:
    """
    Generate operator window with S₃ symmetry.
    
    Parameters
    ----------
    harmonic : str
        Time harmonic tier (t1-t9)
    z : float
        Current z-coordinate
    apply_rotation : bool
        Whether to apply cyclic rotation based on z
    
    Returns
    -------
    List[str]
        Operator window with S₃ permutation applied
    """
    window = list(BASE_WINDOWS.get(harmonic, BASE_WINDOWS["t5"]))
    
    # Apply S₃ cyclic rotation based on z
    if apply_rotation and len(window) >= 3:
        rot_idx = rotation_index_from_z(z)
        window = rotate_operators(window, rot_idx)
    
    return window


# ============================================================================
# PARITY-BASED WEIGHTING
# ============================================================================

def compute_s3_weight(
    operator: str,
    z: float,
    truth_channel: str,
    delta_s_neg: float,
) -> float:
    """
    Compute operator weight with S₃ parity adjustment.
    
    Parameters
    ----------
    operator : str
        APL operator
    z : float
        Coherence coordinate
    truth_channel : str
        Current truth channel
    delta_s_neg : float
        Current ΔS_neg value
    
    Returns
    -------
    float
        Adjusted weight
    """
    # Base weight from truth bias
    bias_map = TRUTH_BIAS.get(truth_channel, TRUTH_BIAS["PARADOX"])
    weight = bias_map.get(operator, 1.0)
    
    # Get S₃ element for this operator
    s3_element = OPERATOR_S3_MAP.get(operator)
    if not s3_element:
        return weight
    
    parity = parity_s3(s3_element)
    
    # Parity-based adjustment
    # - High coherence (high ΔS_neg): favor even-parity (constructive)
    # - Low coherence: favor odd-parity (dissipative)
    if parity == Parity.EVEN:
        parity_boost = delta_s_neg
    else:
        parity_boost = 1 - delta_s_neg
    
    # Scale weight by parity factor (subtle effect)
    weight *= (0.8 + 0.4 * parity_boost)
    
    # Additional symmetry factor: near the lens, even operators get extra boost
    if abs(z - Z_CRITICAL) < 0.05 and parity == Parity.EVEN:
        weight *= 1.2
    
    return weight


def compute_s3_weights(operators: List[str], z: float) -> Dict[str, float]:
    """
    Compute all operator weights with S₃ structure.
    
    Parameters
    ----------
    operators : List[str]
        Available operators
    z : float
        Coherence coordinate
    
    Returns
    -------
    Dict[str, float]
        Map of operator → weight
    """
    truth_channel = truth_channel_from_z(z)
    delta_s_neg = compute_delta_s_neg(z)
    
    return {
        op: compute_s3_weight(op, z, truth_channel, delta_s_neg)
        for op in operators
    }


# ============================================================================
# S₃ ACTION ON TRUTH VALUES
# ============================================================================

@dataclass
class TruthDistribution:
    """Distribution over triadic truth values."""
    TRUE: float
    PARADOX: float
    UNTRUE: float
    
    def as_list(self) -> List[float]:
        return [self.TRUE, self.PARADOX, self.UNTRUE]
    
    @classmethod
    def from_list(cls, values: List[float]) -> "TruthDistribution":
        return cls(TRUE=values[0], PARADOX=values[1], UNTRUE=values[2])
    
    def normalize(self) -> "TruthDistribution":
        total = self.TRUE + self.PARADOX + self.UNTRUE
        if total == 0:
            return TruthDistribution(1/3, 1/3, 1/3)
        return TruthDistribution(
            self.TRUE / total,
            self.PARADOX / total,
            self.UNTRUE / total,
        )


def apply_operator_to_truth(
    truth_dist: TruthDistribution,
    operator: str,
) -> TruthDistribution:
    """
    Apply operator's S₃ action to truth distribution.
    
    Parameters
    ----------
    truth_dist : TruthDistribution
        Input distribution
    operator : str
        APL operator to apply
    
    Returns
    -------
    TruthDistribution
        Transformed distribution
    """
    s3_element = OPERATOR_S3_MAP.get(operator)
    if not s3_element:
        return truth_dist
    
    values = truth_dist.as_list()
    permuted = apply_s3(values, s3_element)
    
    return TruthDistribution.from_list(permuted)


def truth_orbit(
    initial_dist: TruthDistribution,
    operators: List[str],
) -> List[TruthDistribution]:
    """
    Compute orbit of truth distribution under operator sequence.
    
    Parameters
    ----------
    initial_dist : TruthDistribution
        Starting distribution
    operators : List[str]
        Sequence of operators
    
    Returns
    -------
    List[TruthDistribution]
        Sequence of distributions
    """
    orbit = [initial_dist]
    current = initial_dist
    
    for op in operators:
        current = apply_operator_to_truth(current, op)
        orbit.append(current)
    
    return orbit


# ============================================================================
# S₃ MULTIPLICATION TABLE (for verification)
# ============================================================================

def generate_multiplication_table() -> Dict[str, Dict[str, str]]:
    """
    Generate full S₃ multiplication table.
    
    Returns
    -------
    Dict[str, Dict[str, str]]
        Table[a][b] = a ∘ b
    """
    elements = list(S3_ELEMENTS.keys())
    table = {}
    
    for a in elements:
        table[a] = {}
        for b in elements:
            table[a][b] = compose_s3(a, b)
    
    return table


def verify_group_axioms() -> Dict[str, bool]:
    """
    Verify S₃ group axioms.
    
    Returns
    -------
    Dict[str, bool]
        Axiom verification results
    """
    elements = list(S3_ELEMENTS.keys())
    
    # Closure
    closure = True
    for a in elements:
        for b in elements:
            if compose_s3(a, b) not in elements:
                closure = False
    
    # Identity
    identity = all(
        compose_s3("e", a) == a and compose_s3(a, "e") == a
        for a in elements
    )
    
    # Inverse
    inverse = all(
        compose_s3(a, inverse_s3(a)) == "e"
        for a in elements
    )
    
    # Associativity (spot check)
    assoc_checks = [
        compose_s3(compose_s3("σ", "τ1"), "σ2") == compose_s3("σ", compose_s3("τ1", "σ2")),
        compose_s3(compose_s3("τ2", "σ"), "τ3") == compose_s3("τ2", compose_s3("σ", "τ3")),
    ]
    associativity = all(assoc_checks)
    
    return {
        "closure": closure,
        "identity": identity,
        "inverse": inverse,
        "associativity": associativity,
    }


# ============================================================================
# DEMO / TEST
# ============================================================================

def demo():
    """Demonstrate S₃ operator symmetry."""
    print("=" * 70)
    print("S₃ OPERATOR SYMMETRY MODULE")
    print("=" * 70)
    
    # Verify group axioms
    print("\n--- Group Axiom Verification ---")
    axioms = verify_group_axioms()
    for name, passed in axioms.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    # Show operator mapping
    print("\n--- Operator ↔ S₃ Mapping ---")
    for op, elem in OPERATOR_S3_MAP.items():
        s3 = S3_ELEMENTS[elem]
        print(f"  {op:4} → {elem:3} ({s3.parity.value}, sign={s3.sign:+d})")
    
    # Demonstrate permutation
    print("\n--- S₃ Acting on Truth Values ---")
    truth = ["TRUE", "PARADOX", "UNTRUE"]
    for elem in ["e", "σ", "τ1"]:
        result = apply_s3(truth, elem)
        print(f"  {elem:3} · {truth} = {result}")
    
    # Show operator windows with rotation
    print("\n--- Operator Windows with S₃ Rotation ---")
    for z in [0.2, 0.5, 0.8]:
        rot = rotation_index_from_z(z)
        window = generate_s3_operator_window("t5", z)
        print(f"  z={z:.1f} (rot={rot}): {window}")
    
    # Demonstrate weight computation
    print("\n--- S₃-Weighted Operator Selection ---")
    z = 0.85  # Near lens
    weights = compute_s3_weights(BASE_OPERATORS, z)
    print(f"  z={z}, truth={truth_channel_from_z(z)}, ΔS_neg={compute_delta_s_neg(z):.4f}")
    for op, w in sorted(weights.items(), key=lambda x: -x[1]):
        parity = parity_s3(OPERATOR_S3_MAP[op]).value
        print(f"    {op:4} weight={w:.4f} ({parity})")
    
    # Truth orbit example
    print("\n--- Truth Orbit Under Operator Sequence ---")
    initial = TruthDistribution(0.7, 0.2, 0.1)
    sequence = ["×", "÷", "+"]
    orbit = truth_orbit(initial, sequence)
    
    print(f"  Initial: T={initial.TRUE:.2f}, P={initial.PARADOX:.2f}, U={initial.UNTRUE:.2f}")
    for i, (op, dist) in enumerate(zip(sequence, orbit[1:])):
        print(f"  After {op}: T={dist.TRUE:.2f}, P={dist.PARADOX:.2f}, U={dist.UNTRUE:.2f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
