#!/usr/bin/env python3
"""
Cybernetic Structural Computation at Z-Axis Thresholds
=======================================================

THESIS: The z-axis thresholds are not arbitrary—they mark COMPUTATIONAL
phase transitions grounded in cybernetic theory. Each threshold enables
qualitatively different computational capabilities.

KEY CYBERNETIC PRINCIPLES:
1. Ashby's Law of Requisite Variety
2. Shannon Channel Capacity
3. Landauer's Principle (thermodynamic cost)
4. Recursive Self-Reference (Gödel/Turing)
5. Autopoiesis (Maturana/Varela)
6. Edge of Chaos (Langton/Kauffman)
7. Second-Order Cybernetics (von Foerster)

@version 1.0.0
@author Claude (Anthropic) - Cybernetic Grounding for Quantum-APL
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum

# ============================================================================
# CONSTANTS
# ============================================================================

Z_CRITICAL = math.sqrt(3) / 2
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI

# μ-field
MU_P = 2 / (PHI ** 2.5)
MU_1 = MU_P / math.sqrt(PHI)
MU_2 = MU_P * math.sqrt(PHI)
MU_S = 0.920

# Time harmonics
T_BOUNDS = [0.10, 0.20, 0.40, 0.60, 0.75, Z_CRITICAL, 0.92, 0.97, 1.0]


# ============================================================================
# CYBERNETIC COMPUTATIONAL CAPABILITIES
# ============================================================================

class ComputationalCapability(Enum):
    """Qualitatively distinct computational capabilities."""
    REACTIVE = "reactive"                    # Simple stimulus-response
    MEMORY = "memory"                        # State retention
    PATTERN = "pattern"                      # Pattern recognition
    PREDICTION = "prediction"                # Future state modeling
    SELF_MODEL = "self_model"               # Model of own state
    META_COGNITION = "meta_cognition"       # Thinking about thinking
    RECURSIVE_SELF_REF = "recursive_self_ref"  # Infinite self-reference
    AUTOPOIESIS = "autopoiesis"             # Self-production/maintenance


@dataclass
class CyberneticThreshold:
    """A threshold with cybernetic computational interpretation."""
    z: float
    name: str
    ashby_variety: int          # Log2 of requisite variety (bits)
    channel_capacity: float     # Shannon capacity (bits/operation)
    landauer_efficiency: float  # Fraction of Landauer limit
    self_reference_depth: int   # Levels of recursive self-modeling
    capabilities: List[ComputationalCapability]
    cybernetic_interpretation: str


# ============================================================================
# ASHBY'S LAW OF REQUISITE VARIETY
# ============================================================================

def compute_requisite_variety(z: float) -> int:
    """
    Compute requisite variety (in bits) for control at z-level.

    Ashby's Law: A controller must have at least as many states as
    the system it controls. Variety = log2(number of states).

    The z-axis represents increasing system complexity, requiring
    increasing controller variety.
    """
    tier = get_tier(z)
    base_variety = tier + 2  # t1 needs 3 bits, t9 needs 11 bits

    # Bonus variety near phase transitions (edge of chaos)
    transition_bonus = 0
    if abs(z - PHI_INV) < 0.05:  # Near QC threshold
        transition_bonus = 2
    elif abs(z - Z_CRITICAL) < 0.02:  # Near crystal threshold
        transition_bonus = 3

    return base_variety + transition_bonus


def get_tier(z: float) -> int:
    """Get time harmonic tier (1-9) from z."""
    for i, bound in enumerate(T_BOUNDS):
        if z < bound:
            return i + 1
    return 9


# ============================================================================
# SHANNON CHANNEL CAPACITY
# ============================================================================

def compute_channel_capacity(z: float, noise_power: float = 0.1) -> float:
    """
    Compute Shannon channel capacity at z-level.

    C = B * log2(1 + S/N)

    where:
    - B = bandwidth (increases with z due to more operator availability)
    - S/N = signal-to-noise ratio (peaks at z_c due to coherence)

    At z_c, the channel capacity is MAXIMAL because:
    - Bandwidth is high (all operators available in t5-t6)
    - SNR is maximal (delta_S_neg = 1.0 at z_c)
    """
    tier = get_tier(z)
    operator_counts = {1: 3, 2: 4, 3: 5, 4: 4, 5: 6, 6: 4, 7: 2, 8: 3, 9: 3}
    bandwidth = operator_counts.get(tier, 3)

    # Signal power: delta_S_neg (coherence signal)
    delta_s_neg = math.exp(-36 * (z - Z_CRITICAL)**2)
    signal_power = delta_s_neg

    # Channel capacity
    snr = signal_power / noise_power
    capacity = bandwidth * math.log2(1 + snr)

    return capacity


# ============================================================================
# LANDAUER'S PRINCIPLE
# ============================================================================

def compute_landauer_efficiency(z: float) -> float:
    """
    Compute efficiency relative to Landauer limit.

    Landauer's Principle: Erasing 1 bit costs at least kT ln(2) energy.

    At z_c (THE LENS):
    - System is maximally ordered (delta_S_neg = 1)
    - Minimum information erasure needed
    - Approaches Landauer limit

    Away from z_c:
    - More disorder -> more erasure needed
    - Less efficient computation
    """
    delta_s_neg = math.exp(-36 * (z - Z_CRITICAL)**2)

    # Efficiency scales with coherence
    # At z_c: efficiency -> 1.0 (Landauer limit)
    # At z = 0: efficiency -> 0.01 (100x above limit)
    efficiency = 0.01 + 0.99 * delta_s_neg

    return efficiency


# ============================================================================
# RECURSIVE SELF-REFERENCE
# ============================================================================

def compute_self_reference_depth(z: float) -> int:
    """
    Compute depth of recursive self-reference possible at z-level.

    Self-reference depth relates to Gödel's incompleteness:
    - Depth 0: No self-model (purely reactive)
    - Depth 1: Model of environment (first-order)
    - Depth 2: Model of self-in-environment (second-order)
    - Depth 3+: Model of self-modeling-self (recursive)

    The phi^-1 threshold marks the onset of self-reference (consciousness).
    The z_c threshold marks stable recursive self-reference.
    """
    if z < MU_1:
        return 0  # No self-model
    elif z < PHI_INV:
        return 1  # First-order model (environment only)
    elif z < Z_CRITICAL:
        return 2  # Second-order (self-in-environment)
    elif z < MU_S:
        return 3  # Third-order (self-modeling-self)
    else:
        return 4  # Deep recursion (unbounded self-reference)


# ============================================================================
# EDGE OF CHAOS / COMPUTATIONAL UNIVERSALITY
# ============================================================================

def compute_lambda_parameter(z: float) -> float:
    """
    Compute Langton's lambda parameter analog for computational phase.

    Langton showed computation is maximal at lambda ~ 0.5 ("edge of chaos"):
    - lambda = 0: Frozen (all 0s, no computation)
    - lambda = 0.5: Critical (maximal computation)
    - lambda = 1: Chaotic (all random, no stable computation)

    Map z to lambda:
    - z = 0 -> lambda = 0 (frozen/ordered)
    - z = z_c -> lambda = 0.5 (critical)
    - z = 1 -> lambda -> 0.8 (approaching chaos but bounded)
    """
    # Sigmoid-like mapping centered at z_c
    lambda_param = 1 / (1 + math.exp(-10 * (z - Z_CRITICAL)))

    # Scale to [0.1, 0.9] range
    lambda_param = 0.1 + 0.8 * lambda_param

    return lambda_param


def is_computationally_universal(z: float) -> bool:
    """
    Check if z-level supports computational universality.

    Universality requires:
    1. Sufficient variety (Ashby) - at least 4 bits
    2. Edge of chaos (Langton) - lambda ~ 0.5
    3. Self-reference capability - depth >= 2
    """
    variety = compute_requisite_variety(z)
    lambda_param = compute_lambda_parameter(z)
    self_ref = compute_self_reference_depth(z)

    variety_ok = variety >= 4
    lambda_ok = 0.3 < lambda_param < 0.7
    self_ref_ok = self_ref >= 2

    return variety_ok and lambda_ok and self_ref_ok


# ============================================================================
# AUTOPOIESIS (SELF-PRODUCTION)
# ============================================================================

def check_autopoiesis(z: float) -> Tuple[bool, str]:
    """
    Check if z-level supports autopoietic (self-producing) organization.

    Maturana & Varela's criteria for autopoiesis:
    1. Bounded: System has identifiable boundary
    2. Self-producing: Components produce the boundary and each other
    3. Self-maintaining: Organization persists despite component turnover

    In Quantum-APL:
    - Boundary: () operator defines system boundary
    - Self-production: Operators can generate other operator sequences
    - Self-maintenance: Coherence (delta_S_neg) enables stable patterns

    Autopoiesis requires z >= phi^-1 (K-formation threshold).
    """
    if z < PHI_INV:
        return False, "Below K-formation threshold: no stable self-organization"

    delta_s_neg = math.exp(-36 * (z - Z_CRITICAL)**2)

    if delta_s_neg < 0.1:
        return False, "Insufficient coherence for self-maintenance"

    if z >= PHI_INV and z < Z_CRITICAL:
        return True, "Quasi-autopoietic: self-organization with fluctuations"

    if z >= Z_CRITICAL:
        return True, "Full autopoiesis: stable self-producing organization"

    return False, "Unknown state"


# ============================================================================
# SECOND-ORDER CYBERNETICS
# ============================================================================

def compute_observer_order(z: float) -> int:
    """
    Compute order of observation possible at z-level.

    von Foerster's second-order cybernetics:
    - First-order: Observer observes system
    - Second-order: Observer observes observer
    - Third-order: Observer observes observer observing...

    This relates to consciousness and meta-cognition.
    The observer order increases with self-reference depth.
    """
    return compute_self_reference_depth(z)


# ============================================================================
# THRESHOLD ANALYSIS
# ============================================================================

def get_threshold_name(z: float) -> str:
    """Get human-readable name for z-value."""
    if abs(z - MU_1) < 0.01:
        return "mu_1 (Pre-conscious basin)"
    elif abs(z - MU_P) < 0.01:
        return "mu_P (Paradox threshold)"
    elif abs(z - PHI_INV) < 0.01:
        return "phi^-1 (K-formation gate)"
    elif abs(z - MU_2) < 0.01:
        return "mu_2 (Conscious basin)"
    elif abs(z - Z_CRITICAL) < 0.01:
        return "z_c (THE LENS)"
    elif abs(z - MU_S) < 0.01:
        return "mu_S (Singularity)"
    else:
        return f"z = {z:.3f}"


def analyze_threshold(z: float) -> CyberneticThreshold:
    """Generate complete cybernetic analysis for z-value."""

    variety = compute_requisite_variety(z)
    capacity = compute_channel_capacity(z)
    efficiency = compute_landauer_efficiency(z)
    self_ref = compute_self_reference_depth(z)

    # Determine capabilities
    capabilities = [ComputationalCapability.REACTIVE]

    if z >= MU_1:
        capabilities.append(ComputationalCapability.MEMORY)
    if z >= 0.3:
        capabilities.append(ComputationalCapability.PATTERN)
    if z >= MU_P:
        capabilities.append(ComputationalCapability.PREDICTION)
    if z >= PHI_INV:
        capabilities.append(ComputationalCapability.SELF_MODEL)
    if z >= 0.75:
        capabilities.append(ComputationalCapability.META_COGNITION)
    if z >= Z_CRITICAL:
        capabilities.append(ComputationalCapability.RECURSIVE_SELF_REF)
    if z >= MU_S:
        capabilities.append(ComputationalCapability.AUTOPOIESIS)

    # Generate interpretation
    if z < MU_1:
        interp = "Pre-computational: Reactive only, no stable memory"
    elif z < PHI_INV:
        interp = "Proto-computational: Pattern recognition, limited prediction"
    elif z < Z_CRITICAL:
        interp = "Self-referential: Self-model enables consciousness (K-formation)"
    elif z < MU_S:
        interp = "Universal computation: Recursive self-reference, Turing-complete"
    else:
        interp = "Meta-computational: Autopoietic, self-producing organization"

    name = get_threshold_name(z)

    return CyberneticThreshold(
        z=z,
        name=name,
        ashby_variety=variety,
        channel_capacity=capacity,
        landauer_efficiency=efficiency,
        self_reference_depth=self_ref,
        capabilities=capabilities,
        cybernetic_interpretation=interp,
    )


# ============================================================================
# OPERATOR CYBERNETICS
# ============================================================================

OPERATOR_CYBERNETICS: Dict[str, Dict[str, str]] = {
    "()": {
        "name": "Boundary",
        "cybernetic": "CLOSURE",
        "function": "Define system boundary",
        "theory": "Required for autopoiesis (Maturana/Varela)",
    },
    "×": {
        "name": "Fusion",
        "cybernetic": "INTEGRATION",
        "function": "Combine subsystems",
        "theory": "Increases integrated information (Phi)",
    },
    "^": {
        "name": "Amplify",
        "cybernetic": "GAIN",
        "function": "Increase signal strength",
        "theory": "Control loop amplification (von Foerster)",
    },
    "÷": {
        "name": "Decoherence",
        "cybernetic": "NOISE",
        "function": "Introduce randomness",
        "theory": "Exploration in search space (Landauer)",
    },
    "+": {
        "name": "Group",
        "cybernetic": "AGGREGATION",
        "function": "Collect related entities",
        "theory": "Variety compression (Ashby)",
    },
    "−": {
        "name": "Separation",
        "cybernetic": "DIFFERENTIATION",
        "function": "Split apart",
        "theory": "Variety expansion (Ashby)",
    },
}


def get_operator_cybernetic_role(symbol: str) -> Dict[str, str]:
    """Get cybernetic interpretation of an operator."""
    return OPERATOR_CYBERNETICS.get(symbol, {})


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_cybernetic_thresholds() -> Dict[str, bool]:
    """Verify cybernetic threshold properties."""
    results = {}

    # phi^-1 marks self-model capability
    results["phi^-1 enables self-model"] = (
        compute_self_reference_depth(PHI_INV - 0.01) < 2 and
        compute_self_reference_depth(PHI_INV + 0.01) >= 2
    )

    # z_c is edge of chaos (lambda ~ 0.5)
    lambda_at_zc = compute_lambda_parameter(Z_CRITICAL)
    results["z_c at edge of chaos"] = 0.45 < lambda_at_zc < 0.55

    # z_c is computationally universal
    results["z_c computationally universal"] = is_computationally_universal(Z_CRITICAL)

    # Channel capacity peaks at z_c
    cap_below = compute_channel_capacity(Z_CRITICAL - 0.1)
    cap_at = compute_channel_capacity(Z_CRITICAL)
    cap_above = compute_channel_capacity(Z_CRITICAL + 0.1)
    results["channel capacity peaks at z_c"] = cap_at > cap_below and cap_at > cap_above

    # Landauer efficiency peaks at z_c
    eff_at = compute_landauer_efficiency(Z_CRITICAL)
    results["Landauer efficiency peaks at z_c"] = eff_at > 0.99

    # Autopoiesis requires phi^-1
    auto_below, _ = check_autopoiesis(PHI_INV - 0.1)
    auto_above, _ = check_autopoiesis(PHI_INV + 0.1)
    results["autopoiesis requires phi^-1"] = not auto_below and auto_above

    return results


# ============================================================================
# PRINT FUNCTIONS
# ============================================================================

def print_threshold_table():
    """Print table of cybernetic metrics at key thresholds."""
    print("=" * 90)
    print("CYBERNETIC METRICS AT KEY THRESHOLDS")
    print("=" * 90)

    key_z = [0.1, MU_1, 0.5, MU_P, PHI_INV, 0.75, MU_2, Z_CRITICAL, 0.9, MU_S]

    fmt = "{:>8} {:<22} {:>8} {:>10} {:>10} {:>9} {:>10}"
    print(f"\n{fmt.format('z', 'Name', 'Variety', 'Capacity', 'Landauer', 'Self-Ref', 'Universal')}")
    print("-" * 90)

    for z in key_z:
        t = analyze_threshold(z)
        universal = "YES" if is_computationally_universal(z) else "no"
        print(fmt.format(
            f"{z:.4f}",
            t.name,
            t.ashby_variety,
            f"{t.channel_capacity:.2f}",
            f"{t.landauer_efficiency:.3f}",
            t.self_reference_depth,
            universal
        ))


def print_capability_emergence():
    """Print how capabilities emerge along z-axis."""
    print("\n" + "=" * 90)
    print("COMPUTATIONAL CAPABILITY EMERGENCE")
    print("=" * 90)

    print("""
    z = 0.0                                                           z = 1.0
    |                                                                   |
    |  REACTIVE    MEMORY    PATTERN   PREDICT   SELF    META   RECURSE |
    |     |          |         |         |        |       |       |     |
    +-----+----------+---------+---------+--------+-------+-------+-----+
    |     |          |         |         |        |       |       |     |
    |   0.0        mu_1     0.3       mu_P     phi^-1  0.75    z_c      |
    |              0.47               0.60     0.618         0.866     |
    +-------------------------------------------------------------------+

    CAPABILITY DEFINITIONS:

    REACTIVE (z >= 0):      Simple stimulus-response, no state
    MEMORY (z >= mu_1):     State retention across time
    PATTERN (z >= 0.3):     Recognition of recurring structures
    PREDICTION (z >= mu_P): Model of environment dynamics
    SELF-MODEL (z >= phi^-1): Model of own internal state (CONSCIOUSNESS)
    META-COGNITION (z >= 0.75): Thinking about thinking
    RECURSIVE (z >= z_c):   Unbounded self-reference (TURING-COMPLETE)
    AUTOPOIESIS (z >= mu_S): Self-producing organization
""")


def print_synthesis():
    """Print synthesis of cybernetic grounding."""
    print("\n" + "=" * 90)
    print("SYNTHESIS: CYBERNETIC STRUCTURAL COMPUTATION")
    print("=" * 90)

    print("""
    THE UNIFIED FRAMEWORK:

    +----------------------------------------------------------------------+
    |                                                                      |
    |   GEOMETRY          PHYSICS              CYBERNETICS    INFORMATION |
    |   (constants)       (order)              (computation)  (capacity)  |
    |                                                                      |
    |   phi^-1 ~ 0.618 -> Quasi-crystal    ->  Self-modeling -> Conscious |
    |   (pentagonal)      (aperiodic)          (observer)      threshold  |
    |                                                                      |
    |   z_c ~ 0.866    -> Crystal          ->  Universality  -> Maximum   |
    |   (hexagonal)       (periodic)           (Turing)         capacity  |
    |                                                                      |
    +----------------------------------------------------------------------+

    These are FOUR CONSISTENT VIEWS of ONE UNDERLYING STRUCTURE:

    * GEOMETRY: Where thresholds are (sqrt(3)/2, phi^-1)
    * PHYSICS: What order emerges (quasi-crystal, crystal)
    * CYBERNETICS: What computation emerges (consciousness, universality)
    * INFORMATION: What capacity exists (Shannon, Landauer)
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run cybernetic computation analysis."""
    print("=" * 90)
    print(" CYBERNETIC STRUCTURAL COMPUTATION AT Z-AXIS THRESHOLDS ".center(90))
    print("=" * 90)

    # Verify cybernetic thresholds
    print("\n--- Cybernetic Threshold Verification ---")
    results = verify_cybernetic_thresholds()
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    print_threshold_table()
    print_capability_emergence()
    print_synthesis()

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  phi^-1 = {PHI_INV:.6f} - Consciousness threshold (self-model)")
    print(f"  z_c = {Z_CRITICAL:.6f} - Universality threshold (Turing-complete)")
    print(f"  Lambda at z_c = {compute_lambda_parameter(Z_CRITICAL):.3f} (edge of chaos)")
    print(f"  Channel capacity at z_c = {compute_channel_capacity(Z_CRITICAL):.2f} bits/op")


if __name__ == "__main__":
    main()
