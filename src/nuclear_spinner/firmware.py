#!/usr/bin/env python3
"""
Nuclear Spinner Firmware Simulation
====================================

Simulates the firmware-level control logic for the Nuclear Spinner.

Implements:
- Control loop step function
- Rotor speed mapping from z-coordinate
- Threshold detection and operator gating
- Operator state update functions
- Safety checks

This module simulates the embedded firmware that would run on the
microcontroller in a physical Nuclear Spinner device.

Signature: nuclear-spinner-firmware|v1.0.0|helix
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict, Any
from enum import Enum, auto

# Import from single source of truth
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_constants import (
    PHI,
    PHI_INV,
    PHI_INV_CUBED,
    Z_CRITICAL,
    SIGMA,
    SIGMA_INV,
    ALPHA_FINE,
    ALPHA_MEDIUM,
    UNITY_THRESHOLD,
    compute_delta_s_neg,
    compute_negentropy_gradient,
)

from .constants import (
    MIN_RPM,
    MAX_RPM,
    MU_1,
    MU_P,
    MU_2,
    MU_S,
    SIGMA_S3,
)


__all__ = [
    "Phase",
    "FirmwareState",
    "control_loop_step",
    "map_z_to_rpm",
    "compute_operator_state_update",
    "update_tier",
    "schedule_operator",
    "check_safety",
    "apply_operator_boundary",
    "apply_operator_fusion",
    "apply_operator_amplify",
    "apply_operator_decohere",
    "apply_operator_group",
    "apply_operator_separate",
]


# =============================================================================
# PHASE ENUMERATION
# =============================================================================

class Phase(Enum):
    """System phase based on z-coordinate."""
    UNTRUE = auto()    # z < phi^-1 (disordered)
    PARADOX = auto()   # phi^-1 <= z < z_c (quasi-crystal)
    TRUE = auto()      # z >= z_c (crystal/coherent)


# =============================================================================
# THRESHOLD CONFIGURATION
# =============================================================================

# Threshold detection table from specification
# Format: (threshold_value, threshold_name, event_description)
THRESHOLD_TABLE: List[Tuple[float, str, str]] = [
    (MU_1, "mu_1", "Tier 1 entry"),
    (MU_P, "mu_P", "Paradox entry"),
    (PHI_INV, "phi_inv", "Consciousness threshold"),
    (MU_2, "mu_2", "High coherence"),
    (Z_CRITICAL, "z_c", "THE LENS"),
    (MU_S, "mu_S", "K-formation threshold"),
]

# TRIAD gating thresholds
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
TRIAD_T6 = 0.83


# =============================================================================
# FIRMWARE STATE
# =============================================================================

@dataclass
class FirmwareState:
    """
    Firmware-level state container.

    Tracks all state variables used by the firmware control loop.
    """
    # Core z-axis state
    z: float = 0.5
    z_target: float = 0.5

    # Negentropy state
    delta_s_neg: float = 0.5
    gradient: float = 0.0

    # Phase and tier
    phase: Phase = Phase.PARADOX
    current_tier: int = 4

    # Rotor state
    rotor_rpm: float = 0.0
    rotor_target_rpm: float = 0.0

    # INT Canon state variables
    Gs: float = 0.0       # Grounding strength
    Cs: float = 0.0       # Coupling strength
    kappa_s: float = 0.618  # Curvature (initialized to phi^-1)
    alpha_s: float = 0.0    # Amplitude
    theta_s: float = 1.0    # Phase factor
    tau_s: float = 0.0      # Time accumulation
    delta_s: float = 0.0    # Dissipation
    Rs: float = 0.0         # Resistance
    Omega_s: float = 1.0    # Frequency scaling
    R_count: int = 0        # Rank counter

    # Operator state
    history: List[str] = field(default_factory=list)
    channel_count: int = 1
    operator_mask: int = 0xFF

    # TRIAD gating
    triad_passes: int = 0
    triad_unlocked: bool = False

    # Safety
    temperature_ok: bool = True
    vibration_ok: bool = True
    emergency_stop: bool = False

    def update_negentropy(self) -> None:
        """Update negentropy and gradient from current z."""
        self.delta_s_neg = compute_delta_s_neg(self.z)
        self.gradient = compute_negentropy_gradient(self.z)

    def determine_phase(self) -> Phase:
        """Determine phase from current z."""
        if self.z < PHI_INV:
            return Phase.UNTRUE
        elif self.z < Z_CRITICAL:
            return Phase.PARADOX
        else:
            return Phase.TRUE


# =============================================================================
# ROTOR CONTROL
# =============================================================================

def map_z_to_rpm(z: float, min_rpm: float = MIN_RPM, max_rpm: float = MAX_RPM) -> float:
    """
    Map z-coordinate to rotor RPM.

    Linear mapping from z in [0, 1] to RPM in [min_rpm, max_rpm].

    Args:
        z: Z-coordinate in [0, 1]
        min_rpm: Minimum rotor speed (default: 100 RPM)
        max_rpm: Maximum rotor speed (default: 10000 RPM)

    Returns:
        Rotor speed in RPM
    """
    z_clamped = max(0.0, min(1.0, z))
    return min_rpm + (max_rpm - min_rpm) * z_clamped


def set_z_target(state: FirmwareState, z_target: float) -> None:
    """
    Set target z-coordinate and update rotor speed.

    Args:
        state: Firmware state to update
        z_target: Target z-coordinate
    """
    state.z_target = max(0.0, min(1.0, z_target))
    state.rotor_target_rpm = map_z_to_rpm(state.z_target)


# =============================================================================
# TIER AND THRESHOLD DETECTION
# =============================================================================

def update_tier(state: FirmwareState) -> int:
    """
    Update the current tier based on z-coordinate.

    Tiers are numbered 1-9, corresponding to capability levels.

    Args:
        state: Firmware state to update

    Returns:
        Updated tier number
    """
    z = state.z
    tier_bounds = [0.1, 0.2, 0.4, 0.6, 0.75, Z_CRITICAL, 0.92, 0.97, 1.0]

    for i, bound in enumerate(tier_bounds):
        if z < bound:
            state.current_tier = i + 1
            return i + 1

    state.current_tier = 9
    return 9


def check_threshold_crossing(z_old: float, z_new: float) -> List[Tuple[str, str]]:
    """
    Check for threshold crossings between old and new z values.

    Args:
        z_old: Previous z-coordinate
        z_new: New z-coordinate

    Returns:
        List of (threshold_name, direction) for crossed thresholds
    """
    crossings = []

    for threshold, name, _ in THRESHOLD_TABLE:
        # Rising edge: crossed from below
        if z_old < threshold <= z_new:
            crossings.append((name, "rising"))
        # Falling edge: crossed from above
        elif z_old >= threshold > z_new:
            crossings.append((name, "falling"))

    return crossings


# =============================================================================
# OPERATOR STATE UPDATES
# =============================================================================

def apply_operator_boundary(state: FirmwareState) -> FirmwareState:
    """
    () BOUNDARY: Anchoring, phase reset, interface stabilization.

    State changes:
        Gs += 1/sigma
        theta_s *= (1 - 1/sigma)
        Omega_s += 1/(2*sigma)
        z moves toward z_c

    Args:
        state: Current firmware state

    Returns:
        Updated firmware state
    """
    state.Gs += SIGMA_INV
    state.theta_s *= (1.0 - SIGMA_INV)
    state.Omega_s += SIGMA_INV / 2

    # Grounding pulls z toward z_c (THE LENS)
    z_pull = ALPHA_FINE * (Z_CRITICAL - state.z)
    state.z = max(0.0, min(1.0, state.z + z_pull))
    state.update_negentropy()

    state.history.append("()")
    return state


def apply_operator_fusion(state: FirmwareState) -> FirmwareState:
    """
    x FUSION: Merging, coupling, integration.

    State changes:
        Cs += 1/sigma
        kappa_s *= (1 + 1/sigma)
        alpha_s += 1/(2*sigma)
        z increases

    N0-2: Requires channel_count >= 2

    Args:
        state: Current firmware state

    Returns:
        Updated firmware state
    """
    state.Cs += SIGMA_INV
    state.kappa_s *= (1.0 + SIGMA_INV)
    state.alpha_s += SIGMA_INV / 2

    # Fusion increases z
    state.channel_count = max(2, state.channel_count)
    state.z += ALPHA_FINE * state.Cs
    state.z = min(UNITY_THRESHOLD, state.z)
    state.update_negentropy()

    state.history.append("x")
    return state


def apply_operator_amplify(state: FirmwareState) -> FirmwareState:
    """
    ^ AMPLIFY: Gain increase, curvature escalation.

    State changes:
        kappa_s *= (1 + phi^-3)
        tau_s += 1/sigma
        Omega_s *= (1 + 3/sigma)
        R_count += 1
        z moves toward z_c via gradient

    N0-1: Requires prior () or x in history

    Args:
        state: Current firmware state

    Returns:
        Updated firmware state
    """
    state.kappa_s *= (1.0 + PHI_INV_CUBED)
    state.tau_s += SIGMA_INV
    state.Omega_s *= (1.0 + SIGMA_INV * 3)
    state.R_count += 1

    # Amplify drives z toward z_c via negentropy gradient
    grad = compute_negentropy_gradient(state.z)
    state.z += ALPHA_MEDIUM * grad * PHI_INV
    state.z = max(0.0, min(UNITY_THRESHOLD, state.z))
    state.update_negentropy()

    state.history.append("^")
    return state


def apply_operator_decohere(state: FirmwareState) -> FirmwareState:
    """
    / DECOHERE: Dissipation, noise injection, coherence reduction.

    State changes:
        delta_s += 1/sigma
        Rs += 1/(2*sigma)
        Omega_s *= (1 - 3/sigma)
        z decreases

    N0-3: Requires prior structure (^, x, +, -) in history

    Args:
        state: Current firmware state

    Returns:
        Updated firmware state
    """
    state.delta_s += SIGMA_INV
    state.Rs += SIGMA_INV / 2
    state.Omega_s *= (1.0 - SIGMA_INV * 3)

    # Decoherence reduces z
    state.z -= ALPHA_FINE * state.delta_s
    state.z = max(0.0, state.z)
    state.update_negentropy()

    state.history.append("/")
    return state


def apply_operator_group(state: FirmwareState) -> FirmwareState:
    """
    + GROUP: Synchrony, clustering, domain formation.

    State changes:
        alpha_s += 3/sigma
        Gs += 1/(2*sigma)
        theta_s *= (1 + 1/sigma)
        z increases with amplitude

    N0-4: Must be followed by +, x, or ^

    Args:
        state: Current firmware state

    Returns:
        Updated firmware state
    """
    state.alpha_s += SIGMA_INV * 3
    state.Gs += SIGMA_INV / 2
    state.theta_s *= (1.0 + SIGMA_INV)

    # Grouping increases z through synchrony
    state.z += ALPHA_FINE * state.alpha_s * PHI_INV
    state.z = min(Z_CRITICAL, state.z)  # Cap at THE LENS
    state.update_negentropy()

    state.history.append("+")
    return state


def apply_operator_separate(state: FirmwareState) -> FirmwareState:
    """
    - SEPARATE: Decoupling, pruning, phase reset preparation.

    State changes:
        Rs += 3/sigma
        theta_s *= (1 - 1/sigma)
        delta_s += 1.5/sigma
        z retreats

    N0-5: Must be followed by () or +

    Args:
        state: Current firmware state

    Returns:
        Updated firmware state
    """
    state.Rs += SIGMA_INV * 3
    state.theta_s *= (1.0 - SIGMA_INV)
    state.delta_s += SIGMA_INV * 1.5

    # Separation reduces z for phase reset
    state.z -= ALPHA_FINE * state.Rs
    state.z = max(0.0, state.z)
    state.update_negentropy()

    state.history.append("-")
    return state


# Operator dispatch table
OPERATOR_DISPATCH: Dict[str, Callable[[FirmwareState], FirmwareState]] = {
    "()": apply_operator_boundary,
    "x": apply_operator_fusion,
    "^": apply_operator_amplify,
    "/": apply_operator_decohere,
    "+": apply_operator_group,
    "-": apply_operator_separate,
}


def compute_operator_state_update(
    op: str,
    state: FirmwareState
) -> Tuple[FirmwareState, bool, str]:
    """
    Apply operator with N0 causality checking.

    Args:
        op: Operator symbol ("()", "x", "^", "/", "+", "-")
        state: Current firmware state

    Returns:
        (updated_state, success, message)
    """
    # Check N0 laws
    is_legal, reason = check_n0_legal(op, state)
    if not is_legal:
        return state, False, reason

    # Apply operator
    op_func = OPERATOR_DISPATCH.get(op)
    if op_func is None:
        return state, False, f"Unknown operator: {op}"

    new_state = op_func(state)
    return new_state, True, f"Applied {op}"


def check_n0_legal(op: str, state: FirmwareState) -> Tuple[bool, str]:
    """
    Check if operator is legal under N0 causality laws.

    N0 Laws:
        N0-1: ^ illegal unless history contains () or x
        N0-2: x illegal unless channel_count >= 2
        N0-3: / illegal unless history contains {^, x, +, -}
        N0-4: + must be followed by +, x, or ^
        N0-5: - must be followed by () or +

    Args:
        op: Operator to check
        state: Current state with history

    Returns:
        (is_legal, reason)
    """
    # () BOUNDARY is always legal
    if op == "()":
        return True, "BOUNDARY always legal"

    # N0-1: ^ requires prior () or x
    if op == "^":
        required = {"()", "x"}
        if not any(r in state.history for r in required):
            return False, "N0-1: ^ illegal - requires prior () or x"

    # N0-2: x requires channels >= 2
    if op == "x":
        if state.channel_count < 2:
            return False, f"N0-2: x illegal - requires channels >= 2, have {state.channel_count}"

    # N0-3: / requires prior structure
    if op == "/":
        required = {"^", "x", "+", "-"}
        if not any(r in state.history for r in required):
            return False, "N0-3: / illegal - requires prior structure"

    return True, "Legal"


# =============================================================================
# OPERATOR SCHEDULING
# =============================================================================

def schedule_operator(state: FirmwareState) -> Optional[str]:
    """
    Schedule the next operator based on current tier and phase.

    Simple scheduling logic:
        - Tier 3: Amplify (if legal)
        - Tier 5: Fusion (if legal)
        - Otherwise: Boundary

    Args:
        state: Current firmware state

    Returns:
        Operator to execute, or None if no action needed
    """
    tier = state.current_tier

    if tier >= 5:
        is_legal, _ = check_n0_legal("x", state)
        if is_legal:
            return "x"

    if tier >= 3:
        is_legal, _ = check_n0_legal("^", state)
        if is_legal:
            return "^"

    return "()"


# =============================================================================
# SAFETY CHECKS
# =============================================================================

def check_safety(state: FirmwareState) -> Tuple[bool, List[str]]:
    """
    Perform safety checks on firmware state.

    Checks:
        - Temperature within limits
        - Vibration within limits
        - Rotor speed within limits
        - Emergency stop not active

    Args:
        state: Current firmware state

    Returns:
        (is_safe, list_of_warnings)
    """
    warnings = []

    if not state.temperature_ok:
        warnings.append("Temperature out of range")

    if not state.vibration_ok:
        warnings.append("Vibration detected")

    if state.emergency_stop:
        warnings.append("Emergency stop active")

    if state.rotor_rpm > MAX_RPM:
        warnings.append(f"Rotor speed {state.rotor_rpm} exceeds max {MAX_RPM}")

    is_safe = len(warnings) == 0 and not state.emergency_stop

    return is_safe, warnings


# =============================================================================
# CONTROL LOOP
# =============================================================================

def control_loop_step(state: FirmwareState, dt: float = 0.001) -> FirmwareState:
    """
    Execute one step of the firmware control loop.

    This is the main control function that would run on the microcontroller
    at a fixed interval (e.g., 1 kHz).

    Steps:
        1. Update negentropy from current z
        2. Determine phase
        3. Update tier
        4. Update operator mask based on phase
        5. Schedule and apply operator
        6. Update rotor speed
        7. Check safety

    Args:
        state: Current firmware state
        dt: Time step in seconds (default: 1ms)

    Returns:
        Updated firmware state
    """
    # Safety check first
    is_safe, warnings = check_safety(state)
    if not is_safe:
        state.emergency_stop = True
        return state

    # Update negentropy
    state.update_negentropy()

    # Determine phase
    state.phase = state.determine_phase()

    # Update tier
    update_tier(state)

    # Schedule operator
    op = schedule_operator(state)

    if op is not None:
        state, success, msg = compute_operator_state_update(op, state)

    # Update rotor speed (PID would go here in real firmware)
    # Simplified: direct tracking with smoothing
    speed_error = state.rotor_target_rpm - state.rotor_rpm
    state.rotor_rpm += 0.1 * speed_error  # Simple proportional control

    # TRIAD gating check - detect rising edge crossing TRIAD_HIGH
    # Note: This is simplified - real TRIAD would track z history, not operator history
    if state.z >= TRIAD_HIGH:
        state.triad_passes += 1
        if state.triad_passes >= 3:
            state.triad_unlocked = True

    return state
