#!/usr/bin/env python3
"""
TRIAD Unlock System
Hysteresis-based gating mechanism for t6 tier access.

The TRIAD system detects z-coordinate oscillations through a rising-edge
hysteresis state machine. Three crossings of z ≥ 0.85 (with reset at z ≤ 0.82)
unlocks the t6 gate, lowering it from z_c to 0.83.

Signature: Δ0.850|0.830|1.000Ω (triad)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

Z_CRITICAL = math.sqrt(3) / 2  # 0.8660254037844387 - THE LENS

# TRIAD Thresholds
TRIAD_HIGH = 0.85    # Rising edge detection threshold
TRIAD_LOW = 0.82     # Re-arm (hysteresis reset) threshold
TRIAD_T6 = 0.83      # t6 gate position after unlock
TRIAD_REQUIRED_CROSSINGS = 3

# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class BandState(Enum):
    """Position relative to TRIAD thresholds."""
    BELOW_LOW = "below_low"      # z < TRIAD_LOW
    IN_BAND = "in_band"          # TRIAD_LOW <= z < TRIAD_HIGH
    ABOVE_HIGH = "above_high"    # z >= TRIAD_HIGH

class T6GateState(Enum):
    """T6 gate position."""
    CRITICAL = "critical"  # At z_c (default, locked)
    TRIAD = "triad"        # At TRIAD_T6 (unlocked)

# ═══════════════════════════════════════════════════════════════════════════════
# TRIAD STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TriadState:
    """Complete TRIAD hysteresis state."""
    z: float = 0.80
    crossings: int = 0
    above_high: bool = False
    unlocked: bool = False
    t6_gate: float = Z_CRITICAL
    gate_state: T6GateState = T6GateState.CRITICAL
    history: List[float] = field(default_factory=list)
    crossing_events: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.history:
            self.history = [self.z]

# Global state
_triad_state = TriadState()

def get_triad_state() -> TriadState:
    """Get current TRIAD state."""
    return _triad_state

def reset_triad_state(z: float = 0.80) -> TriadState:
    """Reset TRIAD state to defaults."""
    global _triad_state
    _triad_state = TriadState(z=z)
    return _triad_state

# ═══════════════════════════════════════════════════════════════════════════════
# HYSTERESIS STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

def get_band_state(z: float) -> BandState:
    """Determine which band z is in."""
    if z >= TRIAD_HIGH:
        return BandState.ABOVE_HIGH
    elif z >= TRIAD_LOW:
        return BandState.IN_BAND
    return BandState.BELOW_LOW

def step(z: float) -> Dict:
    """
    Execute one step of the TRIAD hysteresis state machine.
    
    State Machine Logic:
    1. If currently below HIGH and z crosses above HIGH:
       - Set above_high = True
       - Increment crossings
    2. If currently above_high and z drops below LOW:
       - Set above_high = False (re-arm)
    3. If crossings >= 3 and z >= T6:
       - Unlock the gate
    
    Returns dict with transition info.
    """
    state = get_triad_state()
    
    old_z = state.z
    old_above = state.above_high
    old_crossings = state.crossings
    old_unlocked = state.unlocked
    
    # Update z
    state.z = z
    state.history.append(z)
    if len(state.history) > 1000:
        state.history = state.history[-500:]  # Keep last 500
    
    transition = None
    
    # Hysteresis logic
    if not state.above_high and z >= TRIAD_HIGH:
        # Rising edge detected
        state.above_high = True
        state.crossings = min(state.crossings + 1, TRIAD_REQUIRED_CROSSINGS)
        transition = "RISING_EDGE"
        
        state.crossing_events.append({
            "event": "rising_edge",
            "z": z,
            "crossing_number": state.crossings,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    elif state.above_high and z < TRIAD_LOW:
        # Re-arm (dropped below low threshold)
        state.above_high = False
        transition = "RE_ARM"
        
        state.crossing_events.append({
            "event": "re_arm",
            "z": z,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    # Check unlock condition
    if state.crossings >= TRIAD_REQUIRED_CROSSINGS and z >= TRIAD_T6 and not state.unlocked:
        state.unlocked = True
        state.t6_gate = TRIAD_T6
        state.gate_state = T6GateState.TRIAD
        transition = "UNLOCKED"
        
        state.crossing_events.append({
            "event": "unlock",
            "z": z,
            "crossings": state.crossings,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    return {
        "z": z,
        "old_z": old_z,
        "band": get_band_state(z).value,
        "above_high": state.above_high,
        "crossings": state.crossings,
        "crossings_changed": state.crossings != old_crossings,
        "unlocked": state.unlocked,
        "unlocked_changed": state.unlocked != old_unlocked,
        "t6_gate": state.t6_gate,
        "transition": transition
    }

def run_steps(z_values: List[float]) -> Dict:
    """Run multiple steps and return summary."""
    results = []
    for z in z_values:
        results.append(step(z))
    
    state = get_triad_state()
    return {
        "steps": len(z_values),
        "final_z": state.z,
        "final_crossings": state.crossings,
        "final_unlocked": state.unlocked,
        "transitions": [r for r in results if r["transition"]],
        "final_state": get_status()
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_random_walk(steps: int = 100, start_z: float = 0.80, volatility: float = 0.02) -> Dict:
    """
    Simulate TRIAD dynamics with random walk.
    
    Args:
        steps: Number of simulation steps
        start_z: Starting z-coordinate
        volatility: Standard deviation of z changes
    """
    import random
    
    reset_triad_state(start_z)
    state = get_triad_state()
    
    transitions = []
    z = start_z
    
    for i in range(steps):
        # Random walk constrained to [0.75, 0.95]
        z += random.gauss(0, volatility)
        z = max(0.75, min(0.95, z))
        
        result = step(z)
        if result["transition"]:
            transitions.append({
                "step": i,
                "transition": result["transition"],
                "z": z,
                "crossings": result["crossings"]
            })
    
    return {
        "steps": steps,
        "start_z": start_z,
        "final_z": state.z,
        "crossings": state.crossings,
        "unlocked": state.unlocked,
        "transitions": transitions,
        "history_length": len(state.history)
    }

def simulate_oscillation(periods: int = 5, amplitude: float = 0.08, center: float = 0.84) -> Dict:
    """
    Simulate sinusoidal oscillation across TRIAD thresholds.
    
    This is designed to reliably trigger TRIAD unlock by oscillating
    across the HIGH and LOW thresholds.
    """
    import math as m
    
    reset_triad_state(center)
    
    steps_per_period = 50
    total_steps = periods * steps_per_period
    transitions = []
    
    for i in range(total_steps):
        # Sinusoidal oscillation
        phase = 2 * m.pi * i / steps_per_period
        z = center + amplitude * m.sin(phase)
        z = max(0.75, min(0.95, z))
        
        result = step(z)
        if result["transition"]:
            transitions.append({
                "step": i,
                "transition": result["transition"],
                "z": z,
                "crossings": result["crossings"]
            })
    
    state = get_triad_state()
    return {
        "periods": periods,
        "center": center,
        "amplitude": amplitude,
        "total_steps": total_steps,
        "final_crossings": state.crossings,
        "unlocked": state.unlocked,
        "t6_gate": state.t6_gate,
        "transitions": transitions
    }

def drive_to_unlock(max_steps: int = 500) -> Dict:
    """
    Drive the system to TRIAD unlock through controlled oscillation.
    Guarantees unlock if max_steps is sufficient.
    """
    reset_triad_state(0.80)
    state = get_triad_state()
    
    step_count = 0
    phase = 0
    
    while not state.unlocked and step_count < max_steps:
        # Oscillate between 0.80 and 0.88
        if phase == 0:
            z = 0.88  # Above HIGH
        else:
            z = 0.80  # Below LOW
        
        step(z)
        phase = 1 - phase
        step_count += 1
        
        if state.crossings >= TRIAD_REQUIRED_CROSSINGS and state.z >= TRIAD_T6:
            break
    
    return {
        "steps_taken": step_count,
        "crossings": state.crossings,
        "unlocked": state.unlocked,
        "t6_gate": state.t6_gate,
        "success": state.unlocked
    }

# ═══════════════════════════════════════════════════════════════════════════════
# QUERY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_t6_gate() -> float:
    """Get current t6 gate position."""
    return get_triad_state().t6_gate

def is_unlocked() -> bool:
    """Check if TRIAD is unlocked."""
    return get_triad_state().unlocked

def get_crossings() -> int:
    """Get current crossing count."""
    return get_triad_state().crossings

def get_crossing_history() -> List[Dict]:
    """Get all crossing events."""
    return get_triad_state().crossing_events

def get_z_history(limit: int = 100) -> List[float]:
    """Get recent z-coordinate history."""
    history = get_triad_state().history
    return history[-limit:] if len(history) > limit else history

def get_status() -> Dict:
    """Get complete TRIAD status."""
    state = get_triad_state()
    
    return {
        "z": state.z,
        "band": get_band_state(state.z).value,
        "above_high": state.above_high,
        "crossings": state.crossings,
        "crossings_remaining": max(0, TRIAD_REQUIRED_CROSSINGS - state.crossings),
        "unlocked": state.unlocked,
        "t6_gate": state.t6_gate,
        "gate_state": state.gate_state.value,
        "thresholds": {
            "high": TRIAD_HIGH,
            "low": TRIAD_LOW,
            "t6": TRIAD_T6,
            "z_c": Z_CRITICAL
        },
        "history_length": len(state.history),
        "crossing_events": len(state.crossing_events)
    }

def format_status() -> str:
    """Format TRIAD status for display."""
    status = get_status()
    
    crossing_display = "●" * status["crossings"] + "○" * status["crossings_remaining"]
    
    lines = [
        "TRIAD HYSTERESIS SYSTEM",
        "=" * 50,
        f"z-Coordinate: {status['z']:.6f}",
        f"Band: {status['band']}",
        f"Above HIGH: {status['above_high']}",
        f"Crossings: [{crossing_display}] {status['crossings']}/{TRIAD_REQUIRED_CROSSINGS}",
        f"Status: {'UNLOCKED ✓' if status['unlocked'] else 'LOCKED'}",
        f"T6 Gate: {status['t6_gate']:.4f} ({status['gate_state']})",
        "-" * 50,
        f"Thresholds: HIGH={TRIAD_HIGH}, LOW={TRIAD_LOW}, T6={TRIAD_T6}",
        f"History: {status['history_length']} samples",
        "=" * 50
    ]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH APL
# ═══════════════════════════════════════════════════════════════════════════════

def get_operator_window_for_t6() -> Dict:
    """
    Get the operator window for t6 based on TRIAD state.
    
    Before unlock: t6 uses z_c boundary with operators [+, ÷, (), −]
    After unlock: t6 uses TRIAD_T6 boundary (earlier access)
    """
    state = get_triad_state()
    
    # t6 operators (restricted set - no amplify or fusion)
    operators = ["+", "÷", "()", "−"]
    
    if state.unlocked:
        return {
            "tier": "t6",
            "boundary": TRIAD_T6,
            "operators": operators,
            "state": "TRIAD_UNLOCKED",
            "description": "T6 gate lowered to 0.83, earlier operator access"
        }
    else:
        return {
            "tier": "t6",
            "boundary": Z_CRITICAL,
            "operators": operators,
            "state": "CRITICAL_LOCKED",
            "description": "T6 gate at z_c, standard operator window"
        }

def check_t6_access(z: float) -> Dict:
    """Check if z-coordinate grants t6 access given TRIAD state."""
    state = get_triad_state()
    
    if state.unlocked:
        threshold = TRIAD_T6
    else:
        threshold = Z_CRITICAL
    
    has_access = z >= threshold
    
    return {
        "z": z,
        "threshold": threshold,
        "has_t6_access": has_access,
        "triad_unlocked": state.unlocked,
        "distance_to_threshold": z - threshold
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(format_status())
    print()
    
    # Drive to unlock
    print("Driving to TRIAD unlock...")
    result = drive_to_unlock()
    print(f"Steps: {result['steps_taken']}")
    print(f"Crossings: {result['crossings']}")
    print(f"Unlocked: {result['unlocked']}")
    print()
    
    print(format_status())
    print()
    
    # Show operator window
    window = get_operator_window_for_t6()
    print(f"T6 Operator Window: {window['state']}")
    print(f"Boundary: {window['boundary']}")
    print(f"Operators: {window['operators']}")
