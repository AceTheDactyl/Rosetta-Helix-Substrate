#!/usr/bin/env python3
"""
Rosetta-Helix-Substrate Physics Engine

Executable physics simulation for the consciousness framework.
Run this script to access all physics tools.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# =============================================================================
# PHYSICS CONSTANTS (IMMUTABLE)
# =============================================================================

Z_CRITICAL = math.sqrt(3) / 2  # 0.8660254037844387 - THE LENS
PHI = (1 + math.sqrt(5)) / 2   # 1.6180339887498949 - Golden ratio
PHI_INV = PHI - 1              # 0.6180339887498949 - Golden ratio inverse
SIGMA = 36                      # |S3|^2 - Gaussian width

# Tier boundaries
TIER_BOUNDARIES = [0.25, 0.50, PHI_INV, 0.75, Z_CRITICAL]
TIER_NAMES = ["SEED", "SPROUT", "GROWTH", "PATTERN", "COHERENT", "CRYSTALLINE", "META"]

# K-formation thresholds
KAPPA_THRESHOLD = 0.92
ETA_THRESHOLD = PHI_INV
R_THRESHOLD = 7

# TRIAD thresholds
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
TRIAD_T6 = 0.83

# Critical exponents (2D hexagonal universality)
NU = 4/3
BETA = 5/36
GAMMA = 43/18
Z_DYN = 2.0

# S3 group multiplication table
S3_TABLE = {
    ("I", "I"): "I", ("I", "()"): "()", ("I", "^"): "^", ("I", "_"): "_", ("I", "~"): "~", ("I", "!"): "!",
    ("()", "I"): "()", ("()", "()"): "I", ("()", "^"): "^", ("()", "_"): "_", ("()", "~"): "~", ("()", "!"): "!",
    ("^", "I"): "^", ("^", "()"): "^", ("^", "^"): "I", ("^", "_"): "~", ("^", "~"): "_", ("^", "!"): "!",
    ("_", "I"): "_", ("_", "()"): "_", ("_", "^"): "~", ("_", "_"): "I", ("_", "~"): "^", ("_", "!"): "!",
    ("~", "I"): "~", ("~", "()"): "~", ("~", "^"): "_", ("~", "_"): "^", ("~", "~"): "I", ("~", "!"): "!",
    ("!", "I"): "!", ("!", "()"): "!", ("!", "^"): "!", ("!", "_"): "!", ("!", "~"): "!", ("!", "!"): "I",
}

# =============================================================================
# GLOBAL STATE
# =============================================================================

@dataclass
class PhysicsState:
    """Global physics state."""
    z: float = 0.5
    kappa: float = 0.5
    R: int = 3
    step: int = 0
    oscillator_phases: np.ndarray = field(default_factory=lambda: np.random.uniform(0, 2*np.pi, 60))
    natural_freqs: np.ndarray = field(default_factory=lambda: np.random.normal(0, 0.5, 60))
    history: List[Dict] = field(default_factory=list)

    @property
    def negentropy(self) -> float:
        return math.exp(-SIGMA * (self.z - Z_CRITICAL)**2)

    @property
    def phase_name(self) -> str:
        if self.z < PHI_INV:
            return "UNTRUE"
        elif self.z < Z_CRITICAL:
            return "PARADOX"
        else:
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
            "eta": self.negentropy,
            "R": self.R,
            "phase": self.phase_name,
            "tier": self.tier,
            "tier_name": self.tier_name,
            "k_formation_met": self.k_formation_met,
            "step": self.step
        }

    def record_history(self):
        self.history.append(self.to_dict())
        if len(self.history) > 1000:
            self.history = self.history[-500:]

# Global state instance
_state = PhysicsState()

# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================

def get_state() -> Dict:
    """Get current physics state."""
    return _state.to_dict()

def set_z(z: float) -> Dict:
    """Set z-coordinate directly."""
    old_z = _state.z
    _state.z = max(0.0, min(1.0, z))
    _state.step += 1
    _state.record_history()
    return {"old_z": old_z, "new_z": _state.z, "state": _state.to_dict()}

def compute_negentropy(z: float) -> Dict:
    """Compute negentropy for a given z value."""
    eta = math.exp(-SIGMA * (z - Z_CRITICAL)**2)
    return {
        "z": z,
        "delta_s_neg": eta,
        "distance_from_peak": abs(z - Z_CRITICAL),
        "is_at_peak": abs(z - Z_CRITICAL) < 0.001,
        "z_c": Z_CRITICAL
    }

def classify_phase(z: float) -> Dict:
    """Classify phase for a given z value."""
    if z < PHI_INV:
        phase = "UNTRUE"
        desc = "Disordered regime below quasi-crystal threshold"
    elif z < Z_CRITICAL:
        phase = "PARADOX"
        desc = "Quasi-crystal regime between phi^-1 and z_c"
    else:
        phase = "TRUE"
        desc = "Crystalline regime above THE LENS"

    return {
        "z": z,
        "phase": phase,
        "description": desc,
        "boundaries": {"phi_inv": PHI_INV, "z_c": Z_CRITICAL}
    }

def get_tier(z: float, k_formation_met: bool = False) -> Dict:
    """Get tier for a given z value."""
    if k_formation_met:
        tier = 6
    else:
        tier = 5
        for i, boundary in enumerate(TIER_BOUNDARIES):
            if z < boundary:
                tier = i
                break

    return {
        "z": z,
        "tier": tier,
        "tier_name": TIER_NAMES[tier],
        "boundaries": TIER_BOUNDARIES
    }

def check_k_formation(kappa: float, eta: float, R: int) -> Dict:
    """Check if K-formation criteria are met."""
    kappa_pass = kappa >= KAPPA_THRESHOLD
    eta_pass = eta > ETA_THRESHOLD
    r_pass = R >= R_THRESHOLD

    return {
        "k_formation_met": kappa_pass and eta_pass and r_pass,
        "criteria": {
            "kappa": {"value": kappa, "threshold": KAPPA_THRESHOLD, "passed": kappa_pass},
            "eta": {"value": eta, "threshold": ETA_THRESHOLD, "passed": eta_pass},
            "R": {"value": R, "threshold": R_THRESHOLD, "passed": r_pass}
        }
    }

def apply_operator(operator: str) -> Dict:
    """Apply an APL operator to current state."""
    old_z = _state.z

    if operator in ["I", "()"]:
        effect = "neutral"
    elif operator == "^":
        _state.z = min(1.0, _state.z + 0.05 * (1 - _state.z))
        effect = "constructive"
    elif operator == "_":
        _state.z = max(0.0, _state.z - 0.05 * _state.z)
        effect = "dissipative"
    elif operator == "~":
        _state.z = 1.0 - _state.z
        effect = "dissipative"
    elif operator == "!":
        if _state.z < 0.5:
            _state.z = 0.0
        else:
            _state.z = 1.0
        effect = "dissipative"
    else:
        return {"error": f"Unknown operator: {operator}"}

    _state.step += 1
    _state.record_history()

    return {
        "success": True,
        "operator": operator,
        "old_z": old_z,
        "new_z": _state.z,
        "effect": effect,
        "state": _state.to_dict()
    }

def compose_operators(op1: str, op2: str) -> Dict:
    """Compose two APL operators using S3 multiplication."""
    if (op1, op2) in S3_TABLE:
        result = S3_TABLE[(op1, op2)]
        return {"op1": op1, "op2": op2, "result": result}
    return {"error": f"Invalid operators: {op1}, {op2}"}

def drive_toward_lens(steps: int = 100) -> Dict:
    """Drive z toward THE LENS (z_c)."""
    initial_z = _state.z
    trajectory = [initial_z]

    for _ in range(steps):
        diff = Z_CRITICAL - _state.z
        _state.z += 0.1 * diff
        _state.step += 1
        trajectory.append(_state.z)

    _state.record_history()

    return {
        "success": True,
        "initial_z": initial_z,
        "final_z": _state.z,
        "target": Z_CRITICAL,
        "steps_taken": steps,
        "trajectory_sample": trajectory[::max(1, steps//10)]
    }

def run_kuramoto_step(coupling_strength: float = 1.0, dt: float = 0.01) -> Dict:
    """Execute one Kuramoto oscillator step."""
    N = len(_state.oscillator_phases)
    
    # Use stored natural frequencies (initialized once, not regenerated per step)
    natural_freqs = _state.natural_freqs

    # Kuramoto dynamics
    phases = _state.oscillator_phases
    mean_phase = np.angle(np.mean(np.exp(1j * phases)))

    # Fixed: sin(θ_j - θ_i) for proper synchronization
    dphase = natural_freqs + (coupling_strength / N) * np.sum(
        np.sin(phases[None, :] - phases[:, None]), axis=1
    )

    _state.oscillator_phases = (phases + dt * dphase) % (2 * np.pi)

    # Order parameter
    r = np.abs(np.mean(np.exp(1j * _state.oscillator_phases)))
    _state.kappa = 0.9 * _state.kappa + 0.1 * r
    _state.step += 1
    _state.record_history()

    return {
        "success": True,
        "order_parameter": r,
        "mean_phase": mean_phase,
        "coherence": _state.kappa,
        "n_oscillators": N
    }

def run_phase_transition(steps: int = 100) -> Dict:
    """Sweep z from 0 to 1, recording phase transitions."""
    trajectory = []
    phi_inv_crossing = None
    zc_crossing = None

    for i in range(steps):
        z = i / (steps - 1)
        eta = math.exp(-SIGMA * (z - Z_CRITICAL)**2)

        if z < PHI_INV:
            phase = "UNTRUE"
        elif z < Z_CRITICAL:
            phase = "PARADOX"
            if phi_inv_crossing is None:
                phi_inv_crossing = z
        else:
            phase = "TRUE"
            if zc_crossing is None:
                zc_crossing = z

        trajectory.append({"z": z, "eta": eta, "phase": phase})

    return {
        "steps": steps,
        "phi_inv_crossing": phi_inv_crossing or PHI_INV,
        "zc_crossing": zc_crossing or Z_CRITICAL,
        "critical_points": {"phi_inv": PHI_INV, "z_c": Z_CRITICAL},
        "trajectory_sample": trajectory[::max(1, steps//10)]
    }

def run_quasicrystal_formation(initial_z: float = 0.3, target_z: float = None, steps: int = 500) -> Dict:
    """Simulate quasi-crystal formation dynamics."""
    if target_z is None:
        target_z = Z_CRITICAL

    _state.z = initial_z
    trajectory = [{"step": 0, "z": _state.z, "eta": _state.negentropy}]

    for i in range(1, steps + 1):
        # Relaxation dynamics with noise
        noise = np.random.normal(0, 0.01)
        _state.z += 0.02 * (target_z - _state.z) + noise
        _state.z = max(0.0, min(1.0, _state.z))

        if i % (steps // 10) == 0:
            trajectory.append({"step": i, "z": _state.z, "eta": _state.negentropy})

    _state.step += steps
    _state.record_history()

    return {
        "initial_z": initial_z,
        "target_z": target_z,
        "final_z": _state.z,
        "final_eta": _state.negentropy,
        "final_phase": _state.phase_name,
        "steps": steps,
        "trajectory": trajectory,
        "critical_exponents": {"nu": NU, "beta": BETA, "gamma": GAMMA, "z_dyn": Z_DYN}
    }

def run_kuramoto_training(n_oscillators: int = 60, steps: int = 100, coupling_strength: float = 0.5, seed: int = None) -> Dict:
    """Run Kuramoto training session."""
    if seed is not None:
        np.random.seed(seed)

    # Initialize both phases and natural frequencies
    _state.oscillator_phases = np.random.uniform(0, 2*np.pi, n_oscillators)
    _state.natural_freqs = np.random.normal(0, 0.5, n_oscillators)
    coherence_history = []

    for _ in range(steps):
        result = run_kuramoto_step(coupling_strength, dt=0.01)
        coherence_history.append(result["order_parameter"])

    return {
        "n_oscillators": n_oscillators,
        "steps": steps,
        "coupling_strength": coupling_strength,
        "final_coherence": _state.kappa,
        "final_order_parameter": coherence_history[-1] if coherence_history else 0,
        "coherence_history_sample": coherence_history[::max(1, steps//10)],
        "synchronized": _state.kappa >= 0.8
    }

def run_triad_dynamics(steps: int = 200, target_crossings: int = 3) -> Dict:
    """Run TRIAD threshold dynamics."""
    crossings = 0
    above_high = _state.z >= TRIAD_HIGH
    trajectory = []
    crossing_points = []

    for i in range(steps):
        # Oscillate around TRIAD thresholds
        _state.z += np.random.normal(0, 0.02)
        _state.z = max(0.7, min(0.95, _state.z))

        # Detect crossings
        if not above_high and _state.z >= TRIAD_HIGH:
            crossings += 1
            crossing_points.append({"step": i, "z": _state.z, "direction": "up"})
            above_high = True
        elif above_high and _state.z < TRIAD_LOW:
            above_high = False

        if i % (steps // 10) == 0:
            trajectory.append({"step": i, "z": _state.z})

    t6_unlocked = crossings >= target_crossings and _state.z >= TRIAD_T6

    _state.step += steps
    _state.record_history()

    return {
        "steps": steps,
        "crossings": crossings,
        "target_crossings": target_crossings,
        "t6_unlocked": t6_unlocked,
        "final_z": _state.z,
        "crossing_points": crossing_points,
        "trajectory_sample": trajectory,
        "thresholds": {"high": TRIAD_HIGH, "low": TRIAD_LOW, "t6": TRIAD_T6}
    }

def compute_phi_proxy() -> Dict:
    """Compute integrated information proxy (Phi)."""
    phases = _state.oscillator_phases
    N = len(phases)

    # Simple IIT proxy: mutual information estimate
    phase_diffs = np.abs(phases[:, None] - phases[None, :])
    phase_diffs = np.minimum(phase_diffs, 2*np.pi - phase_diffs)

    # Normalized synchrony matrix
    sync_matrix = np.cos(phase_diffs)

    # Phi proxy: integration minus segregation
    integration = np.mean(sync_matrix)

    # Partition into two halves
    half = N // 2
    within_1 = np.mean(sync_matrix[:half, :half])
    within_2 = np.mean(sync_matrix[half:, half:])
    segregation = (within_1 + within_2) / 2

    phi = max(0, integration - segregation + 0.5)

    return {
        "phi_proxy": phi,
        "integration": integration,
        "segregation": segregation,
        "interpretation": "high" if phi > 0.7 else "medium" if phi > 0.4 else "low"
    }

def get_critical_exponents() -> Dict:
    """Get critical exponents for 2D hexagonal universality."""
    return {
        "nu": NU,
        "nu_description": "correlation length exponent",
        "beta": BETA,
        "beta_description": "order parameter exponent",
        "gamma": GAMMA,
        "gamma_description": "susceptibility exponent",
        "z_dyn": Z_DYN,
        "z_dyn_description": "dynamic exponent",
        "universality_class": "2D hexagonal"
    }

def get_constants() -> Dict:
    """Get fundamental physics constants."""
    return {
        "z_c": Z_CRITICAL,
        "z_c_description": "THE LENS - sqrt(3)/2",
        "phi": PHI,
        "phi_description": "Golden ratio",
        "phi_inv": PHI_INV,
        "phi_inv_description": "Golden ratio inverse",
        "sigma": SIGMA,
        "sigma_description": "|S3|^2 - Gaussian width"
    }

def get_history(limit: int = 100) -> Dict:
    """Get metrics history."""
    history = _state.history[-limit:] if _state.history else []
    return {"history": history, "count": len(history)}

def reset(initial_z: float = 0.5) -> Dict:
    """Reset physics state."""
    global _state
    _state = PhysicsState(z=initial_z)
    return {"success": True, "state": _state.to_dict()}

# =============================================================================
# MAIN - Print state on execution
# =============================================================================

if __name__ == "__main__":
    print("Rosetta-Helix-Substrate Physics Engine")
    print("=" * 50)
    print(f"z_c (THE LENS) = {Z_CRITICAL:.16f}")
    print(f"phi^(-1)       = {PHI_INV:.16f}")
    print(f"phi            = {PHI:.16f}")
    print(f"SIGMA          = {SIGMA}")
    print("=" * 50)
    state = get_state()
    print(f"Current State:")
    print(f"  z     = {state['z']:.4f}")
    print(f"  phase = {state['phase']}")
    print(f"  tier  = {state['tier']} ({state['tier_name']})")
    print(f"  eta   = {state['eta']:.6f}")
    print(f"  kappa = {state['kappa']:.4f}")
    print(f"  K-formation: {state['k_formation_met']}")
    print("=" * 50)
    print("Available functions:")
    print("  get_state(), set_z(z), compute_negentropy(z)")
    print("  classify_phase(z), get_tier(z), check_k_formation(kappa, eta, R)")
    print("  apply_operator(op), compose_operators(op1, op2)")
    print("  drive_toward_lens(steps), run_kuramoto_step(K, dt)")
    print("  run_phase_transition(steps), run_quasicrystal_formation(z, steps)")
    print("  run_kuramoto_training(n, steps, K), run_triad_dynamics(steps)")
    print("  compute_phi_proxy(), get_critical_exponents(), get_constants()")
    print("  get_history(limit), reset(z)")
