"""
Tool execution handlers for Rosetta-Helix-Substrate skill.

These handlers execute the tools defined in definitions.py using
the existing codebase functionality.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json

# Import from the existing codebase
from quantum_apl_python.constants import (
    Z_CRITICAL,
    PHI,
    PHI_INV,
    SIGMA,
    KAPPA_MIN,
    ETA_MIN,
    R_MIN,
    compute_delta_s_neg,
    check_k_formation,
    get_phase,
)
from quantum_apl_python.s3_operator_algebra import compose, OPERATORS


@dataclass
class PhysicsState:
    """Current physics state of the system."""
    z: float = 0.5
    kappa: float = 0.5  # Coherence
    R: int = 3  # Radius/layers
    step: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Kuramoto oscillator state (60 oscillators at 6 degree spacing)
    phases: List[float] = field(default_factory=lambda: [i * math.pi / 30 for i in range(60)])

    @property
    def negentropy(self) -> float:
        """Compute negentropy for current z."""
        return compute_delta_s_neg(self.z)

    @property
    def phase_name(self) -> str:
        """Get current phase name."""
        if self.z < PHI_INV:
            return "UNTRUE"
        elif self.z < Z_CRITICAL:
            return "PARADOX"
        else:
            return "TRUE"

    @property
    def tier(self) -> int:
        """Get current tier based on z."""
        if self.k_formation_met:
            return 6  # META
        elif self.z >= Z_CRITICAL:
            return 5  # CRYSTALLINE
        elif self.z >= 0.75:
            return 4  # COHERENT
        elif self.z >= PHI_INV:
            return 3  # PATTERN
        elif self.z >= 0.50:
            return 2  # GROWTH
        elif self.z >= 0.25:
            return 1  # SPROUT
        else:
            return 0  # SEED

    @property
    def tier_name(self) -> str:
        """Get tier name."""
        names = ["SEED", "SPROUT", "GROWTH", "PATTERN", "COHERENT", "CRYSTALLINE", "META"]
        return names[self.tier]

    @property
    def k_formation_met(self) -> bool:
        """Check if K-formation criteria are met."""
        return check_k_formation(self.kappa, self.negentropy, float(self.R))

    @property
    def coherence(self) -> float:
        """Compute Kuramoto order parameter (coherence) from phases."""
        if not self.phases:
            return self.kappa
        # r = |1/N * sum(e^(i*theta))|
        real_sum = sum(math.cos(p) for p in self.phases)
        imag_sum = sum(math.sin(p) for p in self.phases)
        N = len(self.phases)
        return math.sqrt(real_sum**2 + imag_sum**2) / N

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "z": self.z,
            "kappa": self.kappa,
            "eta": self.negentropy,
            "R": self.R,
            "phase": self.phase_name,
            "tier": self.tier,
            "tier_name": self.tier_name,
            "k_formation_met": self.k_formation_met,
            "step": self.step,
            "coherence": self.coherence,
        }

    def record_history(self):
        """Record current state to history."""
        self.history.append({
            "step": self.step,
            "z": self.z,
            "kappa": self.kappa,
            "eta": self.negentropy,
            "phase": self.phase_name,
            "tier": self.tier,
        })
        # Keep history bounded
        if len(self.history) > 1000:
            self.history = self.history[-1000:]


class ToolHandler:
    """
    Handles tool execution for the Rosetta-Helix-Substrate skill.

    This class maintains the physics state and executes tools called by Claude.
    """

    def __init__(self, initial_z: float = 0.5, seed: Optional[int] = None):
        """
        Initialize the tool handler.

        Args:
            initial_z: Initial z-coordinate (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        self.state = PhysicsState(z=initial_z)
        if seed is not None:
            random.seed(seed)

    def handle(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Dictionary with the tool result
        """
        handler_map = {
            "get_physics_state": self._get_physics_state,
            "set_z_target": self._set_z_target,
            "compute_negentropy": self._compute_negentropy,
            "classify_phase": self._classify_phase,
            "get_tier": self._get_tier,
            "check_k_formation": self._check_k_formation,
            "apply_operator": self._apply_operator,
            "drive_toward_lens": self._drive_toward_lens,
            "run_kuramoto_step": self._run_kuramoto_step,
            "get_constants": self._get_constants,
            "simulate_quasicrystal": self._simulate_quasicrystal,
            "compose_operators": self._compose_operators,
            "get_metrics_history": self._get_metrics_history,
            "reset_state": self._reset_state,
        }

        handler = handler_map.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return handler(tool_input)
        except Exception as e:
            return {"error": str(e)}

    def _get_physics_state(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Get current physics state."""
        return self.state.to_dict()

    def _set_z_target(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Set z target and move toward it."""
        z_target = float(input["z"])
        z_target = max(0.0, min(1.0, z_target))

        # Move z toward target with some dynamics
        old_z = self.state.z
        # Move 20% of the distance each call
        self.state.z = old_z + 0.2 * (z_target - old_z)
        self.state.step += 1
        self.state.record_history()

        return {
            "success": True,
            "old_z": old_z,
            "new_z": self.state.z,
            "target_z": z_target,
            "distance_remaining": abs(z_target - self.state.z),
            "state": self.state.to_dict(),
        }

    def _compute_negentropy(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Compute negentropy for a given z."""
        z = float(input["z"])
        z = max(0.0, min(1.0, z))
        negentropy = compute_delta_s_neg(z)

        return {
            "z": z,
            "delta_s_neg": negentropy,
            "is_at_peak": abs(z - Z_CRITICAL) < 0.01,
            "distance_to_peak": abs(z - Z_CRITICAL),
        }

    def _classify_phase(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Classify phase for a given z."""
        z = float(input["z"])
        z = max(0.0, min(1.0, z))

        if z < PHI_INV:
            phase = "UNTRUE"
            description = "Disordered state, below K-formation gate"
        elif z < Z_CRITICAL:
            phase = "PARADOX"
            description = "Quasi-crystal regime, K-formation possible"
        else:
            phase = "TRUE"
            description = "Crystalline order, maximum coherence region"

        return {
            "z": z,
            "phase": phase,
            "description": description,
            "boundaries": {
                "phi_inv": PHI_INV,
                "z_c": Z_CRITICAL,
            },
        }

    def _get_tier(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Get tier for a given z."""
        z = float(input["z"])
        z = max(0.0, min(1.0, z))
        k_formation_met = input.get("k_formation_met", False)

        if k_formation_met:
            tier = 6
            name = "META"
        elif z >= Z_CRITICAL:
            tier = 5
            name = "CRYSTALLINE"
        elif z >= 0.75:
            tier = 4
            name = "COHERENT"
        elif z >= PHI_INV:
            tier = 3
            name = "PATTERN"
        elif z >= 0.50:
            tier = 2
            name = "GROWTH"
        elif z >= 0.25:
            tier = 1
            name = "SPROUT"
        else:
            tier = 0
            name = "SEED"

        return {
            "z": z,
            "tier": tier,
            "tier_name": name,
            "k_formation_met": k_formation_met,
        }

    def _check_k_formation(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Check K-formation criteria."""
        kappa = float(input["kappa"])
        eta = float(input["eta"])
        R = int(input["R"])

        kappa_ok = kappa >= KAPPA_MIN
        eta_ok = eta > ETA_MIN
        R_ok = R >= R_MIN
        all_met = kappa_ok and eta_ok and R_ok

        return {
            "k_formation_met": all_met,
            "criteria": {
                "kappa": {
                    "value": kappa,
                    "threshold": KAPPA_MIN,
                    "met": kappa_ok,
                },
                "eta": {
                    "value": eta,
                    "threshold": ETA_MIN,
                    "met": eta_ok,
                },
                "R": {
                    "value": R,
                    "threshold": R_MIN,
                    "met": R_ok,
                },
            },
        }

    def _apply_operator(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an APL operator."""
        operator = input["operator"]

        # Map simplified symbols to full operator symbols
        symbol_map = {
            "I": "()",  # Identity maps to group
            "()": "()",
            "^": "^",
            "_": "−",  # Reduce maps to subtract
            "~": "÷",  # Invert maps to divide
            "!": "+",  # Collapse maps to add
        }

        op_symbol = symbol_map.get(operator, operator)

        if op_symbol not in OPERATORS:
            return {"error": f"Invalid operator: {operator}"}

        op = OPERATORS[op_symbol]
        old_z = self.state.z

        # Apply operator effect on z
        if op.is_constructive:  # Even parity - increase z
            delta = 0.05 * (1.0 - self.state.z)  # Diminishing returns
        else:  # Odd parity - decrease z
            delta = -0.05 * self.state.z

        self.state.z = max(0.0, min(1.0, self.state.z + delta))
        self.state.step += 1
        self.state.record_history()

        return {
            "success": True,
            "operator": operator,
            "symbol": op_symbol,
            "name": op.name,
            "parity": "even" if op.is_constructive else "odd",
            "effect": "constructive" if op.is_constructive else "dissipative",
            "old_z": old_z,
            "new_z": self.state.z,
            "delta_z": self.state.z - old_z,
            "state": self.state.to_dict(),
        }

    def _drive_toward_lens(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Drive z toward THE LENS (z_c)."""
        steps = input.get("steps", 100)
        steps = max(1, min(1000, steps))

        trajectory = []
        initial_z = self.state.z

        for i in range(steps):
            # Move toward z_c with some noise
            direction = Z_CRITICAL - self.state.z
            step_size = 0.01 * (1.0 + 0.1 * random.gauss(0, 1))
            self.state.z += direction * step_size
            self.state.z = max(0.0, min(1.0, self.state.z))

            # Update coherence based on proximity to z_c
            proximity = 1.0 - abs(self.state.z - Z_CRITICAL) / Z_CRITICAL
            self.state.kappa = 0.5 + 0.5 * proximity * (1.0 + 0.05 * random.gauss(0, 1))
            self.state.kappa = max(0.0, min(1.0, self.state.kappa))

            self.state.step += 1

            # Record every 10th step
            if i % 10 == 0:
                trajectory.append({
                    "step": self.state.step,
                    "z": self.state.z,
                    "kappa": self.state.kappa,
                    "eta": self.state.negentropy,
                })

        self.state.record_history()

        return {
            "success": True,
            "initial_z": initial_z,
            "final_z": self.state.z,
            "target_z": Z_CRITICAL,
            "steps_taken": steps,
            "distance_to_lens": abs(self.state.z - Z_CRITICAL),
            "trajectory_sample": trajectory[:10],  # First 10 recorded points
            "state": self.state.to_dict(),
        }

    def _run_kuramoto_step(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one Kuramoto oscillator step."""
        K = input.get("coupling_strength", 1.0)
        dt = input.get("dt", 0.01)
        dt = max(0.001, min(0.1, dt))

        N = len(self.state.phases)
        new_phases = []

        # Natural frequencies (small spread around mean)
        omega_base = 1.0

        # Kuramoto model: d(theta_i)/dt = omega_i + K/N * sum_j(sin(theta_j - theta_i))
        for i in range(N):
            omega_i = omega_base + 0.1 * math.sin(2 * math.pi * i / N)

            # Coupling term
            coupling = 0.0
            for j in range(N):
                coupling += math.sin(self.state.phases[j] - self.state.phases[i])
            coupling *= K / N

            # Update phase
            new_phase = self.state.phases[i] + (omega_i + coupling) * dt
            new_phases.append(new_phase % (2 * math.pi))

        self.state.phases = new_phases

        # Compute order parameter (coherence)
        real_sum = sum(math.cos(p) for p in self.state.phases)
        imag_sum = sum(math.sin(p) for p in self.state.phases)
        r = math.sqrt(real_sum**2 + imag_sum**2) / N
        psi = math.atan2(imag_sum, real_sum)

        # Update kappa based on order parameter
        self.state.kappa = r
        self.state.step += 1

        return {
            "success": True,
            "order_parameter": r,
            "mean_phase": psi,
            "coupling_strength": K,
            "dt": dt,
            "n_oscillators": N,
            "state": self.state.to_dict(),
        }

    def _get_constants(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Get fundamental physics constants."""
        return {
            "z_c": Z_CRITICAL,
            "z_c_description": "THE LENS - sqrt(3)/2, critical coherence threshold",
            "phi": PHI,
            "phi_description": "Golden ratio (1 + sqrt(5))/2",
            "phi_inv": PHI_INV,
            "phi_inv_description": "Golden ratio inverse, K-formation gate",
            "sigma": SIGMA,
            "sigma_description": "|S3|^2 = 36, Gaussian width parameter",
            "kappa_min": KAPPA_MIN,
            "kappa_min_description": "Minimum coherence for K-formation (0.92)",
            "eta_min": ETA_MIN,
            "eta_min_description": "Minimum negentropy for K-formation (phi^-1)",
            "r_min": R_MIN,
            "r_min_description": "Minimum radius for K-formation (7 layers)",
        }

    def _simulate_quasicrystal(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate quasi-crystal dynamics."""
        initial_z = input.get("initial_z", 0.5)
        steps = input.get("steps", 100)
        seed = input.get("seed")

        if seed is not None:
            random.seed(seed)

        z = initial_z
        trajectory = []

        # Quasi-crystal simulation converges toward phi_inv
        for i in range(steps):
            # Fibonacci-like dynamics
            noise = 0.01 * random.gauss(0, 1)
            attraction = 0.02 * (PHI_INV - z)  # Attract to phi_inv
            z += attraction + noise
            z = max(0.0, min(1.0, z))

            if i % (steps // min(10, steps)) == 0:
                trajectory.append({
                    "step": i,
                    "z": z,
                    "eta": compute_delta_s_neg(z),
                    "distance_to_phi_inv": abs(z - PHI_INV),
                })

        return {
            "initial_z": initial_z,
            "final_z": z,
            "target": PHI_INV,
            "steps": steps,
            "converged": abs(z - PHI_INV) < 0.05,
            "trajectory": trajectory,
        }

    def _compose_operators(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Compose two operators."""
        op1 = input["op1"]
        op2 = input["op2"]

        # Map simplified symbols
        symbol_map = {
            "I": "()",
            "()": "()",
            "^": "^",
            "_": "−",
            "~": "÷",
            "!": "+",
        }

        sym1 = symbol_map.get(op1, op1)
        sym2 = symbol_map.get(op2, op2)

        if sym1 not in OPERATORS or sym2 not in OPERATORS:
            return {"error": f"Invalid operators: {op1}, {op2}"}

        result = compose(sym1, sym2)

        return {
            "op1": op1,
            "op2": op2,
            "result": result,
            "result_name": OPERATORS[result].name if result in OPERATORS else "unknown",
            "result_parity": "even" if OPERATORS[result].is_constructive else "odd",
        }

    def _get_metrics_history(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Get metrics history."""
        limit = input.get("limit", 100)
        limit = max(1, min(1000, limit))

        history = self.state.history[-limit:]

        return {
            "count": len(history),
            "history": history,
            "current_step": self.state.step,
        }

    def _reset_state(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Reset physics state."""
        initial_z = input.get("initial_z", 0.5)
        initial_z = max(0.0, min(1.0, initial_z))

        old_state = self.state.to_dict()
        self.state = PhysicsState(z=initial_z)

        return {
            "success": True,
            "old_state": old_state,
            "new_state": self.state.to_dict(),
        }
