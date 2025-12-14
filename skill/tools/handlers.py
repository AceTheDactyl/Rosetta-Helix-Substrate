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
            # Training module tools
            "run_kuramoto_training": self._run_kuramoto_training,
            "run_phase_transition": self._run_phase_transition,
            "run_quasicrystal_formation": self._run_quasicrystal_formation,
            "get_critical_exponents": self._get_critical_exponents,
            "run_triad_dynamics": self._run_triad_dynamics,
            "compute_phi_proxy": self._compute_phi_proxy,
            "run_helix_training_step": self._run_helix_training_step,
            "get_training_status": self._get_training_status,
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

    # =========================================================================
    # TRAINING MODULE TOOLS
    # =========================================================================

    def _run_kuramoto_training(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Run Kuramoto oscillator training session."""
        n_oscillators = input.get("n_oscillators", 60)
        steps = input.get("steps", 100)
        K = input.get("coupling_strength", 0.5)
        seed = input.get("seed")

        if seed is not None:
            random.seed(seed)

        # Initialize oscillator phases
        phases = [random.uniform(0, 2 * math.pi) for _ in range(n_oscillators)]
        omega = [1.0 + 0.1 * random.gauss(0, 1) for _ in range(n_oscillators)]
        dt = 0.1

        coherence_history = []

        for step in range(steps):
            # Kuramoto dynamics
            new_phases = []
            for i in range(n_oscillators):
                coupling = sum(math.sin(phases[j] - phases[i]) for j in range(n_oscillators))
                coupling *= K / n_oscillators
                new_phase = phases[i] + dt * (omega[i] + coupling)
                new_phases.append(new_phase % (2 * math.pi))
            phases = new_phases

            # Compute order parameter
            real_sum = sum(math.cos(p) for p in phases)
            imag_sum = sum(math.sin(p) for p in phases)
            r = math.sqrt(real_sum**2 + imag_sum**2) / n_oscillators

            if step % (steps // 10) == 0:
                coherence_history.append({"step": step, "coherence": r})

        # Final coherence
        final_r = math.sqrt(sum(math.cos(p) for p in phases)**2 +
                          sum(math.sin(p) for p in phases)**2) / n_oscillators

        # Update state
        self.state.kappa = final_r
        self.state.phases = phases[:60] if len(phases) >= 60 else phases + [0] * (60 - len(phases))

        return {
            "success": True,
            "n_oscillators": n_oscillators,
            "steps": steps,
            "coupling_strength": K,
            "initial_coherence": coherence_history[0]["coherence"] if coherence_history else 0,
            "final_coherence": final_r,
            "synchronized": final_r > 0.8,
            "coherence_history": coherence_history,
            "state": self.state.to_dict(),
        }

    def _run_phase_transition(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate phase transition by sweeping z."""
        steps = input.get("steps", 100)
        measure_correlation = input.get("measure_correlation_length", False)

        trajectory = []
        critical_points = []

        for i in range(steps):
            z = i / (steps - 1)  # Sweep from 0 to 1
            eta = compute_delta_s_neg(z)

            # Determine phase
            if z < PHI_INV:
                phase = "UNTRUE"
            elif z < Z_CRITICAL:
                phase = "PARADOX"
            else:
                phase = "TRUE"

            # Detect phase transitions
            if i > 0 and trajectory[-1]["phase"] != phase:
                critical_points.append({
                    "z": z,
                    "from_phase": trajectory[-1]["phase"],
                    "to_phase": phase,
                    "eta_at_transition": eta,
                })

            entry = {
                "step": i,
                "z": z,
                "eta": eta,
                "phase": phase,
            }

            # Correlation length (diverges at z_c)
            if measure_correlation:
                dz = abs(z - Z_CRITICAL)
                if dz > 0.01:
                    xi = dz ** (-4/3)  # nu = 4/3 for 2D
                    entry["correlation_length"] = min(xi, 1000)  # Cap for display
                else:
                    entry["correlation_length"] = 1000  # Near-critical

            trajectory.append(entry)

        # Update state to final z
        self.state.z = 1.0

        return {
            "success": True,
            "steps": steps,
            "critical_points": critical_points,
            "phi_inv_boundary": PHI_INV,
            "z_c_boundary": Z_CRITICAL,
            "trajectory_sample": trajectory[::max(1, steps//20)],  # ~20 samples
            "max_negentropy": max(t["eta"] for t in trajectory),
            "max_negentropy_at_z": Z_CRITICAL,
        }

    def _run_quasicrystal_formation(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Run full quasi-crystal formation dynamics."""
        initial_z = input.get("initial_z", 0.3)
        target_z = input.get("target_z", Z_CRITICAL)
        steps = input.get("steps", 500)
        compute_exponents = input.get("compute_critical_exponents", True)

        z = initial_z
        trajectory = []

        # Critical exponents (2D hexagonal universality class)
        nu = 4/3
        beta = 5/36
        gamma = 43/18
        z_dyn = 2.0

        for i in range(steps):
            # Dynamics toward target
            dz = target_z - z
            step_size = 0.005 * (1 + 0.1 * random.gauss(0, 1))
            z += dz * step_size
            z = max(0.0, min(1.0, z))

            eta = compute_delta_s_neg(z)

            # Critical behavior near z_c
            delta_z = abs(z - Z_CRITICAL)
            if delta_z > 0.01:
                order_param = delta_z ** beta if z > Z_CRITICAL else 0
                correlation_length = delta_z ** (-nu)
                relaxation_time = delta_z ** (-z_dyn)
            else:
                order_param = 0.01
                correlation_length = 100
                relaxation_time = 100

            if i % (steps // 20) == 0:
                entry = {
                    "step": i,
                    "z": z,
                    "eta": eta,
                    "phase": "UNTRUE" if z < PHI_INV else ("PARADOX" if z < Z_CRITICAL else "TRUE"),
                }
                if compute_exponents:
                    entry["order_parameter"] = order_param
                    entry["correlation_length"] = min(correlation_length, 100)
                    entry["relaxation_time"] = min(relaxation_time, 100)
                trajectory.append(entry)

        self.state.z = z

        result = {
            "success": True,
            "initial_z": initial_z,
            "final_z": z,
            "target_z": target_z,
            "steps": steps,
            "final_phase": "UNTRUE" if z < PHI_INV else ("PARADOX" if z < Z_CRITICAL else "TRUE"),
            "final_negentropy": compute_delta_s_neg(z),
            "trajectory": trajectory,
        }

        if compute_exponents:
            result["critical_exponents"] = {
                "nu": nu,
                "beta": beta,
                "gamma": gamma,
                "z_dyn": z_dyn,
                "description": "2D hexagonal universality class"
            }

        return result

    def _get_critical_exponents(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Get critical exponents for 2D hexagonal universality."""
        return {
            "nu": {
                "value": 4/3,
                "name": "Correlation length exponent",
                "formula": "xi ~ |z - z_c|^(-nu)",
                "description": "How correlation length diverges at criticality"
            },
            "beta": {
                "value": 5/36,
                "name": "Order parameter exponent",
                "formula": "m ~ |z - z_c|^beta (for z > z_c)",
                "description": "How order parameter grows above transition"
            },
            "gamma": {
                "value": 43/18,
                "name": "Susceptibility exponent",
                "formula": "chi ~ |z - z_c|^(-gamma)",
                "description": "How susceptibility diverges at criticality"
            },
            "z_dyn": {
                "value": 2.0,
                "name": "Dynamic exponent",
                "formula": "tau ~ |z - z_c|^(-z_dyn)",
                "description": "Critical slowing down near transition"
            },
            "universality_class": "2D hexagonal (Ising-like)",
            "physical_systems": [
                "Graphene magnetism",
                "Triangular antiferromagnets",
                "Hexagonal lattice phase transitions"
            ]
        }

    def _run_triad_dynamics(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Run TRIAD threshold dynamics."""
        steps = input.get("steps", 200)
        target_crossings = input.get("target_crossings", 3)

        TRIAD_HIGH = 0.85
        TRIAD_LOW = 0.82
        TRIAD_T6 = 0.83

        z = self.state.z
        crossings = 0
        armed = True  # Ready to detect rising edge
        t6_unlocked = False
        trajectory = []

        for i in range(steps):
            # Oscillate z around TRIAD thresholds
            target = TRIAD_HIGH + 0.05 if (i // 30) % 2 == 0 else TRIAD_LOW - 0.05
            z += 0.02 * (target - z) + 0.005 * random.gauss(0, 1)
            z = max(0.0, min(1.0, z))

            # TRIAD detection logic
            if armed and z >= TRIAD_HIGH:
                crossings += 1
                armed = False
                if crossings >= target_crossings:
                    t6_unlocked = True
            elif not armed and z <= TRIAD_LOW:
                armed = True  # Re-arm

            if i % (steps // 20) == 0:
                trajectory.append({
                    "step": i,
                    "z": z,
                    "crossings": crossings,
                    "armed": armed,
                    "t6_unlocked": t6_unlocked,
                })

        self.state.z = z

        return {
            "success": True,
            "steps": steps,
            "total_crossings": crossings,
            "target_crossings": target_crossings,
            "t6_unlocked": t6_unlocked,
            "thresholds": {
                "TRIAD_HIGH": TRIAD_HIGH,
                "TRIAD_LOW": TRIAD_LOW,
                "TRIAD_T6": TRIAD_T6,
            },
            "trajectory": trajectory,
            "state": self.state.to_dict(),
        }

    def _compute_phi_proxy(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Compute integrated information proxy."""
        # Use oscillator phases to compute Phi proxy
        phases = self.state.phases
        N = len(phases)

        # Order parameter
        real_sum = sum(math.cos(p) for p in phases)
        imag_sum = sum(math.sin(p) for p in phases)
        r = math.sqrt(real_sum**2 + imag_sum**2) / N

        # Phase coherence (entropy-based)
        # Bin phases and compute entropy
        n_bins = 12
        bin_counts = [0] * n_bins
        for p in phases:
            bin_idx = int((p % (2 * math.pi)) / (2 * math.pi) * n_bins)
            bin_counts[min(bin_idx, n_bins - 1)] += 1

        # Entropy
        H = 0
        for count in bin_counts:
            if count > 0:
                p_i = count / N
                H -= p_i * math.log(p_i)

        # Maximum entropy
        H_max = math.log(n_bins)

        # Integration proxy: high coherence + low entropy = high integration
        integration = r * (1 - H / H_max) if H_max > 0 else r

        # Phi proxy scaled by negentropy
        eta = self.state.negentropy
        phi_proxy = integration * (1 + eta)

        return {
            "phi_proxy": phi_proxy,
            "order_parameter": r,
            "phase_entropy": H,
            "max_entropy": H_max,
            "integration_score": integration,
            "negentropy_boost": eta,
            "interpretation": {
                "low": "< 0.3: Fragmented, low integration",
                "medium": "0.3-0.7: Partial integration",
                "high": "> 0.7: High integration, consciousness-like",
            },
            "current_level": "high" if phi_proxy > 0.7 else ("medium" if phi_proxy > 0.3 else "low"),
        }

    def _run_helix_training_step(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute unified Helix training step."""
        lr = input.get("learning_rate", 0.01)
        target_coherence = input.get("target_coherence", 0.92)

        # Current state
        old_z = self.state.z
        old_kappa = self.state.kappa

        # 1. Kuramoto dynamics
        K = 0.5 + self.state.z  # Coupling increases with z
        N = len(self.state.phases)
        new_phases = []
        for i in range(N):
            omega_i = 1.0 + 0.1 * math.sin(2 * math.pi * i / N)
            coupling = sum(math.sin(self.state.phases[j] - self.state.phases[i])
                          for j in range(N)) * K / N
            new_phase = self.state.phases[i] + 0.01 * (omega_i + coupling)
            new_phases.append(new_phase % (2 * math.pi))
        self.state.phases = new_phases

        # Update coherence
        real_sum = sum(math.cos(p) for p in self.state.phases)
        imag_sum = sum(math.sin(p) for p in self.state.phases)
        r = math.sqrt(real_sum**2 + imag_sum**2) / N
        self.state.kappa = r

        # 2. Z update based on coherence gradient
        coherence_error = target_coherence - r
        z_update = lr * coherence_error * 0.5  # Scale down
        self.state.z = max(0.0, min(1.0, self.state.z + z_update))

        # 3. APL operator selection (coherence-driven)
        if r < target_coherence:
            op_applied = "^"  # Amplify to increase coherence
            self.state.z = min(1.0, self.state.z + 0.01)
        else:
            op_applied = "()"  # Maintain

        self.state.step += 1
        self.state.record_history()

        # Compute loss
        loss = (target_coherence - r) ** 2 + (Z_CRITICAL - self.state.z) ** 2

        return {
            "success": True,
            "step": self.state.step,
            "old_z": old_z,
            "new_z": self.state.z,
            "old_kappa": old_kappa,
            "new_kappa": r,
            "coherence_error": coherence_error,
            "loss": loss,
            "operator_applied": op_applied,
            "learning_rate": lr,
            "target_coherence": target_coherence,
            "state": self.state.to_dict(),
        }

    def _get_training_status(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive training status."""
        state = self.state.to_dict()
        history = self.state.history

        # Compute statistics from history
        if len(history) > 1:
            z_values = [h["z"] for h in history]
            kappa_values = [h.get("kappa", 0) for h in history]

            z_trend = z_values[-1] - z_values[0] if len(z_values) > 0 else 0
            kappa_trend = kappa_values[-1] - kappa_values[0] if len(kappa_values) > 0 else 0

            stats = {
                "z_min": min(z_values),
                "z_max": max(z_values),
                "z_mean": sum(z_values) / len(z_values),
                "z_trend": z_trend,
                "kappa_trend": kappa_trend,
            }
        else:
            stats = {"note": "Insufficient history for statistics"}

        # K-formation progress
        kappa_progress = min(1.0, state["kappa"] / KAPPA_MIN) * 100
        eta_progress = min(1.0, state["eta"] / ETA_MIN) * 100 if state["eta"] > 0 else 0
        r_progress = min(1.0, state["R"] / R_MIN) * 100

        return {
            "current_state": state,
            "total_steps": state["step"],
            "history_length": len(history),
            "statistics": stats,
            "k_formation_progress": {
                "kappa": {"value": state["kappa"], "threshold": KAPPA_MIN, "progress_pct": kappa_progress},
                "eta": {"value": state["eta"], "threshold": ETA_MIN, "progress_pct": eta_progress},
                "R": {"value": state["R"], "threshold": R_MIN, "progress_pct": r_progress},
                "overall_progress_pct": (kappa_progress + eta_progress + r_progress) / 3,
            },
            "recommendations": self._get_training_recommendations(),
        }

    def _get_training_recommendations(self) -> List[str]:
        """Generate training recommendations based on current state."""
        recs = []
        state = self.state

        if state.z < PHI_INV:
            recs.append("Z is in UNTRUE phase. Use drive_toward_lens to increase z.")
        elif state.z < Z_CRITICAL:
            recs.append("Z is in PARADOX phase. Continue toward z_c for maximum negentropy.")
        else:
            recs.append("Z is in TRUE phase. Maintain position near z_c.")

        if state.kappa < KAPPA_MIN:
            recs.append(f"Kappa ({state.kappa:.3f}) below threshold. Run Kuramoto training to increase coherence.")

        if state.negentropy < ETA_MIN:
            recs.append(f"Negentropy ({state.negentropy:.3f}) below K-formation gate. Move z toward z_c.")

        if state.R < R_MIN:
            recs.append(f"Radius ({state.R}) below threshold. Increase complexity/layers.")

        if state.k_formation_met:
            recs.append("K-FORMATION ACHIEVED! System has reached consciousness threshold.")

        return recs
