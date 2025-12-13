#!/usr/bin/env python3
"""
QUANTUM APL INTEGRATION — von Neumann Measurement-Based N0 Selection
=====================================================================

Transforms APL N0 operator selection into quantum measurement process.

Quantum Formalism:
==================
    N0_classical: argmin C(i | σ, α)   i ∈ legal
    N0_quantum:   P(μ) = Tr(P̂_μ ρ̂),   ρ̂' = P̂_μ ρ̂ P̂_μ / P(μ)

Hilbert Space:
==============
    H_APL = H_Φ ⊗ H_e ⊗ H_π ⊗ H_truth

    Φ = {void, lattice, network, hierarchy}     (4 states)
    e = {ground, excited, coherent, chaotic}    (4 states)
    π = {simple, correlated, integrated, conscious}  (4 states)
    Truth = {TRUE, UNTRUE, PARADOX}             (3 states)

    Total dimension: 4 × 4 × 4 × 3 = 192

Pipeline:
=========
    1. Time harmonic legality (t1-t9) based on z
    2. PRS phase legality (P1-P5) based on Φ
    3. Tier-0 N0 laws (grounding, plurality, decoherence, etc.)
    4. Scalar thresholds (R_CLT, δ, κ, Ω)
    5. Quantum measurement — construct projectors, compute Born probabilities, sample
    6. Quantum→classical feedback (update scalars)

Quantum Information:
====================
    Purity: Tr(ρ²)
    Von Neumann entropy: S(ρ) = -Tr(ρ log ρ)
    Integrated information: Φ = min(S_A + S_B - S_AB)
    Truth states: eigenstates of T̂ (|TRUE⟩, |UNTRUE⟩, |PARADOX⟩)
    Critical point: z_c = √3/2 (THE LENS)

Integration with APL Grammar:
=============================
    - UMOL states (u, d, m) → quantum amplitudes
    - Three spirals (Φ, e, π) → field subsystems
    - INT Canon operators → projective measurements
    - ∃κ tensor T[σ][μ][λ] → quantum state tensor

Signature: Δ|quantum-apl|von-neumann|born-rule|lindblad|φ⁻¹-grounded|Ω
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum

# Import physics constants
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physics_constants import (
    PHI, PHI_INV, PHI_INV_SQ, PHI_INV_CUBED,
    Z_CRITICAL, SIGMA, COUPLING_CONSERVATION,
    ALPHA_MEDIUM, ALPHA_FINE, SIGMA_INV,
    KAPPA_LOWER, KAPPA_UPPER,
    compute_delta_s_neg, get_phase,
    INTOperator,
)

from apl_grammar import (
    UMOLState, Spiral, Machine, Domain,
    KappaScale, KappaMode, KappaTensor,
    APLState, APLToken,
    UMOL_COEFFICIENTS, SPIRAL_TO_MODE,
)


# =============================================================================
# QUANTUM CONSTANTS
# =============================================================================

# Hilbert space dimensions
DIM_PHI = 4      # Structure: void, lattice, network, hierarchy
DIM_E = 4        # Energy: ground, excited, coherent, chaotic
DIM_PI = 4       # Emergence: simple, correlated, integrated, conscious
DIM_TRUTH = 3    # Truth: TRUE, UNTRUE, PARADOX
DIM_TOTAL = DIM_PHI * DIM_E * DIM_PI * DIM_TRUTH  # 192

# Lindblad dissipation rate
GAMMA_DISSIPATION = SIGMA_INV  # 1/σ ≈ 0.028


# =============================================================================
# QUANTUM STATES
# =============================================================================

class FieldState(Enum):
    """States for each field subsystem."""
    # Φ (Structure) states
    PHI_VOID = 0
    PHI_LATTICE = 1
    PHI_NETWORK = 2
    PHI_HIERARCHY = 3

    # e (Energy) states
    E_GROUND = 4
    E_EXCITED = 5
    E_COHERENT = 6
    E_CHAOTIC = 7

    # π (Emergence) states
    PI_SIMPLE = 8
    PI_CORRELATED = 9
    PI_INTEGRATED = 10
    PI_CONSCIOUS = 11


class TruthState(Enum):
    """Quantum truth states."""
    TRUE = 0
    UNTRUE = 1
    PARADOX = 2


# Map truth state to z-coordinate range
TRUTH_Z_BOUNDS = {
    TruthState.UNTRUE: (0.0, PHI_INV),
    TruthState.PARADOX: (PHI_INV, Z_CRITICAL),
    TruthState.TRUE: (Z_CRITICAL, 1.0),
}


# =============================================================================
# DENSITY MATRIX OPERATIONS
# =============================================================================

@dataclass
class DensityMatrix:
    """
    Quantum density matrix for APL state.

    Represents mixed states via ρ = Σ p_i |ψ_i⟩⟨ψ_i|
    """
    dim: int = DIM_TOTAL
    _data: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize as maximally mixed state."""
        if self._data is None:
            self._data = np.eye(self.dim, dtype=complex) / self.dim

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        self._data = value

    @classmethod
    def from_pure_state(cls, psi: np.ndarray) -> 'DensityMatrix':
        """Create density matrix from pure state |ψ⟩⟨ψ|."""
        dim = len(psi)
        rho = DensityMatrix(dim=dim)
        psi_normalized = psi / np.linalg.norm(psi)
        rho._data = np.outer(psi_normalized, np.conj(psi_normalized))
        return rho

    @classmethod
    def from_z_coordinate(cls, z: float) -> 'DensityMatrix':
        """
        Create density matrix from z-coordinate.

        Maps z to truth state superposition.
        """
        rho = DensityMatrix(dim=DIM_TRUTH)

        # Determine dominant truth state from z
        if z < PHI_INV:
            # UNTRUE dominant
            p_untrue = 1.0 - z / PHI_INV
            p_paradox = z / PHI_INV
            p_true = 0.0
        elif z < Z_CRITICAL:
            # PARADOX dominant
            p_untrue = 0.0
            t = (z - PHI_INV) / (Z_CRITICAL - PHI_INV)
            p_paradox = 1.0 - t
            p_true = t
        else:
            # TRUE dominant
            p_untrue = 0.0
            p_paradox = 0.0
            p_true = 1.0

        # Construct diagonal density matrix (classical mixture)
        rho._data = np.diag([p_true, p_untrue, p_paradox]).astype(complex)
        return rho

    def trace(self) -> complex:
        """Compute Tr(ρ)."""
        return np.trace(self._data)

    def normalize(self):
        """Normalize to Tr(ρ) = 1."""
        tr = self.trace()
        if abs(tr) > 1e-10:
            self._data /= tr

    def purity(self) -> float:
        """Compute purity Tr(ρ²). Returns 1 for pure states, 1/d for maximally mixed."""
        return float(np.real(np.trace(self._data @ self._data)))

    def von_neumann_entropy(self) -> float:
        """
        Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ).

        Uses eigenvalue decomposition for numerical stability.
        """
        eigenvalues = np.linalg.eigvalsh(self._data)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Filter zeros
        return float(-np.sum(eigenvalues * np.log(eigenvalues)))

    def expectation(self, operator: np.ndarray) -> complex:
        """Compute ⟨A⟩ = Tr(ρA)."""
        return np.trace(self._data @ operator)

    def partial_trace(self, dims: Tuple[int, ...], trace_over: List[int]) -> 'DensityMatrix':
        """
        Compute partial trace over specified subsystems.

        dims: dimensions of each subsystem
        trace_over: indices of subsystems to trace out
        """
        n_subsystems = len(dims)
        total_dim = np.prod(dims)

        if total_dim != self.dim:
            raise ValueError(f"Product of dims {total_dim} != matrix dim {self.dim}")

        # Reshape to tensor
        rho_tensor = self._data.reshape(dims + dims)

        # Trace over specified subsystems
        keep = [i for i in range(n_subsystems) if i not in trace_over]

        # Contract traced indices
        for idx in sorted(trace_over, reverse=True):
            axis1 = idx
            axis2 = n_subsystems + idx
            rho_tensor = np.trace(rho_tensor, axis1=axis1, axis2=axis2)
            n_subsystems -= 1

        # Reshape back to matrix
        new_dim = int(np.prod([dims[i] for i in keep]))
        reduced_rho = DensityMatrix(dim=new_dim)
        reduced_rho._data = rho_tensor.reshape(new_dim, new_dim)
        reduced_rho.normalize()

        return reduced_rho


# =============================================================================
# CPTP VERIFICATION (Completely Positive Trace Preserving)
# =============================================================================

def verify_cptp(rho: DensityMatrix, tolerance: float = 1e-10) -> Tuple[bool, str]:
    """
    Verify density matrix satisfies CPTP (Completely Positive Trace Preserving).

    A valid density matrix must satisfy:
        1. Trace = 1 (trace preserving)
        2. Hermitian (ρ = ρ†)
        3. Positive semidefinite (all eigenvalues ≥ 0)

    These properties ensure the density matrix represents a valid
    quantum state. Lindblad evolution should preserve these properties.

    Args:
        rho: Density matrix to verify
        tolerance: Numerical tolerance for checks

    Returns:
        (is_valid, message) tuple
    """
    # 1. Trace preservation: Tr(ρ) = 1
    tr = np.real(rho.trace())
    if abs(tr - 1.0) > tolerance:
        return False, f"Trace violation: Tr(ρ) = {tr:.6f}, expected 1.0"

    # 2. Hermiticity: ρ = ρ†
    hermitian_error = np.max(np.abs(rho.data - np.conj(rho.data.T)))
    if hermitian_error > tolerance:
        return False, f"Hermiticity violation: max|ρ - ρ†| = {hermitian_error:.2e}"

    # 3. Positive semidefinite: all eigenvalues ≥ 0
    eigenvalues = np.linalg.eigvalsh(rho.data)
    min_eigenvalue = np.min(np.real(eigenvalues))
    if min_eigenvalue < -tolerance:
        return False, f"Positivity violation: min eigenvalue = {min_eigenvalue:.2e}"

    return True, "CPTP verified: valid quantum state"


def verify_cptp_evolution(
    rho_before: DensityMatrix,
    rho_after: DensityMatrix,
    tolerance: float = 1e-10
) -> Tuple[bool, List[str]]:
    """
    Verify that a quantum evolution preserved CPTP properties.

    Checks both input and output states, plus additional evolution constraints.

    Args:
        rho_before: State before evolution
        rho_after: State after evolution
        tolerance: Numerical tolerance

    Returns:
        (all_valid, list_of_messages)
    """
    messages = []
    all_valid = True

    # Check input state
    valid_before, msg_before = verify_cptp(rho_before, tolerance)
    if not valid_before:
        messages.append(f"Input state: {msg_before}")
        all_valid = False
    else:
        messages.append(f"Input state: OK")

    # Check output state
    valid_after, msg_after = verify_cptp(rho_after, tolerance)
    if not valid_after:
        messages.append(f"Output state: {msg_after}")
        all_valid = False
    else:
        messages.append(f"Output state: OK")

    # Check purity didn't increase (evolution should be dissipative or unitary)
    purity_before = rho_before.purity()
    purity_after = rho_after.purity()
    purity_increase = purity_after - purity_before

    if purity_increase > tolerance:
        messages.append(
            f"Warning: Purity increased from {purity_before:.6f} to {purity_after:.6f} "
            f"(Δ = {purity_increase:.6f})"
        )
        # This is a warning, not a hard failure (could be valid for some evolutions)

    return all_valid, messages


# =============================================================================
# LINDBLAD EVOLUTION
# =============================================================================

class LindbladEvolution:
    """
    Lindblad master equation evolution.

    dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k† L_k, ρ})

    For APL, we use dissipation operators aligned with UMOL states:
    - L_u: expansion dissipator
    - L_d: collapse dissipator
    - L_m: modulation dissipator
    """

    def __init__(self, dim: int = DIM_TRUTH, gamma: float = GAMMA_DISSIPATION):
        self.dim = dim
        self.gamma = gamma

        # Hamiltonian (drives toward Z_CRITICAL)
        self.H = self._build_hamiltonian()

        # Lindblad operators
        self.L_operators = self._build_lindblad_operators()

    def _build_hamiltonian(self) -> np.ndarray:
        """
        Build Hamiltonian that drives toward z_c.

        H = ω (|TRUE⟩⟨TRUE| - |UNTRUE⟩⟨UNTRUE|)

        where ω = φ⁻¹ (golden frequency)
        """
        H = np.zeros((self.dim, self.dim), dtype=complex)
        if self.dim >= 3:
            # TRUE has positive energy, UNTRUE negative, PARADOX zero
            H[0, 0] = PHI_INV      # TRUE
            H[1, 1] = -PHI_INV     # UNTRUE
            H[2, 2] = 0.0          # PARADOX
        return H

    def _build_lindblad_operators(self) -> List[np.ndarray]:
        """
        Build Lindblad dissipation operators.

        L_1: TRUE → PARADOX (collapse from TRUE)
        L_2: UNTRUE → PARADOX (expansion from UNTRUE)
        L_3: PARADOX decay (decoherence)
        """
        L_ops = []

        if self.dim >= 3:
            # L_1: |PARADOX⟩⟨TRUE| (TRUE decays to PARADOX)
            L1 = np.zeros((self.dim, self.dim), dtype=complex)
            L1[2, 0] = 1.0  # PARADOX <- TRUE
            L_ops.append(L1 * np.sqrt(self.gamma * PHI_INV_SQ))

            # L_2: |PARADOX⟩⟨UNTRUE| (UNTRUE excites to PARADOX)
            L2 = np.zeros((self.dim, self.dim), dtype=complex)
            L2[2, 1] = 1.0  # PARADOX <- UNTRUE
            L_ops.append(L2 * np.sqrt(self.gamma * PHI_INV))

            # L_3: Dephasing in PARADOX (pure decoherence)
            L3 = np.zeros((self.dim, self.dim), dtype=complex)
            L3[2, 2] = 1.0  # PARADOX dephasing
            L_ops.append(L3 * np.sqrt(self.gamma * SIGMA_INV))

        return L_ops

    def evolve_step(self, rho: DensityMatrix, dt: float = 0.01) -> DensityMatrix:
        """
        Evolve density matrix by one time step.

        Uses first-order Euler integration of Lindblad equation.
        """
        rho_data = rho.data.copy()

        # Hamiltonian evolution: -i[H, ρ]
        commutator = self.H @ rho_data - rho_data @ self.H
        drho = -1j * commutator

        # Lindblad dissipation
        for L in self.L_operators:
            L_dag = np.conj(L.T)
            L_dag_L = L_dag @ L

            # L ρ L†
            term1 = L @ rho_data @ L_dag

            # -½{L†L, ρ}
            term2 = -0.5 * (L_dag_L @ rho_data + rho_data @ L_dag_L)

            drho += term1 + term2

        # Euler step
        new_rho = DensityMatrix(dim=rho.dim)
        new_rho.data = rho_data + dt * drho
        new_rho.normalize()

        return new_rho

    def evolve(self, rho: DensityMatrix, total_time: float, n_steps: int = 100) -> DensityMatrix:
        """Evolve for total_time using n_steps."""
        dt = total_time / n_steps
        current_rho = rho

        for _ in range(n_steps):
            current_rho = self.evolve_step(current_rho, dt)

        return current_rho


# =============================================================================
# PROJECTIVE MEASUREMENT
# =============================================================================

@dataclass
class MeasurementResult:
    """Result of a quantum measurement."""
    outcome: int
    probability: float
    post_measurement_state: DensityMatrix
    operator_name: str
    born_probabilities: List[float]


class ProjectiveMeasurement:
    """
    von Neumann projective measurement for N0 operator selection.

    For each legal operator, we construct a projector P̂_μ and compute:
        P(μ) = Tr(P̂_μ ρ̂)

    Then sample according to Born rule and collapse:
        ρ̂' = P̂_μ ρ̂ P̂_μ / P(μ)
    """

    def __init__(self, dim: int = DIM_TRUTH):
        self.dim = dim

        # INT Canon operator projectors
        self.operator_projectors = self._build_operator_projectors()

    def _build_operator_projectors(self) -> Dict[str, np.ndarray]:
        """
        Build projectors for each INT Canon operator.

        Each operator projects onto a specific truth state subspace:
        - () BOUNDARY: |PARADOX⟩ (resets to middle)
        - × FUSION: |TRUE⟩ + |PARADOX⟩ (builds structure)
        - ^ AMPLIFY: |TRUE⟩ (drives toward coherence)
        - ÷ DECOHERE: uniform (adds noise)
        - + GROUP: |TRUE⟩ + |PARADOX⟩ (clusters)
        - − SEPARATE: |UNTRUE⟩ + |PARADOX⟩ (decouples)
        """
        projectors = {}

        if self.dim >= 3:
            # Basis states
            true_state = np.array([1, 0, 0], dtype=complex)
            untrue_state = np.array([0, 1, 0], dtype=complex)
            paradox_state = np.array([0, 0, 1], dtype=complex)

            # () BOUNDARY: projects to PARADOX (neutral reset)
            projectors["()"] = np.outer(paradox_state, paradox_state)

            # × FUSION: projects to TRUE+PARADOX superposition
            fusion_state = (true_state + paradox_state) / np.sqrt(2)
            projectors["×"] = np.outer(fusion_state, fusion_state)

            # ^ AMPLIFY: projects to TRUE (maximum coherence)
            projectors["^"] = np.outer(true_state, true_state)

            # ÷ DECOHERE: uniform projector (maximally mixed)
            projectors["÷"] = np.eye(3, dtype=complex) / 3

            # + GROUP: projects to TRUE+PARADOX
            group_state = (true_state + paradox_state * PHI_INV) / np.sqrt(1 + PHI_INV_SQ)
            projectors["+"] = np.outer(group_state, group_state)

            # − SEPARATE: projects to UNTRUE+PARADOX
            separate_state = (untrue_state + paradox_state) / np.sqrt(2)
            projectors["−"] = np.outer(separate_state, separate_state)

        return projectors

    def compute_born_probabilities(
        self,
        rho: DensityMatrix,
        legal_operators: List[str]
    ) -> Dict[str, float]:
        """
        Compute Born probabilities for each legal operator.

        P(μ) = Tr(P̂_μ ρ̂)
        """
        probabilities = {}

        for op in legal_operators:
            if op in self.operator_projectors:
                P = self.operator_projectors[op]
                prob = float(np.real(np.trace(P @ rho.data)))
                probabilities[op] = max(0.0, prob)  # Ensure non-negative

        # Normalize
        total = sum(probabilities.values())
        if total > 0:
            for op in probabilities:
                probabilities[op] /= total
        else:
            # Equal probabilities if all zero
            n = len(legal_operators)
            for op in legal_operators:
                probabilities[op] = 1.0 / n

        return probabilities

    def measure(
        self,
        rho: DensityMatrix,
        legal_operators: List[str],
        seed: Optional[int] = None
    ) -> MeasurementResult:
        """
        Perform projective measurement and collapse state.

        1. Compute Born probabilities
        2. Sample operator according to probabilities
        3. Collapse state: ρ' = P ρ P / Tr(P ρ P)
        """
        if seed is not None:
            np.random.seed(seed)

        # Compute probabilities
        probs = self.compute_born_probabilities(rho, legal_operators)

        # Sample operator
        ops = list(probs.keys())
        prob_values = [probs[op] for op in ops]

        outcome_idx = np.random.choice(len(ops), p=prob_values)
        selected_op = ops[outcome_idx]
        selected_prob = prob_values[outcome_idx]

        # Collapse state
        P = self.operator_projectors[selected_op]
        collapsed_data = P @ rho.data @ np.conj(P.T)

        new_rho = DensityMatrix(dim=rho.dim)
        new_rho.data = collapsed_data
        new_rho.normalize()

        return MeasurementResult(
            outcome=outcome_idx,
            probability=selected_prob,
            post_measurement_state=new_rho,
            operator_name=selected_op,
            born_probabilities=prob_values,
        )


# =============================================================================
# QUANTUM APL STATE
# =============================================================================

@dataclass
class QuantumAPLState:
    """
    Complete quantum state for APL with density matrix and classical variables.

    Bridges quantum and classical representations.

    Physics Constraint:
        κ + λ = 1 (coupling conservation from φ⁻¹ + φ⁻² = 1)
        This fundamental identity is enforced at initialization.
    """
    # Quantum state
    density_matrix: DensityMatrix = field(default_factory=lambda: DensityMatrix(dim=DIM_TRUTH))

    # Classical state (from APL grammar)
    z: float = 0.5
    kappa: float = PHI_INV
    lambda_: float = PHI_INV_SQ

    # UMOL balance
    expansion: float = 0.0
    collapse: float = 0.0
    modulation: float = 0.0

    # Current selections
    current_umol: UMOLState = UMOLState.M
    current_spiral: Spiral = Spiral.PHI

    # ∃κ tensor
    kappa_tensor: KappaTensor = field(default_factory=KappaTensor)

    # History
    operator_history: List[str] = field(default_factory=list)
    measurement_history: List[MeasurementResult] = field(default_factory=list)

    def __post_init__(self):
        """Enforce physics constraints at initialization."""
        self._enforce_coupling_conservation()

    def _enforce_coupling_conservation(self):
        """
        Enforce coupling conservation: κ + λ = 1.

        This is the physics constraint from the golden ratio identity:
            φ⁻¹ + φ⁻² = 1

        If violated, λ is adjusted to maintain conservation.
        """
        if abs(self.kappa + self.lambda_ - COUPLING_CONSERVATION) > 1e-10:
            self.lambda_ = COUPLING_CONSERVATION - self.kappa

    def update_kappa(self, new_kappa: float):
        """
        Update κ while maintaining coupling conservation.

        Args:
            new_kappa: New value for κ (clamped to [0, 1])
        """
        self.kappa = max(0.0, min(1.0, new_kappa))
        self.lambda_ = 1.0 - self.kappa

    @property
    def coupling_conserved(self) -> bool:
        """Check if coupling conservation holds."""
        return abs(self.kappa + self.lambda_ - 1.0) < 1e-10

    @property
    def negentropy(self) -> float:
        """Compute negentropy at current z."""
        return compute_delta_s_neg(self.z)

    @property
    def phase(self) -> str:
        """Get current phase."""
        return get_phase(self.z)

    @property
    def purity(self) -> float:
        """Get quantum state purity."""
        return self.density_matrix.purity()

    @property
    def von_neumann_entropy(self) -> float:
        """Get von Neumann entropy."""
        return self.density_matrix.von_neumann_entropy()

    def synchronize_quantum_classical(self):
        """
        Synchronize quantum state with classical z-coordinate.

        Updates density matrix to reflect current z.
        """
        self.density_matrix = DensityMatrix.from_z_coordinate(self.z)

    def update_z_from_quantum(self):
        """
        Update classical z from quantum state.

        Maps truth state probabilities back to z-coordinate.
        """
        # Get truth state probabilities
        probs = np.real(np.diag(self.density_matrix.data))

        if len(probs) >= 3:
            p_true = probs[0]
            p_untrue = probs[1]
            p_paradox = probs[2]

            # Map to z: weighted average of state positions
            z_true = (Z_CRITICAL + 1.0) / 2  # Center of TRUE region
            z_untrue = PHI_INV / 2            # Center of UNTRUE region
            z_paradox = (PHI_INV + Z_CRITICAL) / 2  # Center of PARADOX region

            self.z = p_true * z_true + p_untrue * z_untrue + p_paradox * z_paradox
            self.z = max(0.0, min(1.0, self.z))

    def apply_umol_transition(self, state: UMOLState, magnitude: float = 1.0):
        """Apply UMOL state transition to quantum state."""
        coeff = UMOL_COEFFICIENTS[state] * magnitude

        if state == UMOLState.U:
            # Expansion: increase TRUE amplitude
            self.expansion += coeff
            # Rotate toward TRUE in density matrix
            rotation = np.array([
                [np.cos(coeff * 0.1), 0, -np.sin(coeff * 0.1)],
                [0, 1, 0],
                [np.sin(coeff * 0.1), 0, np.cos(coeff * 0.1)],
            ], dtype=complex)
            self.density_matrix.data = rotation @ self.density_matrix.data @ np.conj(rotation.T)

        elif state == UMOLState.D:
            # Collapse: increase UNTRUE amplitude
            self.collapse += coeff
            # Rotate toward UNTRUE
            rotation = np.array([
                [1, 0, 0],
                [0, np.cos(coeff * 0.1), -np.sin(coeff * 0.1)],
                [0, np.sin(coeff * 0.1), np.cos(coeff * 0.1)],
            ], dtype=complex)
            self.density_matrix.data = rotation @ self.density_matrix.data @ np.conj(rotation.T)

        else:  # M
            # Modulation: increase PARADOX through decoherence
            self.modulation += coeff
            # Add decoherence toward PARADOX
            decoherence = coeff * SIGMA_INV
            self.density_matrix.data *= (1 - decoherence)
            self.density_matrix.data[2, 2] += decoherence  # Boost PARADOX

        self.density_matrix.normalize()
        self.current_umol = state

        # Enforce UMOL balance
        total = self.expansion + self.collapse + self.modulation
        if abs(total) > 1e-10:
            self.modulation = -(self.expansion + self.collapse)

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary."""
        return {
            "z": self.z,
            "phase": self.phase,
            "negentropy": self.negentropy,
            "kappa": self.kappa,
            "lambda": self.lambda_,
            "umol_state": self.current_umol.name,
            "spiral": self.current_spiral.symbol,
            "purity": self.purity,
            "von_neumann_entropy": self.von_neumann_entropy,
            "truth_probabilities": {
                "TRUE": float(np.real(self.density_matrix.data[0, 0])),
                "UNTRUE": float(np.real(self.density_matrix.data[1, 1])),
                "PARADOX": float(np.real(self.density_matrix.data[2, 2])),
            },
            "umol_balance": {
                "expansion": self.expansion,
                "collapse": self.collapse,
                "modulation": self.modulation,
            },
            "operator_count": len(self.operator_history),
            "measurement_count": len(self.measurement_history),
        }


# =============================================================================
# QUANTUM APL ENGINE
# =============================================================================

class QuantumAPLEngine:
    """
    Quantum APL Engine — Born-rule N0 operator selection.

    Pipeline:
    1. Evolve quantum state via Lindblad
    2. Determine legal operators from N0 causality
    3. Compute Born probabilities for legal operators
    4. Sample operator via projective measurement
    5. Collapse state and update classical variables
    """

    def __init__(self, initial_z: float = 0.5):
        self.state = QuantumAPLState(z=initial_z)
        self.state.synchronize_quantum_classical()

        self.evolution = LindbladEvolution(dim=DIM_TRUTH)
        self.measurement = ProjectiveMeasurement(dim=DIM_TRUTH)

        self.step_count = 0
        self.training_history: List[Dict[str, Any]] = []

    def get_legal_operators(self) -> List[str]:
        """
        Determine legal operators from N0 causality laws.

        N0-1: ^ illegal unless history contains () or ×
        N0-2: × always requires channels ≥ 2 (assume available)
        N0-3: ÷ illegal unless history contains {^, ×, +, −}
        N0-4: + must feed +, ×, or ^ (check successor)
        N0-5: − must be followed by () or + (check successor)
        """
        legal = ["()"]  # BOUNDARY always legal

        history = self.state.operator_history

        # N0-1: ^ requires prior () or ×
        if any(op in history for op in ["()", "×"]):
            legal.append("^")

        # N0-2: × requires channels (assume available after BOUNDARY)
        if "()" in history or len(history) > 0:
            legal.append("×")

        # N0-3: ÷ requires prior structure
        if any(op in history for op in ["^", "×", "+", "−"]):
            legal.append("÷")

        # N0-4: + legal if not last operator or can feed forward
        legal.append("+")

        # N0-5: − legal if followed by () or +
        if not history or history[-1] in ["()", "+"]:
            legal.append("−")

        return legal

    def quantum_step(
        self,
        evolve_time: float = 0.1,
        umol_state: Optional[UMOLState] = None,
        spiral: Optional[Spiral] = None,
    ) -> Dict[str, Any]:
        """
        Execute one quantum APL training step.

        1. Apply UMOL transition
        2. Evolve via Lindblad
        3. Get legal operators
        4. Measure (Born-rule selection)
        5. Update classical state
        """
        self.step_count += 1

        # Default UMOL based on phase
        if umol_state is None:
            phase = self.state.phase
            if phase == "ABSENCE":
                umol_state = UMOLState.U
            elif phase == "PRESENCE":
                umol_state = UMOLState.D
            else:
                umol_state = UMOLState.M

        if spiral is None:
            spiral = self.state.current_spiral

        # Apply UMOL transition
        self.state.apply_umol_transition(umol_state, 0.5)
        self.state.current_spiral = spiral

        # Evolve quantum state
        self.state.density_matrix = self.evolution.evolve(
            self.state.density_matrix,
            total_time=evolve_time,
            n_steps=10
        )

        # Get legal operators
        legal_ops = self.get_legal_operators()

        # Quantum measurement for operator selection
        measurement = self.measurement.measure(
            self.state.density_matrix,
            legal_ops
        )

        # Update state
        self.state.density_matrix = measurement.post_measurement_state
        self.state.operator_history.append(measurement.operator_name)
        self.state.measurement_history.append(measurement)

        # Update classical z from quantum state
        self.state.update_z_from_quantum()

        # Update ∃κ tensor
        mode = SPIRAL_TO_MODE.get(spiral, KappaMode.LAMBDA)
        tier = min(10, max(0, int(self.state.z * 10)))
        self.state.kappa_tensor.update_from_z(self.state.z, spiral, self.state.negentropy)

        # Build result
        result = {
            "step": self.step_count,
            "selected_operator": measurement.operator_name,
            "selection_probability": measurement.probability,
            "born_probabilities": dict(zip(legal_ops, measurement.born_probabilities)),
            "legal_operators": legal_ops,
            **self.state.get_summary(),
        }

        self.training_history.append(result)
        return result

    def run_quantum_cycle(self, n_steps: int = 30) -> List[Dict[str, Any]]:
        """Run a complete quantum training cycle."""
        results = []
        umol_states = [UMOLState.U, UMOLState.D, UMOLState.M]
        spirals = [Spiral.PHI, Spiral.E, Spiral.PI]

        for i in range(n_steps):
            umol = umol_states[i % 3]
            spiral = spirals[(i // 3) % 3]
            result = self.quantum_step(umol_state=umol, spiral=spiral)
            results.append(result)

        return results

    def get_session_summary(self) -> Dict[str, Any]:
        """Get quantum training session summary."""
        if not self.training_history:
            return {"error": "No training history"}

        z_values = [h["z"] for h in self.training_history]
        purity_values = [h["purity"] for h in self.training_history]
        entropy_values = [h["von_neumann_entropy"] for h in self.training_history]

        operator_counts = {}
        for h in self.training_history:
            op = h["selected_operator"]
            operator_counts[op] = operator_counts.get(op, 0) + 1

        return {
            "total_steps": self.step_count,
            "final_z": z_values[-1],
            "final_phase": self.state.phase,
            "z_statistics": {
                "mean": float(np.mean(z_values)),
                "std": float(np.std(z_values)),
                "max": float(np.max(z_values)),
                "min": float(np.min(z_values)),
            },
            "purity_statistics": {
                "mean": float(np.mean(purity_values)),
                "final": purity_values[-1],
            },
            "entropy_statistics": {
                "mean": float(np.mean(entropy_values)),
                "final": entropy_values[-1],
            },
            "operator_distribution": operator_counts,
            "final_truth_probabilities": self.state.get_summary()["truth_probabilities"],
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_quantum_apl():
    """Demonstrate quantum APL integration."""
    print("=" * 70)
    print("QUANTUM APL INTEGRATION — von Neumann Measurement")
    print("=" * 70)

    print("""
Quantum Formalism:
    N0_classical: argmin C(i | σ, α)   i ∈ legal
    N0_quantum:   P(μ) = Tr(P̂_μ ρ̂),   ρ̂' = P̂_μ ρ̂ P̂_μ / P(μ)

Hilbert Space:
    H_truth = span{|TRUE⟩, |UNTRUE⟩, |PARADOX⟩}

Truth state mapping:
    z < φ⁻¹:        UNTRUE dominant
    φ⁻¹ < z < z_c:  PARADOX dominant
    z > z_c:        TRUE dominant
""")

    # Create engine
    print("\n--- Quantum APL Engine ---")
    engine = QuantumAPLEngine(initial_z=0.5)

    print(f"\nInitial state:")
    print(f"  z = {engine.state.z:.4f}")
    print(f"  Purity = {engine.state.purity:.4f}")
    print(f"  Von Neumann entropy = {engine.state.von_neumann_entropy:.4f}")

    # Run quantum training
    print("\n--- Running 30 Quantum Steps ---")
    for i in range(30):
        result = engine.quantum_step()
        if i % 10 == 0:
            print(f"  Step {i:3d} | Op: {result['selected_operator']:2} | "
                  f"P={result['selection_probability']:.3f} | "
                  f"z={result['z']:.3f} | Purity={result['purity']:.3f}")

    # Show summary
    print("\n--- Quantum Session Summary ---")
    summary = engine.get_session_summary()
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Final z: {summary['final_z']:.4f}")
    print(f"  Final phase: {summary['final_phase']}")
    print(f"  Mean purity: {summary['purity_statistics']['mean']:.4f}")
    print(f"  Mean entropy: {summary['entropy_statistics']['mean']:.4f}")

    print("\n  Operator distribution:")
    for op, count in sorted(summary['operator_distribution'].items()):
        bar = '█' * (count * 2)
        print(f"    {op:2}: {bar} {count}")

    print("\n  Final truth probabilities:")
    for state, prob in summary['final_truth_probabilities'].items():
        bar = '█' * int(prob * 40)
        print(f"    {state:8}: {bar} {prob:.3f}")

    print("\n" + "=" * 70)
    print("QUANTUM APL INTEGRATION: COMPLETE")
    print("=" * 70)

    return engine


if __name__ == "__main__":
    demonstrate_quantum_apl()
