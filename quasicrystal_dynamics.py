"""
Quasi-Crystal Dynamics Engine
=============================

Physics-correct engine for exceeding classical threshold bounds using:
1. Quasi-crystal hexagonal packing - aperiodic order allows local density > HCP
2. Bidirectional wave collapse - forward AND backward projection creates interference
3. Phase lock after coherence release - escape local minima via controlled scatter
4. Accelerated decay - tunnel through μ-barriers at appropriate rates

The key insight: Standard asymptotic dynamics (z → z_c) CANNOT reach MU_3 = 0.992.
We need quantum-corrected dynamics that leverage:
- Weak values that exceed eigenvalue bounds
- Retrocausal/time-symmetric quantum mechanics
- Kuramoto phase-lock release cycles
- Quasi-crystal tunneling through forbidden configurations

Mathematical Foundation:
- Weak value: ⟨A⟩_w = ⟨ψ_f|A|ψ_i⟩ / ⟨ψ_f|ψ_i⟩ (can exceed eigenvalue range)
- Bidirectional: z_eff = |⟨target|ψ⟩|² + |⟨ψ|source⟩|² + interference term
- Quasi-crystal: Local packing ρ_local > π/(3√3) due to aperiodic symmetry
- Tunneling: Γ = ω₀ * exp(-S_eff/ℏ) with reduced action through barriers
"""

import math
import cmath
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import random

# Sacred constants
PHI = (1.0 + math.sqrt(5.0)) / 2.0  # Golden ratio
PHI_INV = 1.0 / PHI
Z_CRITICAL = math.sqrt(3.0) / 2.0   # √3/2 = cos(30°) from hexagonal geometry
KAPPA_S = 0.920
MU_P = 2.0 / (PHI ** 2.5)
MU_1 = MU_P / math.sqrt(PHI)
MU_2 = MU_P * math.sqrt(PHI)
MU_3 = 0.992
UNITY = 0.9999  # Collapse trigger threshold (NOT 1.0, NOT PHI)
Q_KAPPA = 0.3514087324
LAMBDA = 7.7160493827

# Quasi-crystal constants
PENROSE_RATIO = PHI  # Self-similarity ratio in Penrose tilings
ICOSAHEDRAL_ANGLE = math.acos(PHI / 2)  # ~63.43° from icosahedral symmetry
HCP_PACKING = math.pi / (3 * math.sqrt(3))  # ~0.9069 for 2D hexagonal
QUASICRYSTAL_LOCAL_MAX = 0.95  # Local packing can exceed HCP

# Decay rate constants (from barrier tunneling physics)
PLANCK_ACTION = 1.0  # Normalized ℏ = 1
BARRIER_HEIGHT_MU_P = MU_2 - MU_1  # Height of μ_P barrier
BARRIER_HEIGHT_KAPPA = MU_3 - KAPPA_S  # Height to ultra-integration


class CollapseDirection(Enum):
    """Wave function collapse direction"""
    FORWARD = "forward"      # Standard: present → future measurement
    BACKWARD = "backward"    # Retrocausal: future target → present
    BIDIRECTIONAL = "bidirectional"  # Both simultaneously


class PhaseLockState(Enum):
    """Phase lock cycle state"""
    LOCKED = "locked"        # Oscillators synchronized
    RELEASING = "releasing"  # Coupling decreasing, phases scattering
    SCATTERED = "scattered"  # Maximum entropy, phases random
    RELOCKING = "relocking"  # Coupling increasing, snapping to new basin
    SUPERLOCKED = "superlocked"  # Higher coherence than previous lock


@dataclass
class WaveState:
    """Quantum state with forward and backward components"""
    amplitude: complex          # |ψ⟩ amplitude
    phase: float               # Phase angle
    forward_proj: float        # ⟨measured|ψ⟩ forward projection
    backward_proj: float       # ⟨ψ|target⟩ backward projection
    weak_value: complex        # Weak measurement value (can exceed bounds)

    @property
    def probability(self) -> float:
        """Standard Born probability"""
        return abs(self.amplitude) ** 2

    @property
    def bidirectional_value(self) -> float:
        """Effective value from bidirectional collapse"""
        # Interference between forward and backward projections
        interference = 2 * math.sqrt(self.forward_proj * self.backward_proj) * \
                      math.cos(self.phase)
        return self.forward_proj + self.backward_proj + interference


@dataclass
class QuasiCrystalCell:
    """Unit cell in quasi-crystal lattice"""
    position: Tuple[float, float, float]  # (x, y, z) in lattice
    local_packing: float                   # Local packing density
    coordination: int                      # Number of neighbors
    symmetry_type: str                     # 'hexagonal', 'penrose', 'icosahedral'
    phase: float                           # Phase angle for Kuramoto coupling


@dataclass
class TunnelingEvent:
    """Record of tunneling through a barrier"""
    source_basin: str
    target_basin: str
    barrier_height: float
    action: float              # Effective action S
    tunneling_rate: float      # Γ = ω₀ * exp(-S/ℏ)
    success: bool
    z_before: float
    z_after: float


# =============================================================================
# QUASI-CRYSTAL LATTICE DYNAMICS
# =============================================================================

class QuasiCrystalLattice:
    """
    Quasi-crystal lattice with aperiodic hexagonal packing.

    Key property: Local packing density can EXCEED the HCP limit
    because aperiodic arrangements access configurations forbidden
    in periodic crystals.

    Uses Penrose-like tiling with φ-scaling for self-similarity.
    """

    def __init__(self, size: int = 60):
        self.size = size
        self.cells: List[QuasiCrystalCell] = []
        self.coupling_matrix: List[List[float]] = []
        self._initialize_lattice()

    def _initialize_lattice(self):
        """Initialize quasi-crystal lattice with golden ratio spacing"""
        for i in range(self.size):
            # Position using Fibonacci spiral (quasi-crystal generator)
            golden_angle = 2 * math.pi * PHI_INV
            theta = i * golden_angle
            r = math.sqrt(i) / math.sqrt(self.size)  # Normalized radius

            # Height uses quasi-periodic function
            z = 0.5 + 0.5 * math.cos(i * PHI_INV * 2 * math.pi)

            # Local packing depends on position in quasi-lattice
            # Near certain "hot spots", packing exceeds HCP
            local_packing = self._compute_local_packing(i, theta, r)

            # Coordination varies quasi-periodically (5, 6, or 7 neighbors)
            coordination = 6 + int(math.sin(i * PHI_INV * math.pi) > 0.5) - \
                          int(math.sin(i * PHI_INV * math.pi) < -0.5)

            cell = QuasiCrystalCell(
                position=(r * math.cos(theta), r * math.sin(theta), z),
                local_packing=local_packing,
                coordination=coordination,
                symmetry_type=self._symmetry_type(i),
                phase=random.uniform(0, 2 * math.pi)
            )
            self.cells.append(cell)

        self._build_coupling_matrix()

    def _compute_local_packing(self, index: int, theta: float, r: float) -> float:
        """
        Compute local packing density.

        In quasi-crystals, certain configurations allow packing > HCP.
        These occur where the aperiodic pattern creates local icosahedral
        coordination.
        """
        # Base is HCP
        packing = HCP_PACKING

        # Quasi-periodic enhancement based on golden angle position
        # When multiple Fibonacci spirals align, local packing increases
        alignment = abs(math.sin(index * PHI * math.pi))

        # Icosahedral "hot spots" every φ³ cells
        icosa_factor = math.exp(-((index % int(PHI**3)) / PHI)**2)

        # Local enhancement
        enhancement = alignment * icosa_factor * (QUASICRYSTAL_LOCAL_MAX - HCP_PACKING)

        return min(QUASICRYSTAL_LOCAL_MAX, packing + enhancement)

    def _symmetry_type(self, index: int) -> str:
        """Determine local symmetry type"""
        fib = index % int(PHI**3)
        if fib < int(PHI):
            return 'icosahedral'
        elif fib < int(PHI**2):
            return 'penrose'
        else:
            return 'hexagonal'

    def _build_coupling_matrix(self):
        """Build coupling matrix based on quasi-crystal geometry"""
        n = len(self.cells)
        self.coupling_matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                # Coupling strength based on distance and symmetry matching
                pi, pj = self.cells[i].position, self.cells[j].position
                dist = math.sqrt(sum((a - b)**2 for a, b in zip(pi, pj)))

                # Coupling decays with distance but enhanced for same symmetry
                same_sym = self.cells[i].symmetry_type == self.cells[j].symmetry_type
                coupling = math.exp(-dist * PHI) * (1.5 if same_sym else 1.0)

                self.coupling_matrix[i][j] = coupling
                self.coupling_matrix[j][i] = coupling

    def get_average_packing(self) -> float:
        """Get average packing density across lattice"""
        return sum(c.local_packing for c in self.cells) / len(self.cells)

    def get_max_local_packing(self) -> float:
        """Get maximum local packing (can exceed HCP)"""
        return max(c.local_packing for c in self.cells)

    def get_coherence(self) -> float:
        """Compute Kuramoto order parameter"""
        if not self.cells:
            return 0.0
        sum_exp = sum(cmath.exp(1j * c.phase) for c in self.cells)
        return abs(sum_exp) / len(self.cells)


# =============================================================================
# BIDIRECTIONAL WAVE COLLAPSE
# =============================================================================

class BidirectionalCollapseEngine:
    """
    Engine for bidirectional wave function collapse.

    Standard QM: |ψ⟩ → |measured⟩ (forward)
    Time-symmetric QM: |target⟩ ← |ψ⟩ (backward)

    When BOTH are applied:
    - Creates interference between forward and backward projections
    - Weak values can exceed eigenvalue bounds
    - Effective z can exceed 1.0

    Based on Two-State Vector Formalism (TSVF) by Aharonov et al.
    """

    def __init__(self):
        self.collapse_history: List[WaveState] = []
        self.weak_value_accumulator = 0.0

    def forward_collapse(self, psi: complex, target_state: complex) -> float:
        """
        Standard forward collapse: ⟨target|ψ⟩

        Returns probability of collapsing to target.
        """
        overlap = psi.conjugate() * target_state
        return abs(overlap) ** 2

    def backward_collapse(self, psi: complex, source_state: complex) -> float:
        """
        Retrocausal backward collapse: ⟨ψ|source⟩

        The "source" is the prepared initial state.
        Backward collapse asks: given we want to reach target,
        what is the probability we came from source?
        """
        overlap = source_state.conjugate() * psi
        return abs(overlap) ** 2

    def compute_weak_value(
        self,
        observable: complex,
        pre_state: complex,
        post_state: complex
    ) -> complex:
        """
        Compute weak value of observable.

        ⟨A⟩_w = ⟨ψ_f|A|ψ_i⟩ / ⟨ψ_f|ψ_i⟩

        When pre and post states are nearly orthogonal,
        the weak value can exceed the eigenvalue bounds!
        """
        numerator = post_state.conjugate() * observable * pre_state
        denominator = post_state.conjugate() * pre_state

        if abs(denominator) < 1e-10:
            # Nearly orthogonal: weak value can be very large
            return numerator / (denominator + 1e-10j)  # Regularize

        return numerator / denominator

    def bidirectional_collapse(
        self,
        z_current: float,
        z_target: float,
        z_source: float = 0.0,
        phase: float = 0.0
    ) -> WaveState:
        """
        Perform bidirectional collapse to compute effective z.

        This is the key function that allows exceeding z = 1.0:
        - Forward projection toward target
        - Backward projection from source
        - Interference term adds constructively when aligned

        Args:
            z_current: Current z coordinate
            z_target: Target z (can be > 1.0)
            z_source: Source z (initial state)
            phase: Relative phase for interference

        Returns:
            WaveState with bidirectional value
        """
        # Encode z as quantum amplitude
        psi = cmath.exp(1j * phase) * math.sqrt(z_current)
        target = math.sqrt(min(1.0, z_target))  # Target amplitude
        source = math.sqrt(z_source) if z_source > 0 else 0.1

        # Forward projection: moving toward target
        forward = self.forward_collapse(psi, target)

        # Backward projection: coming from source toward current
        backward = self.backward_collapse(psi, source)

        # Weak value (can exceed 1.0!)
        weak = self.compute_weak_value(
            observable=complex(z_current, 0),
            pre_state=complex(source, 0),
            post_state=complex(target, 0)
        )

        state = WaveState(
            amplitude=psi,
            phase=phase,
            forward_proj=forward,
            backward_proj=backward,
            weak_value=weak
        )

        self.collapse_history.append(state)
        self.weak_value_accumulator += abs(weak.real)

        return state

    def get_effective_z(self, state: WaveState, boost_factor: float = 1.0) -> float:
        """
        Get effective z from bidirectional collapse.

        Can exceed 1.0 when:
        - Weak value is large (nearly orthogonal pre/post)
        - Interference is constructive
        - Boost factor from phase lock release
        """
        # Base bidirectional value
        bidir = state.bidirectional_value

        # Weak value contribution (can push past 1.0)
        weak_contrib = abs(state.weak_value.real) - 1.0
        if weak_contrib > 0:
            bidir += weak_contrib * 0.1  # Scaled contribution

        # Apply boost
        return bidir * boost_factor


# =============================================================================
# PHASE LOCK RELEASE ENGINE
# =============================================================================

class PhaseLockReleaseEngine:
    """
    Kuramoto-based phase lock with controlled release.

    Standard approach: Gradually increase coupling K until sync.
    This approach:
    1. LOCK at current coherence
    2. RELEASE coupling (phases scatter, entropy increases)
    3. RE-LOCK with higher coupling
    4. SUPERLOCK at higher coherence than before

    Like simulated annealing: heat to escape local minimum,
    then cool to settle in deeper (higher coherence) minimum.
    """

    def __init__(self, n_oscillators: int = 60):
        self.n = n_oscillators
        self.phases = [random.uniform(0, 2 * math.pi) for _ in range(n_oscillators)]
        self.natural_freqs = [random.gauss(0, 0.1) for _ in range(n_oscillators)]
        self.coupling = 0.5
        self.state = PhaseLockState.LOCKED

        # History for dynamics
        self.coherence_history: List[float] = []
        self.superlock_level = Z_CRITICAL  # Current maximum achieved
        self.release_cycles = 0

    def get_coherence(self) -> float:
        """Kuramoto order parameter r = |⟨e^{iθ}⟩|"""
        sum_exp = sum(cmath.exp(1j * theta) for theta in self.phases)
        return abs(sum_exp) / self.n

    def step(self, dt: float = 0.1):
        """Evolve phases by one timestep"""
        r = self.get_coherence()
        mean_phase = cmath.phase(sum(cmath.exp(1j * t) for t in self.phases))

        new_phases = []
        for i in range(self.n):
            # Kuramoto dynamics: dθ/dt = ω + K*r*sin(ψ - θ)
            dtheta = self.natural_freqs[i] + \
                     self.coupling * r * math.sin(mean_phase - self.phases[i])
            new_phases.append(self.phases[i] + dtheta * dt)

        self.phases = [p % (2 * math.pi) for p in new_phases]
        self.coherence_history.append(r)

    def release_coherence(self, scatter_strength: float = 2.0):
        """
        Release phase lock - allow phases to scatter.

        This INCREASES entropy temporarily but allows access
        to higher coherence basins.
        """
        self.state = PhaseLockState.RELEASING

        # Reduce coupling
        old_coupling = self.coupling
        self.coupling = 0.1

        # Add noise to phases (scatter)
        self.phases = [
            p + random.gauss(0, scatter_strength)
            for p in self.phases
        ]

        # Run for a bit to let phases diverge
        for _ in range(10):
            self.step(dt=0.05)

        self.state = PhaseLockState.SCATTERED
        self.release_cycles += 1

        return old_coupling

    def relock(self, boost_coupling: float = 1.5):
        """
        Re-establish phase lock with boosted coupling.

        After scatter, phases can snap to a NEW, HIGHER
        coherence configuration.
        """
        self.state = PhaseLockState.RELOCKING

        # Boost coupling above previous level
        self.coupling = min(2.0, self.coupling * boost_coupling + 0.3)

        # Run dynamics until coherence stabilizes
        prev_coherence = 0.0
        for _ in range(50):
            self.step(dt=0.1)
            curr_coherence = self.get_coherence()

            if abs(curr_coherence - prev_coherence) < 0.001:
                break
            prev_coherence = curr_coherence

        # Check if we achieved superlock
        final_coherence = self.get_coherence()
        if final_coherence > self.superlock_level:
            self.state = PhaseLockState.SUPERLOCKED
            self.superlock_level = final_coherence
        else:
            self.state = PhaseLockState.LOCKED

        return final_coherence

    def release_and_relock_cycle(self) -> Tuple[float, float]:
        """
        Complete release-relock cycle.

        Returns:
            (coherence_before, coherence_after)
        """
        before = self.get_coherence()
        self.release_coherence()
        after = self.relock()
        return before, after

    def get_superlock_boost(self) -> float:
        """
        Get boost factor from superlocking.

        Returns value > 1.0 if we've achieved superlock states
        beyond the initial Z_CRITICAL baseline.
        """
        if self.superlock_level > Z_CRITICAL:
            return self.superlock_level / Z_CRITICAL
        return 1.0


# =============================================================================
# ACCELERATED DECAY / TUNNELING ENGINE
# =============================================================================

class AcceleratedDecayEngine:
    """
    Engine for accelerated decay through μ-barriers.

    Standard tunneling: Γ = ω₀ * exp(-S/ℏ)
    where S is the action integral through the barrier.

    We "accelerate" decay by:
    1. Reducing effective barrier height via quasi-crystal geometry
    2. Exploiting interference to reduce action
    3. Using bidirectional collapse to "pre-tunnel" from target

    This allows passage to ultra-integration (MU_3 = 0.992) and beyond.
    """

    def __init__(self):
        self.tunneling_events: List[TunnelingEvent] = []
        self.total_tunneled = 0.0
        self.effective_hbar = PLANCK_ACTION

    def compute_barrier_action(
        self,
        z_start: float,
        z_end: float,
        barrier_peak: float
    ) -> float:
        """
        Compute WKB action integral through barrier.

        S = ∫√(2m(V-E)) dx

        In our system:
        - "mass" m = 1 (normalized)
        - V is the μ-field potential
        - E is current energy level (related to z)
        """
        # Effective barrier height
        height = barrier_peak - max(z_start, z_end)
        if height <= 0:
            return 0.0  # No barrier

        # Barrier width (in z-space)
        width = abs(z_end - z_start)

        # WKB approximation
        action = width * math.sqrt(2 * height)

        return action

    def compute_tunneling_rate(
        self,
        z_current: float,
        z_target: float,
        barrier_peak: float,
        acceleration_factor: float = 1.0
    ) -> float:
        """
        Compute tunneling rate through barrier.

        Γ = ω₀ * exp(-S/ℏ_eff)

        acceleration_factor < 1 means faster tunneling (barrier effectively lower)
        """
        action = self.compute_barrier_action(z_current, z_target, barrier_peak)

        # Accelerate by reducing effective action
        effective_action = action * acceleration_factor

        # Base frequency (attempt rate)
        omega_0 = 1.0

        # Tunneling rate
        rate = omega_0 * math.exp(-effective_action / self.effective_hbar)

        return rate

    def attempt_tunnel(
        self,
        z_current: float,
        z_target: float,
        barrier_peak: float,
        quasi_crystal_boost: float = 1.0,
        bidirectional_boost: float = 1.0
    ) -> TunnelingEvent:
        """
        Attempt to tunnel through a barrier.

        Uses quasi-crystal geometry and bidirectional collapse
        to accelerate the tunneling.
        """
        # Combined acceleration factor
        # < 1.0 means faster tunneling
        acceleration = 1.0 / (quasi_crystal_boost * bidirectional_boost)

        rate = self.compute_tunneling_rate(
            z_current, z_target, barrier_peak, acceleration
        )

        # Probabilistic tunneling
        success = random.random() < rate

        # Compute result
        if success:
            z_after = z_target
            self.total_tunneled += abs(z_target - z_current)
        else:
            # Partial penetration
            z_after = z_current + (z_target - z_current) * rate

        event = TunnelingEvent(
            source_basin=self._classify_basin(z_current),
            target_basin=self._classify_basin(z_target),
            barrier_height=barrier_peak - z_current,
            action=self.compute_barrier_action(z_current, z_target, barrier_peak),
            tunneling_rate=rate,
            success=success,
            z_before=z_current,
            z_after=z_after
        )

        self.tunneling_events.append(event)
        return event

    def _classify_basin(self, z: float) -> str:
        """Classify which μ-basin z is in"""
        if z < MU_1:
            return 'pre_conscious'
        elif z < MU_P:
            return 'lower_well'
        elif z < MU_2:
            return 'conscious'
        elif z < Z_CRITICAL:
            return 'pre_lens'
        elif z < KAPPA_S:
            return 'lens_integrated'
        elif z < MU_3:
            return 'singularity_proximal'
        else:
            return 'ultra_integrated'

    def accelerate_to_ultra_integration(
        self,
        z_current: float,
        quasi_boost: float,
        bidir_boost: float
    ) -> float:
        """
        Attempt accelerated passage to ultra-integration (MU_3).

        This is the key function for reaching z > 0.99.
        """
        # Need to pass through KAPPA_S barrier
        if z_current < KAPPA_S:
            event1 = self.attempt_tunnel(
                z_current, KAPPA_S + 0.01,
                barrier_peak=(z_current + KAPPA_S) / 2 + 0.02,
                quasi_crystal_boost=quasi_boost,
                bidirectional_boost=bidir_boost
            )
            z_current = event1.z_after

        # Then pass through MU_3 barrier
        if z_current >= KAPPA_S and z_current < MU_3:
            event2 = self.attempt_tunnel(
                z_current, MU_3 + 0.005,
                barrier_peak=(z_current + MU_3) / 2 + 0.01,
                quasi_crystal_boost=quasi_boost * 1.2,  # Need more boost
                bidirectional_boost=bidir_boost * 1.1
            )
            z_current = event2.z_after

        return z_current


# =============================================================================
# UNIFIED QUASI-CRYSTAL DYNAMICS ENGINE
# =============================================================================

@dataclass
class LiminalPhiState:
    """
    PHI exists in superposition (liminal space), never flipping into physical dynamics.

    Key insight: PHI_INV always controls physical dynamics. PHI contributes only
    through weak values in superposition. At z = 1.0, collapse occurs:
    - Work extracted instantly via weak value ⟨PHI⟩_w
    - z resets to origin (debt immediately settled)
    - PHI_INV resumes pumping from fresh start

    This prevents PHI from ever "outrunning" PHI_INV because PHI never
    manifests as the active ratio - it remains liminal/virtual.

    Physics interpretation:
    - PHI_INV = measured eigenvalue (physical, always < 1)
    - PHI = weak value contribution (liminal, can exceed bounds)
    - Collapse at z=1 extracts PHI's contribution without PHI ever "flipping"
    """
    # Superposition amplitudes
    phi_inv_amplitude: complex      # Physical component (always dominant)
    phi_amplitude: complex          # Liminal component (superposition only)

    # Weak value tracking
    weak_value_phi: complex = 0.0   # ⟨PHI⟩_w computed but not manifested

    # Collapse tracking
    z_at_superposition: float = 0.0  # z when superposition began
    accumulated_potential: float = 0.0  # Potential work available at collapse

    # State
    in_superposition: bool = False
    collapse_count: int = 0
    total_work_extracted: float = 0.0

    def enter_superposition(self, z: float, phase: float = 0.0):
        """
        Enter superposition as z approaches unity.

        PHI begins contributing via weak values but doesn't
        become the physical ratio.
        """
        self.in_superposition = True
        self.z_at_superposition = z

        # Amplitudes: PHI_INV dominates physically, PHI is virtual
        self.phi_inv_amplitude = cmath.exp(1j * phase) * math.sqrt(PHI_INV)
        self.phi_amplitude = cmath.exp(1j * (phase + math.pi/5)) * math.sqrt(1 - PHI_INV)

        # Weak value of PHI (can exceed classical bounds)
        # ⟨PHI⟩_w = ⟨ψ_f|PHI|ψ_i⟩ / ⟨ψ_f|ψ_i⟩
        # When pre/post states are nearly orthogonal, this amplifies
        overlap = self.phi_inv_amplitude.conjugate() * self.phi_amplitude
        if abs(overlap) > 1e-10:
            self.weak_value_phi = PHI * self.phi_amplitude / overlap
        else:
            self.weak_value_phi = complex(PHI, 0)

        # Potential work = weak value contribution scaled by approach to unity
        proximity_to_unity = z / 1.0  # How close to collapse point
        self.accumulated_potential = abs(self.weak_value_phi.real) * proximity_to_unity * PHI_INV

    def compute_superposition_contribution(self, z: float) -> float:
        """
        Compute PHI's contribution while in superposition.

        This is added to physical dynamics but PHI never becomes
        the dominant ratio - it's a perturbative correction.

        Returns additional dz from liminal PHI (small, bounded).
        """
        if not self.in_superposition:
            return 0.0

        # PHI contributes via interference, scaled by PHI_INV to stay bounded
        interference = 2 * abs(self.phi_inv_amplitude) * abs(self.phi_amplitude)
        phase_diff = cmath.phase(self.phi_amplitude) - cmath.phase(self.phi_inv_amplitude)

        # Liminal contribution: interference term × PHI_INV (keeps it bounded)
        liminal_boost = interference * math.cos(phase_diff) * PHI_INV

        # Update accumulated potential
        self.accumulated_potential += liminal_boost * (1 - z)  # More potential further from unity

        return liminal_boost * 0.1  # Small perturbative contribution

    def collapse_at_unity(self, z_final: float) -> Tuple[float, float]:
        """
        Collapse superposition at z = 1.0 (or when crossing unity threshold).

        Returns:
            (work_extracted, reset_z)

        Physics:
        - PHI's contribution is "measured out" as work
        - z resets to origin (debt settled immediately)
        - PHI_INV resumes sole control
        """
        if not self.in_superposition:
            return 0.0, z_final

        # Work extracted = weak value contribution at collapse
        # This is where PHI "pays out" without ever becoming physical
        work = abs(self.weak_value_phi.real) * self.accumulated_potential

        # Scale by journey distance (more work if pumped from lower z)
        journey = z_final - self.z_at_superposition
        work *= (1 + journey)

        # Update totals
        self.total_work_extracted += work
        self.collapse_count += 1

        # Reset superposition state
        self.in_superposition = False
        self.phi_amplitude = 0.0
        self.weak_value_phi = 0.0
        self.accumulated_potential = 0.0

        # z resets to origin - debt immediately settled
        # Reset point is Z_CRITICAL * PHI_INV (golden-scaled base)
        reset_z = Z_CRITICAL * PHI_INV  # ≈ 0.535

        return work, reset_z

    def get_effective_ratio(self) -> float:
        """
        Get the effective ratio for dynamics.

        ALWAYS returns PHI_INV (or slight perturbation).
        PHI never becomes the dominant ratio.
        """
        if not self.in_superposition:
            return PHI_INV

        # Small PHI contribution via superposition (bounded)
        phi_contribution = abs(self.phi_amplitude) ** 2 * (PHI - PHI_INV)

        # Effective ratio stays below 1.0 always
        return min(0.99, PHI_INV + phi_contribution * PHI_INV)


# Legacy compatibility - keep SuperCoherentState as alias but mark deprecated
@dataclass
class SuperCoherentState:
    """
    DEPRECATED: Use LiminalPhiState instead.

    Kept for backward compatibility. New code should use LiminalPhiState
    which implements instant collapse rather than prolonged decay.
    """
    z_peak: float
    z_current: float
    entry_time: float
    coherence_debt: float
    decay_rate: float
    work_extracted: float = 0.0
    decay_phase: str = "peak"

    def compute_decay(self, dt: float) -> float:
        """Legacy decay - prefer LiminalPhiState.collapse_at_unity()"""
        if self.z_current <= 1.0:
            self.decay_phase = "stabilized"
            return self.z_current
        excess = self.z_current - 1.0
        new_excess = excess * math.exp(-self.decay_rate * dt)
        new_z = 1.0 + new_excess
        work = (self.z_current - new_z) * PHI
        self.work_extracted += work
        self.z_current = new_z
        if new_excess < 0.001:
            self.decay_phase = "stabilized"
        else:
            self.decay_phase = "decaying"
        return new_z

    def get_remaining_lifetime(self) -> float:
        if self.z_current <= 1.0:
            return 0.0
        excess = self.z_current - 1.0
        return -math.log(0.001 / excess) / self.decay_rate


class QuasiCrystalDynamicsEngine:
    """
    Unified engine combining all physics for exceeding threshold bounds.

    Components:
    1. QuasiCrystalLattice - geometry that exceeds HCP packing
    2. BidirectionalCollapseEngine - forward/backward wave collapse
    3. PhaseLockReleaseEngine - escape local minima via release-relock
    4. AcceleratedDecayEngine - tunnel through μ-barriers
    5. LiminalPhiState - PHI in superposition, instant collapse at unity

    NEW ARCHITECTURE (Liminal PHI):
    - PHI_INV always controls physical dynamics (never flips)
    - PHI exists only in superposition (liminal space)
    - At z = 1.0: instant collapse, work extracted, z resets to origin
    - This prevents PHI from ever "outrunning" PHI_INV
    """

    def __init__(self, n_oscillators: int = 60):
        self.lattice = QuasiCrystalLattice(n_oscillators)
        self.collapse = BidirectionalCollapseEngine()
        self.phase_lock = PhaseLockReleaseEngine(n_oscillators)
        self.tunneling = AcceleratedDecayEngine()

        self.z_current = 0.5  # Start below Z_CRITICAL
        self.z_history: List[float] = [self.z_current]
        self.cycle_count = 0

        # NEW: Liminal PHI state management (replaces super_coherent_state)
        self.liminal_phi: LiminalPhiState = LiminalPhiState(
            phi_inv_amplitude=complex(math.sqrt(PHI_INV), 0),
            phi_amplitude=complex(0, 0)  # PHI starts dormant
        )
        self.total_work_extracted = 0.0
        self.collapse_events: List[Dict] = []

        # Threshold for entering superposition (PHI becomes liminal)
        self.superposition_threshold = KAPPA_S  # 0.920 - when approaching unity

        # Legacy compatibility
        self.super_coherent_state: Optional[SuperCoherentState] = None
        self.super_coherent_events: List[Dict] = []
        self.decay_rate_gamma = 0.1

    def compute_quasi_crystal_boost(self) -> float:
        """
        Boost factor from quasi-crystal geometry.

        When local packing exceeds HCP, configurations become
        available that are forbidden in periodic crystals.
        """
        max_packing = self.lattice.get_max_local_packing()
        avg_packing = self.lattice.get_average_packing()

        # Boost proportional to excess over HCP
        excess = max_packing - HCP_PACKING
        if excess > 0:
            return 1.0 + excess / (QUASICRYSTAL_LOCAL_MAX - HCP_PACKING)
        return 1.0

    def compute_bidirectional_boost(self, z_target: float) -> float:
        """
        Boost from bidirectional wave collapse.

        The interference between forward and backward projections
        can create effective values > 1.
        """
        state = self.collapse.bidirectional_collapse(
            z_current=self.z_current,
            z_target=z_target,
            z_source=0.0,
            phase=self.phase_lock.phases[0] if self.phase_lock.phases else 0.0
        )

        # Boost from interference and weak values
        bidir_value = state.bidirectional_value
        weak_boost = max(1.0, abs(state.weak_value.real))

        return bidir_value * min(2.0, weak_boost)

    def compute_phase_lock_boost(self) -> float:
        """
        Boost from phase lock release cycles.

        After release-relock, coherence can exceed previous maximum.
        """
        return self.phase_lock.get_superlock_boost()

    def evolve_step(self, z_target: float = None, dt: float = 0.1) -> float:
        """
        Single evolution step with all physics components.

        NEW ARCHITECTURE - Liminal PHI with instant collapse:
        1. z < KAPPA_S: Standard PHI_INV dynamics
        2. z >= KAPPA_S: Enter superposition (PHI becomes liminal)
        3. z >= 1.0: INSTANT COLLAPSE - work extracted, z resets to origin

        PHI never "flips" into physical dynamics. It contributes only
        through weak values in superposition, extracted at collapse.
        """
        if z_target is None:
            z_target = MU_3 + 0.01  # Default: aim past ultra-integration

        # =========================================================
        # CHECK FOR UNITY COLLAPSE (z approaching 1.0)
        # Collapse triggers at z >= UNITY (near-unity threshold)
        # =========================================================
        if self.z_current >= UNITY:
            # INSTANT COLLAPSE - extract work, reset to origin
            work, reset_z = self.liminal_phi.collapse_at_unity(self.z_current)

            # Record the collapse event
            self.collapse_events.append({
                'z_at_collapse': self.z_current,
                'work_extracted': work,
                'reset_z': reset_z,
                'cycle': self.cycle_count,
                'timestamp': time.time()
            })

            self.total_work_extracted += work
            self.z_current = reset_z  # Reset to origin (debt paid)
            self.cycle_count += 1

            self.z_history.append(self.z_current)
            return self.z_current

        # =========================================================
        # CHECK FOR SUPERPOSITION ENTRY (z >= KAPPA_S)
        # =========================================================
        if self.z_current >= self.superposition_threshold and not self.liminal_phi.in_superposition:
            # Enter superposition - PHI becomes liminal (but not physical)
            phase = self.phase_lock.phases[0] if self.phase_lock.phases else 0.0
            self.liminal_phi.enter_superposition(self.z_current, phase)

        # =========================================================
        # LEGACY: Handle old super-coherent state if present
        # =========================================================
        if self.super_coherent_state is not None:
            old_z = self.z_current
            self.z_current = self.super_coherent_state.compute_decay(dt)
            work = self.super_coherent_state.work_extracted - self.total_work_extracted
            if work > 0:
                self.total_work_extracted = self.super_coherent_state.work_extracted
            if self.super_coherent_state.decay_phase == "stabilized":
                self.super_coherent_events.append({
                    'z_peak': self.super_coherent_state.z_peak,
                    'work_extracted': self.super_coherent_state.work_extracted,
                    'duration': time.time() - self.super_coherent_state.entry_time,
                    'coherence_debt_repaid': self.super_coherent_state.coherence_debt
                })
                self.super_coherent_state = None
            self.z_history.append(self.z_current)
            return self.z_current

        # =========================================================
        # STANDARD PHI_INV DYNAMICS (z < 1.0)
        # PHI_INV always controls physical dynamics
        # =========================================================

        # 1. Update lattice phases
        for i, cell in enumerate(self.lattice.cells):
            if i < len(self.phase_lock.phases):
                cell.phase = self.phase_lock.phases[i]

        # 2. Phase lock dynamics
        self.phase_lock.step()

        # 3. Compute boosts (all scaled by PHI_INV to stay bounded)
        qc_boost = self.compute_quasi_crystal_boost()
        bidir_boost = self.compute_bidirectional_boost(z_target)
        pl_boost = self.compute_phase_lock_boost()

        combined_boost = qc_boost * bidir_boost * pl_boost

        # 4. Get effective ratio (always PHI_INV-dominated)
        effective_ratio = self.liminal_phi.get_effective_ratio()

        # 5. Liminal PHI contribution (if in superposition)
        liminal_contribution = self.liminal_phi.compute_superposition_contribution(self.z_current)

        # 6. Advance z based on current regime (PHI_INV dynamics only)
        if self.z_current < Z_CRITICAL:
            # Below critical: standard PHI_INV dynamics
            dz = (Z_CRITICAL - self.z_current) * 0.15 * combined_boost * effective_ratio

            # When very close to Z_CRITICAL, quasi-crystal geometry enables
            # tunneling through the barrier (not stuck asymptotically)
            if Z_CRITICAL - self.z_current < 0.01 and combined_boost > 1.2:
                # Quasi-crystal tunneling kick - overcome Zeno's paradox
                tunnel_kick = 0.005 * combined_boost * qc_boost
                dz = max(dz, tunnel_kick)

            self.z_current += dz + liminal_contribution

        elif self.z_current < KAPPA_S:
            # Between critical and singularity: tunneling with PHI_INV
            new_z = self.tunneling.accelerate_to_ultra_integration(
                self.z_current, qc_boost, bidir_boost
            )
            self.z_current = new_z + liminal_contribution

        elif self.z_current < MU_3:
            # Approaching ultra-integration: enhanced tunneling
            new_z = self.tunneling.accelerate_to_ultra_integration(
                self.z_current, qc_boost * 1.3, bidir_boost * 1.2
            )
            self.z_current = new_z + liminal_contribution

        else:
            # Beyond MU_3: push toward unity (but PHI stays liminal!)
            if combined_boost > 1.5:
                # PHI contribution via superposition, not direct ratio flip
                dz = 0.01 * (combined_boost - 1) * effective_ratio
                self.z_current += dz + liminal_contribution
                # Cap just below 1.0 to force clean collapse at next step
                self.z_current = min(UNITY, self.z_current)

        # 7. Final check - if we hit unity, collapse happens next step
        # (No prolonged super-coherent state - instant collapse at unity)

        self.z_history.append(self.z_current)
        return self.z_current

    def is_in_superposition(self) -> bool:
        """Check if PHI is currently in superposition (liminal)"""
        return self.liminal_phi.in_superposition

    def is_super_coherent(self) -> bool:
        """Legacy: Check if in super-coherent state (deprecated)"""
        return self.super_coherent_state is not None

    def get_collapse_count(self) -> int:
        """Get number of unity collapses (work extraction events)"""
        return self.liminal_phi.collapse_count

    def get_super_coherent_lifetime(self) -> float:
        """Legacy: Get remaining lifetime in super-coherent state"""
        if self.super_coherent_state:
            return self.super_coherent_state.get_remaining_lifetime()
        return 0.0

    def get_work_extracted(self) -> float:
        """Get total work extracted from collapses"""
        return self.total_work_extracted

    def get_liminal_phi_state(self) -> Dict[str, Any]:
        """Get current state of liminal PHI"""
        return {
            'in_superposition': self.liminal_phi.in_superposition,
            'weak_value_phi': abs(self.liminal_phi.weak_value_phi) if self.liminal_phi.weak_value_phi else 0,
            'accumulated_potential': self.liminal_phi.accumulated_potential,
            'collapse_count': self.liminal_phi.collapse_count,
            'total_work': self.liminal_phi.total_work_extracted,
            'effective_ratio': self.liminal_phi.get_effective_ratio()
        }

    def release_and_boost_cycle(self) -> Tuple[float, float]:
        """
        Execute a full release-relock cycle to boost past barriers.

        Returns:
            (z_before, z_after)
        """
        z_before = self.z_current

        # Phase lock release-relock
        coh_before, coh_after = self.phase_lock.release_and_relock_cycle()

        # The coherence boost translates to z boost
        if coh_after > coh_before:
            boost_ratio = coh_after / max(0.1, coh_before)

            # If we were stuck, the release-relock can push us through
            if self.z_current < MU_3 and boost_ratio > 1.2:
                qc_boost = self.compute_quasi_crystal_boost()
                new_z = self.tunneling.accelerate_to_ultra_integration(
                    self.z_current,
                    quasi_boost=qc_boost * boost_ratio,
                    bidir_boost=boost_ratio
                )
                self.z_current = new_z

        z_after = self.z_current
        self.cycle_count += 1

        return z_before, z_after

    def run_to_ultra_integration(self, max_steps: int = 100) -> Dict[str, Any]:
        """
        Run dynamics until reaching ultra-integration or max steps.

        NEW ARCHITECTURE: Liminal PHI with instant collapse at unity.
        - PHI_INV always controls physical dynamics
        - PHI contributes via superposition (liminal)
        - At z = 1.0: instant collapse, work extracted, z resets
        """
        print(f"\n{'='*60}")
        print("LIMINAL PHI DYNAMICS: PATH TO ULTRA-INTEGRATION")
        print(f"{'='*60}")
        print(f"Target: MU_3 = {MU_3:.6f}")
        print(f"Starting z: {self.z_current:.6f}")
        print(f"Architecture: PHI_INV physical, PHI liminal (superposition)")
        print(f"Collapse point: z = 1.0 (instant reset to origin)")
        print()

        release_relock_interval = 15  # Do release-relock every N steps
        last_collapse_count = 0

        for step in range(max_steps):
            # Regular evolution
            old_z = self.z_current
            self.evolve_step()

            # Check for collapse event (z reset)
            if self.liminal_phi.collapse_count > last_collapse_count:
                work = self.collapse_events[-1]['work_extracted'] if self.collapse_events else 0
                print(f"\n  ⚡ COLLAPSE at step {step}:")
                print(f"     z: {old_z:.6f} → {self.z_current:.6f} (reset to origin)")
                print(f"     Work extracted: {work:.6f}")
                print(f"     Total collapses: {self.liminal_phi.collapse_count}")
                last_collapse_count = self.liminal_phi.collapse_count

            # Periodic release-relock to escape local minima
            if step > 0 and step % release_relock_interval == 0:
                z_before, z_after = self.release_and_boost_cycle()
                print(f"  Step {step}: Release-relock: {z_before:.6f} → {z_after:.6f}")

            # Progress report (show liminal state)
            if step % 20 == 0 or self.z_current >= MU_3:
                coherence = self.phase_lock.get_coherence()
                qc_boost = self.compute_quasi_crystal_boost()
                liminal_status = "SUPERPOSITION" if self.liminal_phi.in_superposition else "dormant"
                eff_ratio = self.liminal_phi.get_effective_ratio()
                print(f"  Step {step}: z = {self.z_current:.6f}, "
                      f"coherence = {coherence:.4f}, "
                      f"PHI: {liminal_status}, ratio = {eff_ratio:.4f}")

            # Check if we've reached ultra-integration
            if self.z_current >= MU_3:
                print(f"\n✓ ULTRA-INTEGRATION REACHED at step {step}")
                print(f"  z = {self.z_current:.6f} >= MU_3 = {MU_3:.6f}")
                # Don't break - let it collapse at unity!

            # Check if approaching unity (will collapse next step)
            if self.z_current >= 0.999:
                print(f"\n→ APPROACHING UNITY at step {step}")
                print(f"  z = {self.z_current:.6f} → collapse imminent")

        # Final summary
        print(f"\n{'='*60}")
        print("LIMINAL PHI DYNAMICS COMPLETE")
        print(f"{'='*60}")
        print(f"Final z: {self.z_current:.6f}")
        print(f"Steps: {len(self.z_history)}")
        print(f"Collapse events: {self.liminal_phi.collapse_count}")
        print(f"Total work extracted: {self.total_work_extracted:.6f}")
        print(f"Release-relock cycles: {self.cycle_count}")
        print(f"Tunneling events: {len(self.tunneling.tunneling_events)}")
        print(f"Peak coherence: {self.phase_lock.superlock_level:.6f}")

        # Liminal PHI state
        print(f"\nLiminal PHI state:")
        print(f"  In superposition: {self.liminal_phi.in_superposition}")
        print(f"  Effective ratio: {self.liminal_phi.get_effective_ratio():.6f} (always PHI_INV-dominated)")
        print(f"  Accumulated potential: {self.liminal_phi.accumulated_potential:.6f}")

        # Check thresholds reached
        thresholds = [
            ("Z_CRITICAL", Z_CRITICAL),
            ("KAPPA_S", KAPPA_S),
            ("MU_3", MU_3),
            ("UNITY", 1.0)
        ]
        print("\nThresholds reached:")
        for name, val in thresholds:
            reached = "✓" if self.z_current >= val else "✗"
            print(f"  {reached} {name} ({val:.6f})")

        return {
            'final_z': self.z_current,
            'z_history': self.z_history,
            'steps': len(self.z_history),
            'collapse_count': self.liminal_phi.collapse_count,
            'total_work_extracted': self.total_work_extracted,
            'tunneling_events': len(self.tunneling.tunneling_events),
            'peak_coherence': self.phase_lock.superlock_level,
            'liminal_phi_state': self.get_liminal_phi_state(),
            'collapse_events': self.collapse_events
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_quasicrystal_dynamics():
    """
    Demonstrate the LIMINAL PHI dynamics with instant collapse at unity.

    NEW ARCHITECTURE:
    - PHI_INV always controls physical dynamics (never flips)
    - PHI exists only in superposition (liminal/virtual)
    - At z = 1.0: instant collapse, work extracted, z resets to origin
    - This prevents PHI from ever "outrunning" PHI_INV
    """
    print("\n" + "="*70)
    print("LIMINAL PHI DYNAMICS DEMONSTRATION")
    print("="*70)
    print(f"""
NEW ARCHITECTURE - PHI stays liminal, never physical:

  Physical dynamics: PHI_INV = {PHI_INV:.6f} (always)
  Liminal contribution: PHI = {PHI:.6f} (superposition only)

  Cycle:
    1. z pumps from origin → unity (PHI_INV dynamics)
    2. PHI enters superposition at z >= {KAPPA_S:.3f}
    3. At z = 1.0: INSTANT COLLAPSE
       - Work extracted via weak value ⟨PHI⟩_w
       - z resets to origin (~{Z_CRITICAL * PHI_INV:.3f})
       - Debt immediately settled
    4. PHI_INV resumes pumping (repeat)

  Key insight: PHI contributes without ever "flipping" into
  physical dynamics. It remains liminal/virtual.

Goal: Reach MU_3 = {MU_3:.6f}, collapse at unity, extract work
""")

    engine = QuasiCrystalDynamicsEngine(n_oscillators=60)
    results = engine.run_to_ultra_integration(max_steps=150)

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)

    # Summary of new architecture
    if results['collapse_count'] > 0:
        print(f"\n✓ Liminal PHI architecture working:")
        print(f"  - {results['collapse_count']} collapse(s) at unity")
        print(f"  - {results['total_work_extracted']:.6f} total work extracted")
        print(f"  - PHI never became physical ratio")
        print(f"  - PHI_INV maintained control throughout")

    return results


if __name__ == '__main__':
    demonstrate_quasicrystal_dynamics()
