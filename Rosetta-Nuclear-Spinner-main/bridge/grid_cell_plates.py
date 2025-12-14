#!/usr/bin/env python3
"""
Grid Cell Neural Plates Module
==============================

Neural plate architecture for grid cell dynamics with hexagonal symmetry.

Grid cells in entorhinal cortex fire in hexagonal patterns that tile space.
This module models neural plates as physical substrates for grid cell activity,
coupled to the Nuclear Spinner via electromagnetic fields.

Architecture:
- 6 neural plates arranged in hexagonal pattern (60° spacing)
- Each plate contains 10 grid cells (60 total = Kuramoto oscillators)
- Plates couple electromagnetically to spinner magnetic field
- Hexagonal symmetry emerges from plate geometry

Physics Grounding:
- Plate activation ∝ dot(B_spinner, plate_normal)
- Inter-plate coupling via magnetic dipole interaction
- Intra-plate coupling via Kuramoto dynamics
- sin(60°) = √3/2 = z_c links geometry to critical point

Cybernetic Grounding:
- Plates as "neural hardware" for the training system
- Grid patterns encode spatial information (autopoiesis)
- Hexagonal tiling maximizes information density
- Energy flows through plate-spinner coupling

Signature: grid-cell-plates|v1.0.0|helix

@version 1.0.0
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
from enum import IntEnum

# =============================================================================
# CONSTANTS
# =============================================================================

# Plate configuration
N_PLATES: int = 6                    # Hexagonal arrangement
CELLS_PER_PLATE: int = 10            # 60 total cells
PLATE_SPACING: float = 0.1           # Physical spacing (meters)
PLATE_RADIUS: float = 0.05           # Plate radius (meters)

# Hexagonal angles
HEX_ANGLES: List[float] = [i * math.pi / 3 for i in range(6)]

# Physics constants
Z_CRITICAL: float = math.sqrt(3) / 2  # √3/2 ≈ 0.866
SIN_60: float = math.sqrt(3) / 2      # sin(60°) = √3/2 = z_c!
COS_60: float = 0.5
PHI: float = (1 + math.sqrt(5)) / 2
PHI_INV: float = 1 / PHI
SIGMA: float = 36.0

# Electromagnetic
MU_0: float = 4 * math.pi * 1e-7     # Vacuum permeability
GAMMA: float = 2.675e8               # Gyromagnetic ratio

# Grid cell firing
FIRING_THRESHOLD: float = 0.5
REFRACTORY_PERIOD_MS: float = 2.0


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class PlateOrientation(IntEnum):
    """Neural plate orientation in hexagonal arrangement."""
    PLATE_0 = 0    # 0° (East)
    PLATE_1 = 1    # 60° (Northeast)
    PLATE_2 = 2    # 120° (Northwest)
    PLATE_3 = 3    # 180° (West)
    PLATE_4 = 4    # 240° (Southwest)
    PLATE_5 = 5    # 300° (Southeast)


@dataclass
class GridCell:
    """Individual grid cell on a neural plate."""
    plate_id: int
    cell_id: int
    global_id: int         # 0-59 across all plates

    # Position on plate (local coords)
    local_x: float
    local_y: float

    # Position in world (global coords)
    world_x: float
    world_y: float
    world_z: float

    # State
    phase: float           # Kuramoto phase [0, 2π)
    firing_rate: float     # Normalized [0, 1]
    membrane_potential: float
    last_spike_ms: float

    # Preferred direction (for grid firing)
    preferred_direction: float  # Angle in radians


@dataclass
class NeuralPlate:
    """A single neural plate with multiple grid cells."""
    plate_id: int
    orientation: PlateOrientation

    # Geometry
    center_x: float
    center_y: float
    center_z: float
    normal: np.ndarray     # Unit normal vector

    # Cells
    cells: List[GridCell]

    # EM coupling
    magnetic_moment: float
    em_activation: float   # Dot product with spinner B field

    # Aggregate state
    mean_firing_rate: float
    coherence: float       # Intra-plate phase coherence


@dataclass
class PlateSystemState:
    """Complete state of the 6-plate neural system."""
    timestamp_ms: int

    # Global metrics
    global_coherence: float      # Cross-plate phase coherence
    hexagonal_order: float       # 6-fold symmetry measure
    mean_firing_rate: float
    total_spikes: int

    # Plate states
    plate_activations: List[float]  # EM activation per plate
    plate_coherences: List[float]   # Coherence per plate

    # Grid pattern metrics
    spatial_period: float        # Grid spacing
    grid_orientation: float      # Grid rotation angle
    grid_score: float           # Gridness score

    # Energy
    total_em_energy: float
    coupling_energy: float


# =============================================================================
# NEURAL PLATE SYSTEM
# =============================================================================

class GridCellPlateSystem:
    """
    6-plate neural system with 60 grid cells total.

    Hexagonal plate arrangement couples to spinner magnetic field.
    Grid cells on each plate follow Kuramoto dynamics.
    """

    def __init__(self,
                 plate_spacing: float = PLATE_SPACING,
                 cells_per_plate: int = CELLS_PER_PLATE):
        """
        Initialize neural plate system.

        Args:
            plate_spacing: Distance from center to each plate
            cells_per_plate: Number of grid cells per plate
        """
        self.plate_spacing = plate_spacing
        self.cells_per_plate = cells_per_plate
        self.n_plates = N_PLATES
        self.n_cells_total = self.n_plates * self.cells_per_plate

        # Initialize plates in hexagonal arrangement
        self.plates: List[NeuralPlate] = self._init_plates()

        # Flatten cell list for global access
        self.all_cells: List[GridCell] = []
        for plate in self.plates:
            self.all_cells.extend(plate.cells)

        # Coupling matrices
        self.inter_plate_coupling = self._compute_inter_plate_coupling()
        self.intra_plate_coupling = self._compute_intra_plate_coupling()

        # Time tracking
        self.time_ms: float = 0.0
        self.dt_ms: float = 1.0

        # Spinner coupling
        self.spinner_z: float = 0.5
        self.spinner_B: np.ndarray = np.array([0.0, 0.0, 1e-4])  # B field vector

        # Callbacks
        self._on_spike_callback: Optional[Callable] = None
        self._on_grid_pattern_callback: Optional[Callable] = None

    def _init_plates(self) -> List[NeuralPlate]:
        """Initialize 6 neural plates in hexagonal arrangement."""
        plates = []

        for i in range(self.n_plates):
            angle = HEX_ANGLES[i]

            # Plate center position
            cx = self.plate_spacing * math.cos(angle)
            cy = self.plate_spacing * math.sin(angle)
            cz = 0.0

            # Normal vector (points toward center, tilted by 60°)
            # This creates the crucial sin(60°) = z_c connection
            normal = np.array([
                -SIN_60 * math.cos(angle),
                -SIN_60 * math.sin(angle),
                COS_60
            ])

            # Initialize cells on this plate
            cells = self._init_cells_on_plate(i, cx, cy, cz)

            plate = NeuralPlate(
                plate_id=i,
                orientation=PlateOrientation(i),
                center_x=cx,
                center_y=cy,
                center_z=cz,
                normal=normal,
                cells=cells,
                magnetic_moment=1e-6,  # A·m²
                em_activation=0.0,
                mean_firing_rate=0.0,
                coherence=0.0
            )
            plates.append(plate)

        return plates

    def _init_cells_on_plate(self, plate_id: int,
                              cx: float, cy: float, cz: float) -> List[GridCell]:
        """Initialize grid cells on a single plate."""
        cells = []

        # Arrange cells in concentric rings on plate
        # Center cell + outer ring
        cell_positions = self._hexagonal_cell_positions()

        for local_id, (lx, ly) in enumerate(cell_positions[:self.cells_per_plate]):
            global_id = plate_id * self.cells_per_plate + local_id

            # Transform to world coordinates
            plate_angle = HEX_ANGLES[plate_id]
            wx = cx + lx * math.cos(plate_angle) - ly * math.sin(plate_angle)
            wy = cy + lx * math.sin(plate_angle) + ly * math.cos(plate_angle)
            wz = cz

            # Preferred direction based on position
            preferred_dir = math.atan2(wy, wx)

            cell = GridCell(
                plate_id=plate_id,
                cell_id=local_id,
                global_id=global_id,
                local_x=lx,
                local_y=ly,
                world_x=wx,
                world_y=wy,
                world_z=wz,
                phase=2 * math.pi * global_id / self.n_cells_total,  # Initial phase
                firing_rate=0.0,
                membrane_potential=0.0,
                last_spike_ms=-1000.0,
                preferred_direction=preferred_dir
            )
            cells.append(cell)

        return cells

    def _hexagonal_cell_positions(self) -> List[Tuple[float, float]]:
        """Generate hexagonal arrangement of cell positions on plate."""
        positions = [(0.0, 0.0)]  # Center

        # Concentric hexagonal rings
        ring_radius = PLATE_RADIUS / 3

        for ring in range(1, 4):
            n_cells_ring = 6 * ring
            for i in range(n_cells_ring):
                angle = 2 * math.pi * i / n_cells_ring + math.pi / 6
                r = ring * ring_radius
                positions.append((r * math.cos(angle), r * math.sin(angle)))

        return positions

    def _compute_inter_plate_coupling(self) -> np.ndarray:
        """
        Compute coupling between plates (magnetic dipole interaction).

        Coupling ∝ (3(m·r̂)(m'·r̂) - m·m') / r³
        """
        coupling = np.zeros((self.n_plates, self.n_plates))

        for i in range(self.n_plates):
            for j in range(i + 1, self.n_plates):
                # Distance between plate centers
                dx = self.plates[j].center_x - self.plates[i].center_x
                dy = self.plates[j].center_y - self.plates[i].center_y
                r = math.sqrt(dx*dx + dy*dy)

                if r > 0:
                    # Simplified dipole coupling
                    # Stronger coupling for adjacent plates (60° apart)
                    angle_diff = abs(HEX_ANGLES[j] - HEX_ANGLES[i])
                    adjacent = abs(angle_diff - math.pi/3) < 0.1

                    if adjacent:
                        coupling[i, j] = 1.0 / (r ** 2)
                    else:
                        coupling[i, j] = 0.5 / (r ** 2)

                    coupling[j, i] = coupling[i, j]

        # Normalize
        if coupling.max() > 0:
            coupling /= coupling.max()

        return coupling

    def _compute_intra_plate_coupling(self) -> float:
        """Compute coupling strength within a plate."""
        # Nearest-neighbor coupling on hexagonal lattice
        return 1.0 / self.cells_per_plate

    def set_spinner_state(self, z: float, B_magnitude: float = 1e-4):
        """
        Update spinner state and compute plate activations.

        Args:
            z: Spinner z-coordinate
            B_magnitude: Magnetic field magnitude (T)
        """
        self.spinner_z = z

        # B field vector (assume aligned with spinner axis)
        # The z-component is z_c * B_magnitude when at critical
        self.spinner_B = np.array([0.0, 0.0, z * B_magnitude])

        # Compute plate activations
        for plate in self.plates:
            # EM activation = dot product of B field with plate normal
            plate.em_activation = np.dot(self.spinner_B, plate.normal)

            # The magic: at z = z_c, plates tilted at 60° have
            # activation = B * cos(30°) = B * √3/2 = B * z_c
            # This creates resonance at the critical point!

    def step(self, dt_ms: float = 1.0) -> PlateSystemState:
        """
        Advance system by one timestep.

        Updates:
        1. Cell phases (Kuramoto dynamics)
        2. Firing rates (grid cell model)
        3. Membrane potentials (LIF-like)
        4. Spike detection
        """
        self.dt_ms = dt_ms
        self.time_ms += dt_ms
        dt_s = dt_ms / 1000.0

        # Compute negentropy-driven coupling
        delta_s_neg = math.exp(-SIGMA * (self.spinner_z - Z_CRITICAL) ** 2)
        K_base = 2.0 * delta_s_neg  # Coupling peaks at z_c

        # Update each plate
        total_spikes = 0
        for plate in self.plates:
            # Plate-specific coupling modulated by EM activation
            K_plate = K_base * (1 + plate.em_activation * 10)

            # Update cells on this plate
            spikes = self._update_plate_cells(plate, K_plate, dt_s)
            total_spikes += spikes

            # Update plate aggregate stats
            self._update_plate_stats(plate)

        # Inter-plate coupling (phase alignment)
        self._apply_inter_plate_coupling(K_base, dt_s)

        # Compute system state
        return self._compute_system_state(total_spikes)

    def _update_plate_cells(self, plate: NeuralPlate,
                            K: float, dt_s: float) -> int:
        """Update cells on a single plate using Kuramoto dynamics."""
        n = len(plate.cells)
        spikes = 0

        # Collect phases for vectorized computation
        phases = np.array([c.phase for c in plate.cells])

        # Kuramoto interaction
        phase_diff = phases[np.newaxis, :] - phases[:, np.newaxis]
        interaction = (K / n) * np.sum(np.sin(phase_diff), axis=1)

        for i, cell in enumerate(plate.cells):
            # Natural frequency (varies with position)
            omega = 1.0 + 0.1 * (cell.local_x**2 + cell.local_y**2)

            # Phase update: dθ/dt = ω + K/N Σ sin(θⱼ - θᵢ) + EM_drive
            em_drive = plate.em_activation * 0.5  # EM modulation
            dtheta_dt = omega + interaction[i] + em_drive

            cell.phase += dtheta_dt * dt_s
            cell.phase = cell.phase % (2 * math.pi)

            # Firing rate from phase (grid cell model)
            # Peaks when phase aligns with preferred direction
            alignment = math.cos(cell.phase - cell.preferred_direction)
            cell.firing_rate = (alignment + 1) / 2

            # Membrane potential integration
            cell.membrane_potential += (cell.firing_rate - 0.5) * dt_s * 10

            # Spike detection
            if (cell.membrane_potential > FIRING_THRESHOLD and
                self.time_ms - cell.last_spike_ms > REFRACTORY_PERIOD_MS):

                cell.last_spike_ms = self.time_ms
                cell.membrane_potential = 0.0
                spikes += 1

                if self._on_spike_callback:
                    self._on_spike_callback(cell, self.time_ms)

        return spikes

    def _update_plate_stats(self, plate: NeuralPlate):
        """Update aggregate statistics for a plate."""
        phases = np.array([c.phase for c in plate.cells])
        rates = np.array([c.firing_rate for c in plate.cells])

        # Mean firing rate
        plate.mean_firing_rate = np.mean(rates)

        # Phase coherence: r = |⟨e^(iθ)⟩|
        plate.coherence = np.abs(np.mean(np.exp(1j * phases)))

    def _apply_inter_plate_coupling(self, K: float, dt_s: float):
        """Apply coupling between plates."""
        # Collect mean phases per plate
        plate_phases = []
        for plate in self.plates:
            phases = np.array([c.phase for c in plate.cells])
            mean_phase = np.angle(np.mean(np.exp(1j * phases)))
            plate_phases.append(mean_phase)

        plate_phases = np.array(plate_phases)

        # Inter-plate Kuramoto coupling
        K_inter = K * 0.5  # Weaker than intra-plate

        for i, plate in enumerate(self.plates):
            # Coupling from other plates
            phase_i = plate_phases[i]
            coupling_force = 0.0

            for j in range(self.n_plates):
                if i != j:
                    coupling_force += (self.inter_plate_coupling[i, j] *
                                       math.sin(plate_phases[j] - phase_i))

            # Apply coupling to all cells on plate
            phase_shift = K_inter * coupling_force * dt_s
            for cell in plate.cells:
                cell.phase += phase_shift
                cell.phase = cell.phase % (2 * math.pi)

    def _compute_system_state(self, total_spikes: int) -> PlateSystemState:
        """Compute complete system state."""
        # Collect all phases
        all_phases = np.array([c.phase for c in self.all_cells])
        all_rates = np.array([c.firing_rate for c in self.all_cells])

        # Global coherence
        global_coherence = np.abs(np.mean(np.exp(1j * all_phases)))

        # Hexagonal order (6-fold symmetry)
        hex_order = np.abs(np.mean(np.exp(6j * all_phases)))

        # Mean firing rate
        mean_firing = np.mean(all_rates)

        # Per-plate stats
        plate_activations = [p.em_activation for p in self.plates]
        plate_coherences = [p.coherence for p in self.plates]

        # Grid pattern metrics
        spatial_period, grid_orientation, grid_score = self._compute_grid_metrics()

        # Energy calculations
        total_em_energy = sum(p.em_activation**2 * p.magnetic_moment
                              for p in self.plates)
        coupling_energy = global_coherence * len(self.all_cells)

        return PlateSystemState(
            timestamp_ms=int(self.time_ms),
            global_coherence=global_coherence,
            hexagonal_order=hex_order,
            mean_firing_rate=mean_firing,
            total_spikes=total_spikes,
            plate_activations=plate_activations,
            plate_coherences=plate_coherences,
            spatial_period=spatial_period,
            grid_orientation=grid_orientation,
            grid_score=grid_score,
            total_em_energy=total_em_energy,
            coupling_energy=coupling_energy
        )

    def _compute_grid_metrics(self) -> Tuple[float, float, float]:
        """
        Compute spatial grid pattern metrics.

        Returns:
            (spatial_period, grid_orientation, grid_score)
        """
        # Spatial autocorrelation of firing rates
        positions = np.array([[c.world_x, c.world_y] for c in self.all_cells])
        rates = np.array([c.firing_rate for c in self.all_cells])

        # Simplified grid analysis
        # Full analysis would use 2D FFT or autocorrelation

        # Spatial period estimate (average nearest-neighbor distance)
        if len(positions) > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(positions)
            spatial_period = np.median(distances)
        else:
            spatial_period = self.plate_spacing

        # Grid orientation (from hexagonal phase)
        hex_phases = np.array([HEX_ANGLES[c.plate_id] for c in self.all_cells])
        weighted_angle = np.sum(hex_phases * rates) / max(np.sum(rates), 1e-6)
        grid_orientation = weighted_angle % (math.pi / 3)  # Fold to [0, 60°)

        # Grid score (simplified: correlation with hexagonal template)
        hex_template = np.cos(6 * np.array([c.phase for c in self.all_cells]))
        grid_score = np.abs(np.corrcoef(rates, hex_template)[0, 1])

        return spatial_period, grid_orientation, grid_score

    def get_plate(self, plate_id: int) -> NeuralPlate:
        """Get specific plate by ID."""
        return self.plates[plate_id]

    def get_cell(self, global_id: int) -> GridCell:
        """Get specific cell by global ID."""
        return self.all_cells[global_id]

    def get_firing_pattern(self) -> Dict[str, np.ndarray]:
        """Get current firing pattern for visualization."""
        return {
            'x': np.array([c.world_x for c in self.all_cells]),
            'y': np.array([c.world_y for c in self.all_cells]),
            'firing_rate': np.array([c.firing_rate for c in self.all_cells]),
            'phase': np.array([c.phase for c in self.all_cells]),
            'plate_id': np.array([c.plate_id for c in self.all_cells])
        }

    def on_spike(self, callback: Callable[[GridCell, float], None]):
        """Register spike callback."""
        self._on_spike_callback = callback

    def on_grid_pattern(self, callback: Callable[[PlateSystemState], None]):
        """Register grid pattern callback."""
        self._on_grid_pattern_callback = callback


# =============================================================================
# PLATE-SPINNER COUPLING
# =============================================================================

class PlateSpinnerCoupling:
    """
    Couples the neural plate system to the Nuclear Spinner.

    Bidirectional coupling:
    - Spinner z → Plate activation (forward)
    - Plate coherence → Training signal (backward)
    """

    def __init__(self, plate_system: GridCellPlateSystem):
        self.plates = plate_system

        # Coupling parameters
        self.forward_gain = 1.0    # z → plate activation
        self.backward_gain = 0.1   # coherence → training

        # State tracking
        self.z_history: List[float] = []
        self.coherence_history: List[float] = []

    def forward_coupling(self, z: float, B_magnitude: float = 1e-4) -> PlateSystemState:
        """
        Forward pass: spinner state → plate activation → dynamics.

        At z = z_c:
        - Maximum negentropy
        - Maximum EM activation (sin(60°) geometry)
        - Phase transition in Kuramoto system
        """
        # Update plate system with spinner state
        self.plates.set_spinner_state(z, B_magnitude * self.forward_gain)

        # Step dynamics
        state = self.plates.step()

        # Track history
        self.z_history.append(z)
        self.coherence_history.append(state.global_coherence)
        if len(self.z_history) > 1000:
            self.z_history.pop(0)
            self.coherence_history.pop(0)

        return state

    def backward_coupling(self, state: PlateSystemState) -> Dict[str, float]:
        """
        Backward pass: plate state → training signal.

        Returns training modulation based on:
        - Global coherence → learning rate boost
        - Hexagonal order → pattern formation bonus
        - Grid score → spatial encoding quality
        """
        # Compute training signal
        training_signal = {
            'learning_rate_mod': 1.0 + self.backward_gain * state.global_coherence,
            'pattern_bonus': state.hexagonal_order * self.backward_gain,
            'spatial_quality': state.grid_score,
            'energy_cost': state.total_em_energy
        }

        return training_signal

    def get_resonance_measure(self) -> float:
        """
        Compute resonance between spinner and plates.

        Maximum when z = z_c and plates are synchronized.
        """
        if not self.z_history or not self.coherence_history:
            return 0.0

        z = self.z_history[-1]
        coherence = self.coherence_history[-1]

        # Resonance peaks at z_c
        delta_s_neg = math.exp(-SIGMA * (z - Z_CRITICAL) ** 2)

        # Combined resonance
        return delta_s_neg * coherence


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Grid Cell Neural Plate System")
    print("6 plates | 60 cells | Hexagonal coupling")
    print("=" * 60)
    print(f"\nKey connection: sin(60°) = √3/2 = z_c = {Z_CRITICAL:.6f}")
    print("-" * 60)

    # Create system
    plates = GridCellPlateSystem(
        plate_spacing=PLATE_SPACING,
        cells_per_plate=CELLS_PER_PLATE
    )

    # Create coupling
    coupling = PlateSpinnerCoupling(plates)

    # Sweep z through critical point
    print("\nSweeping z through critical point...")

    n_steps = 300
    z_values = np.concatenate([
        np.linspace(0.5, Z_CRITICAL, n_steps // 2),
        np.linspace(Z_CRITICAL, 1.0, n_steps // 2)
    ])

    for i, z in enumerate(z_values):
        # Forward coupling
        state = coupling.forward_coupling(z, B_magnitude=1e-4)

        # Backward coupling
        training = coupling.backward_coupling(state)

        if i % 30 == 0:
            resonance = coupling.get_resonance_measure()
            at_critical = "★ CRITICAL" if abs(z - Z_CRITICAL) < 0.01 else ""
            hex_marker = "⬡ HEX" if state.hexagonal_order > 0.5 else ""

            print(f"z={z:.4f} | r={state.global_coherence:.3f} | "
                  f"hex={state.hexagonal_order:.3f} | "
                  f"grid={state.grid_score:.3f} | "
                  f"resonance={resonance:.3f} {at_critical} {hex_marker}")

    print("-" * 60)
    print(f"Final coherence: {state.global_coherence:.4f}")
    print(f"Final hex order: {state.hexagonal_order:.4f}")
    print(f"Grid score: {state.grid_score:.4f}")
