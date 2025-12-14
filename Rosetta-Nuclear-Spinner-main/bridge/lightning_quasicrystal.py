#!/usr/bin/env python3
"""
Lightning-Induced Pentagonal Quasicrystal Phase Transition
==========================================================

Models the physics of quasicrystal nucleation via rapid thermal quench,
analogous to how lightning can produce fullerenes and quasicrystals in nature.

Physical Process:
1. PRE-STRIKE: Energy buildup, field gradient increases
2. STRIKE: Rapid energy release (~30,000K plasma)
3. QUENCH: Ultra-fast cooling (10⁶ K/s)
4. NUCLEATION: Pentagonal seeds form at critical undercooling
5. GROWTH: Quasicrystal domains expand with 5-fold symmetry

Pentagon Physics:
- Interior angle: 108° = 3π/5
- sin(36°) = √(10 - 2√5)/4 ≈ 0.588 (vertex angle)
- cos(36°) = φ/2 (golden ratio connection)
- Fat/thin rhombus ratio → φ as tiling expands

Critical Connection:
- Hexagonal (6-fold): sin(60°) = √3/2 = z_c ≈ 0.866
- Pentagonal (5-fold): sin(72°) = √(10 + 2√5)/4 ≈ 0.951
- Transition: 6-fold → 5-fold at high energy (lightning)

Cybernetic Control:
- Spinner z controls "temperature" (kinetic energy)
- ΔS_neg modulates quench rate
- Kuramoto system tracks collective ordering
- Phase transition detected by symmetry order parameter

Hardware Mapping:
- RF coil: Inductive heating (energy input)
- Peltier array: Rapid cooling (quench)
- Hall sensors: Field gradient monitoring
- Spinner: Mechanical energy reservoir

Signature: lightning-quasicrystal|v1.0.0|helix

@version 1.0.0
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
from enum import IntEnum, auto

# =============================================================================
# CONSTANTS
# =============================================================================

# Golden ratio and related
PHI: float = (1 + math.sqrt(5)) / 2          # φ ≈ 1.618034
PHI_INV: float = 1 / PHI                      # φ⁻¹ ≈ 0.618034
PHI_SQ: float = PHI ** 2                      # φ² ≈ 2.618034

# Critical points
Z_CRITICAL_HEX: float = math.sqrt(3) / 2      # √3/2 ≈ 0.866 (hexagonal)
Z_CRITICAL_PENT: float = math.sqrt((10 + 2*math.sqrt(5))) / 4  # ≈ 0.951 (pentagonal)

# Pentagon geometry
PENT_INTERIOR_ANGLE: float = 3 * math.pi / 5  # 108°
PENT_VERTEX_ANGLE: float = math.pi / 5        # 36°
SIN_36: float = math.sqrt(10 - 2*math.sqrt(5)) / 4  # ≈ 0.588
COS_36: float = PHI / 2                       # ≈ 0.809
SIN_72: float = math.sqrt(10 + 2*math.sqrt(5)) / 4  # ≈ 0.951
COS_72: float = (math.sqrt(5) - 1) / 4        # ≈ 0.309

# Penrose rhombus angles
THIN_RHOMBUS_ANGLE: float = math.pi / 5       # 36°
FAT_RHOMBUS_ANGLE: float = 2 * math.pi / 5    # 72°

# Lightning physics (scaled for simulation)
LIGHTNING_TEMP_K: float = 30000.0             # Peak temperature
AMBIENT_TEMP_K: float = 300.0                 # Room temperature
QUENCH_RATE_K_PER_S: float = 1e6              # Cooling rate
NUCLEATION_UNDERCOOL_K: float = 500.0         # Required undercooling

# Thermodynamics
BOLTZMANN_K: float = 1.380649e-23             # J/K
LANDAUER_LIMIT: float = BOLTZMANN_K * 300 * math.log(2)  # ~2.87e-21 J

# Timing (milliseconds)
PRESTRIKE_DURATION_MS: float = 100.0
STRIKE_DURATION_MS: float = 0.1               # 100 μs
QUENCH_DURATION_MS: float = 10.0
NUCLEATION_DURATION_MS: float = 50.0
GROWTH_DURATION_MS: float = 500.0

# Negentropy
SIGMA: float = 36.0                           # Gaussian width


# =============================================================================
# ENUMS
# =============================================================================

class LightningPhase(IntEnum):
    """Phases of lightning-induced quasicrystal formation."""
    IDLE = 0
    PRE_STRIKE = 1      # Energy buildup
    STRIKE = 2          # Plasma discharge
    QUENCH = 3          # Rapid cooling
    NUCLEATION = 4      # Seed crystal formation
    GROWTH = 5          # Domain expansion
    STABLE = 6          # Quasicrystal formed


class SymmetryOrder(IntEnum):
    """Crystal symmetry types."""
    DISORDERED = 0      # Amorphous/liquid
    SIXFOLD = 6         # Hexagonal
    FIVEFOLD = 5        # Pentagonal (quasicrystal)
    FOURFOLD = 4        # Tetragonal
    THREEFOLD = 3       # Trigonal


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ThermalState:
    """Thermal state of the system."""
    temperature_K: float
    dT_dt: float                  # Cooling/heating rate
    thermal_gradient: float       # Spatial gradient
    entropy: float
    gibbs_free_energy: float


@dataclass
class FieldState:
    """Electromagnetic field state for strike."""
    E_magnitude: float            # Electric field (V/m)
    B_magnitude: float            # Magnetic field (T)
    field_gradient: float         # Gradient (T/m)
    charge_buildup: float         # Normalized [0, 1]
    discharge_current: float      # Amps


@dataclass
class NucleationSeed:
    """A single quasicrystal nucleation seed."""
    x: float
    y: float
    radius: float
    symmetry: SymmetryOrder
    orientation: float            # Rotation angle
    growth_rate: float
    stability: float              # [0, 1]


@dataclass
class QuasicrystalState:
    """State of the quasicrystal domain."""
    fat_tile_count: int
    thin_tile_count: int
    tile_ratio: float             # Should approach φ
    phi_deviation: float          # |ratio - φ|
    pentagonal_order: float       # 5-fold order parameter
    domain_size: float            # Characteristic size
    defect_density: float


@dataclass
class LightningStrikeState:
    """Complete state of lightning-quasicrystal system."""
    timestamp_ms: float
    phase: LightningPhase
    phase_progress: float         # [0, 1] within current phase

    # Thermal
    thermal: ThermalState

    # EM field
    field: FieldState

    # Nucleation
    seeds: List[NucleationSeed]
    total_seeds: int

    # Quasicrystal
    quasicrystal: QuasicrystalState

    # Spinner coupling
    spinner_z: float
    negentropy: float

    # Energy accounting
    energy_input: float
    energy_dissipated: float


# =============================================================================
# LIGHTNING STRIKE SIMULATOR
# =============================================================================

class LightningQuasicrystalSystem:
    """
    Simulates lightning-induced pentagonal quasicrystal formation.

    The system models:
    1. Electromagnetic field buildup (pre-strike)
    2. Plasma discharge (strike)
    3. Rapid thermal quench
    4. Nucleation of pentagonal seeds
    5. Quasicrystal domain growth

    Coupled to Nuclear Spinner:
    - z → effective temperature
    - ΔS_neg → quench efficiency
    - Kuramoto coherence → collective ordering
    """

    def __init__(self,
                 domain_size: float = 1.0,
                 seed_density: float = 10.0,
                 dt_ms: float = 0.1):
        """
        Initialize lightning-quasicrystal system.

        Args:
            domain_size: Simulation domain size (normalized)
            seed_density: Seeds per unit area at nucleation
            dt_ms: Timestep in milliseconds
        """
        self.domain_size = domain_size
        self.seed_density = seed_density
        self.dt_ms = dt_ms

        # State
        self.time_ms: float = 0.0
        self.phase = LightningPhase.IDLE
        self.phase_start_ms: float = 0.0

        # Thermal state
        self.temperature = AMBIENT_TEMP_K
        self.peak_temperature = AMBIENT_TEMP_K

        # Field state
        self.charge_buildup = 0.0
        self.discharge_current = 0.0

        # Nucleation
        self.seeds: List[NucleationSeed] = []
        self.max_seeds = int(seed_density * domain_size ** 2)

        # Quasicrystal tiles
        self.fat_tiles = 0
        self.thin_tiles = 0

        # Spinner coupling
        self.spinner_z = 0.5
        self.kuramoto_r = 0.0

        # Energy tracking
        self.total_energy_input = 0.0
        self.total_energy_dissipated = 0.0

        # History
        self._temperature_history: List[float] = []
        self._order_history: List[float] = []

        # Callbacks
        self._on_strike_callback: Optional[Callable] = None
        self._on_nucleation_callback: Optional[Callable] = None
        self._on_quasicrystal_callback: Optional[Callable] = None

    def trigger_strike(self):
        """Initiate a lightning strike sequence."""
        if self.phase != LightningPhase.IDLE:
            return  # Already in progress

        self.phase = LightningPhase.PRE_STRIKE
        self.phase_start_ms = self.time_ms
        self.charge_buildup = 0.0
        print(f"[LIGHTNING] Strike initiated at t={self.time_ms:.1f}ms")

    def set_spinner_state(self, z: float, kuramoto_r: float = 0.0):
        """
        Set spinner state for coupling.

        Args:
            z: Spinner z-coordinate [0, 1]
            kuramoto_r: Kuramoto order parameter [0, 1]
        """
        self.spinner_z = z
        self.kuramoto_r = kuramoto_r

    def step(self, dt_ms: Optional[float] = None) -> LightningStrikeState:
        """
        Advance simulation by one timestep.

        Args:
            dt_ms: Optional timestep override

        Returns:
            Current system state
        """
        if dt_ms is None:
            dt_ms = self.dt_ms

        self.time_ms += dt_ms

        # Update based on current phase
        if self.phase == LightningPhase.IDLE:
            self._step_idle(dt_ms)
        elif self.phase == LightningPhase.PRE_STRIKE:
            self._step_prestrike(dt_ms)
        elif self.phase == LightningPhase.STRIKE:
            self._step_strike(dt_ms)
        elif self.phase == LightningPhase.QUENCH:
            self._step_quench(dt_ms)
        elif self.phase == LightningPhase.NUCLEATION:
            self._step_nucleation(dt_ms)
        elif self.phase == LightningPhase.GROWTH:
            self._step_growth(dt_ms)
        elif self.phase == LightningPhase.STABLE:
            self._step_stable(dt_ms)

        # Record history
        self._temperature_history.append(self.temperature)
        if len(self._temperature_history) > 10000:
            self._temperature_history.pop(0)

        return self._build_state()

    def _step_idle(self, dt_ms: float):
        """Idle state - system at ambient."""
        # Relax temperature to ambient
        self.temperature += (AMBIENT_TEMP_K - self.temperature) * 0.01

    def _step_prestrike(self, dt_ms: float):
        """Pre-strike - charge buildup."""
        elapsed = self.time_ms - self.phase_start_ms

        # Exponential charge buildup
        tau = PRESTRIKE_DURATION_MS / 3
        self.charge_buildup = 1 - math.exp(-elapsed / tau)

        # Field gradient increases
        # Couple to spinner z: higher z = faster buildup
        buildup_boost = 1 + self.spinner_z

        # Check transition to strike
        if elapsed >= PRESTRIKE_DURATION_MS or self.charge_buildup > 0.95:
            self.phase = LightningPhase.STRIKE
            self.phase_start_ms = self.time_ms
            self.discharge_current = self.charge_buildup * 30000  # 30 kA typical

            if self._on_strike_callback:
                self._on_strike_callback(self._build_state())

            print(f"[LIGHTNING] ⚡ STRIKE! Current={self.discharge_current:.0f}A")

    def _step_strike(self, dt_ms: float):
        """Strike phase - rapid heating."""
        elapsed = self.time_ms - self.phase_start_ms

        # Ultra-rapid heating
        # T(t) = T_peak * exp(-(t/τ)²) for Gaussian pulse
        tau = STRIKE_DURATION_MS / 2
        pulse = math.exp(-(elapsed / tau) ** 2)

        # Peak temperature proportional to discharge
        self.peak_temperature = LIGHTNING_TEMP_K * (self.discharge_current / 30000)
        self.temperature = AMBIENT_TEMP_K + (self.peak_temperature - AMBIENT_TEMP_K) * pulse

        # Energy input (Joule heating)
        power = self.discharge_current ** 2 * 1e-6  # Scaled resistance
        self.total_energy_input += power * dt_ms * 1e-3

        # Charge depletes
        self.charge_buildup *= 0.9
        self.discharge_current *= 0.8

        # Transition to quench
        if elapsed >= STRIKE_DURATION_MS * 3:
            self.phase = LightningPhase.QUENCH
            self.phase_start_ms = self.time_ms
            print(f"[LIGHTNING] Peak T={self.peak_temperature:.0f}K, entering quench")

    def _step_quench(self, dt_ms: float):
        """Quench phase - rapid cooling."""
        elapsed = self.time_ms - self.phase_start_ms

        # Exponential cooling with negentropy-modulated rate
        # Higher ΔS_neg = more efficient quench
        delta_s_neg = math.exp(-SIGMA * (self.spinner_z - Z_CRITICAL_HEX) ** 2)
        quench_efficiency = 1 + delta_s_neg

        # Cooling rate: dT/dt = -k(T - T_ambient)
        k = (QUENCH_RATE_K_PER_S / 1000) * quench_efficiency  # K/ms
        dT = -k * (self.temperature - AMBIENT_TEMP_K) * dt_ms
        self.temperature += dT

        # Energy dissipation
        self.total_energy_dissipated += abs(dT) * 1e-6  # Scaled

        # Check for nucleation threshold
        # Nucleation occurs at critical undercooling below liquidus
        liquidus_temp = 1000.0  # Simplified
        undercooling = liquidus_temp - self.temperature

        if undercooling >= NUCLEATION_UNDERCOOL_K or elapsed >= QUENCH_DURATION_MS:
            self.phase = LightningPhase.NUCLEATION
            self.phase_start_ms = self.time_ms
            print(f"[LIGHTNING] Undercooling={undercooling:.0f}K, nucleating")

    def _step_nucleation(self, dt_ms: float):
        """Nucleation phase - seed crystal formation."""
        elapsed = self.time_ms - self.phase_start_ms

        # Nucleation rate depends on undercooling and spinner state
        # Higher z (near pentagonal critical) favors 5-fold seeds
        pent_preference = math.exp(-20 * (self.spinner_z - Z_CRITICAL_PENT) ** 2)
        hex_preference = math.exp(-20 * (self.spinner_z - Z_CRITICAL_HEX) ** 2)

        # Stochastic nucleation
        if len(self.seeds) < self.max_seeds:
            nucleation_prob = 0.1 * dt_ms * (1 + self.kuramoto_r)

            if np.random.random() < nucleation_prob:
                # Determine symmetry based on spinner state
                if np.random.random() < pent_preference / (pent_preference + hex_preference + 0.1):
                    symmetry = SymmetryOrder.FIVEFOLD
                elif np.random.random() < hex_preference / (hex_preference + 0.1):
                    symmetry = SymmetryOrder.SIXFOLD
                else:
                    symmetry = SymmetryOrder.DISORDERED

                seed = NucleationSeed(
                    x=np.random.uniform(0, self.domain_size),
                    y=np.random.uniform(0, self.domain_size),
                    radius=0.01,
                    symmetry=symmetry,
                    orientation=np.random.uniform(0, 2*np.pi),
                    growth_rate=0.01 * (1 + 0.5 * (symmetry == SymmetryOrder.FIVEFOLD)),
                    stability=0.5 + 0.5 * pent_preference if symmetry == SymmetryOrder.FIVEFOLD else 0.5
                )
                self.seeds.append(seed)

                if self._on_nucleation_callback and symmetry == SymmetryOrder.FIVEFOLD:
                    self._on_nucleation_callback(seed)

        # Continued cooling
        self.temperature += (AMBIENT_TEMP_K - self.temperature) * 0.01

        # Transition to growth
        if elapsed >= NUCLEATION_DURATION_MS or len(self.seeds) >= self.max_seeds * 0.8:
            self.phase = LightningPhase.GROWTH
            self.phase_start_ms = self.time_ms
            pent_count = sum(1 for s in self.seeds if s.symmetry == SymmetryOrder.FIVEFOLD)
            print(f"[LIGHTNING] Nucleated {len(self.seeds)} seeds ({pent_count} pentagonal)")

    def _step_growth(self, dt_ms: float):
        """Growth phase - domain expansion."""
        elapsed = self.time_ms - self.phase_start_ms

        # Grow each seed
        for seed in self.seeds:
            if seed.stability > 0.3:
                # Growth rate modulated by stability and temperature
                T_factor = max(0, 1 - (self.temperature - AMBIENT_TEMP_K) / 1000)
                growth = seed.growth_rate * T_factor * dt_ms

                seed.radius += growth

                # Pentagonal seeds generate tiles
                if seed.symmetry == SymmetryOrder.FIVEFOLD:
                    # Penrose tiling: each growth step adds tiles
                    # Ratio should approach φ
                    new_fat = int(growth * 100 * PHI)
                    new_thin = int(growth * 100)
                    self.fat_tiles += new_fat
                    self.thin_tiles += new_thin

        # Competition: overlapping seeds
        self._resolve_seed_competition()

        # Temperature relaxation
        self.temperature += (AMBIENT_TEMP_K - self.temperature) * 0.005

        # Check for stable state
        pent_order = self._compute_pentagonal_order()
        self._order_history.append(pent_order)

        if elapsed >= GROWTH_DURATION_MS:
            self.phase = LightningPhase.STABLE
            self.phase_start_ms = self.time_ms

            if self._on_quasicrystal_callback:
                self._on_quasicrystal_callback(self._build_state())

            tile_ratio = self.fat_tiles / max(self.thin_tiles, 1)
            print(f"[LIGHTNING] Quasicrystal stable: order={pent_order:.3f}, "
                  f"tile ratio={tile_ratio:.4f} (φ={PHI:.4f})")

    def _step_stable(self, dt_ms: float):
        """Stable state - quasicrystal formed."""
        # Minor thermal fluctuations
        self.temperature = AMBIENT_TEMP_K + np.random.normal(0, 1)

        # Slow annealing improves order
        if np.random.random() < 0.001:
            # Defect annihilation
            for seed in self.seeds:
                if seed.symmetry == SymmetryOrder.FIVEFOLD:
                    seed.stability = min(1.0, seed.stability + 0.01)

    def _resolve_seed_competition(self):
        """Resolve competition between overlapping seeds."""
        # Larger seeds absorb smaller ones of lower stability
        to_remove = []

        for i, seed_i in enumerate(self.seeds):
            for j, seed_j in enumerate(self.seeds):
                if i >= j:
                    continue

                # Check overlap
                dx = seed_i.x - seed_j.x
                dy = seed_i.y - seed_j.y
                dist = math.sqrt(dx*dx + dy*dy)

                if dist < seed_i.radius + seed_j.radius:
                    # Larger/more stable wins
                    score_i = seed_i.radius * seed_i.stability
                    score_j = seed_j.radius * seed_j.stability

                    if score_i > score_j:
                        to_remove.append(j)
                        seed_i.radius += seed_j.radius * 0.5
                    else:
                        to_remove.append(i)
                        seed_j.radius += seed_i.radius * 0.5

        # Remove absorbed seeds
        for idx in sorted(set(to_remove), reverse=True):
            if idx < len(self.seeds):
                self.seeds.pop(idx)

    def _compute_pentagonal_order(self) -> float:
        """
        Compute 5-fold symmetry order parameter.

        ψ₅ = |⟨e^(5iθ)⟩| where θ is seed orientation
        """
        if not self.seeds:
            return 0.0

        pent_seeds = [s for s in self.seeds if s.symmetry == SymmetryOrder.FIVEFOLD]
        if not pent_seeds:
            return 0.0

        # 5-fold order parameter (use cmath for complex exponential)
        import cmath
        phases = [cmath.exp(5j * s.orientation) for s in pent_seeds]
        weights = [s.radius * s.stability for s in pent_seeds]
        total_weight = sum(weights)

        if total_weight == 0:
            return 0.0

        weighted_sum = sum(w * p for w, p in zip(weights, phases))
        return abs(weighted_sum / total_weight)

    def _build_state(self) -> LightningStrikeState:
        """Build complete state snapshot."""
        # Phase progress
        elapsed = self.time_ms - self.phase_start_ms
        durations = {
            LightningPhase.IDLE: 1000.0,
            LightningPhase.PRE_STRIKE: PRESTRIKE_DURATION_MS,
            LightningPhase.STRIKE: STRIKE_DURATION_MS * 3,
            LightningPhase.QUENCH: QUENCH_DURATION_MS,
            LightningPhase.NUCLEATION: NUCLEATION_DURATION_MS,
            LightningPhase.GROWTH: GROWTH_DURATION_MS,
            LightningPhase.STABLE: 1000.0,
        }
        progress = min(1.0, elapsed / durations.get(self.phase, 1000.0))

        # Thermal state
        dT_dt = 0.0
        if len(self._temperature_history) >= 2:
            dT_dt = (self._temperature_history[-1] - self._temperature_history[-2]) / self.dt_ms * 1000

        thermal = ThermalState(
            temperature_K=self.temperature,
            dT_dt=dT_dt,
            thermal_gradient=abs(dT_dt) / 1000,
            entropy=BOLTZMANN_K * math.log(max(1, self.temperature / AMBIENT_TEMP_K)),
            gibbs_free_energy=-BOLTZMANN_K * self.temperature * math.log(max(1, len(self.seeds) + 1))
        )

        # Field state
        field = FieldState(
            E_magnitude=self.charge_buildup * 3e6,  # ~3 MV/m for lightning
            B_magnitude=self.discharge_current * 4e-7,  # μ₀I/2πr
            field_gradient=self.charge_buildup * 1e3,
            charge_buildup=self.charge_buildup,
            discharge_current=self.discharge_current
        )

        # Quasicrystal state
        tile_ratio = self.fat_tiles / max(self.thin_tiles, 1)
        pent_order = self._compute_pentagonal_order()

        quasicrystal = QuasicrystalState(
            fat_tile_count=self.fat_tiles,
            thin_tile_count=self.thin_tiles,
            tile_ratio=tile_ratio,
            phi_deviation=abs(tile_ratio - PHI),
            pentagonal_order=pent_order,
            domain_size=sum(s.radius for s in self.seeds),
            defect_density=sum(1 for s in self.seeds if s.stability < 0.5) / max(len(self.seeds), 1)
        )

        # Negentropy
        delta_s_neg = math.exp(-SIGMA * (self.spinner_z - Z_CRITICAL_HEX) ** 2)

        return LightningStrikeState(
            timestamp_ms=self.time_ms,
            phase=self.phase,
            phase_progress=progress,
            thermal=thermal,
            field=field,
            seeds=self.seeds.copy(),
            total_seeds=len(self.seeds),
            quasicrystal=quasicrystal,
            spinner_z=self.spinner_z,
            negentropy=delta_s_neg,
            energy_input=self.total_energy_input,
            energy_dissipated=self.total_energy_dissipated
        )

    def reset(self):
        """Reset system to initial state."""
        self.time_ms = 0.0
        self.phase = LightningPhase.IDLE
        self.phase_start_ms = 0.0
        self.temperature = AMBIENT_TEMP_K
        self.peak_temperature = AMBIENT_TEMP_K
        self.charge_buildup = 0.0
        self.discharge_current = 0.0
        self.seeds.clear()
        self.fat_tiles = 0
        self.thin_tiles = 0
        self.total_energy_input = 0.0
        self.total_energy_dissipated = 0.0
        self._temperature_history.clear()
        self._order_history.clear()

    def on_strike(self, callback: Callable[[LightningStrikeState], None]):
        """Register strike event callback."""
        self._on_strike_callback = callback

    def on_nucleation(self, callback: Callable[[NucleationSeed], None]):
        """Register nucleation event callback."""
        self._on_nucleation_callback = callback

    def on_quasicrystal(self, callback: Callable[[LightningStrikeState], None]):
        """Register quasicrystal formation callback."""
        self._on_quasicrystal_callback = callback


# =============================================================================
# PENTAGONAL TILING GENERATOR
# =============================================================================

class PenroseTilingGenerator:
    """
    Generates Penrose P2 (kite-dart) or P3 (rhombus) tilings.

    Uses subdivision rules to create aperiodic tilings with 5-fold symmetry.
    The fat/thin tile ratio approaches φ as subdivision continues.
    """

    def __init__(self, tiling_type: str = 'P3'):
        """
        Initialize tiling generator.

        Args:
            tiling_type: 'P2' (kite-dart) or 'P3' (rhombus)
        """
        self.tiling_type = tiling_type
        self.tiles: List[Dict] = []
        self.generation = 0

    def initialize_seed(self, center: Tuple[float, float] = (0, 0),
                        radius: float = 1.0):
        """
        Create initial seed configuration (decagon for P3).

        Args:
            center: Center point
            radius: Size of initial seed
        """
        self.tiles.clear()
        self.generation = 0

        cx, cy = center

        # Create 10 triangles forming a decagon
        for i in range(10):
            angle = i * math.pi / 5  # 36° increments
            next_angle = (i + 1) * math.pi / 5

            # Alternating thin (36°) and fat (72°) rhombus halves
            is_fat = i % 2 == 0

            p0 = (cx, cy)
            p1 = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            p2 = (cx + radius * math.cos(next_angle), cy + radius * math.sin(next_angle))

            self.tiles.append({
                'type': 'fat' if is_fat else 'thin',
                'vertices': [p0, p1, p2],
                'orientation': angle
            })

    def subdivide(self):
        """
        Apply subdivision rules to increase tiling resolution.

        Subdivision preserves the tile ratio approaching φ.
        """
        new_tiles = []
        self.generation += 1

        for tile in self.tiles:
            subdivided = self._subdivide_tile(tile)
            new_tiles.extend(subdivided)

        self.tiles = new_tiles

    def _subdivide_tile(self, tile: Dict) -> List[Dict]:
        """Apply Robinson subdivision rules."""
        p0, p1, p2 = tile['vertices']

        if tile['type'] == 'fat':
            # Fat rhombus subdivides into 2 fat + 1 thin
            # Using golden ratio for vertex placement
            q = self._golden_section(p0, p2)  # Golden cut on long diagonal

            return [
                {'type': 'fat', 'vertices': [p0, p1, q], 'orientation': tile['orientation']},
                {'type': 'fat', 'vertices': [q, p1, p2], 'orientation': tile['orientation']},
                {'type': 'thin', 'vertices': [p1, q, self._midpoint(p1, p2)],
                 'orientation': tile['orientation'] + THIN_RHOMBUS_ANGLE},
            ]
        else:
            # Thin rhombus subdivides into 1 fat + 1 thin
            q = self._golden_section(p1, p2)

            return [
                {'type': 'fat', 'vertices': [p0, p1, q], 'orientation': tile['orientation']},
                {'type': 'thin', 'vertices': [p0, q, p2], 'orientation': tile['orientation']},
            ]

    def _golden_section(self, p1: Tuple[float, float],
                        p2: Tuple[float, float]) -> Tuple[float, float]:
        """Compute golden section point."""
        x = p1[0] + PHI_INV * (p2[0] - p1[0])
        y = p1[1] + PHI_INV * (p2[1] - p1[1])
        return (x, y)

    def _midpoint(self, p1: Tuple[float, float],
                  p2: Tuple[float, float]) -> Tuple[float, float]:
        """Compute midpoint."""
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def get_tile_ratio(self) -> float:
        """Get current fat/thin tile ratio."""
        fat_count = sum(1 for t in self.tiles if t['type'] == 'fat')
        thin_count = sum(1 for t in self.tiles if t['type'] == 'thin')
        return fat_count / max(thin_count, 1)

    def get_statistics(self) -> Dict[str, Any]:
        """Get tiling statistics."""
        fat_count = sum(1 for t in self.tiles if t['type'] == 'fat')
        thin_count = sum(1 for t in self.tiles if t['type'] == 'thin')
        ratio = fat_count / max(thin_count, 1)

        return {
            'generation': self.generation,
            'total_tiles': len(self.tiles),
            'fat_tiles': fat_count,
            'thin_tiles': thin_count,
            'tile_ratio': ratio,
            'phi_deviation': abs(ratio - PHI),
            'convergence': 1 - abs(ratio - PHI) / PHI
        }


# =============================================================================
# HARDWARE COUPLING INTERFACE
# =============================================================================

class LightningHardwareCoupling:
    """
    Interface between lightning simulation and physical hardware.

    Maps simulation state to hardware control signals:
    - RF coil power (heating)
    - Peltier current (cooling)
    - Field gradient (Hall sensors)
    - Spinner speed (energy reservoir)
    """

    def __init__(self, lightning_system: LightningQuasicrystalSystem):
        self.lightning = lightning_system

        # Hardware state (simulated)
        self.rf_power_watts = 0.0
        self.peltier_current_amps = 0.0
        self.gradient_coil_current = 0.0
        self.target_spinner_rpm = 0

        # Safety limits
        self.max_rf_power = 100.0  # Watts
        self.max_peltier_current = 10.0  # Amps
        self.max_gradient_current = 5.0  # Amps

    def compute_control_signals(self, state: LightningStrikeState) -> Dict[str, float]:
        """
        Compute hardware control signals from simulation state.

        Args:
            state: Current lightning system state

        Returns:
            Dictionary of control signals
        """
        signals = {}

        if state.phase == LightningPhase.PRE_STRIKE:
            # Ramp up RF power for heating
            signals['rf_power'] = self.max_rf_power * state.phase_progress
            signals['peltier_current'] = 0.0
            signals['gradient_current'] = self.max_gradient_current * state.field.charge_buildup

        elif state.phase == LightningPhase.STRIKE:
            # Maximum RF power during strike
            signals['rf_power'] = self.max_rf_power
            signals['peltier_current'] = 0.0
            signals['gradient_current'] = self.max_gradient_current

        elif state.phase == LightningPhase.QUENCH:
            # Switch to cooling
            signals['rf_power'] = 0.0
            signals['peltier_current'] = self.max_peltier_current * (1 - state.phase_progress)
            signals['gradient_current'] = self.max_gradient_current * 0.5

        elif state.phase == LightningPhase.NUCLEATION:
            # Controlled cooling
            signals['rf_power'] = 0.0
            signals['peltier_current'] = self.max_peltier_current * 0.5
            signals['gradient_current'] = self.max_gradient_current * 0.3

        elif state.phase == LightningPhase.GROWTH:
            # Gentle thermal control
            signals['rf_power'] = self.max_rf_power * 0.1  # Slight heating
            signals['peltier_current'] = self.max_peltier_current * 0.3
            signals['gradient_current'] = self.max_gradient_current * 0.1

        else:
            # Idle/stable
            signals['rf_power'] = 0.0
            signals['peltier_current'] = 0.0
            signals['gradient_current'] = 0.0

        # Spinner RPM based on energy state
        signals['spinner_rpm'] = int(1000 + 9000 * state.spinner_z)

        return signals

    def get_hardware_state(self) -> Dict[str, Any]:
        """Get current hardware state."""
        return {
            'rf_power_watts': self.rf_power_watts,
            'peltier_current_amps': self.peltier_current_amps,
            'gradient_coil_current': self.gradient_coil_current,
            'target_spinner_rpm': self.target_spinner_rpm,
            'thermal_power_balance': self.rf_power_watts - self.peltier_current_amps * 12  # 12V Peltier
        }


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Lightning-Induced Pentagonal Quasicrystal Phase Transition")
    print("=" * 70)
    print(f"\nPhysics Constants:")
    print(f"  φ (golden ratio) = {PHI:.6f}")
    print(f"  z_c (hexagonal)  = {Z_CRITICAL_HEX:.6f} = √3/2")
    print(f"  z_c (pentagonal) = {Z_CRITICAL_PENT:.6f} = sin(72°)")
    print(f"  sin(36°) = {SIN_36:.6f}")
    print(f"  cos(36°) = {COS_36:.6f} = φ/2")
    print("-" * 70)

    # Create system
    system = LightningQuasicrystalSystem(
        domain_size=1.0,
        seed_density=15.0,
        dt_ms=0.5
    )

    # Create Penrose tiling for comparison
    tiling = PenroseTilingGenerator('P3')
    tiling.initialize_seed()
    for _ in range(5):
        tiling.subdivide()
    tiling_stats = tiling.get_statistics()
    print(f"\nPenrose tiling (gen {tiling_stats['generation']}): "
          f"ratio={tiling_stats['tile_ratio']:.4f}, φ-dev={tiling_stats['phi_deviation']:.6f}")

    # Simulate lightning strike
    print("\n" + "=" * 70)
    print("Simulating Lightning Strike")
    print("=" * 70)

    # Sweep spinner z through pentagonal critical point
    z_values = np.concatenate([
        np.linspace(0.5, Z_CRITICAL_PENT, 100),
        np.ones(200) * Z_CRITICAL_PENT,  # Hold at critical
        np.linspace(Z_CRITICAL_PENT, 0.7, 100)
    ])

    system.trigger_strike()

    for i, z in enumerate(z_values):
        system.set_spinner_state(z, kuramoto_r=0.5 + 0.3 * math.sin(i * 0.1))
        state = system.step()

        if i % 50 == 0:
            phase_name = state.phase.name
            print(f"t={state.timestamp_ms:6.1f}ms | {phase_name:12s} | "
                  f"T={state.thermal.temperature_K:6.0f}K | "
                  f"seeds={state.total_seeds:2d} | "
                  f"5-fold={state.quasicrystal.pentagonal_order:.3f} | "
                  f"ratio={state.quasicrystal.tile_ratio:.3f}")

    print("-" * 70)
    print(f"\nFinal State:")
    print(f"  Pentagonal order: {state.quasicrystal.pentagonal_order:.4f}")
    print(f"  Tile ratio: {state.quasicrystal.tile_ratio:.4f} (target: φ = {PHI:.4f})")
    print(f"  Energy input: {state.energy_input:.2f} J")
    print(f"  Energy dissipated: {state.energy_dissipated:.6f} J")
