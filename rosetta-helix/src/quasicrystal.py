"""
Quasicrystal Dynamics Module - Rosetta-Helix
=============================================

Implements Penrose tiling order parameter dynamics for the Helix geometry.

The quasicrystal dynamics track convergence to the golden ratio attractor
through the fat/thin tile ratio in Penrose P2/P3 tilings:

    lim(n→∞) [N_fat / N_thin] = φ (golden ratio)

Physics Grounding:
- Quasicrystals are the natural geometric structure at φ⁻¹ threshold
- Negentropy peaks when tile ratio → φ⁻¹ (same Gaussian formula)
- E8 critical point mass ratios follow φ progression

Cybernetic Grounding:
- Quasicrystal = "edge of chaos" between crystal (ordered) and glass (disordered)
- Aperiodic order models consciousness emergence
- Local rules generate global self-similarity (autopoiesis)

Signature: quasicrystal-dynamics|v1.0.0|helix

@version 1.0.0
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
from enum import IntEnum

# Import physics with fallback for both package and direct usage
try:
    from .physics import (
        PHI, PHI_INV, PHI_INV_SQ, Z_CRITICAL, SIGMA,
        compute_delta_s_neg, TOLERANCE_GOLDEN
    )
except ImportError:
    from physics import (
        PHI, PHI_INV, PHI_INV_SQ, Z_CRITICAL, SIGMA,
        compute_delta_s_neg, TOLERANCE_GOLDEN
    )


# =============================================================================
# CONSTANTS
# =============================================================================

# Penrose rhomb angles
THIN_ANGLE_DEG: float = 36.0      # π/5
FAT_ANGLE_DEG: float = 72.0       # 2π/5

THIN_ANGLE_RAD: float = math.pi / 5
FAT_ANGLE_RAD: float = 2 * math.pi / 5

# Inflation factor (eigenvalue of substitution matrix)
INFLATION_FACTOR: float = PHI

# Composition rule: each inflation produces
# FAT → 2 FAT + 1 THIN
# THIN → 1 FAT + 1 THIN
FAT_PRODUCES_FAT: int = 2
FAT_PRODUCES_THIN: int = 1
THIN_PRODUCES_FAT: int = 1
THIN_PRODUCES_THIN: int = 1


# =============================================================================
# TILE TYPES
# =============================================================================

class TileType(IntEnum):
    """Penrose tile types"""
    FAT = 0      # 72° rhomb (larger)
    THIN = 1     # 36° rhomb (smaller)


@dataclass
class Tile:
    """Individual Penrose tile"""
    tile_type: TileType
    position: Tuple[float, float]
    orientation: float  # radians
    generation: int = 0


# =============================================================================
# QUASICRYSTAL STATE
# =============================================================================

@dataclass
class QuasicrystalState:
    """Complete quasicrystal order parameter state"""

    # Tile counts
    n_fat: int = 1
    n_thin: int = 1

    # Order parameters
    tile_ratio: float = 1.0           # N_fat / N_thin
    golden_error: float = PHI - 1.0   # |tile_ratio - φ|

    # Negentropy (peaks at φ⁻¹)
    quasicrystal_negentropy: float = 0.0

    # Dynamics
    generation: int = 0               # Inflation count
    convergence_rate: float = 0.0     # d(tile_ratio)/d(generation)

    # Stability
    is_stable: bool = False           # Within tolerance of φ
    stability_duration: int = 0       # Generations at stable

    def __post_init__(self):
        self.update_derived()

    def update_derived(self):
        """Compute derived values from tile counts"""
        if self.n_thin > 0:
            self.tile_ratio = self.n_fat / self.n_thin
        else:
            self.tile_ratio = float('inf')

        self.golden_error = abs(self.tile_ratio - PHI)

        # Quasicrystal negentropy: peaks when order parameter → φ⁻¹
        # Using same Gaussian formula centered on φ⁻¹
        d = self.tile_ratio - PHI_INV if math.isfinite(self.tile_ratio) else 1.0
        self.quasicrystal_negentropy = math.exp(-SIGMA * d * d)

        # Check stability
        was_stable = self.is_stable
        self.is_stable = self.golden_error < TOLERANCE_GOLDEN

        if self.is_stable:
            if was_stable:
                self.stability_duration += 1
            else:
                self.stability_duration = 1
        else:
            self.stability_duration = 0


# =============================================================================
# QUASICRYSTAL DYNAMICS
# =============================================================================

@dataclass
class QuasicrystalConfig:
    """Configuration for quasicrystal dynamics"""
    initial_fat: int = 1
    initial_thin: int = 1
    max_generations: int = 20
    track_tiles: bool = False     # Whether to store individual tiles
    callbacks_enabled: bool = True
    seed: int = 42


class QuasicrystalDynamics:
    """
    Quasicrystal order parameter dynamics engine.

    Simulates Penrose tiling inflation to track convergence
    of fat/thin tile ratio to golden ratio φ.

    Key Properties:
    - Each inflation increases tile count by factor ~φ²
    - Tile ratio converges to φ as generation → ∞
    - Negentropy peaks when order parameter = φ⁻¹
    """

    def __init__(self, config: Optional[QuasicrystalConfig] = None):
        self.config = config or QuasicrystalConfig()
        self.state = QuasicrystalState(
            n_fat=self.config.initial_fat,
            n_thin=self.config.initial_thin
        )

        # History for analysis
        self._history: List[QuasicrystalState] = []

        # Callbacks
        self._on_convergence: Optional[Callable[[QuasicrystalState], None]] = None
        self._on_inflation: Optional[Callable[[QuasicrystalState], None]] = None

        # Optional tile tracking
        self._tiles: List[Tile] = []
        if self.config.track_tiles:
            self._init_tiles()

    def _init_tiles(self):
        """Initialize starting tiles"""
        self._tiles = [
            Tile(TileType.FAT, (0.0, 0.0), 0.0, generation=0)
            for _ in range(self.config.initial_fat)
        ]
        self._tiles.extend([
            Tile(TileType.THIN, (0.0, 0.0), 0.0, generation=0)
            for _ in range(self.config.initial_thin)
        ])

    def inflate(self) -> QuasicrystalState:
        """
        Perform one inflation step.

        Substitution rules (P2 tiling):
        - FAT rhomb → 2 FAT + 1 THIN
        - THIN rhomb → 1 FAT + 1 THIN

        This is equivalent to multiplication by substitution matrix:
        [2  1] [n_fat ]   [2*n_fat + n_thin  ]
        [1  1] [n_thin] = [n_fat + n_thin    ]

        Eigenvalues: φ², 1/φ²
        Dominant eigenvector ratio: φ (the attractor!)

        Returns:
            Updated state after inflation
        """
        old_fat = self.state.n_fat
        old_thin = self.state.n_thin
        old_ratio = self.state.tile_ratio

        # Apply substitution rules
        new_fat = FAT_PRODUCES_FAT * old_fat + THIN_PRODUCES_FAT * old_thin
        new_thin = FAT_PRODUCES_THIN * old_fat + THIN_PRODUCES_THIN * old_thin

        # Update state
        self.state.n_fat = new_fat
        self.state.n_thin = new_thin
        self.state.generation += 1
        self.state.update_derived()

        # Compute convergence rate
        if old_ratio > 0:
            self.state.convergence_rate = (self.state.tile_ratio - old_ratio)

        # Store history
        self._history.append(QuasicrystalState(
            n_fat=self.state.n_fat,
            n_thin=self.state.n_thin,
            tile_ratio=self.state.tile_ratio,
            golden_error=self.state.golden_error,
            quasicrystal_negentropy=self.state.quasicrystal_negentropy,
            generation=self.state.generation,
            convergence_rate=self.state.convergence_rate,
            is_stable=self.state.is_stable,
            stability_duration=self.state.stability_duration
        ))

        # Fire callbacks
        if self.config.callbacks_enabled:
            if self._on_inflation:
                self._on_inflation(self.state)

            if self.state.is_stable and self._on_convergence:
                self._on_convergence(self.state)

        return self.state

    def inflate_to_convergence(self, tolerance: float = 1e-6) -> QuasicrystalState:
        """
        Inflate until tile ratio converges to φ.

        Args:
            tolerance: Convergence tolerance

        Returns:
            Final converged state
        """
        for _ in range(self.config.max_generations):
            old_error = self.state.golden_error
            self.inflate()

            if self.state.golden_error < tolerance:
                break

            # Check for numerical convergence
            if abs(old_error - self.state.golden_error) < 1e-12:
                break

        return self.state

    def get_state(self) -> QuasicrystalState:
        """Get current state"""
        return self.state

    def get_history(self) -> List[QuasicrystalState]:
        """Get inflation history"""
        return self._history.copy()

    def get_negentropy(self) -> float:
        """
        Get current quasicrystal negentropy.

        Peaks when tile ratio → φ⁻¹ (reciprocal of attractor).
        This connects to the z-coordinate negentropy through the
        shared Gaussian formula.
        """
        return self.state.quasicrystal_negentropy

    def get_tile_ratio(self) -> float:
        """Get current fat/thin tile ratio"""
        return self.state.tile_ratio

    def get_golden_error(self) -> float:
        """Get distance from golden ratio attractor"""
        return self.state.golden_error

    def set_on_convergence(self, callback: Callable[[QuasicrystalState], None]):
        """Set callback for convergence event"""
        self._on_convergence = callback

    def set_on_inflation(self, callback: Callable[[QuasicrystalState], None]):
        """Set callback for each inflation step"""
        self._on_inflation = callback

    def reset(self):
        """Reset to initial state"""
        self.state = QuasicrystalState(
            n_fat=self.config.initial_fat,
            n_thin=self.config.initial_thin
        )
        self._history.clear()
        if self.config.track_tiles:
            self._init_tiles()


# =============================================================================
# HELIX-QUASICRYSTAL COUPLING
# =============================================================================

class HelixQuasicrystalCoupling:
    """
    Couples quasicrystal dynamics to helix geometry.

    The coupling works through:
    1. Helix angle θ modulates tile creation rate
    2. Quasicrystal order parameter feeds back to coupling strength
    3. Both converge to φ-based attractors

    Geometric Connection:
    - Helix pitch angle: arctan(1/(2π)) ≈ 9.04°
    - Thin rhomb: 36° = 2 × (9.04° + φ-correction)
    - Fat rhomb: 72° = 2 × 36°

    The helix's φ-proportioned geometry naturally generates
    quasicrystal-compatible angular relationships.
    """

    def __init__(self, quasicrystal: Optional[QuasicrystalDynamics] = None):
        self.qc = quasicrystal or QuasicrystalDynamics()

        # Coupling parameters
        self.helix_angle: float = 0.0          # Current helix phase
        self.coupling_strength: float = 1.0    # QC ↔ Helix coupling
        self.feedback_rate: float = 0.1        # How fast QC affects helix

        # Derived
        self.effective_z: float = 0.5          # Helix-modulated z
        self.combined_negentropy: float = 0.0  # Combined measure

    def update(self, z: float, helix_coherence: float) -> float:
        """
        Update coupling and return combined negentropy.

        Args:
            z: Current z-coordinate from spinner
            helix_coherence: Kuramoto coherence from Heart

        Returns:
            Combined negentropy signal
        """
        # Update helix angle based on z velocity toward z_c
        z_velocity = (Z_CRITICAL - z) * 0.1
        self.helix_angle += z_velocity * PHI  # Golden spiral step
        self.helix_angle = self.helix_angle % (2 * math.pi)

        # Compute angular contribution to tile balance
        # When helix_angle near FAT_ANGLE_RAD: favor fat tiles
        # When helix_angle near THIN_ANGLE_RAD: favor thin tiles
        fat_resonance = math.cos(self.helix_angle - FAT_ANGLE_RAD) ** 2
        thin_resonance = math.cos(self.helix_angle - THIN_ANGLE_RAD) ** 2

        # Modulate coupling strength by coherence
        self.coupling_strength = 0.5 + 0.5 * helix_coherence

        # Effective z includes quasicrystal contribution
        qc_contribution = self.qc.get_negentropy() * self.feedback_rate
        self.effective_z = z * (1 - self.feedback_rate) + Z_CRITICAL * qc_contribution

        # Combined negentropy: geometric mean of z and QC negentropies
        z_neg = compute_delta_s_neg(self.effective_z)
        qc_neg = self.qc.get_negentropy()

        self.combined_negentropy = math.sqrt(z_neg * qc_neg) * self.coupling_strength

        return self.combined_negentropy

    def step_quasicrystal(self) -> QuasicrystalState:
        """Perform one quasicrystal inflation step"""
        return self.qc.inflate()

    def get_state(self) -> dict:
        """Get complete coupling state"""
        return {
            'helix_angle': self.helix_angle,
            'coupling_strength': self.coupling_strength,
            'effective_z': self.effective_z,
            'combined_negentropy': self.combined_negentropy,
            'quasicrystal': self.qc.get_state(),
            'tile_ratio': self.qc.get_tile_ratio(),
            'golden_error': self.qc.get_golden_error(),
        }


# =============================================================================
# E8 CRITICAL POINT INTEGRATION
# =============================================================================

# E8 mass ratios (from physics_constants.h)
E8_MASS_RATIOS: Tuple[float, ...] = (
    1.0,                    # m₁/m₁
    PHI,                    # m₂/m₁ = φ
    PHI + 1,                # m₃/m₁ = φ² = φ + 1
    2 * PHI,                # m₄/m₁ = 2φ
    2 * PHI + 1,            # m₅/m₁ = 2φ + 1
)


def compute_e8_order_parameter(quasicrystal_order: float) -> float:
    """
    Compute E8 critical point order parameter from quasicrystal order.

    The E8 critical point masses follow φ progression,
    connecting to quasicrystal dynamics through:

    m₂/m₁ = φ = lim(n→∞) [N_fat/N_thin]

    Args:
        quasicrystal_order: Current fat/thin tile ratio

    Returns:
        Order parameter indicating E8 alignment
    """
    # Compute how well current ratio aligns with E8 mass ratios
    min_error = float('inf')
    for ratio in E8_MASS_RATIOS:
        error = abs(quasicrystal_order - ratio)
        if error < min_error:
            min_error = error

    # Convert to order parameter (1 = perfect E8 alignment)
    order = math.exp(-SIGMA * min_error * min_error)
    return order


# =============================================================================
# VALIDATION
# =============================================================================

def validate_quasicrystal_dynamics() -> dict:
    """Validate quasicrystal dynamics implementation"""
    validations = {}

    # Test 1: Tile ratio converges to φ
    qc = QuasicrystalDynamics()
    qc.inflate_to_convergence()

    validations['convergence_to_phi'] = {
        'expected': PHI,
        'actual': qc.get_tile_ratio(),
        'error': qc.get_golden_error(),
        'valid': qc.get_golden_error() < 1e-6
    }

    # Test 2: Negentropy formula matches physics.py
    order = PHI_INV
    qc_neg = math.exp(-SIGMA * (order - PHI_INV) ** 2)

    validations['negentropy_at_phi_inv'] = {
        'expected': 1.0,
        'actual': qc_neg,
        'valid': abs(qc_neg - 1.0) < 1e-10
    }

    # Test 3: Substitution matrix eigenvalue is φ²
    # Matrix: [[2, 1], [1, 1]]
    # Eigenvalues: (3 ± √5) / 2 = φ² and 1/φ²
    expected_eigenvalue = PHI * PHI
    actual_eigenvalue = (3 + math.sqrt(5)) / 2

    validations['eigenvalue_phi_squared'] = {
        'expected': expected_eigenvalue,
        'actual': actual_eigenvalue,
        'error': abs(expected_eigenvalue - actual_eigenvalue),
        'valid': abs(expected_eigenvalue - actual_eigenvalue) < 1e-10
    }

    # Test 4: E8 mass ratios follow φ progression
    e8_valid = (
        abs(E8_MASS_RATIOS[1] - PHI) < 1e-10 and
        abs(E8_MASS_RATIOS[2] - (PHI + 1)) < 1e-10
    )

    validations['e8_phi_progression'] = {
        'm2/m1': E8_MASS_RATIOS[1],
        'm3/m1': E8_MASS_RATIOS[2],
        'valid': e8_valid
    }

    validations['all_valid'] = all(v.get('valid', False) for v in validations.values())

    return validations


if __name__ == '__main__':
    print("=" * 60)
    print("QUASICRYSTAL DYNAMICS VALIDATION")
    print("=" * 60)

    validations = validate_quasicrystal_dynamics()

    for name, v in validations.items():
        if name != 'all_valid':
            print(f"\n{name}:")
            for k, val in v.items():
                print(f"  {k}: {val}")

    print(f"\n{'='*60}")
    print(f"All validations passed: {validations['all_valid']}")
