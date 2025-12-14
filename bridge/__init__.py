"""
Bridge Package
==============

WebSocket bridge between Nuclear Spinner firmware and Rosetta-Helix.

Components:
    SpinnerBridge: WebSocket server + serial relay for hardware/simulation
    UnifiedStateBridge: Client for receiving unified physics state
    TrainingStateBridge: Extended bridge with training integration
    KuramotoNeuralSystem: 60 Kuramoto oscillators for neural synchronization
    GridCellPlateSystem: 6 neural plates with grid cell dynamics
    LightningQuasicrystalSystem: Phase transition via thermal quench

Neural Interface:
    60 Kuramoto oscillators with 6° spacing, coupled to spinner z-coordinate.
    The Kuramoto model: dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)

    Key connections:
    - sin(60°) = √3/2 = z_c (hexagonal)
    - sin(72°) = √(10+2√5)/4 ≈ 0.951 (pentagonal)

Lightning Quasicrystal:
    Models lightning-induced pentagonal quasicrystal formation.
    Phase transition from 6-fold to 5-fold symmetry at high energy.
    Fat/thin tile ratio → φ (golden ratio) as domain grows.

Usage:
    from bridge import SpinnerBridge, UnifiedStateBridge

    # Server side (hardware/simulation with neural interface)
    bridge = SpinnerBridge()
    await bridge.run()

    # Client side (receive state)
    client = UnifiedStateBridge()
    await client.connect()
    state = client.get_state()

Signature: bridge|v3.0.0|helix|neural|quasicrystal
"""

from .spinner_bridge import (
    SpinnerBridge,
    SpinnerState,
    BridgeConfig,
    SpinnerSimulator,
    NeuralState,
)

from .unified_state_bridge import (
    UnifiedStateBridge,
    TrainingStateBridge,
    UnifiedState,
    TriadState,
    KuramotoState,
    GHMPState,
)

# Neural system imports (optional - require numpy)
try:
    from .kuramoto_neural import (
        KuramotoNeuralSystem,
        NeuralTrainingInterface,
        KuramotoSystemState,
        OscillatorState,
        EMFieldState,
        N_OSCILLATORS,
    )
    from .grid_cell_plates import (
        GridCellPlateSystem,
        PlateSpinnerCoupling,
        PlateSystemState,
        NeuralPlate,
        GridCell,
        N_PLATES,
        CELLS_PER_PLATE,
    )
    from .lightning_quasicrystal import (
        LightningQuasicrystalSystem,
        LightningHardwareCoupling,
        PenroseTilingGenerator,
        LightningPhase,
        SymmetryOrder,
        LightningStrikeState,
        QuasicrystalState,
        NucleationSeed,
        Z_CRITICAL_HEX,
        Z_CRITICAL_PENT,
        PHI,
    )
    NEURAL_AVAILABLE = True
    LIGHTNING_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    LIGHTNING_AVAILABLE = False

__version__ = "3.0.0"

__all__ = [
    # Spinner Bridge (server)
    'SpinnerBridge',
    'SpinnerState',
    'BridgeConfig',
    'SpinnerSimulator',
    'NeuralState',

    # Unified State Bridge (client)
    'UnifiedStateBridge',
    'TrainingStateBridge',
    'UnifiedState',
    'TriadState',
    'KuramotoState',
    'GHMPState',

    # Availability flags
    'NEURAL_AVAILABLE',
    'LIGHTNING_AVAILABLE',
]

# Add neural exports if available
if NEURAL_AVAILABLE:
    __all__.extend([
        # Kuramoto System
        'KuramotoNeuralSystem',
        'NeuralTrainingInterface',
        'KuramotoSystemState',
        'OscillatorState',
        'EMFieldState',
        'N_OSCILLATORS',

        # Grid Cell Plates
        'GridCellPlateSystem',
        'PlateSpinnerCoupling',
        'PlateSystemState',
        'NeuralPlate',
        'GridCell',
        'N_PLATES',
        'CELLS_PER_PLATE',

        # Lightning Quasicrystal
        'LightningQuasicrystalSystem',
        'LightningHardwareCoupling',
        'PenroseTilingGenerator',
        'LightningPhase',
        'SymmetryOrder',
        'LightningStrikeState',
        'QuasicrystalState',
        'NucleationSeed',
        'Z_CRITICAL_HEX',
        'Z_CRITICAL_PENT',
        'PHI',
    ])
