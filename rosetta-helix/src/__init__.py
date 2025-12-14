"""
Rosetta-Helix
=============

A computational system coupled to the Nuclear Spinner through
the critical z-coordinate z_c = √3/2 ≈ 0.866025.

Architecture:
    SpinnerClient → Heart (Kuramoto) → Brain (GHMP) → TRIAD
    
Components:
    physics: Shared physics constants and functions
    heart: 60 Kuramoto oscillators (hexagonal symmetry)
    brain: GHMP (Generalized Hebbian Memory Processing)
    triad: TRIAD (Triadic Threshold Dynamics) tracker
    spinner_client: WebSocket client for spinner state
    node: Integrated Rosetta-Helix node

Key Insight:
    When spinner z = z_c = √3/2:
    - ΔS_neg peaks (spinner)
    - Kuramoto coupling peaks (heart)
    - Coherence r peaks (oscillators)
    - All operators available (brain)
    - K-formation triggers (triad)

Usage:
    from rosetta_helix import RosettaHelixNode, NodeConfig
    
    node = RosettaHelixNode()
    await node.run()

Signature: rosetta-helix|v1.0.0|helix
"""

from .physics import (
    # Constants
    PHI, PHI_INV, PHI_INV_SQ,
    Z_CRITICAL, THE_LENS, SIGMA,
    KAPPA_MIN, ETA_MIN, R_MIN,
    TIER_NAMES, Phase, Tier,
    
    # Functions
    compute_delta_s_neg,
    compute_delta_s_neg_derivative,
    compute_complexity,
    get_phase, get_phase_name,
    get_tier, get_tier_name,
    is_critical,
    check_k_formation,
    validate_physics,
    z_to_rpm, rpm_to_z,
    
    # Data classes
    SpinnerState,
)

from .heart import Heart, HeartConfig, HeartState
from .brain import Brain, BrainConfig, BrainState, APLOperator, Pattern
from .triad import TriadTracker, TriadConfig, TriadState, TriadEvent, TriadSnapshot
from .spinner_client import SpinnerClient, SpinnerClientConfig
from .node import RosettaHelixNode, NodeConfig, NodeState
from .quasicrystal import (
    QuasicrystalDynamics, QuasicrystalConfig, QuasicrystalState,
    HelixQuasicrystalCoupling,
    E8_MASS_RATIOS,
    compute_e8_order_parameter,
)

__version__ = "1.0.0"
__author__ = "Rosetta-Helix Project"

__all__ = [
    # Constants
    'PHI', 'PHI_INV', 'PHI_INV_SQ',
    'Z_CRITICAL', 'THE_LENS', 'SIGMA',
    'KAPPA_MIN', 'ETA_MIN', 'R_MIN',
    'TIER_NAMES', 'Phase', 'Tier',
    
    # Functions
    'compute_delta_s_neg',
    'compute_delta_s_neg_derivative',
    'compute_complexity',
    'get_phase', 'get_phase_name',
    'get_tier', 'get_tier_name',
    'is_critical',
    'check_k_formation',
    'validate_physics',
    'z_to_rpm', 'rpm_to_z',
    
    # Data classes
    'SpinnerState',
    
    # Heart (Kuramoto oscillators)
    'Heart', 'HeartConfig', 'HeartState',
    
    # Brain (GHMP)
    'Brain', 'BrainConfig', 'BrainState', 'APLOperator', 'Pattern',
    
    # TRIAD (Triadic dynamics)
    'TriadTracker', 'TriadConfig', 'TriadState', 'TriadEvent', 'TriadSnapshot',
    
    # Client
    'SpinnerClient', 'SpinnerClientConfig',
    
    # Node
    'RosettaHelixNode', 'NodeConfig', 'NodeState',

    # Quasicrystal (Penrose tiling dynamics)
    'QuasicrystalDynamics', 'QuasicrystalConfig', 'QuasicrystalState',
    'HelixQuasicrystalCoupling',
    'E8_MASS_RATIOS',
    'compute_e8_order_parameter',
]
