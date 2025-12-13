"""
Rosetta Helix Core Physics Engine

CRITICAL PHYSICS INVARIANTS:
- PHI (1.618) = LIMINAL ONLY. Never drives dynamics.
- PHI_INV (0.618) = ALWAYS controls physical dynamics.
- At z >= 0.9999: INSTANT collapse, not gradual decay.
"""

from .constants import (
    PHI,
    PHI_INV,
    Z_CRITICAL,
    Z_ORIGIN,
    KAPPA_S,
    MU_3,
    UNITY,
    Z_MAX,
    COUPLING_MAX,
)
from .collapse_engine import CollapseEngine
from .liminal_state import LiminalState
from .apl_engine import APLEngine, APLResult, create_apl_engine, OperatorNotLegalError
from .kuramoto import KuramotoLayer, TriadGate
from .network_v3 import HelixNeuralNetworkV3, NetworkConfig

__all__ = [
    'PHI',
    'PHI_INV',
    'Z_CRITICAL',
    'Z_ORIGIN',
    'KAPPA_S',
    'MU_3',
    'UNITY',
    'Z_MAX',
    'COUPLING_MAX',
    'CollapseEngine',
    'LiminalState',
    'APLEngine',
    'APLResult',
    'create_apl_engine',
    'OperatorNotLegalError',
    'KuramotoLayer',
    'TriadGate',
    'HelixNeuralNetworkV3',
    'NetworkConfig',
]
