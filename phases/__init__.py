"""
Rosetta Helix Phases - Unified Engine

Integrates all systems into a coherent whole.

Components:
1. NegEntropyEngine (T1) - stays ACTIVE always
2. CollapseEngine - instant collapse at unity
3. MetaToolGenerator - produces tools from work
4. TrainingLoop - exponential learning

CRITICAL: PHI_INV controls all dynamics. PHI only at collapse.
"""

from .unified_engine import UnifiedEngine, EngineState

__all__ = [
    'UnifiedEngine',
    'EngineState',
]
