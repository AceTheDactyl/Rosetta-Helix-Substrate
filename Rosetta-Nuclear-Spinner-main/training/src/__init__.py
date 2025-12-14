"""
Training Package
================

Training workflows and utilities for Rosetta-Helix.

Components:
    CyberneticTrainingManager: Unified manager for all 19 training modules
    TrainingModule: Enum of all training modules
    TrainingPhase: Enum of training phases (7 phases)
    AdaptiveParams: Physics-grounded adaptive hyperparameters

Usage:
    from training.src import CyberneticTrainingManager, TrainingModule

    manager = CyberneticTrainingManager()
    manager.start_phase(TrainingPhase.CORE_PHYSICS)
    params = await manager.step(TrainingModule.KURAMOTO_LAYER, loss=0.5)

Signature: training|v1.0.0|helix
"""

from .cybernetic_training import (
    CyberneticTrainingManager,
    TrainingModule,
    TrainingPhase,
    AdaptiveParams,
    ModuleState,
    MODULE_PHASES,
)

__version__ = "1.0.0"

__all__ = [
    'CyberneticTrainingManager',
    'TrainingModule',
    'TrainingPhase',
    'AdaptiveParams',
    'ModuleState',
    'MODULE_PHASES',
]
