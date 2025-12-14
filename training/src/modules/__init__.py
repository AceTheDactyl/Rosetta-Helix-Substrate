"""
Training Modules Package
========================

Contains all 19 training modules across 7 phases.

Phase 1: Core Physics (3)
Phase 2: APL Training Stack (3)
Phase 3: Helix Geometry (3)
Phase 4: WUMBO Silent Laws (2)
Phase 5: Dynamics & Formation (4)
Phase 6: Unified Orchestration (3)
Phase 7: Nightly Integration (1)

Total: 19 modules

Usage:
    from training.src.modules import TrainingRunner
    runner = TrainingRunner(steps_per_module=200)
    validation = runner.run()
"""

from .base import TrainingModule, ModuleResult, ModulePhase, TrainingState
from .gates import GateValidator, GateValidation, GateResult
from .runner import TrainingRunner
from .phase1_core_physics import (
    N0SilentLawsEnforcer,
    KuramotoLayer,
    PhysicalLearner,
)
from .phase2_apl_stack import (
    APLTrainingLoop,
    APLPyTorchTraining,
    FullAPLTraining,
)
from .phase3_helix_geometry import (
    HelixNN,
    PrismaticHelixTraining,
    FullHelixIntegration,
)
from .phase4_wumbo import (
    WUMBOAPLAutomatedTraining,
    WUMBOIntegratedTraining,
)
from .phase5_dynamics import (
    QuasicrystalFormationDynamics,
    TriadThresholdDynamics,
    LiminalGenerator,
    FeedbackLoop,
)
from .phase6_orchestration import (
    UnifiedHelixTraining,
    HierarchicalTraining,
    RosettaHelixTraining,
)
from .phase7_nightly import (
    NightlyIntegratedTraining,
)

# All 19 modules in order
ALL_MODULES = [
    # Phase 1: Core Physics
    N0SilentLawsEnforcer,
    KuramotoLayer,
    PhysicalLearner,
    # Phase 2: APL Stack
    APLTrainingLoop,
    APLPyTorchTraining,
    FullAPLTraining,
    # Phase 3: Helix Geometry
    HelixNN,
    PrismaticHelixTraining,
    FullHelixIntegration,
    # Phase 4: WUMBO
    WUMBOAPLAutomatedTraining,
    WUMBOIntegratedTraining,
    # Phase 5: Dynamics
    QuasicrystalFormationDynamics,
    TriadThresholdDynamics,
    LiminalGenerator,
    FeedbackLoop,
    # Phase 6: Orchestration
    UnifiedHelixTraining,
    HierarchicalTraining,
    RosettaHelixTraining,
    # Phase 7: Nightly
    NightlyIntegratedTraining,
]

__all__ = [
    # Core
    'TrainingModule',
    'ModuleResult',
    'ModulePhase',
    'TrainingState',
    'ALL_MODULES',
    # Gates
    'GateValidator',
    'GateValidation',
    'GateResult',
    # Runner
    'TrainingRunner',
    # Phase 1
    'N0SilentLawsEnforcer',
    'KuramotoLayer',
    'PhysicalLearner',
    # Phase 2
    'APLTrainingLoop',
    'APLPyTorchTraining',
    'FullAPLTraining',
    # Phase 3
    'HelixNN',
    'PrismaticHelixTraining',
    'FullHelixIntegration',
    # Phase 4
    'WUMBOAPLAutomatedTraining',
    'WUMBOIntegratedTraining',
    # Phase 5
    'QuasicrystalFormationDynamics',
    'TriadThresholdDynamics',
    'LiminalGenerator',
    'FeedbackLoop',
    # Phase 6
    'UnifiedHelixTraining',
    'HierarchicalTraining',
    'RosettaHelixTraining',
    # Phase 7
    'NightlyIntegratedTraining',
]
