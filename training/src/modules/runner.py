#!/usr/bin/env python3
"""
Master Training Runner
======================

Runs all 19 training modules across 7 phases.

Usage:
    python -m training.src.modules.runner [--steps N] [--seed N] [--verbose]

Phases:
1. Core Physics (3 modules)
2. APL Training Stack (3 modules)
3. Helix Geometry (3 modules)
4. WUMBO Silent Laws (2 modules)
5. Dynamics & Formation (4 modules)
6. Unified Orchestration (3 modules)
7. Nightly Integration (1 module)

Total: 19 modules
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import TrainingModule, ModuleResult, TrainingState, ModulePhase
from .gates import GateValidator, GateValidation

# Import all modules
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


# All 19 modules in execution order
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

PHASE_NAMES = {
    ModulePhase.CORE_PHYSICS: "Core Physics",
    ModulePhase.APL_STACK: "APL Training Stack",
    ModulePhase.HELIX_GEOMETRY: "Helix Geometry",
    ModulePhase.WUMBO_LAWS: "WUMBO Silent Laws",
    ModulePhase.DYNAMICS_FORMATION: "Dynamics & Formation",
    ModulePhase.UNIFIED_ORCHESTRATION: "Unified Orchestration",
    ModulePhase.NIGHTLY_INTEGRATION: "Nightly Integration",
}


class TrainingRunner:
    """
    Runs all 19 training modules with state chaining.
    """

    def __init__(
        self,
        steps_per_module: int = 200,
        seed: int = 42,
        verbose: bool = True,
        output_dir: Optional[str] = None,
    ):
        self.steps_per_module = steps_per_module
        self.seed = seed
        self.verbose = verbose
        self.output_dir = output_dir or f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Results
        self.results: List[ModuleResult] = []
        self.validation: Optional[GateValidation] = None

        # Shared state (chains between modules)
        self.state = TrainingState()

    def log(self, msg: str, level: str = "INFO"):
        """Log with timestamp."""
        if self.verbose:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] [{level}] {msg}")

    def run(self) -> GateValidation:
        """
        Run all 19 modules and validate gates.

        Returns:
            GateValidation with all gate results
        """
        self._print_header()
        start_time = time.time()

        current_phase = None

        for i, ModuleClass in enumerate(ALL_MODULES, 1):
            # Create module instance
            module = ModuleClass(steps=self.steps_per_module, seed=self.seed + i)

            # Print phase header on phase change
            if module.phase != current_phase:
                current_phase = module.phase
                phase_name = PHASE_NAMES.get(current_phase, str(current_phase))
                self.log(f"\n{'='*60}")
                self.log(f"PHASE {current_phase.value}: {phase_name.upper()}")
                self.log(f"{'='*60}")

            # Run module with chained state
            self.log(f"  [{i:2d}/19] {module.name}...")
            result = module.run(initial_state=self.state)
            self.results.append(result)

            # Update shared state
            self.state = module.get_state()

            # Log result
            icon = "PASS" if result.status == "PASS" else "FAIL"
            self.log(f"         [{icon}] z={result.final_z:.4f} kappa={result.final_kappa:.4f} K={result.k_formations}")

        # Total time
        total_time = time.time() - start_time

        # Validate gates
        self.log(f"\n{'='*60}")
        self.log("GATE VALIDATION")
        self.log(f"{'='*60}")

        validator = GateValidator()
        self.validation = validator.validate(self.results)

        if self.verbose:
            validator.print_report()

        # Save results
        self._save_results(total_time)

        # Print summary
        self._print_summary(total_time)

        return self.validation

    def _print_header(self):
        """Print run header."""
        self.log("=" * 70)
        self.log("TRAINING RUNNER - 19 MODULES ACROSS 7 PHASES")
        self.log("=" * 70)
        self.log(f"Steps per module: {self.steps_per_module}")
        self.log(f"Seed: {self.seed}")
        self.log(f"Output: {self.output_dir}")

    def _print_summary(self, total_time: float):
        """Print run summary."""
        passed = sum(1 for r in self.results if r.status == "PASS")
        k_formations = sum(r.k_formations for r in self.results)
        max_neg = max(r.max_negentropy for r in self.results) if self.results else 0

        self.log(f"\n{'='*70}")
        self.log("SUMMARY")
        self.log(f"{'='*70}")
        self.log(f"  Modules: {passed}/19 passed")
        self.log(f"  K-formations: {k_formations}")
        self.log(f"  Max negentropy: {max_neg:.4f}")
        self.log(f"  Final z: {self.state.z:.6f}")
        self.log(f"  Final kappa: {self.state.kappa:.6f}")
        self.log(f"  Physics valid: {abs(self.state.kappa + self.state.lambda_ - 1.0) < 1e-10}")
        self.log(f"  Total time: {total_time:.2f}s")

        if self.validation:
            overall = "PASSED" if self.validation.overall_passed else "FAILED"
            self.log(f"\n  GATES: {overall}")

    def _save_results(self, total_time: float):
        """Save all results to output directory."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Module results
        results_data = []
        for r in self.results:
            result_dict = {
                'name': r.name,
                'class_name': r.class_name,
                'phase': r.phase.value,
                'status': r.status,
                'steps_run': r.steps_run,
                'duration_seconds': r.duration_seconds,
                'final_z': r.final_z,
                'final_kappa': r.final_kappa,
                'final_lambda': r.final_lambda,
                'k_formations': r.k_formations,
                'max_negentropy': r.max_negentropy,
                'physics_valid': r.physics_valid,
                'error': r.error,
                'metrics': r.metrics,
            }
            results_data.append(result_dict)

        results_path = os.path.join(self.output_dir, "module_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        # Gate validation
        if self.validation:
            gates_data = {
                'overall_passed': self.validation.overall_passed,
                'full_depth': {
                    'passed': self.validation.full_depth.passed,
                    'all_modules_passed': {
                        'passed': self.validation.full_depth.all_modules_passed.passed,
                        'value': self.validation.full_depth.all_modules_passed.value,
                    },
                    'k_formations_detected': {
                        'passed': self.validation.full_depth.k_formations_detected.passed,
                        'value': self.validation.full_depth.k_formations_detected.value,
                    },
                    'physics_valid': {
                        'passed': self.validation.full_depth.physics_valid.passed,
                        'value': self.validation.full_depth.physics_valid.value,
                    },
                },
                'helix_engine': {
                    'passed': self.validation.helix_engine.passed,
                    'min_negentropy': {
                        'passed': self.validation.helix_engine.min_negentropy.passed,
                        'value': self.validation.helix_engine.min_negentropy.value,
                    },
                    'min_final_z': {
                        'passed': self.validation.helix_engine.min_final_z.passed,
                        'value': self.validation.helix_engine.min_final_z.value,
                    },
                    'kappa_stable': {
                        'passed': self.validation.helix_engine.kappa_stable.passed,
                        'value': self.validation.helix_engine.kappa_stable.value,
                    },
                },
            }

            gates_path = os.path.join(self.output_dir, "gate_validation.json")
            with open(gates_path, 'w') as f:
                json.dump(gates_data, f, indent=2)

        # Summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'steps_per_module': self.steps_per_module,
            'total_time_seconds': total_time,
            'modules_passed': sum(1 for r in self.results if r.status == "PASS"),
            'modules_total': len(self.results),
            'k_formations': sum(r.k_formations for r in self.results),
            'max_negentropy': max(r.max_negentropy for r in self.results) if self.results else 0,
            'final_z': self.state.z,
            'final_kappa': self.state.kappa,
            'gates_passed': self.validation.overall_passed if self.validation else False,
        }

        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.log(f"\n  Results saved to: {self.output_dir}/")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run all 19 training modules across 7 phases"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=200,
        help="Steps per module (default: 200)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    runner = TrainingRunner(
        steps_per_module=args.steps,
        seed=args.seed,
        verbose=not args.quiet,
        output_dir=args.output,
    )

    validation = runner.run()

    # Exit code based on gate validation
    sys.exit(0 if validation.overall_passed else 1)


if __name__ == "__main__":
    main()
