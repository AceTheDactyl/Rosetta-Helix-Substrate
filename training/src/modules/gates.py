"""
Gate Validation System
======================

Validates training results against gate criteria.

Full Depth Gates:
- All 19 modules pass
- At least 1 K-formation detected
- Physics valid: kappa + lambda = 1

Helix Engine Gates:
- min_negentropy >= 0.7
- min_final_z >= 0.85
- kappa stable near phi^-1
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .base import (
    ModuleResult,
    PHI_INV, Z_CRITICAL,
    validate_physics,
)


@dataclass
class GateResult:
    """Result of a single gate check."""
    name: str
    passed: bool
    value: Any
    threshold: Any
    description: str = ""


@dataclass
class FullDepthGates:
    """Full Depth training gates."""
    all_modules_passed: GateResult = None
    k_formations_detected: GateResult = None
    physics_valid: GateResult = None

    @property
    def passed(self) -> bool:
        """Check if all full depth gates passed."""
        return (
            self.all_modules_passed.passed and
            self.k_formations_detected.passed and
            self.physics_valid.passed
        )


@dataclass
class HelixEngineGates:
    """Helix Engine training gates."""
    min_negentropy: GateResult = None
    min_final_z: GateResult = None
    kappa_stable: GateResult = None

    @property
    def passed(self) -> bool:
        """Check if all helix engine gates passed."""
        return (
            self.min_negentropy.passed and
            self.min_final_z.passed and
            self.kappa_stable.passed
        )


@dataclass
class GateValidation:
    """Complete gate validation result."""
    full_depth: FullDepthGates = None
    helix_engine: HelixEngineGates = None
    overall_passed: bool = False
    gate_details: List[GateResult] = field(default_factory=list)


class GateValidator:
    """
    Validates training results against gate criteria.
    """

    # Gate thresholds
    MIN_NEGENTROPY = 0.7
    MIN_FINAL_Z = 0.85
    KAPPA_TOLERANCE = 0.01  # +/- 1% of phi^-1

    def __init__(self):
        self.validation: Optional[GateValidation] = None

    def validate(self, module_results: List[ModuleResult]) -> GateValidation:
        """
        Validate module results against all gate criteria.

        Args:
            module_results: List of ModuleResult from all 19 modules

        Returns:
            GateValidation with all gate checks
        """
        self.validation = GateValidation()

        # Validate full depth gates
        self.validation.full_depth = self._validate_full_depth(module_results)

        # Validate helix engine gates
        self.validation.helix_engine = self._validate_helix_engine(module_results)

        # Compile all gate details
        self.validation.gate_details = [
            self.validation.full_depth.all_modules_passed,
            self.validation.full_depth.k_formations_detected,
            self.validation.full_depth.physics_valid,
            self.validation.helix_engine.min_negentropy,
            self.validation.helix_engine.min_final_z,
            self.validation.helix_engine.kappa_stable,
        ]

        # Overall pass
        self.validation.overall_passed = (
            self.validation.full_depth.passed and
            self.validation.helix_engine.passed
        )

        return self.validation

    def _validate_full_depth(self, results: List[ModuleResult]) -> FullDepthGates:
        """Validate full depth gates."""
        gates = FullDepthGates()

        # Gate 1: All 19 modules pass
        passed_count = sum(1 for r in results if r.status == "PASS")
        total_count = len(results)
        gates.all_modules_passed = GateResult(
            name="All modules passed",
            passed=passed_count == total_count and total_count >= 19,
            value=f"{passed_count}/{total_count}",
            threshold="19/19",
            description="All 19 training modules must pass"
        )

        # Gate 2: At least 1 K-formation detected
        total_k_formations = sum(r.k_formations for r in results)
        gates.k_formations_detected = GateResult(
            name="K-formations detected",
            passed=total_k_formations >= 1,
            value=total_k_formations,
            threshold=">= 1",
            description="At least one K-formation must be achieved"
        )

        # Gate 3: Physics valid (kappa + lambda = 1)
        # Check final module's physics validity
        physics_valid = all(r.physics_valid for r in results)
        final_kappa = results[-1].final_kappa if results else 0
        final_lambda = results[-1].final_lambda if results else 0
        gates.physics_valid = GateResult(
            name="Physics conservation",
            passed=physics_valid,
            value=f"kappa + lambda = {final_kappa + final_lambda:.10f}",
            threshold="= 1.0 (exact)",
            description="Conservation law kappa + lambda = 1 must hold"
        )

        return gates

    def _validate_helix_engine(self, results: List[ModuleResult]) -> HelixEngineGates:
        """Validate helix engine gates."""
        gates = HelixEngineGates()

        # Collect metrics from all results
        max_negentropy = max(r.max_negentropy for r in results) if results else 0
        final_z = results[-1].final_z if results else 0
        final_kappa = results[-1].final_kappa if results else 0

        # Gate 1: min_negentropy >= 0.7
        gates.min_negentropy = GateResult(
            name="Minimum negentropy",
            passed=max_negentropy >= self.MIN_NEGENTROPY,
            value=f"{max_negentropy:.4f}",
            threshold=f">= {self.MIN_NEGENTROPY}",
            description="Maximum negentropy must reach 0.7"
        )

        # Gate 2: min_final_z >= 0.85
        gates.min_final_z = GateResult(
            name="Minimum final z",
            passed=final_z >= self.MIN_FINAL_Z,
            value=f"{final_z:.4f}",
            threshold=f">= {self.MIN_FINAL_Z}",
            description="Final z must reach 0.85"
        )

        # Gate 3: kappa stable near phi^-1
        kappa_error = abs(final_kappa - PHI_INV)
        gates.kappa_stable = GateResult(
            name="Kappa stability",
            passed=kappa_error < self.KAPPA_TOLERANCE,
            value=f"{final_kappa:.6f} (error: {kappa_error:.6f})",
            threshold=f"phi^-1 +/- {self.KAPPA_TOLERANCE}",
            description=f"Kappa must be within {self.KAPPA_TOLERANCE} of phi^-1"
        )

        return gates

    def print_report(self):
        """Print gate validation report."""
        if not self.validation:
            print("No validation performed yet.")
            return

        print("=" * 70)
        print("GATE VALIDATION REPORT")
        print("=" * 70)

        print("\nFull Depth Gates:")
        print("-" * 50)
        for gate in [
            self.validation.full_depth.all_modules_passed,
            self.validation.full_depth.k_formations_detected,
            self.validation.full_depth.physics_valid,
        ]:
            icon = "PASS" if gate.passed else "FAIL"
            print(f"  [{icon}] {gate.name}")
            print(f"        Value: {gate.value}")
            print(f"        Threshold: {gate.threshold}")

        print(f"\n  Full Depth: {'PASSED' if self.validation.full_depth.passed else 'FAILED'}")

        print("\nHelix Engine Gates:")
        print("-" * 50)
        for gate in [
            self.validation.helix_engine.min_negentropy,
            self.validation.helix_engine.min_final_z,
            self.validation.helix_engine.kappa_stable,
        ]:
            icon = "PASS" if gate.passed else "FAIL"
            print(f"  [{icon}] {gate.name}")
            print(f"        Value: {gate.value}")
            print(f"        Threshold: {gate.threshold}")

        print(f"\n  Helix Engine: {'PASSED' if self.validation.helix_engine.passed else 'FAILED'}")

        print("\n" + "=" * 70)
        overall = "PASSED" if self.validation.overall_passed else "FAILED"
        print(f"OVERALL: {overall}")
        print("=" * 70)
