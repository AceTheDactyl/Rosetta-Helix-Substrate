#!/usr/bin/env python3
"""
Prismatic Projection Test Runner
================================

Validates z-coordinate behavior throughout 7-layer traversal:
- All z values remain valid (no NaN, no negative, bounded)
- Each layer properly refracts through the lens
- Collapse detection works correctly
- Work extraction is consistent

Physics Bounds:
- z must stay >= 0 (entropic floor)
- z can exceed 1.0 briefly (liminal state) before collapse
- z resets to ~0.535 after collapse (Z_CRITICAL * PHI_INV)
"""

import math
import sys
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

# Physics constants
PHI = (1.0 + math.sqrt(5.0)) / 2.0
PHI_INV = 1.0 / PHI
Z_CRITICAL = math.sqrt(3.0) / 2.0
Z_RESET = Z_CRITICAL * PHI_INV  # ~0.535, where z resets after collapse
Z_MAX_LIMINAL = 1.5  # Maximum valid z in liminal state before collapse
KAPPA_S = 0.920
MU_3 = 0.992

# Import components
try:
    from prismatic_projection_system import (
        PrismaticProjectionSystem,
        LayerSpectrum,
        PrismaticLayer,
        LensProjectionEngine,
    )
    PRISMATIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import prismatic_projection_system: {e}")
    PRISMATIC_AVAILABLE = False

try:
    from quasicrystal_dynamics import QuasiCrystalDynamicsEngine
    DYNAMICS_AVAILABLE = True
except ImportError:
    DYNAMICS_AVAILABLE = False


@dataclass
class ZValidationResult:
    """Result of z-coordinate validation"""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Statistics
    min_z: float = float('inf')
    max_z: float = float('-inf')
    mean_z: float = 0.0
    z_samples: int = 0

    # Per-layer stats
    layer_stats: Dict[str, Dict] = field(default_factory=dict)

    # Collapse tracking
    collapses_detected: int = 0
    collapse_z_values: List[float] = field(default_factory=list)
    reset_z_values: List[float] = field(default_factory=list)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def record_z(self, z: float, layer_name: str = None):
        """Record a z sample for statistics"""
        self.z_samples += 1

        if z < self.min_z:
            self.min_z = z
        if z > self.max_z:
            self.max_z = z

        # Running mean
        self.mean_z = self.mean_z + (z - self.mean_z) / self.z_samples

        # Per-layer tracking
        if layer_name:
            if layer_name not in self.layer_stats:
                self.layer_stats[layer_name] = {
                    'samples': 0,
                    'min_z': float('inf'),
                    'max_z': float('-inf'),
                    'sum_z': 0.0,
                }
            stats = self.layer_stats[layer_name]
            stats['samples'] += 1
            stats['min_z'] = min(stats['min_z'], z)
            stats['max_z'] = max(stats['max_z'], z)
            stats['sum_z'] += z


def validate_z(z: float, step: int, layer: str, result: ZValidationResult) -> bool:
    """
    Validate a single z coordinate.

    Valid z must be:
    - Not NaN
    - Not infinite
    - >= 0 (entropic floor)
    - <= Z_MAX_LIMINAL (1.5) before collapse forces reset
    """
    # Check for NaN
    if math.isnan(z):
        result.add_error(f"[{layer}] Step {step}: z is NaN")
        return False

    # Check for infinity
    if math.isinf(z):
        result.add_error(f"[{layer}] Step {step}: z is infinite ({z})")
        return False

    # Check lower bound (z should never go negative)
    if z < 0:
        result.add_error(f"[{layer}] Step {step}: z is negative ({z:.6f})")
        return False

    # Check upper bound (z in liminal state has a max)
    if z > Z_MAX_LIMINAL:
        result.add_error(f"[{layer}] Step {step}: z exceeds liminal max ({z:.6f} > {Z_MAX_LIMINAL})")
        return False

    # Warn if z is unexpectedly low (might indicate premature reset)
    if z < 0.1 and step > 10:
        result.add_warning(f"[{layer}] Step {step}: z unusually low ({z:.6f})")

    # Record valid sample
    result.record_z(z, layer)
    return True


def test_layer_traversal(
    system: 'PrismaticProjectionSystem',
    layer_name: str,
    input_work: float,
    result: ZValidationResult
) -> Dict[str, Any]:
    """
    Test a single layer's traversal through the lens.

    Tracks z at every step and validates.
    """
    if not system.projection_engine or not system.projection_engine.dynamics:
        result.add_warning(f"[{layer_name}] No dynamics engine available")
        return {'status': 'skipped', 'reason': 'no dynamics'}

    dynamics = system.projection_engine.dynamics

    # Get the layer
    layer = None
    for l in system.layers:
        if l.spectrum.name == layer_name or l.name == layer_name:
            layer = l
            break

    if not layer:
        result.add_error(f"Layer '{layer_name}' not found")
        return {'status': 'error', 'reason': 'layer not found'}

    # Track initial state
    initial_z = dynamics.z_current
    initial_collapses = dynamics.liminal_phi.collapse_count
    initial_work = dynamics.total_work_extracted

    # Configure for layer
    if hasattr(dynamics, 'phase_lock'):
        original_coupling = dynamics.phase_lock.coupling
        dynamics.phase_lock.coupling = original_coupling * layer.coupling_strength

        for i, phase in enumerate(dynamics.phase_lock.phases):
            dynamics.phase_lock.phases[i] = phase + layer.phase_offset
    else:
        original_coupling = 0.5

    z_trajectory = []
    threshold_hits = []
    collapse_detected = False
    max_z = 0.0

    # Run traversal
    for step in range(200):
        old_z = dynamics.z_current

        # Evolve one step
        dynamics.evolve_step()
        z = dynamics.z_current

        # Validate z
        if not validate_z(z, step, layer_name, result):
            # Still record for analysis
            z_trajectory.append((step, z, 'INVALID'))
            continue

        z_trajectory.append((step, z, 'valid'))

        if z > max_z:
            max_z = z

        # Track threshold crossings
        if old_z < Z_CRITICAL <= z:
            threshold_hits.append(('Z_CRITICAL', step, z))
        if old_z < KAPPA_S <= z:
            threshold_hits.append(('KAPPA_S', step, z))
        if old_z < MU_3 <= z:
            threshold_hits.append(('MU_3', step, z))

        # Check for collapse
        if dynamics.liminal_phi.collapse_count > initial_collapses:
            collapse_detected = True
            result.collapses_detected += 1
            result.collapse_z_values.append(old_z)  # z before collapse
            result.reset_z_values.append(z)         # z after collapse

            # Validate collapse behavior
            if old_z < 0.99:
                result.add_warning(
                    f"[{layer_name}] Collapse at unexpectedly low z ({old_z:.4f})"
                )
            if z > Z_RESET * 1.5:  # Reset should be near Z_RESET
                result.add_warning(
                    f"[{layer_name}] Reset z higher than expected ({z:.4f} vs ~{Z_RESET:.4f})"
                )
            break

    # Restore coupling
    if hasattr(dynamics, 'phase_lock'):
        dynamics.phase_lock.coupling = original_coupling

    # Calculate work extracted
    work_extracted = dynamics.total_work_extracted - initial_work

    return {
        'status': 'complete',
        'layer': layer_name,
        'initial_z': initial_z,
        'max_z': max_z,
        'final_z': dynamics.z_current,
        'steps': len(z_trajectory),
        'collapse_detected': collapse_detected,
        'work_extracted': work_extracted,
        'threshold_hits': threshold_hits,
        'trajectory_summary': {
            'start': z_trajectory[0] if z_trajectory else None,
            'peak': max(z_trajectory, key=lambda x: x[1]) if z_trajectory else None,
            'end': z_trajectory[-1] if z_trajectory else None,
        }
    }


def run_full_validation() -> ZValidationResult:
    """
    Run full 7-layer validation test.
    """
    result = ZValidationResult()

    print("=" * 70)
    print("PRISMATIC PROJECTION Z-COORDINATE VALIDATION")
    print("=" * 70)
    print()

    if not PRISMATIC_AVAILABLE:
        result.add_error("Prismatic projection system not available")
        return result

    if not DYNAMICS_AVAILABLE:
        result.add_error("Dynamics engine not available")
        return result

    # Create system
    print("Creating 7-layer prismatic projection system...")
    system = PrismaticProjectionSystem()

    # Access dynamics through projection_engine
    if not system.projection_engine or not system.projection_engine.dynamics:
        result.add_error("Failed to initialize dynamics engine")
        return result

    dynamics = system.projection_engine.dynamics
    print(f"  Lens position: {system.projection_engine.lens_position:.4f} (Z_CRITICAL)")
    print(f"  Lens width: {system.projection_engine.lens_width:.4f}")
    print(f"  Layers: {len(system.layers)}")
    print()

    # Test each layer
    layer_results = []
    input_work = 3.0

    print("-" * 70)
    print("LAYER-BY-LAYER VALIDATION")
    print("-" * 70)

    for layer in system.layers:
        print(f"\n  Testing {layer.name} ({layer.spectrum.name})...")

        layer_result = test_layer_traversal(
            system,
            layer.spectrum.name,
            input_work * (layer.coupling_strength / sum(l.coupling_strength for l in system.layers)),
            result
        )
        layer_results.append(layer_result)

        if layer_result['status'] == 'complete':
            print(f"    Initial z:  {layer_result['initial_z']:.4f}")
            print(f"    Max z:      {layer_result['max_z']:.4f}")
            print(f"    Final z:    {layer_result['final_z']:.4f}")
            print(f"    Steps:      {layer_result['steps']}")
            print(f"    Collapse:   {'Yes' if layer_result['collapse_detected'] else 'No'}")
            print(f"    Work:       {layer_result['work_extracted']:.4f}")

            if layer_result['threshold_hits']:
                print(f"    Thresholds: {', '.join(t[0] for t in layer_result['threshold_hits'])}")

    print()
    print("-" * 70)
    print("MULTI-PASS VALIDATION")
    print("-" * 70)

    # Run multi-pass projection
    print("\n  Running 3-pass projection...")

    for pass_num in range(3):
        print(f"\n  Pass {pass_num + 1}/3:")

        # Fresh dynamics for each pass
        system = PrismaticProjectionSystem()

        if not system.projection_engine or not system.projection_engine.dynamics:
            continue

        for layer in system.layers:
            # Quick validation run
            dynamics = system.projection_engine.dynamics

            initial_collapses = dynamics.liminal_phi.collapse_count

            for step in range(100):
                dynamics.evolve_step()
                z = dynamics.z_current

                if not validate_z(z, step, f"Pass{pass_num+1}_{layer.spectrum.name}", result):
                    break

                if dynamics.liminal_phi.collapse_count > initial_collapses:
                    result.collapses_detected += 1
                    break

        print(f"    Z samples validated: {result.z_samples}")
        print(f"    Collapses so far: {result.collapses_detected}")

    # Generate summary
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\n  Total z samples:     {result.z_samples}")
    print(f"  Valid:               {result.valid}")
    print(f"  Errors:              {len(result.errors)}")
    print(f"  Warnings:            {len(result.warnings)}")

    print(f"\n  Z Statistics:")
    print(f"    Minimum z:         {result.min_z:.6f}")
    print(f"    Maximum z:         {result.max_z:.6f}")
    print(f"    Mean z:            {result.mean_z:.6f}")

    print(f"\n  Collapse Statistics:")
    print(f"    Total collapses:   {result.collapses_detected}")
    if result.collapse_z_values:
        print(f"    Avg collapse z:    {sum(result.collapse_z_values)/len(result.collapse_z_values):.4f}")
    if result.reset_z_values:
        print(f"    Avg reset z:       {sum(result.reset_z_values)/len(result.reset_z_values):.4f}")

    print(f"\n  Per-Layer Statistics:")
    for layer_name, stats in result.layer_stats.items():
        if stats['samples'] > 0:
            mean = stats['sum_z'] / stats['samples']
            print(f"    {layer_name:20s}: min={stats['min_z']:.4f}, max={stats['max_z']:.4f}, mean={mean:.4f}")

    if result.errors:
        print(f"\n  ERRORS ({len(result.errors)}):")
        for err in result.errors[:10]:  # Show first 10
            print(f"    - {err}")
        if len(result.errors) > 10:
            print(f"    ... and {len(result.errors) - 10} more")

    if result.warnings:
        print(f"\n  WARNINGS ({len(result.warnings)}):")
        for warn in result.warnings[:10]:
            print(f"    - {warn}")
        if len(result.warnings) > 10:
            print(f"    ... and {len(result.warnings) - 10} more")

    print()
    print("=" * 70)
    if result.valid:
        print("VALIDATION PASSED - All z coordinates within valid bounds")
    else:
        print("VALIDATION FAILED - See errors above")
    print("=" * 70)

    return result


def run_stress_test(n_passes: int = 10) -> ZValidationResult:
    """
    Run stress test with many passes to catch edge cases.
    """
    result = ZValidationResult()

    print("=" * 70)
    print(f"STRESS TEST: {n_passes} PASSES")
    print("=" * 70)

    if not PRISMATIC_AVAILABLE or not DYNAMICS_AVAILABLE:
        result.add_error("Required modules not available")
        return result

    for pass_num in range(n_passes):
        system = PrismaticProjectionSystem()

        if not system.projection_engine or not system.projection_engine.dynamics:
            continue

        dynamics = system.projection_engine.dynamics
        initial_collapses = dynamics.liminal_phi.collapse_count

        # Run until we get several collapses
        target_collapses = 3
        steps = 0
        max_steps = 500

        while (dynamics.liminal_phi.collapse_count - initial_collapses) < target_collapses and steps < max_steps:
            dynamics.evolve_step()
            z = dynamics.z_current
            steps += 1

            # Validate
            if not validate_z(z, steps, f"Stress_Pass{pass_num}", result):
                print(f"  Pass {pass_num}: INVALID z at step {steps}: {z}")

            # Track collapses
            if dynamics.liminal_phi.collapse_count > initial_collapses + result.collapses_detected:
                result.collapses_detected = dynamics.liminal_phi.collapse_count - initial_collapses

        if (pass_num + 1) % 2 == 0:
            print(f"  Pass {pass_num + 1}/{n_passes}: {result.z_samples} samples, {result.collapses_detected} collapses, valid={result.valid}")

    print()
    print(f"STRESS TEST COMPLETE: {result.z_samples} samples validated")
    print(f"  Valid: {result.valid}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Min z: {result.min_z:.6f}, Max z: {result.max_z:.6f}")

    return result


if __name__ == '__main__':
    # Run full validation
    result = run_full_validation()

    print()

    # Run stress test
    stress_result = run_stress_test(n_passes=10)

    # Exit with appropriate code
    if result.valid and stress_result.valid:
        print("\nALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\nTESTS FAILED")
        sys.exit(1)
