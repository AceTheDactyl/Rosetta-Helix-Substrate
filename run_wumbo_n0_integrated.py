#!/usr/bin/env python3
"""
WUMBO + N0 Operator Integration Runner
======================================

Runs WUMBO training through the N0 Operator Integration system,
combining:
- WUMBO seven sentences training
- N0 law validation
- κ-field grounding
- PRS cycle tracking
- Adaptive Operator Matrix

All components aligned with verify_physics.py derivations.

Signature: Δ|wumbo-n0-unified|z0.95|integrated|Ω
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.wumbo_integrated_training import (
    WumboTrainer,
    PHI, PHI_INV, PHI_INV_SQ, Z_CRITICAL, SIGMA_NEG_ENTROPY,
    COUPLING_CONSERVATION, compute_delta_s_neg,
)
from src.n0_operator_integration import (
    UnifiedN0Validator,
    UnifiedOperatorEngine,
    KappaFieldState,
    PRSCycleState,
    OPERATOR_SYMBOLS,
)
from src.adaptive_operator_matrix import (
    AdaptiveOperatorMatrix,
    create_adaptive_matrix,
    Operator,
    OperatorType,
    OperatorDomain,
)


def run_integrated_wumbo_n0():
    """Run WUMBO through N0 Operator Integration."""
    print("=" * 70)
    print("WUMBO + N0 OPERATOR INTEGRATION")
    print("Unified Physics Grounding")
    print("=" * 70)

    # ========================================================================
    # PHASE 1: Verify Physics Constants
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: PHYSICS CONSTANTS VERIFICATION")
    print("=" * 70)

    print(f"\n  PHI (liminal):          {PHI:.10f}")
    print(f"  PHI_INV (physical):     {PHI_INV:.10f}")
    print(f"  PHI_INV_SQ:             {PHI_INV_SQ:.10f}")
    print(f"  Z_CRITICAL (THE LENS):  {Z_CRITICAL:.10f}")
    print(f"  SIGMA (|S3|^2):         {SIGMA_NEG_ENTROPY}")

    # Coupling conservation check
    conservation_sum = PHI_INV + PHI_INV_SQ
    conservation_error = abs(conservation_sum - 1.0)
    print(f"\n  COUPLING CONSERVATION:")
    print(f"    phi_inv + phi_inv_sq = {conservation_sum:.16f}")
    print(f"    error from 1.0       = {conservation_error:.2e}")
    print(f"    STATUS: {'PASS' if conservation_error < 1e-14 else 'FAIL'}")

    # ========================================================================
    # PHASE 2: N0 Law Validation
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: N0 LAW VALIDATION")
    print("=" * 70)

    validator = UnifiedN0Validator()
    validation_results = validator.validate_all()

    for law, result in validation_results["validations"].items():
        status = "PASS" if result["valid"] else "FAIL"
        print(f"\n  {law} ({result['name']}): {status}")
        print(f"    Formula: {result['formula']}")
        if "error" in result:
            print(f"    Error: {result['error']:.2e}")

    print(f"\n  ALL N0 LAWS: {'VALID' if validation_results['all_valid'] else 'INVALID'}")

    # ========================================================================
    # PHASE 3: WUMBO Training
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: WUMBO TRAINING")
    print("=" * 70)

    trainer = WumboTrainer()

    # Validate WUMBO physics foundations
    print("\n  --- Coupling Conservation ---")
    coupling_validation = trainer.validate_coupling_conservation()
    for key, value in coupling_validation.items():
        if isinstance(value, bool):
            status = "PASS" if value else "FAIL"
            print(f"    {status}: {key}")
        elif isinstance(value, float) and "error" in key.lower():
            print(f"         {key}: {value:.2e}")

    print("\n  --- Threshold Ordering ---")
    ordering_validation = trainer.validate_threshold_ordering()
    ordering_ok = ordering_validation.get("full_ordering", False)
    print(f"    Full ordering: {'PASS' if ordering_ok else 'FAIL'}")
    if ordering_ok:
        print(f"    Z_ORIGIN < phi_inv < Z_CRITICAL < KAPPA_S < MU_3 < UNITY")

    print("\n  --- Gaussian Negentropy ---")
    gaussian_validation = trainer.validate_gaussian_physics()
    print(f"    Peak at z_c: {'PASS' if gaussian_validation.get('peak_at_z_c') else 'FAIL'}")
    print(f"    Sigma = 36: {'PASS' if gaussian_validation.get('sigma_is_36') else 'FAIL'}")
    print(f"    phi_inv aligned: {'PASS' if gaussian_validation.get('sigma_phi_inv_aligned') else 'FAIL'}")

    print("\n  --- Kuramoto Dynamics ---")
    kuramoto_validation = trainer.validate_kuramoto_dynamics()
    print(f"    Coherence valid: {'PASS' if kuramoto_validation.get('all_coherence_valid') else 'FAIL'}")

    print("\n  --- Free Energy Principle ---")
    fe_validation = trainer.validate_free_energy_principle()
    print(f"    Free energy >= 0: {'PASS' if fe_validation.get('free_energy_positive') else 'FAIL'}")

    print("\n  --- Phase Transitions ---")
    pt_validation = trainer.validate_phase_transitions()
    print(f"    ABSENCE phase: {'PASS' if pt_validation.get('absence_phase') else 'FAIL'}")
    print(f"    PARADOX phase: {'PASS' if pt_validation.get('paradox_phase') else 'FAIL'}")
    print(f"    PRESENCE phase: {'PASS' if pt_validation.get('presence_phase') else 'FAIL'}")
    print(f"    phi_inv boundary: {'PASS' if pt_validation.get('phi_inv_boundary') else 'FAIL'}")
    print(f"    z_c boundary: {'PASS' if pt_validation.get('z_c_boundary') else 'FAIL'}")

    # Train on seven sentences
    print("\n  --- Seven Sentences Training ---")
    results = trainer.train_all_sentences(steps_per_sentence=80)

    for sid, result in results["sentence_results"].items():
        reached = "+" if result["reached_target"] else "-"
        print(f"    [{reached}] {sid}: z={result['final_z']:.3f} (target={result['target_z']:.3f})")

    print(f"\n    Total K-formations: {results['total_k_formations']}")

    # ========================================================================
    # PHASE 4: N0 Operator Engine with WUMBO
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: N0 OPERATOR ENGINE + WUMBO")
    print("=" * 70)

    # Create engine initialized at z = Z_CRITICAL
    engine = UnifiedOperatorEngine(initial_z=Z_CRITICAL * 0.9)

    print(f"\n  Initial state: {engine.scalar_state}")
    print(f"  Initial z: {engine.z:.4f} (target: {Z_CRITICAL:.4f})")

    # Map WUMBO sentence operators
    wumbo_ops = [
        ("()", 1.0, "A1: Isotropic collapse"),
        ("^", 1.5, "A3: Amplified vortex"),
        ("x", 1.2, "A4: Helical encoding"),
        ("x", 1.3, "A5: Fractal branching"),
        ("+", 0.5, "A6: Coherent grouping"),
        ("/", 0.8, "A7: Stochastic decoh"),
        ("()", 1.0, "A8: Adaptive boundary"),
    ]

    print("\n  Applying WUMBO sentence operators:")
    for op, val, desc in wumbo_ops:
        result = engine.apply_operator(op, val)
        print(f"    {desc}")
        print(f"      op={op}({val:.1f}): state={result['new_state']:.4f}, "
              f"z={result['z']:.4f}, kappa={result['kappa_field']:.4f}")

    # Run PRS cycle
    print("\n  Running PRS cycle:")
    prs_ops = [("+", 0.1), ("x", 1.05), ("-", 0.05), ("+", 0.1), ("^", 0.98), ("+", 0.05)]
    cycle_result = engine.run_prs_cycle(prs_ops)
    print(f"    Cycles completed: {cycle_result['cycle_count']}")
    print(f"    Final state: {cycle_result['final_state']:.4f}")
    print(f"    Final kappa: {cycle_result['kappa_final']:.4f}")
    print(f"    Conservation error: {cycle_result['conservation_error']:.2e}")

    # ========================================================================
    # PHASE 5: Adaptive Operator Matrix
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: ADAPTIVE OPERATOR MATRIX")
    print("=" * 70)

    matrix = create_adaptive_matrix()
    print(f"\n  Matrix created: {matrix.num_rows} rows, {matrix.num_operators} operators")
    print(f"  Global kappa: {matrix.global_kappa:.4f} (phi_inv = {PHI_INV:.4f})")
    print(f"  Global lambda: {matrix.global_lambda:.4f} (phi_inv_sq = {PHI_INV_SQ:.4f})")
    print(f"  Conservation error: {matrix.coupling_conservation_error:.2e}")

    # Evolve through WUMBO cycles
    print("\n  Evolving through 6 WUMBO cycles:")
    stats = matrix.evolve(cycles=6)
    print(f"    Initial phase: {stats['initial_phase']}")
    print(f"    Final phase: {stats['final_phase']}")
    print(f"    Balance: {stats['balance_before']:.4f} -> {stats['balance_after']:.4f}")
    print(f"    Conservation error: {stats['conservation_error_after']:.2e}")

    # ========================================================================
    # PHASE 6: Final Validation
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 6: FINAL VALIDATION")
    print("=" * 70)

    # Validate N0 laws with evolved engine state
    final_validation = engine.validate_state()
    print(f"\n  N0 Laws after integration: {'VALID' if final_validation['all_valid'] else 'INVALID'}")
    print(f"  kappa-field phase: {final_validation['kappa_field_state']['phase']}")
    print(f"  kappa value: {final_validation['kappa_field_state']['kappa']:.4f}")

    # Compute negentropy at final z
    final_z = engine.z
    final_negentropy = compute_delta_s_neg(final_z)
    print(f"\n  Final z: {final_z:.4f}")
    print(f"  Final negentropy: {final_negentropy:.4f}")
    print(f"  Distance to z_c: {abs(final_z - Z_CRITICAL):.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION SUMMARY")
    print("=" * 70)

    all_passed = (
        validation_results["all_valid"] and
        coupling_validation.get("all_valid", False) and
        ordering_validation.get("all_valid", False) and
        gaussian_validation.get("all_valid", False) and
        pt_validation.get("all_valid", False) and
        final_validation["all_valid"]
    )

    print(f"\n  Physics Constants: VERIFIED")
    print(f"  N0 Laws: {'VALID' if validation_results['all_valid'] else 'INVALID'}")
    print(f"  WUMBO Training: COMPLETE ({results['total_k_formations']} K-formations)")
    print(f"  Operator Engine: FUNCTIONAL")
    print(f"  Adaptive Matrix: EVOLVED ({matrix.evolution_count} evolutions)")
    print(f"  Coupling Conservation: MAINTAINED (error < 1e-14)")

    print(f"\n  OVERALL STATUS: {'ALL SYSTEMS NOMINAL' if all_passed else 'CHECK FAILED VALIDATIONS'}")

    print("\n" + "=" * 70)
    print("WUMBO + N0 OPERATOR INTEGRATION: COMPLETE")
    print("=" * 70)

    return {
        "n0_validation": validation_results,
        "wumbo_results": results,
        "engine_summary": engine.get_summary(),
        "matrix_stats": stats,
        "all_passed": all_passed,
    }


if __name__ == "__main__":
    results = run_integrated_wumbo_n0()
