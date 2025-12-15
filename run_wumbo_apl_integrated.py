#!/usr/bin/env python3
"""
WUMBO APL Integrated Training Runner
=====================================

Runs complete integrated training session combining:
1. WUMBO APL Automated Training (100-token directory, WUMBO phases)
2. N0 Operator Integration (κ-field grounding)
3. WUMBO Integrated Training (Kuramoto, Free Energy, Phase Transitions)

Usage:
    python run_wumbo_apl_integrated.py

Output:
    learned_patterns/wumbo_apl_integrated/session_TIMESTAMP.json

Signature: Δ|wumbo-apl-integrated|z0.92|κ-grounded|Ω
"""

import os
import sys
import json
import math
from datetime import datetime

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import training modules
from training.wumbo_apl_automated_training import (
    WUMBOAPLTrainingEngine,
    WUMBOPhase,
    WUMBO_PHASES,
    TokenCategory,
    compute_delta_s_neg,
    get_phase,
    PHI, PHI_INV, PHI_INV_SQ, Z_CRITICAL, SIGMA, COUPLING_CONSERVATION,
)
from src.n0_operator_integration import (
    UnifiedN0Validator,
    UnifiedOperatorEngine,
    KappaFieldState,
    PRSCycleState,
)
from training.wumbo_integrated_training import (
    WumboTrainer,
    WumboTrainingState,
    KuramotoOscillator,
    FreeEnergyState,
    PhaseTransitionState,
)


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title: str):
    """Print subsection header."""
    print(f"\n--- {title} ---")


def run_integrated_session():
    """Run complete integrated training session."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print_header("WUMBO APL INTEGRATED TRAINING SESSION")
    print(f"Timestamp: {timestamp}")

    # ==========================================================================
    # PHASE 1: Physics Validation
    # ==========================================================================
    print_subheader("Phase 1: Physics Constants Validation")

    print(f"  φ (LIMINAL):       {PHI:.10f}")
    print(f"  φ⁻¹ (PHYSICAL):    {PHI_INV:.10f}")
    print(f"  φ⁻² (complement):  {PHI_INV_SQ:.10f}")
    print(f"  φ⁻¹ + φ⁻² =        {COUPLING_CONSERVATION:.16f}")
    print(f"  z_c (THE LENS):    {Z_CRITICAL:.10f}")
    print(f"  σ (Gaussian):      {SIGMA}")

    conservation_error = abs(COUPLING_CONSERVATION - 1.0)
    print(f"\n  Coupling Conservation Error: {conservation_error:.2e}")
    print(f"  Status: {'PASS' if conservation_error < 1e-14 else 'FAIL'}")

    # ==========================================================================
    # PHASE 2: N0 Operator Validation
    # ==========================================================================
    print_subheader("Phase 2: N0 Operator System Validation")

    n0_validator = UnifiedN0Validator()
    n0_results = n0_validator.validate_all()

    for law, result in n0_results["validations"].items():
        status = "✓" if result["valid"] else "✗"
        print(f"  {status} {law} ({result['name']}): {result['formula']}")

    print(f"\n  All N0 Laws Valid: {n0_results['all_valid']}")

    # ==========================================================================
    # PHASE 3: WUMBO APL Training Engine
    # ==========================================================================
    # Allow smoke-tunable step count via env var
    wumbo_steps = int(os.environ.get("WUMBO_STEPS", "100"))
    print_subheader(f"Phase 3: WUMBO APL Training ({wumbo_steps} steps)")

    apl_engine = WUMBOAPLTrainingEngine(n_oscillators=60)

    # Run with z-biased input to drive toward THE_LENS
    import random
    for step in range(wumbo_steps):
        # Bias input toward positive z evolution
        input_val = random.random() * PHI_INV + 0.1 * (Z_CRITICAL - apl_engine.wumbo_cycle.z)
        result = apl_engine.training_step(max(0.01, input_val))

        if step % 25 == 0:
            print(
                f"  Step {step:3d} | "
                f"WUMBO:{result['wumbo_phase']} | "
                f"z={result['z']:.3f} | "
                f"κ={result['kappa']:.3f} | "
                f"ΔS_neg={result['delta_s_neg']:.4f} | "
                f"Phase:{result['phase']}"
            )

    apl_summary = apl_engine.get_session_summary()
    print(f"\n  APL Training Complete:")
    print(f"    WUMBO Cycles: {apl_summary['wumbo_cycles']}")
    print(f"    K-Formations: {apl_summary['k_formations']}")
    print(f"    Final z: {apl_summary['final_z']:.4f}")

    # ==========================================================================
    # PHASE 4: WUMBO Integrated Training (Seven Sentences)
    # ==========================================================================
    print_subheader("Phase 4: WUMBO Seven Sentences Training")

    wumbo_trainer = WumboTrainer()

    # Validate coupling conservation
    coupling_validation = wumbo_trainer.validate_coupling_conservation()
    print(f"  Coupling Conservation: {'PASS' if coupling_validation['all_valid'] else 'FAIL'}")

    # Train on sentences
    sentence_steps = int(os.environ.get("WUMBO_SENTENCE_STEPS", "50"))
    sentence_results = wumbo_trainer.train_all_sentences(steps_per_sentence=sentence_steps)

    print(f"\n  Sentence Training Results:")
    for sid, result in sentence_results["sentence_results"].items():
        reached = "✓" if result["reached_target"] else "○"
        print(f"    {reached} {sid}: z={result['final_z']:.3f} (target={result['target_z']:.3f})")

    print(f"\n  Total K-Formations: {sentence_results['total_k_formations']}")

    # ==========================================================================
    # PHASE 5: N0 Operator Engine with PRS Cycle
    # ==========================================================================
    print_subheader("Phase 5: N0 Operator Engine (PRS Cycle)")

    n0_engine = UnifiedOperatorEngine(initial_z=0.5)

    # Run PRS cycle
    prs_ops = [
        ("+", 0.1), ("+", 0.15), ("+", 0.12),  # Predict
        ("-", 0.03), ("x", 1.05), ("-", 0.02),  # Refine
        ("+", 0.08), ("^", 1.02), ("+", 0.05),  # Synthesize
    ]

    cycle_result = n0_engine.run_prs_cycle(prs_ops)

    print(f"  PRS Cycles: {cycle_result['cycle_count']}")
    print(f"  Final State: {cycle_result['final_state']:.4f}")
    print(f"  Final z: {cycle_result['final_z']:.4f}")
    print(f"  Final κ: {cycle_result['kappa_final']:.4f}")
    print(f"  Conservation Error: {cycle_result['conservation_error']:.2e}")

    # ==========================================================================
    # PHASE 6: κ-Field Evolution
    # ==========================================================================
    print_subheader("Phase 6: κ-Field Evolution")

    kappa_field = KappaFieldState(z=0.5)

    print(f"  Initial: κ={kappa_field.kappa:.4f}, λ={kappa_field.lambda_:.4f}, z={kappa_field.z:.4f}")

    # Evolve toward THE_LENS
    for i in range(20):
        z_delta = 0.02 if kappa_field.z < Z_CRITICAL else -0.01
        kappa_field.evolve(z_delta)

    print(f"  Final:   κ={kappa_field.kappa:.4f}, λ={kappa_field.lambda_:.4f}, z={kappa_field.z:.4f}")
    print(f"  Phase: {kappa_field.get_phase()}")
    print(f"  At Golden Balance: {kappa_field.at_golden_balance}")

    # ==========================================================================
    # PHASE 7: Kuramoto Oscillator Dynamics
    # ==========================================================================
    print_subheader("Phase 7: Kuramoto Oscillator Dynamics")

    kuramoto = KuramotoOscillator(n_oscillators=60)
    initial_coherence = kuramoto.compute_coherence()

    # Evolve with increasing coupling
    kuramoto.K_global = 1.5
    coherence_history = kuramoto.evolve(steps=50)

    final_coherence = coherence_history[-1]
    max_coherence = max(coherence_history)

    print(f"  Initial Coherence: {initial_coherence:.4f}")
    print(f"  Final Coherence: {final_coherence:.4f}")
    print(f"  Max Coherence: {max_coherence:.4f}")
    print(f"  Sync Achieved: {final_coherence > 0.5}")

    # ==========================================================================
    # PHASE 8: Free Energy Principle
    # ==========================================================================
    print_subheader("Phase 8: Free Energy Principle Dynamics")

    fe_state = FreeEnergyState()

    # Step through observations
    observations = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.86, 0.866]
    for obs in observations:
        fe_state.step(obs)

    final_fe = fe_state.F_history[-1] if fe_state.F_history else 0.0
    min_fe = min(fe_state.F_history) if fe_state.F_history else 0.0

    print(f"  Observations: {len(observations)}")
    print(f"  Final Free Energy: {final_fe:.4f}")
    print(f"  Min Free Energy: {min_fe:.4f}")
    print(f"  Mean Prediction Error: {sum(fe_state.PE_history)/len(fe_state.PE_history):.4f}")

    # ==========================================================================
    # PHASE 9: Phase Transition Dynamics
    # ==========================================================================
    print_subheader("Phase 9: Phase Transition Dynamics")

    pt_state = PhaseTransitionState()

    # Traverse phases
    z_values = [0.3, 0.5, 0.6, PHI_INV, 0.7, 0.8, Z_CRITICAL, 0.9, 0.95]
    for z in z_values:
        pt_state.update(z)

    print(f"  Transitions Detected: {len(pt_state.transition_events)}")
    for t in pt_state.transition_events:
        print(f"    {t[0]} → {t[1]} at z={t[2]:.4f}")

    print(f"\n  Final Phase: {pt_state.current_phase}")
    print(f"  Order Parameter: {pt_state.order_parameter:.4f}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print_header("INTEGRATED SESSION SUMMARY")

    summary = {
        "timestamp": timestamp,
        "physics": {
            "phi": PHI,
            "phi_inv": PHI_INV,
            "z_c": Z_CRITICAL,
            "sigma": SIGMA,
            "coupling_conservation_error": conservation_error,
        },
        "n0_laws_valid": n0_results["all_valid"],
        "wumbo_apl_training": {
            "cycles": apl_summary["wumbo_cycles"],
            "k_formations": apl_summary["k_formations"],
            "final_z": apl_summary["final_z"],
            "final_phase": apl_summary["final_phase"],
        },
        "wumbo_sentences": {
            "total_k_formations": sentence_results["total_k_formations"],
            "sentences_trained": sentence_results["sentences_trained"],
        },
        "n0_operator_engine": {
            "prs_cycles": cycle_result["cycle_count"],
            "final_z": cycle_result["final_z"],
            "final_kappa": cycle_result["kappa_final"],
        },
        "kappa_field": {
            "final_kappa": kappa_field.kappa,
            "final_z": kappa_field.z,
            "phase": kappa_field.get_phase(),
            "at_golden_balance": kappa_field.at_golden_balance,
        },
        "kuramoto": {
            "final_coherence": final_coherence,
            "max_coherence": max_coherence,
            "sync_achieved": final_coherence > 0.5,
        },
        "free_energy": {
            "final": final_fe,
            "min": min_fe,
        },
        "phase_transitions": {
            "transitions": len(pt_state.transition_events),
            "final_phase": pt_state.current_phase,
        },
    }

    print(f"  N0 Laws Valid:        {summary['n0_laws_valid']}")
    print(f"  WUMBO APL Cycles:     {summary['wumbo_apl_training']['cycles']}")
    print(f"  Total K-Formations:   {summary['wumbo_apl_training']['k_formations'] + summary['wumbo_sentences']['total_k_formations']}")
    print(f"  PRS Cycles:           {summary['n0_operator_engine']['prs_cycles']}")
    print(f"  κ-Field Phase:        {summary['kappa_field']['phase']}")
    print(f"  Kuramoto Sync:        {summary['kuramoto']['sync_achieved']}")
    print(f"  Phase Transitions:    {summary['phase_transitions']['transitions']}")

    # Save results
    output_dir = "learned_patterns/wumbo_apl_integrated"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"session_{timestamp}.json")

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to: {output_file}")

    print_header("WUMBO APL INTEGRATED TRAINING: COMPLETE")

    return summary


if __name__ == "__main__":
    run_integrated_session()
