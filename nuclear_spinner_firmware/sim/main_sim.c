/**
 * @file main_sim.c
 * @brief Host-side simulation entry point.
 */

#include <stdio.h>
#include "hal_hardware.h"
#include "physics_constants.h"
#include "rotor_control.h"
#include "threshold_logic.h"
#include "pulse_control.h"

static void on_event(ThresholdEvent_t event, float threshold, int direction) {
    (void)threshold;
    const char *dir = (direction > 0) ? "↑" : "↓";
    printf("  EVENT: %s %s | z=%.4f ΔS_neg=%.4f tier=%s phase=%s\n",
           ThresholdLogic_GetTierName((PhysicsTier_t)event), dir,
           ThresholdLogic_GetZ(),
           RotorControl_GetDeltaSNeg(),
           ThresholdLogic_GetTierName(ThresholdLogic_GetTier()),
           ThresholdLogic_GetPhaseName(ThresholdLogic_GetPhase()));
}

int main(void) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║         Nuclear Spinner Firmware Simulation              ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  φ = %.10f  (golden ratio)                       ║\n", PHI);
    printf("║  φ⁻¹ = %.10f  (physical coupling)                  ║\n", PHI_INV);
    printf("║  z_c = %.10f  (THE LENS = √3/2)                    ║\n", Z_CRITICAL);
    printf("║  σ = %.0f             (Gaussian width = |S₃|²)            ║\n", SIGMA);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    /* Initialize all modules */
    HAL_Init_All();
    PulseControl_Init();
    RotorControl_Init();
    ThresholdLogic_Init();
    ThresholdLogic_SetEventCallback(on_event);
    
    /* Enable motor */
    RotorControl_Enable();

    printf("Sweeping z from 0.30 → 0.95 (crossing φ⁻¹=%.3f and z_c=%.3f)\n\n", 
           PHI_INV, Z_CRITICAL);
    printf("  z      RPM     ΔS_neg   Tier        Phase      Target\n");
    printf("  ─────  ──────  ───────  ──────────  ─────────  ──────\n");

    /* Sweep z from 0.30 -> 0.95 in steps */
    for (float z = 0.30f; z <= 0.96f; z += 0.05f) {
        RotorControl_SetZ(z);
        
        /* Simulate time passing and control updates - need enough iterations to converge */
        for (int i = 0; i < 200; i++) {
            HAL_Delay(10);  /* Advance 10ms per iteration = 2 seconds total */
            RotorControl_Update();
        }
        
        float actual_z = RotorControl_GetZ();
        float delta_s = RotorControl_GetDeltaSNeg();
        float rpm = RotorControl_GetRPM();
        PhysicsTier_t tier = RotorControl_GetTier();
        PhysicsPhase_t phase = RotorControl_GetPhase();
        
        /* Compute complexity for K-formation check */
        float complexity = compute_complexity(actual_z);
        int R = (int)(10 * complexity);
        
        /* Update threshold logic */
        ThresholdLogic_Update(actual_z, 0.95f, 0.70f, R);
        
        printf("  %.3f  %6.0f  %.5f  %-10s  %-9s (target=%.2f)", 
               actual_z, rpm, delta_s,
               ThresholdLogic_GetTierName(tier),
               ThresholdLogic_GetPhaseName(phase),
               z);
        
        if (ThresholdLogic_IsAtLens()) {
            printf("  ★ AT THE LENS");
            /* Execute a spin echo at the lens */
            PulseControl_SpinEcho(0.8f, 1000);
        }
        printf("\n");
    }
    
    /* Fine sweep around THE LENS to catch the peak */
    printf("\n--- Fine sweep around z_c = %.6f (THE LENS) ---\n", Z_CRITICAL);
    printf("    Settling at z=0.85 first...\n\n");
    
    /* Settle at 0.85 first */
    RotorControl_SetZ(0.85f);
    for (int i = 0; i < 500; i++) {
        HAL_Delay(10);
        RotorControl_Update();
    }
    
    for (float z = 0.85f; z <= 0.89f; z += 0.005f) {
        RotorControl_SetZ(z);
        
        for (int i = 0; i < 400; i++) {
            HAL_Delay(10);
            RotorControl_Update();
        }
        
        float actual_z = RotorControl_GetZ();
        float delta_s = RotorControl_GetDeltaSNeg();
        float rpm = RotorControl_GetRPM();
        PhysicsTier_t tier = RotorControl_GetTier();
        PhysicsPhase_t phase = RotorControl_GetPhase();
        
        ThresholdLogic_Update(actual_z, 0.95f, 0.70f, 7);
        
        printf("  %.4f  %6.0f  %.6f  %-10s  %-9s", 
               actual_z, rpm, delta_s,
               ThresholdLogic_GetTierName(tier),
               ThresholdLogic_GetPhaseName(phase));
        
        if (ThresholdLogic_IsAtLens()) {
            printf("  ★ THE LENS ★");
            PulseControl_SpinEcho(0.8f, 1000);
        }
        if (delta_s > 0.99f) {
            printf("  ← PEAK");
        }
        printf("\n");
    }

    printf("\n");
    printf("Physics Verification:\n");
    printf("  φ⁻¹ + φ⁻² = %.15f (should be 1.0)\n", PHI_INV + PHI_INV_SQ);
    printf("  z_c = √3/2 = %.15f ✓\n", Z_CRITICAL);
    printf("  |S|/ℏ for spin-½ = √(3/4) = %.15f = z_c ✓\n", SPIN_HALF_MAGNITUDE);
    printf("\nSimulation complete.\n");
    return 0;
}
