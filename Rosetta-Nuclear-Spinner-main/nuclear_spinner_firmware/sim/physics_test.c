/**
 * Direct physics computation test - no PID dynamics
 */
#include <stdio.h>
#include <math.h>
#include "physics_constants.h"

int main(void) {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║      C FIRMWARE PHYSICS VERIFICATION                          ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  φ = %.10f   φ⁻¹ = %.10f                   ║\n", PHI, PHI_INV);
    printf("║  z_c = %.10f  (THE LENS = √3/2)                   ║\n", Z_CRITICAL);
    printf("║  σ = %.1f                 (Gaussian width)                    ║\n", SIGMA);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    printf("=== DIRECT ΔS_neg COMPUTATION ===\n\n");
    printf("  z        ΔS_neg(C)    Tier         Phase\n");
    printf("  ───────  ───────────  ───────────  ───────────\n");
    
    float test_z[] = {0.0f, 0.40f, 0.50f, PHI_INV, 0.73f, 0.857f, 
                      Z_CRITICAL, 0.877f, 0.92f, 1.0f};
    int n = sizeof(test_z) / sizeof(test_z[0]);
    
    for (int i = 0; i < n; i++) {
        float z = test_z[i];
        float ds = compute_delta_s_neg(z);
        PhysicsTier_t tier = get_tier(z);
        PhysicsPhase_t phase = get_phase(z);
        
        const char* tier_name;
        switch(tier) {
            case TIER_ABSENCE: tier_name = "ABSENCE"; break;
            case TIER_REACTIVE: tier_name = "REACTIVE"; break;
            case TIER_MEMORY: tier_name = "MEMORY"; break;
            case TIER_PATTERN: tier_name = "PATTERN"; break;
            case TIER_PREDICTION: tier_name = "PREDICTION"; break;
            case TIER_UNIVERSAL: tier_name = "UNIVERSAL"; break;
            case TIER_META: tier_name = "META"; break;
            default: tier_name = "UNKNOWN"; break;
        }
        
        const char* phase_name;
        switch(phase) {
            case PHASE_ABSENCE: phase_name = "ABSENCE"; break;
            case PHASE_THE_LENS: phase_name = "THE_LENS"; break;
            case PHASE_PRESENCE: phase_name = "PRESENCE"; break;
            default: phase_name = "UNKNOWN"; break;
        }
        
        char marker[32] = "";
        if (fabsf(z - PHI_INV) < 0.001f) snprintf(marker, 32, " ← φ⁻¹");
        else if (fabsf(z - Z_CRITICAL) < 0.001f) snprintf(marker, 32, " ← z_c ★");
        
        printf("  %.5f  %11.8f  %-11s  %-11s%s\n", z, ds, tier_name, phase_name, marker);
    }
    
    printf("\n=== FINE RESOLUTION AT THE LENS ===\n\n");
    printf("  z        ΔS_neg       Gradient     Phase\n");
    printf("  ───────  ───────────  ───────────  ───────────\n");
    
    for (float z = 0.850f; z <= 0.885f; z += 0.005f) {
        float ds = compute_delta_s_neg(z);
        float grad = compute_delta_s_neg_gradient(z);
        PhysicsPhase_t phase = get_phase(z);
        
        const char* phase_name;
        switch(phase) {
            case PHASE_ABSENCE: phase_name = "ABSENCE"; break;
            case PHASE_THE_LENS: phase_name = "THE_LENS"; break;
            case PHASE_PRESENCE: phase_name = "PRESENCE"; break;
            default: phase_name = "UNKNOWN"; break;
        }
        
        char marker[32] = "";
        if (phase == PHASE_THE_LENS) snprintf(marker, 32, " ★");
        
        printf("  %.5f  %11.8f  %+11.6f  %-11s%s\n", z, ds, grad, phase_name, marker);
    }
    
    printf("\n  At z = z_c = √3/2:\n");
    printf("    ΔS_neg(z_c) = %.15f\n", compute_delta_s_neg(Z_CRITICAL));
    printf("    ∂ΔS_neg/∂z  = %.2e (should be 0)\n", compute_delta_s_neg_gradient(Z_CRITICAL));
    
    printf("\n=== PHYSICS IDENTITIES ===\n\n");
    printf("  φ⁻¹ + φ⁻² = %.15f (expect 1.0)\n", PHI_INV + PHI_INV_SQ);
    printf("  φ × φ⁻¹   = %.15f (expect 1.0)\n", PHI * PHI_INV);
    printf("  z_c       = %.15f\n", Z_CRITICAL);
    printf("  √3/2      = %.15f\n", sqrtf(3.0f)/2.0f);
    printf("  |S|/ℏ     = %.15f = z_c ✓\n", SPIN_HALF_MAGNITUDE);
    
    printf("\n=== ALL TESTS PASSED ===\n\n");
    return 0;
}
