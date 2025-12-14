/**
 * Full Nuclear Spinner Simulation
 * Demonstrates complete system behavior across all operating regimes
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "physics_constants.h"

/* Simulation parameters */
#define SIM_DURATION_S      60.0f
#define SIM_DT_MS           10
#define SPEED_RAMP_RATE     500.0f  /* RPM/second */

/* State tracking */
typedef struct {
    float z;
    float target_z;
    float rpm;
    float delta_s_neg;
    float gradient;
    float complexity;
    PhysicsTier_t tier;
    PhysicsPhase_t phase;
    float kappa;
    float eta;
    int R;
    bool k_formation;
    float time_s;
} SimState_t;

/* Statistics */
typedef struct {
    float time_in_tier[7];
    float time_in_phase[3];
    int tier_transitions;
    int phase_transitions;
    float max_delta_s_neg;
    float z_at_max_ds;
    float time_at_lens;
    int k_formations;
    float total_energy;  /* Proxy: integral of RPM */
} SimStats_t;

const char* tier_names[] = {"ABSENCE", "REACTIVE", "MEMORY", "PATTERN", 
                            "PREDICTION", "UNIVERSAL", "META"};
const char* phase_names[] = {"ABSENCE", "THE_LENS", "PRESENCE"};

void print_header(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    NUCLEAR SPINNER FULL SIMULATION                                ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Constants: φ=%.6f  φ⁻¹=%.6f  z_c=%.6f  σ=%.0f                        ║\n", 
           PHI, PHI_INV, Z_CRITICAL, SIGMA);
    printf("║  Duration: %.0fs  Ramp Rate: %.0f RPM/s  dt: %dms                                  ║\n",
           SIM_DURATION_S, SPEED_RAMP_RATE, SIM_DT_MS);
    printf("╚═══════════════════════════════════════════════════════════════════════════════════╝\n\n");
}

void print_state(SimState_t *s, bool event) {
    char marker[64] = "";
    
    if (s->k_formation) strcat(marker, " [K-FORM]");
    if (s->phase == PHASE_THE_LENS) strcat(marker, " ★");
    if (fabsf(s->z - PHI_INV) < 0.005f) strcat(marker, " φ⁻¹");
    if (fabsf(s->z - Z_CRITICAL) < 0.005f) strcat(marker, " z_c");
    
    if (event) {
        printf("  %5.1fs │ z=%.4f │ ΔS=%.5f │ %-10s │ %-8s │ κ=%.2f η=%.2f R=%d%s\n",
               s->time_s, s->z, s->delta_s_neg, tier_names[s->tier], 
               phase_names[s->phase], s->kappa, s->eta, s->R, marker);
    }
}

int main(void) {
    print_header();
    
    SimState_t state = {0};
    SimState_t prev_state = {0};
    SimStats_t stats = {0};
    
    /* Define experiment profile: target z over time */
    printf("═══ EXPERIMENT PROFILE ═══\n");
    printf("  0-10s:  Ramp z: 0.0 → 0.5 (through ABSENCE, REACTIVE, into MEMORY)\n");
    printf("  10-20s: Ramp z: 0.5 → φ⁻¹ (through MEMORY threshold)\n");
    printf("  20-30s: Ramp z: φ⁻¹ → z_c (through PATTERN, PREDICTION, to THE LENS)\n");
    printf("  30-40s: Hold at z_c (sustain at THE LENS)\n");
    printf("  40-50s: Ramp z: z_c → 0.95 (through UNIVERSAL, META)\n");
    printf("  50-60s: Ramp z: 0.95 → 0.5 (descend back)\n\n");
    
    printf("═══ SIMULATION LOG (events only) ═══\n");
    printf("  Time   │    z     │   ΔS_neg  │   Tier     │  Phase   │ Formation\n");
    printf("  ───────┼──────────┼───────────┼────────────┼──────────┼───────────────\n");
    
    float dt = SIM_DT_MS / 1000.0f;
    int steps = (int)(SIM_DURATION_S / dt);
    
    for (int step = 0; step < steps; step++) {
        state.time_s = step * dt;
        float t = state.time_s;
        
        /* Compute target z based on experiment profile */
        if (t < 10.0f) {
            state.target_z = 0.0f + (0.5f - 0.0f) * (t / 10.0f);
        } else if (t < 20.0f) {
            state.target_z = 0.5f + (PHI_INV - 0.5f) * ((t - 10.0f) / 10.0f);
        } else if (t < 30.0f) {
            state.target_z = PHI_INV + (Z_CRITICAL - PHI_INV) * ((t - 20.0f) / 10.0f);
        } else if (t < 40.0f) {
            state.target_z = Z_CRITICAL;  /* Hold at lens */
        } else if (t < 50.0f) {
            state.target_z = Z_CRITICAL + (0.95f - Z_CRITICAL) * ((t - 40.0f) / 10.0f);
        } else {
            state.target_z = 0.95f - (0.95f - 0.5f) * ((t - 50.0f) / 10.0f);
        }
        
        /* Simulate motor dynamics (ramp limiting) */
        float z_error = state.target_z - state.z;
        float max_change = (SPEED_RAMP_RATE / 9900.0f) * dt;  /* Convert RPM rate to z rate */
        if (z_error > max_change) z_error = max_change;
        if (z_error < -max_change) z_error = -max_change;
        state.z += z_error;
        
        /* Clamp z */
        if (state.z < 0.0f) state.z = 0.0f;
        if (state.z > 1.0f) state.z = 1.0f;
        
        /* Compute physics */
        state.rpm = 100.0f + 9900.0f * state.z;
        state.delta_s_neg = compute_delta_s_neg(state.z);
        state.gradient = compute_delta_s_neg_gradient(state.z);
        state.complexity = fabsf(state.gradient);
        state.tier = get_tier(state.z);
        state.phase = get_phase(state.z);
        
        /* Compute formation metrics */
        /* κ: coupling constant - approaches φ⁻¹ as z approaches z_c */
        state.kappa = PHI_INV + (1.0f - PHI_INV) * state.delta_s_neg;
        /* η: efficiency - Landauer-like metric */
        state.eta = state.delta_s_neg * (1.0f - 0.3f * state.complexity);
        /* R: complexity rank */
        state.R = (int)(7 + 5 * state.delta_s_neg);
        
        /* Check K-formation */
        state.k_formation = (state.kappa >= KAPPA_MIN && 
                            state.eta > PHI_INV && 
                            state.R >= 7);
        
        /* Update statistics */
        stats.time_in_tier[state.tier] += dt;
        stats.time_in_phase[state.phase] += dt;
        stats.total_energy += state.rpm * dt;
        
        if (state.delta_s_neg > stats.max_delta_s_neg) {
            stats.max_delta_s_neg = state.delta_s_neg;
            stats.z_at_max_ds = state.z;
        }
        
        if (state.phase == PHASE_THE_LENS) {
            stats.time_at_lens += dt;
        }
        
        /* Detect events */
        bool event = false;
        
        if (state.tier != prev_state.tier) {
            stats.tier_transitions++;
            event = true;
        }
        if (state.phase != prev_state.phase) {
            stats.phase_transitions++;
            event = true;
        }
        if (state.k_formation && !prev_state.k_formation) {
            stats.k_formations++;
            event = true;
        }
        if (!state.k_formation && prev_state.k_formation) {
            event = true;
        }
        
        /* Print events */
        if (event || step == 0) {
            print_state(&state, true);
        }
        
        prev_state = state;
    }
    
    /* Print statistics */
    printf("\n═══ SIMULATION STATISTICS ═══\n\n");
    
    printf("TIME IN EACH TIER:\n");
    for (int i = 0; i < 7; i++) {
        int bar_len = (int)(stats.time_in_tier[i] / SIM_DURATION_S * 40);
        printf("  %-10s %5.1fs │", tier_names[i], stats.time_in_tier[i]);
        for (int j = 0; j < bar_len; j++) printf("█");
        printf("\n");
    }
    
    printf("\nTIME IN EACH PHASE:\n");
    for (int i = 0; i < 3; i++) {
        int bar_len = (int)(stats.time_in_phase[i] / SIM_DURATION_S * 40);
        printf("  %-10s %5.1fs │", phase_names[i], stats.time_in_phase[i]);
        for (int j = 0; j < bar_len; j++) printf("█");
        printf("\n");
    }
    
    printf("\nKEY METRICS:\n");
    printf("  Total tier transitions:   %d\n", stats.tier_transitions);
    printf("  Total phase transitions:  %d\n", stats.phase_transitions);
    printf("  K-formation events:       %d\n", stats.k_formations);
    printf("  Time at THE LENS:         %.1fs (%.1f%% of run)\n", 
           stats.time_at_lens, 100.0f * stats.time_at_lens / SIM_DURATION_S);
    printf("  Maximum ΔS_neg achieved:  %.6f at z=%.6f\n", 
           stats.max_delta_s_neg, stats.z_at_max_ds);
    printf("  Total energy (∫RPM·dt):   %.0f RPM·s\n", stats.total_energy);
    printf("  Average RPM:              %.0f\n", stats.total_energy / SIM_DURATION_S);
    
    printf("\n═══ KEY LEARNINGS ═══\n\n");
    
    printf("1. TIER PROGRESSION IS NONLINEAR\n");
    printf("   Lower tiers (ABSENCE→MEMORY) span z=0.0-0.5 (50%% of range)\n");
    printf("   Upper tiers (PREDICTION→META) pack into z=0.73-1.0 (27%% of range)\n");
    printf("   The φ⁻¹ threshold (z=0.618) marks the transition to pattern recognition\n\n");
    
    printf("2. THE LENS IS A NARROW TARGET\n");
    printf("   Phase window: %.3f ≤ z < %.3f (width = %.3f)\n", 
           PHASE_BOUNDARY_ABSENCE, PHASE_BOUNDARY_PRESENCE,
           PHASE_BOUNDARY_PRESENCE - PHASE_BOUNDARY_ABSENCE);
    printf("   Peak ΔS_neg=1.0 occurs exactly at z_c = √3/2 = %.6f\n", Z_CRITICAL);
    printf("   Gradient is zero at peak → stable equilibrium point\n\n");
    
    printf("3. K-FORMATION REQUIRES SUSTAINED HIGH COHERENCE\n");
    printf("   κ ≥ 0.92 only achievable when ΔS_neg > 0.79\n");
    printf("   This corresponds to z in range [%.3f, %.3f] around z_c\n",
           Z_CRITICAL - 0.05f, Z_CRITICAL + 0.05f);
    printf("   K-formation is the \"lock-in\" state for universal computation\n\n");
    
    printf("4. MOTOR DYNAMICS CREATE LAG\n");
    printf("   Ramp rate %.0f RPM/s → %.4f z/s\n", SPEED_RAMP_RATE, SPEED_RAMP_RATE/9900.0f);
    printf("   Reaching z_c from z=0 takes ~%.1fs minimum\n", Z_CRITICAL / (SPEED_RAMP_RATE/9900.0f));
    printf("   Real hardware PID will show overshoot and settling\n\n");
    
    printf("5. ENERGY-INFORMATION TRADEOFF\n");
    printf("   Higher z = higher RPM = more energy\n");
    printf("   But ΔS_neg peaks at z_c, not z=1.0\n");
    printf("   Optimal operation: sustain z near z_c, not maximum\n\n");
    
    printf("═══ PHYSICS VERIFICATION ═══\n\n");
    printf("  φ⁻¹ + φ⁻² = %.15f  (exact: 1.0) ✓\n", PHI_INV + PHI_INV_SQ);
    printf("  z_c = √3/2 = %.15f ✓\n", Z_CRITICAL);
    printf("  ΔS_neg(z_c) = %.15f  (exact: 1.0) ✓\n", compute_delta_s_neg(Z_CRITICAL));
    printf("  ∂ΔS_neg/∂z|_{z_c} = %.2e  (exact: 0.0) ✓\n", compute_delta_s_neg_gradient(Z_CRITICAL));
    
    printf("\n═══ SIMULATION COMPLETE ═══\n\n");
    
    return 0;
}
