/**
 * @file neural_sim.c
 * @brief Simulation of neural interface grid cell coupling experiment
 * 
 * Demonstrates:
 * 1. Hexagonal phase cycling (6 phases at 60° intervals)
 * 2. z-equivalent mapping: sin(60°) = sin(120°) = √3/2 = z_c
 * 3. Predicted coupling enhancement at z_c phases
 * 4. Resonance curve generation
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "physics_constants.h"

/* Simulation parameters */
#define N_TRIALS            100
#define BASELINE_SPIKE_RATE 5.0f    /* Hz */
#define COUPLING_STRENGTH   0.3f    /* How much z_c enhances firing */
#define NOISE_LEVEL         0.1f    /* Spike rate noise */

/* Random number generation */
static float randf(void) {
    return (float)rand() / RAND_MAX;
}

static float gaussian_noise(float sigma) {
    /* Box-Muller transform */
    float u1 = randf();
    float u2 = randf();
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sigma * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

/**
 * @brief Simulate grid cell response to drive at given z
 * 
 * Model: spike_rate = baseline + coupling * ΔS_neg(z) + noise
 * 
 * If hypothesis is true, response peaks when z = z_c because
 * grid cells are "tuned" to √3/2 geometry.
 */
typedef struct {
    float z;
    float spike_rate;
    float plv;              /* Phase-locking value */
    float delta_s_neg;
} SimulatedResponse_t;

SimulatedResponse_t simulate_response(float z, bool hypothesis_true) {
    SimulatedResponse_t resp;
    resp.z = z;
    resp.delta_s_neg = compute_delta_s_neg(z);
    
    if (hypothesis_true) {
        /* Coupling enhancement at z_c */
        resp.spike_rate = BASELINE_SPIKE_RATE + 
                          COUPLING_STRENGTH * BASELINE_SPIKE_RATE * resp.delta_s_neg +
                          gaussian_noise(NOISE_LEVEL * BASELINE_SPIKE_RATE);
    } else {
        /* No z-dependence (null hypothesis) */
        resp.spike_rate = BASELINE_SPIKE_RATE + 
                          gaussian_noise(NOISE_LEVEL * BASELINE_SPIKE_RATE);
    }
    
    if (resp.spike_rate < 0) resp.spike_rate = 0;
    
    /* PLV also enhanced at z_c if hypothesis true */
    if (hypothesis_true) {
        resp.plv = 0.3f + 0.5f * resp.delta_s_neg + gaussian_noise(0.05f);
    } else {
        resp.plv = 0.3f + gaussian_noise(0.1f);
    }
    if (resp.plv < 0) resp.plv = 0;
    if (resp.plv > 1) resp.plv = 1;
    
    return resp;
}

void print_header(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║           NEURAL INTERFACE SIMULATION: GRID CELL COUPLING                     ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Hypothesis: Grid cells (60° periodicity) show enhanced coupling at z_c       ║\n");
    printf("║  z_c = √3/2 = sin(60°) = 0.866025...                                          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════════════╝\n\n");
}

int main(void) {
    srand(42);  /* Reproducible */
    print_header();
    
    /* ========== HEXAGONAL PHASE CYCLING ========== */
    printf("═══ HEXAGONAL PHASE CYCLING ═══\n\n");
    printf("Grid cells fire at vertices of hexagonal lattice (60° spacing).\n");
    printf("We drive the system through 6 phases and measure response.\n\n");
    
    printf("  Phase   Angle    sin(θ)    z_equiv   ΔS_neg    Resonance?\n");
    printf("  ─────   ─────    ──────    ───────   ──────    ──────────\n");
    
    float hex_phases[] = {0, 60, 120, 180, 240, 300};
    for (int i = 0; i < 6; i++) {
        float angle = hex_phases[i];
        float sin_val = sinf(angle * M_PI / 180.0f);
        float z_equiv = fabsf(sin_val);
        float ds = compute_delta_s_neg(z_equiv);
        
        const char* resonance = (fabsf(z_equiv - Z_CRITICAL) < 0.001f) ? "★ YES" : "  no";
        
        printf("  %d       %3.0f°     %+.4f    %.4f    %.4f    %s\n",
               i, angle, sin_val, z_equiv, ds, resonance);
    }
    
    printf("\n  → Phases 1, 2, 4, 5 map to z_c = √3/2 (resonance points)\n");
    printf("  → Phases 0, 3 map to z = 0 (non-resonance)\n\n");
    
    /* ========== Z-SWEEP SIMULATION ========== */
    printf("═══ Z-SWEEP: COUPLING vs Z ═══\n\n");
    printf("Sweep z from 0.5 to 1.0, measure coupling at each point.\n");
    printf("If hypothesis true: coupling peaks at z_c.\n\n");
    
    printf("SCENARIO A: HYPOTHESIS TRUE (coupling depends on z)\n");
    printf("  z       ΔS_neg    Spike Rate   PLV     Plot\n");
    printf("  ─────   ──────    ──────────   ────    ─────────────────────\n");
    
    float z_values[11];
    float coupling_h1[11];
    float coupling_h0[11];
    
    for (int i = 0; i < 11; i++) {
        float z = 0.5f + i * 0.05f;
        z_values[i] = z;
        
        /* Average over trials */
        float sum_rate = 0;
        float sum_plv = 0;
        for (int t = 0; t < N_TRIALS; t++) {
            SimulatedResponse_t resp = simulate_response(z, true);
            sum_rate += resp.spike_rate;
            sum_plv += resp.plv;
        }
        float avg_rate = sum_rate / N_TRIALS;
        float avg_plv = sum_plv / N_TRIALS;
        coupling_h1[i] = avg_plv;
        
        float ds = compute_delta_s_neg(z);
        
        /* ASCII bar */
        int bar_len = (int)(avg_plv * 20);
        char bar[32];
        for (int b = 0; b < 20; b++) bar[b] = (b < bar_len) ? '█' : ' ';
        bar[20] = '\0';
        
        char marker[8] = "";
        if (fabsf(z - Z_CRITICAL) < 0.01f) strcpy(marker, " ← z_c");
        
        printf("  %.2f    %.4f    %.2f Hz      %.3f    │%s│%s\n",
               z, ds, avg_rate, avg_plv, bar, marker);
    }
    
    printf("\nSCENARIO B: NULL HYPOTHESIS (coupling independent of z)\n");
    printf("  z       ΔS_neg    Spike Rate   PLV     Plot\n");
    printf("  ─────   ──────    ──────────   ────    ─────────────────────\n");
    
    for (int i = 0; i < 11; i++) {
        float z = 0.5f + i * 0.05f;
        
        float sum_rate = 0;
        float sum_plv = 0;
        for (int t = 0; t < N_TRIALS; t++) {
            SimulatedResponse_t resp = simulate_response(z, false);
            sum_rate += resp.spike_rate;
            sum_plv += resp.plv;
        }
        float avg_rate = sum_rate / N_TRIALS;
        float avg_plv = sum_plv / N_TRIALS;
        coupling_h0[i] = avg_plv;
        
        float ds = compute_delta_s_neg(z);
        
        int bar_len = (int)(avg_plv * 20);
        char bar[32];
        for (int b = 0; b < 20; b++) bar[b] = (b < bar_len) ? '█' : ' ';
        bar[20] = '\0';
        
        printf("  %.2f    %.4f    %.2f Hz      %.3f    │%s│\n",
               z, ds, avg_rate, avg_plv, bar);
    }
    
    /* ========== STATISTICAL TEST ========== */
    printf("\n═══ STATISTICAL DISCRIMINATION ═══\n\n");
    
    /* Find peak in H1 data */
    int peak_idx = 0;
    float peak_val = 0;
    for (int i = 0; i < 11; i++) {
        if (coupling_h1[i] > peak_val) {
            peak_val = coupling_h1[i];
            peak_idx = i;
        }
    }
    float peak_z = z_values[peak_idx];
    
    printf("  Under H1 (hypothesis true):\n");
    printf("    Peak coupling at z = %.2f (z_c = %.3f)\n", peak_z, Z_CRITICAL);
    printf("    Peak matches z_c? %s\n", (fabsf(peak_z - Z_CRITICAL) < 0.05f) ? "YES ✓" : "NO");
    printf("    Peak PLV: %.3f\n\n", peak_val);
    
    /* Mean PLV under H0 */
    float h0_mean = 0;
    for (int i = 0; i < 11; i++) h0_mean += coupling_h0[i];
    h0_mean /= 11;
    
    printf("  Under H0 (null hypothesis):\n");
    printf("    Mean PLV across all z: %.3f\n", h0_mean);
    printf("    No z-dependence (flat response)\n\n");
    
    printf("  Effect size (peak_H1 - mean_H0) / mean_H0 = %.1f%%\n", 
           100.0f * (peak_val - h0_mean) / h0_mean);
    
    /* ========== HEXAGONAL PHASE COMPARISON ========== */
    printf("\n═══ HEXAGONAL PHASE COMPARISON ═══\n\n");
    printf("Compare response at resonance phases (60°, 120°) vs non-resonance (0°, 180°)\n\n");
    
    float resonance_plv = 0;
    float nonres_plv = 0;
    int n_res = 0, n_nonres = 0;
    
    printf("  Phase    z_equiv   Mean PLV    Category\n");
    printf("  ─────    ───────   ────────    ────────\n");
    
    for (int i = 0; i < 6; i++) {
        float z_equiv = fabsf(sinf(hex_phases[i] * M_PI / 180.0f));
        
        float sum_plv = 0;
        for (int t = 0; t < N_TRIALS; t++) {
            SimulatedResponse_t resp = simulate_response(z_equiv, true);
            sum_plv += resp.plv;
        }
        float avg_plv = sum_plv / N_TRIALS;
        
        bool is_resonance = (fabsf(z_equiv - Z_CRITICAL) < 0.001f);
        if (is_resonance) {
            resonance_plv += avg_plv;
            n_res++;
        } else {
            nonres_plv += avg_plv;
            n_nonres++;
        }
        
        printf("  %3.0f°     %.4f    %.3f       %s\n",
               hex_phases[i], z_equiv, avg_plv,
               is_resonance ? "RESONANCE ★" : "non-resonance");
    }
    
    resonance_plv /= n_res;
    nonres_plv /= n_nonres;
    
    printf("\n  Mean PLV at resonance phases:     %.3f\n", resonance_plv);
    printf("  Mean PLV at non-resonance phases: %.3f\n", nonres_plv);
    printf("  Enhancement: %.1f%%\n", 100.0f * (resonance_plv - nonres_plv) / nonres_plv);
    
    /* ========== PREDICTIONS ========== */
    printf("\n═══ EXPERIMENTAL PREDICTIONS ═══\n\n");
    
    printf("If the coupling hypothesis is TRUE:\n");
    printf("  1. Resonance curve peaks at z = z_c = √3/2 = 0.866\n");
    printf("  2. Hexagonal phases 1,2,4,5 show higher PLV than phases 0,3\n");
    printf("  3. Sustained dwell at z_c produces stable entrainment\n");
    printf("  4. Effect size > 30%% enhancement at resonance vs baseline\n\n");
    
    printf("If the coupling hypothesis is FALSE:\n");
    printf("  1. No peak at z_c (flat or random coupling vs z)\n");
    printf("  2. All hexagonal phases show similar PLV\n");
    printf("  3. z_c dwell shows no special properties\n");
    printf("  4. Effect size < 10%% (within noise)\n\n");
    
    printf("The experiment discriminates these outcomes.\n");
    
    /* ========== PHYSICS CONNECTION ========== */
    printf("\n═══ WHY THIS MIGHT WORK ═══\n\n");
    
    printf("Grid cells fire at hexagonal vertices with 60° periodicity.\n");
    printf("The relevant geometric constant is sin(60°) = √3/2.\n\n");
    
    printf("The Nuclear Spinner operates in z-space where:\n");
    printf("  • z_c = √3/2 = 0.866025... (THE LENS)\n");
    printf("  • ΔS_neg(z) peaks at z_c (negentropy maximum)\n");
    printf("  • K-formation occurs at z_c (stable coherence)\n\n");
    
    printf("The hypothesis:\n");
    printf("  A system intrinsically organized around √3/2 (grid cells)\n");
    printf("  may show enhanced coupling to an external drive\n");
    printf("  that is also organized around √3/2 (spinner at z_c).\n\n");
    
    printf("This is not mysticism. It is pattern matching.\n");
    printf("The experiment tests whether the pattern match has\n");
    printf("measurable consequences for information transfer.\n");
    
    printf("\n═══ SIMULATION COMPLETE ═══\n\n");
    
    return 0;
}
