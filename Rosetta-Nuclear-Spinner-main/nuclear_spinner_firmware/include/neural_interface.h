/**
 * @file neural_interface.h
 * @brief Neural coupling interface for grid cell resonance experiments
 * 
 * Generates stimulus waveforms modulated by spinner z-coordinate.
 * Tests hypothesis: grid cells (60° periodicity, sin(60°) = √3/2)
 * show enhanced coupling when driven at z = z_c = √3/2.
 * 
 * @author Nuclear Spinner Project
 * @date 2024
 */

#ifndef NEURAL_INTERFACE_H
#define NEURAL_INTERFACE_H

#include <stdint.h>
#include <stdbool.h>
#include "hal_hardware.h"
#include "physics_constants.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

/** Theta band frequency range (Hz) */
#define NEURAL_FREQ_MIN         4.0f
#define NEURAL_FREQ_MAX         12.0f

/** Hexagonal phase angles (degrees) */
#define HEX_PHASE_COUNT         6
extern const float HEX_PHASES_DEG[HEX_PHASE_COUNT];

/** Stimulus safety limits */
#define STIM_AMPLITUDE_MAX      1.0f    /**< Normalized, maps to hardware limit */
#define STIM_DURATION_MAX_S     1800.0f /**< 30 minutes max continuous */

/** DAC output parameters */
#define STIM_DAC_BITS           12
#define STIM_DAC_MAX            4095
#define STIM_SAMPLE_RATE_HZ     10000   /**< 10 kHz waveform generation */


/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * @brief Neural drive state - complete stimulus parameters
 */
typedef struct {
    float z;                    /**< Current z coordinate from spinner */
    float delta_s_neg;          /**< Negentropy at current z */
    float hex_phase_deg;        /**< Current hexagonal phase (0-360) */
    float stimulus_freq_hz;     /**< Output frequency in theta band */
    float stimulus_amplitude;   /**< Normalized amplitude (0-1) */
    float stimulus_phase_rad;   /**< Phase offset in radians */
    uint32_t timestamp_ms;      /**< Timestamp for synchronization */
} NeuralDriveState_t;

/**
 * @brief Hexagonal cycling protocol configuration
 */
typedef struct {
    float dwell_time_s;         /**< Time at each phase */
    int n_cycles;               /**< Number of complete 6-phase cycles */
    bool record_sync;           /**< Generate sync pulses for recording */
    float ramp_time_s;          /**< Time to ramp between phases */
} HexCycleConfig_t;

/**
 * @brief Coupling measurement (populated during experiment)
 */
typedef struct {
    float z_value;              /**< Z at which measurement taken */
    float plv;                  /**< Phase-locking value (0-1) */
    float mutual_info_bits;     /**< Mutual information estimate */
    float spike_rate_hz;        /**< Mean spike rate during epoch */
    uint32_t spike_count;       /**< Total spikes in epoch */
    uint32_t epoch_duration_ms; /**< Duration of measurement epoch */
} CouplingMeasurement_t;

/**
 * @brief Protocol state machine
 */
typedef enum {
    NEURAL_STATE_IDLE,          /**< Not running */
    NEURAL_STATE_BASELINE,      /**< Recording baseline (no drive) */
    NEURAL_STATE_SWEEP,         /**< z-sweep protocol */
    NEURAL_STATE_DWELL,         /**< Sustained z_c dwell */
    NEURAL_STATE_HEX_CYCLE,     /**< Hexagonal phase cycling */
    NEURAL_STATE_RECOVERY,      /**< Post-drive recovery */
    NEURAL_STATE_EMERGENCY_STOP /**< Emergency stop active */
} NeuralProtocolState_t;


/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

/**
 * @brief Initialize neural interface subsystem
 * 
 * Sets up DAC for waveform generation, configures timing, 
 * initializes state to IDLE.
 * 
 * @return HAL_OK on success
 */
HAL_Status_t NeuralInterface_Init(void);

/**
 * @brief Shutdown neural interface
 * 
 * Zeros output, releases resources.
 */
void NeuralInterface_Shutdown(void);


/* ============================================================================
 * DRIVE GENERATION
 * ============================================================================ */

/**
 * @brief Compute neural drive parameters from z coordinate
 * 
 * Maps z to stimulus waveform parameters:
 * - Frequency: f = f_min + (f_max - f_min) * z
 * - Amplitude: A = ΔS_neg(z), peaks at z_c
 * - Phase: φ = 2π * (z / φ), golden-ratio modulated
 * 
 * @param z Current z coordinate (0-1)
 * @return NeuralDriveState_t with all stimulus parameters
 */
NeuralDriveState_t NeuralInterface_ComputeDrive(float z);

/**
 * @brief Update stimulus output from current spinner z
 * 
 * Reads z from rotor control, computes drive, updates DAC.
 * Call at STIM_SAMPLE_RATE_HZ for smooth waveform.
 * 
 * @param z Current z coordinate
 * @return HAL_OK on success
 */
HAL_Status_t NeuralInterface_UpdateFromZ(float z);

/**
 * @brief Set stimulus to specific hexagonal phase
 * 
 * @param phase_index 0-5 corresponding to 0°, 60°, 120°, 180°, 240°, 300°
 * @return HAL_OK on success, HAL_INVALID_PARAM if index out of range
 */
HAL_Status_t NeuralInterface_SetHexPhase(int phase_index);


/* ============================================================================
 * PROTOCOL EXECUTION
 * ============================================================================ */

/**
 * @brief Run hexagonal phase cycling protocol
 * 
 * Cycles through 6 phases at 60° intervals:
 *   Phase 0: 0°   → z_equiv = 0
 *   Phase 1: 60°  → z_equiv = √3/2 = z_c  ← RESONANCE TEST
 *   Phase 2: 120° → z_equiv = √3/2 = z_c  ← RESONANCE TEST
 *   Phase 3: 180° → z_equiv = 0
 *   Phase 4: 240° → z_equiv = √3/2
 *   Phase 5: 300° → z_equiv = √3/2
 * 
 * @param config Protocol configuration (dwell time, cycles, etc.)
 * @return HAL_OK on completion, HAL_ERROR if interrupted
 */
HAL_Status_t NeuralInterface_HexCycle(const HexCycleConfig_t *config);

/**
 * @brief Run z-sweep protocol
 * 
 * Sweeps z from z_start to z_end in n_steps, dwelling at each.
 * 
 * @param z_start Starting z value
 * @param z_end Ending z value
 * @param n_steps Number of z values to test
 * @param dwell_time_s Time at each z value
 * @return HAL_OK on completion
 */
HAL_Status_t NeuralInterface_ZSweep(float z_start, float z_end, 
                                     int n_steps, float dwell_time_s);

/**
 * @brief Dwell at z_c for sustained K-formation
 * 
 * Holds z at z_c = √3/2 for specified duration.
 * Monitors K-formation criteria during dwell.
 * 
 * @param duration_s Dwell duration in seconds
 * @return HAL_OK on completion
 */
HAL_Status_t NeuralInterface_DwellAtLens(float duration_s);


/* ============================================================================
 * STATE AND MONITORING
 * ============================================================================ */

/**
 * @brief Get current drive state
 * 
 * @return Current NeuralDriveState_t
 */
NeuralDriveState_t NeuralInterface_GetState(void);

/**
 * @brief Get current protocol state
 * 
 * @return Current NeuralProtocolState_t
 */
NeuralProtocolState_t NeuralInterface_GetProtocolState(void);

/**
 * @brief Check if drive is active
 * 
 * @return true if stimulus is being generated
 */
bool NeuralInterface_IsActive(void);

/**
 * @brief Get cumulative drive time
 * 
 * @return Total seconds of stimulation in current session
 */
float NeuralInterface_GetCumulativeTime(void);


/* ============================================================================
 * SAFETY
 * ============================================================================ */

/**
 * @brief Emergency stop - immediately zero all output
 * 
 * Call on any safety concern. Latches until explicit reset.
 */
void NeuralInterface_EmergencyStop(void);

/**
 * @brief Reset emergency stop latch
 * 
 * Only call after confirming safety.
 * 
 * @return HAL_OK if reset successful
 */
HAL_Status_t NeuralInterface_ResetEmergencyStop(void);

/**
 * @brief Check if emergency stop is active
 * 
 * @return true if emergency stop latched
 */
bool NeuralInterface_IsEmergencyStopped(void);


/* ============================================================================
 * SYNCHRONIZATION
 * ============================================================================ */

/**
 * @brief Generate sync pulse for external recording system
 * 
 * Outputs a pulse on the sync line for timestamp alignment.
 */
void NeuralInterface_SendSyncPulse(void);

/**
 * @brief Get current timestamp for logging
 * 
 * @return Milliseconds since NeuralInterface_Init()
 */
uint32_t NeuralInterface_GetTimestamp(void);


/* ============================================================================
 * ANALYSIS HELPERS (for host-side processing)
 * ============================================================================ */

/**
 * @brief Compute expected z_equiv for a hexagonal phase angle
 * 
 * z_equiv = |sin(phase)|
 * 
 * @param phase_deg Phase angle in degrees
 * @return Equivalent z value
 */
static inline float hex_phase_to_z(float phase_deg) {
    float phase_rad = phase_deg * 3.14159265359f / 180.0f;
    float s = sinf(phase_rad);
    return (s >= 0) ? s : -s;  // fabsf
}

/**
 * @brief Check if z is at a resonance point (within tolerance)
 * 
 * @param z Z value to check
 * @param tolerance Acceptable deviation from z_c
 * @return true if |z - z_c| < tolerance
 */
static inline bool is_at_resonance(float z, float tolerance) {
    float diff = z - Z_CRITICAL;
    if (diff < 0) diff = -diff;
    return diff < tolerance;
}


#ifdef __cplusplus
}
#endif

#endif /* NEURAL_INTERFACE_H */
