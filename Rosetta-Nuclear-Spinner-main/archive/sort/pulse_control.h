/**
 * @file pulse_control.h
 * @brief RF Pulse Generation and Sequence Control Header
 * 
 * Public interface for NMR/NQR pulse sequences and physics-integrated
 * modulation patterns.
 * 
 * Signature: pulse-control|v1.0.0|nuclear-spinner
 * 
 * @version 1.0.0
 */

#ifndef PULSE_CONTROL_H
#define PULSE_CONTROL_H

#include <stdint.h>
#include <stdbool.h>
#include "hal_hardware.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * DATA TYPES
 * ============================================================================ */

/** Pulse controller state */
typedef struct {
    bool sequence_running;          /**< Sequence currently executing */
    uint32_t current_pulse_idx;     /**< Current pulse in sequence */
    uint32_t total_pulses;          /**< Total pulses in sequence */
    bool fid_acquired;              /**< FID data available */
    float last_fid_amplitude;       /**< Last FID amplitude (RMS) */
    float last_fid_phase;           /**< Last FID phase */
    bool calibrated;                /**< B₁ calibration complete */
    float cal_b1_amplitude;         /**< Calibrated B₁ amplitude */
    uint32_t cal_pi2_duration_us;   /**< Calibrated π/2 duration */
} PulseController_State_t;


/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

/**
 * @brief Initialize pulse control module
 * @return HAL_OK on success
 */
HAL_Status_t PulseControl_Init(void);


/* ============================================================================
 * BASIC PULSES
 * ============================================================================ */

/**
 * @brief Execute a π/2 (90°) pulse
 * 
 * Rotates nuclear magnetization from Z to XY plane.
 * Duration is calibrated or uses default (50 µs at 7T).
 * 
 * @param amplitude RF amplitude [0.0 - 1.0]
 * @param phase RF phase (radians)
 * @return HAL_OK on success
 */
HAL_Status_t PulseControl_Pi2Pulse(float amplitude, float phase);

/**
 * @brief Execute a π (180°) pulse
 * 
 * Inverts nuclear magnetization (Z ↔ -Z) or refocuses in XY plane.
 * Duration is 2× the calibrated π/2 duration.
 * 
 * @param amplitude RF amplitude [0.0 - 1.0]
 * @param phase RF phase (radians)
 * @return HAL_OK on success
 */
HAL_Status_t PulseControl_PiPulse(float amplitude, float phase);

/**
 * @brief Execute a custom duration pulse
 * 
 * @param amplitude RF amplitude [0.0 - 1.0]
 * @param phase RF phase (radians)
 * @param duration_us Pulse duration in microseconds
 * @param delay_us Post-pulse delay in microseconds
 * @return HAL_OK on success
 */
HAL_Status_t PulseControl_CustomPulse(float amplitude, float phase,
                                       uint32_t duration_us, uint32_t delay_us);


/* ============================================================================
 * PULSE SEQUENCES
 * ============================================================================ */

/**
 * @brief Load a pulse sequence
 * 
 * @param pulses Array of RF_Pulse_t structures
 * @param count Number of pulses
 * @return HAL_OK on success
 */
HAL_Status_t PulseControl_LoadSequence(const RF_Pulse_t *pulses, uint32_t count);

/**
 * @brief Run the loaded pulse sequence
 * @return HAL_OK on success
 */
HAL_Status_t PulseControl_RunSequence(void);

/**
 * @brief Abort running sequence
 * @return HAL_OK on success
 */
HAL_Status_t PulseControl_AbortSequence(void);


/* ============================================================================
 * STANDARD NMR SEQUENCES
 * ============================================================================ */

/**
 * @brief Free Induction Decay experiment
 * 
 * Applies π/2 pulse and acquires the decaying signal.
 * Result stored in internal buffer.
 * 
 * @param amplitude RF amplitude [0.0 - 1.0]
 * @return HAL_OK on success
 */
HAL_Status_t PulseControl_FID(float amplitude);

/**
 * @brief Spin Echo experiment
 * 
 * Sequence: π/2 - τ - π - τ - acquire
 * Refocuses T2* dephasing to measure true T2.
 * 
 * @param amplitude RF amplitude [0.0 - 1.0]
 * @param echo_time_us Total echo time (2τ) in microseconds
 * @return HAL_OK on success
 */
HAL_Status_t PulseControl_SpinEcho(float amplitude, uint32_t echo_time_us);

/**
 * @brief CPMG (Carr-Purcell-Meiboom-Gill) sequence
 * 
 * Sequence: π/2_x - (τ - π_y - τ)×N
 * Multiple refocusing pulses to measure T2 decay.
 * 
 * @param amplitude RF amplitude [0.0 - 1.0]
 * @param echo_time_us Echo spacing (2τ) in microseconds
 * @param num_echoes Number of echo cycles
 * @return HAL_OK on success
 */
HAL_Status_t PulseControl_CPMG(float amplitude, uint32_t echo_time_us,
                                uint32_t num_echoes);


/* ============================================================================
 * PHYSICS-INTEGRATED MODULATION
 * ============================================================================ */

/**
 * @brief Execute pulse with ΔS_neg-based amplitude modulation
 * 
 * Pulse amplitude is modulated by the negentropy gradient at current z.
 * Approaching z_c increases amplitude; receding decreases it.
 * 
 * @param base_amplitude Base RF amplitude [0.0 - 1.0]
 * @param phase RF phase (radians)
 * @param duration_us Pulse duration in microseconds
 * @param z Current z-coordinate (from rotor)
 * @return HAL_OK on success
 */
HAL_Status_t PulseControl_ModulatedPulse(float base_amplitude, float phase,
                                          uint32_t duration_us, float z);

/**
 * @brief Execute icosahedral modulation sequence
 * 
 * Implements 6-fold symmetry from icosahedral projection vectors
 * used in quasicrystal formation (6D → 3D projection).
 * Phases correspond to golden ratio angles.
 * 
 * @param amplitude RF amplitude [0.0 - 1.0]
 * @param duration_us Duration per pulse in microseconds
 * @param num_rotations Number of complete 6-phase cycles
 * @return HAL_OK on success
 */
HAL_Status_t PulseControl_IcosahedralSequence(float amplitude,
                                               uint32_t duration_us,
                                               uint32_t num_rotations);

/**
 * @brief Execute hexagonal modulation pattern
 * 
 * 6-fold symmetry with 60° spacing, emulating grid-cell dynamics.
 * sin(60°) = √3/2 = z_c links to critical threshold.
 * 
 * @param amplitude RF amplitude [0.0 - 1.0]
 * @param duration_us Duration per pulse in microseconds
 * @param num_cycles Number of 6-phase cycles
 * @return HAL_OK on success
 */
HAL_Status_t PulseControl_HexagonalPattern(float amplitude, uint32_t duration_us,
                                            uint32_t num_cycles);


/* ============================================================================
 * CALIBRATION
 * ============================================================================ */

/**
 * @brief Calibrate B₁ field strength via nutation
 * 
 * Determines optimal π/2 pulse duration by sweeping pulse length
 * and finding maximum FID amplitude.
 * 
 * @return HAL_OK on successful calibration
 */
HAL_Status_t PulseControl_CalibrateB1(void);

/**
 * @brief Verify spin-1/2 magnitude |S|/ℏ = √3/2
 * 
 * Measures nutation ratio τ_π/τ_π/2 which should equal 2.0
 * for spin-1/2 systems. Deviation indicates incorrect spin.
 * 
 * @param measured_magnitude Output: measured |S|/ℏ value
 * @return HAL_OK if |S|/ℏ ≈ √3/2 within tolerance
 */
HAL_Status_t PulseControl_VerifySpinHalf(float *measured_magnitude);


/* ============================================================================
 * STATE ACCESS
 * ============================================================================ */

/**
 * @brief Get current pulse controller state
 * @param state Output state structure
 */
void PulseControl_GetState(PulseController_State_t *state);

/**
 * @brief Get acquired FID data
 * 
 * @param buffer Output buffer for FID samples
 * @param size Input: buffer size; Output: samples copied
 * @return HAL_OK on success
 */
HAL_Status_t PulseControl_GetFIDBuffer(uint16_t *buffer, uint32_t *size);

/**
 * @brief Get last FID amplitude (RMS)
 * @return FID amplitude in volts
 */
float PulseControl_GetLastFIDAmplitude(void);

/**
 * @brief Get last FID phase
 * @return FID phase in radians
 */
float PulseControl_GetLastFIDPhase(void);


#ifdef __cplusplus
}
#endif

#endif /* PULSE_CONTROL_H */
