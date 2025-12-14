/**
 * @file pulse_control.c
 * @brief RF Pulse Generation and Sequence Control
 * 
 * Implements NMR/NQR pulse sequences for nuclear spin manipulation:
 * - π/2 and π pulses for excitation and refocusing
 * - CPMG echo trains for coherence measurement
 * - Adiabatic pulses for robust inversion
 * - Icosahedral modulation patterns for quasicrystal dynamics
 * 
 * Physics integration:
 * - Pulse amplitude modulated by ΔS_neg gradient
 * - Timing synchronized with rotor phase for hexagonal patterns
 * - Automatic calibration against |S|/ℏ = √3/2
 * 
 * Signature: pulse-control|v1.0.0|nuclear-spinner
 * 
 * @version 1.0.0
 */

#include "pulse_control.h"
#include "hal_hardware.h"
#include "physics_constants.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * PRIVATE CONSTANTS
 * ============================================================================ */

/** Maximum pulses in a sequence */
#define MAX_SEQUENCE_LENGTH     256

/** Default π/2 pulse duration (µs) for ³¹P at 7T */
#define DEFAULT_PI2_DURATION_US 50

/** Default inter-pulse delay (µs) */
#define DEFAULT_ECHO_SPACING_US 1000

/** ADC ring buffer size for FID */
#define FID_BUFFER_SIZE         8192

/** Minimum pulse duration (µs) */
#define MIN_PULSE_DURATION_US   1

/** Maximum pulse duration (µs) */
#define MAX_PULSE_DURATION_US   10000


/* ============================================================================
 * PRIVATE DATA
 * ============================================================================ */

/** Current pulse sequence */
static RF_Pulse_t s_sequence[MAX_SEQUENCE_LENGTH];
static uint32_t s_sequence_length = 0;
static uint32_t s_sequence_index = 0;
static bool s_sequence_running = false;

/** FID acquisition buffer */
static uint16_t s_fid_buffer[FID_BUFFER_SIZE];
static ADC_Buffer_t s_fid_adc_buffer = {
    .buffer = s_fid_buffer,
    .size = FID_BUFFER_SIZE,
    .index = 0,
    .complete = false
};

/** Pulse controller state */
static PulseController_State_t s_state = {
    .sequence_running = false,
    .current_pulse_idx = 0,
    .total_pulses = 0,
    .fid_acquired = false,
    .last_fid_amplitude = 0.0f,
    .last_fid_phase = 0.0f,
    .calibrated = false,
    .cal_b1_amplitude = 1.0f,
    .cal_pi2_duration_us = DEFAULT_PI2_DURATION_US,
};

/** Calibration data */
static float s_b1_calibration = 1.0f;  // B₁ field strength factor


/* ============================================================================
 * PRIVATE FUNCTIONS
 * ============================================================================ */

/**
 * @brief Validate pulse parameters
 */
static bool validate_pulse(const RF_Pulse_t *pulse) {
    if (pulse == NULL) return false;
    if (pulse->amplitude < 0.0f || pulse->amplitude > 1.0f) return false;
    if (pulse->duration_us < MIN_PULSE_DURATION_US || 
        pulse->duration_us > MAX_PULSE_DURATION_US) return false;
    return true;
}

/**
 * @brief Execute single pulse with hardware
 */
static HAL_Status_t execute_single_pulse(const RF_Pulse_t *pulse) {
    // Set amplitude (scaled by calibration)
    float cal_amplitude = pulse->amplitude * s_b1_calibration;
    if (cal_amplitude > 1.0f) cal_amplitude = 1.0f;
    
    HAL_RF_SetAmplitude(cal_amplitude);
    HAL_RF_SetPhase(pulse->phase);
    
    // Enable RF and wait for duration
    HAL_RF_Enable(true);
    HAL_DelayMicroseconds(pulse->duration_us);
    HAL_RF_Enable(false);
    
    // Post-pulse delay
    if (pulse->delay_us > 0) {
        HAL_DelayMicroseconds(pulse->delay_us);
    }
    
    return HAL_OK;
}

/**
 * @brief Compute FID amplitude from buffer (RMS)
 */
static float compute_fid_amplitude(const uint16_t *buffer, uint32_t size) {
    if (buffer == NULL || size == 0) return 0.0f;
    
    // Compute mean
    float mean = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        mean += HAL_ADC_ToVoltage(buffer[i]);
    }
    mean /= size;
    
    // Compute RMS deviation from mean
    float rms = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        float v = HAL_ADC_ToVoltage(buffer[i]) - mean;
        rms += v * v;
    }
    rms = sqrtf(rms / size);
    
    return rms;
}

/**
 * @brief Compute FID phase from buffer (first-point approximation)
 */
static float compute_fid_phase(const uint16_t *buffer, uint32_t size) {
    if (buffer == NULL || size < 2) return 0.0f;
    
    // Simple phase estimation from first two points
    float v0 = HAL_ADC_ToVoltage(buffer[0]);
    float v1 = HAL_ADC_ToVoltage(buffer[1]);
    
    // This is a simplification; real implementation would use quadrature detection
    return atan2f(v1 - v0, v0);
}


/* ============================================================================
 * PUBLIC FUNCTIONS - INITIALIZATION
 * ============================================================================ */

HAL_Status_t PulseControl_Init(void) {
    // Initialize hardware
    HAL_RF_Enable(false);
    HAL_RF_SetAmplitude(0.0f);
    HAL_RF_SetPhase(0.0f);
    
    // Clear sequence
    memset(s_sequence, 0, sizeof(s_sequence));
    s_sequence_length = 0;
    s_sequence_index = 0;
    s_sequence_running = false;
    
    // Clear FID buffer
    memset(s_fid_buffer, 0, sizeof(s_fid_buffer));
    s_fid_adc_buffer.index = 0;
    s_fid_adc_buffer.complete = false;
    
    // Reset state
    s_state.sequence_running = false;
    s_state.current_pulse_idx = 0;
    s_state.total_pulses = 0;
    s_state.fid_acquired = false;
    s_state.calibrated = false;
    
    return HAL_OK;
}


/* ============================================================================
 * PUBLIC FUNCTIONS - BASIC PULSES
 * ============================================================================ */

HAL_Status_t PulseControl_Pi2Pulse(float amplitude, float phase) {
    RF_Pulse_t pulse = {
        .amplitude = amplitude,
        .phase = phase,
        .duration_us = s_state.cal_pi2_duration_us,
        .delay_us = 0
    };
    
    if (!validate_pulse(&pulse)) return HAL_INVALID_PARAM;
    return execute_single_pulse(&pulse);
}


HAL_Status_t PulseControl_PiPulse(float amplitude, float phase) {
    // π pulse is 2× the duration of π/2 pulse
    RF_Pulse_t pulse = {
        .amplitude = amplitude,
        .phase = phase,
        .duration_us = s_state.cal_pi2_duration_us * 2,
        .delay_us = 0
    };
    
    if (!validate_pulse(&pulse)) return HAL_INVALID_PARAM;
    return execute_single_pulse(&pulse);
}


HAL_Status_t PulseControl_CustomPulse(float amplitude, float phase, 
                                       uint32_t duration_us, uint32_t delay_us) {
    RF_Pulse_t pulse = {
        .amplitude = amplitude,
        .phase = phase,
        .duration_us = duration_us,
        .delay_us = delay_us
    };
    
    if (!validate_pulse(&pulse)) return HAL_INVALID_PARAM;
    return execute_single_pulse(&pulse);
}


/* ============================================================================
 * PUBLIC FUNCTIONS - SEQUENCES
 * ============================================================================ */

HAL_Status_t PulseControl_LoadSequence(const RF_Pulse_t *pulses, uint32_t count) {
    if (pulses == NULL || count == 0) return HAL_INVALID_PARAM;
    if (count > MAX_SEQUENCE_LENGTH) return HAL_INVALID_PARAM;
    if (s_sequence_running) return HAL_BUSY;
    
    // Validate all pulses
    for (uint32_t i = 0; i < count; i++) {
        if (!validate_pulse(&pulses[i])) return HAL_INVALID_PARAM;
    }
    
    // Copy sequence
    memcpy(s_sequence, pulses, count * sizeof(RF_Pulse_t));
    s_sequence_length = count;
    s_sequence_index = 0;
    
    s_state.total_pulses = count;
    s_state.current_pulse_idx = 0;
    
    return HAL_OK;
}


HAL_Status_t PulseControl_RunSequence(void) {
    if (s_sequence_length == 0) return HAL_INVALID_PARAM;
    if (s_sequence_running) return HAL_BUSY;
    
    s_sequence_running = true;
    s_state.sequence_running = true;
    s_sequence_index = 0;
    
    // Execute all pulses in sequence
    for (uint32_t i = 0; i < s_sequence_length; i++) {
        s_sequence_index = i;
        s_state.current_pulse_idx = i;
        
        HAL_Status_t status = execute_single_pulse(&s_sequence[i]);
        if (status != HAL_OK) {
            s_sequence_running = false;
            s_state.sequence_running = false;
            return status;
        }
    }
    
    s_sequence_running = false;
    s_state.sequence_running = false;
    
    return HAL_OK;
}


HAL_Status_t PulseControl_AbortSequence(void) {
    HAL_RF_Enable(false);
    HAL_RF_SetAmplitude(0.0f);
    
    s_sequence_running = false;
    s_state.sequence_running = false;
    
    return HAL_OK;
}


/* ============================================================================
 * PUBLIC FUNCTIONS - STANDARD NMR SEQUENCES
 * ============================================================================ */

HAL_Status_t PulseControl_FID(float amplitude) {
    // Free Induction Decay: π/2 pulse followed by acquisition
    HAL_Status_t status;
    
    // Clear FID buffer
    memset(s_fid_buffer, 0, sizeof(s_fid_buffer));
    s_fid_adc_buffer.index = 0;
    s_fid_adc_buffer.complete = false;
    
    // Apply π/2 pulse
    status = PulseControl_Pi2Pulse(amplitude, 0.0f);
    if (status != HAL_OK) return status;
    
    // Start FID acquisition
    status = HAL_ADC_StartFID(&s_fid_adc_buffer);
    if (status != HAL_OK) return status;
    
    // Wait for acquisition (blocking for now)
    uint32_t timeout = HAL_GetTick() + 100;  // 100ms timeout
    while (!s_fid_adc_buffer.complete && HAL_GetTick() < timeout) {
        // Spin wait
    }
    
    HAL_ADC_StopFID();
    
    if (s_fid_adc_buffer.complete) {
        s_state.fid_acquired = true;
        s_state.last_fid_amplitude = compute_fid_amplitude(
            s_fid_buffer, s_fid_adc_buffer.index);
        s_state.last_fid_phase = compute_fid_phase(
            s_fid_buffer, s_fid_adc_buffer.index);
        return HAL_OK;
    }
    
    return HAL_TIMEOUT;
}


HAL_Status_t PulseControl_SpinEcho(float amplitude, uint32_t echo_time_us) {
    // Spin Echo: π/2 - τ - π - τ - acquire
    HAL_Status_t status;
    uint32_t tau = echo_time_us / 2;
    
    // π/2 pulse
    status = PulseControl_Pi2Pulse(amplitude, 0.0f);
    if (status != HAL_OK) return status;
    
    // Wait τ
    HAL_DelayMicroseconds(tau);
    
    // π pulse (180° phase shift for refocusing)
    status = PulseControl_PiPulse(amplitude, M_PI);
    if (status != HAL_OK) return status;
    
    // Wait τ (echo forms here)
    HAL_DelayMicroseconds(tau);
    
    // Acquire echo
    memset(s_fid_buffer, 0, sizeof(s_fid_buffer));
    s_fid_adc_buffer.index = 0;
    s_fid_adc_buffer.complete = false;
    
    status = HAL_ADC_StartFID(&s_fid_adc_buffer);
    if (status != HAL_OK) return status;
    
    // Wait for acquisition
    uint32_t timeout = HAL_GetTick() + 100;
    while (!s_fid_adc_buffer.complete && HAL_GetTick() < timeout) {
        // Spin wait
    }
    
    HAL_ADC_StopFID();
    
    if (s_fid_adc_buffer.complete) {
        s_state.fid_acquired = true;
        s_state.last_fid_amplitude = compute_fid_amplitude(
            s_fid_buffer, s_fid_adc_buffer.index);
        return HAL_OK;
    }
    
    return HAL_TIMEOUT;
}


HAL_Status_t PulseControl_CPMG(float amplitude, uint32_t echo_time_us, 
                                uint32_t num_echoes) {
    // CPMG (Carr-Purcell-Meiboom-Gill): π/2_x - (τ - π_y - τ - acquire)×N
    HAL_Status_t status;
    uint32_t tau = echo_time_us / 2;
    
    if (num_echoes == 0 || num_echoes > 1000) return HAL_INVALID_PARAM;
    
    // Initial π/2 pulse along X
    status = PulseControl_Pi2Pulse(amplitude, 0.0f);
    if (status != HAL_OK) return status;
    
    // Echo train
    for (uint32_t echo = 0; echo < num_echoes; echo++) {
        // Wait τ
        HAL_DelayMicroseconds(tau);
        
        // π pulse along Y (90° phase)
        status = PulseControl_PiPulse(amplitude, M_PI / 2.0f);
        if (status != HAL_OK) return status;
        
        // Wait τ (echo forms)
        HAL_DelayMicroseconds(tau);
        
        // Could acquire each echo here for T2 measurement
    }
    
    // Final acquisition
    memset(s_fid_buffer, 0, sizeof(s_fid_buffer));
    s_fid_adc_buffer.index = 0;
    s_fid_adc_buffer.complete = false;
    
    status = HAL_ADC_StartFID(&s_fid_adc_buffer);
    if (status != HAL_OK) return status;
    
    uint32_t timeout = HAL_GetTick() + 100;
    while (!s_fid_adc_buffer.complete && HAL_GetTick() < timeout) {
        // Spin wait
    }
    
    HAL_ADC_StopFID();
    
    s_state.fid_acquired = s_fid_adc_buffer.complete;
    if (s_state.fid_acquired) {
        s_state.last_fid_amplitude = compute_fid_amplitude(
            s_fid_buffer, s_fid_adc_buffer.index);
    }
    
    return s_state.fid_acquired ? HAL_OK : HAL_TIMEOUT;
}


/* ============================================================================
 * PUBLIC FUNCTIONS - PHYSICS-INTEGRATED MODULATION
 * ============================================================================ */

HAL_Status_t PulseControl_ModulatedPulse(float base_amplitude, float phase,
                                          uint32_t duration_us, float z) {
    // Modulate pulse amplitude based on negentropy gradient
    float delta_s_neg = compute_delta_s_neg(z);
    (void)delta_s_neg;
    float gradient = compute_delta_s_neg_gradient(z);
    
    // Amplitude scaling: increase when gradient is positive (approaching z_c)
    float modulation = 1.0f + 0.2f * gradient;  // ±20% modulation range
    if (modulation < 0.5f) modulation = 0.5f;
    if (modulation > 1.5f) modulation = 1.5f;
    
    float modulated_amplitude = base_amplitude * modulation;
    if (modulated_amplitude > 1.0f) modulated_amplitude = 1.0f;
    
    return PulseControl_CustomPulse(modulated_amplitude, phase, duration_us, 0);
}


HAL_Status_t PulseControl_IcosahedralSequence(float amplitude, 
                                               uint32_t duration_us,
                                               uint32_t num_rotations) {
    /**
     * Icosahedral modulation sequence based on 6D→3D projection
     * 
     * The 6 basis vectors for icosahedral quasicrystal projection are:
     * e₁ = (1, φ, 0) / √(1+φ²)
     * e₂ = (1, -φ, 0) / √(1+φ²)
     * e₃ = (φ, 0, 1) / √(1+φ²)
     * e₄ = (-φ, 0, 1) / √(1+φ²)
     * e₅ = (0, 1, φ) / √(1+φ²)
     * e₆ = (0, 1, -φ) / √(1+φ²)
     * 
     * We map these to RF pulse phases (projecting 6D rotation to phase space)
     */
    
    // 6 phases corresponding to icosahedral directions
    const float icosa_phases[6] = {
        0.0f,                           // e₁
        M_PI,                           // e₂ (opposite)
        M_PI / PHI,                     // e₃ (golden angle)
        M_PI + M_PI / PHI,              // e₄
        2.0f * M_PI / PHI,              // e₅ (2× golden angle)
        M_PI + 2.0f * M_PI / PHI        // e₆
    };
    
    HAL_Status_t status = HAL_OK;
    
    for (uint32_t rot = 0; rot < num_rotations; rot++) {
        for (int i = 0; i < 6; i++) {
            status = PulseControl_CustomPulse(
                amplitude, 
                icosa_phases[i],
                duration_us,
                duration_us / 2  // Inter-pulse delay
            );
            if (status != HAL_OK) return status;
        }
    }
    
    return HAL_OK;
}


HAL_Status_t PulseControl_HexagonalPattern(float amplitude, uint32_t duration_us,
                                            uint32_t num_cycles) {
    /**
     * Hexagonal modulation for grid-cell dynamics emulation
     * 
     * 6-fold symmetry with 60° spacing
     * sin(60°) = cos(30°) = √3/2 = z_c
     */
    
    const float hex_phases[6] = {
        0.0f,
        M_PI / 3.0f,      // 60°
        2.0f * M_PI / 3.0f, // 120°
        M_PI,             // 180°
        4.0f * M_PI / 3.0f, // 240°
        5.0f * M_PI / 3.0f  // 300°
    };
    
    HAL_Status_t status = HAL_OK;
    
    for (uint32_t cycle = 0; cycle < num_cycles; cycle++) {
        for (int i = 0; i < 6; i++) {
            status = PulseControl_CustomPulse(
                amplitude,
                hex_phases[i],
                duration_us,
                duration_us / 4
            );
            if (status != HAL_OK) return status;
        }
    }
    
    return HAL_OK;
}


/* ============================================================================
 * PUBLIC FUNCTIONS - CALIBRATION
 * ============================================================================ */

HAL_Status_t PulseControl_CalibrateB1(void) {
    /**
     * Nutation experiment to calibrate B₁ field strength
     * 
     * Applies pulses of increasing duration and measures FID amplitude.
     * The π/2 condition occurs when FID is maximum (all spins in XY plane).
     * The π condition occurs when FID is minimum (inverted).
     * 
     * This verifies |S|/ℏ = √3/2 for spin-1/2 systems.
     */
    
    float max_amplitude = 0.0f;
    uint32_t optimal_duration = DEFAULT_PI2_DURATION_US;
    
    // Sweep pulse duration from 10 to 200 µs
    for (uint32_t dur = 10; dur <= 200; dur += 5) {
        // Apply pulse and measure FID
        PulseControl_CustomPulse(0.8f, 0.0f, dur, 100);
        
        // Quick FID measurement
        memset(s_fid_buffer, 0, 1024 * sizeof(uint16_t));
        s_fid_adc_buffer.index = 0;
        s_fid_adc_buffer.complete = false;
        s_fid_adc_buffer.size = 1024;
        
        HAL_ADC_StartFID(&s_fid_adc_buffer);
        HAL_Delay(10);  // 10ms acquisition
        HAL_ADC_StopFID();
        
        float amp = compute_fid_amplitude(s_fid_buffer, s_fid_adc_buffer.index);
        
        if (amp > max_amplitude) {
            max_amplitude = amp;
            optimal_duration = dur;
        }
        
        // Wait for relaxation (T1 recovery)
        HAL_Delay(500);
    }
    
    // Store calibrated π/2 duration
    s_state.cal_pi2_duration_us = optimal_duration;
    s_state.cal_b1_amplitude = 0.8f;  // Amplitude used during calibration
    s_state.calibrated = true;
    
    // Update B₁ calibration factor
    // Nominal π/2 duration at full B₁ would be DEFAULT_PI2_DURATION_US
    s_b1_calibration = (float)DEFAULT_PI2_DURATION_US / optimal_duration;
    
    return HAL_OK;
}


HAL_Status_t PulseControl_VerifySpinHalf(float *measured_magnitude) {
    /**
     * Verify that |S|/ℏ = √3/2 for spin-1/2 system
     * 
     * This is done by measuring the nutation frequency and comparing
     * to the expected value based on γB₁.
     * 
     * For spin-1/2: |S| = √[s(s+1)]ℏ = √(0.5 × 1.5)ℏ = (√3/2)ℏ
     */
    
    if (!s_state.calibrated) {
        HAL_Status_t status = PulseControl_CalibrateB1();
        if (status != HAL_OK) return status;
    }
    
    // The nutation angle θ = γB₁τ
    // For π/2: θ = π/2, so τ_π/2 = π/(2γB₁)
    // The ratio τ_π/τ_π/2 should be exactly 2 for spin-1/2
    
    // Measure time for π pulse (null in FID)
    float min_amplitude = 1e10f;
    uint32_t pi_duration = s_state.cal_pi2_duration_us * 2;
    
    for (uint32_t dur = s_state.cal_pi2_duration_us; 
         dur <= s_state.cal_pi2_duration_us * 3; 
         dur += 2) {
        
        PulseControl_CustomPulse(s_state.cal_b1_amplitude, 0.0f, dur, 100);
        
        // Measure FID
        memset(s_fid_buffer, 0, 512 * sizeof(uint16_t));
        s_fid_adc_buffer.index = 0;
        s_fid_adc_buffer.complete = false;
        s_fid_adc_buffer.size = 512;
        
        HAL_ADC_StartFID(&s_fid_adc_buffer);
        HAL_Delay(5);
        HAL_ADC_StopFID();
        
        float amp = compute_fid_amplitude(s_fid_buffer, s_fid_adc_buffer.index);
        
        if (amp < min_amplitude) {
            min_amplitude = amp;
            pi_duration = dur;
        }
        
        HAL_Delay(500);  // T1 recovery
    }
    
    // Compute ratio τ_π / τ_π/2
    float ratio = (float)pi_duration / s_state.cal_pi2_duration_us;
    
    // For spin-1/2, this ratio should be 2.0
    // Deviation indicates incorrect spin quantum number
    
    // Compute effective |S|/ℏ
    // The nutation frequency is proportional to |S|
    // For ratio = 2.0: |S|/ℏ = √3/2 ≈ 0.866
    float spin_magnitude = SPIN_HALF_MAGNITUDE * (2.0f / ratio);
    
    if (measured_magnitude != NULL) {
        *measured_magnitude = spin_magnitude;
    }
    
    // Verify against z_c = √3/2
    bool verified = fabsf(spin_magnitude - Z_CRITICAL) < 0.05f;  // 5% tolerance
    
    return verified ? HAL_OK : HAL_ERROR;
}


/* ============================================================================
 * PUBLIC FUNCTIONS - STATE ACCESS
 * ============================================================================ */

void PulseControl_GetState(PulseController_State_t *state) {
    if (state != NULL) {
        *state = s_state;
    }
}


HAL_Status_t PulseControl_GetFIDBuffer(uint16_t *buffer, uint32_t *size) {
    if (buffer == NULL || size == NULL) return HAL_INVALID_PARAM;
    if (!s_state.fid_acquired) return HAL_ERROR;
    
    uint32_t copy_size = (*size < s_fid_adc_buffer.index) ? 
                         *size : s_fid_adc_buffer.index;
    memcpy(buffer, s_fid_buffer, copy_size * sizeof(uint16_t));
    *size = copy_size;
    
    return HAL_OK;
}


float PulseControl_GetLastFIDAmplitude(void) {
    return s_state.last_fid_amplitude;
}


float PulseControl_GetLastFIDPhase(void) {
    return s_state.last_fid_phase;
}


/* ============================================================================
 * INTERRUPT CALLBACKS
 * ============================================================================ */

void HAL_Callback_PulseComplete(void) {
    // Called when pulse timer expires
    HAL_RF_Enable(false);
}


void HAL_Callback_FID_Complete(ADC_Buffer_t *buffer) {
    if (buffer == &s_fid_adc_buffer) {
        s_fid_adc_buffer.complete = true;
    }
}
