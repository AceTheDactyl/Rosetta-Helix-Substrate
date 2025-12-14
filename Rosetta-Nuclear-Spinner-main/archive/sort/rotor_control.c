/**
 * @file rotor_control.c
 * @brief Precision Rotor Control with Physics z-Mapping
 * 
 * Controls the mechanical rotor that modulates nuclear spin orientation:
 * - PID closed-loop speed control using encoder feedback
 * - z-coordinate ↔ RPM mapping (z ∈ [0,1] → RPM ∈ [100, 10000])
 * - Negentropy-based speed modulation
 * - Icosahedral/hexagonal phase locking
 * 
 * The rotor physically implements:
 * - z < z_c: pre-critical regimes (slow speeds, low complexity)
 * - z ≈ z_c: critical threshold (√3/2 ≈ 0.866)
 * - z > z_c: post-critical (high-speed, saturated)
 * 
 * Signature: rotor-control|v1.0.0|nuclear-spinner
 * 
 * @version 1.0.0
 */

#include "rotor_control.h"
#include "hal_hardware.h"
#include "physics_constants.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * PRIVATE CONSTANTS
 * ============================================================================ */

/** PID update rate (Hz) */
#define PID_UPDATE_RATE_HZ      1000

/** PID gains - tuned for Maxon ECX motor with magnetic bearings */
#define PID_KP                  0.8f
#define PID_KI                  0.15f
#define PID_KD                  0.05f

/** Anti-windup limit for integral term */
#define PID_INTEGRAL_MAX        100.0f

/** Derivative low-pass filter coefficient (0-1, higher = more filtering) */
#define PID_DERIV_FILTER        0.8f

/** Speed ramp rate (RPM/s) for smooth transitions */
#define SPEED_RAMP_RATE         500.0f

/** Minimum duty cycle to overcome static friction */
#define DUTY_MIN                0.05f

/** Maximum duty cycle */
#define DUTY_MAX                0.95f

/** Encoder timeout for stall detection (ms) */
#define ENCODER_TIMEOUT_MS      100

/** RPM measurement averaging window */
#define RPM_AVG_WINDOW          8

/** Index pulse angular position (counts from index to known angle) */
#define INDEX_OFFSET_COUNTS     0


/* ============================================================================
 * PRIVATE DATA
 * ============================================================================ */

/** Rotor controller state */
static RotorController_State_t s_state = {
    .mode = ROTOR_MODE_DISABLED,
    .target_z = 0.0f,
    .target_rpm = ROTOR_RPM_MIN,
    .actual_rpm = 0.0f,
    .actual_z = 0.0f,
    .duty_cycle = 0.0f,
    .encoder_count = 0,
    .revolutions = 0,
    .angular_position = 0.0f,
    .at_target = false,
    .stalled = false,
    .index_detected = false,
};

/** PID controller state */
static struct {
    float integral;
    float prev_error;
    float prev_derivative;
    uint32_t last_update_tick;
} s_pid = {0};

/** RPM measurement */
static struct {
    uint32_t last_count;
    uint32_t last_tick;
    float rpm_samples[RPM_AVG_WINDOW];
    uint32_t sample_idx;
    float rpm_filtered;
} s_rpm_meas = {0};

/** Phase lock for hexagonal patterns */
static struct {
    bool enabled;
    float target_phase;     // Target phase (radians)
    float phase_tolerance;  // Acceptable error (radians)
    uint32_t lock_count;    // Consecutive locked samples
} s_phase_lock = {0};


/* ============================================================================
 * PRIVATE FUNCTIONS
 * ============================================================================ */

/**
 * @brief Compute PID output
 */
static float compute_pid(float setpoint, float measurement, float dt) {
    float error = setpoint - measurement;
    
    // Proportional
    float p_term = PID_KP * error;
    
    // Integral with anti-windup
    s_pid.integral += error * dt;
    if (s_pid.integral > PID_INTEGRAL_MAX) s_pid.integral = PID_INTEGRAL_MAX;
    if (s_pid.integral < -PID_INTEGRAL_MAX) s_pid.integral = -PID_INTEGRAL_MAX;
    float i_term = PID_KI * s_pid.integral;
    
    // Derivative with low-pass filter
    float derivative = (error - s_pid.prev_error) / dt;
    derivative = PID_DERIV_FILTER * s_pid.prev_derivative + 
                 (1.0f - PID_DERIV_FILTER) * derivative;
    s_pid.prev_derivative = derivative;
    float d_term = PID_KD * derivative;
    
    s_pid.prev_error = error;
    
    return p_term + i_term + d_term;
}


/**
 * @brief Measure RPM from encoder
 */
static float measure_rpm(void) {
    uint32_t current_count = HAL_Motor_GetEncoderCount();
    uint32_t current_tick = HAL_GetTick();
    
    uint32_t dt_ms = current_tick - s_rpm_meas.last_tick;
    if (dt_ms == 0) return s_rpm_meas.rpm_filtered;
    
    // Handle encoder overflow
    int32_t d_count = (int32_t)(current_count - s_rpm_meas.last_count);
    
    // Convert to RPM: (counts / CPR) * (60000 ms/min) / dt_ms
    float rpm = (float)d_count / ENCODER_CPR * 60000.0f / dt_ms;
    if (rpm < 0) rpm = -rpm;  // Handle reverse direction
    
    // Update last values
    s_rpm_meas.last_count = current_count;
    s_rpm_meas.last_tick = current_tick;
    
    // Moving average filter
    s_rpm_meas.rpm_samples[s_rpm_meas.sample_idx] = rpm;
    s_rpm_meas.sample_idx = (s_rpm_meas.sample_idx + 1) % RPM_AVG_WINDOW;
    
    float sum = 0.0f;
    for (int i = 0; i < RPM_AVG_WINDOW; i++) {
        sum += s_rpm_meas.rpm_samples[i];
    }
    s_rpm_meas.rpm_filtered = sum / RPM_AVG_WINDOW;
    
    return s_rpm_meas.rpm_filtered;
}


/**
 * @brief Compute angular position from encoder
 */
static float compute_angular_position(void) {
    uint32_t count = HAL_Motor_GetEncoderCount();
    uint32_t position_in_rev = count % ENCODER_CPR;
    return (float)position_in_rev / ENCODER_CPR * 2.0f * M_PI;
}


/**
 * @brief Check for stall condition
 */
static bool check_stall(void) {
    static uint32_t last_moving_tick = 0;
    static float last_rpm = 0.0f;
    
    float current_rpm = s_state.actual_rpm;
    
    // If we're commanding motion but not moving
    if (s_state.duty_cycle > DUTY_MIN && current_rpm < 10.0f) {
        if (last_rpm < 10.0f) {
            if (HAL_GetTick() - last_moving_tick > ENCODER_TIMEOUT_MS) {
                return true;  // Stalled
            }
        }
    } else {
        last_moving_tick = HAL_GetTick();
    }
    
    last_rpm = current_rpm;
    return false;
}


/**
 * @brief Apply duty cycle with soft limits
 */
static void apply_duty(float duty) {
    // Clamp duty cycle
    if (duty < 0.0f) duty = 0.0f;
    if (duty > DUTY_MAX) duty = DUTY_MAX;
    
    // Dead-band for very low duty (motor won't move anyway)
    if (duty < DUTY_MIN && duty > 0.0f) {
        if (s_state.target_rpm > ROTOR_RPM_MIN) {
            duty = DUTY_MIN;  // Minimum to overcome stiction
        } else {
            duty = 0.0f;  // Stop completely
        }
    }
    
    s_state.duty_cycle = duty;
    HAL_Motor_SetDuty(duty);
}


/* ============================================================================
 * PUBLIC FUNCTIONS - INITIALIZATION
 * ============================================================================ */

HAL_Status_t RotorControl_Init(void) {
    // Initialize hardware
    HAL_Motor_Enable(false);
    HAL_Motor_SetDuty(0.0f);
    HAL_Motor_SetDirection(true);  // Clockwise default
    HAL_Motor_ResetEncoder();
    
    // Clear state
    memset(&s_state, 0, sizeof(s_state));
    s_state.mode = ROTOR_MODE_DISABLED;
    s_state.target_rpm = ROTOR_RPM_MIN;
    
    // Clear PID
    memset(&s_pid, 0, sizeof(s_pid));
    s_pid.last_update_tick = HAL_GetTick();
    
    // Clear RPM measurement
    memset(&s_rpm_meas, 0, sizeof(s_rpm_meas));
    s_rpm_meas.last_tick = HAL_GetTick();
    
    // Clear phase lock
    memset(&s_phase_lock, 0, sizeof(s_phase_lock));
    
    return HAL_OK;
}


/* ============================================================================
 * PUBLIC FUNCTIONS - BASIC CONTROL
 * ============================================================================ */

HAL_Status_t RotorControl_Enable(void) {
    if (s_state.stalled) {
        // Clear stall and reset PID
        s_state.stalled = false;
        s_pid.integral = 0.0f;
    }
    
    HAL_Motor_Enable(true);
    s_state.mode = ROTOR_MODE_MANUAL;
    
    return HAL_OK;
}


HAL_Status_t RotorControl_Disable(void) {
    HAL_Motor_Enable(false);
    HAL_Motor_SetDuty(0.0f);
    
    s_state.mode = ROTOR_MODE_DISABLED;
    s_state.duty_cycle = 0.0f;
    
    return HAL_OK;
}


HAL_Status_t RotorControl_SetRPM(float rpm) {
    if (rpm < 0.0f) return HAL_INVALID_PARAM;
    
    // Clamp to valid range
    if (rpm < ROTOR_RPM_MIN) rpm = ROTOR_RPM_MIN;
    if (rpm > ROTOR_RPM_MAX) rpm = ROTOR_RPM_MAX;
    
    s_state.target_rpm = rpm;
    s_state.target_z = rpm_to_z(rpm);
    s_state.at_target = false;
    
    if (s_state.mode == ROTOR_MODE_DISABLED) {
        s_state.mode = ROTOR_MODE_MANUAL;
    }
    
    return HAL_OK;
}


/* ============================================================================
 * PUBLIC FUNCTIONS - Z-COORDINATE CONTROL
 * ============================================================================ */

HAL_Status_t RotorControl_SetZ(float z) {
    if (z < 0.0f || z > 1.0f) return HAL_INVALID_PARAM;
    
    s_state.target_z = z;
    s_state.target_rpm = z_to_rpm(z);
    s_state.at_target = false;
    
    if (s_state.mode != ROTOR_MODE_NEGENTROPY_TRACK) {
        s_state.mode = ROTOR_MODE_Z_TRACK;
    }
    
    return HAL_OK;
}


HAL_Status_t RotorControl_SetZWithModulation(float z_target, float modulation_gain) {
    /**
     * Set z with negentropy-based modulation
     * 
     * The system uses ΔS_neg gradient to:
     * - Speed up when approaching z_c (positive gradient)
     * - Slow down when receding (negative gradient)
     * 
     * This creates a natural attractor at z_c.
     */
    
    float gradient = compute_delta_s_neg_gradient(z_target);
    
    // Modulate target z based on gradient
    // Positive gradient (approaching z_c): increase z slightly
    // Negative gradient (past z_c): decrease z
    float z_modulated = z_target + modulation_gain * gradient * 0.01f;
    
    // Clamp to valid range
    if (z_modulated < 0.0f) z_modulated = 0.0f;
    if (z_modulated > 0.98f) z_modulated = 0.98f;  // Don't exceed 1
    
    s_state.target_z = z_modulated;
    s_state.target_rpm = z_to_rpm(z_modulated);
    s_state.mode = ROTOR_MODE_NEGENTROPY_TRACK;
    
    return HAL_OK;
}


HAL_Status_t RotorControl_SweepZ(float z_start, float z_end, 
                                  float rate_per_second) {
    /**
     * Sweep z linearly from start to end
     * 
     * Useful for:
     * - Calibration (sweeping through z_c)
     * - Mapping negentropy landscape
     * - Detecting phase transitions
     */
    
    if (z_start < 0.0f || z_start > 1.0f) return HAL_INVALID_PARAM;
    if (z_end < 0.0f || z_end > 1.0f) return HAL_INVALID_PARAM;
    if (rate_per_second <= 0.0f) return HAL_INVALID_PARAM;
    
    // Store sweep parameters (would need persistent state for async operation)
    // For now, this is a blocking sweep
    
    float z = z_start;
    float direction = (z_end > z_start) ? 1.0f : -1.0f;
    float dt = 0.001f;  // 1ms steps
    
    while ((direction > 0 && z < z_end) || (direction < 0 && z > z_end)) {
        RotorControl_SetZ(z);
        RotorControl_Update();  // Run control loop
        
        HAL_Delay(1);  // 1ms delay
        z += direction * rate_per_second * dt;
    }
    
    // Set final position
    return RotorControl_SetZ(z_end);
}


/* ============================================================================
 * PUBLIC FUNCTIONS - PHASE LOCKING
 * ============================================================================ */

HAL_Status_t RotorControl_EnablePhaseLock(float target_phase_rad, 
                                           float tolerance_rad) {
    /**
     * Lock rotor to specific angular phase
     * 
     * Used for:
     * - Hexagonal pattern alignment (60° multiples)
     * - Icosahedral projection angles
     * - Synchronization with RF pulses
     */
    
    if (target_phase_rad < 0.0f) target_phase_rad += 2.0f * M_PI;
    if (target_phase_rad >= 2.0f * M_PI) target_phase_rad -= 2.0f * M_PI;
    
    s_phase_lock.enabled = true;
    s_phase_lock.target_phase = target_phase_rad;
    s_phase_lock.phase_tolerance = tolerance_rad;
    s_phase_lock.lock_count = 0;
    
    return HAL_OK;
}


HAL_Status_t RotorControl_DisablePhaseLock(void) {
    s_phase_lock.enabled = false;
    return HAL_OK;
}


HAL_Status_t RotorControl_SetHexagonalPhase(uint8_t sector) {
    /**
     * Lock to one of 6 hexagonal sectors (0-5)
     * 
     * Sector angles: 0°, 60°, 120°, 180°, 240°, 300°
     * sin(60°) = √3/2 = z_c - connects geometry to critical threshold
     */
    
    if (sector > 5) return HAL_INVALID_PARAM;
    
    float phase = sector * (M_PI / 3.0f);  // 60° per sector
    return RotorControl_EnablePhaseLock(phase, M_PI / 36.0f);  // ±5° tolerance
}


/* ============================================================================
 * PUBLIC FUNCTIONS - CONTROL LOOP UPDATE
 * ============================================================================ */

HAL_Status_t RotorControl_Update(void) {
    /**
     * Main control loop - call at PID_UPDATE_RATE_HZ (1 kHz)
     * 
     * Performs:
     * 1. RPM measurement from encoder
     * 2. z-coordinate computation
     * 3. PID control update
     * 4. Phase lock adjustment (if enabled)
     * 5. Safety checks
     */
    
    if (s_state.mode == ROTOR_MODE_DISABLED) {
        return HAL_OK;
    }
    
    // Check for motor fault
    if (HAL_Motor_IsFault()) {
        RotorControl_Disable();
        s_state.stalled = true;
        return HAL_ERROR;
    }
    
    // Measure current RPM
    s_state.actual_rpm = measure_rpm();
    s_state.actual_z = rpm_to_z(s_state.actual_rpm);
    s_state.encoder_count = HAL_Motor_GetEncoderCount();
    s_state.angular_position = compute_angular_position();
    
    // Update revolution counter
    static uint32_t last_count = 0;
    if (s_state.encoder_count < last_count && 
        (last_count - s_state.encoder_count) > ENCODER_CPR / 2) {
        s_state.revolutions++;  // Wrapped around
    }
    last_count = s_state.encoder_count;
    
    // Check for stall
    s_state.stalled = check_stall();
    if (s_state.stalled) {
        // Disable on stall to prevent damage
        RotorControl_Disable();
        return HAL_ERROR;
    }
    
    // Compute time delta
    uint32_t current_tick = HAL_GetTick();
    float dt = (current_tick - s_pid.last_update_tick) / 1000.0f;
    s_pid.last_update_tick = current_tick;
    if (dt <= 0.0f || dt > 0.1f) dt = 0.001f;  // Sanity check
    
    // Speed ramping for smooth transitions
    static float ramped_target = ROTOR_RPM_MIN;
    float ramp_step = SPEED_RAMP_RATE * dt;
    if (ramped_target < s_state.target_rpm) {
        ramped_target += ramp_step;
        if (ramped_target > s_state.target_rpm) ramped_target = s_state.target_rpm;
    } else if (ramped_target > s_state.target_rpm) {
        ramped_target -= ramp_step;
        if (ramped_target < s_state.target_rpm) ramped_target = s_state.target_rpm;
    }
    
    // PID control
    float pid_output = compute_pid(ramped_target, s_state.actual_rpm, dt);
    
    // Convert PID output to duty cycle
    // Feedforward: estimate duty for target RPM
    float feedforward = ramped_target / ROTOR_RPM_MAX * 0.5f;
    float duty = feedforward + pid_output * 0.001f;  // Scale PID output
    
    // Phase lock adjustment (micro-adjustments to duty)
    if (s_phase_lock.enabled) {
        float phase_error = s_phase_lock.target_phase - s_state.angular_position;
        // Wrap to [-π, π]
        while (phase_error > M_PI) phase_error -= 2.0f * M_PI;
        while (phase_error < -M_PI) phase_error += 2.0f * M_PI;
        
        // Small proportional correction
        duty += phase_error * 0.01f;
        
        // Track lock status
        if (fabsf(phase_error) < s_phase_lock.phase_tolerance) {
            s_phase_lock.lock_count++;
        } else {
            s_phase_lock.lock_count = 0;
        }
    }
    
    // Apply duty cycle
    apply_duty(duty);
    
    // Check if at target
    float rpm_error = fabsf(s_state.actual_rpm - s_state.target_rpm);
    s_state.at_target = (rpm_error < s_state.target_rpm * 0.02f);  // Within 2%
    
    return HAL_OK;
}


/* ============================================================================
 * PUBLIC FUNCTIONS - PHYSICS QUERIES
 * ============================================================================ */

float RotorControl_GetZ(void) {
    return s_state.actual_z;
}


float RotorControl_GetDeltaSNeg(void) {
    return compute_delta_s_neg(s_state.actual_z);
}


float RotorControl_GetComplexity(void) {
    return compute_complexity(s_state.actual_z);
}


PhysicsTier_t RotorControl_GetTier(void) {
    return get_tier(s_state.actual_z);
}


PhysicsPhase_t RotorControl_GetPhase(void) {
    return get_phase(s_state.actual_z);
}


bool RotorControl_IsAtCritical(void) {
    return is_at_critical(s_state.actual_z, TOLERANCE_LENS);
}


/* ============================================================================
 * PUBLIC FUNCTIONS - STATE ACCESS
 * ============================================================================ */

void RotorControl_GetState(RotorController_State_t *state) {
    if (state != NULL) {
        *state = s_state;
    }
}


float RotorControl_GetRPM(void) {
    return s_state.actual_rpm;
}


float RotorControl_GetAngularPosition(void) {
    return s_state.angular_position;
}


uint32_t RotorControl_GetRevolutions(void) {
    return s_state.revolutions;
}


bool RotorControl_IsAtTarget(void) {
    return s_state.at_target;
}


bool RotorControl_IsStalled(void) {
    return s_state.stalled;
}


bool RotorControl_IsPhaseLocked(void) {
    return s_phase_lock.enabled && s_phase_lock.lock_count > 10;
}


/* ============================================================================
 * INTERRUPT CALLBACKS
 * ============================================================================ */

void HAL_Callback_EncoderIndex(uint32_t count) {
    // Index pulse detected - once per revolution
    s_state.index_detected = true;
    s_state.revolutions++;
    
    // Could use this for absolute position reference
}


void HAL_Callback_MotorFault(void) {
    // Emergency disable on fault
    HAL_Motor_Enable(false);
    HAL_Motor_SetDuty(0.0f);
    s_state.stalled = true;
    s_state.mode = ROTOR_MODE_DISABLED;
}
