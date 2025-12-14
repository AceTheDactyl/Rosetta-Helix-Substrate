/**
 * @file rotor_control.h
 * @brief Precision Rotor Control with Physics z-Mapping - Public API
 *
 * Signature: rotor-control|v1.0.0|nuclear-spinner
 *
 * @version 1.0.0
 */

#ifndef ROTOR_CONTROL_H
#define ROTOR_CONTROL_H

#include <stdint.h>
#include <stdbool.h>
#include "hal_hardware.h"
#include "physics_constants.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Rotor operating mode */
typedef enum {
    ROTOR_MODE_DISABLED = 0,        /**< Motor disabled */
    ROTOR_MODE_MANUAL,             /**< Direct RPM control */
    ROTOR_MODE_Z_TRACK,            /**< Track target z via RPM mapping */
    ROTOR_MODE_NEGENTROPY_TRACK,   /**< Track z with ΔS_neg-gradient modulation */
} RotorMode_t;

/** Rotor controller state snapshot */
typedef struct {
    RotorMode_t mode;              /**< Current control mode */

    float target_z;                /**< Target z ∈ [0,1] */
    float target_rpm;              /**< Target RPM */

    float actual_rpm;              /**< Measured RPM */
    float actual_z;                /**< Estimated z from RPM */

    float duty_cycle;              /**< PWM duty [0,1] */

    uint32_t encoder_count;        /**< Raw encoder count */
    uint32_t revolutions;          /**< Revolution counter */
    float angular_position;        /**< 0..2π radians */

    bool at_target;                /**< True if within tolerance */
    bool stalled;                  /**< Stall detected */
    bool index_detected;           /**< Index pulse observed */
} RotorController_State_t;

/* Initialization */
HAL_Status_t RotorControl_Init(void);

/* Enable/Disable */
HAL_Status_t RotorControl_Enable(void);
HAL_Status_t RotorControl_Disable(void);

/* Setpoints */
HAL_Status_t RotorControl_SetRPM(float rpm);
HAL_Status_t RotorControl_SetZ(float z);
HAL_Status_t RotorControl_SetZWithModulation(float z_target, float modulation_gain);
HAL_Status_t RotorControl_SweepZ(float z_start, float z_end, float z_step);

/* Phase lock */
HAL_Status_t RotorControl_EnablePhaseLock(float target_phase_rad, float tolerance_rad);
HAL_Status_t RotorControl_DisablePhaseLock(void);
HAL_Status_t RotorControl_SetHexagonalPhase(uint8_t sector);

/* Control loop */
HAL_Status_t RotorControl_Update(void);

/* Physics queries */
float RotorControl_GetZ(void);
float RotorControl_GetDeltaSNeg(void);
float RotorControl_GetComplexity(void);
PhysicsTier_t RotorControl_GetTier(void);
PhysicsPhase_t RotorControl_GetPhase(void);
bool RotorControl_IsAtCritical(void);

/* State access */
void RotorControl_GetState(RotorController_State_t *state);
float RotorControl_GetRPM(void);
float RotorControl_GetAngularPosition(void);
uint32_t RotorControl_GetRevolutions(void);
bool RotorControl_IsAtTarget(void);
bool RotorControl_IsStalled(void);
bool RotorControl_IsPhaseLocked(void);

#ifdef __cplusplus
}
#endif

#endif /* ROTOR_CONTROL_H */
