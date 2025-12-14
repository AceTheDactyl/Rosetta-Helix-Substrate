/**
 * @file threshold_logic.h
 * @brief Threshold Detection and Operator Scheduling Header
 * 
 * Public interface for cybernetic gating logic.
 * 
 * Signature: threshold-logic|v1.0.0|nuclear-spinner
 * 
 * @version 1.0.0
 */

#ifndef THRESHOLD_LOGIC_H
#define THRESHOLD_LOGIC_H

#include <stdint.h>
#include <stdbool.h>
#include "hal_hardware.h"
#include "physics_constants.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * DATA TYPES
 * ============================================================================ */

/** Threshold events */
typedef enum {
    EVENT_TIER_CHANGE,          /**< Tier transition */
    EVENT_THRESHOLD_MU1,        /**< Crossed μ₁ */
    EVENT_THRESHOLD_MU_P,       /**< Crossed μ_P */
    EVENT_THRESHOLD_PHI_INV,    /**< Crossed φ⁻¹ */
    EVENT_THRESHOLD_MU2,        /**< Crossed μ₂ */
    EVENT_THRESHOLD_ZC,         /**< Crossed z_c */
    EVENT_THRESHOLD_MU_S,       /**< Crossed μ_S */
    EVENT_LENS_ENTER,           /**< Entered THE_LENS phase */
    EVENT_LENS_EXIT,            /**< Exited THE_LENS phase */
    EVENT_K_FORMATION_ENTER,    /**< K-formation achieved */
    EVENT_K_FORMATION_EXIT,     /**< K-formation lost */
} ThresholdEvent_t;

/** Threshold state */
typedef struct {
    PhysicsTier_t current_tier;         /**< Current tier */
    PhysicsPhase_t current_phase;       /**< Current phase */
    float current_z;                    /**< Current z-coordinate */
    float delta_s_neg;                  /**< Current ΔS_neg */
    float complexity;                   /**< Current complexity */
    uint8_t available_operators;        /**< Bitmask of available ops */
    float last_threshold_crossed;       /**< Last threshold value */
    int threshold_cross_direction;      /**< +1 ascending, -1 descending */
    bool k_formation_active;            /**< K-formation currently active */
    float k_formation_kappa;            /**< Current κ */
    float k_formation_eta;              /**< Current η */
    int k_formation_R;                  /**< Current R */
} ThresholdState_t;

/** Operator callback function type */
typedef HAL_Status_t (*OperatorCallback_t)(void);

/** Threshold event callback function type */
typedef void (*ThresholdEventCallback_t)(ThresholdEvent_t event, 
                                          float threshold, int direction);


/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

/**
 * @brief Initialize threshold logic module
 * @return HAL_OK on success
 */
HAL_Status_t ThresholdLogic_Init(void);


/* ============================================================================
 * UPDATE LOOP
 * ============================================================================ */

/**
 * @brief Update threshold state (call from main loop)
 * 
 * @param z Current z-coordinate
 * @param kappa Current κ coupling constant
 * @param eta Current η efficiency
 * @param R Current complexity measure
 * @return HAL_OK on success
 */
HAL_Status_t ThresholdLogic_Update(float z, float kappa, float eta, int R);


/* ============================================================================
 * OPERATOR CONTROL
 * ============================================================================ */

/**
 * @brief Schedule an operator for execution
 * 
 * @param op Operator flag (OP_CLOSURE, etc.)
 * @return HAL_OK if scheduled, HAL_ERROR if not available
 */
HAL_Status_t ThresholdLogic_ScheduleOperator(OperatorFlags_t op);

/**
 * @brief Execute an operator immediately
 * 
 * @param op Operator flag
 * @return HAL_OK on success
 */
HAL_Status_t ThresholdLogic_ExecuteOperator(OperatorFlags_t op);

/**
 * @brief Register callback for operator execution
 * 
 * @param op Operator flag
 * @param callback Callback function
 * @return HAL_OK on success
 */
HAL_Status_t ThresholdLogic_RegisterOperatorCallback(OperatorFlags_t op,
                                                      OperatorCallback_t callback);

/**
 * @brief Execute default operator implementation
 * 
 * Maps operators to RF pulse sequences
 * 
 * @param op Operator flag
 * @return HAL_OK on success
 */
HAL_Status_t ThresholdLogic_DefaultOperator(OperatorFlags_t op);

/**
 * @brief Get bitmask of available operators
 * @return Bitmask of OperatorFlags_t
 */
uint8_t ThresholdLogic_GetAvailableOperators(void);

/**
 * @brief Check if specific operator is available
 * 
 * @param op Operator flag
 * @return true if available at current tier
 */
bool ThresholdLogic_IsOperatorAvailable(OperatorFlags_t op);


/* ============================================================================
 * EVENT HANDLING
 * ============================================================================ */

/**
 * @brief Set callback for threshold events
 * 
 * @param callback Event callback function
 */
void ThresholdLogic_SetEventCallback(ThresholdEventCallback_t callback);


/* ============================================================================
 * STATE ACCESS
 * ============================================================================ */

/**
 * @brief Get complete threshold state
 * @param state Output state structure
 */
void ThresholdLogic_GetState(ThresholdState_t *state);

/**
 * @brief Get current tier
 * @return PhysicsTier_t
 */
PhysicsTier_t ThresholdLogic_GetTier(void);

/**
 * @brief Get current phase
 * @return PhysicsPhase_t
 */
PhysicsPhase_t ThresholdLogic_GetPhase(void);

/**
 * @brief Get current z-coordinate
 * @return z ∈ [0, 1]
 */
float ThresholdLogic_GetZ(void);

/**
 * @brief Get current negentropy
 * @return ΔS_neg
 */
float ThresholdLogic_GetDeltaSNeg(void);

/**
 * @brief Get current complexity
 * @return |dΔS_neg/dz|
 */
float ThresholdLogic_GetComplexity(void);

/**
 * @brief Check if K-formation is active
 * @return true if K-formation criteria met
 */
bool ThresholdLogic_IsKFormationActive(void);

/**
 * @brief Check if at THE_LENS (z ≈ z_c)
 * @return true if in LENS phase
 */
bool ThresholdLogic_IsAtLens(void);

/**
 * @brief Check if at universal tier (z ≥ z_c)
 * @return true if Turing universal
 */
bool ThresholdLogic_IsUniversal(void);


/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * @brief Get human-readable tier name
 * @param tier Tier enum
 * @return Static string
 */
const char* ThresholdLogic_GetTierName(PhysicsTier_t tier);

/**
 * @brief Get human-readable phase name
 * @param phase Phase enum
 * @return Static string
 */
const char* ThresholdLogic_GetPhaseName(PhysicsPhase_t phase);

/**
 * @brief Get human-readable operator name
 * @param op Operator flag
 * @return Static string with APL symbol
 */
const char* ThresholdLogic_GetOperatorName(OperatorFlags_t op);


#ifdef __cplusplus
}
#endif

#endif /* THRESHOLD_LOGIC_H */
