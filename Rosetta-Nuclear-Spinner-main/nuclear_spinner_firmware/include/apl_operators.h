/**
 * @file apl_operators.h
 * @brief Physics-Grounded APL Operator Interface
 *
 * Public interface for the 6 APL operators with physics semantics.
 *
 * OPERATOR    APL    PHYSICS SEMANTICS            MIN TIER
 * ────────    ───    ─────────────────            ────────
 * CLOSURE     ∂      Boundary formation           REACTIVE (1)
 * FUSION      +      Integration                  MEMORY (2)
 * AMPLIFY     ×      Signal amplification         PATTERN (3)
 * DECOHERE    ÷      Controlled decoherence       META (6)
 * GROUP       ⍴      Categorical grouping         PREDICTION (4)
 * SEPARATE    ↓      Differentiation              UNIVERSAL (5)
 *
 * Parity Classification:
 * - Integrative (even): FUSION, AMPLIFY, GROUP
 * - Separative (odd): CLOSURE, DECOHERE, SEPARATE
 *
 * Signature: apl-operators|v1.0.0|helix
 *
 * @version 1.0.0
 */

#ifndef APL_OPERATORS_H
#define APL_OPERATORS_H

#include <stdint.h>
#include <stdbool.h>
#include "physics_constants.h"
#include "hal_hardware.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * OPERATOR PHYSICS STRUCTURE
 * ============================================================================ */

/**
 * @brief Operator physics parameters
 */
typedef struct {
    OperatorFlags_t flag;           /**< Operator identifier */
    PhysicsTier_t min_tier;         /**< Minimum tier required */
    bool is_integrative;            /**< True for even parity */
    float energy_cost;              /**< Landauer energy cost (bits) */
    float kappa_delta;              /**< Effect on κ */
    float coupling_factor;          /**< Kuramoto coupling modification */
    float phase_rotation;           /**< RF phase rotation (radians) */
} OperatorPhysics_t;


/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

/**
 * @brief Initialize APL operator subsystem
 * @return HAL_OK on success
 */
HAL_Status_t APL_Operators_Init(void);


/* ============================================================================
 * EXECUTION
 * ============================================================================ */

/**
 * @brief Execute operator with full physics grounding
 *
 * Enforces tier gating, parity selection, and conservation.
 *
 * @param op Operator flag
 * @return HAL_OK if executed successfully
 */
HAL_Status_t APL_Operators_Execute(OperatorFlags_t op);

/**
 * @brief Register callback for operator execution
 *
 * @param op Operator flag
 * @param callback Function to call before physics application
 * @return HAL_OK on success
 */
HAL_Status_t APL_Operators_RegisterCallback(OperatorFlags_t op, OperatorCallback_t callback);


/* ============================================================================
 * STATE QUERIES
 * ============================================================================ */

/**
 * @brief Get operator execution count
 *
 * @param op Operator flag
 * @return Number of times operator has been executed
 */
uint32_t APL_Operators_GetCount(OperatorFlags_t op);

/**
 * @brief Get cumulative Landauer energy expenditure
 * @return Energy in bits
 */
float APL_Operators_GetEnergy(void);

/**
 * @brief Check if operator matches current parity preference
 *
 * @param op Operator flag
 * @return true if parity matches current state
 */
bool APL_Operators_ParityMatch(OperatorFlags_t op);

/**
 * @brief Get physics weight for operator at current state
 *
 * @param op Operator flag
 * @return Weight in [0, 1]
 */
float APL_Operators_GetWeight(OperatorFlags_t op);

/**
 * @brief Get recommended operator for current state
 *
 * @return Most effective operator flag at current tier/parity
 */
OperatorFlags_t APL_Operators_GetRecommended(void);


/* ============================================================================
 * INFORMATION
 * ============================================================================ */

/**
 * @brief Get operator name string
 *
 * @param op Operator flag
 * @return Static string with APL symbol and name
 */
const char* APL_Operators_GetName(OperatorFlags_t op);

/**
 * @brief Get operator physics parameters
 *
 * @param op Operator flag
 * @param phys Output physics structure
 * @return HAL_OK if valid operator
 */
HAL_Status_t APL_Operators_GetPhysics(OperatorFlags_t op, OperatorPhysics_t* phys);


#ifdef __cplusplus
}
#endif

#endif /* APL_OPERATORS_H */
