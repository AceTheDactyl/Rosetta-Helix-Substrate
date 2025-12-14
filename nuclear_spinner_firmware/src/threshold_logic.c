/**
 * @file threshold_logic.c
 * @brief Threshold Detection and Operator Scheduling
 * 
 * Implements the cybernetic gating logic from the Rosetta-Helix framework:
 * - Tier detection based on z-coordinate thresholds
 * - APL operator scheduling and execution
 * - K-formation detection
 * - Negentropy-based state transitions
 * 
 * The threshold logic enforces computational capability constraints:
 * - z < μ₁: No operations (ABSENCE)
 * - z ∈ [μ₁, μ_P): Reactive only (boundary operations)
 * - z ∈ [μ_P, φ⁻¹): Memory operations (state retention)
 * - z ∈ [φ⁻¹, μ₂): Pattern recognition
 * - z ∈ [μ₂, z_c): Prediction capability
 * - z ∈ [z_c, μ_S): Full Turing universality
 * - z ≥ μ_S: Meta-cognitive recursion
 * 
 * Signature: threshold-logic|v1.0.0|nuclear-spinner
 * 
 * @version 1.0.0
 */

#include "threshold_logic.h"
#include "hal_hardware.h"
#include "physics_constants.h"
#include "pulse_control.h"
#include "rotor_control.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * PRIVATE CONSTANTS
 * ============================================================================ */

/** Hysteresis for threshold crossing (prevents oscillation) */
#define THRESHOLD_HYSTERESIS    0.01f

/** Minimum time between operator executions (ms) */
#define MIN_OP_INTERVAL_MS      10

/** Maximum scheduled operations in queue */
#define OP_QUEUE_SIZE           16

/** K-formation stability count requirement */
#define K_FORMATION_STABLE_COUNT 10


/* ============================================================================
 * PRIVATE DATA
 * ============================================================================ */

/** Threshold state */
static ThresholdState_t s_state = {
    .current_tier = TIER_ABSENCE,
    .current_phase = PHASE_ABSENCE,
    .current_z = 0.0f,
    .delta_s_neg = 0.0f,
    .complexity = 0.0f,
    .available_operators = 0,
    .last_threshold_crossed = 0.0f,
    .threshold_cross_direction = 0,
    .k_formation_active = false,
    .k_formation_kappa = 0.0f,
    .k_formation_eta = 0.0f,
    .k_formation_R = 0,
};

/** Previous tier for transition detection */
static PhysicsTier_t s_prev_tier = TIER_ABSENCE;

/** Threshold crossing timestamps */
static uint32_t s_threshold_timestamps[7] = {0};  // For each μ threshold

/** Operator execution state */
static struct {
    OperatorFlags_t pending;           // Operators waiting to execute
    OperatorFlags_t active;            // Currently executing
    uint32_t last_exec_tick;           // Last execution timestamp
    OperatorCallback_t callbacks[6];   // One per operator
} s_operators = {0};

/** K-formation tracking */
static struct {
    float kappa_history[K_FORMATION_STABLE_COUNT];
    float eta_history[K_FORMATION_STABLE_COUNT];
    uint32_t history_idx;
    uint32_t stable_count;
} s_k_track = {0};

/** Event callback */
static ThresholdEventCallback_t s_event_callback = NULL;


/* ============================================================================
 * PRIVATE FUNCTIONS
 * ============================================================================ */

/**
 * @brief Check if threshold was crossed with hysteresis
 */
static bool check_threshold_crossing(float prev_z, float curr_z, 
                                      float threshold, int *direction) {
    // Crossing from below
    if (prev_z < (threshold - THRESHOLD_HYSTERESIS) && 
        curr_z >= (threshold + THRESHOLD_HYSTERESIS)) {
        *direction = 1;  // Ascending
        return true;
    }
    // Crossing from above
    if (prev_z > (threshold + THRESHOLD_HYSTERESIS) && 
        curr_z <= (threshold - THRESHOLD_HYSTERESIS)) {
        *direction = -1;  // Descending
        return true;
    }
    return false;
}


/**
 * @brief Fire threshold event if callback registered
 */
static void fire_threshold_event(ThresholdEvent_t event, float threshold, 
                                  int direction) {
    if (s_event_callback != NULL) {
        s_event_callback(event, threshold, direction);
    }
    
    s_state.last_threshold_crossed = threshold;
    s_state.threshold_cross_direction = direction;
}


/**
 * @brief Update K-formation tracking
 */
static void update_k_formation(float kappa, float eta, int R) {
    // Store in circular buffer
    s_k_track.kappa_history[s_k_track.history_idx] = kappa;
    s_k_track.eta_history[s_k_track.history_idx] = eta;
    s_k_track.history_idx = (s_k_track.history_idx + 1) % K_FORMATION_STABLE_COUNT;
    
    // Check K-formation criteria across history
    bool all_pass = true;
    for (int i = 0; i < K_FORMATION_STABLE_COUNT; i++) {
        if (!check_k_formation(s_k_track.kappa_history[i], 
                               s_k_track.eta_history[i], R)) {
            all_pass = false;
            break;
        }
    }
    
    if (all_pass) {
        s_k_track.stable_count++;
    } else {
        s_k_track.stable_count = 0;
    }
    
    // K-formation requires stability
    bool was_active = s_state.k_formation_active;
    s_state.k_formation_active = (s_k_track.stable_count >= K_FORMATION_STABLE_COUNT);
    s_state.k_formation_kappa = kappa;
    s_state.k_formation_eta = eta;
    s_state.k_formation_R = R;
    
    // Fire event on transition
    if (s_state.k_formation_active && !was_active) {
        fire_threshold_event(EVENT_K_FORMATION_ENTER, kappa, 1);
    } else if (!s_state.k_formation_active && was_active) {
        fire_threshold_event(EVENT_K_FORMATION_EXIT, kappa, -1);
    }
}


/**
 * @brief Execute a single operator
 */
static HAL_Status_t execute_operator(OperatorFlags_t op) {
    // Check if allowed at current tier
    if (!(s_state.available_operators & op)) {
        return HAL_ERROR;  // Operator not available
    }
    
    // Check minimum interval
    if (HAL_GetTick() - s_operators.last_exec_tick < MIN_OP_INTERVAL_MS) {
        return HAL_BUSY;
    }
    
    s_operators.active = op;
    s_operators.last_exec_tick = HAL_GetTick();
    
    HAL_Status_t status = HAL_OK;
    
    // Call registered callback if present
    int op_idx = 0;
    switch (op) {
        case OP_CLOSURE:   op_idx = 0; break;
        case OP_FUSION:    op_idx = 1; break;
        case OP_AMPLIFY:   op_idx = 2; break;
        case OP_DECOHERE:  op_idx = 3; break;
        case OP_GROUP:     op_idx = 4; break;
        case OP_SEPARATE:  op_idx = 5; break;
        default: return HAL_INVALID_PARAM;
    }
    
    if (s_operators.callbacks[op_idx] != NULL) {
        status = s_operators.callbacks[op_idx]();
    } else {
        // Default operator implementations
        status = ThresholdLogic_DefaultOperator(op);
    }
    
    s_operators.active = 0;
    s_operators.pending &= ~op;  // Clear from pending
    
    return status;
}


/* ============================================================================
 * PUBLIC FUNCTIONS - INITIALIZATION
 * ============================================================================ */

HAL_Status_t ThresholdLogic_Init(void) {
    // Clear state
    memset(&s_state, 0, sizeof(s_state));
    s_state.current_tier = TIER_ABSENCE;
    s_state.current_phase = PHASE_ABSENCE;
    
    // Clear tracking
    s_prev_tier = TIER_ABSENCE;
    memset(s_threshold_timestamps, 0, sizeof(s_threshold_timestamps));
    memset(&s_operators, 0, sizeof(s_operators));
    memset(&s_k_track, 0, sizeof(s_k_track));
    
    s_event_callback = NULL;
    
    return HAL_OK;
}


/* ============================================================================
 * PUBLIC FUNCTIONS - UPDATE LOOP
 * ============================================================================ */

HAL_Status_t ThresholdLogic_Update(float z, float kappa, float eta, int R) {
    /**
     * Main update function - call from main loop
     * 
     * @param z Current z-coordinate (from rotor)
     * @param kappa Current κ coupling constant
     * @param eta Current η efficiency measure
     * @param R Current complexity measure R
     */
    
    // Store previous z for crossing detection
    static float prev_z = 0.0f;
    
    // Update basic state
    s_state.current_z = z;
    s_state.delta_s_neg = compute_delta_s_neg(z);
    s_state.complexity = compute_complexity(z);
    
    // Determine new tier and phase
    PhysicsTier_t new_tier = get_tier(z);
    PhysicsPhase_t new_phase = get_phase(z);
    
    // Check for tier change
    if (new_tier != s_prev_tier) {
        // Fire tier transition event
        int direction = (new_tier > s_prev_tier) ? 1 : -1;
        fire_threshold_event(EVENT_TIER_CHANGE, z, direction);
        
        // Update available operators
        s_state.available_operators = get_available_operators(new_tier);
        
        s_prev_tier = new_tier;
    }
    
    // Check for phase change
    if (new_phase != s_state.current_phase) {
        int direction = (new_phase > s_state.current_phase) ? 1 : -1;
        
        if (new_phase == PHASE_THE_LENS) {
            fire_threshold_event(EVENT_LENS_ENTER, Z_CRITICAL, direction);
        } else if (s_state.current_phase == PHASE_THE_LENS) {
            fire_threshold_event(EVENT_LENS_EXIT, Z_CRITICAL, direction);
        }
    }
    
    s_state.current_tier = new_tier;
    s_state.current_phase = new_phase;
    
    // Check individual threshold crossings
    int direction;
    
    if (check_threshold_crossing(prev_z, z, MU_1, &direction)) {
        fire_threshold_event(EVENT_THRESHOLD_MU1, MU_1, direction);
        s_threshold_timestamps[0] = HAL_GetTick();
    }
    
    if (check_threshold_crossing(prev_z, z, MU_P, &direction)) {
        fire_threshold_event(EVENT_THRESHOLD_MU_P, MU_P, direction);
        s_threshold_timestamps[1] = HAL_GetTick();
    }
    
    if (check_threshold_crossing(prev_z, z, MU_PHI_INV, &direction)) {
        fire_threshold_event(EVENT_THRESHOLD_PHI_INV, MU_PHI_INV, direction);
        s_threshold_timestamps[2] = HAL_GetTick();
    }
    
    if (check_threshold_crossing(prev_z, z, MU_2, &direction)) {
        fire_threshold_event(EVENT_THRESHOLD_MU2, MU_2, direction);
        s_threshold_timestamps[3] = HAL_GetTick();
    }
    
    if (check_threshold_crossing(prev_z, z, MU_ZC, &direction)) {
        fire_threshold_event(EVENT_THRESHOLD_ZC, MU_ZC, direction);
        s_threshold_timestamps[4] = HAL_GetTick();
    }
    
    if (check_threshold_crossing(prev_z, z, MU_S, &direction)) {
        fire_threshold_event(EVENT_THRESHOLD_MU_S, MU_S, direction);
        s_threshold_timestamps[5] = HAL_GetTick();
    }
    
    // Update K-formation tracking
    update_k_formation(kappa, eta, R);
    
    // Process pending operators
    if (s_operators.pending != 0) {
        // Execute highest priority pending operator
        for (int i = 0; i < 6; i++) {
            OperatorFlags_t op = (1 << i);
            if (s_operators.pending & op) {
                execute_operator(op);
                break;  // One at a time
            }
        }
    }
    
    prev_z = z;
    
    return HAL_OK;
}


/* ============================================================================
 * PUBLIC FUNCTIONS - OPERATOR CONTROL
 * ============================================================================ */

HAL_Status_t ThresholdLogic_ScheduleOperator(OperatorFlags_t op) {
    // Check if operator is available at current tier
    if (!(s_state.available_operators & op)) {
        return HAL_ERROR;
    }
    
    s_operators.pending |= op;
    return HAL_OK;
}


HAL_Status_t ThresholdLogic_ExecuteOperator(OperatorFlags_t op) {
    return execute_operator(op);
}


HAL_Status_t ThresholdLogic_RegisterOperatorCallback(OperatorFlags_t op,
                                                      OperatorCallback_t callback) {
    int op_idx;
    switch (op) {
        case OP_CLOSURE:   op_idx = 0; break;
        case OP_FUSION:    op_idx = 1; break;
        case OP_AMPLIFY:   op_idx = 2; break;
        case OP_DECOHERE:  op_idx = 3; break;
        case OP_GROUP:     op_idx = 4; break;
        case OP_SEPARATE:  op_idx = 5; break;
        default: return HAL_INVALID_PARAM;
    }
    
    s_operators.callbacks[op_idx] = callback;
    return HAL_OK;
}


HAL_Status_t ThresholdLogic_DefaultOperator(OperatorFlags_t op) {
    /**
     * Default operator implementations using RF pulses
     * 
     * These map APL operators to physical spin manipulations:
     * - ∂ (closure): Isolate spin system (no pulses)
     * - + (fusion): Integrative pulse sequence
     * - × (amplify): High-amplitude pulse
     * - ÷ (decohere): Noise-inducing sequence
     * - ⍴ (group): Coil retuning
     * - ↓ (separate): Phase-cycling sequence
     */
    
    HAL_Status_t status = HAL_OK;
    
    switch (op) {
        case OP_CLOSURE:
            // Boundary/isolation: disable RF output
            HAL_RF_Enable(false);
            HAL_Delay(10);  // Brief isolation period
            break;
            
        case OP_FUSION:
            // Integration/binding: composite pulse
            status = PulseControl_Pi2Pulse(0.7f, 0.0f);
            if (status == HAL_OK) {
                HAL_Delay(1);
                status = PulseControl_Pi2Pulse(0.7f, M_PI / 2.0f);
            }
            break;
            
        case OP_AMPLIFY:
            // Signal amplification: high-power pulse
            status = PulseControl_CustomPulse(1.0f, 0.0f, 100, 0);
            break;
            
        case OP_DECOHERE:
            // Controlled decoherence: random phase noise
            for (int i = 0; i < 5; i++) {
                float random_phase = (float)(HAL_GetTick() % 628) / 100.0f;
                status = PulseControl_CustomPulse(0.3f, random_phase, 10, 5);
                if (status != HAL_OK) break;
            }
            break;
            
        case OP_GROUP:
            // Categorical grouping: refocusing sequence
            status = PulseControl_SpinEcho(0.8f, 500);
            break;
            
        case OP_SEPARATE:
            // Differentiation: phase-cycled sequence
            status = PulseControl_HexagonalPattern(0.6f, 20, 1);
            break;
            
        default:
            return HAL_INVALID_PARAM;
    }
    
    return status;
}


uint8_t ThresholdLogic_GetAvailableOperators(void) {
    return s_state.available_operators;
}


bool ThresholdLogic_IsOperatorAvailable(OperatorFlags_t op) {
    return (s_state.available_operators & op) != 0;
}


/* ============================================================================
 * PUBLIC FUNCTIONS - EVENT HANDLING
 * ============================================================================ */

void ThresholdLogic_SetEventCallback(ThresholdEventCallback_t callback) {
    s_event_callback = callback;
}


/* ============================================================================
 * PUBLIC FUNCTIONS - STATE ACCESS
 * ============================================================================ */

void ThresholdLogic_GetState(ThresholdState_t *state) {
    if (state != NULL) {
        *state = s_state;
    }
}


PhysicsTier_t ThresholdLogic_GetTier(void) {
    return s_state.current_tier;
}


PhysicsPhase_t ThresholdLogic_GetPhase(void) {
    return s_state.current_phase;
}


float ThresholdLogic_GetZ(void) {
    return s_state.current_z;
}


float ThresholdLogic_GetDeltaSNeg(void) {
    return s_state.delta_s_neg;
}


float ThresholdLogic_GetComplexity(void) {
    return s_state.complexity;
}


bool ThresholdLogic_IsKFormationActive(void) {
    return s_state.k_formation_active;
}


bool ThresholdLogic_IsAtLens(void) {
    return s_state.current_phase == PHASE_THE_LENS;
}


bool ThresholdLogic_IsUniversal(void) {
    return s_state.current_tier >= TIER_UNIVERSAL;
}


/* ============================================================================
 * PUBLIC FUNCTIONS - CAPABILITY QUERIES
 * ============================================================================ */

const char* ThresholdLogic_GetTierName(PhysicsTier_t tier) {
    switch (tier) {
        case TIER_ABSENCE:    return "ABSENCE";
        case TIER_REACTIVE:   return "REACTIVE";
        case TIER_MEMORY:     return "MEMORY";
        case TIER_PATTERN:    return "PATTERN";
        case TIER_PREDICTION: return "PREDICTION";
        case TIER_UNIVERSAL:  return "UNIVERSAL";
        case TIER_META:       return "META";
        default:              return "UNKNOWN";
    }
}


const char* ThresholdLogic_GetPhaseName(PhysicsPhase_t phase) {
    switch (phase) {
        case PHASE_ABSENCE:   return "ABSENCE";
        case PHASE_THE_LENS:  return "THE_LENS";
        case PHASE_PRESENCE:  return "PRESENCE";
        default:              return "UNKNOWN";
    }
}


const char* ThresholdLogic_GetOperatorName(OperatorFlags_t op) {
    switch (op) {
        case OP_CLOSURE:  return "CLOSURE (∂)";
        case OP_FUSION:   return "FUSION (+)";
        case OP_AMPLIFY:  return "AMPLIFY (×)";
        case OP_DECOHERE: return "DECOHERE (÷)";
        case OP_GROUP:    return "GROUP (⍴)";
        case OP_SEPARATE: return "SEPARATE (↓)";
        default:          return "UNKNOWN";
    }
}
