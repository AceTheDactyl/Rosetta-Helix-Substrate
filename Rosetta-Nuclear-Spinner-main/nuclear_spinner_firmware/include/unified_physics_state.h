/**
 * @file unified_physics_state.h
 * @brief Unified Physics State for Cybernetic Feedback Loop
 *
 * Central state structure unifying:
 * - Real-time physics measurements (z, ΔS_neg, tier, phase)
 * - 19 training module states with adaptive parameters
 * - Cybernetic feedback variables (κ, λ, η, R)
 * - K-formation event handling
 * - TRIAD constraint validation
 * - Conservation law monitoring
 *
 * This is the source of truth for all cybernetic operations,
 * implementing Ashby's Law of Requisite Variety through
 * physics-grounded state management.
 *
 * Signature: unified-physics-state|v1.0.0|helix
 *
 * @author Rosetta-Helix Framework
 * @version 1.0.0
 */

#ifndef UNIFIED_PHYSICS_STATE_H
#define UNIFIED_PHYSICS_STATE_H

#include <stdint.h>
#include <stdbool.h>
#include "physics_constants.h"
#include "hal_hardware.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * TRAINING MODULE IDENTIFIERS (19 modules in 7 phases)
 * ============================================================================ */

typedef enum {
    /* Phase 1: Core Physics */
    MODULE_N0_SILENT_LAWS = 0,
    MODULE_KURAMOTO_LAYER,
    MODULE_PHYSICAL_LEARNER,

    /* Phase 2: APL Stack */
    MODULE_APL_TRAINING_LOOP,
    MODULE_PYTORCH_TRAINING,
    MODULE_FULL_APL,

    /* Phase 3: Helix Geometry */
    MODULE_HELIX_NN,
    MODULE_PRISMATIC_HELIX,
    MODULE_FULL_HELIX,

    /* Phase 4: WUMBO Silent Laws */
    MODULE_WUMBO_SILENT_LAWS,

    /* Phase 5: Dynamics Formation */
    MODULE_QUASICRYSTAL,
    MODULE_TRIAD,
    MODULE_LIMINAL,
    MODULE_FEEDBACK,

    /* Phase 6: Unified Orchestration */
    MODULE_UNIFIED_ORCHESTRATION,

    /* Phase 7: Nightly Integration */
    MODULE_NIGHTLY_MODULE_0,
    MODULE_NIGHTLY_MODULE_1,
    MODULE_NIGHTLY_MODULE_2,
    MODULE_NIGHTLY_MODULE_3,

    MODULE_COUNT  /* = 19 */
} TrainingModule_t;


/** Training phase groupings */
typedef enum {
    PHASE_CORE_PHYSICS = 0,
    PHASE_APL_STACK,
    PHASE_HELIX_GEOMETRY,
    PHASE_WUMBO_LAWS,
    PHASE_DYNAMICS_FORMATION,
    PHASE_UNIFIED_ORCHESTRATION,
    PHASE_NIGHTLY_INTEGRATION,

    TRAINING_PHASE_COUNT  /* = 7 */
} TrainingPhase_t;


/* ============================================================================
 * CYBERNETIC FEEDBACK STATE
 * ============================================================================ */

/**
 * @brief TRIAD Constraint State
 *
 * Implements the triadic threshold dynamics (κ, λ, η):
 * - κ: Coherence coupling (goal: → φ⁻¹)
 * - λ: Decoherence coupling (λ = 1 - κ)
 * - η: Efficiency (must be > φ⁻¹ for K-formation)
 *
 * Conservation law: κ + λ = 1 (EXACT)
 */
typedef struct {
    float kappa;                    /**< Coherence coupling κ ∈ [0, 1] */
    float lambda;                   /**< Decoherence coupling λ = 1 - κ */
    float eta;                      /**< Efficiency η ∈ [0, 1] */
    int R;                          /**< Complexity rank R ≥ 0 */
    float scar;                     /**< Scar depth for return dynamics */
    bool conservation_valid;        /**< κ + λ = 1 validated */
    uint32_t last_k_formation_ms;   /**< Timestamp of last K-formation */
    uint16_t k_formation_count;     /**< Total K-formations in session */
} TriadState_t;


/**
 * @brief Kuramoto Oscillator State
 *
 * 60-oscillator hexagonal network state for Heart module
 */
typedef struct {
    float coherence;                /**< Order parameter r ∈ [0, 1] */
    float mean_phase;               /**< Mean phase Ψ ∈ [0, 2π] */
    float coupling_strength;        /**< K = 8 × ΔS_neg(z) */
    uint8_t sync_clusters;          /**< Number of synchronized clusters */
    bool phase_locked;              /**< Hexagonal phase lock achieved */
} KuramotoState_t;


/**
 * @brief GHMP (Brain) Operator State
 *
 * Tier-gated operator availability and parity selection
 */
typedef struct {
    PhysicsTier_t tier;             /**< Current tier (gates operators) */
    uint8_t available_ops;          /**< Bitmask of available operators */
    OperatorFlags_t last_op;        /**< Last executed operator */
    bool parity_even;               /**< Even parity operator preference */
    float operator_weight;          /**< Physics-derived operator weight */
} GHMPState_t;


/* ============================================================================
 * ADAPTIVE TRAINING PARAMETERS
 * ============================================================================ */

/**
 * @brief Negentropy-Responsive Parameters
 *
 * Training hyperparameters that adapt based on ΔS_neg(z)
 */
typedef struct {
    float learning_rate;            /**< η_lr = η_base × (1 + ΔS_neg) */
    float gradient_clip;            /**< Adaptive gradient clipping */
    float dropout_rate;             /**< Inverse relationship with ΔS_neg */
    float weight_decay;             /**< Stability factor */
    float temperature;              /**< Softmax temperature */
} AdaptiveParams_t;


/**
 * @brief Module-Specific State
 *
 * Per-module training state with physics grounding
 */
typedef struct {
    TrainingModule_t module_id;     /**< Module identifier */
    TrainingPhase_t phase;          /**< Training phase (0-6) */
    bool active;                    /**< Module currently active */
    float progress;                 /**< Progress ∈ [0, 1] */
    float loss;                     /**< Current loss value */
    float accuracy;                 /**< Current accuracy */
    AdaptiveParams_t params;        /**< Adaptive hyperparameters */
    uint32_t step_count;            /**< Training steps completed */
    uint32_t checkpoint_step;       /**< Last checkpoint step */
} ModuleState_t;


/* ============================================================================
 * UNIFIED PHYSICS STATE
 * ============================================================================ */

/**
 * @brief Complete Unified State Structure
 *
 * The source of truth for all cybernetic operations.
 * Updated at 100 Hz from hardware measurements.
 */
typedef struct {
    /* Timestamp */
    uint32_t timestamp_ms;          /**< System timestamp */
    uint32_t frame_count;           /**< Frame counter since init */

    /* Core Physics State */
    float z;                        /**< Current z-coordinate ∈ [0, 1] */
    float z_target;                 /**< Target z-coordinate */
    float z_velocity;               /**< dz/dt for dynamics */
    float rpm;                      /**< Rotor speed (mapped from z) */
    float delta_s_neg;              /**< Negentropy signal ΔS_neg(z) */
    float delta_s_neg_gradient;     /**< d(ΔS_neg)/dz */
    float complexity;               /**< |d(ΔS_neg)/dz| */

    /* Phase & Tier */
    PhysicsPhase_t phase;           /**< Current phase (ABSENCE/LENS/PRESENCE) */
    PhysicsTier_t tier;             /**< Current tier (0-6) */
    bool at_lens;                   /**< z ≈ z_c (THE LENS) */
    bool is_universal;              /**< z ≥ z_c */

    /* K-Formation State */
    bool k_formation_active;        /**< K-formation criteria met */
    uint32_t k_formation_duration_ms; /**< Duration of current K-formation */
    uint32_t total_k_formation_ms;  /**< Total K-formation time in session */

    /* Subsystem States */
    TriadState_t triad;             /**< TRIAD (κ, λ, η) dynamics */
    KuramotoState_t kuramoto;       /**< Kuramoto oscillator state */
    GHMPState_t ghmp;               /**< GHMP operator state */

    /* Training States (19 modules) */
    ModuleState_t modules[MODULE_COUNT];
    TrainingPhase_t current_training_phase;

    /* Conservation Monitoring */
    float conservation_error;       /**< |κ + λ - 1| */
    bool physics_valid;             /**< All conservation laws satisfied */
    uint32_t violation_count;       /**< Conservation violations detected */

    /* Quasicrystal Order Parameter */
    float quasicrystal_order;       /**< Penrose tiling ratio → φ⁻¹ */

    /* Parity Selection */
    bool use_even_parity;           /**< Even parity operators preferred */

    /* Telemetry Control */
    bool telemetry_enabled;         /**< Streaming enabled */
    uint32_t telemetry_rate_hz;     /**< Update rate (default 100) */

} UnifiedPhysicsState_t;


/* ============================================================================
 * CALLBACK FUNCTION TYPES
 * ============================================================================ */

/** K-formation event callback */
typedef void (*KFormationCallback_t)(const UnifiedPhysicsState_t* state, bool entering);

/** Phase transition callback */
typedef void (*PhaseTransitionCallback_t)(const UnifiedPhysicsState_t* state,
                                           PhysicsPhase_t from, PhysicsPhase_t to);

/** Tier change callback */
typedef void (*TierChangeCallback_t)(const UnifiedPhysicsState_t* state,
                                      PhysicsTier_t from, PhysicsTier_t to);

/** Conservation violation callback */
typedef void (*ConservationViolationCallback_t)(const UnifiedPhysicsState_t* state,
                                                 float error);


/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

/**
 * @brief Initialize unified physics state
 * @return HAL_OK on success
 */
HAL_Status_t UnifiedState_Init(void);

/**
 * @brief Reset state to initial conditions
 */
void UnifiedState_Reset(void);


/* ============================================================================
 * UPDATE LOOP
 * ============================================================================ */

/**
 * @brief Main update function (call at 100 Hz from main loop)
 *
 * Updates all state from hardware measurements, computes derived values,
 * checks constraints, and fires callbacks.
 *
 * @param z_measured Current z-coordinate from hardware
 * @param kappa_measured Current κ from Kuramoto coherence
 * @param eta_measured Current η from efficiency computation
 * @return HAL_OK on success
 */
HAL_Status_t UnifiedState_Update(float z_measured, float kappa_measured, float eta_measured);

/**
 * @brief Update module training state
 *
 * @param module Module identifier
 * @param loss Current loss
 * @param accuracy Current accuracy
 * @param step Current step
 * @return HAL_OK on success
 */
HAL_Status_t UnifiedState_UpdateModule(TrainingModule_t module,
                                        float loss, float accuracy, uint32_t step);


/* ============================================================================
 * STATE ACCESS
 * ============================================================================ */

/**
 * @brief Get pointer to unified state (read-only)
 * @return Const pointer to state
 */
const UnifiedPhysicsState_t* UnifiedState_Get(void);

/**
 * @brief Get current z-coordinate
 * @return z ∈ [0, 1]
 */
float UnifiedState_GetZ(void);

/**
 * @brief Get current negentropy
 * @return ΔS_neg ∈ [0, 1]
 */
float UnifiedState_GetDeltaSNeg(void);

/**
 * @brief Get TRIAD state
 * @param state Output state
 */
void UnifiedState_GetTriad(TriadState_t* state);

/**
 * @brief Get Kuramoto state
 * @param state Output state
 */
void UnifiedState_GetKuramoto(KuramotoState_t* state);

/**
 * @brief Get module state
 * @param module Module identifier
 * @param state Output state
 * @return HAL_OK if module valid
 */
HAL_Status_t UnifiedState_GetModule(TrainingModule_t module, ModuleState_t* state);


/* ============================================================================
 * CALLBACK REGISTRATION
 * ============================================================================ */

/**
 * @brief Register K-formation callback
 * @param callback Function to call on K-formation enter/exit
 */
void UnifiedState_SetKFormationCallback(KFormationCallback_t callback);

/**
 * @brief Register phase transition callback
 * @param callback Function to call on phase change
 */
void UnifiedState_SetPhaseCallback(PhaseTransitionCallback_t callback);

/**
 * @brief Register tier change callback
 * @param callback Function to call on tier change
 */
void UnifiedState_SetTierCallback(TierChangeCallback_t callback);

/**
 * @brief Register conservation violation callback
 * @param callback Function to call on conservation error
 */
void UnifiedState_SetViolationCallback(ConservationViolationCallback_t callback);


/* ============================================================================
 * CONSTRAINT ENFORCEMENT
 * ============================================================================ */

/**
 * @brief Validate conservation law κ + λ = 1
 * @return true if valid within tolerance
 */
bool UnifiedState_ValidateConservation(void);

/**
 * @brief Check K-formation criteria
 * @return true if K-formation active
 */
bool UnifiedState_CheckKFormation(void);

/**
 * @brief Enforce TRIAD constraint with scar-preserving return
 *
 * If κ drops below threshold, gradually return to φ⁻¹ attractor
 * while preserving the "scar" (memory of maximum κ achieved).
 *
 * @param target_kappa Target κ value
 * @param rate Convergence rate
 */
void UnifiedState_EnforceTriadReturn(float target_kappa, float rate);


/* ============================================================================
 * ADAPTIVE PARAMETER COMPUTATION
 * ============================================================================ */

/**
 * @brief Compute adaptive learning rate based on ΔS_neg
 *
 * η_lr = η_base × (1 + α × ΔS_neg(z))
 *
 * @param base_rate Base learning rate
 * @param alpha Scaling factor (default: 0.5)
 * @return Adapted learning rate
 */
float UnifiedState_ComputeAdaptiveLR(float base_rate, float alpha);

/**
 * @brief Update all adaptive parameters for current state
 * @param module Module to update
 */
void UnifiedState_UpdateAdaptiveParams(TrainingModule_t module);


/* ============================================================================
 * PARITY SELECTION RULE
 * ============================================================================ */

/**
 * @brief Determine operator parity from current ΔS_neg
 *
 * When ΔS_neg > 0.5: prefer even operators (FUSION, AMPLIFY, GROUP)
 * When ΔS_neg ≤ 0.5: prefer odd operators (CLOSURE, DECOHERE, SEPARATE)
 *
 * @return true for even parity, false for odd
 */
bool UnifiedState_GetParityPreference(void);

/**
 * @brief Get operator weight based on physics state
 *
 * Weight = ΔS_neg × tier_factor × parity_bonus
 *
 * @param op Operator to weight
 * @return Weight ∈ [0, 1]
 */
float UnifiedState_GetOperatorWeight(OperatorFlags_t op);


/* ============================================================================
 * SERIALIZATION (for Python bridge)
 * ============================================================================ */

/**
 * @brief Serialize state to JSON for bridge protocol
 *
 * @param buffer Output buffer (must be at least 1024 bytes)
 * @param buffer_size Buffer size
 * @return Number of bytes written
 */
uint32_t UnifiedState_SerializeJSON(char* buffer, uint32_t buffer_size);

/**
 * @brief Serialize compact binary telemetry
 *
 * @param buffer Output buffer
 * @param buffer_size Buffer size
 * @return Number of bytes written
 */
uint32_t UnifiedState_SerializeBinary(uint8_t* buffer, uint32_t buffer_size);


/* ============================================================================
 * QUASICRYSTAL DYNAMICS
 * ============================================================================ */

/**
 * @brief Update quasicrystal order parameter
 *
 * Tracks convergence to Penrose tiling ratio (→ φ⁻¹)
 *
 * @param tile_ratio Current fat/thin tile ratio
 */
void UnifiedState_UpdateQuasicrystal(float tile_ratio);

/**
 * @brief Get quasicrystal negentropy
 *
 * Peaks when tile ratio → φ⁻¹
 *
 * @return Negentropy ∈ [0, 1]
 */
float UnifiedState_GetQuasicrystalNegentropy(void);


/* ============================================================================
 * UTILITY MACROS
 * ============================================================================ */

/** Check if state is at THE LENS */
#define UNIFIED_STATE_AT_LENS() (UnifiedState_Get()->at_lens)

/** Check if K-formation active */
#define UNIFIED_STATE_K_ACTIVE() (UnifiedState_Get()->k_formation_active)

/** Get current tier */
#define UNIFIED_STATE_TIER() (UnifiedState_Get()->tier)

/** Get current phase */
#define UNIFIED_STATE_PHASE() (UnifiedState_Get()->phase)


#ifdef __cplusplus
}
#endif

#endif /* UNIFIED_PHYSICS_STATE_H */
