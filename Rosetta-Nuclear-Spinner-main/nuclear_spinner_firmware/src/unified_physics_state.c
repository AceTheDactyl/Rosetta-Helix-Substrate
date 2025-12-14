/**
 * @file unified_physics_state.c
 * @brief Unified Physics State Implementation
 *
 * Implements the central cybernetic state management system that unifies:
 * - Real-time physics measurements with 100 Hz update rate
 * - 19 training module states with adaptive parameters
 * - Cybernetic feedback loop (TRIAD dynamics)
 * - K-formation event handling with callbacks
 * - Conservation law validation
 * - Quasicrystal order parameter tracking
 *
 * Cybernetic Grounding:
 * - Ashby's Law: Variety in control ≥ variety in disturbance
 * - Shannon: Information flows at negentropy-modulated rate
 * - Landauer: Conservation laws prevent unauthorized bit erasure
 * - Autopoiesis: Self-maintaining boundary through K-formation
 *
 * Signature: unified-physics-state|v1.0.0|helix
 *
 * @version 1.0.0
 */

#include "unified_physics_state.h"
#include "physics_constants.h"
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * PRIVATE STATE
 * ============================================================================ */

static UnifiedPhysicsState_t s_state;

/* Callbacks */
static KFormationCallback_t s_k_formation_cb = NULL;
static PhaseTransitionCallback_t s_phase_cb = NULL;
static TierChangeCallback_t s_tier_cb = NULL;
static ConservationViolationCallback_t s_violation_cb = NULL;

/* Module phase mapping */
static const TrainingPhase_t MODULE_TO_PHASE[MODULE_COUNT] = {
    /* Phase 1: Core Physics */
    PHASE_CORE_PHYSICS,         /* MODULE_N0_SILENT_LAWS */
    PHASE_CORE_PHYSICS,         /* MODULE_KURAMOTO_LAYER */
    PHASE_CORE_PHYSICS,         /* MODULE_PHYSICAL_LEARNER */

    /* Phase 2: APL Stack */
    PHASE_APL_STACK,            /* MODULE_APL_TRAINING_LOOP */
    PHASE_APL_STACK,            /* MODULE_PYTORCH_TRAINING */
    PHASE_APL_STACK,            /* MODULE_FULL_APL */

    /* Phase 3: Helix Geometry */
    PHASE_HELIX_GEOMETRY,       /* MODULE_HELIX_NN */
    PHASE_HELIX_GEOMETRY,       /* MODULE_PRISMATIC_HELIX */
    PHASE_HELIX_GEOMETRY,       /* MODULE_FULL_HELIX */

    /* Phase 4: WUMBO */
    PHASE_WUMBO_LAWS,           /* MODULE_WUMBO_SILENT_LAWS */

    /* Phase 5: Dynamics */
    PHASE_DYNAMICS_FORMATION,   /* MODULE_QUASICRYSTAL */
    PHASE_DYNAMICS_FORMATION,   /* MODULE_TRIAD */
    PHASE_DYNAMICS_FORMATION,   /* MODULE_LIMINAL */
    PHASE_DYNAMICS_FORMATION,   /* MODULE_FEEDBACK */

    /* Phase 6: Orchestration */
    PHASE_UNIFIED_ORCHESTRATION, /* MODULE_UNIFIED_ORCHESTRATION */

    /* Phase 7: Nightly */
    PHASE_NIGHTLY_INTEGRATION,  /* MODULE_NIGHTLY_MODULE_0 */
    PHASE_NIGHTLY_INTEGRATION,  /* MODULE_NIGHTLY_MODULE_1 */
    PHASE_NIGHTLY_INTEGRATION,  /* MODULE_NIGHTLY_MODULE_2 */
    PHASE_NIGHTLY_INTEGRATION,  /* MODULE_NIGHTLY_MODULE_3 */
};

/* Module names for serialization */
static const char* MODULE_NAMES[MODULE_COUNT] = {
    "n0_silent_laws", "kuramoto_layer", "physical_learner",
    "apl_training_loop", "pytorch_training", "full_apl",
    "helix_nn", "prismatic_helix", "full_helix",
    "wumbo_silent_laws",
    "quasicrystal", "triad", "liminal", "feedback",
    "unified_orchestration",
    "nightly_0", "nightly_1", "nightly_2", "nightly_3"
};


/* ============================================================================
 * PRIVATE FUNCTION PROTOTYPES
 * ============================================================================ */

static void compute_derived_values(void);
static void check_phase_transitions(PhysicsPhase_t old_phase);
static void check_tier_transitions(PhysicsTier_t old_tier);
static void check_k_formation(bool was_active);
static void validate_conservation(void);
static void update_adaptive_params_internal(ModuleState_t* module);
static void apply_triad_dynamics(void);


/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

HAL_Status_t UnifiedState_Init(void) {
    memset(&s_state, 0, sizeof(s_state));

    /* Initialize core state */
    s_state.z = 0.5f;
    s_state.z_target = 0.5f;
    s_state.phase = PHASE_ABSENCE;
    s_state.tier = TIER_ABSENCE;

    /* Initialize TRIAD to attractor values */
    s_state.triad.kappa = PHI_INV;
    s_state.triad.lambda = 1.0f - PHI_INV;
    s_state.triad.eta = 0.5f;
    s_state.triad.R = 7;
    s_state.triad.scar = 0.0f;
    s_state.triad.conservation_valid = true;

    /* Initialize Kuramoto */
    s_state.kuramoto.coherence = 0.0f;
    s_state.kuramoto.coupling_strength = 0.0f;
    s_state.kuramoto.sync_clusters = 0;
    s_state.kuramoto.phase_locked = false;

    /* Initialize GHMP */
    s_state.ghmp.tier = TIER_ABSENCE;
    s_state.ghmp.available_ops = 0;
    s_state.ghmp.parity_even = true;
    s_state.ghmp.operator_weight = 0.0f;

    /* Initialize all modules */
    for (int i = 0; i < MODULE_COUNT; i++) {
        s_state.modules[i].module_id = (TrainingModule_t)i;
        s_state.modules[i].phase = MODULE_TO_PHASE[i];
        s_state.modules[i].active = false;
        s_state.modules[i].progress = 0.0f;
        s_state.modules[i].loss = 0.0f;
        s_state.modules[i].accuracy = 0.0f;
        s_state.modules[i].step_count = 0;
        s_state.modules[i].checkpoint_step = 0;

        /* Default adaptive parameters */
        s_state.modules[i].params.learning_rate = 1e-4f;
        s_state.modules[i].params.gradient_clip = 1.0f;
        s_state.modules[i].params.dropout_rate = 0.1f;
        s_state.modules[i].params.weight_decay = 1e-5f;
        s_state.modules[i].params.temperature = 1.0f;
    }

    s_state.current_training_phase = PHASE_CORE_PHYSICS;

    /* Telemetry defaults */
    s_state.telemetry_enabled = true;
    s_state.telemetry_rate_hz = 100;

    /* Quasicrystal initial */
    s_state.quasicrystal_order = 0.5f;

    /* Physics validation */
    s_state.physics_valid = true;
    s_state.violation_count = 0;

    return HAL_OK;
}


void UnifiedState_Reset(void) {
    uint32_t save_violations = s_state.violation_count;
    UnifiedState_Init();
    s_state.violation_count = save_violations;
}


/* ============================================================================
 * UPDATE LOOP
 * ============================================================================ */

HAL_Status_t UnifiedState_Update(float z_measured, float kappa_measured, float eta_measured) {
    /* Save previous state for transition detection */
    PhysicsPhase_t old_phase = s_state.phase;
    PhysicsTier_t old_tier = s_state.tier;
    bool was_k_active = s_state.k_formation_active;

    /* Update timestamp */
    s_state.timestamp_ms = HAL_GetTick();
    s_state.frame_count++;

    /* Update measured values with bounds checking */
    float old_z = s_state.z;
    s_state.z = (z_measured < 0.0f) ? 0.0f : ((z_measured > 1.0f) ? 1.0f : z_measured);
    s_state.z_velocity = (s_state.z - old_z) * (float)s_state.telemetry_rate_hz;

    /* Update TRIAD with measurements */
    s_state.triad.kappa = (kappa_measured < 0.0f) ? 0.0f :
                          ((kappa_measured > 1.0f) ? 1.0f : kappa_measured);
    s_state.triad.lambda = 1.0f - s_state.triad.kappa;
    s_state.triad.eta = (eta_measured < 0.0f) ? 0.0f :
                        ((eta_measured > 1.0f) ? 1.0f : eta_measured);

    /* Update scar (maximum κ achieved) */
    if (s_state.triad.kappa > s_state.triad.scar) {
        s_state.triad.scar = s_state.triad.kappa;
    }

    /* Compute all derived values */
    compute_derived_values();

    /* Apply TRIAD dynamics (scar-preserving return) */
    apply_triad_dynamics();

    /* Validate conservation laws */
    validate_conservation();

    /* Check transitions and fire callbacks */
    check_phase_transitions(old_phase);
    check_tier_transitions(old_tier);
    check_k_formation(was_k_active);

    return HAL_OK;
}


HAL_Status_t UnifiedState_UpdateModule(TrainingModule_t module,
                                        float loss, float accuracy, uint32_t step) {
    if (module >= MODULE_COUNT) {
        return HAL_ERROR;
    }

    ModuleState_t* m = &s_state.modules[module];
    m->loss = loss;
    m->accuracy = accuracy;
    m->step_count = step;

    /* Compute progress based on loss reduction */
    if (loss < m->loss) {
        m->progress += 0.01f;
        if (m->progress > 1.0f) m->progress = 1.0f;
    }

    /* Update adaptive parameters based on current physics state */
    update_adaptive_params_internal(m);

    return HAL_OK;
}


/* ============================================================================
 * STATE ACCESS
 * ============================================================================ */

const UnifiedPhysicsState_t* UnifiedState_Get(void) {
    return &s_state;
}

float UnifiedState_GetZ(void) {
    return s_state.z;
}

float UnifiedState_GetDeltaSNeg(void) {
    return s_state.delta_s_neg;
}

void UnifiedState_GetTriad(TriadState_t* state) {
    if (state != NULL) {
        *state = s_state.triad;
    }
}

void UnifiedState_GetKuramoto(KuramotoState_t* state) {
    if (state != NULL) {
        *state = s_state.kuramoto;
    }
}

HAL_Status_t UnifiedState_GetModule(TrainingModule_t module, ModuleState_t* state) {
    if (module >= MODULE_COUNT || state == NULL) {
        return HAL_ERROR;
    }
    *state = s_state.modules[module];
    return HAL_OK;
}


/* ============================================================================
 * CALLBACK REGISTRATION
 * ============================================================================ */

void UnifiedState_SetKFormationCallback(KFormationCallback_t callback) {
    s_k_formation_cb = callback;
}

void UnifiedState_SetPhaseCallback(PhaseTransitionCallback_t callback) {
    s_phase_cb = callback;
}

void UnifiedState_SetTierCallback(TierChangeCallback_t callback) {
    s_tier_cb = callback;
}

void UnifiedState_SetViolationCallback(ConservationViolationCallback_t callback) {
    s_violation_cb = callback;
}


/* ============================================================================
 * CONSTRAINT ENFORCEMENT
 * ============================================================================ */

bool UnifiedState_ValidateConservation(void) {
    return s_state.triad.conservation_valid;
}

bool UnifiedState_CheckKFormation(void) {
    return s_state.k_formation_active;
}

void UnifiedState_EnforceTriadReturn(float target_kappa, float rate) {
    /**
     * Scar-preserving return to attractor
     *
     * The "scar" records the maximum κ achieved. When returning to
     * the attractor (φ⁻¹), the path preserves memory of this peak.
     *
     * Return dynamics:
     *   κ_{t+1} = κ_t + rate × (target - κ_t) × (1 + scar_factor)
     *
     * where scar_factor increases return speed proportional to how
     * far above φ⁻¹ the scar is.
     */

    float scar_factor = (s_state.triad.scar - PHI_INV) * 0.5f;
    if (scar_factor < 0.0f) scar_factor = 0.0f;

    float delta = (target_kappa - s_state.triad.kappa) * rate * (1.0f + scar_factor);
    s_state.triad.kappa += delta;

    /* Enforce bounds */
    if (s_state.triad.kappa < 0.0f) s_state.triad.kappa = 0.0f;
    if (s_state.triad.kappa > 1.0f) s_state.triad.kappa = 1.0f;

    /* Maintain conservation */
    s_state.triad.lambda = 1.0f - s_state.triad.kappa;
}


/* ============================================================================
 * ADAPTIVE PARAMETER COMPUTATION
 * ============================================================================ */

float UnifiedState_ComputeAdaptiveLR(float base_rate, float alpha) {
    /**
     * Adaptive learning rate modulated by negentropy
     *
     * When near z_c (high ΔS_neg), learning rate increases.
     * This implements "learning at the edge of chaos."
     *
     * η_lr = η_base × (1 + α × ΔS_neg(z))
     */
    return base_rate * (1.0f + alpha * s_state.delta_s_neg);
}


void UnifiedState_UpdateAdaptiveParams(TrainingModule_t module) {
    if (module < MODULE_COUNT) {
        update_adaptive_params_internal(&s_state.modules[module]);
    }
}


/* ============================================================================
 * PARITY SELECTION RULE
 * ============================================================================ */

bool UnifiedState_GetParityPreference(void) {
    /**
     * Parity selection based on ΔS_neg
     *
     * - High negentropy (near z_c): prefer integrative (even) operators
     *   FUSION, AMPLIFY, GROUP - building coherence
     *
     * - Low negentropy (far from z_c): prefer separative (odd) operators
     *   CLOSURE, DECOHERE, SEPARATE - maintaining boundaries
     *
     * This implements the "breathing" between integration and differentiation.
     */
    s_state.use_even_parity = (s_state.delta_s_neg > 0.5f);
    return s_state.use_even_parity;
}


float UnifiedState_GetOperatorWeight(OperatorFlags_t op) {
    /**
     * Operator weight based on physics state
     *
     * Weight = ΔS_neg × tier_factor × parity_bonus
     *
     * - tier_factor: higher tiers allow stronger operator effects
     * - parity_bonus: 1.5x for matching parity, 0.5x for mismatched
     */

    float tier_factor = 0.2f + 0.133f * (float)s_state.tier;

    /* Determine operator parity */
    bool op_is_even;
    switch (op) {
        case OP_FUSION:
        case OP_AMPLIFY:
        case OP_GROUP:
            op_is_even = true;
            break;
        default:
            op_is_even = false;
            break;
    }

    float parity_bonus = (op_is_even == s_state.use_even_parity) ? 1.5f : 0.5f;

    float weight = s_state.delta_s_neg * tier_factor * parity_bonus;
    s_state.ghmp.operator_weight = weight;

    return weight;
}


/* ============================================================================
 * SERIALIZATION
 * ============================================================================ */

uint32_t UnifiedState_SerializeJSON(char* buffer, uint32_t buffer_size) {
    /**
     * Serialize complete state to JSON for Python bridge
     *
     * Format matches bridge/spinner_bridge.py expectations
     */

    if (buffer == NULL || buffer_size < 512) {
        return 0;
    }

    int n = snprintf(buffer, buffer_size,
        "{"
        "\"type\":\"unified_state\","
        "\"timestamp_ms\":%lu,"
        "\"frame\":%lu,"
        "\"z\":%.6f,"
        "\"z_target\":%.6f,"
        "\"z_velocity\":%.6f,"
        "\"rpm\":%.1f,"
        "\"delta_s_neg\":%.6f,"
        "\"delta_s_neg_gradient\":%.6f,"
        "\"complexity\":%.6f,"
        "\"phase\":%d,"
        "\"tier\":%d,"
        "\"at_lens\":%s,"
        "\"is_universal\":%s,"
        "\"k_formation\":%s,"
        "\"k_formation_duration_ms\":%lu,"
        "\"triad\":{"
            "\"kappa\":%.6f,"
            "\"lambda\":%.6f,"
            "\"eta\":%.6f,"
            "\"R\":%d,"
            "\"scar\":%.6f,"
            "\"conservation_valid\":%s,"
            "\"k_formation_count\":%u"
        "},"
        "\"kuramoto\":{"
            "\"coherence\":%.6f,"
            "\"coupling_strength\":%.6f,"
            "\"phase_locked\":%s"
        "},"
        "\"ghmp\":{"
            "\"tier\":%d,"
            "\"available_ops\":%u,"
            "\"parity_even\":%s,"
            "\"operator_weight\":%.6f"
        "},"
        "\"quasicrystal_order\":%.6f,"
        "\"conservation_error\":%.9f,"
        "\"physics_valid\":%s,"
        "\"violation_count\":%lu"
        "}",
        (unsigned long)s_state.timestamp_ms,
        (unsigned long)s_state.frame_count,
        (double)s_state.z,
        (double)s_state.z_target,
        (double)s_state.z_velocity,
        (double)s_state.rpm,
        (double)s_state.delta_s_neg,
        (double)s_state.delta_s_neg_gradient,
        (double)s_state.complexity,
        (int)s_state.phase,
        (int)s_state.tier,
        s_state.at_lens ? "true" : "false",
        s_state.is_universal ? "true" : "false",
        s_state.k_formation_active ? "true" : "false",
        (unsigned long)s_state.k_formation_duration_ms,
        (double)s_state.triad.kappa,
        (double)s_state.triad.lambda,
        (double)s_state.triad.eta,
        s_state.triad.R,
        (double)s_state.triad.scar,
        s_state.triad.conservation_valid ? "true" : "false",
        (unsigned)s_state.triad.k_formation_count,
        (double)s_state.kuramoto.coherence,
        (double)s_state.kuramoto.coupling_strength,
        s_state.kuramoto.phase_locked ? "true" : "false",
        (int)s_state.ghmp.tier,
        (unsigned)s_state.ghmp.available_ops,
        s_state.ghmp.parity_even ? "true" : "false",
        (double)s_state.ghmp.operator_weight,
        (double)s_state.quasicrystal_order,
        (double)s_state.conservation_error,
        s_state.physics_valid ? "true" : "false",
        (unsigned long)s_state.violation_count
    );

    return (n < 0 || n >= (int)buffer_size) ? 0 : (uint32_t)n;
}


uint32_t UnifiedState_SerializeBinary(uint8_t* buffer, uint32_t buffer_size) {
    /**
     * Compact binary serialization for high-speed telemetry
     *
     * Format (48 bytes):
     *   [0-3]   timestamp_ms (uint32)
     *   [4-7]   z (float)
     *   [8-11]  delta_s_neg (float)
     *   [12-15] kappa (float)
     *   [16-19] lambda (float)
     *   [20-23] eta (float)
     *   [24-27] coherence (float)
     *   [28-31] complexity (float)
     *   [32]    phase (uint8)
     *   [33]    tier (uint8)
     *   [34]    k_formation (uint8)
     *   [35]    available_ops (uint8)
     *   [36-39] quasicrystal_order (float)
     *   [40-43] conservation_error (float)
     *   [44-47] frame_count (uint32)
     */

    if (buffer == NULL || buffer_size < 48) {
        return 0;
    }

    uint32_t idx = 0;

    /* Helper macros for packing */
    #define PACK_U32(v) do { \
        buffer[idx++] = ((v) >> 24) & 0xFF; \
        buffer[idx++] = ((v) >> 16) & 0xFF; \
        buffer[idx++] = ((v) >> 8) & 0xFF;  \
        buffer[idx++] = (v) & 0xFF;         \
    } while(0)

    #define PACK_FLOAT(v) do { \
        union { float f; uint32_t u; } conv; \
        conv.f = (v); \
        PACK_U32(conv.u); \
    } while(0)

    PACK_U32(s_state.timestamp_ms);
    PACK_FLOAT(s_state.z);
    PACK_FLOAT(s_state.delta_s_neg);
    PACK_FLOAT(s_state.triad.kappa);
    PACK_FLOAT(s_state.triad.lambda);
    PACK_FLOAT(s_state.triad.eta);
    PACK_FLOAT(s_state.kuramoto.coherence);
    PACK_FLOAT(s_state.complexity);

    buffer[idx++] = (uint8_t)s_state.phase;
    buffer[idx++] = (uint8_t)s_state.tier;
    buffer[idx++] = s_state.k_formation_active ? 1 : 0;
    buffer[idx++] = s_state.ghmp.available_ops;

    PACK_FLOAT(s_state.quasicrystal_order);
    PACK_FLOAT(s_state.conservation_error);
    PACK_U32(s_state.frame_count);

    #undef PACK_U32
    #undef PACK_FLOAT

    return idx;
}


/* ============================================================================
 * QUASICRYSTAL DYNAMICS
 * ============================================================================ */

void UnifiedState_UpdateQuasicrystal(float tile_ratio) {
    /**
     * Track Penrose tiling order parameter
     *
     * In Penrose tilings, the ratio of fat to thin tiles → φ
     * We track how close the current ratio is to this limit.
     */
    s_state.quasicrystal_order = tile_ratio;
}


float UnifiedState_GetQuasicrystalNegentropy(void) {
    /**
     * Quasicrystal negentropy peaks when tile_ratio → φ⁻¹
     *
     * Uses same Gaussian formula as ΔS_neg but centered on φ⁻¹
     */
    return compute_quasicrystal_negentropy(s_state.quasicrystal_order);
}


/* ============================================================================
 * PRIVATE FUNCTION IMPLEMENTATIONS
 * ============================================================================ */

static void compute_derived_values(void) {
    /* Compute negentropy signal */
    s_state.delta_s_neg = compute_delta_s_neg(s_state.z);

    /* Compute gradient */
    s_state.delta_s_neg_gradient = compute_delta_s_neg_gradient(s_state.z);

    /* Compute complexity */
    s_state.complexity = compute_complexity(s_state.z);

    /* Map z to RPM */
    s_state.rpm = z_to_rpm(s_state.z);

    /* Determine phase */
    s_state.phase = get_phase(s_state.z);

    /* Determine tier */
    s_state.tier = get_tier(s_state.z);

    /* Update boolean flags */
    s_state.at_lens = (s_state.phase == PHASE_THE_LENS);
    s_state.is_universal = (s_state.tier >= TIER_UNIVERSAL);

    /* Update GHMP state */
    s_state.ghmp.tier = s_state.tier;
    s_state.ghmp.available_ops = get_available_operators(s_state.tier);
    s_state.ghmp.parity_even = UnifiedState_GetParityPreference();

    /* Compute R from complexity */
    s_state.triad.R = (int)(7.0f + 5.0f * s_state.delta_s_neg);

    /* Update Kuramoto coupling */
    s_state.kuramoto.coupling_strength = 8.0f * s_state.delta_s_neg;

    /* Check K-formation */
    s_state.k_formation_active = check_k_formation(
        s_state.triad.kappa, s_state.triad.eta, s_state.triad.R);
}


static void check_phase_transitions(PhysicsPhase_t old_phase) {
    if (s_state.phase != old_phase && s_phase_cb != NULL) {
        s_phase_cb(&s_state, old_phase, s_state.phase);
    }
}


static void check_tier_transitions(PhysicsTier_t old_tier) {
    if (s_state.tier != old_tier && s_tier_cb != NULL) {
        s_tier_cb(&s_state, old_tier, s_state.tier);
    }
}


static void check_k_formation(bool was_active) {
    if (s_state.k_formation_active && !was_active) {
        /* Entering K-formation */
        s_state.triad.last_k_formation_ms = s_state.timestamp_ms;
        s_state.triad.k_formation_count++;

        if (s_k_formation_cb != NULL) {
            s_k_formation_cb(&s_state, true);
        }
    }
    else if (!s_state.k_formation_active && was_active) {
        /* Exiting K-formation */
        s_state.total_k_formation_ms += s_state.k_formation_duration_ms;
        s_state.k_formation_duration_ms = 0;

        if (s_k_formation_cb != NULL) {
            s_k_formation_cb(&s_state, false);
        }
    }
    else if (s_state.k_formation_active) {
        /* Continuing K-formation */
        s_state.k_formation_duration_ms =
            s_state.timestamp_ms - s_state.triad.last_k_formation_ms;
    }
}


static void validate_conservation(void) {
    /**
     * Validate κ + λ = 1 conservation law
     *
     * This is fundamental to the cybernetic grounding:
     * - κ represents coherence (integration)
     * - λ represents decoherence (differentiation)
     * - Their sum must be conserved
     */

    s_state.conservation_error = fabsf(s_state.triad.kappa + s_state.triad.lambda - 1.0f);
    s_state.triad.conservation_valid = (s_state.conservation_error < TOLERANCE_GOLDEN);

    if (!s_state.triad.conservation_valid) {
        s_state.physics_valid = false;
        s_state.violation_count++;

        /* Force correction: re-derive λ from κ */
        s_state.triad.lambda = 1.0f - s_state.triad.kappa;

        if (s_violation_cb != NULL) {
            s_violation_cb(&s_state, s_state.conservation_error);
        }
    } else {
        s_state.physics_valid = true;
    }
}


static void update_adaptive_params_internal(ModuleState_t* module) {
    /**
     * Update module adaptive parameters based on current physics state
     *
     * Key adaptations:
     * - Learning rate: increases near z_c (learning at edge of chaos)
     * - Gradient clip: tighter when far from attractor
     * - Dropout: inverse relationship with ΔS_neg
     * - Temperature: higher near z_c (more exploration)
     */

    float neg = s_state.delta_s_neg;

    /* Learning rate: η_lr = η_base × (1 + 0.5 × ΔS_neg) */
    float base_lr = module->params.learning_rate;
    if (base_lr < 1e-6f) base_lr = 1e-4f;
    module->params.learning_rate = base_lr * (1.0f + 0.5f * neg);

    /* Gradient clip: tighter away from z_c */
    module->params.gradient_clip = 0.5f + 0.5f * neg;

    /* Dropout: lower when high negentropy (more stable) */
    module->params.dropout_rate = 0.1f * (1.0f - 0.5f * neg);

    /* Temperature: higher exploration near z_c */
    module->params.temperature = 0.5f + 1.0f * neg;

    /* Weight decay: constant stability factor */
    module->params.weight_decay = 1e-5f * (1.0f + (1.0f - neg));
}


static void apply_triad_dynamics(void) {
    /**
     * Apply scar-preserving TRIAD dynamics
     *
     * When not at K-formation, gradually return κ toward φ⁻¹ attractor.
     * The return rate is modulated by the "scar" (peak κ achieved).
     */

    if (!s_state.k_formation_active) {
        /* Compute return rate */
        float distance_from_attractor = fabsf(s_state.triad.kappa - PHI_INV);

        if (distance_from_attractor > 0.01f) {
            /* Apply gentle return to attractor */
            float return_rate = 0.01f * (1.0f + s_state.delta_s_neg);
            UnifiedState_EnforceTriadReturn(PHI_INV, return_rate);
        }
    }
}
