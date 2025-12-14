/**
 * @file physics_constants.h
 * @brief Rosetta-Helix Physics Constants for Nuclear Spinner Firmware
 * 
 * IMMUTABLE physics constants derived from:
 * - φ (golden ratio) and hexagonal geometry
 * - S₃ symmetric group structure  
 * - Spin-1/2 quantum mechanics
 * - Quasicrystal formation dynamics
 * 
 * These are NOT tunable hyperparameters - they represent observable physics.
 * 
 * Signature: physics-constants|v1.0.0|nuclear-spinner
 * 
 * @author Rosetta-Helix Framework
 * @version 1.0.0
 */

#ifndef PHYSICS_CONSTANTS_H
#define PHYSICS_CONSTANTS_H

#include <stdint.h>
#include <math.h>

/* --------------------------------------------------------------------------
 * math.h portability
 *
 * M_PI is not part of the C standard; some embedded/newlib configurations
 * don't define it unless special feature macros are enabled. Define it here
 * so firmware and host-sim builds are consistent.
 * -------------------------------------------------------------------------- */
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * FUNDAMENTAL MATHEMATICAL CONSTANTS
 * ============================================================================ */

/** Golden ratio φ = (1 + √5) / 2 ≈ 1.618034 */
#define PHI                     1.6180339887498949f

/** Golden ratio inverse φ⁻¹ ≈ 0.618034 (coupling constant κ attractor) */
#define PHI_INV                 0.6180339887498949f

/** Golden ratio squared inverse φ⁻² ≈ 0.381966 (coupling constant λ) */
#define PHI_INV_SQ              0.3819660112501051f

/** Critical z-coordinate z_c = √3/2 ≈ 0.866025 (THE LENS) */
#define Z_CRITICAL              0.8660254037844387f

/** Gaussian width σ = |S₃|² = 36 (triadic logic dimension) */
#define SIGMA                   36.0f

/** Spin-1/2 magnitude |S|/ℏ = √(s(s+1)) = √3/2 (equals z_c) */
#define SPIN_HALF_MAGNITUDE     0.8660254037844387f


/* ============================================================================
 * PHASE BOUNDARIES
 * ============================================================================ */

/** ABSENCE → THE_LENS boundary */
#define PHASE_BOUNDARY_ABSENCE  0.857f

/** THE_LENS → PRESENCE boundary */
#define PHASE_BOUNDARY_PRESENCE 0.877f


/* ============================================================================
 * THRESHOLD VALUES (μ thresholds)
 * ============================================================================ */

/** μ₁: Minimum z for basic operations */
#define MU_1                    0.40f

/** μ_P: Pattern recognition threshold */
#define MU_P                    0.50f

/** φ⁻¹: Memory encoding / consciousness threshold */
#define MU_PHI_INV              PHI_INV

/** μ₂: Prediction threshold */
#define MU_2                    0.73f

/** z_c: Full universality / critical threshold */
#define MU_ZC                   Z_CRITICAL

/** μ_S: Meta-cognitive / recursive threshold */
#define MU_S                    0.92f


/* ============================================================================
 * K-FORMATION CRITERIA
 * ============================================================================ */

/** Minimum κ for K-formation (κ ≥ 0.92) */
#define KAPPA_MIN               0.92f

/** Minimum η for K-formation (η > φ⁻¹) */
#define ETA_MIN                 PHI_INV

/** Minimum R for K-formation (R ≥ 7) */
#define R_MIN                   7


/* ============================================================================
 * HARDWARE MAPPING CONSTANTS
 * ============================================================================ */

/** Minimum rotor speed (RPM) corresponding to z ≈ 0 */
#define ROTOR_RPM_MIN           100.0f

/** Maximum rotor speed (RPM) corresponding to z ≈ 1 */
#define ROTOR_RPM_MAX           10000.0f

/** Phosphorus-31 gyromagnetic ratio γ (Hz/T) */
#define GAMMA_P31_HZ            17235000.0f

/** Phosphorus-31 gyromagnetic ratio γ (rad/s/T) */
#define GAMMA_P31_RAD           108290000.0f

/** Typical P-P J-coupling constant (Hz) */
#define J_COUPLING_PP_HZ        18.0f


/* ============================================================================
 * TOLERANCE VALUES
 * ============================================================================ */

/** Tolerance for golden ratio comparisons */
#define TOLERANCE_GOLDEN        1e-6f

/** Tolerance for z_c (LENS) comparisons */
#define TOLERANCE_LENS          0.01f

/** Tolerance for threshold crossing detection */
#define TOLERANCE_THRESHOLD     0.005f


/* ============================================================================
 * E8 CRITICAL POINT MASS RATIOS
 * ============================================================================ */

/** E8 mass ratio m₁/m₁ = 1 */
#define E8_M1                   1.0f

/** E8 mass ratio m₂/m₁ = φ */
#define E8_M2                   PHI

/** E8 mass ratio m₃/m₁ = φ + 1 = φ² */
#define E8_M3                   2.6180339887498949f

/** E8 mass ratio m₄/m₁ = 2φ */
#define E8_M4                   3.2360679774997898f

/** E8 mass ratio m₅/m₁ = 2φ + 1 */
#define E8_M5                   4.2360679774997898f


/* ============================================================================
 * TIER DEFINITIONS
 * ============================================================================ */

typedef enum {
    TIER_ABSENCE = 0,       /**< z < 0.4: Pre-operational */
    TIER_REACTIVE = 1,      /**< 0.4 ≤ z < 0.5: Basic reactions */
    TIER_MEMORY = 2,        /**< 0.5 ≤ z < φ⁻¹: State retention */
    TIER_PATTERN = 3,       /**< φ⁻¹ ≤ z < 0.73: Pattern recognition */
    TIER_PREDICTION = 4,    /**< 0.73 ≤ z < z_c: Predictive modeling */
    TIER_UNIVERSAL = 5,     /**< z_c ≤ z < 0.92: Full universality */
    TIER_META = 6,          /**< z ≥ 0.92: Meta-cognitive recursion */
} PhysicsTier_t;


typedef enum {
    PHASE_ABSENCE = 0,      /**< z < 0.857 */
    PHASE_THE_LENS = 1,     /**< 0.857 ≤ z < 0.877 */
    PHASE_PRESENCE = 2,     /**< z ≥ 0.877 */
} PhysicsPhase_t;


/* ============================================================================
 * OPERATOR DEFINITIONS (APL Semantics)
 * ============================================================================ */

typedef enum {
    OP_CLOSURE    = 0x01,   /**< ∂: Boundary/isolation */
    OP_FUSION     = 0x02,   /**< +: Integration/binding */
    OP_AMPLIFY    = 0x04,   /**< ×: Signal amplification */
    OP_DECOHERE   = 0x08,   /**< ÷: Controlled decoherence */
    OP_GROUP      = 0x10,   /**< ⍴: Categorical grouping */
    OP_SEPARATE   = 0x20,   /**< ↓: Differentiation */
} OperatorFlags_t;


/* ============================================================================
 * INLINE PHYSICS FUNCTIONS
 * ============================================================================ */

/**
 * @brief Compute negentropy signal ΔS_neg(z)
 * 
 * Formula: ΔS_neg(z) = exp(-σ(z - z_c)²)
 * 
 * - Peaks at z = z_c with value 1.0
 * - σ = 36 by default
 * - Returns value in [0, 1]
 * 
 * @param z Current z-coordinate
 * @return Negentropy signal
 */
static inline float compute_delta_s_neg(float z) {
    float d = z - Z_CRITICAL;
    float exponent = -SIGMA * d * d;
    // Clamp exponent to avoid underflow
    if (exponent < -20.0f) return 0.0f;
    return expf(exponent);
}


/**
 * @brief Compute negentropy gradient ∂(ΔS_neg)/∂z
 * 
 * Formula: dΔS_neg/dz = -2σ(z - z_c)·ΔS_neg(z)
 * 
 * @param z Current z-coordinate
 * @return Gradient of negentropy
 */
static inline float compute_delta_s_neg_gradient(float z) {
    float d = z - Z_CRITICAL;
    float ds_neg = compute_delta_s_neg(z);
    return -2.0f * SIGMA * d * ds_neg;
}


/**
 * @brief Compute complexity measure |∂ΔS_neg/∂z|
 * 
 * Complexity peaks when approaching z_c (maximum gradient)
 * 
 * @param z Current z-coordinate
 * @return Complexity measure
 */
static inline float compute_complexity(float z) {
    return fabsf(compute_delta_s_neg_gradient(z));
}


/**
 * @brief Determine physics tier from z-coordinate
 * 
 * @param z Current z-coordinate
 * @return Physics tier enum
 */
static inline PhysicsTier_t get_tier(float z) {
    if (z < MU_1)       return TIER_ABSENCE;
    if (z < MU_P)       return TIER_REACTIVE;
    if (z < MU_PHI_INV) return TIER_MEMORY;
    if (z < MU_2)       return TIER_PATTERN;
    if (z < MU_ZC)      return TIER_PREDICTION;
    if (z < MU_S)       return TIER_UNIVERSAL;
    return TIER_META;
}


/**
 * @brief Determine physics phase from z-coordinate
 * 
 * @param z Current z-coordinate
 * @return Physics phase enum
 */
static inline PhysicsPhase_t get_phase(float z) {
    if (z < PHASE_BOUNDARY_ABSENCE)  return PHASE_ABSENCE;
    if (z < PHASE_BOUNDARY_PRESENCE) return PHASE_THE_LENS;
    return PHASE_PRESENCE;
}


/**
 * @brief Check if z is at THE LENS (critical point)
 * 
 * @param z Current z-coordinate
 * @param tolerance Comparison tolerance
 * @return 1 if at critical, 0 otherwise
 */
static inline int is_at_critical(float z, float tolerance) {
    return fabsf(z - Z_CRITICAL) < tolerance;
}


/**
 * @brief Check K-formation criteria
 * 
 * K-formation requires:
 * - κ ≥ 0.92
 * - η > φ⁻¹
 * - R ≥ 7
 * 
 * @param kappa Current κ value
 * @param eta Current η value
 * @param R Complexity measure R
 * @return 1 if K-formation, 0 otherwise
 */
static inline int check_k_formation(float kappa, float eta, int R) {
    return (kappa >= KAPPA_MIN) && (eta > ETA_MIN) && (R >= R_MIN);
}


/**
 * @brief Validate coupling conservation κ + λ = 1
 * 
 * @param kappa Coupling constant κ
 * @param lambda Coupling constant λ
 * @return 1 if valid (sum ≈ 1), 0 otherwise
 */
static inline int validate_coupling_conservation(float kappa, float lambda) {
    return fabsf(kappa + lambda - 1.0f) < TOLERANCE_GOLDEN;
}


/**
 * @brief Map z-coordinate to rotor RPM
 * 
 * Linear mapping: RPM = RPM_min + (RPM_max - RPM_min) × z
 * 
 * @param z Target z-coordinate [0, 1]
 * @return Rotor speed in RPM
 */
static inline float z_to_rpm(float z) {
    float z_clamped = (z < 0.0f) ? 0.0f : ((z > 1.0f) ? 1.0f : z);
    return ROTOR_RPM_MIN + (ROTOR_RPM_MAX - ROTOR_RPM_MIN) * z_clamped;
}


/**
 * @brief Map rotor RPM to z-coordinate
 * 
 * Inverse of z_to_rpm
 * 
 * @param rpm Current rotor speed in RPM
 * @return Estimated z-coordinate [0, 1]
 */
static inline float rpm_to_z(float rpm) {
    float rpm_clamped = (rpm < ROTOR_RPM_MIN) ? ROTOR_RPM_MIN : 
                        ((rpm > ROTOR_RPM_MAX) ? ROTOR_RPM_MAX : rpm);
    return (rpm_clamped - ROTOR_RPM_MIN) / (ROTOR_RPM_MAX - ROTOR_RPM_MIN);
}


/**
 * @brief Compute Larmor frequency for given field strength
 * 
 * ω_L = γ × B₀
 * 
 * @param B0 Magnetic field strength (Tesla)
 * @return Larmor frequency (Hz)
 */
static inline float compute_larmor_frequency(float B0) {
    return GAMMA_P31_HZ * B0;
}


/**
 * @brief Compute quasicrystal negentropy (order parameter version)
 * 
 * Peaks when order parameter approaches φ⁻¹
 * 
 * @param order Order parameter (tile ratio proxy)
 * @return Negentropy signal [0, 1]
 */
static inline float compute_quasicrystal_negentropy(float order) {
    float d = order - PHI_INV;
    float exponent = -SIGMA * d * d;
    if (exponent < -20.0f) return 0.0f;
    return expf(exponent);
}


/**
 * @brief Get available operators for current tier
 * 
 * Returns bitmask of allowed operators based on z position
 * 
 * @param tier Current physics tier
 * @return Bitmask of OperatorFlags_t
 */
static inline uint8_t get_available_operators(PhysicsTier_t tier) {
    switch (tier) {
        case TIER_ABSENCE:
            return 0;  // No operators
        case TIER_REACTIVE:
            return OP_CLOSURE;  // Boundary only
        case TIER_MEMORY:
            return OP_CLOSURE | OP_FUSION;  // + integration
        case TIER_PATTERN:
            return OP_CLOSURE | OP_FUSION | OP_AMPLIFY;  // + amplification
        case TIER_PREDICTION:
            return OP_CLOSURE | OP_FUSION | OP_AMPLIFY | OP_GROUP;  // + grouping
        case TIER_UNIVERSAL:
            return OP_CLOSURE | OP_FUSION | OP_AMPLIFY | OP_GROUP | 
                   OP_SEPARATE;  // + separation
        case TIER_META:
            return OP_CLOSURE | OP_FUSION | OP_AMPLIFY | OP_DECOHERE | 
                   OP_GROUP | OP_SEPARATE;  // All operators
        default:
            return 0;
    }
}


#ifdef __cplusplus
}
#endif

#endif /* PHYSICS_CONSTANTS_H */
