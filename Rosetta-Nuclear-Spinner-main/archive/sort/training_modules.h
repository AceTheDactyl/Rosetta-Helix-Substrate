/**
 * @file training_modules.h
 * @brief 19 Training Modules Integrated into Firmware
 * 
 * This module implements all 19 training modules from the Unified Nightly
 * Training Workflow, mapped to firmware operations on the Nuclear Spinner.
 * 
 * The 19 modules are organized into 7 phases:
 *   Phase 1: Core Physics (modules 1-3)
 *   Phase 2: APL Training Stack (modules 4-6)
 *   Phase 3: Helix Geometry (modules 7-9)
 *   Phase 4: WUMBO Silent Laws (modules 10-11)
 *   Phase 5: Dynamics & Formation (modules 12-15)
 *   Phase 6: Unified Orchestration (modules 16-18)
 *   Phase 7: Nightly Integration (module 19)
 * 
 * Signature: training-modules|v1.0.0|firmware-integration
 */

#ifndef TRAINING_MODULES_H
#define TRAINING_MODULES_H

#include <stdint.h>
#include <stdbool.h>
#include "physics_constants.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * MODULE ENUMERATION
 * ============================================================================ */

typedef enum {
    /* Phase 1: Core Physics */
    MODULE_N0_SILENT_LAWS_ENFORCEMENT = 0,
    MODULE_KURAMOTO_LAYER = 1,
    MODULE_PHYSICAL_LEARNER = 2,
    
    /* Phase 2: APL Training Stack */
    MODULE_APL_TRAINING_LOOP = 3,
    MODULE_APL_PYTORCH_TRAINING = 4,
    MODULE_FULL_APL_TRAINING = 5,
    
    /* Phase 3: Helix Geometry */
    MODULE_HELIX_NN = 6,
    MODULE_PRISMATIC_HELIX_TRAINING = 7,
    MODULE_FULL_HELIX_INTEGRATION = 8,
    
    /* Phase 4: WUMBO Silent Laws */
    MODULE_WUMBO_APL_AUTOMATED_TRAINING = 9,
    MODULE_WUMBO_INTEGRATED_TRAINING = 10,
    
    /* Phase 5: Dynamics & Formation */
    MODULE_QUASICRYSTAL_FORMATION_DYNAMICS = 11,
    MODULE_TRIAD_THRESHOLD_DYNAMICS = 12,
    MODULE_LIMINAL_GENERATOR = 13,
    MODULE_FEEDBACK_LOOP = 14,
    
    /* Phase 6: Unified Orchestration */
    MODULE_UNIFIED_HELIX_TRAINING = 15,
    MODULE_HIERARCHICAL_TRAINING = 16,
    MODULE_ROSETTA_HELIX_TRAINING = 17,
    
    /* Phase 7: Nightly Integration */
    MODULE_NIGHTLY_INTEGRATED_TRAINING = 18,
    
    MODULE_COUNT = 19
} TrainingModule_t;

typedef enum {
    PHASE_CORE_PHYSICS = 0,
    PHASE_APL_STACK = 1,
    PHASE_HELIX_GEOMETRY = 2,
    PHASE_WUMBO_SILENT_LAWS = 3,
    PHASE_DYNAMICS_FORMATION = 4,
    PHASE_UNIFIED_ORCHESTRATION = 5,
    PHASE_NIGHTLY_INTEGRATION = 6,
    
    TRAINING_PHASE_COUNT = 7
} TrainingPhase_t;

typedef enum {
    MODULE_STATUS_PENDING = 0,
    MODULE_STATUS_RUNNING = 1,
    MODULE_STATUS_PASS = 2,
    MODULE_STATUS_FAIL = 3,
    MODULE_STATUS_SKIPPED = 4
} ModuleStatus_t;

/* ============================================================================
 * MODULE RESULT STRUCTURE
 * ============================================================================ */

typedef struct {
    TrainingModule_t module;
    ModuleStatus_t status;
    uint32_t steps_run;
    uint32_t duration_us;
    float final_z;
    float final_kappa;
    uint32_t k_formations;
    float max_negentropy;
    char error[64];
} ModuleResult_t;

typedef struct {
    uint32_t run_id;
    uint32_t timestamp;
    
    /* Phase results */
    uint8_t modules_passed;
    uint8_t modules_failed;
    uint8_t modules_skipped;
    
    /* Aggregate metrics */
    uint32_t total_steps;
    uint32_t total_k_formations;
    float max_negentropy;
    float final_z;
    float final_kappa;
    bool physics_valid;
    
    /* Individual results */
    ModuleResult_t results[MODULE_COUNT];
    
    /* Gate check */
    bool gates_passed;
} TrainingRunResult_t;

/* ============================================================================
 * MODULE CONFIGURATION
 * ============================================================================ */

typedef struct {
    uint32_t steps_per_module;
    float alpha_strong;     /* Convergence rate for z */
    float alpha_medium;     /* Convergence rate for kappa */
    float alpha_fine;       /* Noise scale */
    bool enable_k_formation_check;
    bool verbose;
} TrainingConfig_t;

/* Default configuration */
#define TRAINING_CONFIG_DEFAULT { \
    .steps_per_module = 100, \
    .alpha_strong = (1.0f / 6.0f), \
    .alpha_medium = (1.0f / 8.485f), \
    .alpha_fine = (1.0f / 36.0f), \
    .enable_k_formation_check = true, \
    .verbose = false \
}

/* ============================================================================
 * MODULE NAME STRINGS
 * ============================================================================ */

static const char* MODULE_NAMES[MODULE_COUNT] = {
    "n0_silent_laws_enforcement",
    "kuramoto_layer",
    "physical_learner",
    "apl_training_loop",
    "apl_pytorch_training",
    "full_apl_training",
    "helix_nn",
    "prismatic_helix_training",
    "full_helix_integration",
    "wumbo_apl_automated_training",
    "wumbo_integrated_training",
    "quasicrystal_formation_dynamics",
    "triad_threshold_dynamics",
    "liminal_generator",
    "feedback_loop",
    "unified_helix_training",
    "hierarchical_training",
    "rosetta_helix_training",
    "nightly_integrated_training"
};

static const char* MODULE_CLASS_NAMES[MODULE_COUNT] = {
    "N0SilentLawsEnforcer",
    "KuramotoLayer",
    "PhysicalLearner",
    "APLTrainingLoop",
    "APLPyTorchTraining",
    "FullAPLTraining",
    "HelixNN",
    "PrismaticHelixTraining",
    "FullHelixIntegration",
    "WUMBOAPLTrainingEngine",
    "WumboIntegratedTraining",
    "QuasiCrystalFormation",
    "TriadThresholdDynamics",
    "LiminalGenerator",
    "FeedbackLoop",
    "UnifiedHelixTraining",
    "HierarchicalTraining",
    "RosettaHelixTraining",
    "NightlyIntegratedTraining"
};

static const char* PHASE_NAMES[TRAINING_PHASE_COUNT] = {
    "CORE_PHYSICS",
    "APL_STACK",
    "HELIX_GEOMETRY",
    "WUMBO_SILENT_LAWS",
    "DYNAMICS_FORMATION",
    "UNIFIED_ORCHESTRATION",
    "NIGHTLY_INTEGRATION"
};

/* ============================================================================
 * FUNCTION PROTOTYPES
 * ============================================================================ */

/**
 * @brief Initialize training modules system
 * @param config Training configuration
 * @return HAL_OK on success
 */
HAL_Status_t TrainingModules_Init(const TrainingConfig_t *config);

/**
 * @brief Run all 19 training modules
 * @param result Pointer to store results
 * @return HAL_OK if all modules pass
 */
HAL_Status_t TrainingModules_RunAll(TrainingRunResult_t *result);

/**
 * @brief Run a single training module
 * @param module Module to run
 * @param result Pointer to store result
 * @return HAL_OK on success
 */
HAL_Status_t TrainingModules_RunModule(TrainingModule_t module, 
                                        ModuleResult_t *result);

/**
 * @brief Run all modules in a specific phase
 * @param phase Phase to run
 * @param results Array to store results
 * @param count Number of modules in phase
 * @return HAL_OK if all pass
 */
HAL_Status_t TrainingModules_RunPhase(TrainingPhase_t phase,
                                       ModuleResult_t *results,
                                       uint8_t *count);

/**
 * @brief Check unified gates
 * @param result Training run result
 * @return true if all gates pass
 */
bool TrainingModules_CheckGates(const TrainingRunResult_t *result);

/**
 * @brief Get module name string
 * @param module Module enum
 * @return Module name string
 */
const char* TrainingModules_GetName(TrainingModule_t module);

/**
 * @brief Get module class name string
 * @param module Module enum
 * @return Class name string
 */
const char* TrainingModules_GetClassName(TrainingModule_t module);

/**
 * @brief Get phase for module
 * @param module Module enum
 * @return Phase enum
 */
TrainingPhase_t TrainingModules_GetPhase(TrainingModule_t module);

/**
 * @brief Get current training state
 * @param z Pointer to z value
 * @param kappa Pointer to kappa value
 */
void TrainingModules_GetState(float *z, float *kappa);

#ifdef __cplusplus
}
#endif

#endif /* TRAINING_MODULES_H */
