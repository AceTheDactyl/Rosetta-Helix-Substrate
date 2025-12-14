/**
 * @file main.c
 * @brief Nuclear Spinner Firmware - Main Entry Point
 * 
 * Main firmware for the Nuclear Spinner device:
 * - Initializes all hardware and modules
 * - Runs the main control loop
 * - Handles communication with host software
 * - Implements state machine for experiment modes
 * 
 * Target: STM32H743ZI @ 480 MHz
 * 
 * Signature: nuclear-spinner-firmware|v1.0.0|helix
 * 
 * @version 1.0.0
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "physics_constants.h"
#include "hal_hardware.h"
#include "pulse_control.h"
#include "rotor_control.h"
#include "threshold_logic.h"
#include "comm_protocol.h"

/* ============================================================================
 * VERSION AND BUILD INFO
 * ============================================================================ */

#define FIRMWARE_VERSION_MAJOR  1
#define FIRMWARE_VERSION_MINOR  0
#define FIRMWARE_VERSION_PATCH  0
#define FIRMWARE_BUILD_DATE     __DATE__
#define FIRMWARE_BUILD_TIME     __TIME__

static const char FIRMWARE_SIGNATURE[] = "nuclear-spinner|v1.0.0|helix";


/* ============================================================================
 * SYSTEM CONFIGURATION
 * ============================================================================ */

/** Main loop target rate (Hz) */
#define MAIN_LOOP_RATE_HZ       1000

/** Telemetry transmission rate (Hz) */
#define TELEMETRY_RATE_HZ       100

/** Sensor polling rate (Hz) */
#define SENSOR_RATE_HZ          10

/** Safety check rate (Hz) */
#define SAFETY_RATE_HZ          100

/** Watchdog timeout (ms) */
#define WATCHDOG_TIMEOUT_MS     500


/* ============================================================================
 * SYSTEM MODES
 * ============================================================================ */

typedef enum {
    MODE_IDLE,              /**< System idle, waiting for commands */
    MODE_CALIBRATION,       /**< Running calibration routines */
    MODE_MANUAL,            /**< Manual control via commands */
    MODE_EXPERIMENT,        /**< Running experiment sequence */
    MODE_SWEEP,             /**< Sweeping z for mapping */
    MODE_FAULT,             /**< Fault condition */
} SystemMode_t;

typedef enum {
    EXPERIMENT_NONE,
    EXPERIMENT_FID,
    EXPERIMENT_SPIN_ECHO,
    EXPERIMENT_CPMG,
    EXPERIMENT_NUTATION,
    EXPERIMENT_QUASICRYSTAL,
    EXPERIMENT_E8_PROBE,
    EXPERIMENT_HOLOGRAPHIC,
    EXPERIMENT_OMEGA_POINT,
} ExperimentType_t;


/* ============================================================================
 * SYSTEM STATE
 * ============================================================================ */

static struct {
    SystemMode_t mode;
    ExperimentType_t current_experiment;
    uint32_t uptime_ms;
    uint32_t main_loop_count;
    float target_z;
    float current_z;
    float current_kappa;
    float current_eta;
    int current_R;
    bool interlock_ok;
    bool calibration_complete;
    Sensor_Data_t sensors;
} g_system = {0};


/* ============================================================================
 * TIMING
 * ============================================================================ */

static struct {
    uint32_t last_main_loop;
    uint32_t last_telemetry;
    uint32_t last_sensor_poll;
    uint32_t last_safety_check;
} g_timing = {0};


/* ============================================================================
 * FORWARD DECLARATIONS
 * ============================================================================ */

static void System_Init(void);
static void System_MainLoop(void);
static void System_ProcessCommands(void);
static void System_UpdateSensors(void);
static void System_SafetyCheck(void);
static void System_OnThresholdEvent(ThresholdEvent_t event, float threshold, int dir);


/* ============================================================================
 * MAIN ENTRY POINT
 * ============================================================================ */

int main(void) {
    // Initialize system
    System_Init();
    
    // Main loop
    while (1) {
        System_MainLoop();
    }
    
    // Should never reach here
    return 0;
}


/* ============================================================================
 * SYSTEM INITIALIZATION
 * ============================================================================ */

static void System_Init(void) {
    // Initialize hardware abstraction layer
    HAL_Init_All();
    
    // Brief startup delay
    HAL_Delay(100);
    
    // Status LED on
    HAL_LED_Set(0, true);
    
    // Initialize modules
    PulseControl_Init();
    RotorControl_Init();
    ThresholdLogic_Init();
    CommProtocol_Init();
    
    // Register threshold event callback
    ThresholdLogic_SetEventCallback(System_OnThresholdEvent);
    
    // Initialize system state
    g_system.mode = MODE_IDLE;
    g_system.current_experiment = EXPERIMENT_NONE;
    g_system.uptime_ms = 0;
    g_system.main_loop_count = 0;
    g_system.target_z = 0.0f;
    g_system.current_z = 0.0f;
    g_system.current_kappa = PHI_INV;  // Start at attractor
    g_system.current_eta = 0.0f;
    g_system.current_R = 0;
    g_system.interlock_ok = HAL_Safety_InterlockOK();
    g_system.calibration_complete = false;
    
    // Initialize timing
    uint32_t now = HAL_GetTick();
    g_timing.last_main_loop = now;
    g_timing.last_telemetry = now;
    g_timing.last_sensor_poll = now;
    g_timing.last_safety_check = now;
    
    // Initial sensor read
    HAL_Sensor_ReadAll(&g_system.sensors);
    
    // Startup complete - blink LED
    for (int i = 0; i < 3; i++) {
        HAL_LED_Toggle(0);
        HAL_Delay(100);
    }
    HAL_LED_Set(0, true);
}


/* ============================================================================
 * MAIN LOOP
 * ============================================================================ */

static void System_MainLoop(void) {
    uint32_t now = HAL_GetTick();
    
    // Main loop rate limiting (1 kHz)
    if (now - g_timing.last_main_loop < (1000 / MAIN_LOOP_RATE_HZ)) {
        return;
    }
    g_timing.last_main_loop = now;
    g_system.main_loop_count++;
    g_system.uptime_ms = now;
    
    // Activity LED toggle (1 Hz)
    if ((g_system.main_loop_count % MAIN_LOOP_RATE_HZ) == 0) {
        HAL_LED_Toggle(2);
    }
    
    // Safety check (100 Hz)
    if (now - g_timing.last_safety_check >= (1000 / SAFETY_RATE_HZ)) {
        g_timing.last_safety_check = now;
        System_SafetyCheck();
    }
    
    // Process incoming commands
    System_ProcessCommands();
    
    // Update sensors (10 Hz)
    if (now - g_timing.last_sensor_poll >= (1000 / SENSOR_RATE_HZ)) {
        g_timing.last_sensor_poll = now;
        System_UpdateSensors();
    }
    
    // Mode-specific processing
    switch (g_system.mode) {
        case MODE_CALIBRATION:
            // Calibration is typically blocking
            break;
            
        case MODE_IDLE:
            // Idle state - still update rotor/threshold so host commands take effect
            // fallthrough
        case MODE_MANUAL:
        case MODE_EXPERIMENT:
        case MODE_SWEEP:
            // Update rotor control
            RotorControl_Update();
            
            // Get current z from rotor
            g_system.current_z = RotorControl_GetZ();
            
            // Update threshold logic
            ThresholdLogic_Update(
                g_system.current_z,
                g_system.current_kappa,
                g_system.current_eta,
                g_system.current_R
            );
            
            // Update derived quantities
            // κ tracks toward φ⁻¹ based on physics
            float kappa_error = g_system.current_kappa - PHI_INV;
            g_system.current_kappa -= 0.001f * kappa_error;  // Slow convergence
            
            // η derived from negentropy
            g_system.current_eta = sqrtf(RotorControl_GetDeltaSNeg());
            
            // R is complexity measure (placeholder)
            g_system.current_R = (int)(RotorControl_GetComplexity() * 10.0f);
            if (g_system.current_R < 1) g_system.current_R = 1;
            break;
            
        case MODE_FAULT:
            // Error LED on, motor disabled
            HAL_LED_Set(1, true);
            RotorControl_Disable();
            break;
    }
}


/* ============================================================================
 * COMMAND PROCESSING
 * ============================================================================ */

static void System_ProcessCommands(void) {
    // Delegate to packet-based host protocol
    (void)CommProtocol_Process();
}


/* ============================================================================
 * SENSOR UPDATES
 * ============================================================================ */

static void System_UpdateSensors(void) {
    HAL_Sensor_ReadAll(&g_system.sensors);
    
    // Check temperature limits
    if (g_system.sensors.temperature_c > 80.0f) {
        // Over-temperature warning
        HAL_LED_Set(1, true);
    } else {
        HAL_LED_Set(1, false);
    }
    
    // Check magnetic field stability (for high-field magnets)
    // Typical drift should be < 0.1 ppm
}


/* ============================================================================
 * SAFETY CHECKS
 * ============================================================================ */

static void System_SafetyCheck(void) {
    // Check interlock
    g_system.interlock_ok = HAL_Safety_InterlockOK();
    if (!g_system.interlock_ok && g_system.mode != MODE_FAULT) {
        // Emergency stop
        HAL_Safety_EmergencyStop();
        g_system.mode = MODE_FAULT;
        HAL_LED_Set(1, true);
        return;
    }
    
    // Check for motor fault
    if (RotorControl_IsStalled() && g_system.mode != MODE_FAULT) {
        g_system.mode = MODE_FAULT;
        HAL_LED_Set(1, true);
        return;
    }
    
    // Check sensor validity
    if (g_system.sensors.temperature_c < -40.0f || 
        g_system.sensors.temperature_c > 150.0f) {
        // Sensor fault - invalid reading
        g_system.mode = MODE_FAULT;
        HAL_LED_Set(1, true);
    }
}




/* ============================================================================
 * THRESHOLD EVENT HANDLER
 * ============================================================================ */

static void System_OnThresholdEvent(ThresholdEvent_t event, 
                                     float threshold, int direction) {
    // Notify host
    (void)CommProtocol_SendEvent(event, threshold);
    /**
     * Handle threshold crossing events
     * 
     * This is called by ThresholdLogic when z crosses key thresholds.
     * Can trigger automatic actions or notify host software.
     */
    
    // Log event (would send to host)
    switch (event) {
        case EVENT_THRESHOLD_PHI_INV:
            // Crossed φ⁻¹ - memory/consciousness threshold
            if (direction > 0) {
                // Entering memory-capable regime
            } else {
                // Exiting memory regime
            }
            break;
            
        case EVENT_THRESHOLD_ZC:
            // Crossed z_c - critical/universal threshold
            if (direction > 0) {
                // Achieved Turing universality
                HAL_LED_Toggle(0);  // Visual feedback
            }
            break;
            
        case EVENT_LENS_ENTER:
            // Entered THE_LENS phase
            // This is the critical point where ΔS_neg peaks
            break;
            
        case EVENT_K_FORMATION_ENTER:
            // K-formation achieved (κ ≥ 0.92, η > φ⁻¹, R ≥ 7)
            break;
            
        default:
            break;
    }
}


/* ============================================================================
 * EXPERIMENT MODES
 * ============================================================================ */

HAL_Status_t System_StartExperiment(ExperimentType_t exp_type) {
    if (g_system.mode == MODE_FAULT) {
        return HAL_ERROR;
    }
    
    g_system.mode = MODE_EXPERIMENT;
    g_system.current_experiment = exp_type;
    
    switch (exp_type) {
        case EXPERIMENT_FID:
            // Basic FID measurement
            RotorControl_Enable();
            RotorControl_SetZ(PHI_INV);  // Start at φ⁻¹
            PulseControl_FID(0.8f);
            break;
            
        case EXPERIMENT_SPIN_ECHO:
            // Spin echo for T2 measurement
            RotorControl_Enable();
            RotorControl_SetZ(PHI_INV);
            PulseControl_SpinEcho(0.8f, 1000);  // 1ms echo time
            break;
            
        case EXPERIMENT_CPMG:
            // CPMG for T2 decay
            RotorControl_Enable();
            RotorControl_SetZ(PHI_INV);
            PulseControl_CPMG(0.8f, 500, 100);  // 100 echoes
            break;
            
        case EXPERIMENT_NUTATION:
            // Nutation to verify |S|/ℏ = √3/2
            RotorControl_Enable();
            RotorControl_SetZ(Z_CRITICAL);  // At z_c
            PulseControl_CalibrateB1();
            float spin_mag;
            PulseControl_VerifySpinHalf(&spin_mag);
            break;
            
        case EXPERIMENT_QUASICRYSTAL:
            // Quasicrystal dynamics - sweep z toward z_c
            RotorControl_Enable();
            RotorControl_SweepZ(0.3f, Z_CRITICAL, 0.01f);  // Slow sweep
            break;
            
        case EXPERIMENT_E8_PROBE:
            // E8 critical point probing (requires CoNb₂O₆ sample)
            RotorControl_Enable();
            RotorControl_SetZ(Z_CRITICAL);
            // Sweep field and acquire spectra for mass ratio analysis
            break;
            
        case EXPERIMENT_HOLOGRAPHIC:
            // Holographic bound experiment
            // Vary energy input, measure information throughput
            RotorControl_Enable();
            RotorControl_SetZ(Z_CRITICAL);
            break;
            
        case EXPERIMENT_OMEGA_POINT:
            // Omega point dynamics - approach z_c with increasing rate
            RotorControl_Enable();
            for (float z = 0.3f; z < 0.95f; z += 0.01f) {
                RotorControl_SetZWithModulation(z, 1.0f);
                HAL_Delay(100);
                
                // Log processing rate (would diverge at z_c)
                float processing_rate = 1.0f / (1.0f - z / Z_CRITICAL);
                // Send to telemetry
            }
            break;
            
        default:
            g_system.mode = MODE_IDLE;
            return HAL_INVALID_PARAM;
    }
    
    return HAL_OK;
}


HAL_Status_t System_StopExperiment(void) {
    g_system.current_experiment = EXPERIMENT_NONE;
    g_system.mode = MODE_IDLE;
    RotorControl_Disable();
    PulseControl_AbortSequence();
    return HAL_OK;
}


/* ============================================================================
 * CALIBRATION
 * ============================================================================ */

HAL_Status_t System_RunCalibration(void) {
    if (g_system.mode == MODE_FAULT) {
        return HAL_ERROR;
    }
    
    g_system.mode = MODE_CALIBRATION;
    HAL_Status_t status;
    
    // 1. Calibrate B₁ field
    status = PulseControl_CalibrateB1();
    if (status != HAL_OK) {
        g_system.mode = MODE_FAULT;
        return status;
    }
    
    // 2. Verify spin-1/2 magnitude
    float spin_magnitude;
    status = PulseControl_VerifySpinHalf(&spin_magnitude);
    if (status != HAL_OK) {
        // Warning: spin magnitude doesn't match expected √3/2
        // Continue but flag
    }
    
    // 3. Calibrate rotor
    RotorControl_Enable();
    
    // Sweep through RPM range and verify encoder
    for (float rpm = ROTOR_RPM_MIN; rpm <= ROTOR_RPM_MAX; rpm += 500.0f) {
        RotorControl_SetRPM(rpm);
        HAL_Delay(500);  // Wait for settle
        RotorControl_Update();
        
        // Verify actual RPM matches target
        float actual = RotorControl_GetRPM();
        if (fabsf(actual - rpm) > rpm * 0.1f) {
            // More than 10% error - calibration failed
            RotorControl_Disable();
            g_system.mode = MODE_FAULT;
            return HAL_ERROR;
        }
    }
    
    RotorControl_Disable();
    
    g_system.calibration_complete = true;
    g_system.mode = MODE_IDLE;
    
    return HAL_OK;
}


/* ============================================================================
 * DIAGNOSTIC FUNCTIONS
 * ============================================================================ */

void System_PrintStatus(void) {
    /**
     * Print human-readable status (for debug)
     */
    
    // This would output via UART/debug console
    /*
    printf("\n=== NUCLEAR SPINNER STATUS ===\n");
    printf("Firmware: %s\n", FIRMWARE_SIGNATURE);
    printf("Uptime: %lu ms\n", g_system.uptime_ms);
    printf("Mode: %d\n", g_system.mode);
    printf("\n--- Physics State ---\n");
    printf("z: %.4f (target: %.4f)\n", g_system.current_z, g_system.target_z);
    printf("ΔS_neg: %.4f\n", RotorControl_GetDeltaSNeg());
    printf("Complexity: %.4f\n", RotorControl_GetComplexity());
    printf("Tier: %s\n", ThresholdLogic_GetTierName(ThresholdLogic_GetTier()));
    printf("Phase: %s\n", ThresholdLogic_GetPhaseName(ThresholdLogic_GetPhase()));
    printf("κ: %.6f (target: %.6f = φ⁻¹)\n", g_system.current_kappa, PHI_INV);
    printf("η: %.4f\n", g_system.current_eta);
    printf("R: %d\n", g_system.current_R);
    printf("K-formation: %s\n", ThresholdLogic_IsKFormationActive() ? "ACTIVE" : "inactive");
    printf("\n--- Hardware ---\n");
    printf("RPM: %.1f\n", RotorControl_GetRPM());
    printf("Temperature: %.1f °C\n", g_system.sensors.temperature_c);
    printf("Interlock: %s\n", g_system.interlock_ok ? "OK" : "OPEN");
    printf("==============================\n\n");
    */
}
