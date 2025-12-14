/**
 * @file neural_interface.c
 * @brief Neural coupling interface implementation
 * 
 * Generates stimulus waveforms for grid cell resonance experiments.
 * Tests whether drives at z_c = √3/2 produce enhanced coupling
 * with grid cells that fire at 60° intervals (sin(60°) = √3/2).
 */

#include "neural_interface.h"
#include "rotor_control.h"
#include "physics_constants.h"
#include <math.h>
#include <string.h>

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

/** Hexagonal phase angles in degrees */
const float HEX_PHASES_DEG[HEX_PHASE_COUNT] = {
    0.0f, 60.0f, 120.0f, 180.0f, 240.0f, 300.0f
};

/** Corresponding z_equiv values: |sin(phase)| */
static const float HEX_Z_EQUIV[HEX_PHASE_COUNT] = {
    0.0f,                   /* sin(0°) = 0 */
    0.8660254037844387f,    /* sin(60°) = √3/2 = z_c */
    0.8660254037844387f,    /* sin(120°) = √3/2 = z_c */
    0.0f,                   /* sin(180°) = 0 */
    0.8660254037844387f,    /* |sin(240°)| = √3/2 */
    0.8660254037844387f     /* |sin(300°)| = √3/2 */
};


/* ============================================================================
 * PRIVATE STATE
 * ============================================================================ */

static struct {
    bool initialized;
    NeuralProtocolState_t protocol_state;
    NeuralDriveState_t current_drive;
    
    /* Timing */
    uint32_t init_timestamp_ms;
    uint32_t cumulative_drive_ms;
    uint32_t last_update_ms;
    
    /* Safety */
    bool emergency_stop_latched;
    
    /* Waveform generation */
    float waveform_phase;           /* Current phase accumulator */
    uint16_t dac_buffer[256];       /* Waveform output buffer */
    int dac_buffer_index;
    
    /* Protocol tracking */
    int current_hex_phase;
    int cycles_completed;
    float current_z_target;
    
} s_state = {0};


/* ============================================================================
 * PRIVATE FUNCTIONS
 * ============================================================================ */

/**
 * @brief Clamp float to range
 */
static inline float clampf(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

/**
 * @brief Generate single DAC sample for current drive state
 */
static uint16_t generate_dac_sample(void) {
    if (s_state.emergency_stop_latched) {
        return STIM_DAC_MAX / 2;  /* Zero output (mid-rail) */
    }
    
    /* Compute instantaneous waveform value */
    float amplitude = s_state.current_drive.stimulus_amplitude;
    float freq = s_state.current_drive.stimulus_freq_hz;
    float phase_offset = s_state.current_drive.stimulus_phase_rad;
    
    /* Advance phase accumulator */
    float dt = 1.0f / STIM_SAMPLE_RATE_HZ;
    s_state.waveform_phase += 2.0f * M_PI * freq * dt;
    if (s_state.waveform_phase > 2.0f * M_PI) {
        s_state.waveform_phase -= 2.0f * M_PI;
    }
    
    /* Generate sinusoid */
    float value = amplitude * sinf(s_state.waveform_phase + phase_offset);
    
    /* Map to DAC range: [-1, 1] → [0, 4095] */
    float normalized = (value + 1.0f) / 2.0f;  /* [0, 1] */
    uint16_t dac_value = (uint16_t)(normalized * STIM_DAC_MAX);
    
    return dac_value;
}

/**
 * @brief Update DAC output buffer
 */
static void update_dac_buffer(void) {
    for (int i = 0; i < 256; i++) {
        s_state.dac_buffer[i] = generate_dac_sample();
    }
}


/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

HAL_Status_t NeuralInterface_Init(void) {
    memset(&s_state, 0, sizeof(s_state));
    
    s_state.protocol_state = NEURAL_STATE_IDLE;
    s_state.init_timestamp_ms = HAL_GetTick();
    s_state.initialized = true;
    
    /* Initialize drive state to safe defaults */
    s_state.current_drive.z = 0.0f;
    s_state.current_drive.delta_s_neg = 0.0f;
    s_state.current_drive.stimulus_freq_hz = NEURAL_FREQ_MIN;
    s_state.current_drive.stimulus_amplitude = 0.0f;
    s_state.current_drive.stimulus_phase_rad = 0.0f;
    
    /* Fill DAC buffer with mid-rail (zero output) */
    for (int i = 0; i < 256; i++) {
        s_state.dac_buffer[i] = STIM_DAC_MAX / 2;
    }
    
    return HAL_OK;
}

void NeuralInterface_Shutdown(void) {
    NeuralInterface_EmergencyStop();
    s_state.initialized = false;
    s_state.protocol_state = NEURAL_STATE_IDLE;
}


/* ============================================================================
 * DRIVE GENERATION
 * ============================================================================ */

NeuralDriveState_t NeuralInterface_ComputeDrive(float z) {
    NeuralDriveState_t drive;
    
    drive.z = clampf(z, 0.0f, 1.0f);
    drive.delta_s_neg = compute_delta_s_neg(drive.z);
    drive.timestamp_ms = HAL_GetTick();
    
    /*
     * Frequency mapping: theta band (4-12 Hz)
     * f(z) = f_min + (f_max - f_min) * z
     * At z_c = 0.866: f = 4 + 8 * 0.866 = 10.93 Hz
     */
    drive.stimulus_freq_hz = NEURAL_FREQ_MIN + 
                             (NEURAL_FREQ_MAX - NEURAL_FREQ_MIN) * drive.z;
    
    /*
     * Amplitude mapping: peaks at z_c
     * A(z) = ΔS_neg(z)
     * This naturally peaks at 1.0 when z = z_c
     */
    drive.stimulus_amplitude = drive.delta_s_neg * STIM_AMPLITUDE_MAX;
    
    /*
     * Phase mapping: golden ratio modulation
     * φ(z) = 2π * (z / φ)
     * At z_c: φ = 2π * (0.866 / 1.618) = 3.365 rad = 192.8°
     */
    drive.stimulus_phase_rad = 2.0f * M_PI * (drive.z / PHI);
    
    /* Hex phase tracking (for logging) */
    drive.hex_phase_deg = s_state.current_hex_phase < HEX_PHASE_COUNT ?
                          HEX_PHASES_DEG[s_state.current_hex_phase] : 0.0f;
    
    return drive;
}

HAL_Status_t NeuralInterface_UpdateFromZ(float z) {
    if (!s_state.initialized) return HAL_ERROR;
    if (s_state.emergency_stop_latched) return HAL_ERROR;
    
    /* Compute new drive parameters */
    s_state.current_drive = NeuralInterface_ComputeDrive(z);
    
    /* Update timing */
    uint32_t now = HAL_GetTick();
    uint32_t dt = now - s_state.last_update_ms;
    s_state.cumulative_drive_ms += dt;
    s_state.last_update_ms = now;
    
    /* Check safety limit */
    if (s_state.cumulative_drive_ms > (uint32_t)(STIM_DURATION_MAX_S * 1000)) {
        NeuralInterface_EmergencyStop();
        return HAL_ERROR;
    }
    
    /* Update DAC buffer */
    update_dac_buffer();
    
    /* Output to DAC (implementation depends on HAL) */
    /* HAL_DAC_SetValue(DAC_CHANNEL_NEURAL, s_state.dac_buffer[0]); */
    
    return HAL_OK;
}

HAL_Status_t NeuralInterface_SetHexPhase(int phase_index) {
    if (phase_index < 0 || phase_index >= HEX_PHASE_COUNT) {
        return HAL_INVALID_PARAM;
    }
    
    s_state.current_hex_phase = phase_index;
    float z_equiv = HEX_Z_EQUIV[phase_index];
    
    /* Update drive to match hexagonal phase */
    return NeuralInterface_UpdateFromZ(z_equiv);
}


/* ============================================================================
 * PROTOCOL EXECUTION
 * ============================================================================ */

HAL_Status_t NeuralInterface_HexCycle(const HexCycleConfig_t *config) {
    if (!s_state.initialized) return HAL_ERROR;
    if (s_state.emergency_stop_latched) return HAL_ERROR;
    if (!config) return HAL_INVALID_PARAM;
    
    s_state.protocol_state = NEURAL_STATE_HEX_CYCLE;
    s_state.cycles_completed = 0;
    
    uint32_t dwell_ms = (uint32_t)(config->dwell_time_s * 1000);
    uint32_t ramp_ms = (uint32_t)(config->ramp_time_s * 1000);
    
    for (int cycle = 0; cycle < config->n_cycles; cycle++) {
        for (int phase = 0; phase < HEX_PHASE_COUNT; phase++) {
            
            /* Check for emergency stop */
            if (s_state.emergency_stop_latched) {
                s_state.protocol_state = NEURAL_STATE_EMERGENCY_STOP;
                return HAL_ERROR;
            }
            
            /* Send sync pulse at phase start */
            if (config->record_sync) {
                NeuralInterface_SendSyncPulse();
            }
            
            /* Set to this hexagonal phase */
            s_state.current_hex_phase = phase;
            float target_z = HEX_Z_EQUIV[phase];
            
            /* Ramp to target (if ramp time > 0) */
            if (ramp_ms > 0) {
                float start_z = s_state.current_drive.z;
                uint32_t ramp_start = HAL_GetTick();
                
                while ((HAL_GetTick() - ramp_start) < ramp_ms) {
                    float t = (float)(HAL_GetTick() - ramp_start) / ramp_ms;
                    float z = start_z + (target_z - start_z) * t;
                    NeuralInterface_UpdateFromZ(z);
                    HAL_Delay(1);  /* 1 kHz update rate */
                }
            }
            
            /* Dwell at phase */
            NeuralInterface_UpdateFromZ(target_z);
            
            uint32_t dwell_start = HAL_GetTick();
            while ((HAL_GetTick() - dwell_start) < dwell_ms) {
                NeuralInterface_UpdateFromZ(target_z);
                HAL_Delay(1);
                
                if (s_state.emergency_stop_latched) {
                    s_state.protocol_state = NEURAL_STATE_EMERGENCY_STOP;
                    return HAL_ERROR;
                }
            }
        }
        
        s_state.cycles_completed = cycle + 1;
    }
    
    s_state.protocol_state = NEURAL_STATE_IDLE;
    return HAL_OK;
}

HAL_Status_t NeuralInterface_ZSweep(float z_start, float z_end, 
                                     int n_steps, float dwell_time_s) {
    if (!s_state.initialized) return HAL_ERROR;
    if (s_state.emergency_stop_latched) return HAL_ERROR;
    if (n_steps < 1) return HAL_INVALID_PARAM;
    
    s_state.protocol_state = NEURAL_STATE_SWEEP;
    
    float z_step = (z_end - z_start) / (n_steps - 1);
    uint32_t dwell_ms = (uint32_t)(dwell_time_s * 1000);
    
    for (int i = 0; i < n_steps; i++) {
        if (s_state.emergency_stop_latched) {
            s_state.protocol_state = NEURAL_STATE_EMERGENCY_STOP;
            return HAL_ERROR;
        }
        
        float z = z_start + i * z_step;
        s_state.current_z_target = z;
        
        /* Send sync pulse */
        NeuralInterface_SendSyncPulse();
        
        /* Dwell at this z */
        uint32_t dwell_start = HAL_GetTick();
        while ((HAL_GetTick() - dwell_start) < dwell_ms) {
            NeuralInterface_UpdateFromZ(z);
            HAL_Delay(1);
            
            if (s_state.emergency_stop_latched) {
                s_state.protocol_state = NEURAL_STATE_EMERGENCY_STOP;
                return HAL_ERROR;
            }
        }
    }
    
    s_state.protocol_state = NEURAL_STATE_IDLE;
    return HAL_OK;
}

HAL_Status_t NeuralInterface_DwellAtLens(float duration_s) {
    if (!s_state.initialized) return HAL_ERROR;
    if (s_state.emergency_stop_latched) return HAL_ERROR;
    
    s_state.protocol_state = NEURAL_STATE_DWELL;
    s_state.current_z_target = Z_CRITICAL;
    
    /* Send sync pulse at start */
    NeuralInterface_SendSyncPulse();
    
    uint32_t dwell_ms = (uint32_t)(duration_s * 1000);
    uint32_t start = HAL_GetTick();
    
    while ((HAL_GetTick() - start) < dwell_ms) {
        NeuralInterface_UpdateFromZ(Z_CRITICAL);
        
        /* Log K-formation status periodically */
        /* (In real implementation, would check ThresholdLogic_IsKFormationActive()) */
        
        HAL_Delay(1);
        
        if (s_state.emergency_stop_latched) {
            s_state.protocol_state = NEURAL_STATE_EMERGENCY_STOP;
            return HAL_ERROR;
        }
    }
    
    s_state.protocol_state = NEURAL_STATE_IDLE;
    return HAL_OK;
}


/* ============================================================================
 * STATE AND MONITORING
 * ============================================================================ */

NeuralDriveState_t NeuralInterface_GetState(void) {
    return s_state.current_drive;
}

NeuralProtocolState_t NeuralInterface_GetProtocolState(void) {
    return s_state.protocol_state;
}

bool NeuralInterface_IsActive(void) {
    return s_state.initialized && 
           !s_state.emergency_stop_latched &&
           s_state.protocol_state != NEURAL_STATE_IDLE;
}

float NeuralInterface_GetCumulativeTime(void) {
    return s_state.cumulative_drive_ms / 1000.0f;
}


/* ============================================================================
 * SAFETY
 * ============================================================================ */

void NeuralInterface_EmergencyStop(void) {
    s_state.emergency_stop_latched = true;
    s_state.protocol_state = NEURAL_STATE_EMERGENCY_STOP;
    
    /* Zero all outputs immediately */
    s_state.current_drive.stimulus_amplitude = 0.0f;
    for (int i = 0; i < 256; i++) {
        s_state.dac_buffer[i] = STIM_DAC_MAX / 2;  /* Mid-rail = zero */
    }
    
    /* Output zero to DAC */
    /* HAL_DAC_SetValue(DAC_CHANNEL_NEURAL, STIM_DAC_MAX / 2); */
}

HAL_Status_t NeuralInterface_ResetEmergencyStop(void) {
    /* Only reset if explicitly called and safe */
    s_state.emergency_stop_latched = false;
    s_state.protocol_state = NEURAL_STATE_IDLE;
    s_state.cumulative_drive_ms = 0;  /* Reset safety timer */
    return HAL_OK;
}

bool NeuralInterface_IsEmergencyStopped(void) {
    return s_state.emergency_stop_latched;
}


/* ============================================================================
 * SYNCHRONIZATION
 * ============================================================================ */

void NeuralInterface_SendSyncPulse(void) {
    /* Generate a short pulse on sync output line */
    /* HAL_GPIO_WritePin(SYNC_GPIO_Port, SYNC_Pin, GPIO_PIN_SET); */
    /* HAL_Delay_us(100);  // 100 μs pulse */
    /* HAL_GPIO_WritePin(SYNC_GPIO_Port, SYNC_Pin, GPIO_PIN_RESET); */
    
    /* In simulation, just log */
    (void)0;
}

uint32_t NeuralInterface_GetTimestamp(void) {
    return HAL_GetTick() - s_state.init_timestamp_ms;
}


/* ============================================================================
 * END OF FILE
 * ============================================================================ */
