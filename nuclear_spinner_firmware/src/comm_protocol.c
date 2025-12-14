/**
 * @file comm_protocol.c
 * @brief Host Communication Protocol Implementation
 * 
 * Implements bidirectional communication between firmware and host software:
 * 
 * COMMAND PROTOCOL (Host → Device):
 * ┌─────────┬─────────┬──────────────┬──────────┐
 * │ Header  │ Command │   Payload    │ Checksum │
 * │ 0xAA55  │ 1 byte  │ 0-64 bytes   │ 2 bytes  │
 * └─────────┴─────────┴──────────────┴──────────┘
 * 
 * TELEMETRY PROTOCOL (Device → Host):
 * ┌─────────┬────────┬───────────────┬──────────┐
 * │ Header  │ Length │   Payload     │ Checksum │
 * │ 0xAA55  │ 2 bytes│ Variable      │ 2 bytes  │
 * └─────────┴────────┴───────────────┴──────────┘
 * 
 * Commands map directly to physics operations:
 * - SET_Z: Drive rotor to z-coordinate target
 * - PULSE: Execute RF pulse with physics modulation
 * - SWEEP: Perform z-sweep for negentropy mapping
 * - EXPERIMENT: Run predefined physics experiments
 * 
 * Signature: comm-protocol|v1.0.0|nuclear-spinner
 * 
 * @version 1.0.0
 */

#include "comm_protocol.h"
#include "hal_hardware.h"
#include "physics_constants.h"
#include "pulse_control.h"
#include "rotor_control.h"
#include "threshold_logic.h"
#include <string.h>

/* ============================================================================
 * PROTOCOL CONSTANTS
 * ============================================================================ */

#define PROTOCOL_HEADER         0xAA55
#define PROTOCOL_VERSION        0x01

#define MAX_PAYLOAD_SIZE        64
#define RX_BUFFER_SIZE          128
#define TX_BUFFER_SIZE          256

#define CMD_TIMEOUT_MS          100


/* ============================================================================
 * COMMAND DEFINITIONS
 * ============================================================================ */

typedef enum {
    // System commands (0x00-0x0F)
    CMD_NOP             = 0x00,
    CMD_PING            = 0x01,
    CMD_VERSION         = 0x02,
    CMD_RESET           = 0x03,
    CMD_STATUS          = 0x04,
    
    // Motor/Rotor commands (0x10-0x1F)
    CMD_MOTOR_ENABLE    = 0x10,
    CMD_MOTOR_DISABLE   = 0x11,
    CMD_SET_RPM         = 0x12,
    CMD_SET_Z           = 0x13,
    CMD_SET_Z_MODULATED = 0x14,
    CMD_SWEEP_Z         = 0x15,
    CMD_PHASE_LOCK      = 0x16,
    
    // RF Pulse commands (0x20-0x2F)
    CMD_PULSE           = 0x20,
    CMD_PULSE_PI2       = 0x21,
    CMD_PULSE_PI        = 0x22,
    CMD_PULSE_SEQUENCE  = 0x23,
    CMD_PULSE_ICOSA     = 0x24,
    CMD_PULSE_HEX       = 0x25,
    CMD_PULSE_ABORT     = 0x26,
    
    // Experiment commands (0x30-0x3F)
    CMD_EXP_FID         = 0x30,
    CMD_EXP_ECHO        = 0x31,
    CMD_EXP_CPMG        = 0x32,
    CMD_EXP_NUTATION    = 0x33,
    CMD_EXP_QUASICRYSTAL= 0x34,
    CMD_EXP_E8          = 0x35,
    CMD_EXP_HOLOGRAPHIC = 0x36,
    CMD_EXP_OMEGA       = 0x37,
    CMD_EXP_STOP        = 0x3F,
    
    // Calibration commands (0x40-0x4F)
    CMD_CAL_B1          = 0x40,
    CMD_CAL_VERIFY_SPIN = 0x41,
    CMD_CAL_ROTOR       = 0x42,
    CMD_CAL_SENSORS     = 0x43,
    
    // Data commands (0x50-0x5F)
    CMD_GET_FID         = 0x50,
    CMD_GET_SENSORS     = 0x51,
    CMD_GET_THRESHOLD   = 0x52,
    CMD_GET_PHYSICS     = 0x53,
    
    // Operator commands (0x60-0x6F)
    CMD_OP_CLOSURE      = 0x60,
    CMD_OP_FUSION       = 0x61,
    CMD_OP_AMPLIFY      = 0x62,
    CMD_OP_DECOHERE     = 0x63,
    CMD_OP_GROUP        = 0x64,
    CMD_OP_SEPARATE     = 0x65,
    
    // Telemetry control (0x70-0x7F)
    CMD_TELEM_START     = 0x70,
    CMD_TELEM_STOP      = 0x71,
    CMD_TELEM_RATE      = 0x72,
    
} CommandCode_t;


/* ============================================================================
 * RESPONSE CODES
 * ============================================================================ */

typedef enum {
    RESP_OK             = 0x00,
    RESP_ERROR          = 0x01,
    RESP_INVALID_CMD    = 0x02,
    RESP_INVALID_PARAM  = 0x03,
    RESP_BUSY           = 0x04,
    RESP_TIMEOUT        = 0x05,
    RESP_NOT_CALIBRATED = 0x06,
    RESP_FAULT          = 0x07,
} ResponseCode_t;


/* ============================================================================
 * TELEMETRY PACKET TYPES
 * ============================================================================ */

typedef enum {
    TELEM_STATUS        = 0x01,
    TELEM_PHYSICS       = 0x02,
    TELEM_THRESHOLD     = 0x03,
    TELEM_FID_DATA      = 0x04,
    TELEM_SENSOR_DATA   = 0x05,
    TELEM_EVENT         = 0x06,
    TELEM_DEBUG         = 0xFF,
} TelemetryType_t;


/* ============================================================================
 * PRIVATE DATA
 * ============================================================================ */

static uint8_t s_rx_buffer[RX_BUFFER_SIZE];
static uint8_t s_tx_buffer[TX_BUFFER_SIZE];
static uint32_t s_rx_index = 0;

static bool s_telemetry_enabled = true;
static uint32_t s_telemetry_rate_ms = 10;  // 100 Hz default
static uint32_t s_last_telemetry_tick = 0;

static CommStats_t s_stats = {0};


/* ============================================================================
 * PRIVATE FUNCTION PROTOTYPES
 * ============================================================================ */

static uint16_t compute_crc16(const uint8_t *data, uint32_t len);
static HAL_Status_t send_response(uint8_t cmd, ResponseCode_t resp, 
                                   const uint8_t *data, uint32_t len);
static HAL_Status_t process_command(uint8_t cmd, const uint8_t *payload, uint32_t len);
static void pack_float(uint8_t *buf, float value);
static float unpack_float(const uint8_t *buf);
static void pack_uint32(uint8_t *buf, uint32_t value);
static uint32_t unpack_uint32(const uint8_t *buf);


/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

HAL_Status_t CommProtocol_Init(void) {
    memset(s_rx_buffer, 0, sizeof(s_rx_buffer));
    memset(s_tx_buffer, 0, sizeof(s_tx_buffer));
    s_rx_index = 0;
    
    s_telemetry_enabled = false;
    s_telemetry_rate_ms = 10;
    s_last_telemetry_tick = 0;
    
    memset(&s_stats, 0, sizeof(s_stats));
    
    // Initialize UART/USB peripheral here
    (void)HAL_Comm_Init();
    
    return HAL_OK;
}


/* ============================================================================
 * MAIN PROCESSING LOOP
 * ============================================================================ */

HAL_Status_t CommProtocol_Process(void) {
    /**
     * Main communication processing function
     * Call from main loop to handle incoming commands and send telemetry
     */
    


// Pull any available bytes from transport into RX buffer
if (s_rx_index < RX_BUFFER_SIZE) {
    uint32_t space = RX_BUFFER_SIZE - s_rx_index;
    uint32_t n = HAL_Comm_Read(&s_rx_buffer[s_rx_index], space);
    if (n > space) n = space;
    s_rx_index += n;
} else {
    // Overflow guard: drop buffer and resync
    s_rx_index = 0;
    s_stats.sync_errors++;
}

    // Process any pending received data
    // In production, this would read from UART/USB receive buffer
    // For demonstration, assume data is already in s_rx_buffer
    
    // Check for complete packet
    if (s_rx_index >= 5) {  // Minimum packet: header(2) + cmd(1) + checksum(2)
        // Verify header
        uint16_t header = (s_rx_buffer[0] << 8) | s_rx_buffer[1];
        if (header == PROTOCOL_HEADER) {
            uint8_t cmd = s_rx_buffer[2];
            uint8_t payload_len = s_rx_buffer[3];
            
            if (s_rx_index >= (uint32_t)(4 + payload_len + 2)) {
                // Full packet received
                uint16_t rx_crc = (s_rx_buffer[4 + payload_len] << 8) | 
                                   s_rx_buffer[4 + payload_len + 1];
                uint16_t calc_crc = compute_crc16(s_rx_buffer, 4 + payload_len);
                
                if (rx_crc == calc_crc) {
                    // Valid packet
                    s_stats.packets_received++;
                    process_command(cmd, &s_rx_buffer[4], payload_len);
                } else {
                    // CRC error
                    s_stats.crc_errors++;
                }
                
                // Reset buffer
                s_rx_index = 0;
            }
        } else {
            // Invalid header, shift buffer
            memmove(s_rx_buffer, s_rx_buffer + 1, s_rx_index - 1);
            s_rx_index--;
            s_stats.sync_errors++;
        }
    }
    
    // Send telemetry if enabled and interval elapsed
    if (s_telemetry_enabled) {
        uint32_t now = HAL_GetTick();
        if (now - s_last_telemetry_tick >= s_telemetry_rate_ms) {
            s_last_telemetry_tick = now;
            CommProtocol_SendTelemetry();
        }
    }
    
    return HAL_OK;
}


/* ============================================================================
 * COMMAND PROCESSING
 * ============================================================================ */

static HAL_Status_t process_command(uint8_t cmd, const uint8_t *payload, uint32_t len) {
    HAL_Status_t status = HAL_OK;
    ResponseCode_t resp = RESP_OK;
    uint8_t resp_data[32];
    uint32_t resp_len = 0;
    
    switch (cmd) {
        // ===== System Commands =====
        case CMD_NOP:
            // No operation
            break;
            
        case CMD_PING:
            // Echo back timestamp
            pack_uint32(resp_data, HAL_GetTick());
            resp_len = 4;
            break;
            
        case CMD_VERSION: {
            // Return firmware version and signature
            resp_data[0] = 1;  // Major
            resp_data[1] = 0;  // Minor
            resp_data[2] = 0;  // Patch
            resp_data[3] = PROTOCOL_VERSION;
            resp_len = 4;
            break;
        }
        
        case CMD_RESET:
            // System reset (careful!)
            // NVIC_SystemReset();
            break;
            
        case CMD_STATUS: {
            // Return system status
            resp_data[0] = RotorControl_IsStalled() ? 1 : 0;
            resp_data[1] = HAL_Safety_InterlockOK() ? 1 : 0;
            resp_data[2] = (uint8_t)ThresholdLogic_GetTier();
            resp_data[3] = (uint8_t)ThresholdLogic_GetPhase();
            pack_float(&resp_data[4], RotorControl_GetZ());
            pack_float(&resp_data[8], RotorControl_GetDeltaSNeg());
            resp_len = 12;
            break;
        }
        
        // ===== Motor/Rotor Commands =====
        case CMD_MOTOR_ENABLE:
            status = RotorControl_Enable();
            resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            break;
            
        case CMD_MOTOR_DISABLE:
            status = RotorControl_Disable();
            resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            break;
            
        case CMD_SET_RPM:
            if (len >= 4) {
                float rpm = unpack_float(payload);
                status = RotorControl_SetRPM(rpm);
                resp = (status == HAL_OK) ? RESP_OK : RESP_INVALID_PARAM;
            } else {
                resp = RESP_INVALID_PARAM;
            }
            break;
            
        case CMD_SET_Z:
            if (len >= 4) {
                float z = unpack_float(payload);
                status = RotorControl_SetZ(z);
                resp = (status == HAL_OK) ? RESP_OK : RESP_INVALID_PARAM;
            } else {
                resp = RESP_INVALID_PARAM;
            }
            break;
            
        case CMD_SET_Z_MODULATED:
            if (len >= 8) {
                float z = unpack_float(payload);
                float gain = unpack_float(payload + 4);
                status = RotorControl_SetZWithModulation(z, gain);
                resp = (status == HAL_OK) ? RESP_OK : RESP_INVALID_PARAM;
            } else {
                resp = RESP_INVALID_PARAM;
            }
            break;
            
        case CMD_SWEEP_Z:
            if (len >= 12) {
                float z_start = unpack_float(payload);
                float z_end = unpack_float(payload + 4);
                float rate = unpack_float(payload + 8);
                status = RotorControl_SweepZ(z_start, z_end, rate);
                resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            } else {
                resp = RESP_INVALID_PARAM;
            }
            break;
            
        case CMD_PHASE_LOCK:
            if (len >= 5) {
                uint8_t sector = payload[0];
                float tolerance = unpack_float(payload + 1);
                if (sector <= 5) {
                    RotorControl_SetHexagonalPhase(sector);
                    status = HAL_OK;
                } else {
                    status = RotorControl_EnablePhaseLock(
                        unpack_float(payload + 1), tolerance);
                }
                resp = (status == HAL_OK) ? RESP_OK : RESP_INVALID_PARAM;
            } else {
                resp = RESP_INVALID_PARAM;
            }
            break;
            
        // ===== RF Pulse Commands =====
        case CMD_PULSE:
            if (len >= 12) {
                float amplitude = unpack_float(payload);
                float phase = unpack_float(payload + 4);
                uint32_t duration = unpack_uint32(payload + 8);
                status = PulseControl_CustomPulse(amplitude, phase, duration, 0);
                resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            } else {
                resp = RESP_INVALID_PARAM;
            }
            break;
            
        case CMD_PULSE_PI2:
            if (len >= 8) {
                float amplitude = unpack_float(payload);
                float phase = unpack_float(payload + 4);
                status = PulseControl_Pi2Pulse(amplitude, phase);
                resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            } else {
                resp = RESP_INVALID_PARAM;
            }
            break;
            
        case CMD_PULSE_PI:
            if (len >= 8) {
                float amplitude = unpack_float(payload);
                float phase = unpack_float(payload + 4);
                status = PulseControl_PiPulse(amplitude, phase);
                resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            } else {
                resp = RESP_INVALID_PARAM;
            }
            break;
            
        case CMD_PULSE_ICOSA:
            if (len >= 12) {
                float amplitude = unpack_float(payload);
                uint32_t duration = unpack_uint32(payload + 4);
                uint32_t rotations = unpack_uint32(payload + 8);
                status = PulseControl_IcosahedralSequence(amplitude, duration, rotations);
                resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            } else {
                resp = RESP_INVALID_PARAM;
            }
            break;
            
        case CMD_PULSE_HEX:
            if (len >= 12) {
                float amplitude = unpack_float(payload);
                uint32_t duration = unpack_uint32(payload + 4);
                uint32_t cycles = unpack_uint32(payload + 8);
                status = PulseControl_HexagonalPattern(amplitude, duration, cycles);
                resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            } else {
                resp = RESP_INVALID_PARAM;
            }
            break;
            
        case CMD_PULSE_ABORT:
            status = PulseControl_AbortSequence();
            resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            break;
            
        // ===== Experiment Commands =====
        case CMD_EXP_FID:
            if (len >= 4) {
                float amplitude = unpack_float(payload);
                status = PulseControl_FID(amplitude);
                resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            } else {
                resp = RESP_INVALID_PARAM;
            }
            break;
            
        case CMD_EXP_ECHO:
            if (len >= 8) {
                float amplitude = unpack_float(payload);
                uint32_t echo_time = unpack_uint32(payload + 4);
                status = PulseControl_SpinEcho(amplitude, echo_time);
                resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            } else {
                resp = RESP_INVALID_PARAM;
            }
            break;
            
        case CMD_EXP_CPMG:
            if (len >= 12) {
                float amplitude = unpack_float(payload);
                uint32_t echo_time = unpack_uint32(payload + 4);
                uint32_t num_echoes = unpack_uint32(payload + 8);
                status = PulseControl_CPMG(amplitude, echo_time, num_echoes);
                resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            } else {
                resp = RESP_INVALID_PARAM;
            }
            break;
            
        case CMD_EXP_NUTATION:
            status = PulseControl_CalibrateB1();
            resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            break;
            
        // ===== Calibration Commands =====
        case CMD_CAL_B1:
            status = PulseControl_CalibrateB1();
            resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            break;
            
        case CMD_CAL_VERIFY_SPIN: {
            float magnitude;
            status = PulseControl_VerifySpinHalf(&magnitude);
            pack_float(resp_data, magnitude);
            pack_float(resp_data + 4, Z_CRITICAL);  // Expected value
            resp_data[8] = (status == HAL_OK) ? 1 : 0;  // Verified flag
            resp_len = 9;
            resp = RESP_OK;  // Always respond, even if verification failed
            break;
        }
        
        // ===== Data Commands =====
        case CMD_GET_FID: {
            // Return FID amplitude and phase
            pack_float(resp_data, PulseControl_GetLastFIDAmplitude());
            pack_float(resp_data + 4, PulseControl_GetLastFIDPhase());
            resp_len = 8;
            break;
        }
        
        case CMD_GET_SENSORS: {
            Sensor_Data_t sensors;
            HAL_Sensor_ReadAll(&sensors);
            pack_float(resp_data, sensors.temperature_c);
            pack_float(resp_data + 4, sensors.mag_field_t);
            pack_float(resp_data + 8, sensors.accel_x);
            pack_float(resp_data + 12, sensors.accel_y);
            pack_float(resp_data + 16, sensors.accel_z);
            resp_len = 20;
            break;
        }
        
        case CMD_GET_THRESHOLD: {
            ThresholdState_t state;
            ThresholdLogic_GetState(&state);
            resp_data[0] = (uint8_t)state.current_tier;
            resp_data[1] = (uint8_t)state.current_phase;
            resp_data[2] = state.available_operators;
            resp_data[3] = state.k_formation_active ? 1 : 0;
            pack_float(resp_data + 4, state.current_z);
            pack_float(resp_data + 8, state.delta_s_neg);
            pack_float(resp_data + 12, state.complexity);
            pack_float(resp_data + 16, state.k_formation_kappa);
            resp_len = 20;
            break;
        }
        
        case CMD_GET_PHYSICS: {
            // Return key physics values
            pack_float(resp_data, (float)PHI);
            pack_float(resp_data + 4, (float)PHI_INV);
            pack_float(resp_data + 8, (float)Z_CRITICAL);
            pack_float(resp_data + 12, (float)SIGMA);
            pack_float(resp_data + 16, RotorControl_GetZ());
            pack_float(resp_data + 20, RotorControl_GetDeltaSNeg());
            resp_len = 24;
            break;
        }
        
        // ===== Operator Commands =====
        case CMD_OP_CLOSURE:
            status = ThresholdLogic_ExecuteOperator(OP_CLOSURE);
            resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            break;
            
        case CMD_OP_FUSION:
            status = ThresholdLogic_ExecuteOperator(OP_FUSION);
            resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            break;
            
        case CMD_OP_AMPLIFY:
            status = ThresholdLogic_ExecuteOperator(OP_AMPLIFY);
            resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            break;
            
        case CMD_OP_DECOHERE:
            status = ThresholdLogic_ExecuteOperator(OP_DECOHERE);
            resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            break;
            
        case CMD_OP_GROUP:
            status = ThresholdLogic_ExecuteOperator(OP_GROUP);
            resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            break;
            
        case CMD_OP_SEPARATE:
            status = ThresholdLogic_ExecuteOperator(OP_SEPARATE);
            resp = (status == HAL_OK) ? RESP_OK : RESP_ERROR;
            break;
            
        // ===== Telemetry Control =====
        case CMD_TELEM_START:
            s_telemetry_enabled = true;
            break;
            
        case CMD_TELEM_STOP:
            s_telemetry_enabled = false;
            break;
            
        case CMD_TELEM_RATE:
            if (len >= 4) {
                s_telemetry_rate_ms = unpack_uint32(payload);
                if (s_telemetry_rate_ms < 1) s_telemetry_rate_ms = 1;
                if (s_telemetry_rate_ms > 1000) s_telemetry_rate_ms = 1000;
            } else {
                resp = RESP_INVALID_PARAM;
            }
            break;
            
        default:
            resp = RESP_INVALID_CMD;
            break;
    }
    
    // Send response
    return send_response(cmd, resp, resp_data, resp_len);
}


/* ============================================================================
 * TELEMETRY
 * ============================================================================ */

HAL_Status_t CommProtocol_SendTelemetry(void) {
    /**
     * Send periodic telemetry packet
     * 
     * Packet structure:
     * - Header (2 bytes): 0xAA55
     * - Type (1 byte): TELEM_PHYSICS
     * - Length (2 bytes): payload length
     * - Payload:
     *   - timestamp_ms (4 bytes)
     *   - z (4 bytes)
     *   - delta_s_neg (4 bytes)
     *   - complexity (4 bytes)
     *   - rpm (4 bytes)
     *   - tier (1 byte)
     *   - phase (1 byte)
     *   - operators (1 byte)
     *   - k_formation (1 byte)
     *   - kappa (4 bytes)
     *   - eta (4 bytes)
     * - Checksum (2 bytes)
     */
    
    uint8_t payload[40];
    uint32_t idx = 0;
    
    // Timestamp
    pack_uint32(&payload[idx], HAL_GetTick()); idx += 4;
    
    // Physics state
    pack_float(&payload[idx], RotorControl_GetZ()); idx += 4;
    pack_float(&payload[idx], RotorControl_GetDeltaSNeg()); idx += 4;
    pack_float(&payload[idx], RotorControl_GetComplexity()); idx += 4;
    pack_float(&payload[idx], RotorControl_GetRPM()); idx += 4;
    
    // Threshold state
    ThresholdState_t state;
    ThresholdLogic_GetState(&state);
    payload[idx++] = (uint8_t)state.current_tier;
    payload[idx++] = (uint8_t)state.current_phase;
    payload[idx++] = state.available_operators;
    payload[idx++] = state.k_formation_active ? 1 : 0;
    
    // K-formation parameters
    pack_float(&payload[idx], state.k_formation_kappa); idx += 4;
    pack_float(&payload[idx], state.k_formation_eta); idx += 4;
    
    // Build packet
    uint32_t tx_idx = 0;
    s_tx_buffer[tx_idx++] = 0xAA;
    s_tx_buffer[tx_idx++] = 0x55;
    s_tx_buffer[tx_idx++] = TELEM_PHYSICS;
    s_tx_buffer[tx_idx++] = (idx >> 8) & 0xFF;
    s_tx_buffer[tx_idx++] = idx & 0xFF;
    memcpy(&s_tx_buffer[tx_idx], payload, idx);
    tx_idx += idx;
    
    // Checksum
    uint16_t crc = compute_crc16(s_tx_buffer, tx_idx);
    s_tx_buffer[tx_idx++] = (crc >> 8) & 0xFF;
    s_tx_buffer[tx_idx++] = crc & 0xFF;
    
    // Send via UART/USB
    if (HAL_Comm_Write(s_tx_buffer, tx_idx) != HAL_OK) {
        return HAL_ERROR;
    }
    
    s_stats.packets_sent++;
    
    return HAL_OK;
}


HAL_Status_t CommProtocol_SendEvent(ThresholdEvent_t event, float value) {
    /**
     * Send event notification to host
     */
    
    uint8_t payload[9];
    payload[0] = (uint8_t)event;
    pack_float(&payload[1], value);
    pack_uint32(&payload[5], HAL_GetTick());
    
    uint32_t tx_idx = 0;
    s_tx_buffer[tx_idx++] = 0xAA;
    s_tx_buffer[tx_idx++] = 0x55;
    s_tx_buffer[tx_idx++] = TELEM_EVENT;
    s_tx_buffer[tx_idx++] = 0;
    s_tx_buffer[tx_idx++] = 9;  // payload length
    memcpy(&s_tx_buffer[tx_idx], payload, 9);
    tx_idx += 9;
    
    uint16_t crc = compute_crc16(s_tx_buffer, tx_idx);
    s_tx_buffer[tx_idx++] = (crc >> 8) & 0xFF;
    s_tx_buffer[tx_idx++] = crc & 0xFF;
    
    // Send
    if (HAL_Comm_Write(s_tx_buffer, tx_idx) != HAL_OK) {
        return HAL_ERROR;
    }
    
    return HAL_OK;
}


/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

static HAL_Status_t send_response(uint8_t cmd, ResponseCode_t resp,
                                   const uint8_t *data, uint32_t len) {
    uint32_t tx_idx = 0;
    
    s_tx_buffer[tx_idx++] = 0xAA;
    s_tx_buffer[tx_idx++] = 0x55;
    s_tx_buffer[tx_idx++] = cmd | 0x80;  // Response flag
    s_tx_buffer[tx_idx++] = resp;
    s_tx_buffer[tx_idx++] = len;
    
    if (len > 0 && data != NULL) {
        memcpy(&s_tx_buffer[tx_idx], data, len);
        tx_idx += len;
    }
    
    uint16_t crc = compute_crc16(s_tx_buffer, tx_idx);
    s_tx_buffer[tx_idx++] = (crc >> 8) & 0xFF;
    s_tx_buffer[tx_idx++] = crc & 0xFF;
    
    // Send via UART/USB
    if (HAL_Comm_Write(s_tx_buffer, tx_idx) != HAL_OK) {
        return HAL_ERROR;
    }
    
    s_stats.packets_sent++;
    
    return HAL_OK;
}


static uint16_t compute_crc16(const uint8_t *data, uint32_t len) {
    /**
     * CRC-16-CCITT (polynomial 0x1021)
     */
    uint16_t crc = 0xFFFF;
    
    for (uint32_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x8000) {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    
    return crc;
}


static void pack_float(uint8_t *buf, float value) {
    union { float f; uint32_t u; } conv;
    conv.f = value;
    buf[0] = (conv.u >> 24) & 0xFF;
    buf[1] = (conv.u >> 16) & 0xFF;
    buf[2] = (conv.u >> 8) & 0xFF;
    buf[3] = conv.u & 0xFF;
}


static float unpack_float(const uint8_t *buf) {
    union { float f; uint32_t u; } conv;
    conv.u = ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
             ((uint32_t)buf[2] << 8) | buf[3];
    return conv.f;
}


static void pack_uint32(uint8_t *buf, uint32_t value) {
    buf[0] = (value >> 24) & 0xFF;
    buf[1] = (value >> 16) & 0xFF;
    buf[2] = (value >> 8) & 0xFF;
    buf[3] = value & 0xFF;
}


static uint32_t unpack_uint32(const uint8_t *buf) {
    return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8) | buf[3];
}


/* ============================================================================
 * STATISTICS
 * ============================================================================ */

void CommProtocol_GetStats(CommStats_t *stats) {
    if (stats != NULL) {
        *stats = s_stats;
    }
}


void CommProtocol_ResetStats(void) {
    memset(&s_stats, 0, sizeof(s_stats));
}
