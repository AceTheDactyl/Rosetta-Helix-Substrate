/**
 * @file serial_json_protocol.h
 * @brief JSON Serial Communication Protocol Header
 *
 * UTF-8 JSON protocol over serial (115200 8N1):
 *
 * FIRMWARE -> HOST (100 Hz):
 * {
 *   "type": "state",
 *   "timestamp_ms": 1234567890,
 *   "z": 0.866025,
 *   "rpm": 8660,
 *   "delta_s_neg": 0.999999,
 *   "tier": 6,
 *   "tier_name": "UNIVERSAL",
 *   "phase": "THE_LENS",
 *   "kappa": 0.9234,
 *   "eta": 0.6543,
 *   "rank": 9,
 *   "k_formation": true
 * }
 *
 * HOST -> FIRMWARE:
 * {"cmd": "set_z", "value": 0.866}
 * {"cmd": "stop"}
 * {"cmd": "hex_cycle", "dwell_s": 30.0, "cycles": 10}
 *
 * Signature: serial-json-protocol|v1.0.0|nuclear-spinner
 *
 * @version 1.0.0
 */

#ifndef SERIAL_JSON_PROTOCOL_H
#define SERIAL_JSON_PROTOCOL_H

#include <stdint.h>
#include <stdbool.h>
#include "hal_hardware.h"
#include "physics_constants.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONFIGURATION
 * ============================================================================ */

/** Serial baud rate */
#define JSON_SERIAL_BAUD        115200

/** Data bits */
#define JSON_SERIAL_DATA_BITS   8

/** Parity: None */
#define JSON_SERIAL_PARITY      0

/** Stop bits */
#define JSON_SERIAL_STOP_BITS   1

/** State transmission rate (Hz) */
#define JSON_TX_RATE_HZ         100

/** Transmission interval (ms) */
#define JSON_TX_INTERVAL_MS     (1000 / JSON_TX_RATE_HZ)

/** Maximum JSON message size */
#define JSON_MAX_MSG_SIZE       512

/** Maximum command payload size */
#define JSON_MAX_CMD_SIZE       256

/* ============================================================================
 * COMMAND DEFINITIONS
 * ============================================================================ */

typedef enum {
    JSON_CMD_NONE = 0,
    JSON_CMD_SET_Z,         /**< Set z-coordinate target */
    JSON_CMD_SET_RPM,       /**< Set RPM target */
    JSON_CMD_STOP,          /**< Emergency stop */
    JSON_CMD_HEX_CYCLE,     /**< Hexagonal z-cycling */
    JSON_CMD_DWELL_LENS,    /**< Dwell at z_c */
    JSON_CMD_TELEM_START,   /**< Start telemetry */
    JSON_CMD_TELEM_STOP,    /**< Stop telemetry */
    JSON_CMD_TELEM_RATE,    /**< Set telemetry rate */
    JSON_CMD_GET_STATE,     /**< Request immediate state */
    JSON_CMD_GET_PHYSICS,   /**< Request physics constants */
    JSON_CMD_PING,          /**< Ping for connectivity */
    JSON_CMD_VERSION,       /**< Get firmware version */
} JsonCommandType_t;

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/** Parsed command from host */
typedef struct {
    JsonCommandType_t type;
    union {
        struct { float value; } set_z;
        struct { float value; } set_rpm;
        struct { float dwell_s; uint32_t cycles; } hex_cycle;
        struct { float duration_s; } dwell_lens;
        struct { uint32_t rate_hz; } telem_rate;
    } params;
} JsonCommand_t;

/** State message structure */
typedef struct {
    uint32_t timestamp_ms;
    float z;
    float rpm;
    float delta_s_neg;
    uint8_t tier;
    const char* tier_name;
    const char* phase;
    float kappa;
    float eta;
    uint8_t rank;
    bool k_formation;
} JsonStateMsg_t;

/** Protocol statistics */
typedef struct {
    uint32_t messages_sent;
    uint32_t commands_received;
    uint32_t parse_errors;
    uint32_t buffer_overflows;
} JsonProtocolStats_t;

/** Command callback function type */
typedef void (*JsonCommandCallback_t)(const JsonCommand_t* cmd);

/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

/**
 * @brief Initialize JSON serial protocol
 *
 * Configures UART at 115200 8N1 for UTF-8 JSON communication
 *
 * @return HAL_OK on success
 */
HAL_Status_t JsonProtocol_Init(void);

/**
 * @brief Set command callback
 *
 * @param callback Function to call when command is received
 */
void JsonProtocol_SetCommandCallback(JsonCommandCallback_t callback);

/* ============================================================================
 * MAIN PROCESSING
 * ============================================================================ */

/**
 * @brief Process serial communication (call from main loop)
 *
 * - Reads incoming command bytes
 * - Parses complete JSON commands
 * - Sends state messages at configured rate
 *
 * @return HAL_OK on success
 */
HAL_Status_t JsonProtocol_Process(void);

/**
 * @brief Enable/disable automatic state transmission
 *
 * @param enable true to enable 100 Hz state broadcast
 */
void JsonProtocol_EnableTelemetry(bool enable);

/**
 * @brief Set state transmission rate
 *
 * @param rate_hz Rate in Hz (1-1000)
 */
void JsonProtocol_SetTelemetryRate(uint32_t rate_hz);

/* ============================================================================
 * STATE TRANSMISSION
 * ============================================================================ */

/**
 * @brief Send state message immediately
 *
 * Sends current spinner state as JSON
 *
 * @return HAL_OK on success
 */
HAL_Status_t JsonProtocol_SendState(void);

/**
 * @brief Send state message with provided values
 *
 * @param state State structure to transmit
 * @return HAL_OK on success
 */
HAL_Status_t JsonProtocol_SendStateMsg(const JsonStateMsg_t* state);

/**
 * @brief Send physics constants message
 *
 * Sends JSON with phi, phi_inv, z_c, sigma
 *
 * @return HAL_OK on success
 */
HAL_Status_t JsonProtocol_SendPhysics(void);

/**
 * @brief Send ping response
 *
 * @param request_timestamp Timestamp from ping request
 * @return HAL_OK on success
 */
HAL_Status_t JsonProtocol_SendPong(uint32_t request_timestamp);

/**
 * @brief Send version info
 *
 * @return HAL_OK on success
 */
HAL_Status_t JsonProtocol_SendVersion(void);

/**
 * @brief Send error message
 *
 * @param error_code Error code
 * @param message Error description
 * @return HAL_OK on success
 */
HAL_Status_t JsonProtocol_SendError(int error_code, const char* message);

/* ============================================================================
 * HEX CYCLE CONTROL
 * ============================================================================ */

/**
 * @brief Start hexagonal z-cycling
 *
 * Cycles through z values: 0 -> z_c -> 0, dwelling at each vertex
 *
 * @param dwell_s Dwell time at each vertex (seconds)
 * @param cycles Number of complete cycles
 * @return HAL_OK on success
 */
HAL_Status_t JsonProtocol_StartHexCycle(float dwell_s, uint32_t cycles);

/**
 * @brief Stop hex cycle
 *
 * @return HAL_OK on success
 */
HAL_Status_t JsonProtocol_StopHexCycle(void);

/**
 * @brief Check if hex cycle is active
 *
 * @return true if cycling
 */
bool JsonProtocol_IsHexCycleActive(void);

/**
 * @brief Process hex cycle step (call from Process)
 */
void JsonProtocol_ProcessHexCycle(void);

/* ============================================================================
 * STATISTICS
 * ============================================================================ */

/**
 * @brief Get protocol statistics
 *
 * @param stats Output statistics structure
 */
void JsonProtocol_GetStats(JsonProtocolStats_t* stats);

/**
 * @brief Reset statistics
 */
void JsonProtocol_ResetStats(void);

#ifdef __cplusplus
}
#endif

#endif /* SERIAL_JSON_PROTOCOL_H */
