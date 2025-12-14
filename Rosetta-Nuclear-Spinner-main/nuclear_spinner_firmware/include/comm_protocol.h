/**
 * @file comm_protocol.h
 * @brief Host Communication Protocol Header
 * 
 * Public interface for host-device communication.
 * 
 * Signature: comm-protocol|v1.0.0|nuclear-spinner
 * 
 * @version 1.0.0
 */

#ifndef COMM_PROTOCOL_H
#define COMM_PROTOCOL_H

#include <stdint.h>
#include <stdbool.h>
#include "hal_hardware.h"
#include "threshold_logic.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * DATA TYPES
 * ============================================================================ */

/** Communication statistics */
typedef struct {
    uint32_t packets_received;
    uint32_t packets_sent;
    uint32_t crc_errors;
    uint32_t sync_errors;
    uint32_t timeouts;
} CommStats_t;


/* ============================================================================
 * FUNCTIONS
 * ============================================================================ */

/**
 * @brief Initialize communication protocol
 * @return HAL_OK on success
 */
HAL_Status_t CommProtocol_Init(void);

/**
 * @brief Process communication (call from main loop)
 * 
 * Handles incoming commands and sends telemetry
 * 
 * @return HAL_OK on success
 */
HAL_Status_t CommProtocol_Process(void);

/**
 * @brief Send telemetry packet
 * @return HAL_OK on success
 */
HAL_Status_t CommProtocol_SendTelemetry(void);

/**
 * @brief Send event notification to host
 * 
 * @param event Event type
 * @param value Associated value
 * @return HAL_OK on success
 */
HAL_Status_t CommProtocol_SendEvent(ThresholdEvent_t event, float value);

/**
 * @brief Get communication statistics
 * @param stats Output statistics structure
 */
void CommProtocol_GetStats(CommStats_t *stats);

/**
 * @brief Reset communication statistics
 */
void CommProtocol_ResetStats(void);


#ifdef __cplusplus
}
#endif

#endif /* COMM_PROTOCOL_H */
