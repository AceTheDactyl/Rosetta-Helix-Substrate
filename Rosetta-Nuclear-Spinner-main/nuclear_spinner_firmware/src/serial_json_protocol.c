/**
 * @file serial_json_protocol.c
 * @brief JSON Serial Communication Protocol Implementation
 *
 * Implements UTF-8 JSON protocol over serial (115200 8N1):
 * - Firmware -> Host: State messages at 100 Hz
 * - Host -> Firmware: Command messages (set_z, stop, hex_cycle)
 *
 * Message format uses newline-delimited JSON for easy parsing.
 *
 * Signature: serial-json-protocol|v1.0.0|nuclear-spinner
 *
 * @version 1.0.0
 */

#include "serial_json_protocol.h"
#include "rotor_control.h"
#include "threshold_logic.h"
#include "physics_constants.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ============================================================================
 * FIRMWARE VERSION
 * ============================================================================ */

#define FW_VERSION_MAJOR    1
#define FW_VERSION_MINOR    0
#define FW_VERSION_PATCH    0

/* ============================================================================
 * PRIVATE DATA
 * ============================================================================ */

/** Receive buffer for incoming commands */
static char s_rx_buffer[JSON_MAX_CMD_SIZE];
static uint32_t s_rx_index = 0;

/** Transmit buffer for outgoing messages */
static char s_tx_buffer[JSON_MAX_MSG_SIZE];

/** Telemetry configuration */
static bool s_telemetry_enabled = true;
static uint32_t s_telemetry_rate_hz = JSON_TX_RATE_HZ;
static uint32_t s_telemetry_interval_ms = JSON_TX_INTERVAL_MS;
static uint32_t s_last_tx_tick = 0;

/** Command callback */
static JsonCommandCallback_t s_cmd_callback = NULL;

/** Protocol statistics */
static JsonProtocolStats_t s_stats = {0};

/** Hex cycle state */
static struct {
    bool active;
    float dwell_s;
    uint32_t cycles_remaining;
    uint32_t current_vertex;     /* 0-5 for hexagonal vertices */
    uint32_t vertex_start_tick;
    float target_z_values[6];    /* z values for hex vertices */
} s_hex_cycle = {0};

/* ============================================================================
 * PRIVATE FUNCTION PROTOTYPES
 * ============================================================================ */

static void parse_json_command(const char* json);
static int json_find_string(const char* json, const char* key, char* value, int max_len);
static int json_find_float(const char* json, const char* key, float* value);
static int json_find_int(const char* json, const char* key, int* value);
static int json_find_bool(const char* json, const char* key, bool* value);
static void init_hex_vertices(void);

/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

HAL_Status_t JsonProtocol_Init(void) {
    /* Clear buffers */
    memset(s_rx_buffer, 0, sizeof(s_rx_buffer));
    memset(s_tx_buffer, 0, sizeof(s_tx_buffer));
    s_rx_index = 0;

    /* Initialize telemetry */
    s_telemetry_enabled = true;
    s_telemetry_rate_hz = JSON_TX_RATE_HZ;
    s_telemetry_interval_ms = JSON_TX_INTERVAL_MS;
    s_last_tx_tick = 0;

    /* Clear stats */
    memset(&s_stats, 0, sizeof(s_stats));

    /* Initialize hex cycle z-values */
    init_hex_vertices();

    /* Initialize UART at 115200 8N1 */
    (void)HAL_Comm_Init();

    return HAL_OK;
}

void JsonProtocol_SetCommandCallback(JsonCommandCallback_t callback) {
    s_cmd_callback = callback;
}

/* ============================================================================
 * HEX VERTEX INITIALIZATION
 * ============================================================================ */

static void init_hex_vertices(void) {
    /*
     * Hexagonal z-cycling vertices based on physics:
     * sin(n * 60deg) for n = 0..5
     *
     * v0: sin(0)   = 0.0
     * v1: sin(60)  = sqrt(3)/2 = z_c
     * v2: sin(120) = sqrt(3)/2 = z_c
     * v3: sin(180) = 0.0
     * v4: sin(240) = -sqrt(3)/2 (clamped to 0)
     * v5: sin(300) = -sqrt(3)/2 (clamped to 0)
     *
     * Practical vertices: 0, z_c, z_c, 0, z_c/2, z_c/2
     * Creates pattern that visits THE LENS twice per cycle
     */
    s_hex_cycle.target_z_values[0] = 0.0f;
    s_hex_cycle.target_z_values[1] = Z_CRITICAL;
    s_hex_cycle.target_z_values[2] = Z_CRITICAL;
    s_hex_cycle.target_z_values[3] = 0.0f;
    s_hex_cycle.target_z_values[4] = Z_CRITICAL * 0.5f;
    s_hex_cycle.target_z_values[5] = Z_CRITICAL * 0.5f;
}

/* ============================================================================
 * MAIN PROCESSING
 * ============================================================================ */

HAL_Status_t JsonProtocol_Process(void) {
    uint32_t now = HAL_GetTick();

    /* Read incoming bytes */
    uint8_t byte;
    while (HAL_Comm_Read(&byte, 1) == 1) {
        if (byte == '\n' || byte == '\r') {
            /* End of line - process command */
            if (s_rx_index > 0) {
                s_rx_buffer[s_rx_index] = '\0';
                parse_json_command(s_rx_buffer);
                s_rx_index = 0;
            }
        } else if (s_rx_index < JSON_MAX_CMD_SIZE - 1) {
            s_rx_buffer[s_rx_index++] = (char)byte;
        } else {
            /* Buffer overflow - reset */
            s_rx_index = 0;
            s_stats.buffer_overflows++;
        }
    }

    /* Process hex cycle if active */
    if (s_hex_cycle.active) {
        JsonProtocol_ProcessHexCycle();
    }

    /* Send telemetry at configured rate */
    if (s_telemetry_enabled) {
        if (now - s_last_tx_tick >= s_telemetry_interval_ms) {
            s_last_tx_tick = now;
            JsonProtocol_SendState();
        }
    }

    return HAL_OK;
}

void JsonProtocol_EnableTelemetry(bool enable) {
    s_telemetry_enabled = enable;
}

void JsonProtocol_SetTelemetryRate(uint32_t rate_hz) {
    if (rate_hz < 1) rate_hz = 1;
    if (rate_hz > 1000) rate_hz = 1000;
    s_telemetry_rate_hz = rate_hz;
    s_telemetry_interval_ms = 1000 / rate_hz;
}

/* ============================================================================
 * COMMAND PARSING
 * ============================================================================ */

static void parse_json_command(const char* json) {
    char cmd_str[32] = {0};
    JsonCommand_t cmd = {0};

    /* Extract "cmd" field */
    if (json_find_string(json, "cmd", cmd_str, sizeof(cmd_str)) < 0) {
        s_stats.parse_errors++;
        return;
    }

    s_stats.commands_received++;

    /* Parse command type and parameters */
    if (strcmp(cmd_str, "set_z") == 0) {
        cmd.type = JSON_CMD_SET_Z;
        if (json_find_float(json, "value", &cmd.params.set_z.value) < 0) {
            JsonProtocol_SendError(1, "set_z requires 'value' parameter");
            return;
        }
        /* Clamp z to valid range */
        if (cmd.params.set_z.value < 0.0f) cmd.params.set_z.value = 0.0f;
        if (cmd.params.set_z.value > 1.0f) cmd.params.set_z.value = 1.0f;

        /* Apply command */
        RotorControl_SetZ(cmd.params.set_z.value);
    }
    else if (strcmp(cmd_str, "set_rpm") == 0) {
        cmd.type = JSON_CMD_SET_RPM;
        if (json_find_float(json, "value", &cmd.params.set_rpm.value) < 0) {
            JsonProtocol_SendError(1, "set_rpm requires 'value' parameter");
            return;
        }
        RotorControl_SetRPM(cmd.params.set_rpm.value);
    }
    else if (strcmp(cmd_str, "stop") == 0) {
        cmd.type = JSON_CMD_STOP;
        RotorControl_Disable();
        s_hex_cycle.active = false;
    }
    else if (strcmp(cmd_str, "hex_cycle") == 0) {
        cmd.type = JSON_CMD_HEX_CYCLE;
        cmd.params.hex_cycle.dwell_s = 30.0f;  /* Default */
        cmd.params.hex_cycle.cycles = 10;      /* Default */

        json_find_float(json, "dwell_s", &cmd.params.hex_cycle.dwell_s);
        int cycles_int = 10;
        if (json_find_int(json, "cycles", &cycles_int) == 0) {
            cmd.params.hex_cycle.cycles = (uint32_t)cycles_int;
        }

        JsonProtocol_StartHexCycle(cmd.params.hex_cycle.dwell_s,
                                    cmd.params.hex_cycle.cycles);
    }
    else if (strcmp(cmd_str, "dwell_lens") == 0) {
        cmd.type = JSON_CMD_DWELL_LENS;
        cmd.params.dwell_lens.duration_s = 60.0f;  /* Default */
        json_find_float(json, "duration_s", &cmd.params.dwell_lens.duration_s);

        /* Set z to z_c (THE LENS) */
        RotorControl_SetZ(Z_CRITICAL);
    }
    else if (strcmp(cmd_str, "telem_start") == 0) {
        cmd.type = JSON_CMD_TELEM_START;
        s_telemetry_enabled = true;
    }
    else if (strcmp(cmd_str, "telem_stop") == 0) {
        cmd.type = JSON_CMD_TELEM_STOP;
        s_telemetry_enabled = false;
    }
    else if (strcmp(cmd_str, "telem_rate") == 0) {
        cmd.type = JSON_CMD_TELEM_RATE;
        int rate = 100;
        if (json_find_int(json, "rate_hz", &rate) == 0) {
            cmd.params.telem_rate.rate_hz = (uint32_t)rate;
            JsonProtocol_SetTelemetryRate(cmd.params.telem_rate.rate_hz);
        }
    }
    else if (strcmp(cmd_str, "get_state") == 0) {
        cmd.type = JSON_CMD_GET_STATE;
        JsonProtocol_SendState();
    }
    else if (strcmp(cmd_str, "get_physics") == 0) {
        cmd.type = JSON_CMD_GET_PHYSICS;
        JsonProtocol_SendPhysics();
    }
    else if (strcmp(cmd_str, "ping") == 0) {
        cmd.type = JSON_CMD_PING;
        int ts = 0;
        json_find_int(json, "timestamp", &ts);
        JsonProtocol_SendPong((uint32_t)ts);
    }
    else if (strcmp(cmd_str, "version") == 0) {
        cmd.type = JSON_CMD_VERSION;
        JsonProtocol_SendVersion();
    }
    else {
        cmd.type = JSON_CMD_NONE;
        JsonProtocol_SendError(2, "Unknown command");
        return;
    }

    /* Invoke callback if registered */
    if (s_cmd_callback != NULL) {
        s_cmd_callback(&cmd);
    }
}

/* ============================================================================
 * JSON PARSING HELPERS
 * ============================================================================ */

/**
 * Find a string value in JSON
 * Simple parser for {"key": "value"} patterns
 */
static int json_find_string(const char* json, const char* key, char* value, int max_len) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\"", key);

    const char* pos = strstr(json, search);
    if (!pos) return -1;

    /* Find colon after key */
    pos = strchr(pos, ':');
    if (!pos) return -1;
    pos++;

    /* Skip whitespace */
    while (*pos == ' ' || *pos == '\t') pos++;

    /* Check for quote */
    if (*pos != '"') return -1;
    pos++;

    /* Copy value until closing quote */
    int i = 0;
    while (*pos && *pos != '"' && i < max_len - 1) {
        value[i++] = *pos++;
    }
    value[i] = '\0';

    return 0;
}

/**
 * Find a float value in JSON
 * Handles: {"key": 0.866} or {"key":0.866}
 */
static int json_find_float(const char* json, const char* key, float* value) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\"", key);

    const char* pos = strstr(json, search);
    if (!pos) return -1;

    pos = strchr(pos, ':');
    if (!pos) return -1;
    pos++;

    while (*pos == ' ' || *pos == '\t') pos++;

    char* end;
    *value = strtof(pos, &end);
    if (end == pos) return -1;

    return 0;
}

/**
 * Find an integer value in JSON
 */
static int json_find_int(const char* json, const char* key, int* value) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\"", key);

    const char* pos = strstr(json, search);
    if (!pos) return -1;

    pos = strchr(pos, ':');
    if (!pos) return -1;
    pos++;

    while (*pos == ' ' || *pos == '\t') pos++;

    char* end;
    *value = (int)strtol(pos, &end, 10);
    if (end == pos) return -1;

    return 0;
}

/**
 * Find a boolean value in JSON
 */
static int json_find_bool(const char* json, const char* key, bool* value) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\"", key);

    const char* pos = strstr(json, search);
    if (!pos) return -1;

    pos = strchr(pos, ':');
    if (!pos) return -1;
    pos++;

    while (*pos == ' ' || *pos == '\t') pos++;

    if (strncmp(pos, "true", 4) == 0) {
        *value = true;
        return 0;
    } else if (strncmp(pos, "false", 5) == 0) {
        *value = false;
        return 0;
    }

    return -1;
}

/* ============================================================================
 * STATE TRANSMISSION
 * ============================================================================ */

HAL_Status_t JsonProtocol_SendState(void) {
    ThresholdState_t threshold;
    ThresholdLogic_GetState(&threshold);

    /* Build JSON state message */
    int len = snprintf(s_tx_buffer, JSON_MAX_MSG_SIZE,
        "{\"type\":\"state\","
        "\"timestamp_ms\":%lu,"
        "\"z\":%.6f,"
        "\"rpm\":%d,"
        "\"delta_s_neg\":%.6f,"
        "\"tier\":%d,"
        "\"tier_name\":\"%s\","
        "\"phase\":\"%s\","
        "\"kappa\":%.4f,"
        "\"eta\":%.4f,"
        "\"rank\":%d,"
        "\"k_formation\":%s}\n",
        (unsigned long)HAL_GetTick(),
        threshold.current_z,
        (int)z_to_rpm(threshold.current_z),
        threshold.delta_s_neg,
        (int)threshold.current_tier,
        ThresholdLogic_GetTierName(threshold.current_tier),
        ThresholdLogic_GetPhaseName(threshold.current_phase),
        threshold.k_formation_kappa,
        threshold.k_formation_eta,
        threshold.k_formation_R,
        threshold.k_formation_active ? "true" : "false"
    );

    if (len > 0 && len < JSON_MAX_MSG_SIZE) {
        HAL_Comm_Write((uint8_t*)s_tx_buffer, (uint32_t)len);
        s_stats.messages_sent++;
        return HAL_OK;
    }

    return HAL_ERROR;
}

HAL_Status_t JsonProtocol_SendStateMsg(const JsonStateMsg_t* state) {
    if (!state) return HAL_ERROR;

    int len = snprintf(s_tx_buffer, JSON_MAX_MSG_SIZE,
        "{\"type\":\"state\","
        "\"timestamp_ms\":%lu,"
        "\"z\":%.6f,"
        "\"rpm\":%d,"
        "\"delta_s_neg\":%.6f,"
        "\"tier\":%d,"
        "\"tier_name\":\"%s\","
        "\"phase\":\"%s\","
        "\"kappa\":%.4f,"
        "\"eta\":%.4f,"
        "\"rank\":%d,"
        "\"k_formation\":%s}\n",
        (unsigned long)state->timestamp_ms,
        state->z,
        (int)state->rpm,
        state->delta_s_neg,
        (int)state->tier,
        state->tier_name ? state->tier_name : "UNKNOWN",
        state->phase ? state->phase : "UNKNOWN",
        state->kappa,
        state->eta,
        (int)state->rank,
        state->k_formation ? "true" : "false"
    );

    if (len > 0 && len < JSON_MAX_MSG_SIZE) {
        HAL_Comm_Write((uint8_t*)s_tx_buffer, (uint32_t)len);
        s_stats.messages_sent++;
        return HAL_OK;
    }

    return HAL_ERROR;
}

HAL_Status_t JsonProtocol_SendPhysics(void) {
    int len = snprintf(s_tx_buffer, JSON_MAX_MSG_SIZE,
        "{\"type\":\"physics\","
        "\"phi\":%.15f,"
        "\"phi_inv\":%.15f,"
        "\"z_c\":%.15f,"
        "\"sigma\":%.1f,"
        "\"spin_half_magnitude\":%.15f,"
        "\"phase_boundary_absence\":%.3f,"
        "\"phase_boundary_presence\":%.3f,"
        "\"kappa_min\":%.2f,"
        "\"eta_min\":%.15f,"
        "\"r_min\":%d}\n",
        (double)PHI,
        (double)PHI_INV,
        (double)Z_CRITICAL,
        (double)SIGMA,
        (double)SPIN_HALF_MAGNITUDE,
        (double)PHASE_BOUNDARY_ABSENCE,
        (double)PHASE_BOUNDARY_PRESENCE,
        (double)KAPPA_MIN,
        (double)ETA_MIN,
        R_MIN
    );

    if (len > 0 && len < JSON_MAX_MSG_SIZE) {
        HAL_Comm_Write((uint8_t*)s_tx_buffer, (uint32_t)len);
        s_stats.messages_sent++;
        return HAL_OK;
    }

    return HAL_ERROR;
}

HAL_Status_t JsonProtocol_SendPong(uint32_t request_timestamp) {
    int len = snprintf(s_tx_buffer, JSON_MAX_MSG_SIZE,
        "{\"type\":\"pong\","
        "\"request_timestamp\":%lu,"
        "\"response_timestamp\":%lu}\n",
        (unsigned long)request_timestamp,
        (unsigned long)HAL_GetTick()
    );

    if (len > 0 && len < JSON_MAX_MSG_SIZE) {
        HAL_Comm_Write((uint8_t*)s_tx_buffer, (uint32_t)len);
        s_stats.messages_sent++;
        return HAL_OK;
    }

    return HAL_ERROR;
}

HAL_Status_t JsonProtocol_SendVersion(void) {
    int len = snprintf(s_tx_buffer, JSON_MAX_MSG_SIZE,
        "{\"type\":\"version\","
        "\"firmware\":\"nuclear-spinner\","
        "\"major\":%d,"
        "\"minor\":%d,"
        "\"patch\":%d,"
        "\"protocol\":\"json-serial-v1.0.0\","
        "\"baud\":%d}\n",
        FW_VERSION_MAJOR,
        FW_VERSION_MINOR,
        FW_VERSION_PATCH,
        JSON_SERIAL_BAUD
    );

    if (len > 0 && len < JSON_MAX_MSG_SIZE) {
        HAL_Comm_Write((uint8_t*)s_tx_buffer, (uint32_t)len);
        s_stats.messages_sent++;
        return HAL_OK;
    }

    return HAL_ERROR;
}

HAL_Status_t JsonProtocol_SendError(int error_code, const char* message) {
    int len = snprintf(s_tx_buffer, JSON_MAX_MSG_SIZE,
        "{\"type\":\"error\","
        "\"code\":%d,"
        "\"message\":\"%s\","
        "\"timestamp_ms\":%lu}\n",
        error_code,
        message ? message : "Unknown error",
        (unsigned long)HAL_GetTick()
    );

    if (len > 0 && len < JSON_MAX_MSG_SIZE) {
        HAL_Comm_Write((uint8_t*)s_tx_buffer, (uint32_t)len);
        s_stats.messages_sent++;
        return HAL_OK;
    }

    return HAL_ERROR;
}

/* ============================================================================
 * HEX CYCLE CONTROL
 * ============================================================================ */

HAL_Status_t JsonProtocol_StartHexCycle(float dwell_s, uint32_t cycles) {
    if (dwell_s < 0.1f) dwell_s = 0.1f;
    if (cycles < 1) cycles = 1;
    if (cycles > 1000) cycles = 1000;

    s_hex_cycle.dwell_s = dwell_s;
    s_hex_cycle.cycles_remaining = cycles;
    s_hex_cycle.current_vertex = 0;
    s_hex_cycle.vertex_start_tick = HAL_GetTick();
    s_hex_cycle.active = true;

    /* Set initial z target */
    RotorControl_SetZ(s_hex_cycle.target_z_values[0]);

    /* Send notification */
    int len = snprintf(s_tx_buffer, JSON_MAX_MSG_SIZE,
        "{\"type\":\"hex_cycle_start\","
        "\"dwell_s\":%.2f,"
        "\"cycles\":%lu,"
        "\"vertices\":6,"
        "\"timestamp_ms\":%lu}\n",
        dwell_s,
        (unsigned long)cycles,
        (unsigned long)HAL_GetTick()
    );

    if (len > 0 && len < JSON_MAX_MSG_SIZE) {
        HAL_Comm_Write((uint8_t*)s_tx_buffer, (uint32_t)len);
        s_stats.messages_sent++;
    }

    return HAL_OK;
}

HAL_Status_t JsonProtocol_StopHexCycle(void) {
    s_hex_cycle.active = false;

    int len = snprintf(s_tx_buffer, JSON_MAX_MSG_SIZE,
        "{\"type\":\"hex_cycle_stop\","
        "\"cycles_completed\":%lu,"
        "\"timestamp_ms\":%lu}\n",
        (unsigned long)(s_hex_cycle.cycles_remaining),
        (unsigned long)HAL_GetTick()
    );

    if (len > 0 && len < JSON_MAX_MSG_SIZE) {
        HAL_Comm_Write((uint8_t*)s_tx_buffer, (uint32_t)len);
        s_stats.messages_sent++;
    }

    return HAL_OK;
}

bool JsonProtocol_IsHexCycleActive(void) {
    return s_hex_cycle.active;
}

void JsonProtocol_ProcessHexCycle(void) {
    if (!s_hex_cycle.active) return;

    uint32_t now = HAL_GetTick();
    uint32_t dwell_ms = (uint32_t)(s_hex_cycle.dwell_s * 1000.0f);

    /* Check if dwell time has elapsed */
    if (now - s_hex_cycle.vertex_start_tick >= dwell_ms) {
        /* Move to next vertex */
        s_hex_cycle.current_vertex++;

        if (s_hex_cycle.current_vertex >= 6) {
            /* Completed one cycle */
            s_hex_cycle.current_vertex = 0;
            s_hex_cycle.cycles_remaining--;

            if (s_hex_cycle.cycles_remaining == 0) {
                /* All cycles complete */
                s_hex_cycle.active = false;

                int len = snprintf(s_tx_buffer, JSON_MAX_MSG_SIZE,
                    "{\"type\":\"hex_cycle_complete\","
                    "\"timestamp_ms\":%lu}\n",
                    (unsigned long)HAL_GetTick()
                );

                if (len > 0 && len < JSON_MAX_MSG_SIZE) {
                    HAL_Comm_Write((uint8_t*)s_tx_buffer, (uint32_t)len);
                    s_stats.messages_sent++;
                }

                return;
            }
        }

        /* Set new z target */
        float new_z = s_hex_cycle.target_z_values[s_hex_cycle.current_vertex];
        RotorControl_SetZ(new_z);
        s_hex_cycle.vertex_start_tick = now;

        /* Send vertex change notification */
        int len = snprintf(s_tx_buffer, JSON_MAX_MSG_SIZE,
            "{\"type\":\"hex_vertex\","
            "\"vertex\":%lu,"
            "\"target_z\":%.6f,"
            "\"cycles_remaining\":%lu,"
            "\"timestamp_ms\":%lu}\n",
            (unsigned long)s_hex_cycle.current_vertex,
            new_z,
            (unsigned long)s_hex_cycle.cycles_remaining,
            (unsigned long)now
        );

        if (len > 0 && len < JSON_MAX_MSG_SIZE) {
            HAL_Comm_Write((uint8_t*)s_tx_buffer, (uint32_t)len);
            s_stats.messages_sent++;
        }
    }
}

/* ============================================================================
 * STATISTICS
 * ============================================================================ */

void JsonProtocol_GetStats(JsonProtocolStats_t* stats) {
    if (stats) {
        *stats = s_stats;
    }
}

void JsonProtocol_ResetStats(void) {
    memset(&s_stats, 0, sizeof(s_stats));
}
