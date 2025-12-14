/**
 * @file hal_hardware_sim.c
 * @brief Host-side simulation HAL implementation.
 *
 * Provides the same HAL_* API as the STM32 implementation, but uses
 * in-memory state only so it can run on a desktop.
 */

#include "hal_hardware.h"
#include "physics_constants.h"
#include <string.h>
#include <math.h>
#include <time.h>

static uint32_t s_tick_ms = 0;
static Motor_State_t s_motor = {0};
static uint32_t s_enc = 0;
static uint32_t s_last_enc_tick = 0;
static bool s_interlock = true;

/* Simulation parameters */
#define SIM_ENCODER_CPR 4096
#define SIM_RPM_MAX 10000.0f

HAL_Status_t HAL_Init_All(void) { s_last_enc_tick = 0; return HAL_OK; }
HAL_Status_t HAL_Init_Clocks(void) { return HAL_OK; }
HAL_Status_t HAL_Init_GPIO(void) { return HAL_OK; }
HAL_Status_t HAL_Init_Timers(void) { return HAL_OK; }
HAL_Status_t HAL_Init_DAC(void) { return HAL_OK; }
HAL_Status_t HAL_Init_ADC(void) { return HAL_OK; }
HAL_Status_t HAL_Init_I2C(void) { return HAL_OK; }
HAL_Status_t HAL_Init_DMA(void) { return HAL_OK; }

void HAL_RF_Enable(bool enable) { (void)enable; }
HAL_Status_t HAL_RF_SetAmplitude(float amplitude) { (void)amplitude; return HAL_OK; }
HAL_Status_t HAL_RF_SetPhase(float phase) { (void)phase; return HAL_OK; }
HAL_Status_t HAL_RF_ExecutePulse(const RF_Pulse_t *pulse) { (void)pulse; return HAL_OK; }
HAL_Status_t HAL_RF_ExecuteSequence(const RF_Pulse_t *pulses, uint32_t count) { (void)pulses; (void)count; return HAL_OK; }
HAL_Status_t HAL_RF_ConfigureTimer(uint32_t duration_us) { (void)duration_us; return HAL_OK; }

void HAL_Motor_Enable(bool enable) { 
    s_motor.enabled = enable; 
    if (enable && s_last_enc_tick == 0) {
        s_last_enc_tick = s_tick_ms;
    }
}
HAL_Status_t HAL_Motor_SetDuty(float duty) {
    if (duty < 0.0f) duty = 0.0f;
    if (duty > 1.0f) duty = 1.0f;
    s_motor.duty_cycle = duty;
    // simple mapping: duty -> rpm
    s_motor.actual_rpm = duty * SIM_RPM_MAX;
    return HAL_OK;
}
void HAL_Motor_SetDirection(bool clockwise) { (void)clockwise; }
HAL_Status_t HAL_Motor_GetState(Motor_State_t *state) { if (!state) return HAL_INVALID_PARAM; *state = s_motor; return HAL_OK; }
uint32_t HAL_Motor_GetEncoderCount(void) {
    /* Simulate encoder counts based on elapsed time and RPM */
    uint32_t now = s_tick_ms;
    if (now > s_last_enc_tick && s_motor.enabled) {
        uint32_t dt = now - s_last_enc_tick;
        /* counts = rpm * CPR * dt_ms / 60000 */
        float counts_per_ms = s_motor.actual_rpm * SIM_ENCODER_CPR / 60000.0f;
        s_enc += (uint32_t)(counts_per_ms * dt);
    }
    s_last_enc_tick = now;
    return s_enc;
}
void HAL_Motor_ResetEncoder(void) { s_enc = 0; }
bool HAL_Motor_IsFault(void) { return s_motor.fault; }
void HAL_Motor_ClearFault(void) { s_motor.fault = false; }

HAL_Status_t HAL_ADC_StartFID(ADC_Buffer_t *buffer) {
    if (!buffer || !buffer->buffer || buffer->size == 0) return HAL_INVALID_PARAM;
    // Fill with a synthetic decaying sinusoid (12-bit raw), immediate completion
    for (uint32_t i = 0; i < buffer->size; i++) {
        float t = (float)i / (float)ADC_SAMPLE_RATE;
        float v = 0.5f + 0.45f * expf(-t * 2000.0f) * sinf(2.0f * 3.1415926f * 25000.0f * t);
        uint16_t raw = (uint16_t)(v * 4095.0f);
        buffer->buffer[i] = raw;
    }
    buffer->index = buffer->size;
    buffer->complete = true;
    return HAL_OK;
}
HAL_Status_t HAL_ADC_StopFID(void) { return HAL_OK; }
uint16_t HAL_ADC_ReadSingle(uint8_t channel) { (void)channel; return 2048; }
float HAL_ADC_ToVoltage(uint16_t raw) { return (3.3f * raw) / 4095.0f; }
float HAL_ADC_ReadTemperature(void) { return 25.0f; }

HAL_Status_t HAL_Sensor_ReadAll(Sensor_Data_t *data) {
    if (!data) return HAL_INVALID_PARAM;
    memset(data, 0, sizeof(*data));
    data->temperature_c = 25.0f;
    data->timestamp_ms = HAL_GetTick();
    return HAL_OK;
}
HAL_Status_t HAL_Sensor_ReadMag(float *x, float *y, float *z) { if (x) *x = 0; if (y) *y = 0; if (z) *z = 0; return HAL_OK; }
HAL_Status_t HAL_Sensor_ReadIMU(float accel[3], float gyro[3]) {
    if (accel) accel[0]=accel[1]=0, accel[2]=9.81f;
    if (gyro) gyro[0]=gyro[1]=gyro[2]=0;
    return HAL_OK;
}
float HAL_Sensor_ReadPT100(void) { return 25.0f; }

bool HAL_Safety_InterlockOK(void) { return s_interlock; }
void HAL_Safety_EmergencyStop(void) { HAL_Motor_Enable(false); HAL_Motor_SetDuty(0.0f); }
void HAL_LED_Set(uint8_t led, bool state) { (void)led; (void)state; }
void HAL_LED_Toggle(uint8_t led) { (void)led; }

uint32_t HAL_GetTick(void) { return s_tick_ms; }
void HAL_Delay(uint32_t ms) { s_tick_ms += ms; }
void HAL_DelayMicroseconds(uint32_t us) { s_tick_ms += (us + 999) / 1000; }
void HAL_Timer_StartProfile(void) {}
uint32_t HAL_Timer_GetElapsed(void) { return 0; }

// Optional callbacks (modules may provide their own strong definitions)
void HAL_Callback_FID_HalfComplete(ADC_Buffer_t *buffer) { (void)buffer; }
void HAL_Callback_InterlockChange(bool engaged) { (void)engaged; }


HAL_Status_t HAL_Comm_Init(void) { return HAL_OK; }

uint32_t HAL_Comm_Read(uint8_t *dst, uint32_t max_len) {
    (void)dst; (void)max_len;
    return 0;
}

HAL_Status_t HAL_Comm_Write(const uint8_t *src, uint32_t len) {
    // In sim, drop bytes on the floor (or route to stdout if desired).
    (void)src; (void)len;
    return HAL_OK;
}
