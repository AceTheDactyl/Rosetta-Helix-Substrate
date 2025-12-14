/**
 * @file hal_hardware.h
 * @brief Hardware Abstraction Layer for Nuclear Spinner
 * 
 * Abstracts STM32H7 peripherals for:
 * - Timer-based RF pulse generation
 * - DAC output for amplitude/phase control
 * - ADC input for FID signal acquisition
 * - Motor driver interface for rotor control
 * - Sensor interfaces (temperature, magnetometer, accelerometer)
 * 
 * Target: STM32H743ZI (480 MHz ARM Cortex-M7)
 * 
 * Signature: hal-hardware|v1.0.0|nuclear-spinner
 * 
 * @version 1.0.0
 */

#ifndef HAL_HARDWARE_H
#define HAL_HARDWARE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * HARDWARE CONFIGURATION
 * ============================================================================ */

/** System clock frequency (Hz) */
#define SYSCLK_FREQ             480000000UL

/** Timer clock for pulse generation (Hz) */
#define TIM_PULSE_FREQ          240000000UL

/** ADC sampling rate (Hz) */
#define ADC_SAMPLE_RATE         1000000UL

/** DAC update rate (Hz) */
#define DAC_UPDATE_RATE         10000000UL

/** PWM frequency for motor control (Hz) */
#define MOTOR_PWM_FREQ          20000UL

/** Encoder counts per revolution */
#define ENCODER_CPR             4096UL


/* ============================================================================
 * GPIO PIN DEFINITIONS
 * ============================================================================ */

// RF Coil Control
#define RF_ENABLE_PORT          GPIOA
#define RF_ENABLE_PIN           0
#define RF_AMP_PORT             GPIOA
#define RF_AMP_PIN              4       // DAC1 output
#define RF_PHASE_PORT           GPIOA
#define RF_PHASE_PIN            5       // DAC2 output

// Motor Control
#define MOTOR_PWM_PORT          GPIOB
#define MOTOR_PWM_PIN           6       // TIM4_CH1
#define MOTOR_DIR_PORT          GPIOB
#define MOTOR_DIR_PIN           7
#define MOTOR_ENABLE_PORT       GPIOB
#define MOTOR_ENABLE_PIN        8
#define MOTOR_FAULT_PORT        GPIOB
#define MOTOR_FAULT_PIN         9

// Encoder
#define ENC_A_PORT              GPIOC
#define ENC_A_PIN               6       // TIM3_CH1
#define ENC_B_PORT              GPIOC
#define ENC_B_PIN               7       // TIM3_CH2
#define ENC_INDEX_PORT          GPIOC
#define ENC_INDEX_PIN           8

// ADC Inputs
#define ADC_FID_CHANNEL         0       // PA0 - FID signal
#define ADC_TEMP_CHANNEL        1       // PA1 - Temperature sensor
#define ADC_VBAT_CHANNEL        2       // PA2 - Battery voltage

// I2C Sensors
#define I2C_SCL_PORT            GPIOB
#define I2C_SCL_PIN             10
#define I2C_SDA_PORT            GPIOB
#define I2C_SDA_PIN             11

// Status LEDs
#define LED_STATUS_PORT         GPIOD
#define LED_STATUS_PIN          12      // Green
#define LED_ERROR_PORT          GPIOD
#define LED_ERROR_PIN           14      // Red
#define LED_ACTIVITY_PORT       GPIOD
#define LED_ACTIVITY_PIN        15      // Blue

// Safety Interlock
#define INTERLOCK_PORT          GPIOE
#define INTERLOCK_PIN           0


/* ============================================================================
 * DATA TYPES
 * ============================================================================ */

/** HAL Status codes */
typedef enum {
    HAL_OK = 0,
    HAL_ERROR,
    HAL_BUSY,
    HAL_TIMEOUT,
    HAL_INVALID_PARAM,
    HAL_NOT_INITIALIZED,
} HAL_Status_t;

/** RF Pulse parameters */
typedef struct {
    float amplitude;        /**< Amplitude [0.0 - 1.0] */
    float phase;            /**< Phase [0.0 - 2π] radians */
    uint32_t duration_us;   /**< Duration in microseconds */
    uint32_t delay_us;      /**< Post-pulse delay in microseconds */
} RF_Pulse_t;

/** Motor state */
typedef struct {
    float target_rpm;       /**< Target speed (RPM) */
    float actual_rpm;       /**< Measured speed (RPM) */
    float duty_cycle;       /**< Current PWM duty cycle [0.0 - 1.0] */
    bool enabled;           /**< Motor enabled flag */
    bool fault;             /**< Fault detected flag */
} Motor_State_t;

/** Sensor readings */
typedef struct {
    float temperature_c;    /**< Temperature (Celsius) */
    float mag_field_t;      /**< Magnetic field (Tesla) */
    float accel_x;          /**< Acceleration X (m/s²) */
    float accel_y;          /**< Acceleration Y (m/s²) */
    float accel_z;          /**< Acceleration Z (m/s²) */
    float gyro_x;           /**< Angular rate X (rad/s) */
    float gyro_y;           /**< Angular rate Y (rad/s) */
    float gyro_z;           /**< Angular rate Z (rad/s) */
    uint32_t timestamp_ms;  /**< Reading timestamp */
} Sensor_Data_t;

/** ADC buffer for FID acquisition */
typedef struct {
    uint16_t *buffer;       /**< Sample buffer pointer */
    uint32_t size;          /**< Buffer size (samples) */
    uint32_t index;         /**< Current write index */
    bool complete;          /**< Acquisition complete flag */
} ADC_Buffer_t;


/* ============================================================================
 * INITIALIZATION FUNCTIONS
 * ============================================================================ */

/**
 * @brief Initialize all hardware peripherals
 * @return HAL_OK on success
 */
HAL_Status_t HAL_Init_All(void);

/**
 * @brief Initialize system clocks (480 MHz core)
 * @return HAL_OK on success
 */
HAL_Status_t HAL_Init_Clocks(void);

/**
 * @brief Initialize GPIO pins
 * @return HAL_OK on success
 */
HAL_Status_t HAL_Init_GPIO(void);

/**
 * @brief Initialize timers for pulse generation and encoder
 * @return HAL_OK on success
 */
HAL_Status_t HAL_Init_Timers(void);

/**
 * @brief Initialize DACs for amplitude/phase control
 * @return HAL_OK on success
 */
HAL_Status_t HAL_Init_DAC(void);

/**
 * @brief Initialize ADC for FID acquisition
 * @return HAL_OK on success
 */
HAL_Status_t HAL_Init_ADC(void);

/**
 * @brief Initialize I2C for sensor communication
 * @return HAL_OK on success
 */
HAL_Status_t HAL_Init_I2C(void);

/**
 * @brief Initialize DMA for high-speed transfers
 * @return HAL_OK on success
 */
HAL_Status_t HAL_Init_DMA(void);


/* ============================================================================
 * RF PULSE CONTROL
 * ============================================================================ */

/**
 * @brief Enable/disable RF amplifier
 * @param enable True to enable
 */
void HAL_RF_Enable(bool enable);

/**
 * @brief Set RF amplitude via DAC
 * @param amplitude Normalized amplitude [0.0 - 1.0]
 * @return HAL_OK on success
 */
HAL_Status_t HAL_RF_SetAmplitude(float amplitude);

/**
 * @brief Set RF phase via DAC
 * @param phase Phase in radians [0 - 2π]
 * @return HAL_OK on success
 */
HAL_Status_t HAL_RF_SetPhase(float phase);

/**
 * @brief Execute a single RF pulse
 * @param pulse Pulse parameters
 * @return HAL_OK on success
 */
HAL_Status_t HAL_RF_ExecutePulse(const RF_Pulse_t *pulse);

/**
 * @brief Execute a sequence of RF pulses
 * @param pulses Array of pulse parameters
 * @param count Number of pulses
 * @return HAL_OK on success
 */
HAL_Status_t HAL_RF_ExecuteSequence(const RF_Pulse_t *pulses, uint32_t count);

/**
 * @brief Configure pulse timer for specific duration
 * @param duration_us Pulse duration in microseconds
 * @return HAL_OK on success
 */
HAL_Status_t HAL_RF_ConfigureTimer(uint32_t duration_us);


/* ============================================================================
 * MOTOR CONTROL
 * ============================================================================ */

/**
 * @brief Enable/disable motor driver
 * @param enable True to enable
 */
void HAL_Motor_Enable(bool enable);

/**
 * @brief Set motor PWM duty cycle
 * @param duty Duty cycle [0.0 - 1.0]
 * @return HAL_OK on success
 */
HAL_Status_t HAL_Motor_SetDuty(float duty);

/**
 * @brief Set motor direction
 * @param clockwise True for clockwise rotation
 */
void HAL_Motor_SetDirection(bool clockwise);

/**
 * @brief Get current motor state
 * @param state Output state structure
 * @return HAL_OK on success
 */
HAL_Status_t HAL_Motor_GetState(Motor_State_t *state);

/**
 * @brief Read encoder position (counts)
 * @return Current encoder count
 */
uint32_t HAL_Motor_GetEncoderCount(void);

/**
 * @brief Reset encoder count to zero
 */
void HAL_Motor_ResetEncoder(void);

/**
 * @brief Check for motor fault condition
 * @return True if fault detected
 */
bool HAL_Motor_IsFault(void);

/**
 * @brief Clear motor fault (if clearable)
 */
void HAL_Motor_ClearFault(void);


/* ============================================================================
 * ADC / FID ACQUISITION
 * ============================================================================ */

/**
 * @brief Start continuous FID acquisition
 * @param buffer ADC buffer configuration
 * @return HAL_OK on success
 */
HAL_Status_t HAL_ADC_StartFID(ADC_Buffer_t *buffer);

/**
 * @brief Stop FID acquisition
 * @return HAL_OK on success
 */
HAL_Status_t HAL_ADC_StopFID(void);

/**
 * @brief Read single ADC value
 * @param channel ADC channel number
 * @return Raw ADC value (12-bit)
 */
uint16_t HAL_ADC_ReadSingle(uint8_t channel);

/**
 * @brief Convert raw ADC to voltage
 * @param raw Raw ADC value
 * @return Voltage in volts
 */
float HAL_ADC_ToVoltage(uint16_t raw);

/**
 * @brief Read temperature from sensor ADC
 * @return Temperature in Celsius
 */
float HAL_ADC_ReadTemperature(void);


/* ============================================================================
 * I2C SENSOR INTERFACE
 * ============================================================================ */

/**
 * @brief Read all sensor data
 * @param data Output sensor data structure
 * @return HAL_OK on success
 */
HAL_Status_t HAL_Sensor_ReadAll(Sensor_Data_t *data);

/**
 * @brief Read magnetometer (HMC5883L or similar)
 * @param x Output X-axis field (Tesla)
 * @param y Output Y-axis field (Tesla)
 * @param z Output Z-axis field (Tesla)
 * @return HAL_OK on success
 */
HAL_Status_t HAL_Sensor_ReadMag(float *x, float *y, float *z);

/**
 * @brief Read IMU accelerometer/gyroscope (BMI160 or similar)
 * @param accel Output acceleration array [x, y, z] in m/s²
 * @param gyro Output angular rate array [x, y, z] in rad/s
 * @return HAL_OK on success
 */
HAL_Status_t HAL_Sensor_ReadIMU(float accel[3], float gyro[3]);

/**
 * @brief Read PT100 temperature sensor (via MAX31865)
 * @return Temperature in Celsius
 */
float HAL_Sensor_ReadPT100(void);


/* ============================================================================
 * SAFETY AND STATUS
 * ============================================================================ */

/**
 * @brief Check interlock status
 * @return True if interlock is engaged (safe)
 */
bool HAL_Safety_InterlockOK(void);

/**
 * @brief Emergency stop - disable all outputs
 */
void HAL_Safety_EmergencyStop(void);

/**
 * @brief Set status LED
 * @param led LED index (0=status, 1=error, 2=activity)
 * @param state True to turn on
 */
void HAL_LED_Set(uint8_t led, bool state);

/**
 * @brief Toggle status LED
 * @param led LED index
 */
void HAL_LED_Toggle(uint8_t led);




/* ============================================================================
 * COMMUNICATION (UART/USB CDC)
 * ============================================================================
 *
 * Minimal byte-stream API used by comm_protocol.c.
 * Implement these for your chosen transport (USART, USB-CDC, etc).
 */

/**
 * @brief Initialize comm transport (UART/USB).
 */
HAL_Status_t HAL_Comm_Init(void);

/**
 * @brief Read up to max_len bytes into dst.
 * @return number of bytes read (0 if none).
 */
uint32_t HAL_Comm_Read(uint8_t *dst, uint32_t max_len);

/**
 * @brief Write len bytes from src.
 */
HAL_Status_t HAL_Comm_Write(const uint8_t *src, uint32_t len);

/* ============================================================================
 * TIMING AND DELAYS
 * ============================================================================ */

/**
 * @brief Get system tick count (milliseconds)
 * @return Current tick count
 */
uint32_t HAL_GetTick(void);

/**
 * @brief Delay for specified milliseconds
 * @param ms Delay in milliseconds
 */
void HAL_Delay(uint32_t ms);

/**
 * @brief Delay for specified microseconds (blocking)
 * @param us Delay in microseconds
 */
void HAL_DelayMicroseconds(uint32_t us);

/**
 * @brief Start high-resolution timer for profiling
 */
void HAL_Timer_StartProfile(void);

/**
 * @brief Get elapsed time since StartProfile (microseconds)
 * @return Elapsed microseconds
 */
uint32_t HAL_Timer_GetElapsed(void);


/* ============================================================================
 * INTERRUPT CALLBACKS (weak, to be overridden)
 * ============================================================================ */

/**
 * @brief Called when RF pulse timer expires
 */
void HAL_Callback_PulseComplete(void);

/**
 * @brief Called when FID acquisition buffer is half full
 * @param buffer Pointer to buffer
 */
void HAL_Callback_FID_HalfComplete(ADC_Buffer_t *buffer);

/**
 * @brief Called when FID acquisition buffer is full
 * @param buffer Pointer to buffer
 */
void HAL_Callback_FID_Complete(ADC_Buffer_t *buffer);

/**
 * @brief Called on encoder index pulse (once per revolution)
 * @param count Current encoder count
 */
void HAL_Callback_EncoderIndex(uint32_t count);

/**
 * @brief Called on motor fault detection
 */
void HAL_Callback_MotorFault(void);

/**
 * @brief Called on interlock state change
 * @param engaged True if interlock is now engaged
 */
void HAL_Callback_InterlockChange(bool engaged);


#ifdef __cplusplus
}
#endif

#endif /* HAL_HARDWARE_H */
