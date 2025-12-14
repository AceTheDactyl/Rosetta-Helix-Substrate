/**
 * @file hal_hardware.c
 * @brief STM32H7 Hardware Abstraction Layer Implementation
 * 
 * Maps physics operations to STM32H743ZI hardware:
 * - TIM1/TIM8: High-resolution RF pulse timing (16-bit, 240 MHz clock)
 * - DAC1/DAC2: RF amplitude and phase control (12-bit, 1 MHz update)
 * - ADC1: FID signal acquisition (14-bit equivalent via oversampling, 2 MS/s)
 * - TIM3: Encoder interface (32-bit quadrature)
 * - TIM4: Motor PWM (20 kHz, 0.1% resolution)
 * - I2C1: Sensor communication (magnetometer, IMU, temperature)
 * 
 * Hardware-Physics Mapping:
 * - z ∈ [0,1] → Rotor RPM ∈ [100, 10000] → TIM4 PWM duty
 * - ΔS_neg(z) = exp(-36(z - z_c)²) computed in real-time
 * - RF pulse amplitude → DAC1 (0-3.3V → 0-1.0 normalized)
 * - RF pulse phase → DAC2 (0-3.3V → 0-2π radians via I/Q modulator)
 * - FID signal → ADC1 with DMA circular buffer
 * 
 * Signature: hal-hardware|v1.0.0|nuclear-spinner
 * 
 * @version 1.0.0
 */

#include "hal_hardware.h"
#include "physics_constants.h"
#include <string.h>

/* ============================================================================
 * STM32H7 REGISTER DEFINITIONS (Simplified for firmware demonstration)
 * In production, use STM32H7xx HAL or LL libraries
 * ============================================================================ */

// Memory-mapped peripheral base addresses (STM32H743)
#define PERIPH_BASE             0x40000000UL
#define APB1PERIPH_BASE         PERIPH_BASE
#define APB2PERIPH_BASE         (PERIPH_BASE + 0x00010000UL)
#define AHB1PERIPH_BASE         (PERIPH_BASE + 0x00020000UL)
#define AHB2PERIPH_BASE         (PERIPH_BASE + 0x08020000UL)

// GPIO
#define GPIOA_BASE              (AHB2PERIPH_BASE + 0x0000UL)
#define GPIOB_BASE              (AHB2PERIPH_BASE + 0x0400UL)
#define GPIOC_BASE              (AHB2PERIPH_BASE + 0x0800UL)
#define GPIOD_BASE              (AHB2PERIPH_BASE + 0x0C00UL)
#define GPIOE_BASE              (AHB2PERIPH_BASE + 0x1000UL)

// Timers
#define TIM1_BASE               (APB2PERIPH_BASE + 0x0000UL)
#define TIM3_BASE               (APB1PERIPH_BASE + 0x0400UL)
#define TIM4_BASE               (APB1PERIPH_BASE + 0x0800UL)
#define TIM8_BASE               (APB2PERIPH_BASE + 0x0400UL)

// DAC
#define DAC1_BASE               (APB1PERIPH_BASE + 0x7400UL)

// ADC
#define ADC1_BASE               (AHB1PERIPH_BASE + 0x2000UL)
#define ADC2_BASE               (AHB1PERIPH_BASE + 0x2100UL)

// I2C
#define I2C1_BASE               (APB1PERIPH_BASE + 0x5400UL)

// DMA
#define DMA1_BASE               (AHB1PERIPH_BASE + 0x0000UL)
#define DMA2_BASE               (AHB1PERIPH_BASE + 0x0400UL)

// RCC (Reset and Clock Control)
#define RCC_BASE                (AHB2PERIPH_BASE + 0x4400UL)

// SysTick
#define SYSTICK_BASE            0xE000E010UL


/* ============================================================================
 * PRIVATE DATA
 * ============================================================================ */

/** System tick counter (incremented every 1ms by SysTick ISR) */
static volatile uint32_t s_systick_count = 0;

/** High-resolution timer for profiling (TIM5, 32-bit, 1 MHz) */
static volatile uint32_t s_profile_start = 0;

/** ADC DMA buffer and state */
static ADC_Buffer_t *s_current_adc_buffer = NULL;
static volatile bool s_adc_running = false;

/** Motor state cache */
static Motor_State_t s_motor_state = {0};

/** Sensor data cache */
static Sensor_Data_t s_sensor_cache = {0};

/** Initialization flags */
static bool s_initialized = false;
static bool s_clocks_initialized = false;
static bool s_gpio_initialized = false;


/* ============================================================================
 * PRIVATE FUNCTION PROTOTYPES
 * ============================================================================ */

static void configure_system_clock_480mhz(void);
static void configure_gpio_af(uint32_t port_base, uint8_t pin, uint8_t af);
static void configure_timer_pwm(uint32_t tim_base, uint32_t freq_hz, uint16_t resolution);
static uint16_t voltage_to_dac(float voltage);
static float adc_to_voltage(uint16_t raw);


/* ============================================================================
 * INITIALIZATION FUNCTIONS
 * ============================================================================ */

HAL_Status_t HAL_Init_All(void) {
    HAL_Status_t status;
    
    status = HAL_Init_Clocks();
    if (status != HAL_OK) return status;
    
    status = HAL_Init_GPIO();
    if (status != HAL_OK) return status;
    
    status = HAL_Init_Timers();
    if (status != HAL_OK) return status;
    
    status = HAL_Init_DAC();
    if (status != HAL_OK) return status;
    
    status = HAL_Init_ADC();
    if (status != HAL_OK) return status;
    
    status = HAL_Init_I2C();
    if (status != HAL_OK) return status;
    
    status = HAL_Init_DMA();
    if (status != HAL_OK) return status;
    
    s_initialized = true;
    
    return HAL_OK;
}


HAL_Status_t HAL_Init_Clocks(void) {
    /**
     * Configure STM32H743 clocks for maximum performance:
     * - HSE: 25 MHz external crystal
     * - PLL1: 480 MHz system clock (SYSCLK)
     * - AHB: 240 MHz (HCLK)
     * - APB1: 120 MHz
     * - APB2: 120 MHz
     * - ADC clock: 75 MHz (from PLL2)
     * - Timer clock: 240 MHz (2x APB)
     * 
     * This provides:
     * - ~4.2 ns resolution for pulse timing
     * - 1 MHz ADC sampling with 16x oversampling for 14-bit effective
     */
    
    configure_system_clock_480mhz();
    
    // Enable peripheral clocks
    // In production: use RCC->AHB1ENR, RCC->APB1ENR, etc.
    // Placeholder for register writes
    
    // Configure SysTick for 1ms interrupts
    // SysTick reload = SYSCLK / 1000 - 1 = 479999
    volatile uint32_t *SYST_RVR = (uint32_t *)(SYSTICK_BASE + 0x04);
    volatile uint32_t *SYST_CVR = (uint32_t *)(SYSTICK_BASE + 0x08);
    volatile uint32_t *SYST_CSR = (uint32_t *)(SYSTICK_BASE + 0x00);
    
    *SYST_RVR = (SYSCLK_FREQ / 1000) - 1;
    *SYST_CVR = 0;
    *SYST_CSR = 0x07;  // Enable, interrupt, use processor clock
    
    s_clocks_initialized = true;
    
    return HAL_OK;
}


HAL_Status_t HAL_Init_GPIO(void) {
    /**
     * Configure GPIO pins:
     * 
     * PA0: ADC1_IN0 (FID signal input)
     * PA4: DAC1_OUT1 (RF amplitude)
     * PA5: DAC1_OUT2 (RF phase)
     * PA6: TIM3_CH1 (Encoder A) - AF2
     * PA7: TIM3_CH2 (Encoder B) - AF2
     * 
     * PB6: TIM4_CH1 (Motor PWM) - AF2
     * PB7: GPIO output (Motor direction)
     * PB8: GPIO output (Motor enable)
     * PB9: GPIO input (Motor fault)
     * PB10: I2C1_SCL - AF4
     * PB11: I2C1_SDA - AF4
     * 
     * PC6: TIM3_CH1 (Encoder A alt) - AF2
     * PC7: TIM3_CH2 (Encoder B alt) - AF2
     * PC8: GPIO input (Encoder index)
     * 
     * PD12: GPIO output (Status LED - Green)
     * PD14: GPIO output (Error LED - Red)
     * PD15: GPIO output (Activity LED - Blue)
     * 
     * PE0: GPIO input (Interlock)
     */
    
    // Enable GPIO clocks (RCC->AHB4ENR)
    // Configure mode, speed, pull-up/down, alternate function
    // Placeholder for actual register configuration
    
    // Configure analog inputs (PA0 for ADC, PA4/PA5 for DAC)
    // Mode = Analog (0b11)
    
    // Configure alternate function pins
    configure_gpio_af(GPIOA_BASE, 6, 2);   // TIM3_CH1
    configure_gpio_af(GPIOA_BASE, 7, 2);   // TIM3_CH2
    configure_gpio_af(GPIOB_BASE, 6, 2);   // TIM4_CH1
    configure_gpio_af(GPIOB_BASE, 10, 4);  // I2C1_SCL
    configure_gpio_af(GPIOB_BASE, 11, 4);  // I2C1_SDA
    
    // Configure GPIO outputs (Motor control, LEDs)
    // Mode = Output (0b01), Speed = High, Push-pull
    
    // Configure GPIO inputs (Fault, Index, Interlock)
    // Mode = Input (0b00), Pull-up enabled
    
    s_gpio_initialized = true;
    
    return HAL_OK;
}


HAL_Status_t HAL_Init_Timers(void) {
    /**
     * Timer configuration:
     * 
     * TIM1: RF pulse timing (16-bit, 240 MHz, one-pulse mode)
     *       Resolution: 4.17 ns
     *       Max duration: 273 µs (use prescaler for longer)
     * 
     * TIM3: Encoder interface (32-bit quadrature mode)
     *       4096 CPR encoder → 16384 counts/rev with 4x decoding
     *       Max RPM = 240 MHz / 16384 = 14,648 RPM (sufficient)
     * 
     * TIM4: Motor PWM (20 kHz, 12-bit resolution)
     *       Prescaler = 0, ARR = 12000-1 → 20 kHz
     *       0.0083% duty resolution
     * 
     * TIM5: Profile timer (32-bit, 1 MHz free-running)
     *       For microsecond-resolution timing
     */
    
    // TIM1: One-pulse mode for RF timing
    // - Prescaler = 0 (240 MHz clock)
    // - ARR set dynamically based on pulse duration
    // - OPM bit set for single pulse
    // - CH1 output to RF enable gate
    
    // TIM3: Encoder mode
    // - SMS = 0b011 (encoder mode 3 - count both edges)
    // - CC1S = 0b01, CC2S = 0b01 (inputs on TI1, TI2)
    // - IC1F, IC2F = 0b0011 (filtering)
    
    // TIM4: PWM mode
    configure_timer_pwm(TIM4_BASE, MOTOR_PWM_FREQ, 12000);
    
    // TIM5: Free-running microsecond counter
    // - Prescaler = 240-1 (1 MHz from 240 MHz)
    // - ARR = 0xFFFFFFFF (32-bit)
    // - Enable and let run
    
    return HAL_OK;
}


HAL_Status_t HAL_Init_DAC(void) {
    /**
     * DAC configuration for RF amplitude and phase:
     * 
     * DAC1_CH1 (PA4): RF amplitude
     *   - 12-bit resolution
     *   - Output range: 0-3.3V → 0-1.0 normalized amplitude
     *   - Buffered output for driving RF VGA
     * 
     * DAC1_CH2 (PA5): RF phase
     *   - 12-bit resolution
     *   - Output range: 0-3.3V → 0-2π radians
     *   - Drives I/Q modulator phase input
     * 
     * Both channels use DMA for smooth updates during sequences
     */
    
    // Enable DAC clock
    // Configure DAC1 channel 1 and 2
    // - TEN = 0 (no trigger, write directly)
    // - BOFF = 0 (buffer enabled)
    // - EN = 1 (channel enabled)
    
    // Initialize to zero output
    // DAC1->DHR12R1 = 0;
    // DAC1->DHR12R2 = 0;
    
    return HAL_OK;
}


HAL_Status_t HAL_Init_ADC(void) {
    /**
     * ADC configuration for FID acquisition:
     * 
     * ADC1:
     *   - 16-bit resolution with oversampling
     *   - 2 MS/s sampling rate
     *   - Single channel (PA0 = IN0)
     *   - DMA circular mode for continuous acquisition
     *   - Triggered by TIM1 TRGO (synchronized with RF pulses)
     * 
     * Effective resolution: 14-bit (16x oversampling, 4-bit shift)
     * 
     * Signal chain:
     *   RF coil → LNA → Mixer → IF filter → ADC
     *   Baseband signal centered at DC after quadrature detection
     */
    
    // Enable ADC clock (from PLL2, 75 MHz)
    // Configure ADC1:
    // - ADCALDIF = 0 (single-ended calibration)
    // - Run calibration
    // - RES = 0b00 (16-bit)
    // - OVRMOD = 1 (overwrite on overrun)
    // - ROVSM = 0 (regular oversampling mode)
    // - OVSR = 0b011 (16x oversampling)
    // - OVSS = 0b0100 (4-bit right shift)
    
    // Configure regular sequence:
    // - L = 0 (1 conversion)
    // - SQ1 = 0 (channel 0)
    
    // Configure sampling time:
    // - SMP0 = 0b010 (6.5 cycles) → ~11.5 µs total conversion time
    
    // Enable DMA request
    // - DMAEN = 1, DMACFG = 1 (circular mode)
    
    return HAL_OK;
}


HAL_Status_t HAL_Init_I2C(void) {
    /**
     * I2C1 configuration for sensors:
     * 
     * - 400 kHz Fast Mode
     * - 7-bit addressing
     * - DMA for bulk transfers
     * 
     * Connected devices:
     * - HMC5883L magnetometer (addr: 0x1E)
     * - BMI160 IMU (addr: 0x68 or 0x69)
     * - MAX31865 PT100 interface (SPI, but can be I2C adapter)
     */
    
    // Configure I2C1:
    // - TIMING = calculated for 400 kHz from 120 MHz APB1
    // - ANFOFF = 0 (analog filter enabled)
    // - DNF = 0b0001 (1 clock cycle digital filter)
    // - PE = 1 (enable)
    
    return HAL_OK;
}


HAL_Status_t HAL_Init_DMA(void) {
    /**
     * DMA configuration:
     * 
     * DMA1 Stream 0: ADC1 → Memory (FID acquisition)
     *   - Circular mode
     *   - Half-transfer and transfer-complete interrupts
     *   - Memory increment, peripheral fixed
     *   - 16-bit transfers
     * 
     * DMA1 Stream 1: Memory → DAC (pulse waveforms)
     *   - Normal mode (single shot)
     *   - Memory increment
     *   - 16-bit transfers
     */
    
    // Configure DMA streams
    // Enable DMA interrupts
    
    return HAL_OK;
}


/* ============================================================================
 * RF PULSE CONTROL
 * ============================================================================ */

void HAL_RF_Enable(bool enable) {
    /**
     * Enable/disable RF amplifier
     * 
     * Controls RF gate signal (PA0 via TIM1 or direct GPIO)
     * When disabled, RF output is zero regardless of DAC settings
     */
    
    if (enable) {
        // Set RF enable GPIO high
        // Or configure TIM1 to output
    } else {
        // Set RF enable GPIO low
        // Disable TIM1 output
        HAL_RF_SetAmplitude(0.0f);  // Also zero the DAC
    }
}


HAL_Status_t HAL_RF_SetAmplitude(float amplitude) {
    /**
     * Set RF amplitude via DAC1_CH1
     * 
     * amplitude ∈ [0.0, 1.0] → DAC value ∈ [0, 4095]
     * DAC output drives Variable Gain Amplifier (VGA) control input
     */
    
    if (amplitude < 0.0f) amplitude = 0.0f;
    if (amplitude > 1.0f) amplitude = 1.0f;
    
    // Convert to 12-bit DAC value
    uint16_t dac_value = (uint16_t)(amplitude * 4095.0f);
    
    // Write to DAC1 Channel 1
    // DAC1->DHR12R1 = dac_value;
    
    // Placeholder for register write
    (void)dac_value;
    
    return HAL_OK;
}


HAL_Status_t HAL_RF_SetPhase(float phase) {
    /**
     * Set RF phase via DAC1_CH2
     * 
     * phase ∈ [0, 2π] → DAC value ∈ [0, 4095]
     * DAC output controls I/Q modulator phase:
     *   0V → 0°
     *   3.3V → 360° (wraps to 0°)
     */
    
    // Normalize phase to [0, 2π]
    while (phase < 0.0f) phase += 2.0f * 3.14159265f;
    while (phase >= 2.0f * 3.14159265f) phase -= 2.0f * 3.14159265f;
    
    // Convert to 12-bit DAC value
    uint16_t dac_value = (uint16_t)(phase / (2.0f * 3.14159265f) * 4095.0f);
    
    // Write to DAC1 Channel 2
    // DAC1->DHR12R2 = dac_value;
    
    (void)dac_value;
    
    return HAL_OK;
}


HAL_Status_t HAL_RF_ExecutePulse(const RF_Pulse_t *pulse) {
    if (pulse == NULL) return HAL_INVALID_PARAM;
    
    // Set amplitude and phase
    HAL_RF_SetAmplitude(pulse->amplitude);
    HAL_RF_SetPhase(pulse->phase);
    
    // Enable RF for duration
    HAL_RF_Enable(true);
    HAL_DelayMicroseconds(pulse->duration_us);
    HAL_RF_Enable(false);
    
    // Post-pulse delay
    if (pulse->delay_us > 0) {
        HAL_DelayMicroseconds(pulse->delay_us);
    }
    
    return HAL_OK;
}


HAL_Status_t HAL_RF_ExecuteSequence(const RF_Pulse_t *pulses, uint32_t count) {
    if (pulses == NULL || count == 0) return HAL_INVALID_PARAM;
    
    for (uint32_t i = 0; i < count; i++) {
        HAL_Status_t status = HAL_RF_ExecutePulse(&pulses[i]);
        if (status != HAL_OK) return status;
    }
    
    return HAL_OK;
}


HAL_Status_t HAL_RF_ConfigureTimer(uint32_t duration_us) {
    /**
     * Configure TIM1 for precise pulse timing
     * 
     * Timer clock: 240 MHz
     * Resolution: 4.17 ns
     * 
     * For durations > 273 µs, use prescaler
     */
    
    uint32_t timer_counts;
    uint16_t prescaler = 0;
    
    // Calculate timer counts
    // counts = duration_us * 240 (for 240 MHz clock)
    timer_counts = duration_us * 240;
    
    // If counts > 65535, need prescaler
    while (timer_counts > 65535 && prescaler < 65535) {
        prescaler++;
        timer_counts = (duration_us * 240) / (prescaler + 1);
    }
    
    // Configure TIM1
    // TIM1->PSC = prescaler;
    // TIM1->ARR = timer_counts;
    // TIM1->CR1 |= TIM_CR1_OPM;  // One-pulse mode
    
    return HAL_OK;
}


/* ============================================================================
 * MOTOR CONTROL
 * ============================================================================ */

void HAL_Motor_Enable(bool enable) {
    s_motor_state.enabled = enable;
    
    // Set motor enable GPIO (PB8)
    if (enable) {
        // GPIOB->BSRR = (1 << 8);  // Set PB8 high
    } else {
        // GPIOB->BSRR = (1 << (8 + 16));  // Reset PB8
        HAL_Motor_SetDuty(0.0f);
    }
}


HAL_Status_t HAL_Motor_SetDuty(float duty) {
    if (duty < 0.0f) duty = 0.0f;
    if (duty > 1.0f) duty = 1.0f;
    
    s_motor_state.duty_cycle = duty;
    
    // Calculate CCR value for TIM4
    // ARR = 12000-1, so CCR = duty * 11999
    uint16_t ccr = (uint16_t)(duty * 11999.0f);
    
    // TIM4->CCR1 = ccr;
    
    (void)ccr;
    
    return HAL_OK;
}


void HAL_Motor_SetDirection(bool clockwise) {
    // Set direction GPIO (PB7)
    if (clockwise) {
        // GPIOB->BSRR = (1 << 7);
    } else {
        // GPIOB->BSRR = (1 << (7 + 16));
    }
}


HAL_Status_t HAL_Motor_GetState(Motor_State_t *state) {
    if (state == NULL) return HAL_INVALID_PARAM;
    
    // Read encoder and compute RPM
    static uint32_t last_count = 0;
    static uint32_t last_tick = 0;
    
    uint32_t count = HAL_Motor_GetEncoderCount();
    uint32_t tick = HAL_GetTick();
    
    if (tick != last_tick) {
        int32_t delta = (int32_t)(count - last_count);
        uint32_t dt_ms = tick - last_tick;
        
        // RPM = (delta / ENCODER_CPR) * (60000 / dt_ms)
        float rpm = (float)delta / ENCODER_CPR * 60000.0f / dt_ms;
        s_motor_state.actual_rpm = (rpm < 0) ? -rpm : rpm;
        
        last_count = count;
        last_tick = tick;
    }
    
    *state = s_motor_state;
    
    return HAL_OK;
}


uint32_t HAL_Motor_GetEncoderCount(void) {
    // Read TIM3 counter (32-bit encoder interface)
    // return TIM3->CNT;
    
    // Placeholder: return simulated value
    static uint32_t sim_count = 0;
    sim_count += (uint32_t)(s_motor_state.duty_cycle * 100);
    return sim_count;
}


void HAL_Motor_ResetEncoder(void) {
    // TIM3->CNT = 0;
}


bool HAL_Motor_IsFault(void) {
    // Read fault GPIO (PB9)
    // return !(GPIOB->IDR & (1 << 9));  // Active low
    return false;
}


void HAL_Motor_ClearFault(void) {
    s_motor_state.fault = false;
}


/* ============================================================================
 * ADC / FID ACQUISITION
 * ============================================================================ */

HAL_Status_t HAL_ADC_StartFID(ADC_Buffer_t *buffer) {
    if (buffer == NULL || buffer->buffer == NULL) return HAL_INVALID_PARAM;
    if (s_adc_running) return HAL_BUSY;
    
    s_current_adc_buffer = buffer;
    buffer->index = 0;
    buffer->complete = false;
    s_adc_running = true;
    
    // Configure DMA for circular transfer
    // DMA1_Stream0->M0AR = (uint32_t)buffer->buffer;
    // DMA1_Stream0->NDTR = buffer->size;
    // DMA1_Stream0->CR |= DMA_SxCR_EN;
    
    // Start ADC
    // ADC1->CR |= ADC_CR_ADSTART;
    
    return HAL_OK;
}


HAL_Status_t HAL_ADC_StopFID(void) {
    s_adc_running = false;
    
    // Stop ADC
    // ADC1->CR |= ADC_CR_ADSTP;
    
    // Disable DMA
    // DMA1_Stream0->CR &= ~DMA_SxCR_EN;
    
    if (s_current_adc_buffer != NULL) {
        s_current_adc_buffer->complete = true;
    }
    
    return HAL_OK;
}


uint16_t HAL_ADC_ReadSingle(uint8_t channel) {
    (void)channel;
    
    // Configure channel, start conversion, wait, read
    // ADC1->SQR1 = (channel << ADC_SQR1_SQ1_Pos);
    // ADC1->CR |= ADC_CR_ADSTART;
    // while (!(ADC1->ISR & ADC_ISR_EOC));
    // return ADC1->DR;
    
    return 2048;  // Mid-scale placeholder
}


float HAL_ADC_ToVoltage(uint16_t raw) {
    // 16-bit ADC, 3.3V reference
    return (float)raw / 65535.0f * 3.3f;
}


float HAL_ADC_ReadTemperature(void) {
    // Read from ADC channel connected to temperature sensor
    uint16_t raw = HAL_ADC_ReadSingle(ADC_TEMP_CHANNEL);
    
    // Convert based on sensor characteristics
    // For example, LM35: 10mV/°C
    float voltage = HAL_ADC_ToVoltage(raw);
    return voltage * 100.0f;  // 10mV/°C → °C
}


/* ============================================================================
 * I2C SENSOR INTERFACE
 * ============================================================================ */

/**
 * @brief Low-level I2C write
 */
static HAL_Status_t i2c_write(uint8_t addr, uint8_t reg, uint8_t *data, uint8_t len) {
    (void)addr; (void)reg; (void)data; (void)len;
    // I2C1->CR2 = (addr << 1) | (len + 1) << I2C_CR2_NBYTES_Pos;
    // I2C1->CR2 |= I2C_CR2_START;
    // ... write register address and data ...
    return HAL_OK;
}

/**
 * @brief Low-level I2C read
 */
static HAL_Status_t i2c_read(uint8_t addr, uint8_t reg, uint8_t *data, uint8_t len) {
    (void)addr; (void)reg; (void)data; (void)len;
    // Write register address, then read data
    return HAL_OK;
}


HAL_Status_t HAL_Sensor_ReadAll(Sensor_Data_t *data) {
    if (data == NULL) return HAL_INVALID_PARAM;
    
    // Read magnetometer
    float mag[3];
    HAL_Sensor_ReadMag(&mag[0], &mag[1], &mag[2]);
    data->mag_field_t = sqrtf(mag[0]*mag[0] + mag[1]*mag[1] + mag[2]*mag[2]);
    
    // Read IMU
    float accel[3], gyro[3];
    HAL_Sensor_ReadIMU(accel, gyro);
    data->accel_x = accel[0];
    data->accel_y = accel[1];
    data->accel_z = accel[2];
    data->gyro_x = gyro[0];
    data->gyro_y = gyro[1];
    data->gyro_z = gyro[2];
    
    // Read temperature
    data->temperature_c = HAL_Sensor_ReadPT100();
    
    data->timestamp_ms = HAL_GetTick();
    
    s_sensor_cache = *data;
    
    return HAL_OK;
}


HAL_Status_t HAL_Sensor_ReadMag(float *x, float *y, float *z) {
    /**
     * Read HMC5883L magnetometer
     * Address: 0x1E
     * Data registers: 0x03-0x08 (X, Z, Y in that order)
     * Scale: 1090 LSB/Gauss (default gain)
     */
    
    uint8_t raw[6];
    i2c_read(0x1E, 0x03, raw, 6);
    
    // Convert to Tesla (1 Gauss = 1e-4 Tesla)
    int16_t raw_x = (raw[0] << 8) | raw[1];
    int16_t raw_z = (raw[2] << 8) | raw[3];
    int16_t raw_y = (raw[4] << 8) | raw[5];
    
    *x = (float)raw_x / 1090.0f * 1e-4f;
    *y = (float)raw_y / 1090.0f * 1e-4f;
    *z = (float)raw_z / 1090.0f * 1e-4f;
    
    return HAL_OK;
}


HAL_Status_t HAL_Sensor_ReadIMU(float accel[3], float gyro[3]) {
    /**
     * Read BMI160 IMU
     * Address: 0x68
     * Accel: 0x12-0x17 (X, Y, Z, 16-bit each)
     * Gyro: 0x0C-0x11 (X, Y, Z, 16-bit each)
     * 
     * Accel range: ±2g (default), scale = 16384 LSB/g
     * Gyro range: ±2000°/s (default), scale = 16.4 LSB/(°/s)
     */
    
    uint8_t raw[12];
    i2c_read(0x68, 0x0C, raw, 12);  // Read gyro and accel together
    
    // Parse gyroscope (first 6 bytes)
    int16_t gx = (raw[1] << 8) | raw[0];
    int16_t gy = (raw[3] << 8) | raw[2];
    int16_t gz = (raw[5] << 8) | raw[4];
    
    // Parse accelerometer (next 6 bytes)
    int16_t ax = (raw[7] << 8) | raw[6];
    int16_t ay = (raw[9] << 8) | raw[8];
    int16_t az = (raw[11] << 8) | raw[10];
    
    // Convert to m/s² and rad/s
    accel[0] = (float)ax / 16384.0f * 9.81f;
    accel[1] = (float)ay / 16384.0f * 9.81f;
    accel[2] = (float)az / 16384.0f * 9.81f;
    
    gyro[0] = (float)gx / 16.4f * 3.14159f / 180.0f;
    gyro[1] = (float)gy / 16.4f * 3.14159f / 180.0f;
    gyro[2] = (float)gz / 16.4f * 3.14159f / 180.0f;
    
    return HAL_OK;
}


float HAL_Sensor_ReadPT100(void) {
    /**
     * Read PT100 via MAX31865 RTD-to-digital converter
     * 
     * The MAX31865 provides 15-bit resistance measurement
     * PT100 at 0°C: 100Ω
     * Temperature coefficient: ~0.385Ω/°C
     * 
     * For I2C interface (using adapter), address depends on adapter
     * Alternatively, use SPI directly to MAX31865
     */
    
    // Placeholder: return simulated temperature
    return 25.0f + (float)(HAL_GetTick() % 100) * 0.01f;
}


/* ============================================================================
 * SAFETY AND STATUS
 * ============================================================================ */

bool HAL_Safety_InterlockOK(void) {
    // Read interlock GPIO (PE0)
    // Active low: low = engaged (safe), high = open (unsafe)
    // return !(GPIOE->IDR & (1 << 0));
    return true;  // Placeholder
}


void HAL_Safety_EmergencyStop(void) {
    // Disable all outputs immediately
    HAL_RF_Enable(false);
    HAL_RF_SetAmplitude(0.0f);
    HAL_Motor_Enable(false);
    HAL_Motor_SetDuty(0.0f);
    
    // Set error LED
    HAL_LED_Set(1, true);
}


void HAL_LED_Set(uint8_t led, bool state) {
    uint8_t pin;
    uint32_t port = GPIOD_BASE;
    
    switch (led) {
        case 0: pin = 12; break;  // Status (Green)
        case 1: pin = 14; break;  // Error (Red)
        case 2: pin = 15; break;  // Activity (Blue)
        default: return;
    }
    
    (void)port;
    
    if (state) {
        // GPIOx->BSRR = (1 << pin);
    } else {
        // GPIOx->BSRR = (1 << (pin + 16));
    }
}


void HAL_LED_Toggle(uint8_t led) {
    uint8_t pin;
    
    switch (led) {
        case 0: pin = 12; break;
        case 1: pin = 14; break;
        case 2: pin = 15; break;
        default: return;
    }
    
    (void)pin;
    // GPIOx->ODR ^= (1 << pin);
}


/* ============================================================================
 * TIMING AND DELAYS
 * ============================================================================ */

uint32_t HAL_GetTick(void) {
    return s_systick_count;
}


void HAL_Delay(uint32_t ms) {
    uint32_t start = HAL_GetTick();
    while ((HAL_GetTick() - start) < ms) {
        // Spin wait
        // In production, use WFI for power savings
    }
}


void HAL_DelayMicroseconds(uint32_t us) {
    /**
     * Microsecond delay using TIM5 (1 MHz counter)
     * For very short delays (< 10 µs), use cycle counting
     */
    
    if (us < 10) {
        // Use CPU cycle counting for short delays
        // At 480 MHz, 1 µs = 480 cycles
        uint32_t cycles = us * 480;
        while (cycles--) {
            __asm volatile("nop");
        }
    } else {
        // Use TIM5 for longer delays
        // uint32_t start = TIM5->CNT;
        // while ((TIM5->CNT - start) < us);
        
        // Placeholder: use tick-based approximation
        uint32_t ms = us / 1000;
        if (ms > 0) HAL_Delay(ms);
    }
}


void HAL_Timer_StartProfile(void) {
    // s_profile_start = TIM5->CNT;
    s_profile_start = HAL_GetTick() * 1000;  // Approximation
}


uint32_t HAL_Timer_GetElapsed(void) {
    // return TIM5->CNT - s_profile_start;
    return HAL_GetTick() * 1000 - s_profile_start;
}


/* ============================================================================
 * PRIVATE HELPER FUNCTIONS
 * ============================================================================ */

static void configure_system_clock_480mhz(void) {
    /**
     * Configure PLL for 480 MHz SYSCLK from 25 MHz HSE
     * 
     * PLL1: HSE / M * N / P = 25 / 5 * 192 / 2 = 480 MHz
     * 
     * This is a simplified representation; actual implementation
     * requires careful sequencing of RCC registers.
     */
    
    // Enable HSE
    // RCC->CR |= RCC_CR_HSEON;
    // while (!(RCC->CR & RCC_CR_HSERDY));
    
    // Configure PLL1
    // RCC->PLLCKSELR = (5 << 4) | RCC_PLLCKSELR_PLLSRC_HSE;  // M = 5, HSE source
    // RCC->PLL1DIVR = ((192 - 1) << 0) | ((2 - 1) << 9);     // N = 192, P = 2
    
    // Enable PLL1
    // RCC->CR |= RCC_CR_PLL1ON;
    // while (!(RCC->CR & RCC_CR_PLL1RDY));
    
    // Switch to PLL
    // RCC->CFGR |= RCC_CFGR_SW_PLL1;
    // while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL1);
}


static void configure_gpio_af(uint32_t port_base, uint8_t pin, uint8_t af) {
    /**
     * Configure GPIO pin for alternate function
     */
    (void)port_base;
    (void)pin;
    (void)af;
    
    // volatile uint32_t *MODER = (uint32_t *)(port_base + 0x00);
    // volatile uint32_t *AFRL = (uint32_t *)(port_base + 0x20);
    // volatile uint32_t *AFRH = (uint32_t *)(port_base + 0x24);
    
    // Set mode to alternate function (0b10)
    // *MODER = (*MODER & ~(0x3 << (pin * 2))) | (0x2 << (pin * 2));
    
    // Set alternate function
    // if (pin < 8) {
    //     *AFRL = (*AFRL & ~(0xF << (pin * 4))) | (af << (pin * 4));
    // } else {
    //     *AFRH = (*AFRH & ~(0xF << ((pin - 8) * 4))) | (af << ((pin - 8) * 4));
    // }
}


static void configure_timer_pwm(uint32_t tim_base, uint32_t freq_hz, uint16_t resolution) {
    /**
     * Configure timer for PWM output
     */
    (void)tim_base;
    (void)freq_hz;
    (void)resolution;
    
    // uint32_t timer_clock = 240000000;  // 240 MHz
    // uint32_t arr = timer_clock / freq_hz - 1;
    
    // TIMx->PSC = 0;
    // TIMx->ARR = arr;
    // TIMx->CCR1 = 0;
    // TIMx->CCMR1 = (0b110 << 4);  // PWM mode 1
    // TIMx->CCER = TIM_CCER_CC1E;  // Enable output
    // TIMx->CR1 = TIM_CR1_CEN;     // Enable timer
}


static uint16_t voltage_to_dac(float voltage) {
    if (voltage < 0.0f) voltage = 0.0f;
    if (voltage > 3.3f) voltage = 3.3f;
    return (uint16_t)(voltage / 3.3f * 4095.0f);
}


static float adc_to_voltage(uint16_t raw) {
    return (float)raw / 65535.0f * 3.3f;
}


/* ============================================================================
 * INTERRUPT SERVICE ROUTINES
 * ============================================================================ */

/**
 * @brief SysTick interrupt handler (1 ms)
 */
void SysTick_Handler(void) {
    s_systick_count++;
}


/**
 * @brief DMA1 Stream 0 interrupt (ADC transfer complete)
 */
void DMA1_Stream0_IRQHandler(void) {
    // Check half-transfer flag
    // if (DMA1->LISR & DMA_LISR_HTIF0) {
    //     DMA1->LIFCR = DMA_LIFCR_CHTIF0;
    //     if (s_current_adc_buffer) {
    //         s_current_adc_buffer->index = s_current_adc_buffer->size / 2;
    //         HAL_Callback_FID_HalfComplete(s_current_adc_buffer);
    //     }
    // }
    
    // Check transfer-complete flag
    // if (DMA1->LISR & DMA_LISR_TCIF0) {
    //     DMA1->LIFCR = DMA_LIFCR_CTCIF0;
    //     if (s_current_adc_buffer) {
    //         s_current_adc_buffer->index = s_current_adc_buffer->size;
    //         s_current_adc_buffer->complete = true;
    //         HAL_Callback_FID_Complete(s_current_adc_buffer);
    //     }
    // }
}


/**
 * @brief TIM1 Update interrupt (pulse complete)
 */
void TIM1_UP_IRQHandler(void) {
    // TIM1->SR &= ~TIM_SR_UIF;
    HAL_Callback_PulseComplete();
}


/**
 * @brief TIM3 interrupt (encoder index pulse)
 */
void TIM3_IRQHandler(void) {
    // if (TIM3->SR & TIM_SR_CC3IF) {  // Assuming index on CH3
    //     TIM3->SR &= ~TIM_SR_CC3IF;
    //     HAL_Callback_EncoderIndex(TIM3->CNT);
    // }
}


/**
 * @brief EXTI interrupt (motor fault, interlock change)
 */
void EXTI9_5_IRQHandler(void) {
    // Motor fault on PB9 (EXTI9)
    // if (EXTI->PR1 & (1 << 9)) {
    //     EXTI->PR1 = (1 << 9);
    //     HAL_Callback_MotorFault();
    // }
}


void EXTI0_IRQHandler(void) {
    // Interlock on PE0 (EXTI0)
    // if (EXTI->PR1 & (1 << 0)) {
    //     EXTI->PR1 = (1 << 0);
    //     HAL_Callback_InterlockChange(HAL_Safety_InterlockOK());
    // }
}


/* ============================================================================
 * WEAK CALLBACK STUBS
 * ============================================================================ */

__attribute__((weak)) void HAL_Callback_PulseComplete(void) {
    // Override in application
}

__attribute__((weak)) void HAL_Callback_FID_HalfComplete(ADC_Buffer_t *buffer) {
    (void)buffer;
}

__attribute__((weak)) void HAL_Callback_FID_Complete(ADC_Buffer_t *buffer) {
    (void)buffer;
}

__attribute__((weak)) void HAL_Callback_EncoderIndex(uint32_t count) {
    (void)count;
}

__attribute__((weak)) void HAL_Callback_MotorFault(void) {
    HAL_Safety_EmergencyStop();
}

__attribute__((weak)) void HAL_Callback_InterlockChange(bool engaged) {
    if (!engaged) {
        HAL_Safety_EmergencyStop();
    }
}
