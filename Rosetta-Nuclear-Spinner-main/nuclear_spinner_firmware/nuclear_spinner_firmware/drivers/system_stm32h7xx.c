/**
 * @file system_stm32h7xx.c
 * @brief Minimal system init stub for Nuclear Spinner firmware.
 *
 * This repo uses a custom HAL layer (drivers/hal_hardware.c) that configures
 * clocks and SysTick inside HAL_Init_Clocks().
 *
 * In a full STM32CubeH7 integration, replace this file with the vendor
 * SystemInit/SystemCoreClock implementation.
 */

#include <stdint.h>

// CMSIS typically provides this.
uint32_t SystemCoreClock = 480000000UL;

void SystemInit(void) {
    // Intentionally minimal.
}
