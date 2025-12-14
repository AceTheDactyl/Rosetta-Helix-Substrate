/**
 * @file stm32h7xx_it.c
 * @brief Interrupt vector stubs.
 *
 * NOTE: SysTick_Handler is implemented in drivers/hal_hardware.c
 * to keep the demo self-contained.
 */

#include <stdint.h>

void HardFault_Handler(void) {
    while (1) {
        __asm__ volatile ("nop");
    }
}

void MemManage_Handler(void) { while (1) { __asm__ volatile ("nop"); } }
void BusFault_Handler(void)  { while (1) { __asm__ volatile ("nop"); } }
void UsageFault_Handler(void){ while (1) { __asm__ volatile ("nop"); } }
