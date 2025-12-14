/*
 * Minimal startup for STM32H743 (Cortex-M7)
 * This is a placeholder suitable for demo builds.
 * Replace with STM32CubeH7 startup for production.
 */

.syntax unified
.cpu cortex-m7
.fpu fpv5-d16
.thumb

.global g_pfnVectors
.global Reset_Handler

.section .isr_vector,"a",%progbits
.type g_pfnVectors, %object
.size g_pfnVectors, .-g_pfnVectors

g_pfnVectors:
  .word _estack
  .word Reset_Handler
  .word 0 /* NMI */
  .word HardFault_Handler
  .word MemManage_Handler
  .word BusFault_Handler
  .word UsageFault_Handler
  .word 0
  .word 0
  .word 0
  .word 0
  .word 0 /* SVC */
  .word 0 /* DebugMon */
  .word 0
  .word 0 /* PendSV */
  .word SysTick_Handler

.section .text.Reset_Handler,"ax",%progbits
Reset_Handler:
  bl  SystemInit
  bl  main
1: b 1b
