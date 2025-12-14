/* Minimal linker script for STM32H743ZITx (2MB FLASH, 1MB RAM)
 * Replace with vendor linker script for production.
 */

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 2048K
  RAM_D1 (xrw): ORIGIN = 0x24000000, LENGTH = 512K
  RAM_D2 (xrw): ORIGIN = 0x30000000, LENGTH = 288K
  RAM_D3 (xrw): ORIGIN = 0x38000000, LENGTH = 64K
}

_estack = ORIGIN(RAM_D1) + LENGTH(RAM_D1);

SECTIONS
{
  .isr_vector :
  {
    KEEP(*(.isr_vector))
  } > FLASH

  .text :
  {
    *(.text*)
    *(.rodata*)
    _etext = .;
  } > FLASH

  .data : AT ( _etext )
  {
    _sdata = .;
    *(.data*)
    _edata = .;
  } > RAM_D1

  .bss :
  {
    _sbss = .;
    *(.bss*)
    *(COMMON)
    _ebss = .;
  } > RAM_D1
}
