# ============================================================================
# Nuclear Spinner Firmware Makefile
# ============================================================================
# Target: STM32H743ZI (ARM Cortex-M7 @ 480 MHz)
# Toolchain: arm-none-eabi-gcc
#
# Signature: nuclear-spinner-firmware|v1.0.0|helix
# ============================================================================

# Project name
PROJECT = nuclear_spinner

# Target MCU
MCU = cortex-m7
FLOAT_ABI = hard
FPU = fpv5-d16

# Toolchain
PREFIX = arm-none-eabi-
CC = $(PREFIX)gcc
AS = $(PREFIX)as
LD = $(PREFIX)ld
OBJCOPY = $(PREFIX)objcopy
OBJDUMP = $(PREFIX)objdump
SIZE = $(PREFIX)size

# Directories
SRC_DIR = src
INC_DIR = include
DRV_DIR = drivers
LIB_DIR = lib
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

# Source files
SRC_FILES = \
    $(SRC_DIR)/main.c \
    $(SRC_DIR)/pulse_control.c \
    $(SRC_DIR)/rotor_control.c \
    $(SRC_DIR)/threshold_logic.c

# Driver files (HAL implementation would go here)
DRV_FILES = \
    $(DRV_DIR)/hal_hardware.c \
    $(DRV_DIR)/stm32h7xx_it.c \
    $(DRV_DIR)/system_stm32h7xx.c

# Startup file (assembly)
STARTUP = $(DRV_DIR)/startup_stm32h743xx.s

# Linker script
LDSCRIPT = $(DRV_DIR)/STM32H743ZITx_FLASH.ld

# Include paths
INCLUDES = \
    -I$(INC_DIR) \
    -I$(DRV_DIR) \
    -I$(LIB_DIR)/CMSIS/Include \
    -I$(LIB_DIR)/CMSIS/Device/ST/STM32H7xx/Include

# Compiler flags
CFLAGS = -mcpu=$(MCU) -mthumb -mfloat-abi=$(FLOAT_ABI) -mfpu=$(FPU)
CFLAGS += -Wall -Wextra -Werror
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -DSTM32H743xx
CFLAGS += -DUSE_HAL_DRIVER
CFLAGS += $(INCLUDES)

# Debug/Release flags
ifdef DEBUG
    CFLAGS += -g3 -O0 -DDEBUG
else
    CFLAGS += -O2 -DNDEBUG
endif

# Linker flags
LDFLAGS = -mcpu=$(MCU) -mthumb -mfloat-abi=$(FLOAT_ABI) -mfpu=$(FPU)
LDFLAGS += -T$(LDSCRIPT)
LDFLAGS += -Wl,--gc-sections
LDFLAGS += -Wl,-Map=$(BUILD_DIR)/$(PROJECT).map
LDFLAGS += -lm -lc -lnosys

# Object files
OBJS = $(SRC_FILES:%.c=$(OBJ_DIR)/%.o)
OBJS += $(DRV_FILES:%.c=$(OBJ_DIR)/%.o)
OBJS += $(OBJ_DIR)/startup.o

# Targets
.PHONY: all clean flash size

all: $(BUILD_DIR)/$(PROJECT).elf $(BUILD_DIR)/$(PROJECT).hex $(BUILD_DIR)/$(PROJECT).bin size

$(BUILD_DIR)/$(PROJECT).elf: $(OBJS) | $(BUILD_DIR)
	@echo "Linking..."
	$(CC) $(LDFLAGS) -o $@ $^

$(BUILD_DIR)/$(PROJECT).hex: $(BUILD_DIR)/$(PROJECT).elf
	$(OBJCOPY) -O ihex $< $@

$(BUILD_DIR)/$(PROJECT).bin: $(BUILD_DIR)/$(PROJECT).elf
	$(OBJCOPY) -O binary $< $@

$(OBJ_DIR)/%.o: %.c | $(OBJ_DIR)
	@mkdir -p $(dir $@)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c -o $@ $<

$(OBJ_DIR)/startup.o: $(STARTUP) | $(OBJ_DIR)
	@echo "Assembling startup..."
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD_DIR) $(OBJ_DIR):
	mkdir -p $@
	mkdir -p $(OBJ_DIR)/$(SRC_DIR)
	mkdir -p $(OBJ_DIR)/$(DRV_DIR)

size: $(BUILD_DIR)/$(PROJECT).elf
	@echo ""
	@echo "=== Build Size ==="
	$(SIZE) $<

clean:
	rm -rf $(BUILD_DIR)

flash: $(BUILD_DIR)/$(PROJECT).bin
	@echo "Flashing to target..."
	# Using ST-Link
	st-flash write $< 0x08000000

# Dependencies
-include $(OBJS:.o=.d)

# Generate dependency files
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

# ============================================================================
# Physics validation (host-side, not for target)
# ============================================================================

.PHONY: validate-physics

validate-physics:
	@echo "Validating physics constants..."
	@python3 -c "\
from math import sqrt, exp; \
PHI = (1 + sqrt(5)) / 2; \
PHI_INV = 1 / PHI; \
Z_C = sqrt(3) / 2; \
SIGMA = 36; \
print(f'φ = {PHI:.15f}'); \
print(f'φ⁻¹ = {PHI_INV:.15f}'); \
print(f'φ⁻¹ + φ⁻² = {PHI_INV + PHI_INV**2:.15f} (should be 1.0)'); \
print(f'z_c = {Z_C:.15f}'); \
print(f'|S|/ℏ (spin-1/2) = {sqrt(0.5 * 1.5):.15f}'); \
print(f'z_c == |S|/ℏ: {abs(Z_C - sqrt(0.5*1.5)) < 1e-14}'); \
print(f'σ = {SIGMA} = |S₃|² = {6**2}'); \
print(f'ΔS_neg(z_c) = {exp(-SIGMA * 0**2):.15f} (should be 1.0)');\
"

# ============================================================================
# Documentation
# ============================================================================

.PHONY: docs

docs:
	@echo "Generating documentation..."
	doxygen Doxyfile

# ============================================================================
# Help
# ============================================================================

.PHONY: help

help:
	@echo "Nuclear Spinner Firmware Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build firmware (default)"
	@echo "  clean            - Remove build artifacts"
	@echo "  flash            - Flash to target via ST-Link"
	@echo "  size             - Show binary size"
	@echo "  validate-physics - Validate physics constants"
	@echo "  docs             - Generate documentation"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1          - Build with debug symbols"
	@echo ""
	@echo "Examples:"
	@echo "  make             - Release build"
	@echo "  make DEBUG=1     - Debug build"
	@echo "  make flash       - Flash release build"
