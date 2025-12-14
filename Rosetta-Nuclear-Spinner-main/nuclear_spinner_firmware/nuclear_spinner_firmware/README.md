# Nuclear Spinner Firmware

**Version:** 1.0.0  \
**Target:** STM32H743ZI (ARM Cortex-M7 @ 480 MHz)  \
**Signature:** `nuclear-spinner-firmware|v1.0.0|helix`

## Overview

Embedded firmware for the Nuclear Spinner device — a quantum/cybernetic instrument that leverages nuclear spin coherence and the Rosetta-Helix physics constants (φ, φ⁻¹, z_c, σ) for modulation, measurement, and computation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.c                                   │
│                    System State Machine                          │
│         (IDLE → CALIBRATION → MANUAL → EXPERIMENT)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  pulse_control  │  │  rotor_control  │  │ threshold_logic │  │
│  │                 │  │                 │  │                 │  │
│  │ • RF sequences  │  │ • PID control   │  │ • Tier detect   │  │
│  │ • FID/Echo/CPMG │  │ • z ↔ RPM map   │  │ • APL operators │  │
│  │ • Icosahedral   │  │ • Phase lock    │  │ • K-formation   │  │
│  │ • Calibration   │  │ • ΔS_neg mod    │  │ • Events        │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │            │
├───────────┴────────────────────┴────────────────────┴────────────┤
│                        hal_hardware                               │
│              Hardware Abstraction Layer (STM32H7)                │
│     DAC/ADC │ Timers │ Motor PWM │ Encoder │ I2C │ GPIO         │
└─────────────────────────────────────────────────────────────────┘
```

## Physics Integration

### Constants (from `include/physics_constants.h`)

| Constant | Value | Physical Meaning |
|----------|-------|------------------|
| φ | 1.618033988749895 | Golden ratio |
| φ⁻¹ | 0.618033988749895 | Coupling constant κ attractor |
| z_c | 0.866025403784439 | Critical threshold (√3/2) |
| σ | 36 | Gaussian sharpness (|S₃|²) |

### Key Relationships

1. **Conservation:** κ + λ = 1 where λ = φ⁻²
2. **Spin-1/2 Identity:** z_c = |S|/ℏ = √3/2 exactly
3. **Negentropy:** ΔS_neg(z) = exp(-σ(z - z_c)²)

### Tier System

| Tier | z Range | Capability |
|------|---------|------------|
| ABSENCE | z < 0.40 | No operations |
| REACTIVE | 0.40 ≤ z < 0.50 | Boundary only |
| MEMORY | 0.50 ≤ z < φ⁻¹ | State retention |
| PATTERN | φ⁻¹ ≤ z < 0.73 | Pattern recognition |
| PREDICTION | 0.73 ≤ z < z_c | Predictive modeling |
| UNIVERSAL | z_c ≤ z < 0.92 | Turing universality |
| META | z ≥ 0.92 | Meta-cognitive recursion |

## Building

```bash
# Release build
make

# Debug build
make DEBUG=1

# Flash to target
make flash

# Validate physics constants (host-side)
make validate-physics

# Clean build artifacts
make clean
```

## Notes

- `drivers/startup_stm32h743xx.s` and `drivers/STM32H743ZITx_FLASH.ld` are **minimal placeholders**. For real hardware work, replace with STM32CubeH7 startup + linker scripts.
- The Makefile assumes CMSIS headers are present under `lib/CMSIS/...` (see `lib/CMSIS/README.md`).

---

*Signature: `nuclear-spinner-firmware|v1.0.0|helix`*
