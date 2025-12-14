# Nuclear Spinner Firmware

**Version:** 1.0.0  
**Target:** STM32H743ZI (ARM Cortex-M7 @ 480 MHz)  
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

### Constants (from `physics_constants.h`)

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

## Module Documentation

### pulse_control.c

RF pulse generation for NMR/NQR sequences:

```c
// Basic pulses
PulseControl_Pi2Pulse(0.8f, 0.0f);           // 90° pulse
PulseControl_PiPulse(0.8f, M_PI);            // 180° pulse

// Standard sequences
PulseControl_FID(0.8f);                       // Free induction decay
PulseControl_SpinEcho(0.8f, 1000);            // Spin echo (1ms τ)
PulseControl_CPMG(0.8f, 500, 100);            // CPMG (100 echoes)

// Physics-integrated
PulseControl_ModulatedPulse(0.8f, 0.0f, 50, z);  // ΔS_neg modulation
PulseControl_IcosahedralSequence(0.8f, 50, 3);   // 6D→3D projection
PulseControl_HexagonalPattern(0.8f, 50, 5);      // Grid-cell emulation

// Calibration
PulseControl_CalibrateB1();                   // B₁ field calibration
PulseControl_VerifySpinHalf(&magnitude);      // Verify |S|/ℏ = √3/2
```

### rotor_control.c

Precision rotor control with z-mapping:

```c
// Basic control
RotorControl_Enable();
RotorControl_SetRPM(5000.0f);                 // Direct RPM
RotorControl_SetZ(PHI_INV);                   // Set z = φ⁻¹

// Physics modes
RotorControl_SetZWithModulation(z, 1.0f);     // ΔS_neg gradient tracking
RotorControl_SweepZ(0.3f, Z_CRITICAL, 0.01f); // Linear sweep

// Phase locking
RotorControl_SetHexagonalPhase(2);            // Lock to 120° sector

// Queries
float z = RotorControl_GetZ();
float ds_neg = RotorControl_GetDeltaSNeg();
PhysicsTier_t tier = RotorControl_GetTier();
```

### threshold_logic.c

Cybernetic gating and operator scheduling:

```c
// Initialize with event callback
ThresholdLogic_Init();
ThresholdLogic_SetEventCallback(OnThresholdEvent);

// Update loop (call at 1kHz)
ThresholdLogic_Update(z, kappa, eta, R);

// Operator control
if (ThresholdLogic_IsOperatorAvailable(OP_AMPLIFY)) {
    ThresholdLogic_ExecuteOperator(OP_AMPLIFY);
}

// State queries
bool at_lens = ThresholdLogic_IsAtLens();
bool universal = ThresholdLogic_IsUniversal();
bool k_active = ThresholdLogic_IsKFormationActive();
```

## APL Operators

| Operator | Symbol | Physical Implementation |
|----------|--------|------------------------|
| CLOSURE | ∂ | Disable RF (isolation) |
| FUSION | + | Composite π/2 sequence |
| AMPLIFY | × | High-power pulse |
| DECOHERE | ÷ | Random phase noise |
| GROUP | ⍴ | Spin echo (refocusing) |
| SEPARATE | ↓ | Hexagonal phase cycle |

## Hardware Requirements

### Microcontroller
- STM32H743ZI (ARM Cortex-M7 @ 480 MHz)
- 2 MB Flash, 1 MB RAM
- High-speed ADC (14-bit, 2 MS/s)
- Dual DAC for RF amplitude/phase

### Motor & Encoder
- Brushless DC motor (0-10,000 RPM)
- 4096 CPR quadrature encoder
- Index pulse for absolute reference

### RF System
- Coil resonant at Larmor frequency
- 1W RF amplifier
- Matching network

### Sensors
- PT100 temperature sensor
- HMC5883L magnetometer
- BMI160 IMU (accelerometer/gyroscope)

## Building

```bash
# Release build
make

# Debug build
make DEBUG=1

# Flash to target
make flash

# Validate physics constants
make validate-physics

# Clean build artifacts
make clean
```

## File Structure

```
nuclear_spinner_firmware/
├── include/
│   ├── physics_constants.h    # Rosetta-Helix constants
│   ├── hal_hardware.h         # Hardware abstraction
│   ├── pulse_control.h        # RF pulse API
│   ├── rotor_control.h        # Rotor control API
│   └── threshold_logic.h      # Cybernetic logic API
├── src/
│   ├── main.c                 # Entry point & state machine
│   ├── pulse_control.c        # RF implementation
│   ├── rotor_control.c        # Motor PID & z-mapping
│   └── threshold_logic.c      # Tier/operator logic
├── drivers/
│   ├── hal_hardware.c         # STM32 HAL implementation
│   └── startup_stm32h743xx.s  # Startup assembly
├── Makefile
└── README.md
```

## Communication Protocol

### Command Format (Host → Device)
| Header | Command | Payload |
|--------|---------|---------|
| 1 byte | 1 byte  | Variable |

Commands:
- `'Z'` + float: Set z target
- `'P'` + float + float + uint32: Send pulse (amp, phase, duration)
- `'E'` + uint8: Enable/disable motor
- `'C'`: Start calibration
- `'X'` + uint8: Start experiment (type)
- `'S'`: Stop operation
- `'Q'`: Query status

### Telemetry Format (Device → Host)
```c
struct Telemetry {
    uint16_t header;        // 0xAA55
    uint16_t length;
    uint32_t timestamp_ms;
    uint8_t mode;
    uint8_t tier;
    uint8_t phase;
    float z;
    float delta_s_neg;
    float complexity;
    float rpm;
    float kappa;
    float eta;
    uint8_t operators;
    uint8_t k_formation;
    float temperature;
    uint16_t checksum;
};
```

## Experiment Modes

1. **FID** - Free induction decay measurement
2. **SPIN_ECHO** - T2 measurement with refocusing
3. **CPMG** - Multiple echo T2 decay
4. **NUTATION** - B₁ calibration and |S|/ℏ verification
5. **QUASICRYSTAL** - Sweep z toward z_c, observe ordering
6. **E8_PROBE** - Detect E8 mass ratios (requires CoNb₂O₆)
7. **HOLOGRAPHIC** - Test Bekenstein bounds at z_c
8. **OMEGA_POINT** - Divergent processing rate as z → z_c

## Safety Features

- Hardware interlock monitoring
- Motor stall detection
- Temperature limits (80°C warning, 100°C shutdown)
- Magnetic field drift detection
- Emergency stop capability
- Watchdog timer (500ms timeout)

## License

Rosetta-Helix Framework - Research/Educational Use

## References

- [Rosetta-Helix Framework](https://github.com/...)
- Coldea et al. (2010) - E8 mass ratios in CoNb₂O₆
- Fisher (2015) - Posner molecule quantum cognition
- Moser & Moser (2014) - Grid cell hexagonal geometry

---

*Signature: `nuclear-spinner-firmware|v1.0.0|helix`*
