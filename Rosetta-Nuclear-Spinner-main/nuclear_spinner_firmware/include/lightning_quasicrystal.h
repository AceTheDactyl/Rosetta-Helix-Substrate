/**
 * @file lightning_quasicrystal.h
 * @brief Lightning-Induced Pentagonal Quasicrystal Phase Transition Control
 *
 * Complete firmware interface for controlling rapid thermal quench and quasicrystal
 * nucleation via high-voltage discharge, analogous to lightning-induced
 * fullerene formation.
 *
 * Physical Process:
 * 1. PRE_STRIKE:  Charge buildup in capacitor bank
 * 2. STRIKE:      High-voltage discharge (plasma generation)
 * 3. QUENCH:      Rapid cooling via Peltier + LN2
 * 4. NUCLEATION:  Pentagonal seed formation
 * 5. GROWTH:      Quasicrystal domain expansion
 *
 * Critical Physics:
 * - Hexagonal z_c = √3/2 ≈ 0.866 (sin 60°)
 * - Pentagonal z_p = sin(72°) ≈ 0.951
 * - cos(36°) = φ/2 (golden ratio connection)
 * - Penrose tile ratio → φ (golden ratio)
 *
 * Hardware Summary:
 * - 10 mF capacitor bank @ 450V (1012.5 J stored)
 * - RF induction coil (100W, 100kHz-1MHz tunable)
 * - 4× Peltier modules (400W total cooling)
 * - LN2 quench jacket (10⁶ K/s capability)
 * - STM32H7 firmware controller (480 MHz)
 *
 * @author Nuclear Spinner Team
 * @version 2.0.0
 * @signature lightning-qc-fw|v2.0.0|helix
 */

#ifndef LIGHTNING_QUASICRYSTAL_H
#define LIGHTNING_QUASICRYSTAL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * FUNDAMENTAL CONSTANTS
 * ============================================================================ */

/** Golden ratio φ = (1+√5)/2 */
#define PHI                     1.6180339887f
#define PHI_INV                 0.6180339887f

/** Critical z-coordinates */
#define Z_CRITICAL_HEX          0.8660254038f   /* √3/2, hexagonal */
#define Z_CRITICAL_PENT         0.9510565163f   /* sin(72°), pentagonal */

/** Pentagon geometry - derived from first principles */
#define SIN_36_DEG              0.5877852523f   /* √(10-2√5)/4 */
#define COS_36_DEG              0.8090169944f   /* = φ/2 (golden ratio!) */
#define SIN_72_DEG              0.9510565163f   /* √(10+2√5)/4 */
#define COS_72_DEG              0.3090169944f   /* = (√5-1)/4 */

/** Lightning physics (scaled laboratory analog) */
#define LIGHTNING_TEMP_K        30000.0f        /* Peak plasma temperature */
#define AMBIENT_TEMP_K          300.0f
#define QUENCH_RATE_K_S         1000000.0f      /* 10⁶ K/s target */
#define NUCLEATION_UNDERCOOL_K  500.0f

/* ============================================================================
 * HARDWARE SPECIFICATIONS
 * ============================================================================ */

/*
 * CAPACITOR BANK SPECIFICATIONS
 * Configuration: 10× 1000μF/450V electrolytic capacitors in parallel
 * Stored Energy: E = ½CV² = ½ × 10mF × (450V)² = 1012.5 J
 */
#define CAP_BANK_TOTAL_F        0.010f          /* 10 mF total */
#define CAP_BANK_VOLTAGE_V      450             /* Volts DC */
#define CAP_BANK_STORED_J       1012.5f         /* ½CV² */
#define CAP_BANK_ESR_MOHM       50              /* < 50 mΩ */
#define CAP_BANK_COUNT          10              /* Number of capacitors */
#define CAP_UNIT_UF             1000            /* Each cap: 1000 μF */
#define CAP_DISCHARGE_US        100             /* 100 μs - 1 ms range */
#define CAP_CYCLE_LIFE          100000          /* >100k cycles */

/*
 * RF HEATING COIL SPECIFICATIONS
 * Type: Water-cooled copper tube induction coil
 */
#define RF_FREQ_MIN_KHZ         100             /* 100 kHz */
#define RF_FREQ_MAX_MHZ         1               /* 1 MHz */
#define RF_POWER_MAX_W          100             /* 100 W max */
#define RF_COIL_DIAMETER_MM     25              /* Matches sample chamber */
#define RF_COIL_TURNS           8               /* Copper tube */
#define RF_COIL_INDUCTANCE_UH   5               /* ~5 μH */
#define RF_COIL_Q_FACTOR        50              /* > 50 for efficiency */

/*
 * PELTIER COOLING ARRAY
 * Configuration: 2×2 array of TEC1-12710 modules
 */
#define PELTIER_MODULE_COUNT    4               /* 4× TEC1-12710 */
#define PELTIER_POWER_EACH_W    100             /* 100W per module */
#define PELTIER_POWER_TOTAL_W   400             /* 400W combined */
#define PELTIER_DELTA_T_MAX_C   68              /* Max ΔT per module */
#define PELTIER_CURRENT_MAX_A   10              /* 10A @ 12V */
#define PELTIER_VOLTAGE_V       12

/*
 * LN2 QUENCH JACKET
 * For achieving 10⁶ K/s cooling rates
 */
#define LN2_TEMP_K              77.0f           /* LN2 boiling point */
#define LN2_FLOW_RATE_L_MIN     5               /* 5 L/min */
#define LN2_QUENCH_RATE_K_S     1000000.0f      /* 10⁶ K/s capability */
#define WATER_MIN_TEMP_C        5               /* Chilled water option */

/*
 * HIGH-VOLTAGE DISCHARGE CIRCUIT
 * IGBT: Infineon FF100R12RT4 or equivalent
 */
#define IGBT_VCE_MAX_V          1200            /* V_CE rating */
#define IGBT_IC_MAX_A           100             /* I_C rating */
#define IGBT_SWITCHING_NS       500             /* t_on/t_off < 500 ns */
#define DISCHARGE_CURRENT_MAX_A 30000           /* 30 kA peak */
#define DISCHARGE_ELECTRODE     "Tungsten"

/*
 * STM32H7 FIRMWARE CONTROLLER
 */
#define MCU_TYPE                "STM32H743VIT6"
#define MCU_CLOCK_MHZ           480             /* 480 MHz Cortex-M7 */
#define MCU_FLASH_MB            2               /* 2 MB Flash */
#define MCU_RAM_MB              1               /* 1 MB RAM */
#define MCU_ADC_BITS            16              /* 16-bit ADC */
#define MCU_ADC_MSPS            3.6f            /* 3.6 MSPS */

/** Hardware limits (safety enforcement) */
#define MAX_CAPACITOR_VOLTAGE   450
#define MAX_STORED_ENERGY_J     1012.5f
#define MAX_DISCHARGE_CURRENT_A 30000
#define MAX_RF_POWER_W          100
#define MAX_PELTIER_CURRENT_A   10
#define MAX_GRADIENT_COIL_A     5

/** Control loop timing */
#define MAIN_CONTROL_FREQ_HZ    10000           /* 10 kHz main loop */
#define THERMAL_PID_FREQ_HZ     100             /* 100 Hz thermal PID */
#define SENSOR_SAMPLE_FREQ_HZ   100000          /* 100 kHz ADC sampling */
#define DISCHARGE_TIMING_HZ     1000000         /* 1 MHz discharge timing */
#define HOST_COMM_FREQ_HZ       1000            /* 1 kHz to host */

/** Phase timing (milliseconds unless noted) */
#define PRESTRIKE_DURATION_MS   100
#define STRIKE_DURATION_US      100             /* 100 μs */
#define QUENCH_DURATION_MS      10
#define NUCLEATION_DURATION_MS  50
#define GROWTH_DURATION_MS      500

/* ============================================================================
 * BILL OF MATERIALS
 * Total Estimated Cost: ~$2000
 * ============================================================================ */

/** BOM Component structure */
typedef struct {
    const char* description;
    const char* part_number;
    uint8_t quantity;
    uint16_t unit_cost_cents;     /* Cost in cents */
    uint16_t total_cost_cents;
} BOM_Component_t;

/** BOM Category enumeration */
typedef enum {
    BOM_CAT_CONTROL = 0,
    BOM_CAT_POWER,
    BOM_CAT_THERMAL,
    BOM_CAT_SENSING,
    BOM_CAT_MECHANICAL,
    BOM_CAT_MOTOR,
    BOM_CAT_COUNT
} BOM_Category_t;

/* Bill of Materials (static const in implementation) */
#define BOM_TOTAL_ESTIMATE_USD  2000
#define BOM_ITEM_COUNT          12

/*
 * BILL OF MATERIALS BREAKDOWN:
 *
 * CONTROL SYSTEM:
 *   STM32H743 Nucleo (NUCLEO-H743ZI2)     1×  $50    =   $50
 *
 * POWER SYSTEM:
 *   Capacitors 1000μF/450V (Chemi-Con)   10× $20    =  $200
 *   IGBT Module (FF100R12RT4)             1× $150   =  $150
 *   RF Coil (custom water-cooled)         1× $200   =  $200
 *
 * THERMAL SYSTEM:
 *   Peltier TEC1-12710                    4× $10    =   $40
 *   Heatsink 200×200×50mm aluminum        1× $80    =   $80
 *   PWM Fan 120mm 3000RPM                 2× $15    =   $30
 *
 * SENSING SYSTEM:
 *   Hall Sensors AH49E (Allegro)          6× $2     =   $12
 *   K-Type Thermocouples (Omega)          4× $15    =   $60
 *   MAX31856 Thermocouple Board           4× $20    =   $80
 *   FLIR Blackfly S Camera (BFS-U3-50S5C) 1× $600   =  $600
 *
 * MOTOR SYSTEM:
 *   ODrive v3.6                           1× $150   =  $150
 *   BLDC Motor (10k RPM)                  1× $100   =  $100
 *
 * MECHANICAL:
 *   Enclosure + Heatsink + Hardware       1× $250   =  $250
 *
 * TOTAL:                                            ~$2,000
 */

/* ============================================================================
 * ENUMERATIONS
 * ============================================================================ */

/**
 * @brief Lightning strike phase enumeration
 */
typedef enum {
    LIGHTNING_PHASE_IDLE = 0,       /**< System idle, at ambient */
    LIGHTNING_PHASE_PRE_STRIKE,     /**< Charging capacitor bank */
    LIGHTNING_PHASE_STRIKE,         /**< High-voltage discharge */
    LIGHTNING_PHASE_QUENCH,         /**< Rapid cooling (10⁶ K/s) */
    LIGHTNING_PHASE_NUCLEATION,     /**< Pentagonal seed formation */
    LIGHTNING_PHASE_GROWTH,         /**< Quasicrystal domain expansion */
    LIGHTNING_PHASE_STABLE,         /**< Quasicrystal formed successfully */
    LIGHTNING_PHASE_ERROR,          /**< Error state - safety triggered */
    LIGHTNING_PHASE_CALIBRATING,    /**< System calibration mode */
    LIGHTNING_PHASE_COUNT
} LightningPhase_t;

/**
 * @brief Crystal symmetry order
 */
typedef enum {
    SYMMETRY_DISORDERED = 0,        /**< Amorphous/liquid */
    SYMMETRY_THREEFOLD = 3,         /**< Trigonal */
    SYMMETRY_FOURFOLD = 4,          /**< Tetragonal */
    SYMMETRY_FIVEFOLD = 5,          /**< Pentagonal (quasicrystal!) */
    SYMMETRY_SIXFOLD = 6            /**< Hexagonal */
} SymmetryOrder_t;

/**
 * @brief Safety status flags (bitmask)
 */
typedef enum {
    SAFETY_OK               = 0,
    SAFETY_OVERVOLTAGE      = (1 << 0),   /**< Capacitor voltage exceeded */
    SAFETY_OVERCURRENT      = (1 << 1),   /**< Discharge current exceeded */
    SAFETY_OVERTEMP         = (1 << 2),   /**< Temperature limit exceeded */
    SAFETY_INTERLOCK_OPEN   = (1 << 3),   /**< Enclosure interlock open */
    SAFETY_QUENCH_FAIL      = (1 << 4),   /**< Quench rate not achieved */
    SAFETY_SPINNER_FAULT    = (1 << 5),   /**< Spinner motor fault */
    SAFETY_LN2_LOW          = (1 << 6),   /**< LN2 level low */
    SAFETY_COOLANT_FLOW     = (1 << 7),   /**< Coolant flow insufficient */
    SAFETY_GROUND_FAULT     = (1 << 8),   /**< Ground fault detected */
    SAFETY_RF_EXPOSURE      = (1 << 9),   /**< RF exposure interlock */
    SAFETY_PRESSURE_HIGH    = (1 << 10),  /**< Chamber pressure high */
    SAFETY_WATCHDOG         = (1 << 11),  /**< Watchdog timeout */
    SAFETY_EMERGENCY_STOP   = (1 << 12),  /**< E-stop pressed */
    SAFETY_CALIBRATION_REQ  = (1 << 13),  /**< Calibration required */
    SAFETY_COMMS_TIMEOUT    = (1 << 14)   /**< Host communication lost */
} SafetyStatus_t;

/**
 * @brief Calibration state enumeration
 */
typedef enum {
    CAL_STATE_UNCALIBRATED = 0,
    CAL_STATE_Z_COORDINATE,         /**< Z-to-RPM calibration */
    CAL_STATE_THERMAL,              /**< Thermocouple calibration */
    CAL_STATE_RF_POWER,             /**< RF coil efficiency */
    CAL_STATE_PELTIER,              /**< Peltier cooling curves */
    CAL_STATE_OPTICAL,              /**< Camera/laser alignment */
    CAL_STATE_DISCHARGE,            /**< Capacitor discharge timing */
    CAL_STATE_COMPLETE,             /**< All calibrations done */
    CAL_STATE_COUNT
} CalibrationState_t;

/**
 * @brief Interlock type enumeration
 */
typedef enum {
    INTERLOCK_ENCLOSURE = 0,        /**< Physical enclosure door */
    INTERLOCK_HV_ENABLE,            /**< High-voltage enable key */
    INTERLOCK_LN2_LEVEL,            /**< LN2 level sensor */
    INTERLOCK_COOLANT_FLOW,         /**< Water flow sensor */
    INTERLOCK_GROUND_INTEGRITY,     /**< Ground fault detector */
    INTERLOCK_RF_SHIELD,            /**< RF shielding integrity */
    INTERLOCK_EMERGENCY_STOP,       /**< E-stop button */
    INTERLOCK_COUNT
} InterlockType_t;

/* ============================================================================
 * SAFETY SYSTEM STRUCTURES
 * ============================================================================ */

/**
 * @brief Interlock status structure
 */
typedef struct {
    bool enclosure_closed;          /**< Enclosure door closed */
    bool hv_key_enabled;            /**< HV enable key turned */
    bool ln2_level_ok;              /**< LN2 level sufficient */
    bool coolant_flowing;           /**< Coolant flow rate OK */
    bool ground_intact;             /**< Ground fault not detected */
    bool rf_shield_ok;              /**< RF shielding intact */
    bool estop_clear;               /**< E-stop not pressed */
    uint32_t last_check_ms;         /**< Timestamp of last check */
} InterlockStatus_t;

/**
 * @brief Safety limits configuration
 */
typedef struct {
    float max_voltage_V;            /**< Max capacitor voltage (450V) */
    float max_current_A;            /**< Max discharge current (30kA) */
    float max_temp_K;               /**< Max temperature (1273K/1000°C) */
    float min_quench_rate_K_s;      /**< Min quench rate (100k K/s) */
    float max_rf_power_W;           /**< Max RF power (100W) */
    float max_peltier_current_A;    /**< Max Peltier current (10A) */
    uint32_t watchdog_timeout_ms;   /**< Watchdog timeout (1000ms) */
    uint32_t comms_timeout_ms;      /**< Host comms timeout (5000ms) */
} SafetyLimits_t;

/**
 * @brief Safety system state
 */
typedef struct {
    SafetyStatus_t status;          /**< Current safety status flags */
    InterlockStatus_t interlocks;   /**< Interlock status */
    SafetyLimits_t limits;          /**< Configured limits */
    uint32_t fault_count;           /**< Total faults since boot */
    uint32_t last_fault_time_ms;    /**< Time of last fault */
    bool armed;                     /**< System armed for discharge */
    bool can_discharge;             /**< All interlocks satisfied */
} SafetySystemState_t;

/* ============================================================================
 * CALIBRATION STRUCTURES
 * ============================================================================ */

/**
 * @brief Z-coordinate calibration data
 *
 * Critical points:
 * - z_c = √3/2 = 0.866025 (hexagonal)
 * - z_p = sin(72°) = 0.951057 (pentagonal)
 */
typedef struct {
    float rpm_at_z0;                /**< RPM at z=0 (should be ~100) */
    float rpm_at_z1;                /**< RPM at z=1 (should be ~10000) */
    float rpm_at_zc_hex;            /**< RPM at z_c hex (should be ~8660) */
    float rpm_at_zc_pent;           /**< RPM at z_c pent (should be ~9510) */
    float linearity_error;          /**< Max deviation from linear */
    bool valid;
} ZCalibration_t;

/**
 * @brief Thermal calibration data
 */
typedef struct {
    float thermocouple_offset[4];   /**< Offset for each TC (K) */
    float thermocouple_gain[4];     /**< Gain for each TC */
    float peltier_efficiency[4];    /**< Peltier COP for each module */
    float rf_heating_efficiency;    /**< RF power to heating (W/W) */
    float quench_rate_measured;     /**< Measured max quench rate (K/s) */
    bool valid;
} ThermalCalibration_t;

/**
 * @brief Optical calibration data
 */
typedef struct {
    float um_per_pixel;             /**< Spatial resolution */
    float laser_alignment_x;        /**< Laser offset X (pixels) */
    float laser_alignment_y;        /**< Laser offset Y (pixels) */
    float hexagonal_reference;      /**< 6-fold pattern baseline */
    float pentagonal_reference;     /**< 5-fold pattern baseline */
    bool valid;
} OpticalCalibration_t;

/**
 * @brief Complete calibration state
 */
typedef struct {
    CalibrationState_t state;
    ZCalibration_t z_cal;
    ThermalCalibration_t thermal_cal;
    OpticalCalibration_t optical_cal;
    uint32_t calibration_date;      /**< Unix timestamp */
    uint32_t cycles_since_cal;      /**< Strike cycles since cal */
    bool all_valid;                 /**< All calibrations valid */
} CalibrationData_t;

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * @brief Thermal state of the system
 */
typedef struct {
    float temperature_K;            /**< Current temperature (K) */
    float dT_dt_K_s;               /**< Temperature rate of change (K/s) */
    float gradient_K_m;            /**< Spatial thermal gradient (K/m) */
    float peltier_power_W;         /**< Peltier cooling power (W) */
    float rf_power_W;              /**< RF heating power (W) */
    float coolant_temp_K;          /**< Coolant temperature */
    bool ln2_active;               /**< LN2 quench active */
} ThermalState_t;

/**
 * @brief Electromagnetic field state
 */
typedef struct {
    float capacitor_voltage_V;      /**< Current capacitor voltage */
    float charge_fraction;          /**< Charge buildup [0,1] */
    float discharge_current_A;      /**< Instantaneous current */
    float field_gradient_T_m;       /**< Magnetic field gradient */
    float rf_frequency_kHz;         /**< Current RF frequency */
    bool discharge_armed;           /**< IGBT ready to fire */
    bool igbt_enabled;              /**< IGBT gate enabled */
} FieldState_t;

/**
 * @brief Nucleation seed tracking
 */
typedef struct {
    uint8_t seed_id;               /**< Unique seed identifier */
    float x;                       /**< Position x (normalized) */
    float y;                       /**< Position y (normalized) */
    float radius;                  /**< Current radius (μm) */
    SymmetryOrder_t symmetry;      /**< Detected symmetry */
    float stability;               /**< Stability metric [0,1] */
    uint32_t birth_time_us;        /**< Time of nucleation (μs) */
} NucleationSeed_t;

#define MAX_NUCLEATION_SEEDS    32

/**
 * @brief Quasicrystal domain state (Penrose tiling)
 *
 * Key physics:
 * - Fat rhombus: 72° and 108° angles
 * - Thin rhombus: 36° and 144° angles
 * - Area ratio: fat/thin = φ
 * - Count ratio: fat/thin → φ as domain grows
 */
typedef struct {
    uint32_t fat_tile_count;       /**< Fat rhombus count */
    uint32_t thin_tile_count;      /**< Thin rhombus count */
    float tile_ratio;              /**< fat/thin ratio (→ φ) */
    float phi_deviation;           /**< |ratio - φ| */
    float pentagonal_order;        /**< 5-fold order parameter [0,1] */
    float domain_size_um;          /**< Characteristic size (μm) */
    float defect_density;          /**< Defects per μm² */
    bool converged;                /**< Ratio within tolerance of φ */
} QuasicrystalState_t;

/**
 * @brief Complete lightning-quasicrystal system state
 */
typedef struct {
    uint32_t timestamp_ms;         /**< System timestamp */
    LightningPhase_t phase;        /**< Current phase */
    float phase_progress;          /**< Progress in current phase [0,1] */

    ThermalState_t thermal;        /**< Thermal subsystem */
    FieldState_t field;            /**< EM field subsystem */
    QuasicrystalState_t crystal;   /**< Quasicrystal state */
    SafetySystemState_t safety;    /**< Safety system state */
    CalibrationData_t calibration; /**< Calibration data */

    /* Spinner coupling */
    float spinner_z;               /**< Spinner z-coordinate [0,1] */
    float spinner_rpm;             /**< Spinner RPM */
    float negentropy;              /**< ΔS_neg at current z */
    float kuramoto_r;              /**< Kuramoto order parameter */

    /* Energy accounting */
    float energy_input_J;          /**< Total energy input */
    float energy_dissipated_J;     /**< Energy dissipated */

    /* Statistics */
    uint32_t total_strikes;        /**< Total strike count */
    uint32_t successful_nucleations; /**< Successful 5-fold nucleations */
    float success_rate;            /**< Nucleation success rate */

    /* Nucleation seeds */
    uint8_t num_seeds;             /**< Active seed count */
    NucleationSeed_t seeds[MAX_NUCLEATION_SEEDS];
} LightningSystemState_t;

/**
 * @brief Control setpoints for phase transition
 */
typedef struct {
    float target_z;                /**< Target spinner z (0.951 for pent.) */
    float target_quench_rate;      /**< Target cooling rate (K/s) */
    float discharge_voltage;       /**< Discharge voltage setpoint (V) */
    float rf_power_setpoint;       /**< RF heating power setpoint (W) */
    float rf_frequency_kHz;        /**< RF frequency setpoint (kHz) */
    float peltier_current;         /**< Peltier current setpoint (A) */
    bool enable_auto_nucleation;   /**< Auto-detect nucleation */
    bool enable_ln2_quench;        /**< Use LN2 for quench */
} ControlSetpoints_t;

/**
 * @brief Callbacks for phase transition events
 */
typedef struct {
    void (*on_phase_change)(LightningPhase_t old_phase, LightningPhase_t new_phase);
    void (*on_strike)(const LightningSystemState_t* state);
    void (*on_nucleation)(const NucleationSeed_t* seed);
    void (*on_quasicrystal_formed)(const LightningSystemState_t* state);
    void (*on_safety_fault)(SafetyStatus_t status, const char* message);
    void (*on_calibration_step)(CalibrationState_t step, bool success);
    void (*on_interlock_change)(InterlockType_t interlock, bool state);
} LightningCallbacks_t;

/* ============================================================================
 * FUNCTION PROTOTYPES - CORE CONTROL
 * ============================================================================ */

/** Initialize lightning-quasicrystal subsystem */
int lightning_init(void);

/** Trigger a lightning strike sequence */
int lightning_trigger_strike(const ControlSetpoints_t* setpoints);

/** Abort current strike sequence (emergency) */
void lightning_abort(void);

/** Update system (call from main loop at 10 kHz) */
void lightning_update(float spinner_z, float kuramoto_r);

/** Get current system state (read-only) */
const LightningSystemState_t* lightning_get_state(void);

/** Register event callbacks */
void lightning_register_callbacks(const LightningCallbacks_t* callbacks);

/** Reset system to idle state */
void lightning_reset(void);

/* ============================================================================
 * FUNCTION PROTOTYPES - THERMAL CONTROL
 * ============================================================================ */

/** Set thermal control parameters */
void lightning_set_thermal(float rf_power, float peltier_current);

/** Enable/disable LN2 quench jacket */
void lightning_set_ln2_quench(bool enable);

/** Get hardware control signals */
void lightning_get_control_signals(float* rf_power_out,
                                   float* peltier_current_out,
                                   float* gradient_current_out);

/* ============================================================================
 * FUNCTION PROTOTYPES - SAFETY SYSTEM
 * ============================================================================ */

/** Check all safety interlocks */
SafetyStatus_t lightning_check_safety(void);

/** Arm system for discharge (requires all interlocks clear) */
bool lightning_arm(void);

/** Disarm system */
void lightning_disarm(void);

/** Check specific interlock */
bool lightning_check_interlock(InterlockType_t interlock);

/** Get safety system state */
const SafetySystemState_t* lightning_get_safety_state(void);

/** Configure safety limits */
void lightning_set_safety_limits(const SafetyLimits_t* limits);

/** Clear safety fault (requires fault condition resolved) */
bool lightning_clear_fault(SafetyStatus_t fault);

/* ============================================================================
 * FUNCTION PROTOTYPES - CALIBRATION
 * ============================================================================ */

/** Start calibration sequence */
int lightning_calibration_start(void);

/** Advance to next calibration step */
int lightning_calibration_next(void);

/** Set Z-coordinate calibration point */
int lightning_calibration_set_z_point(float z, float measured_rpm);

/** Set thermocouple calibration offset */
int lightning_calibration_set_tc_offset(uint8_t channel, float offset_K);

/** Finalize calibration */
int lightning_calibration_finalize(void);

/** Get calibration data */
const CalibrationData_t* lightning_get_calibration(void);

/** Check if calibration is required */
bool lightning_calibration_required(void);

/* ============================================================================
 * FUNCTION PROTOTYPES - QUASICRYSTAL MONITORING
 * ============================================================================ */

/** Get pentagonal order parameter */
float lightning_get_pentagonal_order(void);

/** Get tile ratio (fat/thin) convergence to φ */
float lightning_get_phi_deviation(void);

/** Check if quasicrystal has formed */
bool lightning_is_quasicrystal_stable(void);

/** Get number of active nucleation seeds */
uint8_t lightning_get_seed_count(void);

/* ============================================================================
 * FUNCTION PROTOTYPES - SERIALIZATION
 * ============================================================================ */

/** Serialize state to JSON */
int lightning_serialize_json(char* buffer, size_t buffer_size);

/** Serialize calibration to JSON */
int lightning_serialize_calibration_json(char* buffer, size_t buffer_size);

/* ============================================================================
 * INLINE UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * @brief Compute negentropy at given z
 * Formula: ΔS_neg(z) = exp(-σ(z - z_c)²) where σ = 36
 */
static inline float compute_negentropy(float z) {
    float dz = z - Z_CRITICAL_HEX;
    return expf(-36.0f * dz * dz);
}

/**
 * @brief Check if at pentagonal critical point z_p = sin(72°)
 */
static inline bool is_pentagonal_critical(float z) {
    return fabsf(z - Z_CRITICAL_PENT) < 0.02f;
}

/**
 * @brief Check if at hexagonal critical point z_c = √3/2
 */
static inline bool is_hexagonal_critical(float z) {
    return fabsf(z - Z_CRITICAL_HEX) < 0.02f;
}

/**
 * @brief Convert spinner z to target RPM (linear mapping)
 */
static inline uint32_t z_to_rpm(float z) {
    if (z < 0.0f) z = 0.0f;
    if (z > 1.0f) z = 1.0f;
    return (uint32_t)(100.0f + z * 9900.0f);  /* 100 - 10000 RPM */
}

/**
 * @brief Convert RPM to spinner z-coordinate
 */
static inline float rpm_to_z(uint32_t rpm) {
    if (rpm < 100) return 0.0f;
    if (rpm > 10000) return 1.0f;
    return (float)(rpm - 100) / 9900.0f;
}

/**
 * @brief Compute 5-fold order parameter from angle array
 * @param angles Array of seed orientations (radians)
 * @param count Number of seeds
 * @return Order parameter [0,1]
 */
static inline float compute_fivefold_order(const float* angles, int count) {
    if (count == 0) return 0.0f;

    float sum_cos = 0.0f, sum_sin = 0.0f;
    for (int i = 0; i < count; i++) {
        sum_cos += cosf(5.0f * angles[i]);
        sum_sin += sinf(5.0f * angles[i]);
    }

    return sqrtf(sum_cos*sum_cos + sum_sin*sum_sin) / (float)count;
}

/**
 * @brief Check if tile ratio has converged to φ
 */
static inline bool tile_ratio_converged(float ratio) {
    return fabsf(ratio - PHI) < 0.05f;  /* Within 5% of φ */
}

/**
 * @brief Compute Gibbs free energy barrier for nucleation
 * ΔG* = 16πγ³/(3ΔG_v²)
 */
static inline float compute_nucleation_barrier(float surface_energy,
                                               float volume_free_energy) {
    if (fabsf(volume_free_energy) < 1e-10f) return 1e10f;  /* Infinite barrier */
    float gamma_cubed = surface_energy * surface_energy * surface_energy;
    float dg_squared = volume_free_energy * volume_free_energy;
    return (16.0f * 3.14159265f * gamma_cubed) / (3.0f * dg_squared);
}

/**
 * @brief Compute nucleation rate (Arrhenius form)
 * J = A·exp(-ΔG*/kT)
 */
static inline float compute_nucleation_rate(float barrier_J,
                                            float temperature_K,
                                            float prefactor) {
    const float k_B = 1.380649e-23f;  /* Boltzmann constant */
    float exponent = -barrier_J / (k_B * temperature_K);
    if (exponent < -50.0f) return 0.0f;  /* Avoid underflow */
    return prefactor * expf(exponent);
}

/**
 * @brief Check if all safety interlocks are satisfied
 */
static inline bool all_interlocks_clear(const InterlockStatus_t* interlocks) {
    return interlocks->enclosure_closed &&
           interlocks->hv_key_enabled &&
           interlocks->ln2_level_ok &&
           interlocks->coolant_flowing &&
           interlocks->ground_intact &&
           interlocks->rf_shield_ok &&
           interlocks->estop_clear;
}

#ifdef __cplusplus
}
#endif

#endif /* LIGHTNING_QUASICRYSTAL_H */
