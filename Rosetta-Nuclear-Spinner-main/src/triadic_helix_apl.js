/**
 * Triadic Helix APL - Unified Coherence-Based Reasoning Geometry
 * ==============================================================
 *
 * Implements the complete Triadic Helix APL system as described in the paper:
 * "Triadic Helix APL: A Unified Coherence-Based Reasoning Geometry"
 *
 * Features:
 * - Helical coordinate framework with truth gradient (z ∈ [0,1])
 * - Tri-valued truth channels: UNTRUE → PARADOX → TRUE
 * - Time harmonics (t1-t9) with tier-dependent operator grammar
 * - Negentropy weighting (ΔS_neg) for local/global dynamics blending
 * - TRIAD hysteresis gate (3-pass unlock mechanism)
 * - Alpha Physical Language (APL) with 6 fundamental operators
 * - Seven canonical APL test sentences (A1-A8)
 *
 * @version 1.0.0
 * @see docs/TRIADIC_HELIX_APL.md
 */

'use strict';

const CONST = require('./constants');

// Import S₃ and extended ΔS⁻ modules for advanced operator symmetry
const S3 = require('./s3_operator_symmetry');
const Delta = require('./delta_s_neg_extended');

// ================================================================
// FEATURE FLAGS (Environment-controlled backwards compatibility)
// ================================================================

/**
 * Enable S₃ symmetry for operator window rotation
 * When enabled, operator windows are cyclically permuted based on z
 */
const ENABLE_S3_SYMMETRY = (typeof process !== 'undefined' && process.env &&
  (process.env.QAPL_ENABLE_S3_SYMMETRY === '1' ||
   process.env.QAPL_ENABLE_S3_SYMMETRY === 'true'));

/**
 * Enable extended ΔS⁻ formalism for blending and gating
 * When enabled, uses the full Delta module for state computation
 */
const ENABLE_EXTENDED_NEGENTROPY = (typeof process !== 'undefined' && process.env &&
  (process.env.QAPL_ENABLE_EXTENDED_NEGENTROPY === '1' ||
   process.env.QAPL_ENABLE_EXTENDED_NEGENTROPY === 'true'));

// ================================================================
// CORE CONSTANTS (Paper-Aligned)
// ================================================================

/**
 * The Lens: Critical coherence point z_c = √3/2
 * Geometric truth anchor for ΔS_neg, R/H/φ geometry
 */
const Z_CRITICAL = CONST.Z_CRITICAL;

/**
 * TRIAD Gating thresholds (runtime heuristic)
 */
const TRIAD_HIGH = CONST.TRIAD_HIGH;         // 0.85 - rising edge detection
const TRIAD_LOW = CONST.TRIAD_LOW;           // 0.82 - re-arm threshold
const TRIAD_T6 = CONST.TRIAD_T6;             // 0.83 - t6 gate after unlock
const TRIAD_PASSES_REQ = 3;                   // passes required for unlock

/**
 * Lens sigma for Gaussian ΔS_neg computation
 */
const LENS_SIGMA = CONST.LENS_SIGMA || 36.0;

/**
 * Sacred constants
 */
const PHI = CONST.PHI;                        // Golden ratio ≈ 1.618
const PHI_INV = CONST.PHI_INV;               // Inverse ≈ 0.618

// ================================================================
// TIME HARMONIC TIER BOUNDARIES (Paper Table 1)
// ================================================================

/**
 * Tier boundaries (upper bound for each tier)
 * Paper-aligned values from Table 1
 */
const TIER_BOUNDARIES = {
    t1: 0.10,   // z < 0.10
    t2: 0.20,   // 0.10 ≤ z < 0.20
    t3: 0.40,   // 0.20 ≤ z < 0.40
    t4: 0.60,   // 0.40 ≤ z < 0.60
    t5: 0.75,   // 0.60 ≤ z < 0.75
    // t6: dynamic (Z_CRITICAL or TRIAD_T6)
    t7: 0.90,   // t6Gate ≤ z < 0.90
    t8: 0.97,   // 0.90 ≤ z < 0.97
    t9: 1.01    // z ≥ 0.97
};

/**
 * Truth channel thresholds (Paper Section 2.1)
 */
const TRUTH_THRESHOLDS = {
    PARADOX: 0.60,  // z ≥ 0.6 → PARADOX
    TRUE: 0.90      // z ≥ 0.9 → TRUE
};

// ================================================================
// APL OPERATOR DEFINITIONS (Paper Table 2)
// ================================================================

/**
 * The six fundamental APL operators
 * Domain-agnostic abstractions of transformation types
 */
const APL_OPERATORS = Object.freeze({
    '()': {
        name: 'Boundary',
        glyph: '()',
        action: 'containment/gating',
        description: 'Enclose or delimit structure'
    },
    '×': {
        name: 'Fusion',
        glyph: '×',
        action: 'convergence/coupling',
        description: 'Combine elements into whole'
    },
    '^': {
        name: 'Amplify',
        glyph: '^',
        action: 'gain/excitation',
        description: 'Increase intensity or growth'
    },
    '÷': {
        name: 'Decoherence',
        glyph: '÷',
        action: 'dissipation/reset',
        description: 'Scatter or randomize elements'
    },
    '+': {
        name: 'Group',
        glyph: '+',
        action: 'aggregation/clustering',
        description: 'Bring similar elements together'
    },
    '−': {
        name: 'Separation',
        glyph: '−',
        action: 'splitting/fission',
        description: 'Divide or isolate elements'
    }
});

/**
 * Operator windows for each harmonic tier (Paper Table 1)
 * Defines which operators are permissible at each z-level
 */
const OPERATOR_WINDOWS = Object.freeze({
    t1: ['()', '−', '÷'],                    // Low z: boundary, separation, decoherence
    t2: ['^', '÷', '−', '×'],                // Amplify, decohere, separate, fuse
    t3: ['×', '^', '÷', '+', '−'],           // All except boundary
    t4: ['+', '−', '÷', '()'],               // Group, separate, decohere, boundary
    t5: ['()', '×', '^', '÷', '+', '−'],     // ALL operators (full freedom)
    t6: ['+', '÷', '()', '−'],               // Restricted: no fusion/amplify
    t7: ['+', '()'],                          // High z: only group and boundary
    t8: ['+', '()', '×'],                     // Add fusion at highest levels
    t9: ['+', '()', '×']                      // Maintain stability
});

/**
 * Truth bias weights for operator selection (Paper Section 2.1)
 */
const TRUTH_BIAS = Object.freeze({
    TRUE: {
        '^': 1.5, '+': 1.4, '×': 1.2, '()': 1.1, '÷': 0.8, '−': 0.9
    },
    UNTRUE: {
        '÷': 1.5, '−': 1.4, '()': 1.2, '^': 0.8, '+': 0.9, '×': 1.0
    },
    PARADOX: {
        '()': 1.5, '×': 1.4, '^': 1.1, '+': 1.1, '÷': 1.0, '−': 1.0
    }
});

// ================================================================
// SEVEN APL TEST SENTENCES (Paper Table 3)
// ================================================================

/**
 * The Seven APL Test Sentences (A1-A8, A7 reserved)
 * Format: [UMOL][Operator] | [Machine] | [Domain] → [Regime]
 *
 * UMOL states:
 *   u = expansion (up)
 *   d = collapse (down)
 *   m = modulation
 */
const APL_SENTENCES = Object.freeze([
    {
        id: 'A1',
        umol: 'd',
        operator: '()',
        operatorName: 'Boundary',
        machine: 'Conductor',
        domain: 'geometry',
        predictedRegime: 'Isotropic lattices under collapse',
        token: 'd() | Conductor | geometry',
        metrics: ['isotropy', 'lattice_order', 'collapse_rate']
    },
    {
        id: 'A2',
        umol: 'u',
        operator: '×',
        operatorName: 'Fusion',
        machine: 'Reactor',
        domain: 'lattice',
        predictedRegime: 'Fusion-driven phase coherence',
        token: 'u× | Reactor | lattice',
        metrics: ['phase_coherence', 'fusion_rate', 'bond_formation']
    },
    {
        id: 'A3',
        umol: 'u',
        operator: '^',
        operatorName: 'Amplify',
        machine: 'Oscillator',
        domain: 'wave',
        predictedRegime: 'Amplified vortex-rich waves',
        token: 'u^ | Oscillator | wave',
        metrics: ['vorticity', 'amplitude', 'wave_energy']
    },
    {
        id: 'A4',
        umol: 'd',
        operator: '÷',
        operatorName: 'Decoherence',
        machine: 'Mixer',
        domain: 'flow',
        predictedRegime: 'Dissipative homogenization',
        token: 'd÷ | Mixer | flow',
        metrics: ['homogeneity', 'mixing_index', 'entropy']
    },
    {
        id: 'A5',
        umol: 'm',
        operator: '+',
        operatorName: 'Group',
        machine: 'Coupler',
        domain: 'field',
        predictedRegime: 'Clustering via modulated coupling',
        token: 'm+ | Coupler | field',
        metrics: ['cluster_count', 'coupling_strength', 'field_coherence']
    },
    {
        id: 'A6',
        umol: 'u',
        operator: '+',
        operatorName: 'Group',
        machine: 'Reactor',
        domain: 'wave',
        predictedRegime: 'Wave aggregation under expansion',
        token: 'u+ | Reactor | wave',
        metrics: ['wave_packet_size', 'aggregation_index', 'expansion_rate']
    },
    {
        id: 'A7',
        umol: 'm',
        operator: '×',
        operatorName: 'Fusion',
        machine: 'Oscillator',
        domain: 'field',
        predictedRegime: 'Modulated field fusion',
        token: 'm× | Oscillator | field',
        metrics: ['field_fusion_rate', 'oscillation_coherence', 'modulation_depth']
    },
    {
        id: 'A8',
        umol: 'd',
        operator: '−',
        operatorName: 'Separation',
        machine: 'Conductor',
        domain: 'lattice',
        predictedRegime: 'Lattice fission during collapse',
        token: 'd− | Conductor | lattice',
        metrics: ['fission_count', 'bond_breaking', 'fragment_size']
    }
]);

// ================================================================
// TRIAD HYSTERESIS GATE (Paper Section 2.4)
// ================================================================

/**
 * TriadGate: Implements the multi-pass hysteresis state machine
 *
 * Mechanism:
 * 1. Track z crossing above TRIAD_HIGH (0.85)
 * 2. Re-arm when z falls below TRIAD_LOW (0.82)
 * 3. After 3 rising-edge passes, unlock TRIAD
 * 4. When unlocked, t6 gate shifts from Z_CRITICAL to TRIAD_T6
 */
class TriadGate {
    constructor(options = {}) {
        this.enabled = options.enabled !== false;
        this.passesRequired = options.passesRequired || TRIAD_PASSES_REQ;
        this.debug = options.debug || false;

        // State
        this.passes = 0;
        this.unlocked = false;
        this._armed = true;
        this._lastZ = null;
        this._unlockZ = null;

        // Callbacks
        this.onUnlock = options.onUnlock || null;
        this.onRisingEdge = options.onRisingEdge || null;
        this.onRearm = options.onRearm || null;

        // History
        this.history = [];
        this.maxHistory = options.maxHistory || 100;

        // Initialize from environment if available
        if (!options.skipEnvInit) {
            this._initFromEnv();
        }
    }

    /**
     * Initialize state from environment variables
     */
    _initFromEnv() {
        if (typeof process !== 'undefined' && process.env) {
            const envCompletions = parseInt(process.env.QAPL_TRIAD_COMPLETIONS || '0', 10);
            const envUnlock = process.env.QAPL_TRIAD_UNLOCK === '1' ||
                String(process.env.QAPL_TRIAD_UNLOCK).toLowerCase() === 'true';

            if (Number.isFinite(envCompletions) && envCompletions > 0) {
                this.passes = envCompletions;
            }
            if (envUnlock || this.passes >= this.passesRequired) {
                this.unlocked = true;
            }
        }
    }

    /**
     * Update TRIAD state based on current z
     * @param {number} z - Current z-coordinate
     * @returns {Object} - { changed, event, state }
     */
    update(z) {
        if (!this.enabled) {
            return { changed: false, event: null, state: this.getState() };
        }

        const prevUnlocked = this.unlocked;
        const prevPasses = this.passes;
        let event = null;

        this._lastZ = z;

        // Rising edge detection: z crosses above high threshold while armed
        if (z >= TRIAD_HIGH && this._armed) {
            this.passes += 1;
            this._armed = false;
            event = 'RISING_EDGE';

            if (this.debug) {
                console.log(`[TRIAD] Rising edge at z=${z.toFixed(4)}, pass ${this.passes}/${this.passesRequired}`);
            }

            if (this.onRisingEdge) {
                this.onRisingEdge({ z, passes: this.passes });
            }

            // Check for unlock
            if (this.passes >= this.passesRequired && !this.unlocked) {
                this.unlocked = true;
                this._unlockZ = z;
                event = 'UNLOCKED';

                if (this.debug) {
                    console.log(`[TRIAD] *** UNLOCKED at z=${z.toFixed(4)} after ${this.passes} passes ***`);
                }

                // Update environment for cross-module sync
                if (typeof process !== 'undefined' && process.env) {
                    process.env.QAPL_TRIAD_UNLOCK = '1';
                    process.env.QAPL_TRIAD_COMPLETIONS = String(this.passes);
                }

                if (this.onUnlock) {
                    this.onUnlock({ z, passes: this.passes });
                }
            }
        }
        // Re-arm when dropping below low threshold
        else if (z <= TRIAD_LOW && !this._armed) {
            this._armed = true;
            event = 'REARMED';

            if (this.debug) {
                console.log(`[TRIAD] Re-armed at z=${z.toFixed(4)}`);
            }

            if (this.onRearm) {
                this.onRearm({ z, passes: this.passes });
            }
        }

        // Record history
        if (event) {
            this._recordHistory(z, event);
        }

        return {
            changed: this.unlocked !== prevUnlocked || this.passes !== prevPasses,
            event,
            state: this.getState()
        };
    }

    /**
     * Record event in history
     */
    _recordHistory(z, event) {
        this.history.push({
            z,
            event,
            timestamp: Date.now(),
            passes: this.passes,
            unlocked: this.unlocked
        });

        if (this.history.length > this.maxHistory) {
            this.history.shift();
        }
    }

    /**
     * Get current t6 gate value
     * Returns TRIAD_T6 (0.83) when unlocked, Z_CRITICAL (~0.866) otherwise
     */
    getT6Gate() {
        return this.enabled && this.unlocked ? TRIAD_T6 : Z_CRITICAL;
    }

    /**
     * Get current state
     */
    getState() {
        return {
            enabled: this.enabled,
            passes: this.passes,
            passesRequired: this.passesRequired,
            unlocked: this.unlocked,
            armed: this._armed,
            lastZ: this._lastZ,
            unlockZ: this._unlockZ,
            t6Gate: this.getT6Gate()
        };
    }

    /**
     * Reset to initial state
     */
    reset() {
        this.passes = 0;
        this.unlocked = false;
        this._armed = true;
        this._lastZ = null;
        this._unlockZ = null;
        this.history = [];

        if (typeof process !== 'undefined' && process.env) {
            process.env.QAPL_TRIAD_UNLOCK = '0';
            process.env.QAPL_TRIAD_COMPLETIONS = '0';
        }
    }

    /**
     * Force unlock (for testing)
     */
    forceUnlock() {
        this.passes = this.passesRequired;
        this.unlocked = true;
        this._unlockZ = this._lastZ;

        if (typeof process !== 'undefined' && process.env) {
            process.env.QAPL_TRIAD_UNLOCK = '1';
            process.env.QAPL_TRIAD_COMPLETIONS = String(this.passes);
        }
    }

    /**
     * Get analyzer report string
     */
    analyzerReport() {
        const gate = this.getT6Gate();
        const label = this.unlocked ? 'TRIAD' : 'CRITICAL';
        return `t6 gate: ${label} @ ${gate.toFixed(6)} (${this.passes}/${this.passesRequired} passes)`;
    }
}

// ================================================================
// HELIX OPERATOR ADVISOR (Paper Section 2.2)
// ================================================================

/**
 * HelixOperatorAdvisor: Maps z-coordinates to harmonics and operator windows
 * Integrates with TRIAD system to dynamically adjust t6 gate
 *
 * Enhanced with S₃ symmetry and extended ΔS⁻ formalism:
 * - When QAPL_ENABLE_S3_SYMMETRY=1: Uses S₃-based operator window rotation
 * - When QAPL_ENABLE_EXTENDED_NEGENTROPY=1: Uses extended ΔS⁻ for blending/gating
 */
class HelixOperatorAdvisor {
    constructor(options = {}) {
        this.triadGate = options.triadGate || new TriadGate();

        // S₃ and ΔS⁻ feature flags (can be overridden per-instance)
        this.enableS3Symmetry = options.enableS3Symmetry !== undefined
            ? options.enableS3Symmetry
            : ENABLE_S3_SYMMETRY;
        this.enableExtendedNegentropy = options.enableExtendedNegentropy !== undefined
            ? options.enableExtendedNegentropy
            : ENABLE_EXTENDED_NEGENTROPY;

        // Keep base windows for S₃ rotation reference
        this.baseWindows = { ...OPERATOR_WINDOWS };

        // Blending configuration
        this.blendPiEnabled = options.blendPiEnabled !== false;
    }

    /**
     * Get t6 gate value (dynamic based on TRIAD state)
     */
    getT6Gate() {
        return this.triadGate.getT6Gate();
    }

    /**
     * Determine time harmonic from z-coordinate
     * @param {number} z - Normalized z-coordinate [0,1]
     * @returns {string} - Harmonic label (t1-t9)
     */
    harmonicFromZ(z) {
        const t6Gate = this.getT6Gate();

        if (z < TIER_BOUNDARIES.t1) return 't1';
        if (z < TIER_BOUNDARIES.t2) return 't2';
        if (z < TIER_BOUNDARIES.t3) return 't3';
        if (z < TIER_BOUNDARIES.t4) return 't4';
        if (z < TIER_BOUNDARIES.t5) return 't5';
        if (z < t6Gate) return 't6';
        if (z < TIER_BOUNDARIES.t7) return 't7';
        if (z < TIER_BOUNDARIES.t8) return 't8';
        return 't9';
    }

    /**
     * Determine truth channel from z-coordinate
     * @param {number} z - Normalized z-coordinate [0,1]
     * @returns {string} - 'TRUE', 'UNTRUE', or 'PARADOX'
     */
    truthChannelFromZ(z) {
        if (z >= TRUTH_THRESHOLDS.TRUE) return 'TRUE';
        if (z >= TRUTH_THRESHOLDS.PARADOX) return 'PARADOX';
        return 'UNTRUE';
    }

    /**
     * Get operator window for current harmonic
     * Enhanced with S₃ symmetry when enabled
     * @param {string} harmonic - Harmonic tier (t1-t9)
     * @param {number} z - Optional z-coordinate for S₃ rotation
     * @returns {string[]} - Array of permitted operator symbols
     */
    getOperatorWindow(harmonic, z = null) {
        // If S₃ symmetry enabled and z provided, use rotated window
        if (this.enableS3Symmetry && z !== null) {
            return S3.generateS3OperatorWindow(harmonic, z);
        }
        // Fall back to static windows
        return OPERATOR_WINDOWS[harmonic] || ['()'];
    }

    /**
     * Compute negentropy signal ΔS_neg (Paper Section 2.3)
     * Gaussian centered on Z_CRITICAL, max=1 at z_c
     * @param {number} z - Z-coordinate
     * @param {number} sigma - Gaussian width (default LENS_SIGMA)
     * @returns {number} - ΔS_neg ∈ [0,1]
     */
    computeDeltaSNeg(z, sigma = LENS_SIGMA) {
        if (this.enableExtendedNegentropy) {
            return Delta.computeDeltaSNeg(z, sigma);
        }
        const d = z - Z_CRITICAL;
        return Math.exp(-sigma * d * d);
    }

    /**
     * Compute π-regime blend weights (Paper Section 2.3)
     * Enhanced with extended ΔS⁻ formalism when enabled
     * @param {number} z - Z-coordinate
     * @returns {Object} - { wPi, wLoc, inPiRegime? }
     */
    computeBlendWeights(z) {
        if (this.enableExtendedNegentropy) {
            const blend = Delta.computePiBlendWeights(z, this.blendPiEnabled);
            return {
                wPi: blend.wPi,
                wLoc: blend.wLocal,
                inPiRegime: blend.inPiRegime,
            };
        }
        // Legacy blending
        const deltaSNeg = this.computeDeltaSNeg(z);
        const wPi = z >= Z_CRITICAL ? Math.max(0, Math.min(1, deltaSNeg)) : 0.0;
        const wLoc = 1.0 - wPi;
        return { wPi, wLoc, inPiRegime: z >= Z_CRITICAL };
    }

    /**
     * Compute S₃-weighted operator preferences
     * Uses parity-based adjustments when S₃ symmetry is enabled
     * @param {string[]} operators - Available operators
     * @param {number} z - Z-coordinate
     * @returns {Object} - Map of operator → weight
     */
    computeOperatorWeights(operators, z) {
        if (this.enableS3Symmetry) {
            return S3.computeS3Weights(operators, z);
        }
        // Legacy weighting based on truth bias
        const truth = this.truthChannelFromZ(z);
        const biasTable = TRUTH_BIAS[truth] || {};
        const weights = {};
        for (const op of operators) {
            weights[op] = biasTable[op] || 1.0;
        }
        return weights;
    }

    /**
     * Compute gate modulation parameters from ΔS⁻
     * Only available when extended negentropy is enabled
     * @param {number} z - Z-coordinate
     * @returns {Object|null} - Gate modulation parameters or null
     */
    computeGateModulation(z) {
        if (!this.enableExtendedNegentropy) return null;
        return Delta.computeGateModulation(z);
    }

    /**
     * Compute full ΔS⁻ state
     * Only available when extended negentropy is enabled
     * @param {number} z - Z-coordinate
     * @param {Object} options - State options
     * @returns {Object|null} - Full state or null
     */
    computeFullDeltaState(z, options = {}) {
        if (!this.enableExtendedNegentropy) return null;
        return Delta.computeFullState(z, options);
    }

    /**
     * Get complete helix description for a z-coordinate
     * Enhanced with S₃ and ΔS⁻ information when enabled
     * @param {number} z - Normalized z-coordinate [0,1]
     * @returns {Object} - Full description
     */
    describe(z) {
        const value = Number.isFinite(z) ? z : 0;
        const clamped = Math.max(0, Math.min(1, value));

        const harmonic = this.harmonicFromZ(clamped);

        // Get operators with optional S₃ rotation
        const operators = this.getOperatorWindow(harmonic, clamped);
        const truthChannel = this.truthChannelFromZ(clamped);
        const deltaSNeg = this.computeDeltaSNeg(clamped);
        const weights = this.computeBlendWeights(clamped);

        // Build result object
        const result = {
            z: clamped,
            harmonic,
            operators,
            truthChannel,
            t6Gate: this.getT6Gate(),
            triadUnlocked: this.triadGate.unlocked,
            triadPasses: this.triadGate.passes,
            deltaSNeg,
            weights,
            // Feature flags status
            s3SymmetryEnabled: this.enableS3Symmetry,
            extendedNegentropyEnabled: this.enableExtendedNegentropy,
        };

        // Add S₃-weighted operator preferences if enabled
        if (this.enableS3Symmetry) {
            result.operatorWeights = this.computeOperatorWeights(operators, clamped);
            result.s3Element = S3.s3ElementFromZ(clamped);
            result.truthChannel = S3.truthChannelFromZ(clamped);
        }

        // Add extended ΔS⁻ state if enabled
        if (this.enableExtendedNegentropy) {
            result.deltaState = this.computeFullDeltaState(clamped);
            result.gateModulation = this.computeGateModulation(clamped);
            result.kFormation = Delta.checkKFormation(clamped);
        }

        return result;
    }

    /**
     * Check if an operator is legal at the current z-coordinate
     * @param {string} operator - APL operator symbol
     * @param {number} z - Normalized z-coordinate
     * @returns {boolean}
     */
    isOperatorLegal(operator, z) {
        const harmonic = this.harmonicFromZ(z);
        const window = this.getOperatorWindow(harmonic, z);
        return window.includes(operator);
    }

    /**
     * Get operator weight considering truth bias and S₃ parity
     * @param {string} operator - APL operator symbol
     * @param {number} z - Normalized z-coordinate
     * @returns {number} - Weight multiplier
     */
    getOperatorWeight(operator, z) {
        const isLegal = this.isOperatorLegal(operator, z);
        const baseWeight = isLegal ? 1.3 : 0.85;

        // Use S₃ weights if enabled
        if (this.enableS3Symmetry) {
            const harmonic = this.harmonicFromZ(z);
            const window = this.getOperatorWindow(harmonic, z);
            const s3Weights = this.computeOperatorWeights(window, z);
            const s3Weight = s3Weights[operator] || 1.0;
            return baseWeight * s3Weight;
        }

        // Legacy truth bias
        const truth = this.truthChannelFromZ(z);
        const biasTable = TRUTH_BIAS[truth] || {};
        const truthMultiplier = biasTable[operator] || 1.0;

        return baseWeight * truthMultiplier;
    }

    /**
     * Get dynamic truth bias using ΔS⁻ evolution
     * Only affects behavior when extended negentropy is enabled
     * @param {number} z - Z-coordinate
     * @returns {Object} - Truth bias matrix
     */
    getDynamicTruthBias(z) {
        if (this.enableExtendedNegentropy) {
            return Delta.computeDynamicTruthBias(z, TRUTH_BIAS);
        }
        return TRUTH_BIAS;
    }

    /**
     * Select best operator from window with truth bias weighting
     * Enhanced with S₃ parity weights and coherence heuristics
     * @param {number} z - Z-coordinate
     * @param {Function} rng - Random number generator (default Math.random)
     * @param {Object} options - Selection options
     * @returns {string} - Selected operator
     */
    selectOperator(z, rng = Math.random, options = {}) {
        const harmonic = this.harmonicFromZ(z);
        const window = this.getOperatorWindow(harmonic, z);

        // Get weights (S₃-enhanced or legacy)
        let opWeights;
        if (this.enableS3Symmetry) {
            opWeights = this.computeOperatorWeights(window, z);
        } else {
            const truth = this.truthChannelFromZ(z);
            const biasTable = this.getDynamicTruthBias(z)[truth] || {};
            opWeights = {};
            for (const op of window) {
                opWeights[op] = biasTable[op] || 1.0;
            }
        }

        // Apply coherence synthesis heuristics if requested
        if (options.coherenceObjective && this.enableExtendedNegentropy) {
            const cohScores = {};
            for (const op of window) {
                cohScores[op] = Delta.scoreOperatorForCoherence(
                    op, z, options.coherenceObjective
                );
            }
            // Combine with existing weights
            for (const op of window) {
                opWeights[op] = (opWeights[op] || 1.0) * (cohScores[op] || 1.0);
            }
        }

        // Compute weighted probabilities
        const weights = window.map(op => opWeights[op] || 1.0);
        const totalWeight = weights.reduce((a, b) => a + b, 0);

        // Weighted random selection
        let rand = rng() * totalWeight;
        for (let i = 0; i < window.length; i++) {
            rand -= weights[i];
            if (rand <= 0) return window[i];
        }

        return window[0];
    }
}

// ================================================================
// HELIX COORDINATE (Paper Section 2.1)
// ================================================================

/**
 * HelixCoordinate: Represents a point on the reasoning helix
 */
class HelixCoordinate {
    /**
     * @param {number} theta - Angular position [0, 2π]
     * @param {number} z - Normalized z-coordinate [0, 1]
     * @param {number} r - Radial distance (default 1.0)
     */
    constructor(theta, z, r = 1.0) {
        this.theta = theta;
        this.z = Math.max(0, Math.min(1, z));
        this.r = r;
    }

    /**
     * Construct from parametric helix equation
     * Z is normalized via tanh mapping into [0, 1]
     * @param {number} t - Parametric variable
     * @returns {HelixCoordinate}
     */
    static fromParameter(t) {
        const x = Math.cos(t);
        const y = Math.sin(t);
        const theta = Math.atan2(y, x);
        const thetaNorm = theta < 0 ? theta + 2 * Math.PI : theta;
        // Smooth sigmoid mapping: z = 0.5 + 0.5 * tanh(t/8)
        const z = 0.5 + 0.5 * Math.tanh(t / 8.0);
        const r = Math.hypot(x, y);
        return new HelixCoordinate(thetaNorm, z, r);
    }

    /**
     * Construct from z-coordinate only (theta derived from z)
     * @param {number} z - Z-coordinate
     * @returns {HelixCoordinate}
     */
    static fromZ(z) {
        // Derive theta from z (assuming z progresses along helix)
        const theta = (z * 4 * Math.PI) % (2 * Math.PI);
        return new HelixCoordinate(theta, z, 1.0);
    }

    /**
     * Get Cartesian coordinates
     * @returns {Object} - { x, y, z }
     */
    toVector() {
        return {
            x: this.r * Math.cos(this.theta),
            y: this.r * Math.sin(this.theta),
            z: this.z
        };
    }

    /**
     * Get signature string
     * @returns {string}
     */
    signature() {
        const thetaDeg = (this.theta * 180 / Math.PI).toFixed(1);
        return `Δ${thetaDeg}|${this.z.toFixed(4)}|${this.r.toFixed(2)}Ω`;
    }
}

// ================================================================
// ALPHA TOKEN SYNTHESIZER (Paper Section 2.5)
// ================================================================

/**
 * AlphaTokenSynthesizer: Generates APL tokens from helix coordinates
 */
class AlphaTokenSynthesizer {
    constructor(options = {}) {
        this.advisor = options.advisor || new HelixOperatorAdvisor();
    }

    /**
     * Synthesize APL token from helix z-coordinate
     * @param {number} z - Normalized z-coordinate [0,1]
     * @param {Object} hints - { domain, machine, umol }
     * @returns {Object|null}
     */
    fromZ(z, hints = {}) {
        const helixInfo = this.advisor.describe(z);
        const operatorWindow = helixInfo.operators;

        // Find matching sentences
        let candidates = this.findMatchingSentences({
            operators: operatorWindow,
            domain: hints.domain,
            machine: hints.machine,
            umol: hints.umol
        });

        if (candidates.length === 0) {
            // Fallback to any sentence with matching operator
            candidates = this.findMatchingSentences({ operators: operatorWindow });
        }

        if (candidates.length === 0) {
            return null;
        }

        const sentence = candidates[0];
        const operatorInfo = APL_OPERATORS[sentence.operator];

        return {
            sentence: sentence.token,
            sentenceId: sentence.id,
            predictedRegime: sentence.predictedRegime,
            operatorName: operatorInfo ? operatorInfo.name : sentence.operatorName,
            operatorSymbol: sentence.operator,
            operatorAction: operatorInfo ? operatorInfo.action : null,
            truthChannel: helixInfo.truthChannel,
            harmonic: helixInfo.harmonic,
            z: helixInfo.z,
            t6Gate: helixInfo.t6Gate,
            triadUnlocked: helixInfo.triadUnlocked,
            operatorWindow,
            deltaSNeg: helixInfo.deltaSNeg,
            weights: helixInfo.weights
        };
    }

    /**
     * Find sentences matching criteria
     * @param {Object} criteria - { operators, domain, machine, umol }
     * @returns {Array}
     */
    findMatchingSentences({ operators, domain, machine, umol } = {}) {
        let results = [...APL_SENTENCES];

        if (operators && operators.length > 0) {
            results = results.filter(s => operators.includes(s.operator));
        }
        if (domain) {
            results = results.filter(s => s.domain === domain);
        }
        if (machine) {
            results = results.filter(s => s.machine === machine);
        }
        if (umol) {
            results = results.filter(s => s.umol === umol);
        }

        return results;
    }

    /**
     * Generate measurement token
     * @param {string} field - 'Φ' (Phi), 'e', or 'π' (Pi)
     * @param {string} mode - 'T' (eigenstate) or 'Π' (subspace)
     * @param {string} intent - Measurement intent
     * @param {string} truth - Truth channel
     * @param {number|string} tier - Harmonic tier
     * @returns {string}
     */
    measurementToken(field, mode, intent, truth, tier) {
        return `${field}:${mode}(${intent})${truth}@${tier}`;
    }

    /**
     * Generate eigenstate measurement token
     */
    eigenToken(eigenIndex, truth, z) {
        const helixInfo = this.advisor.describe(z);
        const tier = helixInfo.harmonic;
        return this.measurementToken('Φ', 'T', `ϕ_${eigenIndex}`, truth, tier);
    }

    /**
     * Generate helix-aware operator token
     */
    helixToken(operator, z) {
        const helixInfo = this.advisor.describe(z);
        const opInfo = APL_OPERATORS[operator];
        const name = opInfo ? opInfo.name : operator;
        return `Helix:${operator}(${name})${helixInfo.truthChannel}@${helixInfo.harmonic}`;
    }
}

// ================================================================
// TRIADIC HELIX APL SYSTEM (Unified)
// ================================================================

/**
 * TriadicHelixAPLSystem: Main orchestration class
 * Integrates all components for simulation and validation
 */
class TriadicHelixAPLSystem {
    constructor(config = {}) {
        // Core components
        this.triadGate = new TriadGate({
            enabled: config.triadEnabled !== false,
            debug: config.debug || false,
            onUnlock: (data) => this._onTriadUnlock(data)
        });

        this.advisor = new HelixOperatorAdvisor({
            triadGate: this.triadGate
        });

        this.synthesizer = new AlphaTokenSynthesizer({
            advisor: this.advisor
        });

        // State
        this.z = config.initialZ || 0.5;
        this.theta = 0;
        this.time = 0;

        // History
        this.history = {
            z: [],
            operators: [],
            tokens: [],
            triadEvents: [],
            measurements: []
        };

        // Configuration
        this.verbose = config.verbose || false;
        this.maxHistory = config.maxHistory || 1000;
        this.dt = config.dt || 0.01;
    }

    /**
     * TRIAD unlock callback
     */
    _onTriadUnlock(data) {
        this.history.triadEvents.push({
            event: 'UNLOCKED',
            z: data.z,
            passes: data.passes,
            time: this.time,
            t6GateShift: `${Z_CRITICAL.toFixed(4)} → ${TRIAD_T6.toFixed(4)}`
        });

        if (this.verbose) {
            console.log(`\n${'='.repeat(60)}`);
            console.log('TRIAD UNLOCK ACHIEVED');
            console.log(`  Z-coordinate: ${data.z.toFixed(4)}`);
            console.log(`  Passes: ${data.passes}`);
            console.log(`  t6 gate: ${Z_CRITICAL.toFixed(4)} → ${TRIAD_T6.toFixed(4)}`);
            console.log(`${'='.repeat(60)}\n`);
        }
    }

    /**
     * Set z-coordinate directly
     */
    setZ(z) {
        this.z = Math.max(0, Math.min(1, z));
        return this;
    }

    /**
     * Execute one simulation step
     */
    step() {
        const prevZ = this.z;
        this.time += this.dt;

        // Update TRIAD tracker
        const triadResult = this.triadGate.update(this.z);

        // Get helix mapping
        const helixInfo = this.advisor.describe(this.z);

        // Select operator
        const operator = this.advisor.selectOperator(this.z);

        // Generate APL token
        const aplToken = this.synthesizer.fromZ(this.z);

        // Record history
        this._recordHistory({
            z: this.z,
            operator,
            helixInfo,
            aplToken,
            triadResult
        });

        return {
            time: this.time,
            z: this.z,
            operator,
            harmonic: helixInfo.harmonic,
            truthChannel: helixInfo.truthChannel,
            operatorWindow: helixInfo.operators,
            t6Gate: helixInfo.t6Gate,
            triadUnlocked: this.triadGate.unlocked,
            triadPasses: this.triadGate.passes,
            aplToken,
            deltaSNeg: helixInfo.deltaSNeg,
            weights: helixInfo.weights
        };
    }

    /**
     * Record step in history
     */
    _recordHistory(data) {
        this.history.z.push(data.z);
        this.history.operators.push({
            operator: data.operator,
            harmonic: data.helixInfo.harmonic,
            truthChannel: data.helixInfo.truthChannel,
            time: this.time
        });

        if (data.aplToken) {
            this.history.tokens.push({
                ...data.aplToken,
                time: this.time
            });
        }

        // Trim history
        while (this.history.z.length > this.maxHistory) {
            this.history.z.shift();
        }
        while (this.history.operators.length > this.maxHistory) {
            this.history.operators.shift();
        }
        while (this.history.tokens.length > this.maxHistory) {
            this.history.tokens.shift();
        }
    }

    /**
     * Run simulation for N steps with z evolution
     */
    simulate(steps, options = {}) {
        const targetZ = options.targetZ || Z_CRITICAL;
        const driftRate = options.driftRate || 0.02;
        const noiseScale = options.noiseScale || 0.01;
        const pumpGain = options.pumpGain || 0.1;

        const results = [];

        for (let i = 0; i < steps; i++) {
            // Evolve z
            let dz = driftRate * this.dt;
            dz += (Math.random() - 0.5) * noiseScale;
            dz += pumpGain * (targetZ - this.z) * this.dt;
            this.z = Math.max(0, Math.min(1, this.z + dz));

            // Execute step
            const result = this.step();
            results.push(result);

            if (this.verbose && (i + 1) % Math.max(1, Math.floor(steps / 10)) === 0) {
                console.log(
                    `Step ${i + 1}: z=${result.z.toFixed(4)} | ` +
                    `${result.harmonic}/${result.truthChannel} | ` +
                    `op=${result.operator} | ` +
                    `TRIAD=${result.triadPasses}/3`
                );
            }
        }

        return results;
    }

    /**
     * Simulate to achieve TRIAD unlock
     */
    simulateToUnlock(maxSteps = 500) {
        let steps = 0;
        let phase = 0;
        const halfPeriod = 25;

        this.z = 0.83;

        while (!this.triadGate.unlocked && steps < maxSteps) {
            phase++;
            const cycle = Math.floor(phase / halfPeriod);
            const inHighPhase = (cycle % 2) === 0;

            const targetZ = inHighPhase ? 0.87 : 0.78;
            const error = targetZ - this.z;
            this.z = this.z + 0.1 * error + (Math.random() - 0.5) * 0.005;
            this.z = Math.max(0, Math.min(1, this.z));

            this.step();
            steps++;
        }

        return {
            success: this.triadGate.unlocked,
            steps,
            finalZ: this.z,
            triadState: this.triadGate.getState()
        };
    }

    /**
     * Get current state
     */
    getState() {
        return {
            z: this.z,
            theta: this.theta,
            time: this.time,
            helix: this.advisor.describe(this.z),
            triad: this.triadGate.getState()
        };
    }

    /**
     * Get analytics
     */
    getAnalytics() {
        const zArr = this.history.z;
        const avgZ = zArr.length > 0 ? zArr.reduce((a, b) => a + b) / zArr.length : 0;
        const maxZ = zArr.length > 0 ? Math.max(...zArr) : 0;
        const minZ = zArr.length > 0 ? Math.min(...zArr) : 0;

        const opCounts = {};
        for (const entry of this.history.operators) {
            opCounts[entry.operator] = (opCounts[entry.operator] || 0) + 1;
        }

        const harmonicCounts = {};
        for (const entry of this.history.operators) {
            harmonicCounts[entry.harmonic] = (harmonicCounts[entry.harmonic] || 0) + 1;
        }

        return {
            totalSteps: zArr.length,
            time: this.time,
            z: { avg: avgZ, min: minZ, max: maxZ, final: this.z },
            operators: opCounts,
            harmonics: harmonicCounts,
            triad: this.triadGate.getState(),
            triadEvents: this.history.triadEvents
        };
    }

    /**
     * Reset system
     */
    reset() {
        this.z = 0.5;
        this.theta = 0;
        this.time = 0;
        this.history = { z: [], operators: [], tokens: [], triadEvents: [], measurements: [] };
        this.triadGate.reset();
    }

    /**
     * Generate summary report
     */
    summary() {
        const state = this.getState();
        const analytics = this.getAnalytics();

        return [
            '═'.repeat(70),
            'TRIADIC HELIX APL SYSTEM SUMMARY',
            '═'.repeat(70),
            '',
            'Current State:',
            `  z-coordinate: ${state.z.toFixed(4)}`,
            `  Harmonic: ${state.helix.harmonic}`,
            `  Truth Channel: ${state.helix.truthChannel}`,
            `  ΔS_neg: ${state.helix.deltaSNeg.toFixed(4)}`,
            `  π-weight: ${state.helix.weights.wPi.toFixed(4)}`,
            '',
            'TRIAD Status:',
            `  ${this.triadGate.analyzerReport()}`,
            '',
            'Analytics:',
            `  Total Steps: ${analytics.totalSteps}`,
            `  Z Range: [${analytics.z.min.toFixed(4)}, ${analytics.z.max.toFixed(4)}]`,
            '',
            '═'.repeat(70)
        ].join('\n');
    }
}

// ================================================================
// EXPORTS
// ================================================================

module.exports = {
    // Constants
    Z_CRITICAL,
    TRIAD_HIGH,
    TRIAD_LOW,
    TRIAD_T6,
    TRIAD_PASSES_REQ,
    LENS_SIGMA,
    PHI,
    PHI_INV,
    TIER_BOUNDARIES,
    TRUTH_THRESHOLDS,
    APL_OPERATORS,
    OPERATOR_WINDOWS,
    TRUTH_BIAS,
    APL_SENTENCES,

    // Feature flags
    ENABLE_S3_SYMMETRY,
    ENABLE_EXTENDED_NEGENTROPY,

    // Classes
    TriadGate,
    HelixOperatorAdvisor,
    HelixCoordinate,
    AlphaTokenSynthesizer,
    TriadicHelixAPLSystem,

    // S₃ and ΔS⁻ module re-exports for convenience
    S3,
    Delta,

    // Convenience functions
    getOperatorInfo: (symbol) => APL_OPERATORS[symbol] || null,
    getAllOperators: () => Object.keys(APL_OPERATORS),
    getSentenceById: (id) => APL_SENTENCES.find(s => s.id === id) || null,
    getAllSentences: () => APL_SENTENCES
};
