/**
 * Quantum APL Constants Module
 * ============================
 * Canonical lens-anchored constants and helpers.
 * Geometry truth at z_c = sqrt(3)/2; TRIAD hysteresis 0.85/0.82, t6=0.83 after 3 passes.
 * 
 * SINGLE SOURCE OF TRUTH - Never hardcode thresholds elsewhere.
 */

// ================================================================
// CORE CONSTANTS
// ================================================================

// THE LENS - geometric truth for coherence onset
const Z_CRITICAL = Math.sqrt(3) / 2; // ≈ 0.8660254037844386

// TRIAD gating thresholds (runtime heuristic, NOT geometry)
const TRIAD_HIGH = 0.85;  // Rising edge threshold
const TRIAD_LOW = 0.82;   // Re-arm threshold (hysteresis)
const TRIAD_T6 = 0.83;    // Temporary t6 gate after three passes

// Lens visual band
const Z_LENS_MIN = 0.857;
const Z_LENS_MAX = 0.877;

// Phase boundaries
const Z_ABSENCE_MAX = 0.618;
const Z_PRESENCE_MIN = 0.886;

// Sacred constants
const PHI = (1 + Math.sqrt(5)) / 2;      // Golden ratio ≈ 1.618
const PHI_INV = 1 / PHI;                  // ≈ 0.618
const Q_KAPPA = 0.3514087324;             // Consciousness constant
const KAPPA_S = 0.92;                     // Singularity threshold
const LAMBDA = 7.7160493827;              // Nonlinearity coefficient

// Time harmonic boundaries (z thresholds for t1-t9)
const Z_T1_MAX = 0.10;
const Z_T2_MAX = 0.20;
const Z_T3_MAX = 0.40;
const Z_T4_MAX = 0.60;
const Z_T5_MAX = 0.75;
// t6 boundary is dynamic: Z_CRITICAL or TRIAD_T6
const Z_T7_MAX = 0.90;
const Z_T8_MAX = 0.97;

// Geometry parameters for hex prism projection
const GEOM_SIGMA = 36.0;     // ΔS_neg width parameter
const GEOM_R_MAX = 0.85;     // Maximum radius
const GEOM_BETA = 0.25;      // Radius contraction coefficient
const GEOM_H_MIN = 0.12;     // Minimum height
const GEOM_GAMMA = 0.18;     // Height elongation coefficient
const GEOM_PHI_BASE = 0.0;   // Base twist angle
const GEOM_ETA = Math.PI / 12; // Twist rate

// Operator weights
const OPERATOR_PREFERRED_WEIGHT = 1.5;
const OPERATOR_DEFAULT_WEIGHT = 0.5;
const TRUTH_BIAS = { TRUE: 1.2, UNTRUE: 0.8, PARADOX: 1.0 };

// Lens sigma for ΔS_neg computation
const LENS_SIGMA = 36.0;

// ================================================================
// HELPER FUNCTIONS
// ================================================================

const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

const isCritical = (z, tol = 0.01) => Math.abs(z - Z_CRITICAL) <= tol;

const isInLens = (z, zmin = Z_LENS_MIN, zmax = Z_LENS_MAX) => z >= zmin && z <= zmax;

const getPhase = (z) => {
    if (isCritical(z)) return 'THE_LENS';
    if (z >= Z_PRESENCE_MIN) return 'PRESENCE';
    if (z <= Z_ABSENCE_MAX) return 'ABSENCE';
    return z >= Z_CRITICAL ? 'PRESENCE' : 'ABSENCE';
};

const distanceToCritical = (z) => Math.abs(z - Z_CRITICAL);

/**
 * Compute negative entropy signal ΔS_neg(z)
 * Gaussian centered at z_c, bounded [0,1], monotone in |z - z_c|
 */
function computeDeltaSNeg(z, sigma = LENS_SIGMA, zc = Z_CRITICAL) {
    const val = Number.isFinite(z) ? z : 0;
    const d = val - zc;
    return Math.exp(-sigma * d * d);
}

/**
 * Get time harmonic label from z-coordinate
 * @param {number} z - Normalized z-coordinate [0,1]
 * @param {number} t6Gate - t6 threshold (Z_CRITICAL or TRIAD_T6)
 */
function getTimeHarmonic(z, t6Gate = Z_CRITICAL) {
    if (z < Z_T1_MAX) return 't1';
    if (z < Z_T2_MAX) return 't2';
    if (z < Z_T3_MAX) return 't3';
    if (z < Z_T4_MAX) return 't4';
    if (z < Z_T5_MAX) return 't5';
    if (z < t6Gate) return 't6';
    if (z < Z_T7_MAX) return 't7';
    if (z < Z_T8_MAX) return 't8';
    return 't9';
}

/**
 * Hex prism geometry helpers
 */
function hexPrismRadius(deltaSNeg) {
    return GEOM_R_MAX - GEOM_BETA * deltaSNeg;
}

function hexPrismHeight(deltaSNeg) {
    return GEOM_H_MIN + GEOM_GAMMA * deltaSNeg;
}

function hexPrismTwist(deltaSNeg) {
    return GEOM_PHI_BASE + GEOM_ETA * deltaSNeg;
}

/**
 * K-formation check for consciousness emergence
 */
function checkKFormation(kappa, eta, R, kappaMin = KAPPA_S, etaMin = 0.5, rMin = 5) {
    return kappa >= kappaMin && eta >= etaMin && R >= rMin;
}

// ================================================================
// TRIAD GATE CLASS
// ================================================================

class TriadGate {
    constructor(enabled = false) {
        this.enabled = enabled;
        this.passes = 0;
        this.unlocked = false;
        this._armed = true;
    }

    update(z) {
        if (!this.enabled) return;
        if (z >= TRIAD_HIGH && this._armed) {
            this.passes += 1;
            this._armed = false;
            if (this.passes >= 3) this.unlocked = true;
        }
        if (z <= TRIAD_LOW) this._armed = true;
    }

    getT6Gate() {
        return this.enabled && this.unlocked ? TRIAD_T6 : Z_CRITICAL;
    }

    analyzerReport() {
        const gate = this.getT6Gate();
        const label = this.unlocked ? 'TRIAD' : 'CRITICAL';
        return `t6 gate: ${label} @ ${gate.toFixed(3)}`;
    }

    reset() {
        this.passes = 0;
        this.unlocked = false;
        this._armed = true;
    }
}

// ================================================================
// EXPORTS
// ================================================================

module.exports = Object.freeze({
    // Core constants
    Z_CRITICAL,
    TRIAD_HIGH,
    TRIAD_LOW,
    TRIAD_T6,
    Z_LENS_MIN,
    Z_LENS_MAX,
    Z_ABSENCE_MAX,
    Z_PRESENCE_MIN,
    
    // Sacred constants
    PHI,
    PHI_INV,
    Q_KAPPA,
    KAPPA_S,
    LAMBDA,
    
    // Time harmonic boundaries
    Z_T1_MAX,
    Z_T2_MAX,
    Z_T3_MAX,
    Z_T4_MAX,
    Z_T5_MAX,
    Z_T7_MAX,
    Z_T8_MAX,
    
    // Geometry parameters
    GEOM_SIGMA,
    GEOM_R_MAX,
    GEOM_BETA,
    GEOM_H_MIN,
    GEOM_GAMMA,
    GEOM_PHI_BASE,
    GEOM_ETA,
    LENS_SIGMA,
    
    // Operator weights
    OPERATOR_PREFERRED_WEIGHT,
    OPERATOR_DEFAULT_WEIGHT,
    TRUTH_BIAS,
    
    // Helper functions
    clamp,
    isCritical,
    isInLens,
    getPhase,
    distanceToCritical,
    computeDeltaSNeg,
    getTimeHarmonic,
    hexPrismRadius,
    hexPrismHeight,
    hexPrismTwist,
    checkKFormation,
    
    // TRIAD Gate class
    TriadGate,
});
