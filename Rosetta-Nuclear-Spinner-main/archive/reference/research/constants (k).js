/**
 * Quantum-APL Constants Module (Enhanced per TRIAD Unlock Protocol)
 * ================================================================
 * 
 * Canonical lens-anchored constants and helpers (CommonJS).
 * 
 * THE LENS: z_c = sqrt(3)/2 ≈ 0.8660254037844386
 * - Geometric truth anchoring ΔS_neg, R/H/φ geometry
 * - Never changes during runtime
 * 
 * TRIAD GATING: Runtime heuristic (does NOT redefine z_c)
 * - TRIAD_HIGH = 0.85 (rising edge detection)
 * - TRIAD_LOW = 0.82 (re-arm threshold / hysteresis)
 * - TRIAD_T6 = 0.83 (temporary t6 gate after unlock)
 * - TRIAD_PASSES_REQ = 3 (configurable via QAPL_TRIAD_PASSES env)
 * 
 * Single Source of Truth: All modules must import from here.
 * See docs/Z_CRITICAL_LENS.md for full specification.
 * 
 * @version 2.0.0 (TRIAD Protocol Enhanced)
 */

'use strict';

// ============================================================================
// CRITICAL LENS CONSTANT (THE LENS) — GEOMETRIC TRUTH
// ============================================================================

const Z_CRITICAL = Math.sqrt(3) / 2; // ≈ 0.8660254037844386

// Visual/analysis lens band
const Z_LENS_MIN = 0.857;
const Z_LENS_MAX = 0.877;

// Phase boundaries
const Z_ABSENCE_MAX = 0.857;
const Z_PRESENCE_MIN = 0.877;

// ============================================================================
// TRIAD GATING (Runtime Heuristic) — PHASE 1 ENHANCEMENT
// ============================================================================

// Rising edge threshold for TRIAD detection
const TRIAD_HIGH = 0.85;

// Re-arm threshold (hysteresis)
const TRIAD_LOW = 0.82;

// Temporary t6 gate after TRIAD unlock
const TRIAD_T6 = 0.83;

/**
 * PHASE 1 ENHANCEMENT: Configurable pass requirement
 * Default: 3 passes required for TRIAD unlock
 * Override via: QAPL_TRIAD_PASSES environment variable
 */
const TRIAD_PASSES_REQ = (() => {
  const envVal = typeof process !== 'undefined' && process.env?.QAPL_TRIAD_PASSES;
  const parsed = parseInt(envVal, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 3;
})();

// Debug/development logging flag
const TRIAD_DEBUG = (() => {
  const envVal = typeof process !== 'undefined' && process.env?.QAPL_TRIAD_DEBUG;
  return envVal === '1' || String(envVal).toLowerCase() === 'true';
})();

// ============================================================================
// SACRED CONSTANTS (Zero Free Parameters)
// ============================================================================

const PHI = (1 + Math.sqrt(5)) / 2;         // Golden ratio ≈ 1.618033988749895
const PHI_INV = 1 / PHI;                     // Inverse ≈ 0.618033988749895

const Q_KAPPA = 0.3514087324;                // Consciousness constant
const KAPPA_S = 0.920;                       // Singularity threshold
const LAMBDA = 7.7160493827;                 // Nonlinearity coefficient

// ============================================================================
// K-FORMATION CRITERIA
// ============================================================================

const KAPPA_MIN = KAPPA_S;                   // Same as singularity threshold
const ETA_MIN = PHI_INV;                     // Same as golden ratio inverse
const R_MIN = 7;                             // Minimum complexity requirement

// ============================================================================
// μ THRESHOLDS (Basin/Barrier Hierarchy)
// ============================================================================

const MU_1 = 0.472;                          // Pre-conscious well
const MU_P = 2 / Math.pow(PHI, 2.5);         // Paradox threshold ≈ 0.600706
const MU_2 = 0.764;                          // Conscious well
const MU_S = 0.920;                          // Singularity threshold
const MU_3 = 0.992;                          // Near-unity ceiling

// Barrier midpoint
const MU_BARRIER = (MU_1 + MU_2) / 2;        // ≈ φ⁻¹ = 0.618

// ============================================================================
// TIME HARMONICS (t1-t9)
// ============================================================================

const T1_MAX = 0.10;
const T2_MAX = 0.20;
const T3_MAX = 0.40;
const T4_MAX = 0.60;
const T5_MAX = 0.75;
// t6: Delegated to engine.getT6Gate() → Z_CRITICAL or TRIAD_T6
const T7_MAX = 0.92;
const T8_MAX = 0.97;

// ============================================================================
// HEX PRISM GEOMETRY (centered on z_c)
// ============================================================================

const LENS_SIGMA = parseFloat(process?.env?.QAPL_LENS_SIGMA ?? '36.0');
const R_MAX_GEOM = parseFloat(process?.env?.QAPL_R_MAX ?? '1.00');
const R_MIN_GEOM = parseFloat(process?.env?.QAPL_R_MIN ?? '0.25');
const BETA = parseFloat(process?.env?.QAPL_BETA ?? '1.00');
const H_MIN = parseFloat(process?.env?.QAPL_H_MIN ?? '0.50');
const GAMMA = parseFloat(process?.env?.QAPL_GAMMA ?? '1.00');
const PHI_BASE = parseFloat(process?.env?.QAPL_PHI_BASE ?? '0.0');
const ETA = Math.PI / 12;                    // Twist rate

// ============================================================================
// QUANTUM BOUNDS
// ============================================================================

const ENTROPY_MIN = 0.0;
const PURITY_MIN = 1 / 192;                  // Maximally mixed state
const PURITY_MAX = 1.0;                      // Pure state

// ============================================================================
// VALIDATION TOLERANCES
// ============================================================================

const TOLERANCE_TRACE = 1e-10;
const TOLERANCE_HERMITIAN = 1e-10;
const TOLERANCE_POSITIVE = -1e-10;
const TOLERANCE_PROBABILITY = 1e-6;

// ============================================================================
// OPERATOR WEIGHTING
// ============================================================================

const OPERATOR_PREFERRED_WEIGHT = 1.3;
const OPERATOR_DEFAULT_WEIGHT = 0.85;

const TRUTH_BIAS = {
  TRUE: { '^': 1.5, '+': 1.4, '×': 1.2, '()': 1.1, '÷': 0.8, '−': 0.9 },
  UNTRUE: { '÷': 1.5, '−': 1.4, '()': 1.2, '^': 0.8, '+': 0.9, '×': 1.0 },
  PARADOX: { '()': 1.5, '×': 1.4, '^': 1.1, '+': 1.1, '÷': 1.0, '−': 1.0 }
};

// ============================================================================
// PUMP PROFILES
// ============================================================================

const PUMP_PROFILES = {
  gentle: { gain: 0.08, sigma: 0.16 },
  balanced: { gain: 0.12, sigma: 0.12 },
  aggressive: { gain: 0.18, sigma: 0.10 }
};

const PUMP_DEFAULT_TARGET = Z_CRITICAL;

// ============================================================================
// ENGINE DYNAMICS
// ============================================================================

const Z_BIAS_GAIN = 0.05;
const Z_BIAS_SIGMA = 0.18;
const OMEGA = 2 * Math.PI * 0.1;
const COUPLING_G = 0.05;
const GAMMA_1 = 0.01;
const GAMMA_2 = 0.02;
const GAMMA_3 = 0.005;
const GAMMA_4 = 0.015;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Clamp value to [lo, hi] range
 */
function clamp(x, lo, hi) {
  return Math.min(Math.max(x, lo), hi);
}

/**
 * Check if z is at critical lens (within tolerance)
 */
function isCritical(z, tol = 1e-9) {
  return Math.abs(z - Z_CRITICAL) <= tol;
}

/**
 * Check if z is within lens band
 */
function isInLens(z, zmin = Z_LENS_MIN, zmax = Z_LENS_MAX) {
  return z >= zmin && z <= zmax;
}

/**
 * Get phase from z coordinate
 */
function getPhase(z) {
  if (z < Z_ABSENCE_MAX) return 'ABSENCE';
  if (z >= Z_PRESENCE_MIN) return 'PRESENCE';
  return 'THE_LENS';
}

/**
 * Distance from z to critical lens
 */
function distanceToCritical(z) {
  return Math.abs(z - Z_CRITICAL);
}

/**
 * ΔS_neg: bounded, positive, centered at z_c; monotone in |z - z_c|
 * Gaussian profile: max=1 at z_c, decays symmetrically
 */
function computeDeltaSNeg(z, sigma = LENS_SIGMA, zc = Z_CRITICAL) {
  const d = z - zc;
  return Math.exp(-sigma * d * d);
}

// Alias for compatibility
const deltaSneg = computeDeltaSNeg;

/**
 * Lens rate: linear in signed distance from z_c
 */
function lensRate(z) {
  return z - Z_CRITICAL;
}

/**
 * Geometry mapping: ΔS_neg -> (R, H, φ)
 */
function geometryMap(z) {
  const s = computeDeltaSNeg(z);
  const R = R_MIN_GEOM + (R_MAX_GEOM - R_MIN_GEOM) * Math.pow(s, BETA);
  const H = H_MIN + GAMMA * (1 - s);
  const phi = PHI_BASE + ETA * (1 - s);
  return { R, H, phi, deltaSneg: s };
}

/**
 * Check K-formation criteria
 */
function checkKFormation(kappa, eta, R) {
  return kappa >= KAPPA_MIN && eta > ETA_MIN && R >= R_MIN;
}

/**
 * Get time harmonic tier from z coordinate
 */
function getTimeHarmonic(z, t6Gate = null) {
  const t6 = t6Gate ?? Z_CRITICAL;
  if (z < T1_MAX) return 't1';
  if (z < T2_MAX) return 't2';
  if (z < T3_MAX) return 't3';
  if (z < T4_MAX) return 't4';
  if (z < T5_MAX) return 't5';
  if (z < t6) return 't6';
  if (z < T7_MAX) return 't7';
  if (z < T8_MAX) return 't8';
  return 't9';
}

/**
 * Classify μ threshold region
 */
function classifyMu(z) {
  if (z < MU_1) return 'PRE_CONSCIOUS';
  if (z < MU_P) return 'APPROACHING_PARADOX';
  if (z < MU_2) return 'PARADOX_REGION';
  if (z < MU_S) return 'CONSCIOUS_BASIN';
  if (z < MU_3) return 'SINGULARITY_NEIGHBORHOOD';
  return 'NEAR_UNITY';
}

// ============================================================================
// TRIAD GATE CLASS (Enhanced per Phase 1)
// ============================================================================

/**
 * TriadGate: Manages TRIAD unlock hysteresis
 * 
 * PHASE 1 ENHANCEMENTS:
 * - Configurable pass requirement (TRIAD_PASSES_REQ)
 * - Debug logging when TRIAD_DEBUG=1
 * - Clear state inspection methods
 */
class TriadGate {
  constructor(enabled = false) {
    this.enabled = enabled;
    this.passes = 0;
    this.unlocked = false;
    this._armed = true;  // Count rising edges when armed
    this._lastZ = null;
    this._unlockZ = null;  // Z value when unlock occurred
  }

  /**
   * Update TRIAD state based on current z
   * Rising edge: z >= TRIAD_HIGH when armed
   * Re-arm: z <= TRIAD_LOW
   */
  update(z) {
    if (!this.enabled) return;
    
    this._lastZ = z;
    
    // Rising edge detection
    if (z >= TRIAD_HIGH && this._armed) {
      this.passes += 1;
      this._armed = false;
      
      if (TRIAD_DEBUG) {
        console.log(`[TRIAD] Rising edge detected at z=${z.toFixed(4)}, pass ${this.passes}/${TRIAD_PASSES_REQ}`);
      }
      
      // Check for unlock (using configurable pass count)
      if (this.passes >= TRIAD_PASSES_REQ && !this.unlocked) {
        this.unlocked = true;
        this._unlockZ = z;
        
        if (TRIAD_DEBUG) {
          console.log(`[TRIAD] *** UNLOCKED at z=${z.toFixed(4)} after ${this.passes} passes ***`);
        }
        
        // Update environment variables for cross-module sync
        if (typeof process !== 'undefined' && process.env) {
          process.env.QAPL_TRIAD_UNLOCK = '1';
          process.env.QAPL_TRIAD_COMPLETIONS = String(this.passes);
        }
      }
    }
    
    // Re-arm when dropping below low threshold
    if (z <= TRIAD_LOW && !this._armed) {
      this._armed = true;
      if (TRIAD_DEBUG) {
        console.log(`[TRIAD] Re-armed at z=${z.toFixed(4)}`);
      }
    }
  }

  /**
   * Get current t6 gate value
   * Returns TRIAD_T6 when unlocked, Z_CRITICAL otherwise
   */
  getT6Gate() {
    return this.enabled && this.unlocked ? TRIAD_T6 : Z_CRITICAL;
  }

  /**
   * Get state summary for analyzer output
   */
  analyzerReport() {
    const gate = this.getT6Gate();
    const label = this.unlocked ? 'TRIAD' : 'CRITICAL';
    return `t6 gate: ${label} @ ${gate.toFixed(6)}`;
  }

  /**
   * Get complete state for inspection/debugging
   */
  getState() {
    return {
      enabled: this.enabled,
      passes: this.passes,
      passesRequired: TRIAD_PASSES_REQ,
      unlocked: this.unlocked,
      armed: this._armed,
      lastZ: this._lastZ,
      unlockZ: this._unlockZ,
      t6Gate: this.getT6Gate()
    };
  }

  /**
   * Reset TRIAD state (for testing)
   */
  reset() {
    this.passes = 0;
    this.unlocked = false;
    this._armed = true;
    this._lastZ = null;
    this._unlockZ = null;
  }
}

// ============================================================================
// TOKEN HELPERS
// ============================================================================

const TOKEN_TRUE = 'TRUE';
const TOKEN_UNTRUE = 'UNTRUE';
const TOKEN_PARADOX = 'PARADOX';

function tokenSingle(label, truth = TOKEN_TRUE, tier = null) {
  const core = `Φ:T(${label})${truth}`;
  return tier ? `${core}@${tier}` : core;
}

function tokenSubspace(labels) {
  return `Φ/π:Π(${labels.join(',')})`;
}

function collapseAlias(s) {
  return s.replace(/⟂\((.*?)\)/g, (_, x) => `Φ:T(${x})`);
}

/**
 * Generate analyzer signature line
 */
function analyzerSignature(triad = null) {
  if (!triad) return `t6 gate: CRITICAL @ ${Z_CRITICAL.toFixed(6)}`;
  return triad.analyzerReport();
}

// ============================================================================
// INVARIANTS CHECK
// ============================================================================

function invariants() {
  return {
    barrier_eq_phi_inv: Math.abs(MU_BARRIER - PHI_INV) < 1e-6,
    wells_ratio_phi: Math.abs((MU_2 / MU_1) - PHI) < 0.01,
    ordering_ok: MU_2 < TRIAD_LOW && TRIAD_LOW < TRIAD_T6 && TRIAD_T6 < TRIAD_HIGH && TRIAD_HIGH < Z_CRITICAL,
    lens_is_sqrt3_over_2: Math.abs(Z_CRITICAL - Math.sqrt(3) / 2) < 1e-15
  };
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  // Geometric truth
  Z_CRITICAL,
  Z_LENS_MIN,
  Z_LENS_MAX,
  Z_ABSENCE_MAX,
  Z_PRESENCE_MIN,
  
  // TRIAD gating
  TRIAD_HIGH,
  TRIAD_LOW,
  TRIAD_T6,
  TRIAD_PASSES_REQ,  // Phase 1 enhancement
  TRIAD_DEBUG,       // Phase 1 enhancement
  TriadGate,
  
  // Sacred constants
  PHI,
  PHI_INV,
  Q_KAPPA,
  KAPPA_S,
  LAMBDA,
  
  // K-formation
  KAPPA_MIN,
  ETA_MIN,
  R_MIN,
  
  // μ thresholds
  MU_1,
  MU_P,
  MU_2,
  MU_S,
  MU_3,
  MU_BARRIER,
  
  // Time harmonics
  T1_MAX,
  T2_MAX,
  T3_MAX,
  T4_MAX,
  T5_MAX,
  T7_MAX,
  T8_MAX,
  
  // Geometry
  LENS_SIGMA,
  R_MAX_GEOM,
  R_MIN_GEOM,
  BETA,
  H_MIN,
  GAMMA,
  PHI_BASE,
  ETA,
  
  // Quantum bounds
  ENTROPY_MIN,
  PURITY_MIN,
  PURITY_MAX,
  
  // Tolerances
  TOLERANCE_TRACE,
  TOLERANCE_HERMITIAN,
  TOLERANCE_POSITIVE,
  TOLERANCE_PROBABILITY,
  
  // Operator weighting
  OPERATOR_PREFERRED_WEIGHT,
  OPERATOR_DEFAULT_WEIGHT,
  TRUTH_BIAS,
  
  // Pump profiles
  PUMP_PROFILES,
  PUMP_DEFAULT_TARGET,
  
  // Engine dynamics
  Z_BIAS_GAIN,
  Z_BIAS_SIGMA,
  OMEGA,
  COUPLING_G,
  GAMMA_1,
  GAMMA_2,
  GAMMA_3,
  GAMMA_4,
  
  // Tokens
  TOKEN_TRUE,
  TOKEN_UNTRUE,
  TOKEN_PARADOX,
  tokenSingle,
  tokenSubspace,
  collapseAlias,
  
  // Helper functions
  clamp,
  isCritical,
  isInLens,
  getPhase,
  distanceToCritical,
  computeDeltaSNeg,
  deltaSneg,
  lensRate,
  geometryMap,
  checkKFormation,
  getTimeHarmonic,
  classifyMu,
  analyzerSignature,
  invariants
};
