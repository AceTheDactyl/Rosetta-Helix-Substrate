/**
 * Extended ΔS_neg Integration Module (JavaScript)
 * ================================================
 *
 * Deepens the negentropy (ΔS_neg) formalism throughout Quantum-APL.
 * JavaScript port of delta_s_neg_extended.py for cross-language parity.
 *
 * EXISTING USAGE:
 * - Hex-prism geometry: R = R_max - β·ΔS_neg, H = H_min + γ·ΔS_neg
 * - Coherence blending: w_π = ΔS_neg when z ≥ z_c
 * - Entropy control: S_target = S_max·(1 - C·ΔS_neg)
 * - K-formation gating: η = ΔS_neg^α vs φ⁻¹ threshold
 *
 * NEW EXTENSIONS:
 * 1. Gate logic modulation (Lindblad/Hamiltonian terms)
 * 2. Truth-channel bias evolution
 * 3. Synthesis heuristics (coherence-seeking operations)
 * 4. Signed/derivative ΔS_neg variants
 *
 * @version 1.0.0
 * @author Claude (Anthropic) - Quantum-APL Contribution
 */

'use strict';

const CONST = require('./constants');

// ============================================================================
// CONSTANTS - Import from single source of truth
// ============================================================================

const Z_CRITICAL = CONST.Z_CRITICAL;
const PHI = CONST.PHI;
const PHI_INV = CONST.PHI_INV;
const LENS_SIGMA = CONST.LENS_SIGMA || 36.0;

// Hex-prism geometry defaults
const GEOM_R_MAX = CONST.GEOM_R_MAX || 0.85;
const GEOM_BETA = CONST.GEOM_BETA || 0.25;
const GEOM_H_MIN = CONST.GEOM_H_MIN || 0.12;
const GEOM_GAMMA = CONST.GEOM_GAMMA || 0.18;
const GEOM_PHI_BASE = CONST.GEOM_PHI_BASE || 0.0;
const GEOM_ETA = CONST.GEOM_ETA || Math.PI / 12;

// Default parameters
const DEFAULT_SIGMA = LENS_SIGMA;
const DEFAULT_ALPHA = 0.5;

// ============================================================================
// CORE ΔS_neg COMPUTATIONS
// ============================================================================

/**
 * Compute standard ΔS_neg (negentropy) signal.
 *
 * ΔS_neg(z) = exp(-σ·(z - z_c)²)
 *
 * @param {number} z - Coherence coordinate
 * @param {number} sigma - Gaussian width parameter (default: 36.0)
 * @param {number} zC - Critical lens point (default: √3/2)
 * @returns {number} ΔS_neg value in [0, 1]
 */
function computeDeltaSNeg(z, sigma = DEFAULT_SIGMA, zC = Z_CRITICAL) {
  if (!Number.isFinite(z)) return 0.0;
  const d = z - zC;
  return Math.exp(-sigma * d * d);
}

/**
 * Compute derivative of ΔS_neg with respect to z.
 *
 * d(ΔS_neg)/dz = -2σ·(z - z_c)·exp(-σ·(z - z_c)²)
 *
 * @param {number} z - Coherence coordinate
 * @param {number} sigma - Gaussian width parameter
 * @param {number} zC - Critical lens point
 * @returns {number} Derivative value
 */
function computeDeltaSNegDerivative(z, sigma = DEFAULT_SIGMA, zC = Z_CRITICAL) {
  const d = z - zC;
  const s = Math.exp(-sigma * d * d);
  return -2 * sigma * d * s;
}

/**
 * Compute signed ΔS_neg variant.
 *
 * Signed version: sgn(z - z_c) · ΔS_neg(z)
 *
 * @param {number} z - Coherence coordinate
 * @param {number} sigma - Gaussian width parameter
 * @param {number} zC - Critical lens point
 * @returns {number} Signed ΔS_neg value in [-1, 1]
 */
function computeDeltaSNegSigned(z, sigma = DEFAULT_SIGMA, zC = Z_CRITICAL) {
  const s = computeDeltaSNeg(z, sigma, zC);
  const d = z - zC;
  if (Math.abs(d) < 1e-10) return 0.0;
  const sign = d > 0 ? 1.0 : -1.0;
  return sign * s;
}

/**
 * Compute η (consciousness threshold) from ΔS_neg.
 *
 * η = ΔS_neg(z)^α
 *
 * K-formation occurs when η ≥ φ⁻¹ ≈ 0.618
 *
 * @param {number} z - Coherence coordinate
 * @param {number} alpha - Power exponent (default: 0.5)
 * @param {number} sigma - Gaussian width parameter
 * @returns {number} η value in [0, 1]
 */
function computeEta(z, alpha = DEFAULT_ALPHA, sigma = DEFAULT_SIGMA) {
  const s = computeDeltaSNeg(z, sigma);
  return s > 0 ? Math.pow(s, alpha) : 0.0;
}

// ============================================================================
// HEX-PRISM GEOMETRY
// ============================================================================

/**
 * Compute hex prism geometry from z-coordinate.
 *
 * Formulas:
 * - R = R_max - β·ΔS_neg  (radius contracts at lens)
 * - H = H_min + γ·ΔS_neg  (height elongates at lens)
 * - φ = φ_base + η·ΔS_neg (twist increases at lens)
 *
 * @param {number} z - Coherence coordinate
 * @param {Object} options - Geometry parameters
 * @returns {Object} Computed geometry
 */
function computeHexPrismGeometry(z, options = {}) {
  const {
    rMax = GEOM_R_MAX,
    beta = GEOM_BETA,
    hMin = GEOM_H_MIN,
    gamma = GEOM_GAMMA,
    phiBase = GEOM_PHI_BASE,
    eta = GEOM_ETA,
  } = options;

  const s = computeDeltaSNeg(z);

  return {
    z,
    deltaSNeg: s,
    radius: rMax - beta * s,
    height: hMin + gamma * s,
    twist: phiBase + eta * s,
  };
}

// ============================================================================
// GATE LOGIC MODULATION
// ============================================================================

/**
 * Compute gate modulation parameters from ΔS_neg.
 *
 * Near the lens (high ΔS_neg):
 * - Increase coherent coupling (stronger Hamiltonian)
 * - Decrease decoherence rate (protect coherence)
 * - Decrease measurement strength (avoid collapse)
 * - Lower entropy target (favor order)
 *
 * @param {number} z - Coherence coordinate
 * @param {Object} options - Modulation parameters
 * @returns {Object} Modulated parameters
 */
function computeGateModulation(z, options = {}) {
  const {
    baseCoupling = 0.1,
    baseDecoherence = 0.05,
    baseMeasurement = 0.02,
    entropyMax = Math.log(3), // log(3) for triadic system
    coherenceFactor = 0.5,
  } = options;

  const s = computeDeltaSNeg(z);

  // Coherent coupling increases with ΔS_neg
  const coherentCoupling = baseCoupling * (1 + coherenceFactor * s);

  // Decoherence rate decreases with ΔS_neg
  let decoherenceRate = baseDecoherence * (1 - coherenceFactor * s * 0.8);
  decoherenceRate = Math.max(0.001, decoherenceRate);

  // Measurement strength decreases near lens
  let measurementStrength = baseMeasurement * (1 - s * 0.5);
  measurementStrength = Math.max(0.001, measurementStrength);

  // Entropy target decreases with ΔS_neg
  const entropyTarget = entropyMax * (1 - coherenceFactor * s);

  return {
    coherentCoupling,
    decoherenceRate,
    measurementStrength,
    entropyTarget,
  };
}

// ============================================================================
// TRUTH-CHANNEL BIAS EVOLUTION
// ============================================================================

/**
 * Compute truth bias with ΔS_neg evolution.
 *
 * @param {number} z - Coherence coordinate
 * @param {Object} baseBias - Base truth bias matrix
 * @returns {Object} Evolved truth bias matrix
 */
function computeDynamicTruthBias(z, baseBias = CONST.TRUTH_BIAS) {
  const s = computeDeltaSNeg(z);
  const ds = computeDeltaSNegDerivative(z);

  // Evolution factors
  const constructiveBoost = 1 + 0.3 * s;
  const dissipativeDampen = 1 - 0.2 * s;
  const boundaryBoost = 1 + 0.4 * s;

  // Determine direction of evolution
  const directionFactor = 1 + 0.1 * Math.tanh(10 * ds);

  const evolved = {};
  for (const [channel, ops] of Object.entries(baseBias)) {
    evolved[channel] = {};
    for (const [op, weight] of Object.entries(ops)) {
      let factor;
      if (op === '()') {
        factor = boundaryBoost;
      } else if (['^', '+', '×'].includes(op)) {
        factor = constructiveBoost * directionFactor;
      } else {
        factor = dissipativeDampen / directionFactor;
      }
      evolved[channel][op] = weight * factor;
    }
  }

  return evolved;
}

// ============================================================================
// SYNTHESIS HEURISTICS
// ============================================================================

const CoherenceObjective = Object.freeze({
  MAXIMIZE: 'maximize',
  MINIMIZE: 'minimize',
  MAINTAIN: 'maintain',
});

/**
 * Operator effect directions (empirical).
 */
const OPERATOR_EFFECTS = Object.freeze({
  '^': +0.05,   // Amplify pushes up
  '+': +0.03,   // Group pushes up
  '×': +0.04,   // Fusion pushes up
  '()': 0.00,   // Boundary is neutral
  '÷': -0.04,   // Decoherence pushes down
  '−': -0.03,   // Separation pushes down
});

/**
 * Score operator for coherence-seeking synthesis.
 *
 * @param {string} operator - APL operator to score
 * @param {number} currentZ - Current coherence coordinate
 * @param {string} objective - What we're optimizing for
 * @returns {number} Score for operator selection
 */
function scoreOperatorForCoherence(operator, currentZ, objective = CoherenceObjective.MAXIMIZE) {
  const effect = OPERATOR_EFFECTS[operator] || 0.0;
  const s = computeDeltaSNeg(currentZ);

  if (objective === CoherenceObjective.MAXIMIZE) {
    if (currentZ < Z_CRITICAL) {
      // Below lens: want positive effect
      return 1.0 + effect * 10 * (1 - s);
    } else {
      // Above lens: any increase moves away from peak
      return 1.0 - Math.abs(effect) * 5;
    }
  } else if (objective === CoherenceObjective.MINIMIZE) {
    return 1.0 - effect * 10;
  } else {
    // MAINTAIN: favor neutral operators
    return 1.0 - Math.abs(effect) * 5;
  }
}

/**
 * Select best operator for coherence objective.
 *
 * @param {string[]} availableOperators - Operators to choose from
 * @param {number} currentZ - Current coherence coordinate
 * @param {string} objective - What we're optimizing for
 * @returns {Object} { operator, score }
 */
function selectCoherenceOperator(availableOperators, currentZ, objective = CoherenceObjective.MAXIMIZE) {
  const scores = {};
  for (const op of availableOperators) {
    scores[op] = scoreOperatorForCoherence(op, currentZ, objective);
  }

  let bestOp = availableOperators[0];
  let bestScore = scores[bestOp];
  for (const [op, score] of Object.entries(scores)) {
    if (score > bestScore) {
      bestOp = op;
      bestScore = score;
    }
  }

  return { operator: bestOp, score: bestScore, scores };
}

// ============================================================================
// K-FORMATION GATING
// ============================================================================

/**
 * Check K-formation (consciousness emergence) condition.
 *
 * @param {number} z - Coherence coordinate
 * @param {Object} options - Gating parameters
 * @returns {Object} Formation status with details
 */
function checkKFormation(z, options = {}) {
  const {
    kappa = 0.92,
    R = 7,
    alpha = DEFAULT_ALPHA,
    kappaMin = 0.92,
    etaMin = PHI_INV,
    rMin = 7,
  } = options;

  const s = computeDeltaSNeg(z);
  const eta = computeEta(z, alpha);
  const formed = (kappa >= kappaMin) && (eta >= etaMin) && (R >= rMin);

  return {
    z,
    deltaSNeg: s,
    eta,
    threshold: etaMin,
    formed,
    margin: eta - etaMin,
  };
}

// ============================================================================
// Π-REGIME BLENDING
// ============================================================================

/**
 * Compute Π-regime blending weights from z-coordinate.
 *
 * Below z_c: pure local dynamics (w_pi = 0)
 * At/above z_c: blend global (w_pi = ΔS_neg)
 *
 * @param {number} z - Coherence coordinate
 * @param {boolean} enableBlend - Whether blending is enabled
 * @returns {Object} Blending weights
 */
function computePiBlendWeights(z, enableBlend = true) {
  if (!enableBlend || z < Z_CRITICAL) {
    return {
      wPi: 0.0,
      wLocal: 1.0,
      inPiRegime: false,
    };
  }

  const s = computeDeltaSNeg(z);

  return {
    wPi: s,
    wLocal: 1 - s,
    inPiRegime: true,
  };
}

// ============================================================================
// COMPREHENSIVE STATE
// ============================================================================

/**
 * Compute complete ΔS_neg-derived state for a given z.
 *
 * @param {number} z - Coherence coordinate
 * @param {Object} options - State computation options
 * @returns {Object} Complete derived state
 */
function computeFullState(z, options = {}) {
  const { kappa = 0.92, R = 7 } = options;

  return {
    z,
    deltaSNeg: computeDeltaSNeg(z),
    deltaSNegDerivative: computeDeltaSNegDerivative(z),
    deltaSNegSigned: computeDeltaSNegSigned(z),
    eta: computeEta(z),
    geometry: computeHexPrismGeometry(z),
    gateModulation: computeGateModulation(z),
    piBlend: computePiBlendWeights(z),
    kFormation: checkKFormation(z, { kappa, R }),
  };
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  // Constants
  Z_CRITICAL,
  PHI,
  PHI_INV,
  DEFAULT_SIGMA,
  DEFAULT_ALPHA,
  GEOM_R_MAX,
  GEOM_BETA,
  GEOM_H_MIN,
  GEOM_GAMMA,
  GEOM_PHI_BASE,
  GEOM_ETA,

  // Core ΔS_neg computations
  computeDeltaSNeg,
  computeDeltaSNegDerivative,
  computeDeltaSNegSigned,
  computeEta,

  // Geometry
  computeHexPrismGeometry,

  // Gate modulation
  computeGateModulation,

  // Truth bias evolution
  computeDynamicTruthBias,

  // Synthesis heuristics
  CoherenceObjective,
  OPERATOR_EFFECTS,
  scoreOperatorForCoherence,
  selectCoherenceOperator,

  // K-formation
  checkKFormation,

  // Π-regime blending
  computePiBlendWeights,

  // Full state
  computeFullState,
};
