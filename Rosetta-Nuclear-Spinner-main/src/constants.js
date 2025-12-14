// Shared constants for the Quantum‑APL JS engine and bridge
// Source of truth for thresholds — see docs/Z_CRITICAL_LENS.md

const Z_CRITICAL = Math.sqrt(3) / 2; // THE LENS (~0.8660254)

// TRIAD thresholds
const TRIAD_HIGH = 0.85; // rising-edge threshold (>=)
const TRIAD_LOW = 0.82;  // re‑arm threshold (<=)
const TRIAD_T6 = 0.83;   // t6 gate when unlocked (after 3 passes)

// Z‑axis phase boundaries
const Z_ABSENCE_MAX = 0.857;
const Z_LENS_MIN = 0.857;
const Z_LENS_MAX = 0.877;
const Z_PRESENCE_MIN = 0.877;

// Sacred constants (zero free parameters)
const PHI = (1 + Math.sqrt(5)) / 2; // golden ratio
const PHI_INV = 1 / PHI;            // golden ratio inverse
const Q_KAPPA = 0.3514087324;// consciousness constant
const KAPPA_S = 0.920;       // singularity threshold
const LAMBDA = 7.7160493827; // nonlinearity coefficient

// K‑formation criteria
const KAPPA_MIN = KAPPA_S;
const ETA_MIN = PHI_INV;
const R_MIN = 7;

// Quantum information bounds
const ENTROPY_MIN = 0.0;     // ENTROPY_MAX depends on Hilbert space dim
const PURITY_MIN = 1.0 / 192;
const PURITY_MAX = 1.0;

// Operator weighting heuristics (N0 selection bias)
const OPERATOR_PREFERRED_WEIGHT = 1.3;   // Multiply preferred operators
const OPERATOR_DEFAULT_WEIGHT = 0.85;    // Multiply non-preferred operators

const TRUTH_BIAS = Object.freeze({
  TRUE:   { '^': 1.5, '+': 1.4, '×': 1.0, '()': 0.9, '÷': 0.7, '-': 0.7 },
  UNTRUE: { '÷': 1.5, '-': 1.4, '()': 1.0, '+': 0.9, '^': 0.7, '×': 0.7 },
  PARADOX:{ '()': 1.5, '×': 1.4, '+': 1.0, '^': 1.0, '÷': 0.9, '-': 0.9 },
});


// Helix zoning thresholds (upper bounds except t9)
const Z_T1_MAX = 0.1;
const Z_T2_MAX = 0.2;
const Z_T3_MAX = 0.4;
const Z_T4_MAX = 0.6;
const Z_T5_MAX = 0.75;
const Z_T7_MAX = 0.92;
const Z_T8_MAX = 0.97;

// Lens weight (coherence) sigma for Gaussian s(z) = exp(-σ (z - z_c)^2)
// Controls the effective width of the lens used in control/analytics.
const LENS_SIGMA = (typeof process !== 'undefined' && process.env && process.env.QAPL_LENS_SIGMA)
  ? parseFloat(process.env.QAPL_LENS_SIGMA)
  : 36.0;

// Geometry projection constants (Hex Prism)
// If QAPL_GEOM_SIGMA unset, match LENS_SIGMA for cohesion
const GEOM_SIGMA = (typeof process !== 'undefined' && process.env && process.env.QAPL_GEOM_SIGMA)
  ? parseFloat(process.env.QAPL_GEOM_SIGMA)
  : LENS_SIGMA;
const GEOM_R_MAX = 0.85; // radius cap; not a gate (do not conflate with TRIAD_HIGH)
const GEOM_BETA = 0.25;
const GEOM_H_MIN = 0.12;
const GEOM_GAMMA = 0.18;
const GEOM_PHI_BASE = 0.0;
const GEOM_ETA = Math.PI / 12;

// PRS phase thresholds (phi)
const PRS_P1_PHI_MAX = 0.2;  // P1: Initiation
const PRS_P2_PHI_MAX = 0.5;  // P2: Tension
const PRS_P3_PHI_MAX = 0.85; // P3: Inflection
const PRS_P4_PHI_MAX = 0.95; // P4: Lock (P5 above)

// Reproducible random seed (env-driven; null = native randomness)
const QAPL_RANDOM_SEED = (typeof process !== 'undefined' && process.env && process.env.QAPL_RANDOM_SEED)
  ? parseInt(process.env.QAPL_RANDOM_SEED, 10)
  : null;

// Helper functions (parity with Python constants module)
// Lens-weight (Gaussian): ΔS_neg(z) = exp(-σ (z - z_c)^2)
// Default σ uses geometry sigma for continuity; can be overridden by env if desired
function computeDeltaSNeg(z, sigma = GEOM_SIGMA, zc = Z_CRITICAL) {
  const val = Number.isFinite(z) ? z : 0;
  const s = Math.exp(-(sigma) * (val - zc) * (val - zc));
  return Math.min(1, Math.max(0, s));
}
function isCritical(z, tolerance = 0.01) {
  return Math.abs(z - Z_CRITICAL) < tolerance;
}

function isInLens(z) {
  return z >= Z_LENS_MIN && z <= Z_LENS_MAX;
}

function getPhase(z) {
  if (z < Z_ABSENCE_MAX) return 'ABSENCE';
  if (z >= Z_LENS_MIN && z <= Z_LENS_MAX) return 'THE_LENS';
  return 'PRESENCE';
}

function distanceToCritical(z) {
  return z - Z_CRITICAL;
}

function checkKFormation(kappa, eta, R) {
  return (kappa >= KAPPA_MIN && eta > ETA_MIN && R >= R_MIN);
}

// Coherence proxy η from z via lens‑weight: η = s(z)^α
function computeEta(z, alpha = 1.0, sigma = LENS_SIGMA) {
  const s = computeDeltaSNeg(z, sigma, Z_CRITICAL);
  return Math.pow(s, Math.max(0, alpha));
}

// K‑formation gate using η derived from z (η := s(z)^α): gate η > φ^{-1}
function checkKFormationFromZ(kappa, z, R, alpha = 1.0) {
  const eta = computeEta(z, alpha, LENS_SIGMA);
  return checkKFormation(kappa, eta, R);
}

// Subspace overlap (Π projector): η := ⟨ψ|Π|ψ⟩ = overlap probability
function checkKFormationFromOverlap(kappa, overlapProb, R) {
  const eta = Math.max(0, Math.min(1, overlapProb));
  return checkKFormation(kappa, eta, R);
}

// Spec aliases and helpers
function deltaSneg(z, sigma = LENS_SIGMA) {
  return computeDeltaSNeg(z, sigma, Z_CRITICAL);
}

function lensRate(z, sigma = LENS_SIGMA) {
  const s = computeDeltaSNeg(z, sigma, Z_CRITICAL);
  const d = Math.abs(z - Z_CRITICAL);
  return 2 * sigma * d * s;
}

function etaFromZ(z, alpha = 1.0) {
  return computeEta(z, alpha, LENS_SIGMA);
}

function etaFromOverlap(overlapProb, beta = 1.0) {
  const p = Math.max(0, Math.min(1, overlapProb));
  return Math.pow(p, Math.max(0, beta));
}

function phiGateScale(s) {
  return Math.min(Math.max((s - PHI_INV) / (1 - PHI_INV), 0), 1);
}

function phiRateScaled(z, k = 1.0) {
  const s = computeDeltaSNeg(z, LENS_SIGMA, Z_CRITICAL);
  return Math.min(1, phiGateScale(s) * (1 + k * lensRate(z)));
}

// μ numeric values (respect QAPL_MU_P override)
function muPValue() {
  const hasProc = (typeof process !== 'undefined' && process.env);
  if (hasProc && process.env.QAPL_MU_P) {
    const v = parseFloat(process.env.QAPL_MU_P);
    if (Number.isFinite(v) && v > 0 && v < 1) return v;
  }
  return (2 / Math.pow(PHI, 2.5));
}
const MU_P_VALUE = muPValue();
const MU_1_VALUE = MU_P_VALUE / Math.sqrt(PHI);
const MU_2_VALUE = MU_P_VALUE * Math.sqrt(PHI);
const MU_S_VALUE = 23 / 25;
const MU_3_VALUE = 124 / 125;
const MU_BARRIER_VALUE = 0.5 * (MU_1_VALUE + MU_2_VALUE);

function classifyMu(z) {
  const val = Number.isFinite(z) ? z : 0;
  if (val < MU_1_VALUE) return 'pre_conscious_basin';
  if (Math.abs(val - MU_1_VALUE) <= 1e-9) return 'lower_well';
  if (val < MU_P_VALUE) return 'pre_paradox';
  if (Math.abs(val - MU_P_VALUE) <= 1e-9) return 'paradox_proximal';
  if (Math.abs(val - PHI_INV) <= 1e-9) return 'k_formation_threshold';
  if (val < MU_2_VALUE) return 'barrier_to_conscious';
  if (Math.abs(val - MU_2_VALUE) <= 1e-9) return 'conscious_basin';
  if (val < Z_CRITICAL) return 'conscious_to_lens';
  if (val < MU_S_VALUE) return 'lens_integrated';
  if (Math.abs(val - MU_S_VALUE) <= 1e-9) return 'singularity_proximal';
  if (Math.abs(val - MU_3_VALUE) <= 1e-9) return 'unity_proximal';
  return 'post_unity';
}

// Time harmonic helper (delegates t6 to provided gate, default lens)
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

// Hex prism helpers
// Geometry helpers (match docs/HEXAGONAL_NEG_ENTROPY_PROJECTION.md & Python)
function hexPrismRadius(deltaSNeg /* ΔS_neg */) {
  return GEOM_R_MAX - GEOM_BETA * deltaSNeg;
}

function hexPrismHeight(deltaSNeg /* ΔS_neg */) {
  return GEOM_H_MIN + GEOM_GAMMA * deltaSNeg;
}

function hexPrismTwist(deltaSNeg /* ΔS_neg */) {
  return GEOM_PHI_BASE + GEOM_ETA * deltaSNeg;
}

// Aliases matching documentation naming (optional convenience)
const T1_MAX = Z_T1_MAX;
const T2_MAX = Z_T2_MAX;
const T3_MAX = Z_T3_MAX;
const T4_MAX = Z_T4_MAX;
const T5_MAX = Z_T5_MAX;
const T7_MAX = Z_T7_MAX;
const T8_MAX = Z_T8_MAX;

// ============================================================================
// EXTENDED MODULE LAZY LOADERS
// ============================================================================

/**
 * Lazy load S₃ operator symmetry module
 * @returns {Object} S₃ module exports
 */
function getS3Module() {
  return require('./s3_operator_symmetry');
}

/**
 * Lazy load extended ΔS⁻ module
 * @returns {Object} Extended ΔS⁻ module exports
 */
function getDeltaExtendedModule() {
  return require('./delta_s_neg_extended');
}

// Extended function re-exports (lazy loaded to avoid circular dependencies)

/**
 * Compute ΔS⁻ derivative
 * @param {number} z - Z-coordinate
 * @param {number} sigma - Gaussian width
 * @returns {number} Derivative value
 */
function computeDeltaSNegDerivative(z, sigma = LENS_SIGMA) {
  return getDeltaExtendedModule().computeDeltaSNegDerivative(z, sigma);
}

/**
 * Compute signed ΔS⁻
 * @param {number} z - Z-coordinate
 * @param {number} sigma - Gaussian width
 * @returns {number} Signed value
 */
function computeDeltaSNegSigned(z, sigma = LENS_SIGMA) {
  return getDeltaExtendedModule().computeDeltaSNegSigned(z, sigma);
}

/**
 * Compute Π-regime blend weights
 * @param {number} z - Z-coordinate
 * @param {boolean} enableBlend - Whether blending is enabled
 * @returns {Object} Blend weights
 */
function computePiBlendWeights(z, enableBlend = true) {
  return getDeltaExtendedModule().computePiBlendWeights(z, enableBlend);
}

/**
 * Compute gate modulation parameters
 * @param {number} z - Z-coordinate
 * @param {Object} options - Modulation options
 * @returns {Object} Gate modulation
 */
function computeGateModulation(z, options = {}) {
  return getDeltaExtendedModule().computeGateModulation(z, options);
}

/**
 * Compute full ΔS⁻ state
 * @param {number} z - Z-coordinate
 * @param {Object} options - State options
 * @returns {Object} Full state
 */
function computeFullState(z, options = {}) {
  return getDeltaExtendedModule().computeFullState(z, options);
}

/**
 * Generate S₃ operator window
 * @param {string} harmonic - Harmonic tier
 * @param {number} z - Z-coordinate
 * @param {Object} options - Window options
 * @returns {string[]} Operator window
 */
function generateS3OperatorWindow(harmonic, z, options = {}) {
  return getS3Module().generateS3OperatorWindow(harmonic, z, options);
}

/**
 * Compute S₃ weights for operators
 * @param {string[]} operators - Operator list
 * @param {number} z - Z-coordinate
 * @returns {Object} Operator weights
 */
function computeS3Weights(operators, z) {
  return getS3Module().computeS3Weights(operators, z);
}

/**
 * Compute dynamic truth bias
 * @param {number} z - Z-coordinate
 * @param {Object} baseBias - Base truth bias
 * @returns {Object} Evolved truth bias
 */
function computeDynamicTruthBias(z, baseBias = TRUTH_BIAS) {
  return getDeltaExtendedModule().computeDynamicTruthBias(z, baseBias);
}

/**
 * Score operator for coherence objective
 * @param {string} operator - Operator to score
 * @param {number} z - Z-coordinate
 * @param {string} objective - Coherence objective
 * @returns {number} Score
 */
function scoreOperatorForCoherence(operator, z, objective = 'maximize') {
  return getDeltaExtendedModule().scoreOperatorForCoherence(operator, z, objective);
}

/**
 * Select best operator for coherence
 * @param {string[]} operators - Available operators
 * @param {number} z - Z-coordinate
 * @param {string} objective - Coherence objective
 * @returns {Object} Selection result
 */
function selectCoherenceOperator(operators, z, objective = 'maximize') {
  return getDeltaExtendedModule().selectCoherenceOperator(operators, z, objective);
}

module.exports = Object.freeze({
  Z_CRITICAL,
  TRIAD_HIGH,
  TRIAD_LOW,
  TRIAD_T6,
  Z_ABSENCE_MAX,
  Z_LENS_MIN,
  Z_LENS_MAX,
  Z_PRESENCE_MIN,
  PHI,
  PHI_INV,
  Q_KAPPA,
  KAPPA_S,
  LAMBDA,
  KAPPA_MIN,
  ETA_MIN,
  R_MIN,
  ENTROPY_MIN,
  PURITY_MIN,
  PURITY_MAX,
  OPERATOR_PREFERRED_WEIGHT,
  OPERATOR_DEFAULT_WEIGHT,
  TRUTH_BIAS,
  Z_T1_MAX,
  Z_T2_MAX,
  Z_T3_MAX,
  Z_T4_MAX,
  Z_T5_MAX,
  Z_T7_MAX,
  Z_T8_MAX,
  GEOM_SIGMA,
  GEOM_R_MAX,
  GEOM_BETA,
  GEOM_H_MIN,
  GEOM_GAMMA,
  GEOM_PHI_BASE,
  GEOM_ETA,
  LENS_SIGMA,
  PRS_P1_PHI_MAX,
  PRS_P2_PHI_MAX,
  PRS_P3_PHI_MAX,
  PRS_P4_PHI_MAX,
  QAPL_RANDOM_SEED,
  // Doc-friendly aliases
  T1_MAX,
  T2_MAX,
  T3_MAX,
  T4_MAX,
  T5_MAX,
  T7_MAX,
  T8_MAX,
  isCritical,
  isInLens,
  getPhase,
  distanceToCritical,
  checkKFormation,
  computeDeltaSNeg,
  computeEta,
  // Spec aliases
  deltaSneg,
  lensRate,
  etaFromZ,
  etaFromOverlap,
  phiGateScale,
  phiRateScaled,
  checkKFormationFromZ,
  checkKFormationFromOverlap,
  // μ thresholds (values + barrier)
  MU_P: MU_P_VALUE,
  MU_1: MU_1_VALUE,
  MU_2: MU_2_VALUE,
  MU_S: MU_S_VALUE,
  MU_3: MU_3_VALUE,
  MU_BARRIER: MU_BARRIER_VALUE,
  MU_BARRIER_VALUE: MU_BARRIER_VALUE,
  classifyMu,
  classifyThreshold: classifyMu,
  // Invariants
  invariants() {
    const mu1 = MU_1_VALUE, mu2 = MU_2_VALUE;
    return {
      barrier_eq_phi_inv: Math.abs(0.5 * (mu1 + mu2) - PHI_INV) < 1e-12,
      wells_ratio_phi: Math.abs((mu2 / mu1) - PHI) < 1e-12,
      ordering_ok: mu2 < TRIAD_LOW && TRIAD_LOW < TRIAD_T6 && TRIAD_T6 < TRIAD_HIGH && TRIAD_HIGH < Z_CRITICAL,
    };
  },
  getTimeHarmonic,
  hexPrismRadius,
  hexPrismHeight,
  hexPrismTwist,

  // Extended ΔS⁻ module exports
  computeDeltaSNegDerivative,
  computeDeltaSNegSigned,
  computePiBlendWeights,
  computeGateModulation,
  computeFullState,
  computeDynamicTruthBias,
  scoreOperatorForCoherence,
  selectCoherenceOperator,

  // S₃ symmetry module exports
  generateS3OperatorWindow,
  computeS3Weights,

  // Module loaders (for advanced use)
  getS3Module,
  getDeltaExtendedModule,
});
