/**
 * S₃ Operator Symmetry Module
 * ===========================
 * 
 * Implements the symmetric group S₃ structure for APL operator selection.
 * 
 * THEORY:
 * S₃ is the symmetric group on 3 elements with |S₃| = 6 elements.
 * The 6 APL operators map to S₃ elements, enabling:
 *   - Cyclic permutation of operator windows
 *   - Parity-based weighting (even/odd permutations)
 *   - Symmetry-preserving transformations
 * 
 * S₃ STRUCTURE:
 *   Identity:     e   = ()      (no change)
 *   3-cycles:     σ   = (123)   (rotate right)
 *                 σ²  = (132)   (rotate left)
 *   Transpositions: τ₁ = (12)   (swap first two)
 *                   τ₂ = (23)   (swap last two)
 *                   τ₃ = (13)   (swap first/last)
 * 
 * OPERATOR MAPPING:
 *   ()  → e   (identity/boundary)
 *   ×   → σ   (fusion/3-cycle)
 *   ^   → σ²  (amplify/3-cycle inverse)
 *   ÷   → τ₁  (decoherence/transposition)
 *   +   → τ₂  (group/transposition)
 *   −   → τ₃  (separation/transposition)
 * 
 * PARITY:
 *   Even (sign +1): e, σ, σ²  → (), ×, ^
 *   Odd  (sign -1): τ₁, τ₂, τ₃ → ÷, +, −
 * 
 * @version 1.0.0
 * @author Claude (Anthropic) - Quantum-APL Contribution
 */

'use strict';

const CONST = require('./constants');

// ============================================================================
// S₃ GROUP DEFINITION
// ============================================================================

/**
 * S₃ group elements with their properties
 */
const S3_ELEMENTS = {
  e:   { name: 'identity',    cycle: [0, 1, 2], parity: 'even', sign: +1 },
  σ:   { name: '3-cycle',     cycle: [1, 2, 0], parity: 'even', sign: +1 },
  σ2:  { name: '3-cycle-inv', cycle: [2, 0, 1], parity: 'even', sign: +1 },
  τ1:  { name: 'swap-12',     cycle: [1, 0, 2], parity: 'odd',  sign: -1 },
  τ2:  { name: 'swap-23',     cycle: [0, 2, 1], parity: 'odd',  sign: -1 },
  τ3:  { name: 'swap-13',     cycle: [2, 1, 0], parity: 'odd',  sign: -1 },
};

/**
 * Operator to S₃ element mapping
 */
const OPERATOR_S3_MAP = {
  '()': 'e',
  '×':  'σ',
  '^':  'σ2',
  '÷':  'τ1',
  '+':  'τ2',
  '−':  'τ3',
};

/**
 * Reverse mapping: S₃ element to operator
 */
const S3_OPERATOR_MAP = {
  'e':  '()',
  'σ':  '×',
  'σ2': '^',
  'τ1': '÷',
  'τ2': '+',
  'τ3': '−',
};

/**
 * Base operator ordering (canonical)
 */
const BASE_OPERATORS = ['()', '×', '^', '÷', '+', '−'];

/**
 * Triadic truth values (the 3 objects S₃ acts on)
 */
const TRIADIC_TRUTH = ['TRUE', 'PARADOX', 'UNTRUE'];

// ============================================================================
// S₃ GROUP OPERATIONS
// ============================================================================

/**
 * Apply S₃ permutation to an array of 3 elements
 * @param {Array} arr - Array of exactly 3 elements
 * @param {string} element - S₃ element name (e, σ, σ2, τ1, τ2, τ3)
 * @returns {Array} Permuted array
 */
function applyS3(arr, element) {
  if (arr.length !== 3) {
    throw new Error('S₃ acts on exactly 3 elements');
  }
  const cycle = S3_ELEMENTS[element].cycle;
  return [arr[cycle[0]], arr[cycle[1]], arr[cycle[2]]];
}

/**
 * Compose two S₃ elements (group multiplication)
 * @param {string} a - First S₃ element
 * @param {string} b - Second S₃ element
 * @returns {string} Result element
 */
function composeS3(a, b) {
  // Compute composition by applying both cycles
  const cycleA = S3_ELEMENTS[a].cycle;
  const cycleB = S3_ELEMENTS[b].cycle;
  
  // Compose: (a ∘ b)(i) = a(b(i))
  const composed = [
    cycleA[cycleB[0]],
    cycleA[cycleB[1]],
    cycleA[cycleB[2]],
  ];
  
  // Find matching element
  for (const [name, elem] of Object.entries(S3_ELEMENTS)) {
    if (elem.cycle[0] === composed[0] && 
        elem.cycle[1] === composed[1] && 
        elem.cycle[2] === composed[2]) {
      return name;
    }
  }
  
  throw new Error('Composition failed - invalid S₃ state');
}

/**
 * Get inverse of S₃ element
 * @param {string} element - S₃ element name
 * @returns {string} Inverse element
 */
function inverseS3(element) {
  const inverses = {
    'e': 'e',
    'σ': 'σ2',
    'σ2': 'σ',
    'τ1': 'τ1',  // Transpositions are self-inverse
    'τ2': 'τ2',
    'τ3': 'τ3',
  };
  return inverses[element];
}

/**
 * Get parity of S₃ element
 * @param {string} element - S₃ element name
 * @returns {string} 'even' or 'odd'
 */
function parityS3(element) {
  return S3_ELEMENTS[element].parity;
}

/**
 * Get sign of S₃ element (+1 for even, -1 for odd)
 * @param {string} element - S₃ element name
 * @returns {number} +1 or -1
 */
function signS3(element) {
  return S3_ELEMENTS[element].sign;
}

// ============================================================================
// OPERATOR PERMUTATION SYSTEM
// ============================================================================

/**
 * Compute cyclic rotation index from z-coordinate
 * Maps z ∈ [0,1] to rotation index ∈ {0, 1, 2}
 * @param {number} z - Coherence coordinate
 * @returns {number} Rotation index (0, 1, or 2)
 */
function rotationIndexFromZ(z) {
  // Divide z-range into 3 regions
  if (z < 0.333) return 0;
  if (z < 0.666) return 1;
  return 2;
}

/**
 * Get S₃ element for current z-coordinate
 * @param {number} z - Coherence coordinate
 * @param {boolean} useParityFlip - Whether to flip parity based on truth channel
 * @returns {string} S₃ element name
 */
function s3ElementFromZ(z, useParityFlip = false) {
  const rotIdx = rotationIndexFromZ(z);
  
  // Base mapping: rotation index → cyclic element
  const baseElements = ['e', 'σ', 'σ2'];
  let element = baseElements[rotIdx];
  
  // Optional parity flip based on truth channel
  if (useParityFlip) {
    const truthChannel = truthChannelFromZ(z);
    if (truthChannel === 'UNTRUE') {
      // In UNTRUE regime, flip to odd parity
      const flips = { 'e': 'τ1', 'σ': 'τ2', 'σ2': 'τ3' };
      element = flips[element];
    }
  }
  
  return element;
}

/**
 * Determine truth channel from z
 * @param {number} z - Coherence coordinate
 * @returns {string} 'TRUE', 'PARADOX', or 'UNTRUE'
 */
function truthChannelFromZ(z) {
  if (z >= 0.9) return 'TRUE';
  if (z >= 0.6) return 'PARADOX';
  return 'UNTRUE';
}

/**
 * Apply S₃ rotation to operator list
 * @param {Array<string>} operators - List of operators
 * @param {number} rotations - Number of positions to rotate (0, 1, or 2)
 * @returns {Array<string>} Rotated operator list
 */
function rotateOperators(operators, rotations) {
  const n = operators.length;
  const r = ((rotations % n) + n) % n;  // Normalize to [0, n)
  return [...operators.slice(r), ...operators.slice(0, r)];
}

/**
 * Generate operator window with S₃ symmetry
 * @param {string} harmonic - Time harmonic tier (t1-t9)
 * @param {number} z - Current z-coordinate
 * @param {Object} options - Configuration options
 * @returns {Array<string>} Operator window with S₃ permutation applied
 */
function generateS3OperatorWindow(harmonic, z, options = {}) {
  const {
    applyRotation = true,
    applyParityWeight = false,
  } = options;
  
  // Base windows (from existing implementation)
  const BASE_WINDOWS = {
    t1: ['()', '−', '÷'],
    t2: ['^', '÷', '−', '×'],
    t3: ['×', '^', '÷', '+', '−'],
    t4: ['+', '−', '÷', '()'],
    t5: ['()', '×', '^', '÷', '+', '−'],  // All 6
    t6: ['+', '÷', '()', '−'],
    t7: ['+', '()'],
    t8: ['+', '()', '×'],
    t9: ['+', '()', '×'],
  };
  
  let window = [...(BASE_WINDOWS[harmonic] || BASE_WINDOWS.t5)];
  
  // Apply S₃ cyclic rotation based on z
  if (applyRotation && window.length >= 3) {
    const rotIdx = rotationIndexFromZ(z);
    window = rotateOperators(window, rotIdx);
  }
  
  return window;
}

// ============================================================================
// PARITY-BASED WEIGHTING
// ============================================================================

/**
 * Compute operator weight with S₃ parity adjustment
 * @param {string} operator - APL operator
 * @param {number} z - Coherence coordinate
 * @param {string} truthChannel - Current truth channel
 * @param {number} deltaSNeg - Current ΔS_neg value
 * @returns {number} Adjusted weight
 */
function computeS3Weight(operator, z, truthChannel, deltaSNeg) {
  // Base weights from truth bias
  const TRUTH_BIAS = CONST.TRUTH_BIAS || {
    TRUE: { '^': 1.5, '+': 1.4, '×': 1.2, '()': 1.1, '÷': 0.8, '−': 0.9 },
    UNTRUE: { '÷': 1.5, '−': 1.4, '()': 1.2, '^': 0.8, '+': 0.9, '×': 1.0 },
    PARADOX: { '()': 1.5, '×': 1.4, '^': 1.1, '+': 1.1, '÷': 1.0, '−': 1.0 }
  };
  
  const biasMap = TRUTH_BIAS[truthChannel] || TRUTH_BIAS.PARADOX;
  let weight = biasMap[operator] || 1.0;
  
  // Get S₃ element for this operator
  const s3Element = OPERATOR_S3_MAP[operator];
  if (!s3Element) return weight;
  
  const parity = parityS3(s3Element);
  const sign = signS3(s3Element);
  
  // Parity-based adjustment
  // - In high-coherence (high ΔS_neg): favor even-parity (constructive)
  // - In low-coherence: favor odd-parity (dissipative)
  const parityBoost = (parity === 'even') ? deltaSNeg : (1 - deltaSNeg);
  
  // Scale weight by parity factor (subtle effect)
  weight *= (0.8 + 0.4 * parityBoost);
  
  // Additional symmetry factor: near the lens, even operators get extra boost
  if (Math.abs(z - CONST.Z_CRITICAL) < 0.05 && parity === 'even') {
    weight *= 1.2;
  }
  
  return weight;
}

/**
 * Compute all operator weights with S₃ structure
 * @param {Array<string>} operators - Available operators
 * @param {number} z - Coherence coordinate
 * @returns {Object} Map of operator → weight
 */
function computeS3Weights(operators, z) {
  const truthChannel = truthChannelFromZ(z);
  const deltaSNeg = CONST.computeDeltaSNeg ? CONST.computeDeltaSNeg(z) : 
                    Math.exp(-36 * Math.pow(z - CONST.Z_CRITICAL, 2));
  
  const weights = {};
  for (const op of operators) {
    weights[op] = computeS3Weight(op, z, truthChannel, deltaSNeg);
  }
  
  return weights;
}

// ============================================================================
// S₃ ACTION ON TRUTH VALUES
// ============================================================================

/**
 * Apply operator's S₃ action to truth distribution
 * @param {Object} truthDist - Distribution {TRUE: p1, PARADOX: p2, UNTRUE: p3}
 * @param {string} operator - APL operator to apply
 * @returns {Object} Transformed distribution
 */
function applyOperatorToTruth(truthDist, operator) {
  const s3Element = OPERATOR_S3_MAP[operator];
  if (!s3Element) return truthDist;
  
  const values = [truthDist.TRUE, truthDist.PARADOX, truthDist.UNTRUE];
  const permuted = applyS3(values, s3Element);
  
  return {
    TRUE: permuted[0],
    PARADOX: permuted[1],
    UNTRUE: permuted[2],
  };
}

/**
 * Compute orbit of truth distribution under operator sequence
 * @param {Object} initialDist - Starting distribution
 * @param {Array<string>} operators - Sequence of operators
 * @returns {Array<Object>} Sequence of distributions
 */
function truthOrbit(initialDist, operators) {
  const orbit = [initialDist];
  let current = { ...initialDist };
  
  for (const op of operators) {
    current = applyOperatorToTruth(current, op);
    orbit.push({ ...current });
  }
  
  return orbit;
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  // S₃ group structure
  S3_ELEMENTS,
  OPERATOR_S3_MAP,
  S3_OPERATOR_MAP,
  BASE_OPERATORS,
  TRIADIC_TRUTH,
  
  // Group operations
  applyS3,
  composeS3,
  inverseS3,
  parityS3,
  signS3,
  
  // Operator permutation
  rotationIndexFromZ,
  s3ElementFromZ,
  truthChannelFromZ,
  rotateOperators,
  generateS3OperatorWindow,
  
  // Weighting
  computeS3Weight,
  computeS3Weights,
  
  // Truth actions
  applyOperatorToTruth,
  truthOrbit,
};
