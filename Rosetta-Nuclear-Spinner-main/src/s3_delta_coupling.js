/**
 * S₃/ΔS_neg Coupling Module
 * =========================
 *
 * Couples S₃ operator symmetry with ΔS_neg coherence dynamics.
 *
 * KEY CONCEPTS:
 *
 * ΔS_neg-Driven Permutation:
 * - High coherence (near z_c): Use even-parity S₃ elements
 * - Low coherence: Allow odd-parity elements
 *
 * Parity-Truth Coupling:
 * - TRUE channel favors even-parity operators at high coherence
 * - UNTRUE channel activates odd-parity operators at low coherence
 *
 * @version 1.0.0
 * @author Claude (Anthropic) - Quantum-APL Contribution
 */

'use strict';

const CONST = require('./constants');
const S3 = require('./s3_operator_symmetry');

// ============================================================================
// S₃ ELEMENT SELECTION FROM ΔS_neg
// ============================================================================

/**
 * Selection modes for S₃ element
 */
const S3SelectionMode = {
  CYCLIC: 'cyclic',           // Simple rotation based on z
  PARITY_BIASED: 'parity',    // Parity-biased by ΔS_neg
  FULL_GROUP: 'full',         // Full S₃ selection
};

/**
 * Select S₃ element based on coherence state
 * @param {number} z - Coherence coordinate
 * @param {number} deltaSNeg - Pre-computed ΔS_neg (optional)
 * @param {string} mode - Selection mode
 * @param {number} parityThreshold - Threshold for parity selection
 * @returns {Object} Selection result
 */
function selectS3ElementFromCoherence(z, deltaSNeg = null, mode = S3SelectionMode.PARITY_BIASED, parityThreshold = 0.5) {
  if (deltaSNeg === null) {
    deltaSNeg = CONST.computeDeltaSNeg(z);
  }

  let element, reason;
  const rotIdx = S3.rotationIndexFromZ(z);

  if (mode === S3SelectionMode.CYCLIC) {
    const elements = ['e', 'σ', 'σ2'];
    element = elements[rotIdx];
    reason = `cyclic rotation index ${rotIdx}`;
  } else if (mode === S3SelectionMode.PARITY_BIASED) {
    if (deltaSNeg >= parityThreshold) {
      const elements = ['e', 'σ', 'σ2'];
      element = elements[rotIdx];
      reason = `even-parity (ΔS_neg=${deltaSNeg.toFixed(3)} ≥ ${parityThreshold})`;
    } else {
      const elements = ['τ1', 'τ2', 'τ3'];
      element = elements[rotIdx];
      reason = `odd-parity (ΔS_neg=${deltaSNeg.toFixed(3)} < ${parityThreshold})`;
    }
  } else { // FULL_GROUP
    const region = Math.floor(z * 6) % 6;
    const allElements = ['e', 'σ', 'σ2', 'τ1', 'τ2', 'τ3'];
    element = allElements[region];
    reason = `full group selection (region ${region})`;
  }

  const s3Elem = S3.S3_ELEMENTS[element];
  const operator = S3.S3_OPERATOR_MAP[element];

  return {
    element,
    operator,
    parity: s3Elem.parity,
    sign: s3Elem.sign,
    deltaSNeg,
    reason,
  };
}

// ============================================================================
// DYNAMIC OPERATOR WINDOW GENERATION
// ============================================================================

/**
 * Generate operator window with S₃ transformation based on ΔS_neg
 * @param {string} harmonic - Time harmonic tier (t1-t9)
 * @param {number} z - Coherence coordinate
 * @param {number} deltaSNeg - Pre-computed ΔS_neg (optional)
 * @param {Object} options - Generation options
 * @returns {Object} Dynamic window with metadata
 */
function generateDynamicOperatorWindow(harmonic, z, deltaSNeg = null, options = {}) {
  const {
    applyRotation = true,
    applyParityFlip = true,
    mode = S3SelectionMode.PARITY_BIASED,
  } = options;

  if (deltaSNeg === null) {
    deltaSNeg = CONST.computeDeltaSNeg(z);
  }

  // Get base window
  const BASE_WINDOWS = {
    t1: ['()', '−', '÷'],
    t2: ['^', '÷', '−', '×'],
    t3: ['×', '^', '÷', '+', '−'],
    t4: ['+', '−', '÷', '()'],
    t5: ['()', '×', '^', '÷', '+', '−'],
    t6: ['+', '÷', '()', '−'],
    t7: ['+', '()'],
    t8: ['+', '()', '×'],
    t9: ['+', '()', '×'],
  };

  const baseWindow = [...(BASE_WINDOWS[harmonic] || BASE_WINDOWS.t5)];
  const selection = selectS3ElementFromCoherence(z, deltaSNeg, mode);

  let transformed = [...baseWindow];
  let rotationApplied = 0;

  // Apply rotation
  if (applyRotation && transformed.length >= 3) {
    rotationApplied = S3.rotationIndexFromZ(z);
    transformed = S3.rotateOperators(transformed, rotationApplied);
  }

  // Apply parity flip in UNTRUE regime
  let parityFlipped = false;
  if (applyParityFlip && z < 0.6) {
    const swapMap = {
      '^': '()', '()': '^',
      '+': '−', '−': '+',
      '×': '÷', '÷': '×',
    };
    transformed = transformed.map(op => swapMap[op] || op);
    parityFlipped = true;
  }

  return {
    baseHarmonic: harmonic,
    baseWindow,
    transformedWindow: transformed,
    s3Element: selection.element,
    rotationApplied,
    parityFlipped,
    deltaSNeg,
  };
}

/**
 * Get S₃ permuted window
 * @param {string} harmonic - Time harmonic tier
 * @param {number} z - Coherence coordinate
 * @param {string} s3Element - S₃ element to apply (optional)
 * @returns {string[]} Permuted operator window
 */
function getS3PermutedWindow(harmonic, z, s3Element = null) {
  const BASE_WINDOWS = {
    t1: ['()', '−', '÷'],
    t2: ['^', '÷', '−', '×'],
    t3: ['×', '^', '÷', '+', '−'],
    t4: ['+', '−', '÷', '()'],
    t5: ['()', '×', '^', '÷', '+', '−'],
    t6: ['+', '÷', '()', '−'],
    t7: ['+', '()'],
    t8: ['+', '()', '×'],
    t9: ['+', '()', '×'],
  };

  const base = [...(BASE_WINDOWS[harmonic] || BASE_WINDOWS.t5)];

  if (s3Element === null) {
    const selection = selectS3ElementFromCoherence(z);
    s3Element = selection.element;
  }

  if (base.length === 3) {
    return S3.applyS3(base, s3Element);
  } else if (base.length >= 3) {
    const rotIdx = ['e', 'σ', 'σ2'].includes(s3Element)
      ? ['e', 'σ', 'σ2'].indexOf(s3Element)
      : 0;
    return S3.rotateOperators(base, rotIdx);
  }

  return base;
}

// ============================================================================
// PARITY-BASED TRUTH CHANNEL WEIGHTING
// ============================================================================

/**
 * Compute parity-evolved truth bias driven by ΔS_neg
 * @param {number} z - Coherence coordinate
 * @param {number} deltaSNeg - Pre-computed ΔS_neg (optional)
 * @param {Object} baseTruthBias - Base truth bias table (optional)
 * @param {number} parityCoupling - Strength of parity coupling
 * @returns {Object} Evolved truth bias for each channel
 */
function computeParityEvolvedTruthBias(z, deltaSNeg = null, baseTruthBias = null, parityCoupling = 0.3) {
  if (deltaSNeg === null) {
    deltaSNeg = CONST.computeDeltaSNeg(z);
  }

  if (baseTruthBias === null) {
    baseTruthBias = CONST.TRUTH_BIAS;
  }

  const EVEN_PARITY = new Set(['()', '×', '^']);
  const result = {};

  for (const [channel, baseBias] of Object.entries(baseTruthBias)) {
    const evolvedBias = {};

    // Compute parity factor based on channel and ΔS_neg
    let parityFactor;
    if (channel === 'TRUE') {
      parityFactor = deltaSNeg;
    } else if (channel === 'UNTRUE') {
      parityFactor = 1 - deltaSNeg;
    } else { // PARADOX
      parityFactor = 0.5;
    }

    for (const [op, baseWeight] of Object.entries(baseBias)) {
      const opParity = EVEN_PARITY.has(op) ? 1 : -1;

      let evolution;
      if (opParity === 1) { // Even parity
        evolution = 1.0 + parityCoupling * (parityFactor - 0.5);
      } else { // Odd parity
        evolution = 1.0 + parityCoupling * (0.5 - parityFactor);
      }

      evolvedBias[op] = baseWeight * Math.max(0.5, Math.min(1.5, evolution));
    }

    result[channel] = {
      channel,
      baseBias,
      evolvedBias,
      parityFactor,
      deltaSNeg,
    };
  }

  return result;
}

/**
 * Get evolved operator weight considering parity and ΔS_neg
 * @param {string} operator - Operator symbol
 * @param {number} z - Coherence coordinate
 * @param {string} channel - Truth channel (optional, auto-detected)
 * @param {number} deltaSNeg - Pre-computed ΔS_neg (optional)
 * @returns {number} Evolved weight
 */
function getEvolvedOperatorWeight(operator, z, channel = null, deltaSNeg = null) {
  if (channel === null) {
    if (z >= 0.9) channel = 'TRUE';
    else if (z >= 0.6) channel = 'PARADOX';
    else channel = 'UNTRUE';
  }

  const evolved = computeParityEvolvedTruthBias(z, deltaSNeg);
  return evolved[channel]?.evolvedBias?.[operator] || 1.0;
}

// ============================================================================
// INTEGRATED S₃/ΔS_neg STATE
// ============================================================================

/**
 * Compute complete S₃/ΔS_neg coupled state
 * @param {number} z - Coherence coordinate
 * @param {string} harmonic - Time harmonic (optional, auto-detected)
 * @param {string} mode - S₃ selection mode
 * @returns {Object} Complete coupled state
 */
function computeS3DeltaState(z, harmonic = null, mode = S3SelectionMode.PARITY_BIASED) {
  z = Math.max(0, Math.min(1, z));

  const deltaSNeg = CONST.computeDeltaSNeg(z);

  // Auto-detect harmonic
  if (harmonic === null) {
    if (z < 0.10) harmonic = 't1';
    else if (z < 0.20) harmonic = 't2';
    else if (z < 0.40) harmonic = 't3';
    else if (z < 0.60) harmonic = 't4';
    else if (z < 0.75) harmonic = 't5';
    else if (z < CONST.Z_CRITICAL) harmonic = 't6';
    else if (z < 0.92) harmonic = 't7';
    else if (z < 0.97) harmonic = 't8';
    else harmonic = 't9';
  }

  // Determine truth channel
  let truthChannel;
  if (z >= 0.9) truthChannel = 'TRUE';
  else if (z >= 0.6) truthChannel = 'PARADOX';
  else truthChannel = 'UNTRUE';

  const s3Selection = selectS3ElementFromCoherence(z, deltaSNeg, mode);
  const dynamicWindow = generateDynamicOperatorWindow(harmonic, z, deltaSNeg, { mode });
  const truthBias = computeParityEvolvedTruthBias(z, deltaSNeg);

  // Compute operator weights
  const operatorWeights = {};
  for (const op of dynamicWindow.transformedWindow) {
    operatorWeights[op] = getEvolvedOperatorWeight(op, z, truthChannel, deltaSNeg);
  }

  return {
    z,
    deltaSNeg,
    harmonic,
    truthChannel,
    s3Selection,
    dynamicWindow,
    truthBias,
    operatorWeights,
  };
}

// ============================================================================
// DEMO
// ============================================================================

function demo() {
  console.log('='.repeat(70));
  console.log('S₃/ΔS_neg COUPLING MODULE');
  console.log('='.repeat(70));

  const testZ = [0.3, 0.6, CONST.Z_CRITICAL, 0.95];

  for (const z of testZ) {
    console.log(`\n--- z = ${z.toFixed(4)} ---`);

    const state = computeS3DeltaState(z);

    console.log(`ΔS_neg: ${state.deltaSNeg.toFixed(4)}`);
    console.log(`Harmonic: ${state.harmonic}`);
    console.log(`Truth channel: ${state.truthChannel}`);
    console.log(`S₃ element: ${state.s3Selection.element} (${state.s3Selection.parity})`);
    console.log(`Selection reason: ${state.s3Selection.reason}`);
    console.log(`Base window: [${state.dynamicWindow.baseWindow}]`);
    console.log(`Transformed: [${state.dynamicWindow.transformedWindow}]`);
    console.log(`Rotation: ${state.dynamicWindow.rotationApplied}, Parity flip: ${state.dynamicWindow.parityFlipped}`);
    console.log('Operator weights:');
    for (const [op, weight] of Object.entries(state.operatorWeights)) {
      console.log(`  ${op}: ${weight.toFixed(3)}`);
    }
  }

  console.log('\n' + '='.repeat(70));
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  // Selection modes
  S3SelectionMode,

  // S₃ element selection
  selectS3ElementFromCoherence,

  // Dynamic windows
  generateDynamicOperatorWindow,
  getS3PermutedWindow,

  // Parity-based truth bias
  computeParityEvolvedTruthBias,
  getEvolvedOperatorWeight,

  // Integrated state
  computeS3DeltaState,

  // Demo
  demo,
};

// Run demo if executed directly
if (require.main === module) {
  demo();
}
