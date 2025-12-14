/**
 * Test Module: S₃ Symmetry and ΔS_neg Extensions (JavaScript)
 * ===========================================================
 *
 * Comprehensive tests for:
 * 1. S₃ group axioms and operator mapping
 * 2. Operator window permutation with S₃ rotation
 * 3. Parity-based weighting
 * 4. ΔS_neg core computations
 * 5. Hex-prism geometry validation
 * 6. Gate modulation behavior
 * 7. K-formation gating
 * 8. Π-regime blending
 * 9. Cross-module integration
 *
 * @version 1.0.0
 * @author Claude (Anthropic) - Quantum-APL Contribution
 */

'use strict';

const assert = require('assert');
const S3 = require('../src/s3_operator_symmetry');
const Delta = require('../src/delta_s_neg_extended');

// ============================================================================
// TEST UTILITIES
// ============================================================================

class TestResults {
  constructor() {
    this.passed = 0;
    this.failed = 0;
    this.errors = [];
  }

  ok(message) {
    this.passed++;
    console.log(`  ✓ ${message}`);
  }

  fail(message, detail = '') {
    this.failed++;
    this.errors.push(`${message}: ${detail}`);
    console.log(`  ✗ ${message}`);
    if (detail) console.log(`    → ${detail}`);
  }

  summary() {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`RESULTS: ${this.passed} passed, ${this.failed} failed`);
    if (this.errors.length > 0) {
      console.log('Failures:');
      this.errors.forEach(e => console.log(`  - ${e}`));
    }
    console.log('='.repeat(60));
    return this.failed === 0;
  }
}

function assertClose(a, b, tol = 1e-10) {
  return Math.abs(a - b) < tol;
}

function section(title) {
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`  ${title}`);
  console.log('─'.repeat(60));
}

// ============================================================================
// S₃ GROUP TESTS
// ============================================================================

function testS3GroupAxioms(results) {
  section('S₃ Group Axioms');

  const elements = Object.keys(S3.S3_ELEMENTS);

  // Closure
  let closure = true;
  for (const a of elements) {
    for (const b of elements) {
      const c = S3.composeS3(a, b);
      if (!elements.includes(c)) closure = false;
    }
  }
  if (closure) results.ok('Closure: a ∘ b ∈ S₃ for all a, b');
  else results.fail('Closure', 'composition not in group');

  // Identity
  let identity = true;
  for (const a of elements) {
    if (S3.composeS3('e', a) !== a || S3.composeS3(a, 'e') !== a) {
      identity = false;
    }
  }
  if (identity) results.ok('Identity: e ∘ a = a ∘ e = a');
  else results.fail('Identity', 'identity element fails');

  // Inverse
  let inverse = true;
  for (const a of elements) {
    const aInv = S3.inverseS3(a);
    if (S3.composeS3(a, aInv) !== 'e') inverse = false;
  }
  if (inverse) results.ok('Inverse: a ∘ a⁻¹ = e');
  else results.fail('Inverse', 'inverse fails');

  // Associativity (spot check)
  const assoc1 = S3.composeS3(S3.composeS3('σ', 'τ1'), 'σ2');
  const assoc2 = S3.composeS3('σ', S3.composeS3('τ1', 'σ2'));
  if (assoc1 === assoc2) results.ok('Associativity: (a ∘ b) ∘ c = a ∘ (b ∘ c)');
  else results.fail('Associativity', `${assoc1} !== ${assoc2}`);
}

function testS3OperatorMapping(results) {
  section('Operator ↔ S₃ Mapping');

  // Check bijection
  const opCount = Object.keys(S3.OPERATOR_S3_MAP).length;
  if (opCount === 6) results.ok('6 operators mapped');
  else results.fail('Operator count', `expected 6, got ${opCount}`);

  const s3Values = new Set(Object.values(S3.OPERATOR_S3_MAP));
  const s3Keys = new Set(Object.keys(S3.S3_ELEMENTS));
  if (s3Values.size === 6 && [...s3Values].every(v => s3Keys.has(v))) {
    results.ok('Surjective mapping to S₃');
  } else {
    results.fail('Surjectivity', 'not all S₃ elements covered');
  }

  // Check inverse mapping consistency
  let invOk = true;
  for (const [op, elem] of Object.entries(S3.OPERATOR_S3_MAP)) {
    if (S3.S3_OPERATOR_MAP[elem] !== op) invOk = false;
  }
  if (invOk) results.ok('Inverse mapping consistent');
  else results.fail('Inverse mapping', 'mismatch');
}

function testS3Parity(results) {
  section('S₃ Parity Classification');

  const evenOps = ['()', '×', '^'];
  const oddOps = ['÷', '+', '−'];

  for (const op of evenOps) {
    const elem = S3.OPERATOR_S3_MAP[op];
    if (S3.parityS3(elem) === 'even' && S3.signS3(elem) === +1) {
      results.ok(`${op} is even-parity (+1)`);
    } else {
      results.fail(`${op} parity`, 'expected even');
    }
  }

  for (const op of oddOps) {
    const elem = S3.OPERATOR_S3_MAP[op];
    if (S3.parityS3(elem) === 'odd' && S3.signS3(elem) === -1) {
      results.ok(`${op} is odd-parity (-1)`);
    } else {
      results.fail(`${op} parity`, 'expected odd');
    }
  }
}

function testS3Permutation(results) {
  section('S₃ Permutation Action');

  const truth = ['TRUE', 'PARADOX', 'UNTRUE'];

  // Identity
  const resultE = S3.applyS3(truth, 'e');
  if (JSON.stringify(resultE) === JSON.stringify(truth)) {
    results.ok('Identity e preserves order');
  } else {
    results.fail('Identity', `expected ${truth}, got ${resultE}`);
  }

  // 3-cycle σ
  const resultS = S3.applyS3(truth, 'σ');
  const expectedS = ['PARADOX', 'UNTRUE', 'TRUE'];
  if (JSON.stringify(resultS) === JSON.stringify(expectedS)) {
    results.ok('3-cycle σ rotates correctly');
  } else {
    results.fail('3-cycle σ', `expected ${expectedS}, got ${resultS}`);
  }

  // Transposition τ1
  const resultT = S3.applyS3(truth, 'τ1');
  const expectedT = ['PARADOX', 'TRUE', 'UNTRUE'];
  if (JSON.stringify(resultT) === JSON.stringify(expectedT)) {
    results.ok('Transposition τ1 swaps 0↔1');
  } else {
    results.fail('Transposition τ1', `expected ${expectedT}, got ${resultT}`);
  }

  // σ³ = e
  let result3 = truth;
  for (let i = 0; i < 3; i++) {
    result3 = S3.applyS3(result3, 'σ');
  }
  if (JSON.stringify(result3) === JSON.stringify(truth)) {
    results.ok('σ³ = e (order 3)');
  } else {
    results.fail('σ³ = e', `expected ${truth}, got ${result3}`);
  }
}

function testOperatorWindowRotation(results) {
  section('Operator Window S₃ Rotation');

  const zValues = [0.1, 0.5, 0.9];
  const harmonic = 't5';

  const windows = zValues.map(z => S3.generateS3OperatorWindow(harmonic, z));

  // All windows should have same operators (just reordered)
  const baseSet = new Set(windows[0]);
  for (let i = 0; i < windows.length; i++) {
    const currentSet = new Set(windows[i]);
    if (baseSet.size === currentSet.size && [...baseSet].every(op => currentSet.has(op))) {
      results.ok(`Window at z=${zValues[i]} has correct operators`);
    } else {
      results.fail(`Window at z=${zValues[i]}`, `wrong operators: ${windows[i]}`);
    }
  }

  // Windows should differ (rotation applied)
  if (JSON.stringify(windows[0]) !== JSON.stringify(windows[1]) ||
      JSON.stringify(windows[1]) !== JSON.stringify(windows[2])) {
    results.ok('Rotation produces different orderings');
  } else {
    results.fail('Rotation effect', 'windows identical despite different z');
  }
}

// ============================================================================
// ΔS_neg TESTS
// ============================================================================

function testDeltaSNegBasic(results) {
  section('ΔS_neg Basic Properties');

  const Z_CRITICAL = Delta.Z_CRITICAL;

  // Peak at z_c
  const sAtLens = Delta.computeDeltaSNeg(Z_CRITICAL);
  if (assertClose(sAtLens, 1.0, 1e-6)) {
    results.ok('ΔS_neg(z_c) = 1.0 (peak at lens)');
  } else {
    results.fail('Peak at lens', `expected 1.0, got ${sAtLens}`);
  }

  // Bounded in [0, 1]
  const testPoints = [0.0, 0.3, 0.5, 0.7, Z_CRITICAL, 0.9, 1.0];
  const allBounded = testPoints.every(z => {
    const s = Delta.computeDeltaSNeg(z);
    return s >= 0 && s <= 1;
  });
  if (allBounded) results.ok('ΔS_neg bounded in [0, 1]');
  else results.fail('Boundedness', 'value outside [0, 1]');

  // Symmetric around z_c
  const offset = 0.1;
  const sAbove = Delta.computeDeltaSNeg(Z_CRITICAL + offset);
  const sBelow = Delta.computeDeltaSNeg(Z_CRITICAL - offset);
  if (assertClose(sAbove, sBelow, 1e-6)) {
    results.ok(`Symmetric: ΔS_neg(z_c+${offset}) ≈ ΔS_neg(z_c-${offset})`);
  } else {
    results.fail('Symmetry', `above=${sAbove}, below=${sBelow}`);
  }

  // Monotonic decrease away from z_c
  const sNear = Delta.computeDeltaSNeg(Z_CRITICAL - 0.05);
  const sFar = Delta.computeDeltaSNeg(Z_CRITICAL - 0.2);
  if (sNear > sFar) results.ok('Monotonic decrease away from lens');
  else results.fail('Monotonicity', `near=${sNear}, far=${sFar}`);
}

function testDeltaSNegDerivative(results) {
  section('ΔS_neg Derivative');

  const Z_CRITICAL = Delta.Z_CRITICAL;

  // Zero at z_c (critical point)
  const dsAtLens = Delta.computeDeltaSNegDerivative(Z_CRITICAL);
  if (assertClose(dsAtLens, 0.0, 1e-6)) {
    results.ok('d(ΔS_neg)/dz = 0 at z_c (critical point)');
  } else {
    results.fail('Critical point', `expected 0, got ${dsAtLens}`);
  }

  // Positive below z_c
  const dsBelow = Delta.computeDeltaSNegDerivative(Z_CRITICAL - 0.1);
  if (dsBelow > 0) results.ok('d(ΔS_neg)/dz > 0 below z_c');
  else results.fail('Sign below z_c', `expected positive, got ${dsBelow}`);

  // Negative above z_c
  const dsAbove = Delta.computeDeltaSNegDerivative(Z_CRITICAL + 0.1);
  if (dsAbove < 0) results.ok('d(ΔS_neg)/dz < 0 above z_c');
  else results.fail('Sign above z_c', `expected negative, got ${dsAbove}`);
}

function testDeltaSNegSigned(results) {
  section('Signed ΔS_neg');

  const Z_CRITICAL = Delta.Z_CRITICAL;

  // Zero at z_c
  const signedLens = Delta.computeDeltaSNegSigned(Z_CRITICAL);
  if (assertClose(signedLens, 0.0, 1e-6)) {
    results.ok('Signed ΔS_neg = 0 at z_c');
  } else {
    results.fail('Zero at lens', `expected 0, got ${signedLens}`);
  }

  // Positive above z_c
  const signedAbove = Delta.computeDeltaSNegSigned(Z_CRITICAL + 0.05);
  if (signedAbove > 0) results.ok('Signed ΔS_neg > 0 above z_c');
  else results.fail('Sign above', `expected positive, got ${signedAbove}`);

  // Negative below z_c
  const signedBelow = Delta.computeDeltaSNegSigned(Z_CRITICAL - 0.05);
  if (signedBelow < 0) results.ok('Signed ΔS_neg < 0 below z_c');
  else results.fail('Sign below', `expected negative, got ${signedBelow}`);
}

// ============================================================================
// HEX-PRISM GEOMETRY TESTS
// ============================================================================

function testHexPrismGeometry(results) {
  section('Hex-Prism Geometry');

  const Z_CRITICAL = Delta.Z_CRITICAL;

  const geomLens = Delta.computeHexPrismGeometry(Z_CRITICAL);
  const expectedR = 0.85 - 0.25 * 1.0;
  if (assertClose(geomLens.radius, expectedR, 1e-6)) {
    results.ok(`R(z_c) = ${expectedR} (radius contracts at lens)`);
  } else {
    results.fail('Radius at lens', `expected ${expectedR}, got ${geomLens.radius}`);
  }

  const expectedH = 0.12 + 0.18 * 1.0;
  if (assertClose(geomLens.height, expectedH, 1e-6)) {
    results.ok(`H(z_c) = ${expectedH} (height elongates at lens)`);
  } else {
    results.fail('Height at lens', `expected ${expectedH}, got ${geomLens.height}`);
  }

  const geomFar = Delta.computeHexPrismGeometry(0.5);
  if (geomFar.radius > geomLens.radius) results.ok('Radius larger far from lens');
  else results.fail('Radius comparison', 'far should be larger');

  if (geomFar.height < geomLens.height) results.ok('Height smaller far from lens');
  else results.fail('Height comparison', 'far should be smaller');
}

// ============================================================================
// K-FORMATION TESTS
// ============================================================================

function testKFormation(results) {
  section('K-Formation Gating');

  const Z_CRITICAL = Delta.Z_CRITICAL;
  const PHI_INV = Delta.PHI_INV;

  // At z_c
  const kLens = Delta.checkKFormation(Z_CRITICAL);
  if (kLens.formed) results.ok('K-formation at z_c (η=1 > φ⁻¹)');
  else results.fail('K-formation at lens', `η=${kLens.eta}, threshold=${kLens.threshold}`);

  // Far from lens
  const kFar = Delta.checkKFormation(0.5);
  if (!kFar.formed) results.ok('No K-formation far from lens');
  else results.fail('K-formation at z=0.5', `unexpected formation, η=${kFar.eta}`);

  // Check threshold value
  if (assertClose(kLens.threshold, PHI_INV, 1e-6)) {
    results.ok(`Threshold = φ⁻¹ ≈ ${PHI_INV.toFixed(6)}`);
  } else {
    results.fail('Threshold', `expected ${PHI_INV}, got ${kLens.threshold}`);
  }
}

// ============================================================================
// Π-REGIME BLENDING TESTS
// ============================================================================

function testPiBlending(results) {
  section('Π-Regime Blending');

  const Z_CRITICAL = Delta.Z_CRITICAL;

  // Below z_c
  const blendBelow = Delta.computePiBlendWeights(Z_CRITICAL - 0.1);
  if (blendBelow.wPi === 0 && blendBelow.wLocal === 1) {
    results.ok('Below z_c: w_π=0, w_local=1 (pure local)');
  } else {
    results.fail('Below z_c', `w_π=${blendBelow.wPi}`);
  }

  if (!blendBelow.inPiRegime) results.ok('Below z_c: not in Π-regime');
  else results.fail('Π-regime flag below z_c', 'should be False');

  // At z_c
  const blendLens = Delta.computePiBlendWeights(Z_CRITICAL);
  if (assertClose(blendLens.wPi, 1.0, 1e-6)) {
    results.ok('At z_c: w_π = 1.0');
  } else {
    results.fail('At z_c', `w_π=${blendLens.wPi}`);
  }

  if (blendLens.inPiRegime) results.ok('At z_c: in Π-regime');
  else results.fail('Π-regime flag at z_c', 'should be True');

  // Above z_c
  const blendAbove = Delta.computePiBlendWeights(Z_CRITICAL + 0.05);
  if (blendAbove.wPi > 0 && blendAbove.wPi < 1) {
    results.ok(`Above z_c: 0 < w_π < 1 (w_π=${blendAbove.wPi.toFixed(4)})`);
  } else {
    results.fail('Above z_c', `w_π=${blendAbove.wPi} not in (0,1)`);
  }
}

// ============================================================================
// GATE MODULATION TESTS
// ============================================================================

function testGateModulation(results) {
  section('Gate Modulation');

  const Z_CRITICAL = Delta.Z_CRITICAL;

  const modFar = Delta.computeGateModulation(0.5);
  const modLens = Delta.computeGateModulation(Z_CRITICAL);

  if (modLens.coherentCoupling > modFar.coherentCoupling) {
    results.ok('Coherent coupling increases at lens');
  } else {
    results.fail('Coherent coupling',
      `lens=${modLens.coherentCoupling}, far=${modFar.coherentCoupling}`);
  }

  if (modLens.decoherenceRate < modFar.decoherenceRate) {
    results.ok('Decoherence rate decreases at lens');
  } else {
    results.fail('Decoherence rate',
      `lens=${modLens.decoherenceRate}, far=${modFar.decoherenceRate}`);
  }

  if (modLens.entropyTarget < modFar.entropyTarget) {
    results.ok('Entropy target decreases at lens');
  } else {
    results.fail('Entropy target',
      `lens=${modLens.entropyTarget}, far=${modFar.entropyTarget}`);
  }
}

// ============================================================================
// COHERENCE SYNTHESIS TESTS
// ============================================================================

function testCoherenceSynthesis(results) {
  section('Coherence Synthesis Heuristics');

  const operators = ['()', '×', '^', '÷', '+', '−'];
  const zBelow = 0.5;

  // Maximize coherence below lens
  const scoresMax = {};
  for (const op of operators) {
    scoresMax[op] = Delta.scoreOperatorForCoherence(op, zBelow, Delta.CoherenceObjective.MAXIMIZE);
  }

  const constructive = ['^', '+', '×'];
  const dissipative = ['÷', '−'];

  const avgConstructive = constructive.reduce((a, op) => a + scoresMax[op], 0) / 3;
  const avgDissipative = dissipative.reduce((a, op) => a + scoresMax[op], 0) / 2;

  if (avgConstructive > avgDissipative) {
    results.ok('MAXIMIZE below lens favors constructive operators');
  } else {
    results.fail('MAXIMIZE heuristic',
      `constructive=${avgConstructive}, dissipative=${avgDissipative}`);
  }

  // Minimize coherence
  const scoresMin = {};
  for (const op of operators) {
    scoresMin[op] = Delta.scoreOperatorForCoherence(op, zBelow, Delta.CoherenceObjective.MINIMIZE);
  }

  const avgConstructiveMin = constructive.reduce((a, op) => a + scoresMin[op], 0) / 3;
  const avgDissipativeMin = dissipative.reduce((a, op) => a + scoresMin[op], 0) / 2;

  if (avgDissipativeMin > avgConstructiveMin) {
    results.ok('MINIMIZE favors dissipative operators');
  } else {
    results.fail('MINIMIZE heuristic',
      `constructive=${avgConstructiveMin}, dissipative=${avgDissipativeMin}`);
  }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

function testFullStateIntegration(results) {
  section('Full State Integration');

  const Z_CRITICAL = Delta.Z_CRITICAL;
  const state = Delta.computeFullState(Z_CRITICAL);

  if (state.z === Z_CRITICAL) results.ok('z-coordinate preserved');
  else results.fail('z-coordinate', `expected ${Z_CRITICAL}, got ${state.z}`);

  if (assertClose(state.deltaSNeg, 1.0, 1e-6)) {
    results.ok('ΔS_neg computed correctly');
  } else {
    results.fail('ΔS_neg', `expected 1.0, got ${state.deltaSNeg}`);
  }

  if (state.geometry) results.ok('Geometry computed');
  else results.fail('Geometry', 'missing');

  if (state.gateModulation) results.ok('Gate modulation computed');
  else results.fail('Gate modulation', 'missing');

  if (state.piBlend.inPiRegime) results.ok('Π-blend computed and correct');
  else results.fail('Π-blend', 'should be in Π-regime at z_c');

  if (state.kFormation.formed) results.ok('K-formation computed and correct');
  else results.fail('K-formation', 'should be formed at z_c');
}

function testS3TruthOrbit(results) {
  section('S₃ Truth Orbit');

  const initial = { TRUE: 0.7, PARADOX: 0.2, UNTRUE: 0.1 };

  // Apply × (σ) 3 times should return to initial
  let current = { ...initial };
  for (let i = 0; i < 3; i++) {
    current = S3.applyOperatorToTruth(current, '×');
  }

  if (assertClose(current.TRUE, initial.TRUE, 1e-10) &&
      assertClose(current.PARADOX, initial.PARADOX, 1e-10) &&
      assertClose(current.UNTRUE, initial.UNTRUE, 1e-10)) {
    results.ok('3-cycle returns to initial (σ³ = e)');
  } else {
    results.fail('3-cycle orbit', `expected ${JSON.stringify(initial)}, got ${JSON.stringify(current)}`);
  }

  // Single transposition is self-inverse
  const dist1 = S3.applyOperatorToTruth(initial, '÷');
  const dist2 = S3.applyOperatorToTruth(dist1, '÷');

  if (assertClose(dist2.TRUE, initial.TRUE, 1e-10) &&
      assertClose(dist2.PARADOX, initial.PARADOX, 1e-10) &&
      assertClose(dist2.UNTRUE, initial.UNTRUE, 1e-10)) {
    results.ok('Transposition is self-inverse (τ² = e)');
  } else {
    results.fail('Transposition inverse', `expected ${JSON.stringify(initial)}, got ${JSON.stringify(dist2)}`);
  }
}

// ============================================================================
// CROSS-LANGUAGE PARITY TESTS
// ============================================================================

function testCrossLanguageParity(results) {
  section('Cross-Language Parity');

  // These values should match the Python implementation
  const Z_CRITICAL = Math.sqrt(3) / 2;
  const testZ = [0.3, 0.5, 0.7, Z_CRITICAL, 0.9];

  // Check ΔS_neg values match expected
  for (const z of testZ) {
    const s = Delta.computeDeltaSNeg(z);
    const expected = Math.exp(-36.0 * Math.pow(z - Z_CRITICAL, 2));
    if (assertClose(s, expected, 1e-10)) {
      results.ok(`ΔS_neg(${z.toFixed(2)}) matches formula`);
    } else {
      results.fail(`ΔS_neg(${z.toFixed(2)})`, `expected ${expected}, got ${s}`);
    }
  }

  // Check rotation index
  if (S3.rotationIndexFromZ(0.1) === 0 &&
      S3.rotationIndexFromZ(0.5) === 1 &&
      S3.rotationIndexFromZ(0.8) === 2) {
    results.ok('Rotation index z-mapping correct');
  } else {
    results.fail('Rotation index', 'mapping incorrect');
  }
}

// ============================================================================
// MAIN
// ============================================================================

function main() {
  console.log('='.repeat(60));
  console.log('S₃ SYMMETRY AND ΔS_neg EXTENSION TESTS (JavaScript)');
  console.log('='.repeat(60));

  const results = new TestResults();

  // S₃ tests
  testS3GroupAxioms(results);
  testS3OperatorMapping(results);
  testS3Parity(results);
  testS3Permutation(results);
  testOperatorWindowRotation(results);
  testS3TruthOrbit(results);

  // ΔS_neg tests
  testDeltaSNegBasic(results);
  testDeltaSNegDerivative(results);
  testDeltaSNegSigned(results);

  // Geometry tests
  testHexPrismGeometry(results);

  // Gating tests
  testKFormation(results);
  testPiBlending(results);
  testGateModulation(results);

  // Synthesis tests
  testCoherenceSynthesis(results);

  // Integration tests
  testFullStateIntegration(results);

  // Cross-language parity
  testCrossLanguageParity(results);

  // Summary
  const success = results.summary();
  process.exit(success ? 0 : 1);
}

main();
