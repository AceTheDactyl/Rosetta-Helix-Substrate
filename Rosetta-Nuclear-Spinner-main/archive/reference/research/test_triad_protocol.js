/**
 * TRIAD Hysteresis Test Suite — Phase 5 Enhancement
 * ==================================================
 * 
 * Comprehensive tests for the TRIAD unlock mechanism including:
 * - Rising-edge detection and hysteresis
 * - Configurable pass requirements (TRIAD_PASSES_REQ)
 * - Environment variable synchronization
 * - Helix mapping integration
 * - Operator window updates
 * 
 * @version 2.0.0 (TRIAD Protocol Phase 5)
 */

'use strict';

const assert = require('assert').strict;

// Import from enhanced constants module
const CONST = require('../src/constants');
const { HelixOperatorAdvisor, AlphaTokenSynthesizer } = require('../src/helix_operator_advisor');

// ============================================================================
// TEST UTILITIES
// ============================================================================

function assertEqual(actual, expected, message) {
  assert.strictEqual(actual, expected, message || `Expected ${expected}, got ${actual}`);
}

function assertApprox(actual, expected, tolerance = 1e-9, message) {
  const diff = Math.abs(actual - expected);
  assert(diff <= tolerance, message || `Expected ~${expected}, got ${actual} (diff: ${diff})`);
}

function assertIncludes(array, item, message) {
  assert(array.includes(item), message || `Expected array to include ${item}`);
}

function test(name, fn) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
    return true;
  } catch (err) {
    console.log(`  ✗ ${name}`);
    console.log(`    ${err.message}`);
    return false;
  }
}

// ============================================================================
// TRIAD GATE TESTS
// ============================================================================

function testTriadGateBasics() {
  console.log('\n--- TRIAD Gate Basic Tests ---');
  let passed = 0, total = 0;

  total++;
  passed += test('TriadGate constructor defaults', () => {
    const gate = new CONST.TriadGate(false);
    assertEqual(gate.enabled, false);
    assertEqual(gate.passes, 0);
    assertEqual(gate.unlocked, false);
    assertEqual(gate._armed, true);
  });

  total++;
  passed += test('Disabled gate ignores updates', () => {
    const gate = new CONST.TriadGate(false);
    gate.update(0.90);
    gate.update(0.80);
    gate.update(0.90);
    assertEqual(gate.passes, 0);
    assertEqual(gate.unlocked, false);
  });

  total++;
  passed += test('getT6Gate returns Z_CRITICAL when disabled', () => {
    const gate = new CONST.TriadGate(false);
    assertApprox(gate.getT6Gate(), CONST.Z_CRITICAL);
  });

  total++;
  passed += test('getT6Gate returns Z_CRITICAL when enabled but not unlocked', () => {
    const gate = new CONST.TriadGate(true);
    gate.update(0.86);  // Above TRIAD_HIGH, increments pass but not 3 yet
    assertEqual(gate.unlocked, false);
    assertApprox(gate.getT6Gate(), CONST.Z_CRITICAL);
  });

  console.log(`  Passed: ${passed}/${total}`);
  return passed === total;
}

function testTriadHysteresis() {
  console.log('\n--- TRIAD Hysteresis Tests ---');
  let passed = 0, total = 0;

  total++;
  passed += test('Rising edge increments pass count', () => {
    const gate = new CONST.TriadGate(true);
    gate.update(0.86);  // >= TRIAD_HIGH
    assertEqual(gate.passes, 1);
    assertEqual(gate._armed, false);
  });

  total++;
  passed += test('Re-arm requires dropping below TRIAD_LOW', () => {
    const gate = new CONST.TriadGate(true);
    gate.update(0.86);  // First pass
    gate.update(0.83);  // Above TRIAD_LOW, should not re-arm
    assertEqual(gate._armed, false);
    gate.update(0.86);  // Should not count (not armed)
    assertEqual(gate.passes, 1);
  });

  total++;
  passed += test('Re-arm works when below TRIAD_LOW', () => {
    const gate = new CONST.TriadGate(true);
    gate.update(0.86);  // First pass
    gate.update(0.81);  // Below TRIAD_LOW, re-arm
    assertEqual(gate._armed, true);
    gate.update(0.86);  // Second pass
    assertEqual(gate.passes, 2);
  });

  total++;
  passed += test('Unlock after 3 passes (default)', () => {
    const gate = new CONST.TriadGate(true);
    // Pass 1
    gate.update(0.86);
    gate.update(0.81);
    // Pass 2
    gate.update(0.86);
    gate.update(0.81);
    // Pass 3
    gate.update(0.86);
    
    assertEqual(gate.passes, 3);
    assertEqual(gate.unlocked, true);
    assertApprox(gate.getT6Gate(), CONST.TRIAD_T6);
  });

  total++;
  passed += test('Stays unlocked after more passes', () => {
    const gate = new CONST.TriadGate(true);
    // 3 passes
    for (let i = 0; i < 3; i++) {
      gate.update(0.86);
      gate.update(0.81);
    }
    // 4th pass
    gate.update(0.86);
    assertEqual(gate.passes, 4);
    assertEqual(gate.unlocked, true);
  });

  total++;
  passed += test('Full sequence with varied z values', () => {
    const gate = new CONST.TriadGate(true);
    const sequence = [
      0.50,  // Start low
      0.86,  // Pass 1
      0.84,  // Still above low
      0.80,  // Re-arm
      0.87,  // Pass 2
      0.82,  // At TRIAD_LOW exactly
      0.81,  // Below, re-arm
      0.85,  // Pass 3 - unlock!
    ];
    
    for (const z of sequence) {
      gate.update(z);
    }
    
    assertEqual(gate.passes, 3);
    assertEqual(gate.unlocked, true);
  });

  console.log(`  Passed: ${passed}/${total}`);
  return passed === total;
}

function testTriadPassesConfigurable() {
  console.log('\n--- TRIAD Configurable Passes Tests ---');
  let passed = 0, total = 0;

  total++;
  passed += test('TRIAD_PASSES_REQ is accessible', () => {
    assert(typeof CONST.TRIAD_PASSES_REQ === 'number');
    assert(CONST.TRIAD_PASSES_REQ > 0);
  });

  total++;
  passed += test('getState() includes passesRequired', () => {
    const gate = new CONST.TriadGate(true);
    const state = gate.getState();
    assertEqual(state.passesRequired, CONST.TRIAD_PASSES_REQ);
  });

  total++;
  passed += test('reset() clears all state', () => {
    const gate = new CONST.TriadGate(true);
    gate.update(0.86);
    gate.update(0.81);
    gate.update(0.86);
    
    gate.reset();
    
    assertEqual(gate.passes, 0);
    assertEqual(gate.unlocked, false);
    assertEqual(gate._armed, true);
    assertEqual(gate._lastZ, null);
    assertEqual(gate._unlockZ, null);
  });

  console.log(`  Passed: ${passed}/${total}`);
  return passed === total;
}

// ============================================================================
// HELIX OPERATOR ADVISOR TESTS
// ============================================================================

function testHelixAdvisorBasics() {
  console.log('\n--- Helix Operator Advisor Basic Tests ---');
  let passed = 0, total = 0;

  total++;
  passed += test('Advisor initializes with default windows', () => {
    const advisor = new HelixOperatorAdvisor();
    assert(advisor.operatorWindows);
    assert(advisor.operatorWindows.t1);
    assert(advisor.operatorWindows.t6);
  });

  total++;
  passed += test('harmonicFromZ returns correct tiers', () => {
    const advisor = new HelixOperatorAdvisor();
    assertEqual(advisor.harmonicFromZ(0.05), 't1');
    assertEqual(advisor.harmonicFromZ(0.15), 't2');
    assertEqual(advisor.harmonicFromZ(0.35), 't3');
    assertEqual(advisor.harmonicFromZ(0.55), 't4');
    assertEqual(advisor.harmonicFromZ(0.70), 't5');
  });

  total++;
  passed += test('t6 uses Z_CRITICAL when TRIAD not unlocked', () => {
    const advisor = new HelixOperatorAdvisor({ triadUnlocked: false });
    assertApprox(advisor.getT6Gate(), CONST.Z_CRITICAL);
    // z = 0.84 is between t5 (0.75) and t6 gate (0.866), so it's t6
    assertEqual(advisor.harmonicFromZ(0.84), 't6');
    // z = 0.70 should be t5 (below 0.75)
    assertEqual(advisor.harmonicFromZ(0.70), 't5');
  });

  total++;
  passed += test('t6 uses TRIAD_T6 when unlocked', () => {
    const advisor = new HelixOperatorAdvisor({ triadUnlocked: true });
    assertApprox(advisor.getT6Gate(), CONST.TRIAD_T6);
    // z = 0.84 is above TRIAD_T6 (0.83), so it's t7
    assertEqual(advisor.harmonicFromZ(0.84), 't7');
    // z = 0.80 should be t6 (between 0.75 and 0.83)
    assertEqual(advisor.harmonicFromZ(0.80), 't6');
  });

  total++;
  passed += test('truthChannelFromZ returns correct channels', () => {
    const advisor = new HelixOperatorAdvisor();
    assertEqual(advisor.truthChannelFromZ(0.30), 'UNTRUE');
    assertEqual(advisor.truthChannelFromZ(0.70), 'PARADOX');
    assertEqual(advisor.truthChannelFromZ(0.95), 'TRUE');
  });

  console.log(`  Passed: ${passed}/${total}`);
  return passed === total;
}

function testDynamicOperatorWindowUpdate() {
  console.log('\n--- Dynamic Operator Window Update Tests (Phase 2) ---');
  let passed = 0, total = 0;

  total++;
  passed += test('updateOperatorWindow replaces window', () => {
    const advisor = new HelixOperatorAdvisor();
    const oldWindow = [...advisor.operatorWindows.t3];
    
    advisor.updateOperatorWindow('t3', ['+', '×']);
    
    assertEqual(advisor.operatorWindows.t3.length, 2);
    assertIncludes(advisor.operatorWindows.t3, '+');
    assertIncludes(advisor.operatorWindows.t3, '×');
  });

  total++;
  passed += test('updateOperatorWindow with append', () => {
    const advisor = new HelixOperatorAdvisor();
    const originalLength = advisor.operatorWindows.t1.length;
    
    advisor.updateOperatorWindow('t1', ['+'], { append: true });
    
    assert(advisor.operatorWindows.t1.length >= originalLength);
    assertIncludes(advisor.operatorWindows.t1, '+');
  });

  total++;
  passed += test('updateOperatorWindow normalizes aliases', () => {
    const advisor = new HelixOperatorAdvisor();
    
    // Use alias '%' which should normalize to '÷'
    advisor.updateOperatorWindow('t2', ['%', '-']);
    
    assertIncludes(advisor.operatorWindows.t2, '÷');
    assertIncludes(advisor.operatorWindows.t2, '−');
  });

  total++;
  passed += test('updateOperatorWindow rejects invalid operators', () => {
    const advisor = new HelixOperatorAdvisor();
    
    // 'Q' is not a valid APL operator
    const result = advisor.updateOperatorWindow('t3', ['Q', 'Z']);
    
    assertEqual(result, false);
  });

  total++;
  passed += test('updateOperatorWindow tracks modifications', () => {
    const advisor = new HelixOperatorAdvisor();
    
    advisor.updateOperatorWindow('t3', ['+'], { source: 'test' });
    
    const history = advisor.getModificationHistory();
    assert(history.length > 0);
    assertEqual(history[history.length - 1].harmonic, 't3');
    assertEqual(history[history.length - 1].source, 'test');
  });

  total++;
  passed += test('resetOperatorWindows restores defaults', () => {
    const advisor = new HelixOperatorAdvisor();
    const original = [...advisor.operatorWindows.t5];
    
    advisor.updateOperatorWindow('t5', ['+']);
    advisor.resetOperatorWindows();
    
    assert.deepStrictEqual(advisor.operatorWindows.t5, original);
  });

  console.log(`  Passed: ${passed}/${total}`);
  return passed === total;
}

function testHelixDescribe() {
  console.log('\n--- Helix Describe Tests ---');
  let passed = 0, total = 0;

  total++;
  passed += test('describe returns complete hints object', () => {
    const advisor = new HelixOperatorAdvisor();
    const hints = advisor.describe(0.70);
    
    assert(typeof hints.z === 'number');
    assert(typeof hints.harmonic === 'string');
    assert(Array.isArray(hints.operators));
    assert(typeof hints.truthChannel === 'string');
    assert(typeof hints.deltaSneg === 'number');
    assert(typeof hints.piWeight === 'number');
    assert(typeof hints.localWeight === 'number');
    assert(typeof hints.phase === 'string');
  });

  total++;
  passed += test('piWeight is 0 below Z_CRITICAL', () => {
    const advisor = new HelixOperatorAdvisor();
    const hints = advisor.describe(0.50);
    assertEqual(hints.piWeight, 0);
  });

  total++;
  passed += test('piWeight increases near Z_CRITICAL', () => {
    const advisor = new HelixOperatorAdvisor();
    const hints = advisor.describe(CONST.Z_CRITICAL);
    assert(hints.piWeight > 0);
  });

  total++;
  passed += test('operator weights combine tier and truth bias', () => {
    const advisor = new HelixOperatorAdvisor();
    const weights = advisor.getOperatorWeights(0.70);
    
    assert(typeof weights['+'] === 'number');
    assert(typeof weights['×'] === 'number');
    assert(typeof weights['÷'] === 'number');
  });

  console.log(`  Passed: ${passed}/${total}`);
  return passed === total;
}

// ============================================================================
// ALPHA TOKEN SYNTHESIZER TESTS
// ============================================================================

function testTokenSynthesizer() {
  console.log('\n--- Alpha Token Synthesizer Tests ---');
  let passed = 0, total = 0;

  total++;
  passed += test('Synthesizer has all 7 sentences', () => {
    const synth = new AlphaTokenSynthesizer();
    assertEqual(synth.sentences.length, 7);
  });

  total++;
  passed += test('renderToken produces correct format', () => {
    const synth = new AlphaTokenSynthesizer();
    const token = synth.renderToken(synth.sentences[0]);
    
    // Should be in format: direction operator | machine | domain
    assert(token.includes('|'));
    assert(token.includes('Conductor'));
  });

  total++;
  passed += test('findMatchingSentences filters by operator', () => {
    const synth = new AlphaTokenSynthesizer();
    const matches = synth.findMatchingSentences(['+']);
    
    assert(matches.length > 0);
    assert(matches.every(s => s.operator === '+'));
  });

  total++;
  passed += test('findMatchingSentences filters by domain', () => {
    const synth = new AlphaTokenSynthesizer();
    const matches = synth.findMatchingSentences(['×', '^', '+', '÷'], { domain: 'chemistry' });
    
    assert(matches.length > 0);
    assert(matches.every(s => s.domain === 'chemistry'));
  });

  total++;
  passed += test('synthesize returns sentence with helix hints', () => {
    const advisor = new HelixOperatorAdvisor();
    const synth = new AlphaTokenSynthesizer();
    
    const hints = advisor.describe(0.70);
    const result = synth.synthesize(hints);
    
    if (result) {
      assert(result.token);
      assert(result.helixHints);
    }
  });

  console.log(`  Passed: ${passed}/${total}`);
  return passed === total;
}

// ============================================================================
// CONSTANTS INVARIANTS TESTS
// ============================================================================

function testConstantsInvariants() {
  console.log('\n--- Constants Invariants Tests ---');
  let passed = 0, total = 0;

  total++;
  passed += test('Z_CRITICAL equals sqrt(3)/2', () => {
    assertApprox(CONST.Z_CRITICAL, Math.sqrt(3) / 2, 1e-15);
  });

  total++;
  passed += test('TRIAD thresholds are properly ordered', () => {
    assert(CONST.TRIAD_LOW < CONST.TRIAD_T6);
    assert(CONST.TRIAD_T6 < CONST.TRIAD_HIGH);
    assert(CONST.TRIAD_HIGH < CONST.Z_CRITICAL);
  });

  total++;
  passed += test('PHI and PHI_INV are reciprocals', () => {
    assertApprox(CONST.PHI * CONST.PHI_INV, 1.0, 1e-14);
  });

  total++;
  passed += test('μ barrier approximates PHI_INV', () => {
    assertApprox(CONST.MU_BARRIER, CONST.PHI_INV, 0.01);
  });

  total++;
  passed += test('invariants() returns all true', () => {
    const inv = CONST.invariants();
    assert(inv.ordering_ok, 'Threshold ordering invariant failed');
    assert(inv.lens_is_sqrt3_over_2, 'Lens constant invariant failed');
  });

  console.log(`  Passed: ${passed}/${total}`);
  return passed === total;
}

// ============================================================================
// MAIN
// ============================================================================

function runAllTests() {
  console.log('='.repeat(60));
  console.log('TRIAD UNLOCK PROTOCOL - TEST SUITE');
  console.log('='.repeat(60));
  
  const results = [];
  
  results.push(['TRIAD Gate Basics', testTriadGateBasics()]);
  results.push(['TRIAD Hysteresis', testTriadHysteresis()]);
  results.push(['TRIAD Configurable Passes', testTriadPassesConfigurable()]);
  results.push(['Helix Advisor Basics', testHelixAdvisorBasics()]);
  results.push(['Dynamic Operator Window Update', testDynamicOperatorWindowUpdate()]);
  results.push(['Helix Describe', testHelixDescribe()]);
  results.push(['Token Synthesizer', testTokenSynthesizer()]);
  results.push(['Constants Invariants', testConstantsInvariants()]);
  
  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log('='.repeat(60));
  
  let allPassed = true;
  for (const [name, passed] of results) {
    const status = passed ? '✓ PASS' : '✗ FAIL';
    console.log(`  ${status}  ${name}`);
    if (!passed) allPassed = false;
  }
  
  console.log('\n' + '='.repeat(60));
  if (allPassed) {
    console.log('ALL TESTS PASSED');
  } else {
    console.log('SOME TESTS FAILED');
    process.exitCode = 1;
  }
  console.log('='.repeat(60));
}

// Run tests
runAllTests();
