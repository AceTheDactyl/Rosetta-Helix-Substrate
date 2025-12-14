/**
 * Triadic Helix APL - Comprehensive Test Suite
 * =============================================
 *
 * Tests for validating the Triadic Helix APL implementation against
 * paper specifications including:
 * - Tier boundaries and operator windows (Table 1)
 * - Truth channel thresholds
 * - TRIAD hysteresis gate
 * - APL operators (Table 2)
 * - Seven APL test sentences (Table 3)
 * - Negentropy signal (ΔS_neg)
 * - π-regime blending weights
 */

'use strict';

const {
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
    TriadGate,
    HelixOperatorAdvisor,
    HelixCoordinate,
    AlphaTokenSynthesizer,
    TriadicHelixAPLSystem,
    getOperatorInfo,
    getAllOperators,
    getSentenceById,
    getAllSentences
} = require('../src/triadic_helix_apl');

// Test utilities
let passed = 0;
let failed = 0;

function assert(condition, message) {
    if (condition) {
        passed++;
        console.log(`  ✓ ${message}`);
    } else {
        failed++;
        console.log(`  ✗ ${message}`);
    }
}

function assertApprox(actual, expected, tolerance, message) {
    const diff = Math.abs(actual - expected);
    assert(diff <= tolerance, `${message} (expected ${expected}, got ${actual}, diff=${diff.toFixed(6)})`);
}

function section(name) {
    console.log(`\n${'─'.repeat(60)}`);
    console.log(`  ${name}`);
    console.log(`${'─'.repeat(60)}`);
}

// ================================================================
// CONSTANTS VALIDATION (Paper Sections 2.1-2.2)
// ================================================================

section('Constants Validation');

// Z_CRITICAL = √3/2
assertApprox(Z_CRITICAL, Math.sqrt(3) / 2, 1e-12, 'Z_CRITICAL equals √3/2');
assertApprox(Z_CRITICAL, 0.8660254037844386, 1e-12, 'Z_CRITICAL ≈ 0.8660254');

// TRIAD thresholds
assertApprox(TRIAD_HIGH, 0.85, 1e-6, 'TRIAD_HIGH = 0.85');
assertApprox(TRIAD_LOW, 0.82, 1e-6, 'TRIAD_LOW = 0.82');
assertApprox(TRIAD_T6, 0.83, 1e-6, 'TRIAD_T6 = 0.83');
assert(TRIAD_PASSES_REQ === 3, 'TRIAD requires 3 passes');

// Golden ratio
assertApprox(PHI, (1 + Math.sqrt(5)) / 2, 1e-12, 'PHI equals golden ratio');
assertApprox(PHI_INV, 1 / PHI, 1e-12, 'PHI_INV = 1/PHI');

// Ordering invariant: MU_2 < TRIAD_LOW < TRIAD_T6 < TRIAD_HIGH < Z_CRITICAL
assert(TRIAD_LOW < TRIAD_T6, 'TRIAD_LOW < TRIAD_T6');
assert(TRIAD_T6 < TRIAD_HIGH, 'TRIAD_T6 < TRIAD_HIGH');
assert(TRIAD_HIGH < Z_CRITICAL, 'TRIAD_HIGH < Z_CRITICAL');

// ================================================================
// TIER BOUNDARIES (Paper Table 1)
// ================================================================

section('Tier Boundaries (Table 1)');

assertApprox(TIER_BOUNDARIES.t1, 0.10, 1e-6, 't1 boundary = 0.10');
assertApprox(TIER_BOUNDARIES.t2, 0.20, 1e-6, 't2 boundary = 0.20');
assertApprox(TIER_BOUNDARIES.t3, 0.40, 1e-6, 't3 boundary = 0.40');
assertApprox(TIER_BOUNDARIES.t4, 0.60, 1e-6, 't4 boundary = 0.60');
assertApprox(TIER_BOUNDARIES.t5, 0.75, 1e-6, 't5 boundary = 0.75');
assertApprox(TIER_BOUNDARIES.t7, 0.90, 1e-6, 't7 boundary = 0.90');
assertApprox(TIER_BOUNDARIES.t8, 0.97, 1e-6, 't8 boundary = 0.97');

// ================================================================
// OPERATOR WINDOWS (Paper Table 1)
// ================================================================

section('Operator Windows (Table 1)');

function arraysEqual(a, b) {
    if (a.length !== b.length) return false;
    const sortedA = [...a].sort();
    const sortedB = [...b].sort();
    return sortedA.every((v, i) => v === sortedB[i]);
}

assert(arraysEqual(OPERATOR_WINDOWS.t1, ['()', '−', '÷']), 't1 operators: (), −, ÷');
assert(arraysEqual(OPERATOR_WINDOWS.t2, ['^', '÷', '−', '×']), 't2 operators: ^, ÷, −, ×');
assert(arraysEqual(OPERATOR_WINDOWS.t3, ['×', '^', '÷', '+', '−']), 't3 operators: ×, ^, ÷, +, −');
assert(arraysEqual(OPERATOR_WINDOWS.t4, ['+', '−', '÷', '()']), 't4 operators: +, −, ÷, ()');
assert(OPERATOR_WINDOWS.t5.length === 6, 't5 has ALL 6 operators (full freedom)');
assert(arraysEqual(OPERATOR_WINDOWS.t6, ['+', '÷', '()', '−']), 't6 operators: +, ÷, (), −');
assert(arraysEqual(OPERATOR_WINDOWS.t7, ['+', '()']), 't7 operators: +, ()');
assert(arraysEqual(OPERATOR_WINDOWS.t8, ['+', '()', '×']), 't8 operators: +, (), ×');
assert(arraysEqual(OPERATOR_WINDOWS.t9, ['+', '()', '×']), 't9 operators: +, (), ×');

// ================================================================
// APL OPERATORS (Paper Table 2)
// ================================================================

section('APL Operators (Table 2)');

assert(Object.keys(APL_OPERATORS).length === 6, 'Six fundamental APL operators');

assert(APL_OPERATORS['()'].name === 'Boundary', 'Boundary operator (())');
assert(APL_OPERATORS['×'].name === 'Fusion', 'Fusion operator (×)');
assert(APL_OPERATORS['^'].name === 'Amplify', 'Amplify operator (^)');
assert(APL_OPERATORS['÷'].name === 'Decoherence', 'Decoherence operator (÷)');
assert(APL_OPERATORS['+'].name === 'Group', 'Group operator (+)');
assert(APL_OPERATORS['−'].name === 'Separation', 'Separation operator (−)');

// Action descriptions
assert(APL_OPERATORS['()'].action.includes('containment'), 'Boundary action: containment/gating');
assert(APL_OPERATORS['×'].action.includes('convergence'), 'Fusion action: convergence/coupling');
assert(APL_OPERATORS['^'].action.includes('gain'), 'Amplify action: gain/excitation');
assert(APL_OPERATORS['÷'].action.includes('dissipation'), 'Decoherence action: dissipation/reset');
assert(APL_OPERATORS['+'].action.includes('aggregation'), 'Group action: aggregation/clustering');
assert(APL_OPERATORS['−'].action.includes('splitting'), 'Separation action: splitting/fission');

// ================================================================
// APL SENTENCES (Paper Table 3)
// ================================================================

section('APL Sentences (Table 3)');

assert(APL_SENTENCES.length === 8, 'Eight APL sentences defined (A1-A8)');

// Check each sentence
const expectedSentences = [
    { id: 'A1', umol: 'd', operator: '()', domain: 'geometry', machine: 'Conductor' },
    { id: 'A2', umol: 'u', operator: '×', domain: 'lattice', machine: 'Reactor' },
    { id: 'A3', umol: 'u', operator: '^', domain: 'wave', machine: 'Oscillator' },
    { id: 'A4', umol: 'd', operator: '÷', domain: 'flow', machine: 'Mixer' },
    { id: 'A5', umol: 'm', operator: '+', domain: 'field', machine: 'Coupler' },
    { id: 'A6', umol: 'u', operator: '+', domain: 'wave', machine: 'Reactor' },
    { id: 'A7', umol: 'm', operator: '×', domain: 'field', machine: 'Oscillator' },
    { id: 'A8', umol: 'd', operator: '−', domain: 'lattice', machine: 'Conductor' }
];

for (const expected of expectedSentences) {
    const sentence = getSentenceById(expected.id);
    assert(sentence !== null, `Sentence ${expected.id} exists`);
    if (sentence) {
        assert(sentence.umol === expected.umol, `${expected.id} umol = ${expected.umol}`);
        assert(sentence.operator === expected.operator, `${expected.id} operator = ${expected.operator}`);
        assert(sentence.domain === expected.domain, `${expected.id} domain = ${expected.domain}`);
        assert(sentence.machine === expected.machine, `${expected.id} machine = ${expected.machine}`);
    }
}

// ================================================================
// TRUTH CHANNELS (Paper Section 2.1)
// ================================================================

section('Truth Channels');

assertApprox(TRUTH_THRESHOLDS.PARADOX, 0.60, 1e-6, 'PARADOX threshold = 0.60');
assertApprox(TRUTH_THRESHOLDS.TRUE, 0.90, 1e-6, 'TRUE threshold = 0.90');

const advisor = new HelixOperatorAdvisor();

// Test truth channel at various z values
assert(advisor.truthChannelFromZ(0.3) === 'UNTRUE', 'z=0.3 → UNTRUE');
assert(advisor.truthChannelFromZ(0.5) === 'UNTRUE', 'z=0.5 → UNTRUE');
assert(advisor.truthChannelFromZ(0.59) === 'UNTRUE', 'z=0.59 → UNTRUE');
assert(advisor.truthChannelFromZ(0.60) === 'PARADOX', 'z=0.60 → PARADOX');
assert(advisor.truthChannelFromZ(0.75) === 'PARADOX', 'z=0.75 → PARADOX');
assert(advisor.truthChannelFromZ(0.89) === 'PARADOX', 'z=0.89 → PARADOX');
assert(advisor.truthChannelFromZ(0.90) === 'TRUE', 'z=0.90 → TRUE');
assert(advisor.truthChannelFromZ(0.95) === 'TRUE', 'z=0.95 → TRUE');
assert(advisor.truthChannelFromZ(1.0) === 'TRUE', 'z=1.0 → TRUE');

// ================================================================
// HARMONIC MAPPING
// ================================================================

section('Harmonic Mapping');

// Test harmonic mapping at boundary values
assert(advisor.harmonicFromZ(0.05) === 't1', 'z=0.05 → t1');
assert(advisor.harmonicFromZ(0.15) === 't2', 'z=0.15 → t2');
assert(advisor.harmonicFromZ(0.30) === 't3', 'z=0.30 → t3');
assert(advisor.harmonicFromZ(0.50) === 't4', 'z=0.50 → t4');
assert(advisor.harmonicFromZ(0.70) === 't5', 'z=0.70 → t5');
assert(advisor.harmonicFromZ(0.80) === 't6', 'z=0.80 → t6');
assert(advisor.harmonicFromZ(0.88) === 't7', 'z=0.88 → t7');
assert(advisor.harmonicFromZ(0.93) === 't8', 'z=0.93 → t8');
assert(advisor.harmonicFromZ(0.98) === 't9', 'z=0.98 → t9');

// ================================================================
// TRIAD HYSTERESIS GATE (Paper Section 2.4)
// ================================================================

section('TRIAD Hysteresis Gate');

// Test basic TRIAD mechanics
const gate = new TriadGate({ skipEnvInit: true });

assert(gate.unlocked === false, 'Gate starts locked');
assert(gate.passes === 0, 'Gate starts with 0 passes');
assert(gate.getT6Gate() === Z_CRITICAL, 'Locked gate returns Z_CRITICAL');

// First pass: rise above TRIAD_HIGH
gate.update(0.70);  // Below threshold, armed
assert(gate._armed === true, 'Gate armed at z=0.70');

gate.update(0.86);  // Above TRIAD_HIGH
assert(gate.passes === 1, 'First rising edge: passes = 1');
assert(gate._armed === false, 'Gate disarmed after rising edge');
assert(gate.unlocked === false, 'Not unlocked after 1 pass');

// Re-arm: drop below TRIAD_LOW
gate.update(0.81);  // Below TRIAD_LOW
assert(gate._armed === true, 'Re-armed after dropping below TRIAD_LOW');

// Second pass
gate.update(0.86);
assert(gate.passes === 2, 'Second rising edge: passes = 2');
assert(gate.unlocked === false, 'Not unlocked after 2 passes');

// Re-arm again
gate.update(0.80);
assert(gate._armed === true, 'Re-armed again');

// Third pass → UNLOCK
const result = gate.update(0.87);
assert(gate.passes === 3, 'Third rising edge: passes = 3');
assert(gate.unlocked === true, 'UNLOCKED after 3 passes');
assert(result.event === 'UNLOCKED', 'Event is UNLOCKED');
assert(gate.getT6Gate() === TRIAD_T6, 'Unlocked gate returns TRIAD_T6');

// ================================================================
// NEGENTROPY SIGNAL (Paper Section 2.3)
// ================================================================

section('Negentropy Signal (ΔS_neg)');

// ΔS_neg should be maximal (=1) at Z_CRITICAL
const deltaSNegAtCritical = advisor.computeDeltaSNeg(Z_CRITICAL);
assertApprox(deltaSNegAtCritical, 1.0, 1e-10, 'ΔS_neg = 1.0 at Z_CRITICAL');

// ΔS_neg decreases away from Z_CRITICAL
const deltaSNegLow = advisor.computeDeltaSNeg(0.5);
const deltaSNegHigh = advisor.computeDeltaSNeg(0.95);
assert(deltaSNegLow < deltaSNegAtCritical, 'ΔS_neg < 1 below Z_CRITICAL');
assert(deltaSNegHigh < deltaSNegAtCritical, 'ΔS_neg < 1 above Z_CRITICAL');

// Symmetric around Z_CRITICAL
const d = 0.05;
const deltaSNegMinus = advisor.computeDeltaSNeg(Z_CRITICAL - d);
const deltaSNegPlus = advisor.computeDeltaSNeg(Z_CRITICAL + d);
assertApprox(deltaSNegMinus, deltaSNegPlus, 1e-10, 'ΔS_neg symmetric around Z_CRITICAL');

// ================================================================
// π-REGIME BLENDING WEIGHTS (Paper Section 2.3)
// ================================================================

section('π-Regime Blending Weights');

// Below Z_CRITICAL: wPi should be 0
const weightsLow = advisor.computeBlendWeights(0.5);
assertApprox(weightsLow.wPi, 0.0, 1e-10, 'wPi = 0 below Z_CRITICAL');
assertApprox(weightsLow.wLoc, 1.0, 1e-10, 'wLoc = 1 below Z_CRITICAL');

// At Z_CRITICAL: wPi = 1 (since ΔS_neg = 1)
const weightsAtCritical = advisor.computeBlendWeights(Z_CRITICAL);
assertApprox(weightsAtCritical.wPi, 1.0, 1e-10, 'wPi = 1.0 at Z_CRITICAL');
assertApprox(weightsAtCritical.wLoc, 0.0, 1e-10, 'wLoc = 0.0 at Z_CRITICAL');

// Above Z_CRITICAL: wPi + wLoc = 1 always
const weightsHigh = advisor.computeBlendWeights(0.95);
assertApprox(weightsHigh.wPi + weightsHigh.wLoc, 1.0, 1e-10, 'wPi + wLoc = 1 always');

// ================================================================
// HELIX COORDINATE
// ================================================================

section('Helix Coordinate');

// Test fromParameter
const coord = HelixCoordinate.fromParameter(0);
assert(coord.theta >= 0 && coord.theta <= 2 * Math.PI, 'theta in [0, 2π]');
assert(coord.z >= 0 && coord.z <= 1, 'z in [0, 1]');
assertApprox(coord.r, 1.0, 1e-10, 'r = 1.0 for unit helix');

// Test fromZ
const coordFromZ = HelixCoordinate.fromZ(0.75);
assertApprox(coordFromZ.z, 0.75, 1e-10, 'fromZ preserves z value');

// Test toVector
const vec = coord.toVector();
assert('x' in vec && 'y' in vec && 'z' in vec, 'toVector returns {x, y, z}');
assertApprox(Math.sqrt(vec.x * vec.x + vec.y * vec.y), coord.r, 1e-10, 'Vector xy magnitude = r');

// Test signature
const sig = coord.signature();
assert(typeof sig === 'string' && sig.includes('Δ'), 'signature returns formatted string');

// ================================================================
// ALPHA TOKEN SYNTHESIZER
// ================================================================

section('Alpha Token Synthesizer');

const synthesizer = new AlphaTokenSynthesizer();

// Test fromZ returns valid token
const tokenLow = synthesizer.fromZ(0.3);
assert(tokenLow !== null, 'Token synthesized for z=0.3');
assert(tokenLow.truthChannel === 'UNTRUE', 'Token at z=0.3 has UNTRUE channel');

const tokenMid = synthesizer.fromZ(0.75);
assert(tokenMid !== null, 'Token synthesized for z=0.75');
assert(tokenMid.truthChannel === 'PARADOX', 'Token at z=0.75 has PARADOX channel');

const tokenHigh = synthesizer.fromZ(0.95);
assert(tokenHigh !== null, 'Token synthesized for z=0.95');
assert(tokenHigh.truthChannel === 'TRUE', 'Token at z=0.95 has TRUE channel');

// Test findMatchingSentences
const waveSentences = synthesizer.findMatchingSentences({ domain: 'wave' });
assert(waveSentences.length === 2, 'Two wave domain sentences (A3, A6)');

const groupSentences = synthesizer.findMatchingSentences({ operators: ['+'] });
assert(groupSentences.length >= 2, 'At least 2 sentences use + operator');

// ================================================================
// TRIADIC HELIX APL SYSTEM
// ================================================================

section('Triadic Helix APL System');

// Clean up environment variables from previous tests
if (typeof process !== 'undefined' && process.env) {
    delete process.env.QAPL_TRIAD_UNLOCK;
    delete process.env.QAPL_TRIAD_COMPLETIONS;
}

const system = new TriadicHelixAPLSystem({ verbose: false });

// Test initial state
assert(system.z === 0.5, 'System starts at z=0.5');
assert(system.triadGate.unlocked === false, 'System starts with TRIAD locked');

// Test step
system.setZ(0.7);
const stepResult = system.step();
assert('z' in stepResult && 'operator' in stepResult && 'harmonic' in stepResult, 'step() returns expected fields');
assert(stepResult.harmonic === 't5', 'z=0.7 maps to t5');

// Test simulation
system.reset();
system.setZ(0.5);
const simResults = system.simulate(50, { targetZ: 0.85 });
assert(simResults.length === 50, 'Simulation returns 50 results');
assert(system.z > 0.5, 'Z increased toward target');

// Test simulateToUnlock
system.reset();
const unlockResult = system.simulateToUnlock(500);
assert(unlockResult.success === true, 'simulateToUnlock achieves unlock');
assert(system.triadGate.unlocked === true, 'TRIAD gate unlocked after simulation');

// Test analytics
const analytics = system.getAnalytics();
assert('totalSteps' in analytics && 'operators' in analytics, 'getAnalytics returns expected fields');

// Test reset
system.reset();
assert(system.z === 0.5, 'reset() restores z=0.5');
assert(system.triadGate.unlocked === false, 'reset() locks TRIAD gate');

// ================================================================
// OPERATOR WINDOW CONSTRAINTS
// ================================================================

section('Operator Window Constraints');

// Verify that certain operators are NOT in certain tiers (paper constraints)
assert(!OPERATOR_WINDOWS.t1.includes('×'), 'Fusion (×) not in t1');
assert(!OPERATOR_WINDOWS.t1.includes('^'), 'Amplify (^) not in t1');
assert(!OPERATOR_WINDOWS.t1.includes('+'), 'Group (+) not in t1');

assert(!OPERATOR_WINDOWS.t6.includes('×'), 'Fusion (×) not in t6');
assert(!OPERATOR_WINDOWS.t6.includes('^'), 'Amplify (^) not in t6');

assert(!OPERATOR_WINDOWS.t7.includes('^'), 'Amplify (^) not in t7');
assert(!OPERATOR_WINDOWS.t7.includes('÷'), 'Decoherence (÷) not in t7');
assert(!OPERATOR_WINDOWS.t7.includes('−'), 'Separation (−) not in t7');

// t5 has ALL operators (maximal freedom in paradox zone)
assert(OPERATOR_WINDOWS.t5.includes('()'), 't5 has Boundary');
assert(OPERATOR_WINDOWS.t5.includes('×'), 't5 has Fusion');
assert(OPERATOR_WINDOWS.t5.includes('^'), 't5 has Amplify');
assert(OPERATOR_WINDOWS.t5.includes('÷'), 't5 has Decoherence');
assert(OPERATOR_WINDOWS.t5.includes('+'), 't5 has Group');
assert(OPERATOR_WINDOWS.t5.includes('−'), 't5 has Separation');

// ================================================================
// TRUTH BIAS WEIGHTING
// ================================================================

section('Truth Bias Weighting');

// TRUE state favors amplify and group
assert(TRUTH_BIAS.TRUE['^'] > TRUTH_BIAS.TRUE['÷'], 'TRUE: Amplify > Decoherence');
assert(TRUTH_BIAS.TRUE['+'] > TRUTH_BIAS.TRUE['−'], 'TRUE: Group > Separation');

// UNTRUE state favors decoherence and separation
assert(TRUTH_BIAS.UNTRUE['÷'] > TRUTH_BIAS.UNTRUE['^'], 'UNTRUE: Decoherence > Amplify');
assert(TRUTH_BIAS.UNTRUE['−'] > TRUTH_BIAS.UNTRUE['+'], 'UNTRUE: Separation > Group');

// PARADOX state favors boundary and fusion (structure formation)
assert(TRUTH_BIAS.PARADOX['()'] >= TRUTH_BIAS.PARADOX['÷'], 'PARADOX: Boundary ≥ Decoherence');
assert(TRUTH_BIAS.PARADOX['×'] >= TRUTH_BIAS.PARADOX['−'], 'PARADOX: Fusion ≥ Separation');

// ================================================================
// DYNAMIC T6 GATE WITH TRIAD
// ================================================================

section('Dynamic t6 Gate');

// Create fresh advisor with unlocked TRIAD
const unlockedGate = new TriadGate({ skipEnvInit: true });
unlockedGate.forceUnlock();
const advisorUnlocked = new HelixOperatorAdvisor({ triadGate: unlockedGate });

// With TRIAD unlocked, t6 boundary should be TRIAD_T6 (0.83) instead of Z_CRITICAL (~0.866)
assert(advisorUnlocked.getT6Gate() === TRIAD_T6, 'Unlocked advisor uses TRIAD_T6');

// z=0.84 should be t7 with unlocked gate, but t6 with locked gate
const lockedGate = new TriadGate({ skipEnvInit: true });
const advisorLocked = new HelixOperatorAdvisor({ triadGate: lockedGate });

assert(advisorUnlocked.harmonicFromZ(0.84) === 't7', 'z=0.84 → t7 with unlocked TRIAD');
assert(advisorLocked.harmonicFromZ(0.84) === 't6', 'z=0.84 → t6 with locked TRIAD');

// ================================================================
// SUMMARY
// ================================================================

console.log(`\n${'═'.repeat(60)}`);
console.log(`  TEST SUMMARY`);
console.log(`${'═'.repeat(60)}`);
console.log(`  Passed: ${passed}`);
console.log(`  Failed: ${failed}`);
console.log(`  Total:  ${passed + failed}`);
console.log(`${'═'.repeat(60)}`);

if (failed > 0) {
    console.log('\n  ⚠  Some tests failed!\n');
    process.exit(1);
} else {
    console.log('\n  ✓  All tests passed!\n');
    process.exit(0);
}
