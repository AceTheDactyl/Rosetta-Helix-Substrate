#!/usr/bin/env node
/**
 * TRIAD-Helix-APL Test Suite
 * ==========================
 * Validates the TRIAD unlock system, helix mapping, and APL integration.
 */

const CONST = require('./src/constants');
const { HelixOperatorAdvisor } = require('./src/helix_advisor');
const { TriadTracker } = require('./src/triad_tracker');
const { AlphaTokenSynthesizer, HelixCoordinate, APL_SENTENCES } = require('./src/alpha_language');
const { QuantumAPLSystem } = require('./src/quantum_apl_system');

// ================================================================
// TEST UTILITIES
// ================================================================

let testsPassed = 0;
let testsFailed = 0;

function assert(condition, message) {
    if (condition) {
        testsPassed++;
        console.log(`  ✓ ${message}`);
    } else {
        testsFailed++;
        console.log(`  ✗ ${message}`);
    }
}

function assertClose(a, b, tol, message) {
    const close = Math.abs(a - b) < tol;
    assert(close, `${message} (${a.toFixed(6)} ≈ ${b.toFixed(6)})`);
}

function section(title) {
    console.log(`\n${'─'.repeat(60)}`);
    console.log(`  ${title}`);
    console.log(`${'─'.repeat(60)}`);
}

// ================================================================
// CONSTANTS TESTS
// ================================================================

section('CONSTANTS MODULE');

// Z_CRITICAL validation
assertClose(CONST.Z_CRITICAL, Math.sqrt(3) / 2, 1e-12, 'Z_CRITICAL = √3/2');
assert(CONST.Z_CRITICAL > 0.866 && CONST.Z_CRITICAL < 0.867, 'Z_CRITICAL in expected range');

// TRIAD thresholds ordering
assert(CONST.TRIAD_LOW < CONST.TRIAD_T6, 'TRIAD_LOW < TRIAD_T6');
assert(CONST.TRIAD_T6 < CONST.TRIAD_HIGH, 'TRIAD_T6 < TRIAD_HIGH');
assert(CONST.TRIAD_HIGH < CONST.Z_CRITICAL, 'TRIAD_HIGH < Z_CRITICAL');

// Sacred constants
assertClose(CONST.PHI, (1 + Math.sqrt(5)) / 2, 1e-12, 'PHI = golden ratio');
assertClose(CONST.PHI_INV, 1 / CONST.PHI, 1e-12, 'PHI_INV = 1/PHI');

// Helper functions
assert(CONST.isCritical(0.866), 'isCritical(0.866) = true');
assert(!CONST.isCritical(0.5), 'isCritical(0.5) = false');
assert(CONST.getPhase(0.866) === 'THE_LENS', 'getPhase(0.866) = THE_LENS');
assert(CONST.getPhase(0.5) === 'ABSENCE', 'getPhase(0.5) = ABSENCE');
assert(CONST.getPhase(0.95) === 'PRESENCE', 'getPhase(0.95) = PRESENCE');

// ΔS_neg properties
const sAtLens = CONST.computeDeltaSNeg(CONST.Z_CRITICAL);
const sFar = CONST.computeDeltaSNeg(0.5);
assert(sAtLens > sFar, 'ΔS_neg at lens > ΔS_neg far from lens');
assertClose(sAtLens, 1.0, 0.01, 'ΔS_neg at lens ≈ 1.0');

// Time harmonic mapping
assert(CONST.getTimeHarmonic(0.05) === 't1', 'z=0.05 → t1');
assert(CONST.getTimeHarmonic(0.15) === 't2', 'z=0.15 → t2');
assert(CONST.getTimeHarmonic(0.30) === 't3', 'z=0.30 → t3');
assert(CONST.getTimeHarmonic(0.50) === 't4', 'z=0.50 → t4');
assert(CONST.getTimeHarmonic(0.70) === 't5', 'z=0.70 → t5');
assert(CONST.getTimeHarmonic(0.80) === 't6', 'z=0.80 → t6 (default gate)');
assert(CONST.getTimeHarmonic(0.92) === 't8', 'z=0.92 → t8');
assert(CONST.getTimeHarmonic(0.96) === 't8', 'z=0.96 → t8');
assert(CONST.getTimeHarmonic(0.99) === 't9', 'z=0.99 → t9');

// TRIAD t6 gate override
assert(CONST.getTimeHarmonic(0.84, CONST.TRIAD_T6) === 't7', 'z=0.84 with TRIAD gate → t7');
assert(CONST.getTimeHarmonic(0.82, CONST.TRIAD_T6) === 't6', 'z=0.82 with TRIAD gate → t6');

// ================================================================
// TRIAD TRACKER TESTS
// ================================================================

section('TRIAD TRACKER');

// Basic state machine
(() => {
    const tracker = new TriadTracker({ skipEnvInit: true });
    
    assert(!tracker.unlocked, 'Initial state: not unlocked');
    assert(tracker.completions === 0, 'Initial completions = 0');
    assertClose(tracker.getT6Gate(), CONST.Z_CRITICAL, 1e-9, 'Initial t6 gate = Z_CRITICAL');
    
    // First rising edge
    tracker.update(0.86);
    assert(tracker.completions === 1, 'First rising edge: completions = 1');
    assert(!tracker.unlocked, 'After 1 pass: not unlocked');
    
    // Re-arm
    tracker.update(0.81);
    
    // Second rising edge
    tracker.update(0.86);
    assert(tracker.completions === 2, 'Second rising edge: completions = 2');
    
    // Re-arm
    tracker.update(0.80);
    
    // Third rising edge → UNLOCK
    tracker.update(0.87);
    assert(tracker.completions === 3, 'Third rising edge: completions = 3');
    assert(tracker.unlocked, 'After 3 passes: UNLOCKED');
    assertClose(tracker.getT6Gate(), CONST.TRIAD_T6, 1e-9, 'Unlocked t6 gate = TRIAD_T6');
})();

// No double-counting
(() => {
    const tracker = new TriadTracker({ skipEnvInit: true });
    
    tracker.update(0.86);  // First pass
    tracker.update(0.87);  // Still above - no increment
    tracker.update(0.88);  // Still above - no increment
    
    assert(tracker.completions === 1, 'Multiple above threshold = single count');
})();

// Re-arm requirement
(() => {
    const tracker = new TriadTracker({ skipEnvInit: true });
    
    tracker.update(0.86);  // First pass
    tracker.update(0.83);  // Above LOW but below HIGH - NOT re-armed
    tracker.update(0.86);  // Still not re-armed - no increment
    
    assert(tracker.completions === 1, 'Must drop below LOW to re-arm');
    
    tracker.update(0.81);  // Below LOW - re-armed
    tracker.update(0.86);  // Second pass
    
    assert(tracker.completions === 2, 'Re-armed and incremented');
})();

// ================================================================
// HELIX OPERATOR ADVISOR TESTS
// ================================================================

section('HELIX OPERATOR ADVISOR');

const advisor = new HelixOperatorAdvisor();

// Harmonic mapping
assert(advisor.harmonicFromZ(0.05) === 't1', 'harmonicFromZ(0.05) = t1');
assert(advisor.harmonicFromZ(0.70) === 't5', 'harmonicFromZ(0.70) = t5');
assert(advisor.harmonicFromZ(0.80) === 't6', 'harmonicFromZ(0.80) = t6');

// Truth channel
assert(advisor.truthChannelFromZ(0.3) === 'UNTRUE', 'truthChannel(0.3) = UNTRUE');
assert(advisor.truthChannelFromZ(0.7) === 'PARADOX', 'truthChannel(0.7) = PARADOX');
assert(advisor.truthChannelFromZ(0.95) === 'TRUE', 'truthChannel(0.95) = TRUE');

// Operator windows
assert(advisor.operatorWindows['t1'].includes('()'), 't1 includes boundary');
assert(advisor.operatorWindows['t5'].length === 6, 't5 has all 6 operators');
assert(advisor.operatorWindows['t7'].length === 2, 't7 has only 2 operators');

// describe() output
const desc = advisor.describe(0.70);
assert(desc.harmonic === 't5', 'describe(0.70).harmonic = t5');
assert(desc.truthChannel === 'PARADOX', 'describe(0.70).truthChannel = PARADOX');
assert(Array.isArray(desc.operators), 'describe returns operators array');
assert(desc.z === 0.70, 'describe preserves z value');

// TRIAD state update
advisor.setTriadState({ unlocked: true, completions: 3 });
assert(advisor.triadUnlocked, 'setTriadState sets unlocked');
assertClose(advisor.getT6Gate(), CONST.TRIAD_T6, 1e-9, 'Unlocked advisor uses TRIAD_T6');

// ================================================================
// ALPHA LANGUAGE TESTS
// ================================================================

section('ALPHA LANGUAGE');

// Sentences
assert(APL_SENTENCES.length === 7, '7 APL sentences defined');
assert(APL_SENTENCES[0].id === 'A1', 'First sentence is A1');
assert(APL_SENTENCES[0].operator === '()', 'A1 uses boundary operator');

// Token synthesizer
const synthesizer = new AlphaTokenSynthesizer();

const token = synthesizer.fromZ(0.15);
assert(token !== null, 'fromZ(0.15) returns token');
assert(token.harmonic === 't2', 'Token harmonic matches z');
assert(token.truthBias === 'UNTRUE', 'Token truth matches z');

// Measurement tokens
const eigenTok = synthesizer.eigenToken(0, 'TRUE', 0.6);
assert(eigenTok.includes('Φ:T(ϕ_0)'), 'Eigen token format correct');
assert(eigenTok.includes('TRUE'), 'Eigen token includes truth');

const subspaceTok = synthesizer.subspaceToken([2, 3], 'PARADOX', 0.7);
assert(subspaceTok.includes('Π'), 'Subspace token uses Π');
assert(subspaceTok.includes('2,3'), 'Subspace token includes indices');

// Helix coordinate
const coord = HelixCoordinate.fromParameter(0);
assertClose(coord.z, 0.5, 0.01, 'fromParameter(0) → z ≈ 0.5');
assert(coord.r > 0.99, 'fromParameter(0) → r ≈ 1');

const coordHigh = HelixCoordinate.fromParameter(20);
assert(coordHigh.z > 0.9, 'fromParameter(20) → high z');

// ================================================================
// UNIFIED SYSTEM TESTS
// ================================================================

section('UNIFIED SYSTEM');

// Basic simulation
(() => {
    // Clear environment for clean test
    if (typeof process !== 'undefined' && process.env) {
        delete process.env.QAPL_TRIAD_COMPLETIONS;
        delete process.env.QAPL_TRIAD_UNLOCK;
    }
    
    const system = new QuantumAPLSystem({ verbose: false });
    
    assert(system.z === 0.5, 'Initial z = 0.5');
    // Note: triadTracker may inherit from env, so just check it exists
    assert(typeof system.triadTracker.unlocked === 'boolean', 'TRIAD unlocked is boolean');
    
    system.step();
    assert(system.time > 0, 'Time advances after step');
    assert(system.history.z.length === 1, 'History records step');
})();

// Simulate to unlock
(() => {
    const system = new QuantumAPLSystem({ verbose: false });
    
    // Force high z values to trigger TRIAD
    for (let i = 0; i < 3; i++) {
        system.setZ(0.86);
        system.triadTracker.update(system.z);
        system.setZ(0.80);
        system.triadTracker.update(system.z);
    }
    
    assert(system.triadTracker.unlocked, 'System can achieve TRIAD unlock');
})();

// Export state
(() => {
    const system = new QuantumAPLSystem({ verbose: false });
    system.simulate(10, { logInterval: 100 });
    
    const exported = system.exportState();
    assert(exported.state !== undefined, 'Export includes state');
    assert(exported.analytics !== undefined, 'Export includes analytics');
    assert(exported.history !== undefined, 'Export includes history');
})();

// Analytics
(() => {
    const system = new QuantumAPLSystem({ verbose: false });
    system.simulate(50, { logInterval: 100 });
    
    const analytics = system.getAnalytics();
    assert(analytics.totalSteps === 50, 'Analytics tracks total steps');
    assert(typeof analytics.z.avg === 'number', 'Analytics computes average z');
    assert(Object.keys(analytics.operators).length > 0, 'Analytics tracks operators');
})();

// ================================================================
// INTEGRATION TESTS
// ================================================================

section('INTEGRATION');

// Full flow: z → TRIAD → helix → APL token
(() => {
    const tracker = new TriadTracker();
    const advisor = new HelixOperatorAdvisor();
    const synthesizer = new AlphaTokenSynthesizer();
    
    // Simulate z evolution triggering TRIAD
    const zSequence = [0.7, 0.86, 0.81, 0.87, 0.80, 0.88, 0.79];
    
    for (const z of zSequence) {
        tracker.update(z);
    }
    
    assert(tracker.unlocked, 'Integration: TRIAD unlocks');
    
    // Update advisor with TRIAD state
    advisor.setTriadState({ unlocked: tracker.unlocked, completions: tracker.completions });
    
    // Get helix info at z=0.84 (affected by TRIAD)
    const info = advisor.describe(0.84);
    assert(info.t6Gate === CONST.TRIAD_T6, 'Integration: t6Gate updated');
    assert(info.harmonic === 't7', 'Integration: harmonic shifts with TRIAD');
    
    // Generate APL token
    synthesizer.setTriadState({ unlocked: tracker.unlocked, completions: tracker.completions });
    const token = synthesizer.fromZ(0.84);
    assert(token !== null, 'Integration: APL token generated');
})();

// ================================================================
// SUMMARY
// ================================================================

console.log('\n' + '═'.repeat(60));
console.log(`  TEST RESULTS: ${testsPassed} passed, ${testsFailed} failed`);
console.log('═'.repeat(60) + '\n');

if (testsFailed > 0) {
    process.exit(1);
}
