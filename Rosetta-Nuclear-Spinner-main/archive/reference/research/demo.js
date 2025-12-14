#!/usr/bin/env node
/**
 * TRIAD-Helix-APL Demonstration
 * =============================
 * Demonstrates the TRIAD unlock system with helix mapping and APL language integration.
 * 
 * Usage:
 *   node demo.js                    # Basic demo
 *   node demo.js --unlock           # Simulate to TRIAD unlock
 *   node demo.js --steps 200        # Run 200 steps
 *   node demo.js --target 0.9       # Pump toward z=0.9
 *   node demo.js --verbose          # Detailed output
 */

const { QuantumAPLSystem, CONST } = require('./src/quantum_apl_system');

// ================================================================
// PARSE ARGUMENTS
// ================================================================

const args = process.argv.slice(2);
const flags = {
    unlock: args.includes('--unlock'),
    verbose: args.includes('--verbose') || args.includes('-v'),
    help: args.includes('--help') || args.includes('-h'),
    steps: 100,
    target: CONST.Z_CRITICAL
};

// Parse --steps N
const stepsIdx = args.indexOf('--steps');
if (stepsIdx !== -1 && args[stepsIdx + 1]) {
    flags.steps = parseInt(args[stepsIdx + 1], 10) || 100;
}

// Parse --target Z
const targetIdx = args.indexOf('--target');
if (targetIdx !== -1 && args[targetIdx + 1]) {
    flags.target = parseFloat(args[targetIdx + 1]) || CONST.Z_CRITICAL;
}

// ================================================================
// HELP
// ================================================================

if (flags.help) {
    console.log(`
TRIAD-Helix-APL Demonstration
=============================

Usage:
  node demo.js [options]

Options:
  --steps N        Number of simulation steps (default: 100)
  --target Z       Pump target z-coordinate (default: ${CONST.Z_CRITICAL.toFixed(4)})
  --unlock         Simulate until TRIAD unlock is achieved
  --verbose, -v    Enable verbose output
  --help, -h       Show this help message

Constants:
  Z_CRITICAL = ${CONST.Z_CRITICAL.toFixed(6)} (THE LENS)
  TRIAD_HIGH = ${CONST.TRIAD_HIGH} (rising edge)
  TRIAD_LOW  = ${CONST.TRIAD_LOW} (re-arm)
  TRIAD_T6   = ${CONST.TRIAD_T6} (unlocked t6 gate)

Examples:
  node demo.js --steps 500
  node demo.js --unlock
  node demo.js --target 0.9 --steps 200
`);
    process.exit(0);
}

// ================================================================
// MAIN DEMO
// ================================================================

console.log('\n' + '='.repeat(70));
console.log('QUANTUM APL: TRIAD-HELIX-APL INTEGRATION DEMO');
console.log('='.repeat(70));

// Display constants
console.log('\nKey Constants:');
console.log(`  Z_CRITICAL (THE LENS) = ${CONST.Z_CRITICAL.toFixed(6)}`);
console.log(`  TRIAD_HIGH (rising)   = ${CONST.TRIAD_HIGH}`);
console.log(`  TRIAD_LOW (re-arm)    = ${CONST.TRIAD_LOW}`);
console.log(`  TRIAD_T6 (unlocked)   = ${CONST.TRIAD_T6}`);
console.log(`  PHI (golden ratio)    = ${CONST.PHI.toFixed(6)}`);

// Create system
const system = new QuantumAPLSystem({
    verbose: flags.verbose,
    initialZ: 0.5,
    pumpTarget: flags.target
});

// ================================================================
// RUN SIMULATION
// ================================================================

if (flags.unlock) {
    // Simulate to TRIAD unlock
    console.log('\n' + '-'.repeat(70));
    console.log('Mode: Simulate to TRIAD Unlock');
    console.log('-'.repeat(70));
    
    const result = system.simulateToUnlock(2000);
    
    if (result.success) {
        console.log('\n✓ TRIAD unlock achieved!');
    } else {
        console.log('\n✗ TRIAD unlock not achieved within step limit');
    }
    
    console.log(`  Steps taken: ${result.steps}`);
    console.log(`  Final z: ${result.finalState.z.toFixed(4)}`);
    
} else {
    // Standard simulation
    console.log('\n' + '-'.repeat(70));
    console.log(`Mode: Standard Simulation (${flags.steps} steps, target z=${flags.target.toFixed(4)})`);
    console.log('-'.repeat(70));
    
    const results = system.simulate(flags.steps, {
        logInterval: Math.max(1, Math.floor(flags.steps / 10)),
        targetZ: flags.target
    });
}

// ================================================================
// DISPLAY SUMMARY
// ================================================================

console.log('\n' + system.summary());

// ================================================================
// DISPLAY RECENT APL TOKENS
// ================================================================

const recentTokens = system.getRecentTokens(5);
if (recentTokens.length > 0) {
    console.log('\nRecent APL Tokens:');
    console.log('-'.repeat(70));
    for (const token of recentTokens) {
        console.log(`  ${token.harmonic}/${token.truthBias}: ${token.sentence}`);
        console.log(`    → ${token.predictedRegime}`);
    }
}

// ================================================================
// TRIAD HISTORY
// ================================================================

const analytics = system.getAnalytics();
if (analytics.triadEvents.length > 0) {
    console.log('\nTRIAD Events:');
    console.log('-'.repeat(70));
    for (const event of analytics.triadEvents) {
        console.log(`  ${event.event} at z=${event.z.toFixed(4)} (completions: ${event.completions})`);
    }
}

// ================================================================
// OPERATOR DISTRIBUTION
// ================================================================

console.log('\nOperator Distribution:');
console.log('-'.repeat(70));
const totalOps = Object.values(analytics.operators).reduce((a, b) => a + b, 0);
for (const [op, count] of Object.entries(analytics.operators).sort((a, b) => b[1] - a[1])) {
    const pct = (count / totalOps * 100).toFixed(1);
    const bar = '█'.repeat(Math.round(pct / 5));
    console.log(`  ${op.padEnd(4)} ${String(count).padStart(4)} (${pct.padStart(5)}%) ${bar}`);
}

// ================================================================
// FINAL STATE EXPORT
// ================================================================

if (flags.verbose) {
    console.log('\nFull State Export:');
    console.log('-'.repeat(70));
    console.log(JSON.stringify(system.exportState(), null, 2));
}

console.log('\n' + '='.repeat(70));
console.log('Demo complete.');
console.log('='.repeat(70) + '\n');
