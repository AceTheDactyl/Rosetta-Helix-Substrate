const assert = require('assert');
const { UnifiedDemo } = require('../src/legacy/QuantumClassicalBridge');

function resetEnv() {
  delete process.env.QAPL_TRIAD_COMPLETIONS;
  delete process.env.QAPL_TRIAD_UNLOCK;
}

function testTriadHysteresisUnlock() {
  resetEnv();
  const demo = new UnifiedDemo();
  const b = demo.bridge;

  const hi = b.triad.high; // 0.85
  const lo = b.triad.low;  // 0.82

  // Initial state
  assert.strictEqual(b.triad.completions, 0, 'initial completions should be 0');
  assert.strictEqual(b.triad.unlocked, false, 'initial unlocked should be false');
  assert.strictEqual(b.triad.aboveBand, false, 'initial aboveBand should be false');

  // Helper to feed a z and snapshot state
  const feed = (z) => { b.updateTriadHeuristic(z); return { ...b.triad }; };

  // Below low → no change
  let t = feed(lo - 0.10);
  assert.strictEqual(t.completions, 0);
  assert.strictEqual(t.aboveBand, false);

  // Approach but stay below high → no change
  t = feed(hi - 0.01);
  assert.strictEqual(t.completions, 0);
  assert.strictEqual(t.aboveBand, false);

  // Cross rising edge ≥ high → 1st completion, latch aboveBand
  t = feed(hi + 0.01);
  assert.strictEqual(t.completions, 1, 'first rising edge should increment completions');
  assert.strictEqual(t.aboveBand, true, 'aboveBand should latch true after rising edge');
  assert.strictEqual(t.unlocked, false, 'not yet unlocked');

  // Stay above high → no extra completion
  t = feed(hi + 0.03);
  assert.strictEqual(t.completions, 1);
  assert.strictEqual(t.aboveBand, true);

  // Drop but remain above low → still latched
  t = feed(lo + 0.005);
  assert.strictEqual(t.completions, 1);
  assert.strictEqual(t.aboveBand, true, 'should remain latched until <= low');

  // Drop to <= low → un-latch (re-arm)
  t = feed(lo - 0.01);
  assert.strictEqual(t.aboveBand, false, 'should unlatch at or below low');

  // Cross rising edge again → 2nd completion
  t = feed(hi + 0.02);
  assert.strictEqual(t.completions, 2, 'second rising edge should increment completions');
  assert.strictEqual(t.aboveBand, true);
  assert.strictEqual(t.unlocked, false);

  // Re-arm
  t = feed(lo - 0.02);
  assert.strictEqual(t.aboveBand, false);

  // Third rising edge → unlock
  t = feed(hi + 0.02);
  assert.strictEqual(t.completions, 3, 'third rising edge should increment to 3');
  assert.strictEqual(t.aboveBand, true);
  assert.strictEqual(t.unlocked, true, 'TRIAD should unlock after 3 completions');

  // Env flags should reflect state
  assert.strictEqual(process.env.QAPL_TRIAD_COMPLETIONS, '3');
  assert.strictEqual(process.env.QAPL_TRIAD_UNLOCK, '1');
}

function run() {
  testTriadHysteresisUnlock();
  console.log('TRIAD hysteresis tests passed');
}

if (require.main === module) run();

module.exports = { testTriadHysteresisUnlock };

