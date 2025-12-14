const assert = require('assert');

function withEnv(env, fn) {
  const prev = {};
  for (const k of Object.keys(env)) {
    prev[k] = process.env[k];
    process.env[k] = env[k];
  }
  try { fn(); } finally {
    for (const k of Object.keys(env)) {
      if (prev[k] === undefined) delete process.env[k]; else process.env[k] = prev[k];
    }
  }
}

function runOnce(expectCollapse) {
  // Lazy require inside to respect env toggling
  const { UnifiedDemo } = require('../src/legacy/QuantumClassicalBridge');
  const demo = new UnifiedDemo();
  const r1 = demo.bridge.aplMeasureEigen(0, 'Phi');
  const r2 = demo.bridge.aplMeasureSubspace([2,3], 'Phi');
  if (expectCollapse) {
    assert(/Φ:⟂\(ϕ_0\)TRUE@/.test(r1.aplToken), `expected collapse glyph in eigen token, got ${r1.aplToken}`);
    assert(/Φ:⟂\(subspace\)PARADOX@/.test(r2.aplToken), `expected collapse glyph in subspace token, got ${r2.aplToken}`);
  } else {
    assert(/Φ:T\(ϕ_0\)TRUE@/.test(r1.aplToken), `expected T() in eigen token, got ${r1.aplToken}`);
    assert(/Φ:Π\(subspace\)PARADOX@/.test(r2.aplToken), `expected Π(subspace) in subspace token, got ${r2.aplToken}`);
  }
}

function run() {
  withEnv({ QAPL_EMIT_COLLAPSE_GLYPH: '0' }, () => runOnce(false));
  withEnv({ QAPL_EMIT_COLLAPSE_GLYPH: '1' }, () => runOnce(true));
  console.log('Collapse alias emission tests passed');
}

if (require.main === module) run();

module.exports = { runOnce };

