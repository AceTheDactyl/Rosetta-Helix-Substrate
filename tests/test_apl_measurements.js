const assert = require('assert');
const { UnifiedDemo } = require('../src/legacy/QuantumClassicalBridge');

function testEigenMeasurementToken() {
  const demo = new UnifiedDemo();
  const res = demo.bridge.aplMeasureEigen(0, 'Phi');
  assert(res.aplToken && res.aplToken.startsWith('Φ:T(ϕ_0)'), 'Missing eigen APL token');
  const hist = demo.bridge.operatorHistory || [];
  const last = hist[hist.length - 1];
  assert(last && last.aplToken === res.aplToken, 'Token not recorded in history');
  assert(typeof last.aplProb === 'number' && last.aplProb >= 0 && last.aplProb <= 1, 'Probability missing/bounds');
}

function testSubspaceMeasurementTokens() {
  const demo = new UnifiedDemo();
  const resPhi = demo.bridge.aplMeasureSubspace([2,3], 'Phi');
  assert(/Φ:Π\(subspace\)/.test(resPhi.aplToken), 'Missing Φ subspace token');
  const resPi = demo.bridge.aplMeasureSubspace([2,3], 'Pi');
  assert(/π:Π\(subspace\)/.test(resPi.aplToken), 'Missing π subspace token');
}

function testCompositeMeasurementTokens() {
  const demo = new UnifiedDemo();
  const comps = [
    { eigenIndex: 0, field: 'Phi', truthChannel: 'TRUE', weight: 0.3 },
    { eigenIndex: 1, field: 'Pi',  truthChannel: 'UNTRUE', weight: 0.3 },
    { subspaceIndices: [2,3], field: 'Phi', truthChannel: 'PARADOX', weight: 0.4 },
  ];
  const res = demo.bridge.aplMeasureComposite(comps);
  assert(Array.isArray(res.aplTokens) && res.aplTokens.length === 3, 'Composite tokens missing');
  const hist = demo.bridge.operatorHistory.filter(e => e.operator === 'APL_MEAS');
  assert(hist.length >= 3, 'Composite tokens not recorded');
}

function run() {
  testEigenMeasurementToken();
  testSubspaceMeasurementTokens();
  testCompositeMeasurementTokens();
  console.log('APL measurement tests passed');
}

if (require.main === module) run();

