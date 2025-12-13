const assert = require('assert');
const { UnifiedDemo } = require('../src/legacy/QuantumClassicalBridge');

function approx(a, b, eps = 1e-6) {
  return Math.abs(a - b) <= eps;
}

function tailAplMeas(history, n) {
  return history.filter(e => e && e.operator === 'APL_MEAS').slice(-n);
}

function testCompositeAllZeroWeightsUniform() {
  const demo = new UnifiedDemo();
  const b = demo.bridge;
  const comps = [
    { eigenIndex: 0, field: 'Phi', truthChannel: 'TRUE', weight: 0 },
    { subspaceIndices: [2, 3], field: 'Pi', truthChannel: 'UNTRUE', weight: 0 },
    { eigenIndex: 2, field: 'Phi', truthChannel: 'TRUE', weight: 0 },
  ];
  const res = b.aplMeasureComposite(comps);
  assert(Array.isArray(res.allProbabilities) && res.allProbabilities.length === 3, 'composite probs missing');
  res.allProbabilities.forEach(p => assert(approx(p.probability, 1 / 3), `expected uniform 1/3, got ${p.probability}`));
  const last = tailAplMeas(b.operatorHistory, 3);
  assert.strictEqual(last.length, 3, 'expected 3 measurement entries appended');
  last.forEach(e => assert(approx(e.aplProb, 1 / 3), `history aplProb not uniform: ${e.aplProb}`));
}

function testCompositeNaNWeightsClampToZero() {
  const demo = new UnifiedDemo();
  const b = demo.bridge;
  const comps = [
    { eigenIndex: 0, field: 'Phi', truthChannel: 'TRUE', weight: Number.NaN },
    { subspaceIndices: [2, 3], field: 'Phi', truthChannel: 'PARADOX', weight: Number.NaN },
  ];
  const res = b.aplMeasureComposite(comps);
  assert(Array.isArray(res.allProbabilities) && res.allProbabilities.length === 2);
  // res probabilities likely NaN; recording path maps non-finite to 0
  const last = tailAplMeas(b.operatorHistory, 2);
  assert.strictEqual(last.length, 2);
  last.forEach(e => assert(approx(e.aplProb, 0), `aplProb should clamp to 0 for NaN weights, got ${e.aplProb}`));
}

function testCompositeNegativeTotalsUniform() {
  const demo = new UnifiedDemo();
  const b = demo.bridge;
  const comps = [
    { eigenIndex: 1, field: 'Pi', truthChannel: 'UNTRUE', weight: -1 },
    { subspaceIndices: [2, 3], field: 'Phi', truthChannel: 'PARADOX', weight: -2 },
  ];
  const res = b.aplMeasureComposite(comps);
  assert(Array.isArray(res.allProbabilities) && res.allProbabilities.length === 2);
  res.allProbabilities.forEach(p => assert(approx(p.probability, 0.5), `expected uniform 0.5, got ${p.probability}`));
  const last = tailAplMeas(b.operatorHistory, 2);
  last.forEach(e => assert(approx(e.aplProb, 0.5), `history aplProb not uniform 0.5: ${e.aplProb}`));
}

function run() {
  testCompositeAllZeroWeightsUniform();
  testCompositeNaNWeightsClampToZero();
  testCompositeNegativeTotalsUniform();
  console.log('APL measurement edge-case probability tests passed');
}

if (require.main === module) run();

module.exports = {
  testCompositeAllZeroWeightsUniform,
  testCompositeNaNWeightsClampToZero,
  testCompositeNegativeTotalsUniform,
};

