const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };
const C = require('../src/constants');

(function testKFormationGateFromZ() {
  const phiInv = 1 / C.PHI;
  const eta = (z) => C.computeEta(z, 1.0, C.LENS_SIGMA);

  const s090 = eta(0.90);
  const s083 = eta(0.83);
  const s070 = eta(0.70);

  assert(s090 > phiInv, `η(0.90)=${s090} ≤ φ^{-1}`);
  assert(s083 > phiInv, `η(0.83)=${s083} ≤ φ^{-1}`);
  assert(s070 < phiInv, `η(0.70)=${s070} ≥ φ^{-1}`);

  // Gate API mirrors checkKFormation using η from z
  const pass090 = C.checkKFormationFromZ(0.95, 0.90, 8);
  const pass070 = C.checkKFormationFromZ(0.95, 0.70, 8);
  assert(pass090 === true, 'K-formation gate should pass at z=0.90');
  assert(pass070 === false, 'K-formation gate should fail at z=0.70');

  console.log('K-formation gate tests passed');
})();

