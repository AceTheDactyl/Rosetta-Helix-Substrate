const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };
const C = require('../src/constants');

(function testGateScaleEdges() {
  const s1 = 1.0;
  const scale1 = C.phiGateScale(s1);
  assert(Math.abs(scale1 - 1.0) < 1e-12, 'phiGateScale(1) must be 1');

  const sphi = C.PHI_INV;
  const scale0 = C.phiGateScale(sphi);
  assert(Math.abs(scale0 - 0.0) < 1e-12, 'phiGateScale(φ^{-1}) must be 0');
  console.log('Gate scale edge tests passed');
})();

