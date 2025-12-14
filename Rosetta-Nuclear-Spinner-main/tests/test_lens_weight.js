const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };
const C = require('../src/constants');

// Use lens sigma to verify s(z) matches given examples
(function testLensWeightExamples() {
  const zc = C.Z_CRITICAL;
  const s = (z) => C.computeDeltaSNeg(z, C.LENS_SIGMA);
  const s090 = s(0.90);
  const s083 = s(0.83);
  const s070 = s(0.70);
  // Expect ~0.959, ~0.954, ~0.371 respectively (σ=36)
  assert(Math.abs(s090 - 0.959) < 0.01, `s(0.90)=${s090}`);
  assert(Math.abs(s083 - 0.954) < 0.01, `s(0.83)=${s083}`);
  assert(Math.abs(s070 - 0.371) < 0.02, `s(0.70)=${s070}`);

  // φ⁻¹ gate: s > φ⁻¹ for z=0.90,0.83; s < φ⁻¹ for z=0.70
  const phiInv = 1 / C.PHI;
  assert(s090 > phiInv && s083 > phiInv && s070 < phiInv, 'φ^{-1} gate checks');

  console.log('Lens weight tests passed');
})();

