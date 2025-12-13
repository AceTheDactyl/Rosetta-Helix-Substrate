const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };
const C = require('../src/constants');

(function testLensRateProperties() {
  const zc = C.Z_CRITICAL;
  const r0 = C.lensRate(zc);
  assert(Math.abs(r0 - 0) < 1e-12, 'lensRate(z_c) must be 0');

  const r1 = C.lensRate(zc + 0.01);
  const r2 = C.lensRate(zc + 0.12);
  const r3 = C.lensRate(zc + 0.30);

  assert(r1 > 0, 'lensRate(z_c+ε) > 0');
  assert(r2 > r1, 'lensRate increases near lens to a peak');
  assert(r3 < r2, 'lensRate decreases as s becomes very small');
  console.log('Lens rate properties passed');
})();
