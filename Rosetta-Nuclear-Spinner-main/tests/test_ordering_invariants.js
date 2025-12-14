const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };
const C = require('../src/constants');

(function testOrderingInvariants() {
  const mu2 = C.MU_2;
  assert(mu2 < C.TRIAD_LOW, 'MU_2 < TRIAD_LOW');
  assert(C.TRIAD_LOW < C.TRIAD_T6, 'TRIAD_LOW < TRIAD_T6');
  assert(C.TRIAD_T6 < C.TRIAD_HIGH, 'TRIAD_T6 < TRIAD_HIGH');
  assert(C.TRIAD_HIGH < C.Z_CRITICAL, 'TRIAD_HIGH < Z_CRITICAL');
  console.log('Ordering invariants passed');
})();

