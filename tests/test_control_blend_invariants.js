const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };
const { QuantumAPL } = require('../src/quantum_apl_engine');
const C = require('../src/constants');

(function testBlendAdmissibilityAndScale() {
  const engine = new QuantumAPL();
  const zc = C.Z_CRITICAL;

  // Blend admissibility: below lens, w_pi == 0; at/above, w_pi == s
  const below = zc - 1e-3;
  const above = zc + 1e-3;
  const hBelow = engine.helixAdvisor.describe(below);
  const hAbove = engine.helixAdvisor.describe(above);
  assert(Math.abs(hBelow.weights.pi - 0.0) < 1e-12, 'w_pi should be 0 below lens');
  const sAbove = C.computeDeltaSNeg(above, C.LENS_SIGMA);
  assert(Math.abs(hAbove.weights.pi - sAbove) < 1e-12, 'w_pi should equal s above lens');

  // Gate scale invariants for entropy control
  const phiInv = 1 / C.PHI;
  const s1 = 1.0; // at z=z_c
  const scale1 = Math.max(0, s1 - phiInv) / (1 - phiInv);
  assert(Math.abs(scale1 - 1.0) < 1e-12, 'scale must be 1 at s=1');

  const s0 = Math.min(phiInv, 0.5);
  const scale0 = Math.max(0, s0 - phiInv) / (1 - phiInv);
  assert(Math.abs(scale0 - 0.0) < 1e-12, 'scale must be 0 when s <= φ^{-1}');

  console.log('Control/blend invariants tests passed');
})();

