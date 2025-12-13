const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };
const C = require('../src/constants');

// Validate μ ordering, ratios, and barrier proximity to φ^{-1}
(() => {
  const muP = C.MU_P;
  const mu1 = C.MU_1;
  const mu2 = C.MU_2;
  const muS = C.KAPPA_S || C.MU_S;
  const mu3 = 0.992;

  assert(mu1 < muP && muP < (1/C.PHI) + 0.1, 'ordering around μ_P');
  assert((1/C.PHI) < mu2 && mu2 < C.Z_CRITICAL, 'ordering around μ_2 and z_c');
  assert(C.Z_CRITICAL < muS && muS < mu3 && mu3 < 1.0, 'ordering above lens');

  // Double-well ratio
  const ratio = mu2 / mu1;
  const rel = Math.abs(ratio - C.PHI) / C.PHI;
  assert(rel < 1e-12, 'μ2/μ1 should equal φ (within fp rounding)');

  // Barrier proximity
  const barrier = C.MU_BARRIER;
  const diff = Math.abs(barrier - (1/C.PHI));
  // Default is exact; if user explicitly overrides QAPL_MU_P, relax check
  const hasOverride = (typeof process !== 'undefined' && process.env && process.env.QAPL_MU_P);
  const tol = hasOverride ? 2e-3 : 1e-12;
  assert(diff < tol, `barrier not aligned with φ^{-1}: Δ=${diff}, tol=${tol}`);
  console.log('μ thresholds tests passed');
})();
