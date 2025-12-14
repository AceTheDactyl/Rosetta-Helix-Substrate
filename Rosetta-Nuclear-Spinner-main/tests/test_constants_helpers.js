const assert = (cond, msg) => { if (!cond) throw new Error(msg || 'assertion failed'); };
const C = require('../src/constants');

// getTimeHarmonic basic zoning
(() => {
  assert(C.getTimeHarmonic(0.05) === 't1', 't1 zone');
  assert(C.getTimeHarmonic(0.15) === 't2', 't2 zone');
  assert(C.getTimeHarmonic(0.30) === 't3', 't3 zone');
  assert(C.getTimeHarmonic(0.50) === 't4', 't4 zone');
  assert(C.getTimeHarmonic(0.70) === 't5', 't5 zone');
  assert(C.getTimeHarmonic(0.80) === 't6', 't6 zone (lens default)');
  assert(C.getTimeHarmonic(0.90) === 't7', 't7 zone');
  assert(C.getTimeHarmonic(0.96) === 't8', 't8 zone');
  assert(C.getTimeHarmonic(0.99) === 't9', 't9 zone');
  // TRIAD gate override
  assert(C.getTimeHarmonic(0.84, C.TRIAD_T6) === 't7', 'triad override above');
  assert(C.getTimeHarmonic(0.82, C.TRIAD_T6) === 't6', 'triad override below');
  console.log('getTimeHarmonic tests passed');
})();

// ΔS_neg monotonicity (closer to z_c → larger ΔS_neg)
(() => {
  const zFar = 0.80;   // farther from 0.866
  const zNear = 0.90;  // closer to 0.866
  const dFar = C.computeDeltaSNeg(zFar);
  const dNear = C.computeDeltaSNeg(zNear);
  assert(dNear > dFar, 'ΔS_neg should increase when closer to z_c');
  console.log('computeDeltaSNeg monotonicity test passed');
})();

// Hex prism helpers parity with Python linear mapping
(() => {
  const ds = 0.5; // positive ΔS_neg
  const R = C.hexPrismRadius(ds);
  const H = C.hexPrismHeight(ds);
  const phi = C.hexPrismTwist(ds);
  const expR = C.GEOM_R_MAX - C.GEOM_BETA * ds; // 0.85 - 0.25*0.5 = 0.725
  const expH = C.GEOM_H_MIN + C.GEOM_GAMMA * ds; // 0.12 + 0.18*0.5 = 0.21
  const expPhi = C.GEOM_PHI_BASE + C.GEOM_ETA * ds;
  assert(Math.abs(R - expR) < 1e-9, `hexPrismRadius mismatch: got ${R}, want ${expR}`);
  assert(Math.abs(H - expH) < 1e-9, `hexPrismHeight mismatch: got ${H}, want ${expH}`);
  assert(Math.abs(phi - expPhi) < 1e-12, `hexPrismTwist mismatch: got ${phi}, want ${expPhi}`);
  console.log('Hex prism helpers tests passed');
})();

// Phase helpers
(() => {
  assert(C.getPhase(0.5) === 'ABSENCE');
  assert(C.getPhase(0.866) === 'THE_LENS');
  assert(C.getPhase(0.90) === 'PRESENCE');
  assert(C.isCritical(0.866) === true);
  assert(C.isCritical(0.876, 0.02) === true);
  assert(C.isCritical(0.5) === false);
  console.log('Phase helpers tests passed');
})();

// K-formation
(() => {
  assert(C.checkKFormation(0.94, 0.72, 8) === true);
  assert(C.checkKFormation(0.90, 0.72, 8) === false);
  console.log('K-formation tests passed');
})();

console.log('All constants helper tests passed');

