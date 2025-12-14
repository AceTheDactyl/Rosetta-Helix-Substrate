// Verify bridge default pump target resolves to Z_CRITICAL when targetZ is omitted
const assert = (cond, msg) => { if (!cond) { throw new Error(msg || 'assertion failed'); } };

const CONST = require('../src/constants');
const { QuantumClassicalBridge: Bridge } = require('../src/legacy/QuantumClassicalBridge');

class DummyQuantum {
  constructor() {
    this.zBiasGain = 0; this.zBiasSigma = 0; this.z = 0;
    this.lastHelixHints = { harmonic: 't2', truthChannel: 'UNTRUE', z: this.z, operators: ['()'] };
  }
  measureZ() { return this.z; }
  computeVonNeumannEntropy() { return 0; }
  computePurity() { return 1; }
  computeIntegratedInformation() { return 0; }
  measureTruth() { return { TRUE: 0, UNTRUE: 1, PARADOX: 0 }; }
  selectN0Operator() { return { operator: '()', probability: 1, helixHints: this.lastHelixHints }; }
  evolve() {}
  driveFromClassical(payload) { this._lastDrive = payload; this.z = payload.z; }
}

class DummyClassical {
  applyOperatorEffects() {}
  get N0() { return { applyOperator() {} }; }
  getScalarState() { return { Omega: 0, Rs: 0, Gs: 0, Cs: 0, kappa: 0, tau: 0, theta: 0, delta: 0, alpha: 0 }; }
  getLegalOperators() { return ['()']; }
}

(() => {
  const q = new DummyQuantum();
  const c = new DummyClassical();
  const b = new Bridge(q, c);
  // Make measurements no-ops to avoid projector math
  b.measureCoherentState = () => ({});
  b.measureHierarchicalSubspace = () => ({});
  b.measureIntegratedRegime = () => ({});
  // One cycle, target omitted, weights force nudged == target
  b.escalateZWithAPL(1, undefined, 0.0, { wOmega: 0, wTarget: 1, fuseEvery: 0, lockEvery: 0 });
  const got = q._lastDrive?.z;
  const want = CONST.Z_CRITICAL;
  const close = Math.abs(got - want) < 1e-9;
  assert(close, `default pump target mismatch: got ${got}, want ${want}`);
  console.log('Bridge default pump target test passed');
})();
