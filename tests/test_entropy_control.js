// Simple smoke test: entropy control nudges entropy toward a target
const assert = (cond, msg) => { if (!cond) throw new Error(msg || 'assert'); };

const { QuantumAPL } = require('../src/quantum_apl_engine');
const CONST = require('../src/constants');

(function testEntropyControl() {
  const engineBase = new QuantumAPL();
  const engineCtrl = new QuantumAPL({ entropyCtrlEnabled: true, entropyCtrlGain: 0.3, entropyCtrlCoeff: 0.6 });

  // Measure initial entropy
  const S0_base = engineBase.computeVonNeumannEntropy();
  const S0_ctrl = engineCtrl.computeVonNeumannEntropy();

  // Target near lens
  const targetZ = CONST.Z_CRITICAL;

  // Apply a few bias steps without and with entropy control
  for (let i = 0; i < 5; i++) {
    engineBase.applyZBias(targetZ, 0.05, 0.18);
    engineCtrl.applyZBias(targetZ, 0.05, 0.18);
  }

  const S_base = engineBase.computeVonNeumannEntropy();
  const S_ctrl = engineCtrl.computeVonNeumannEntropy();

  // Expect the controlled engine to be at least as low entropy as baseline
  assert(Number.isFinite(S_base) && Number.isFinite(S_ctrl), 'entropy should be finite');
  assert(S_ctrl <= S_base + 1e-9, `entropy control did not reduce entropy: S_ctrl=${S_ctrl}, S_base=${S_base}`);

  console.log('Entropy control smoke test passed');
})();

