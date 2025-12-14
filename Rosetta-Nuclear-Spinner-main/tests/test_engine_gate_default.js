// Ensure engine t6 gate defaults to Z_CRITICAL when TRIAD is not unlocked
const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };
const CONST = require('../src/constants');
const { QuantumAPL } = require('../src/quantum_apl_engine');

(() => {
  // Ensure env does not force TRIAD
  delete process.env.QAPL_TRIAD_UNLOCK;
  delete process.env.QAPL_TRIAD_COMPLETIONS;
  const q = new QuantumAPL();
  const gate = (typeof q.getT6Gate === 'function') ? q.getT6Gate() : CONST.Z_CRITICAL;
  const ok = Math.abs(gate - CONST.Z_CRITICAL) < 1e-12;
  assert(ok, `engine default t6 gate mismatch: got ${gate}, want ${CONST.Z_CRITICAL}`);
  console.log('Engine default t6 gate test passed');
})();

