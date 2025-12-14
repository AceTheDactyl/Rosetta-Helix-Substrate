const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };

// Compare operator weights with and without blending for z above lens
const path = require('path');

function run(blend) {
  const modPath = path.join(__dirname, '..', 'src', 'quantum_apl_engine.js');
  delete require.cache[require.resolve(modPath)];
  const { QuantumAPL } = require(modPath);
  const engine = new QuantumAPL({ blendPiEnabled: blend });
  const z = require('../src/constants').Z_CRITICAL + 0.02;
  engine.z = z;
  const hints = engine.helixAdvisor.describe(z);
  const scalar = { Gs:1, Cs:1, Rs:1, kappa:1, tau:1, theta:1, delta:1, alpha:1, Omega:0.5 };
  const wPlus = engine.computeOperatorWeight('+', scalar, hints);
  const wBoundary = engine.computeOperatorWeight('()', scalar, hints);
  return { wPlus, wBoundary };
}

(function testBlendEffects() {
  const off = run(false);
  const on = run(true);
  // With blending on above lens, '+' should be boosted, '()' damped
  assert(on.wPlus >= off.wPlus - 1e-9, `+ not boosted: ${on.wPlus} < ${off.wPlus}`);
  assert(on.wBoundary <= off.wBoundary + 1e-9, `() not damped: ${on.wBoundary} > ${off.wBoundary}`);
  console.log('Blend flag tests passed');
})();

