const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };
const { spawnSync } = require('node:child_process');

// With QAPL_MU_P override, barrier_eq_phi_inv should be false in invariants()
const code = `
  process.env.QAPL_MU_P = '0.600';
  const C = require('./src/constants');
  const inv = C.invariants();
  process.stdout.write(JSON.stringify(inv));
`;

const out = spawnSync(process.execPath, ['-e', code], { env: { ...process.env } });
if (out.status !== 0) throw new Error(out.stderr.toString());
const inv = JSON.parse(out.stdout.toString());
assert(inv.barrier_eq_phi_inv === false, 'barrier_eq_phi_inv should be false when MU_P overridden');
console.log('MU override invariants test passed');

