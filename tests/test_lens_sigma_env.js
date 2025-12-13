const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };
const { spawnSync } = require('node:child_process');

// Verify that QAPL_LENS_SIGMA env var controls exported LENS_SIGMA
const code = `
  process.env.QAPL_LENS_SIGMA = '50';
  const C = require('./src/constants');
  process.stdout.write(String(C.LENS_SIGMA));
`;

const out = spawnSync(process.execPath, ['-e', code], { env: { ...process.env } });
if (out.status !== 0) throw new Error(out.stderr.toString());
const val = parseFloat(out.stdout.toString());
assert(Math.abs(val - 50.0) < 1e-12, `LENS_SIGMA env override failed: ${val}`);
console.log('Lens sigma env test passed');
