const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };
const { spawnSync } = require('node:child_process');

// Ensure that when QAPL_GEOM_SIGMA is unset, GEOM_SIGMA === LENS_SIGMA
const code = `
  delete process.env.QAPL_GEOM_SIGMA;
  delete process.env.QAPL_LENS_SIGMA;
  const C = require('./src/constants');
  process.stdout.write(JSON.stringify({ lens: C.LENS_SIGMA, geom: C.GEOM_SIGMA }));
`;

const out = spawnSync(process.execPath, ['-e', code], { env: { ...process.env } });
if (out.status !== 0) throw new Error(out.stderr.toString());
const obj = JSON.parse(out.stdout.toString());
assert(obj.geom === obj.lens, `GEOM_SIGMA (${obj.geom}) !== LENS_SIGMA (${obj.lens}) when unset`);
console.log('Geom sigma cohesion test passed');

