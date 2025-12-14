const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };
const { spawnSync } = require('node:child_process');

// When QAPL_GEOM_SIGMA is set, GEOM_SIGMA should equal that value and differ from LENS_SIGMA if different
const code = `
  process.env.QAPL_LENS_SIGMA = '36';
  process.env.QAPL_GEOM_SIGMA = '12';
  const C = require('./src/constants');
  process.stdout.write(JSON.stringify({ lens: C.LENS_SIGMA, geom: C.GEOM_SIGMA }));
`;

const out = spawnSync(process.execPath, ['-e', code], { env: { ...process.env } });
if (out.status !== 0) throw new Error(out.stderr.toString());
const obj = JSON.parse(out.stdout.toString());
assert(obj.geom === 12, 'GEOM_SIGMA should match env override');
assert(obj.geom !== obj.lens, 'GEOM_SIGMA should differ from LENS_SIGMA when set');
console.log('Geom sigma override test passed');

