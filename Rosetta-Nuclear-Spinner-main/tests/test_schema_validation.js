const path = require('path');
const fs = require('fs');
const Ajv = require('ajv');

const ajv = new Ajv({ allErrors: true });

function loadSchema(rel) {
  const p = path.join(__dirname, '..', rel);
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function assertOk(ok, validate) {
  if (!ok) {
    const msg = ajv.errorsText(validate.errors, { separator: '\n' });
    throw new Error(`Schema validation failed:\n${msg}`);
  }
}

// Geometry sidecar tests
(function testGeometrySidecar() {
  const geomSchema = loadSchema('schemas/geometry-sidecar.schema.json');
  const validate = ajv.compile(geomSchema);

  // Build via JS module if available; else synthesize minimal valid example.
  let sidecar;
  try {
    const hexPrism = require('../src/hex_prism');
    sidecar = hexPrism.exportGeometrySidecar(0.866, '1.0.0');
  } catch {
    // Fallback synthetic fixture (keeps test independent)
    sidecar = {
      version: '1.0.0',
      z: 0.8660254,
      delta_S_neg: 0.5,
      vertices: Array.from({ length: 63 }, (_, id) => ({ id, x: 0, y: 0, z: id < 32 ? 0.1 : -0.1 })),
      geometry: { R: 0.72, H: 0.21, phi: 0.05 },
      constants: {
        Z_CRITICAL: 0.8660254037844386,
        GEOM_SIGMA: 0.12,
        GEOM_R_MAX: 0.85,
        GEOM_BETA: 0.25,
        GEOM_H_MIN: 0.12,
        GEOM_GAMMA: 0.18,
        GEOM_PHI_BASE: 0.0,
        GEOM_ETA: Math.PI / 12
      }
    };
  }

  const ok = validate(sidecar);
  assertOk(ok, validate);

  // Negative cases
  const brokenLen = { ...sidecar, vertices: sidecar.vertices.slice(0, 10) };
  const okLen = validate(brokenLen);
  if (okLen) throw new Error('Expected failure for wrong vertices length');

  const brokenBounds = { ...sidecar, delta_S_neg: 1.5 };
  const okBounds = validate(brokenBounds);
  if (okBounds) throw new Error('Expected failure for delta_S_neg out of [0,1]');

  const missingGeom = { ...sidecar };
  delete missingGeom.geometry;
  const okMissing = validate(missingGeom);
  if (okMissing) throw new Error('Expected failure for missing geometry');

  console.log('✓ geometry-sidecar.schema.json validation passed (positive and negative cases)');
})();

// APL bundle tests
(function testAPLBundle() {
  const bundleSchema = loadSchema('schemas/apl-bundle.schema.json');
  const validate = ajv.compile(bundleSchema);

  const good = [
    'Φ:T(ϕ_0)TRUE@2',
    'π:Π(subspace)UNTRUE@3',
    'Φ:Π(my_subspace)PARADOX@4'
  ];
  const okGood = validate(good);
  assertOk(okGood, validate);

  const bad = [
    'Φ:T(ϕ_0)@2',              // missing truth
    'π:Π(subspace)UNTRUE',     // missing @Tier
    'X:Π(subspace)TRUE@2'      // bad channel symbol
  ];
  const okBad = validate(bad);
  if (okBad) throw new Error('Expected failure for malformed APL bundle');

  console.log('✓ apl-bundle.schema.json validation passed (positive and negative cases)');
})();
