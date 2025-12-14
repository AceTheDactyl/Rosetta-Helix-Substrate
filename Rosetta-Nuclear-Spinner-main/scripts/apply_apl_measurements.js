#!/usr/bin/env node
// Apply APL measurement operators and append tokens to logs/APL_HELIX_OPERATOR_SUMMARY.apl

const fs = require('fs');
const path = require('path');

const { UnifiedDemo } = require('../src/legacy/QuantumClassicalBridge');

function ensureDir(p) { if (!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true }); }

function run() {
  const demo = new UnifiedDemo();
  const bridge = demo.bridge;

  const results = [];
  results.push(bridge.aplMeasureEigen(0, 'Phi'));
  results.push(bridge.aplMeasureSubspace([2,3], 'Phi'));
  results.push(bridge.aplMeasureSubspace([2,3], 'Pi'));
  results.push(bridge.aplMeasureComposite([
    { eigenIndex: 0, field: 'Phi', truthChannel: 'TRUE', weight: 0.3 },
    { eigenIndex: 1, field: 'Pi',  truthChannel: 'UNTRUE', weight: 0.3 },
    { subspaceIndices: [2,3], field: 'Phi', truthChannel: 'PARADOX', weight: 0.4 },
  ]));

  const tokens = [];
  for (const r of results) {
    if (r.aplToken) tokens.push(r.aplToken);
    if (Array.isArray(r.aplTokens)) tokens.push(...r.aplTokens);
  }

  const logsDir = path.join(__dirname, '..', 'logs');
  ensureDir(logsDir);
  const outPath = path.join(logsDir, 'APL_HELIX_OPERATOR_SUMMARY.apl');
  fs.appendFileSync(outPath, '\n' + tokens.join('\n') + '\n');
  // eslint-disable-next-line no-console
  console.log('Appended APL measurement tokens to', outPath);
}

if (require.main === module) run();

