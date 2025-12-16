#!/usr/bin/env node
/*
 Rosetta-Helix Node Wrapper CLI
 - Orchestrates Python virtualenv setup and common commands via npm
*/
const { spawn } = require('child_process');
const { existsSync } = require('fs');
const { join } = require('path');

function sh(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    // Preserve ANTHROPIC_API_KEY in environment
    const env = {
      ...process.env,
      ...(opts.env || {})
    };
    const p = spawn(cmd, args, { stdio: 'inherit', shell: false, env, ...opts });
    p.on('exit', code => code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}`)));
  });
}

function venvBin(name) {
  const v = '.venv';
  const unix = join(v, 'bin', name);
  const win = join(v, 'Scripts', name + '.exe');
  if (existsSync(unix)) return unix;
  if (existsSync(win)) return win;
  return name; // fallback to PATH
}

async function setup() {
  const py = process.platform === 'win32' ? 'python' : 'python3';
  await sh(py, ['-m', 'venv', '.venv']);
  const pip = venvBin('pip');
  // Upgrade basics
  await sh(pip, ['install', '--upgrade', 'pip', 'setuptools', 'wheel']);
  // Install requirements
  if (existsSync('requirements.txt')) await sh(pip, ['install', '-r', 'requirements.txt']);
  await sh(pip, ['install', '-e', '.']);
  if (existsSync('kira-local-system/requirements.txt')) await sh(pip, ['install', '-r', 'kira-local-system/requirements.txt']);
  if (existsSync('requirements.spinner.txt')) await sh(pip, ['install', '-r', 'requirements.spinner.txt']);
}

async function runUnified() {
  const py = venvBin('python');
  if (existsSync('unified_rosetta_server.py')) {
    console.log('Starting Unified Rosetta Server...');
    console.log('HTTP API: http://localhost:5000');
    console.log('WebSocket: ws://localhost:8765');
    console.log('Web Interface: http://localhost:5000/unified');
    await sh(py, ['unified_rosetta_server.py']);
  } else {
    console.error('unified_rosetta_server.py not found in current directory.');
    console.error('Run this command from the Rosetta-Helix-Substrate repo root.');
    process.exit(1);
  }
}

// Legacy functions for backward compatibility
async function runKira() {
  console.log('Note: Running unified server (kira command is deprecated)');
  await runUnified();
}

async function runViz() {
  console.log('Note: Running unified server (viz command is deprecated)');
  await runUnified();
}

async function runHelixTrain() {
  const helix = venvBin('helix');
  try {
    await sh(helix, ['train', '--config', 'configs/full.yaml']);
  } catch (e) {
    const py = venvBin('python');
    await sh(py, ['train_helix.py']);
  }
}

async function runHelixNightly() {
  const py = venvBin('python');
  await sh(py, ['nightly_training_runner.py']);
}

async function runSmoke() {
  const py = venvBin('python');
  await sh(py, ['-m', 'pytest', '-q', 'tests/smoke']);
}

async function runApiTests() {
  const py = venvBin('python');
  await sh(py, ['-m', 'pytest', '-q', 'tests/api']);
}

async function runStart() {
  console.log('Starting Unified Rosetta Server...');
  await runUnified();
}

async function runHealth() {
  console.log(JSON.stringify({
    kira: 'http://localhost:5000/api/health',
    viz: 'http://localhost:8765/state'
  }, null, 2));
}

async function runDoctor() {
  console.log('Running environment checks...');
  const checks = [];

  // Check Python
  try {
    await sh('python3', ['--version']);
    checks.push('✓ Python 3 available');
  } catch {
    checks.push('✗ Python 3 not found');
  }

  // Check venv
  if (existsSync('.venv')) {
    checks.push('✓ Virtual environment exists');
  } else {
    checks.push('✗ Virtual environment missing (run: npx rosetta-helix setup)');
  }

  // Check ANTHROPIC_API_KEY
  if (process.env.ANTHROPIC_API_KEY) {
    checks.push('✓ ANTHROPIC_API_KEY is set');
  } else {
    checks.push('⚠ ANTHROPIC_API_KEY not set (optional)');
  }

  checks.forEach(c => console.log(c));
}

async function runCompose(target) {
  // shim docker compose via npm scripts
  const args = {
    'docker:build': ['compose', 'build'],
    'docker:up': ['compose', 'up', '-d', 'kira', 'viz'],
    'docker:down': ['compose', 'down'],
    'docker:logs': ['compose', 'logs', '-f', '--tail=200']
  }[target];
  if (!args) return;
  await sh('docker', args);
}

(async () => {
  const cmd = process.argv[2] || '';
  try {
    if (cmd === 'setup') await setup();
    else if (cmd === 'unified') await runUnified();
    else if (cmd === 'kira') await runKira();
    else if (cmd === 'viz') await runViz();
    else if (cmd === 'start' || cmd === 'star') await runStart();
    else if (cmd === 'health') await runHealth();
    else if (cmd === 'doctor') await runDoctor();
    else if (cmd === 'helix:train') await runHelixTrain();
    else if (cmd === 'helix:nightly') await runHelixNightly();
    else if (cmd === 'smoke') await runSmoke();
    else if (cmd === 'api:test') await runApiTests();
    else if (cmd.startsWith('docker:')) await runCompose(cmd);
    else {
      console.log(`Usage: rosetta-helix <command>\n` +
        `  setup           Create .venv and install deps\n` +
        `  unified         Start Unified Rosetta Server (NEW!)\n` +
        `  start           Start Unified server (alias for unified)\n` +
        `  kira            [Deprecated] Use 'unified' instead\n` +
        `  viz             [Deprecated] Use 'unified' instead\n` +
        `  health          Check service health endpoints\n` +
        `  doctor          Run environment checks\n` +
        `  helix:train     Run helix training\n` +
        `  helix:nightly   Run nightly training\n` +
        `  smoke           Run smoke tests\n` +
        `  api:test        Run API contract tests\n` +
        `  docker:build|up|down|logs  Compose helpers\n\n` +
        `Web Interface: http://localhost:5000/unified (after starting)\n`);
      process.exit(1);
    }
  } catch (err) {
    console.error(err.message || err);
    process.exit(1);
  }
})();
