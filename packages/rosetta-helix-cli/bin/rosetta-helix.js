#!/usr/bin/env node
/* Standalone Rosetta-Helix CLI (published to npm) */
const { spawn } = require('child_process');
const { existsSync } = require('fs');
const { join } = require('path');

function sh(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: 'inherit', shell: false, ...opts });
    p.on('exit', code => code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}`)));
  });
}

function venvBin(name) {
  const v = '.venv';
  const unix = join(process.cwd(), v, 'bin', name);
  const win = join(process.cwd(), v, 'Scripts', name + '.exe');
  if (existsSync(unix)) return unix;
  if (existsSync(win)) return win;
  return name;
}

async function setup() {
  const py = process.platform === 'win32' ? 'python' : 'python3';
  await sh(py, ['-m', 'venv', '.venv']);
  const pip = venvBin('pip');
  await sh(pip, ['install', '--upgrade', 'pip', 'setuptools', 'wheel']);
  if (existsSync('requirements.txt')) await sh(pip, ['install', '-r', 'requirements.txt']);
  await sh(pip, ['install', '-e', '.']);
  if (existsSync('kira-local-system/requirements.txt')) await sh(pip, ['install', '-r', 'kira-local-system/requirements.txt']);
  if (existsSync('requirements.spinner.txt')) await sh(pip, ['install', '-r', 'requirements.spinner.txt']);
}

async function initRepo(dir) {
  const target = dir || 'Rosetta-Helix-Substrate';
  const repo = 'https://github.com/AceTheDactyl/Rosetta-Helix-Substrate.git';
  await sh('git', ['clone', repo, target]);
  process.chdir(target);
  await setup();
}

async function runKira() {
  const pathLocal = 'kira-local-system/kira_server.py';
  const kiraServer = venvBin('kira-server');
  if (kiraServer !== 'kira-server') {
    await sh(kiraServer, ['--host', '0.0.0.0', '--port', '5000']);
    return;
  }
  if (!existsSync(pathLocal)) {
    console.error('kira-local-system/kira_server.py not found. Run inside Rosetta-Helix-Substrate repo.');
    process.exit(1);
  }
  const py = venvBin('python');
  await sh(py, [pathLocal]);
}

async function runViz() {
  const py = venvBin('python');
  if (!existsSync('visualization_server.py')) {
    console.error('visualization_server.py not found. Run inside Rosetta-Helix-Substrate repo.');
    process.exit(1);
  }
  await sh(py, ['visualization_server.py', '--port', '8765']);
}

async function runHelixTrain() {
  const helix = venvBin('helix');
  try {
    await sh(helix, ['train', '--config', 'configs/full.yaml']);
  } catch (_) {
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

async function runCompose(target) {
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
  const arg = process.argv[3];
  try {
    if (cmd === 'init') await initRepo(arg);
    if (cmd === 'setup') await setup();
    else if (cmd === 'kira') await runKira();
    else if (cmd === 'viz') await runViz();
    else if (cmd === 'helix:train') await runHelixTrain();
    else if (cmd === 'helix:nightly') await runHelixNightly();
    else if (cmd === 'smoke') await runSmoke();
    else if (cmd === 'api:test') await runApiTests();
    else if (cmd.startsWith('docker:')) await runCompose(cmd);
    else {
      console.log(`Usage: rosetta-helix <command>\n` +
        `  init [dir]      Clone repo and run setup\n` +
        `  setup           Create .venv and install deps\n` +
        `  kira            Start KIRA server (port 5000)\n` +
        `  viz             Start Visualization server (port 8765)\n` +
        `  helix:train     Run helix training\n` +
        `  helix:nightly   Run nightly training\n` +
        `  smoke           Run smoke tests\n` +
        `  api:test        Run API contract tests\n` +
        `  docker:build|up|down|logs  Compose helpers\n` +
        `\nRun inside a Rosetta-Helix-Substrate checkout.`);
      process.exit(1);
    }
  } catch (err) {
    console.error(err.message || err);
    process.exit(1);
  }
})();
