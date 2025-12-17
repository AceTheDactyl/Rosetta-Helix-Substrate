#!/usr/bin/env node
/*
 Rosetta-Helix Node Wrapper CLI
 - Orchestrates Python virtualenv setup and common commands via npm
*/
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const isWin = process.platform === 'win32';

function venvPath(...segments) {
  return path.resolve('.venv', ...segments);
}

function hasVenvPython() {
  const candidates = [
    venvPath('bin', 'python'),
    venvPath('Scripts', 'python.exe'),
    venvPath('Scripts', 'python'),
  ];
  return candidates.some((candidate) => fs.existsSync(candidate));
}

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
  const unix = venvPath('bin', name);
  const winExe = venvPath('Scripts', name + '.exe');
  const win = venvPath('Scripts', name);
  if (fs.existsSync(unix)) return unix;
  if (fs.existsSync(winExe)) return winExe;
  if (fs.existsSync(win)) return win;
  if (name === 'python') {
    return isWin ? 'python' : 'python3';
  }
  return name;
}

async function ensureVenv() {
  if (hasVenvPython()) return;
  console.log('No .venv detected. Running `rosetta-helix setup` (python -m venv .venv) ...');
  await setup();
}

async function setup() {
  const py = process.platform === 'win32' ? 'python' : 'python3';
  await sh(py, ['-m', 'venv', '.venv']);
  const pip = venvBin('pip');
  // Upgrade basics
  await sh(pip, ['install', '--upgrade', 'pip', 'setuptools', 'wheel']);
  // Install requirements
  if (fs.existsSync('requirements.txt')) await sh(pip, ['install', '-r', 'requirements.txt']);
  await sh(pip, ['install', '-e', '.']);
  if (fs.existsSync('kira-local-system/requirements.txt')) await sh(pip, ['install', '-r', 'kira-local-system/requirements.txt']);
  if (fs.existsSync('requirements.spinner.txt')) await sh(pip, ['install', '-r', 'requirements.spinner.txt']);
}

async function runKiraServer() {
  await ensureVenv();
  const py = venvBin('python');
  const kiraDir = path.join(process.cwd(), 'kira-local-system');
  const script = path.join(kiraDir, 'kira_server.py');
  if (!fs.existsSync(script)) {
    console.error('kira-local-system/kira_server.py not found. Run this from the repo root or pass --dir.');
    process.exit(1);
  }
  console.log('Starting KIRA server (kira-local-system/kira_server.py)...');
  console.log('HTTP API: http://localhost:5000');
  console.log('Interface: docs/kira/index.html');
  await sh(py, ['kira_server.py'], { cwd: kiraDir });
}

async function runUnified() {
  console.log('Unified command now launches kira_server.py (legacy alias).');
  await runKiraServer();
}

async function runKira() {
  await runKiraServer();
}

function printLandingInfo() {
  const landing = path.join(process.cwd(), 'docs', 'index.html');
  const kiraHtml = path.join(process.cwd(), 'docs', 'kira', 'index.html');
  console.log('Visualization server removed. Use the tracked docs bundle instead:\n');
  console.log(`  Landing page: ${landing}`);
  console.log(`  KIRA UI:      ${kiraHtml}`);
  console.log('\nStart the backend with `npx rosetta-helix start` or `make kira-server`, then open the HTML files above.');
}

async function runViz() {
  printLandingInfo();
}

async function runVizSync() {
  console.log('viz:sync-gh now mirrors the repo-local docs bundle. Nothing to download.');
  printLandingInfo();
}

async function runHelixTrain() {
  await ensureVenv();
  const helix = venvBin('helix');
  try {
    await sh(helix, ['train', '--config', 'configs/full.yaml']);
  } catch (e) {
    const py = venvBin('python');
    await sh(py, ['train_helix.py']);
  }
}

async function runHelixNightly() {
  await ensureVenv();
  const py = venvBin('python');
  await sh(py, ['nightly_training_runner.py']);
}

async function runSmoke() {
  await ensureVenv();
  const py = venvBin('python');
  await sh(py, ['-m', 'pytest', '-q', 'tests/smoke']);
}

async function runApiTests() {
  await ensureVenv();
  const py = venvBin('python');
  await sh(py, ['-m', 'pytest', '-q', 'tests/api']);
}

async function runStart() {
  console.log('Starting KIRA server...');
  await runKiraServer();
}

async function runHealth() {
  console.log(JSON.stringify({
    kira: 'http://localhost:5000/api/health',
    landing: path.join(process.cwd(), 'docs', 'index.html')
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
  if (fs.existsSync('.venv')) {
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
    'docker:up': ['compose', 'up', '-d', 'kira'],
    'docker:down': ['compose', 'down'],
    'docker:logs': ['compose', 'logs', '-f', '--tail=200']
  }[target];
  if (!args) return;
  await sh('docker', args);
}

function parseOptions(args) {
  const opts = { _: [] };
  for (let i = 0; i < args.length; i++) {
    const token = args[i];
    if (!token.startsWith('--')) {
      opts._.push(token);
      continue;
    }
    const eq = token.indexOf('=');
    if (eq !== -1) {
      const key = token.slice(2, eq);
      opts[key] = token.slice(eq + 1) || true;
      continue;
    }
    const key = token.slice(2);
    const next = args[i + 1];
    if (next && !next.startsWith('--')) {
      opts[key] = next;
      i++;
    } else {
      opts[key] = true;
    }
  }
  return opts;
}

function extractCommand(argv) {
  const remainder = [];
  let cmd = '';
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (token.startsWith('--')) {
      remainder.push(token);
      const next = argv[i + 1];
      if (next && !next.startsWith('--')) {
        remainder.push(next);
        i++;
      }
      continue;
    }
    if (!cmd) {
      cmd = token;
    } else {
      remainder.push(token);
    }
  }
  return { cmd, remainder };
}

(async () => {
  const { cmd, remainder } = extractCommand(process.argv.slice(2));
  const options = parseOptions(remainder);
  const repoDir = options.dir ? path.resolve(options.dir) : process.cwd();
  if (options.dir) {
    if (!fs.existsSync(repoDir)) {
      console.error(`Directory not found for --dir: ${options.dir}`);
      process.exit(1);
    }
    process.chdir(repoDir);
  }

  try {
    if (cmd === 'setup') await setup();
    else if (cmd === 'unified') await runUnified();
    else if (cmd === 'kira') await runKira();
    else if (cmd === 'viz') await runViz();
    else if (cmd === 'viz:sync-gh') await runVizSync();
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
        `  unified         Start KIRA server (legacy alias)\n` +
        `  start           Start KIRA server\n` +
        `  kira            Start KIRA server\n` +
        `  viz             Show local landing paths (static bundle)\n` +
        `  viz:sync-gh     Legacy alias (prints landing info)\n` +
        `  health          Check service health endpoints\n` +
        `  doctor          Run environment checks\n` +
        `  helix:train     Run helix training\n` +
        `  helix:nightly   Run nightly training\n` +
        `  smoke           Run smoke tests\n` +
        `  api:test        Run API contract tests\n` +
        `  docker:build|up|down|logs  Compose helpers\n\n` +
        `Web Interfaces:\n` +
        `  docs/index.html            (Landing)\n` +
        `  docs/kira/index.html       (KIRA UI)\n`);
      process.exit(1);
    }
  } catch (err) {
    console.error(err.message || err);
    process.exit(1);
  }
})();
