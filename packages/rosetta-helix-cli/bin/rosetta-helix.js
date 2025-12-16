#!/usr/bin/env node
/* Standalone Rosetta-Helix CLI (published to npm) */
const { spawn } = require('child_process');
const { existsSync, readFileSync } = require('fs');
const { join } = require('path');
const http = require('http');

function sh(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: 'inherit', shell: false, ...opts });
    p.on('exit', code => code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}`)));
  });
}

function venvBin(name, opts = {}) {
  const root = opts.venvRoot || process.cwd();
  const v = '.venv';
  const unix = join(root, v, 'bin', name);
  const win = join(root, v, 'Scripts', name + '.exe');
  if (existsSync(unix)) return unix;
  if (existsSync(win)) return win;
  return name;
}

function parseFlags(argv) {
  const out = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('-')) { out._.push(a); continue; }
    const [k, v] = a.includes('=') ? a.split(/=(.*)/) : [a, argv[i + 1] && !argv[i + 1].startsWith('-') ? argv[++i] : true];
    switch (k) {
      case '--dir': out.dir = v; break;
      case '--venv': out.venvRoot = v; break;
      case '--python': out.python = v; break;
      case '--host': out.host = v; break;
      case '--port': out.port = v; break;
      case '--verbose': case '-v': out.verbose = true; break;
      case '--help': case '-h': out.help = true; break;
      default: out[k.replace(/^--?/, '')] = v === true ? true : v; break;
    }
  }
  return out;
}

async function setup(flags = {}) {
  if (flags.dir) process.chdir(flags.dir);
  const py = flags.python || (process.platform === 'win32' ? 'python' : 'python3');
  await sh(py, ['-m', 'venv', '.venv']);
  const pip = venvBin('pip', { venvRoot: process.cwd() });
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

async function runKira(flags = {}) {
  if (flags.dir) process.chdir(flags.dir);
  const host = flags.host || '0.0.0.0';
  const port = String(flags.port || 5000);
  const pathLocal = 'kira-local-system/kira_server.py';
  const kiraServer = venvBin('kira-server', { venvRoot: process.cwd() });
  if (kiraServer !== 'kira-server') {
    await sh(kiraServer, ['--host', host, '--port', port]);
    return;
  }
  if (!existsSync(pathLocal)) {
    console.error('kira-local-system/kira_server.py not found. Run inside Rosetta-Helix-Substrate repo.');
    process.exit(1);
  }
  const py = flags.python || venvBin('python', { venvRoot: process.cwd() });
  await sh(py, [pathLocal, '--host', host, '--port', port]);
}

async function runViz(flags = {}) {
  if (flags.dir) process.chdir(flags.dir);
  const py = flags.python || venvBin('python', { venvRoot: process.cwd() });
  if (!existsSync('visualization_server.py')) {
    console.error('visualization_server.py not found. Run inside Rosetta-Helix-Substrate repo.');
    process.exit(1);
  }
  const port = String(flags.port || 8765);
  await sh(py, ['visualization_server.py', '--port', port]);
}

async function runHelixTrain(flags = {}) {
  if (flags.dir) process.chdir(flags.dir);
  const helix = venvBin('helix', { venvRoot: process.cwd() });
  try {
    await sh(helix, ['train', '--config', 'configs/full.yaml']);
  } catch (_) {
    const py = flags.python || venvBin('python', { venvRoot: process.cwd() });
    await sh(py, ['train_helix.py']);
  }
}

async function runHelixNightly(flags = {}) {
  if (flags.dir) process.chdir(flags.dir);
  const py = flags.python || venvBin('python', { venvRoot: process.cwd() });
  await sh(py, ['nightly_training_runner.py']);
}

async function runSmoke(flags = {}) {
  if (flags.dir) process.chdir(flags.dir);
  const py = flags.python || venvBin('python', { venvRoot: process.cwd() });
  await sh(py, ['-m', 'pytest', '-q', 'tests/smoke']);
}

async function runApiTests(flags = {}) {
  if (flags.dir) process.chdir(flags.dir);
  const py = flags.python || venvBin('python', { venvRoot: process.cwd() });
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

async function startBoth(flags = {}) {
  if (flags.dir) process.chdir(flags.dir);
  const host = flags.host || '0.0.0.0';
  const kiraPort = String(flags.kiraPort || flags.port || 5000);
  const vizPort = String(flags.vizPort || 8765);
  const py = flags.python || venvBin('python', { venvRoot: process.cwd() });
  const procs = [];
  if (!existsSync('kira-local-system/kira_server.py')) {
    console.error('kira-local-system/kira_server.py not found. Run inside repo or pass --dir.');
    process.exit(1);
  }
  if (!existsSync('visualization_server.py')) {
    console.error('visualization_server.py not found. Run inside repo or pass --dir.');
    process.exit(1);
  }
  procs.push(spawn(py, ['kira-local-system/kira_server.py', '--host', host, '--port', kiraPort], { stdio: 'inherit' }));
  procs.push(spawn(py, ['visualization_server.py', '--port', vizPort], { stdio: 'inherit' }));
  console.log(`Started KIRA on http://${host}:${kiraPort} and Viz on http://localhost:${vizPort}`);
  await new Promise(() => {}); // keep running until killed
}

function httpGetJson(url) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (res) => {
      let data = '';
      res.on('data', chunk => (data += chunk));
      res.on('end', () => {
        try { resolve(JSON.parse(data)); } catch (_) { resolve({ statusCode: res.statusCode, body: data }); }
      });
    });
    req.on('error', reject);
    req.setTimeout(2000, () => { req.destroy(new Error('timeout')); });
  });
}

async function health(flags = {}) {
  const host = flags.host || 'localhost';
  const kiraPort = String(flags.kiraPort || flags.port || 5000);
  const vizPort = String(flags.vizPort || 8765);
  const results = { kira: null, viz: null };
  try { results.kira = await httpGetJson(`http://${host}:${kiraPort}/api/health`); } catch (e) { results.kira = { error: e.message }; }
  try { results.viz = await httpGetJson(`http://${host}:${vizPort}/state`); } catch (e) { results.viz = { error: e.message }; }
  console.log(JSON.stringify(results, null, 2));
}

function showVersion() {
  try {
    const pkg = JSON.parse(readFileSync(join(__dirname, '..', 'package.json'), 'utf8'));
    console.log(pkg.version);
  } catch (_) {
    console.log('unknown');
  }
}

async function doctor(flags = {}) {
  if (flags.dir) process.chdir(flags.dir);
  const checks = [];
  function ok(name, ok, extra = '') { checks.push({ name, ok, extra }); }
  // Node version
  const nodeOk = parseInt(process.versions.node.split('.')[0], 10) >= 16;
  ok('node>=16', nodeOk, process.versions.node);
  // Python present
  const pyCmd = flags.python || (process.platform === 'win32' ? 'python' : 'python3');
  try { await sh(pyCmd, ['--version']); ok('python present', true); } catch { ok('python present', false); }
  // .venv present
  ok('.venv exists', existsSync(join(process.cwd(), '.venv')));
  // Repo files
  ok('kira_server.py exists', existsSync('kira-local-system/kira_server.py'));
  ok('visualization_server.py exists', existsSync('visualization_server.py'));
  // Docker
  try { await sh('docker', ['--version']); ok('docker present', true); } catch { ok('docker present', false); }
  // Print
  for (const c of checks) {
    console.log(`${c.ok ? '✔' : '✖'} ${c.name}${c.extra ? ' (' + c.extra + ')' : ''}`);
  }
  const allOk = checks.every(c => c.ok);
  process.exit(allOk ? 0 : 1);
}

(async () => {
  const cmd = process.argv[2] || '';
  const args = process.argv.slice(3);
  const flags = parseFlags(args);
  try {
    if (cmd === 'init') await initRepo(flags._[0]);
    else if (cmd === 'setup') await setup(flags);
    else if (cmd === 'kira') await runKira(flags);
    else if (cmd === 'viz') await runViz(flags);
    else if (cmd === 'helix:train') await runHelixTrain(flags);
    else if (cmd === 'helix:nightly') await runHelixNightly(flags);
    else if (cmd === 'smoke') await runSmoke(flags);
    else if (cmd === 'api:test') await runApiTests(flags);
    else if (cmd === 'start') await startBoth(flags);
    else if (cmd === 'health') await health(flags);
    else if (cmd === 'version' || cmd === '--version' || cmd === '-v') showVersion();
    else if (cmd === 'doctor') await doctor(flags);
    else if (cmd && cmd.startsWith('docker:')) await runCompose(cmd);
    else {
      console.log(`Usage: rosetta-helix <command>\n` +
        `  init [dir]           Clone repo and run setup\n` +
        `  setup [--dir D]      Create .venv and install deps\n` +
        `  kira [--host H --port P]   Start KIRA (default 0.0.0.0:5000)\n` +
        `  viz [--port P]       Start Visualization server (default 8765)\n` +
        `  start                Start KIRA + Viz together\n` +
        `  health               Check http://localhost:{5000,8765} health\n` +
        `  doctor [--dir D]     Check environment & repo files\n` +
        `  helix:train          Run helix training\n` +
        `  helix:nightly        Run nightly training\n` +
        `  smoke                Run smoke tests\n` +
        `  api:test             Run API contract tests\n` +
        `  version              Print CLI version\n` +
        `  docker:build|up|down|logs  Docker compose helpers\n` +
        `\nOptions: --dir <path> --python <path> --host <host> --port <num>\n` +
        `Run inside a Rosetta-Helix-Substrate checkout (or use init/--dir).`);
      process.exit(1);
    }
  } catch (err) {
    console.error(err.message || err);
    process.exit(1);
  }
})();
