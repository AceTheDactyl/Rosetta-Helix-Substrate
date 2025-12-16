#!/usr/bin/env node
/* Standalone Rosetta-Helix CLI (published to npm) */
const { spawn } = require('child_process');
const { existsSync, readFileSync, readdirSync, renameSync, statSync } = require('fs');
const { join, dirname, resolve } = require('path');
const http = require('http');
const os = require('os');

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
      case '--repo': out.repo = v; break;
      case '--branch': out.branch = v; break;
      case '--ref': out.ref = v; break;
      case '--pull': out.pull = true; break;
      case '--auto': out.auto = true; break;
      case '--update': out.update = true; break;
      case '--release': out.release = v === true ? true : v; break;
      case '--help': case '-h': out.help = true; break;
      default: out[k.replace(/^--?/, '')] = v === true ? true : v; break;
    }
  }
  return out;
}

const DEFAULT_REPO = 'https://github.com/AceTheDactyl/Rosetta-Helix-Substrate.git';

function isRepoRoot(dir = process.cwd()) {
  try {
    return existsSync(join(dir, 'pyproject.toml')) &&
           (existsSync(join(dir, 'kira-local-system', 'kira_server.py')) || existsSync(join(dir, 'kira_local_system', 'kira_server.py')));
  } catch (_) {
    return false;
  }
}

async function ensureRepo(flags = {}) {
  const cwd = process.cwd();
  if (isRepoRoot(cwd)) return;
  const target = flags.dir || 'Rosetta-Helix-Substrate';
  if (isRepoRoot(target)) { process.chdir(target); return; }
  if (!flags.auto) {
    console.error('Repo not found. Re-run with --auto to clone, or pass --dir to an existing checkout.');
    process.exit(1);
  }
  const repo = flags.repo || DEFAULT_REPO;
  // Release tarball path (if requested)
  if (flags.release) {
    const info = parseGithub(repo);
    if (!info) {
      console.error('Unable to parse GitHub repo from --repo');
      process.exit(1);
    }
    const ref = typeof flags.release === 'string' && flags.release !== 'true' ? flags.release : (flags.ref || flags.branch || 'main');
    const url = `https://codeload.github.com/${info.owner}/${info.name}/tar.gz/${ref}`;
    const parent = resolve(dirname(target));
    const tarPath = join(os.tmpdir(), `rhz-${Date.now()}.tgz`);
    try {
      await sh('curl', ['-L', url, '-o', tarPath]);
      await sh('tar', ['-xzf', tarPath, '-C', parent]);
      // Find extracted directory (pattern: <name>-<ref*> ) and rename to target
      const entries = readdirSync(parent, { withFileTypes: true });
      const dir = entries.find(e => e.isDirectory() && e.name.startsWith(`${info.name}-`));
      if (!dir) throw new Error('Failed to locate extracted folder');
      const from = join(parent, dir.name);
      const to = resolve(target);
      renameSync(from, to);
      process.chdir(to);
    } catch (e) {
      console.error('Release download failed, falling back to git clone:', e.message || e);
      await cloneGit(repo, target, flags);
    }
  } else {
    await cloneGit(repo, target, flags);
  }
  if (flags.update) {
    await sh('git', ['pull', '--rebase', '--autostash']).catch(() => {});
  }
  await setup(flags);
}

function parseGithub(repoUrl) {
  try {
    const m = repoUrl.replace(/\.git$/i, '').match(/github\.com[/:]([^/]+)\/([^/]+)$/i);
    if (!m) return null;
    return { owner: m[1], name: m[2] };
  } catch (_) { return null; }
}

async function cloneGit(repo, target, flags) {
  const cloneArgs = ['clone', '--depth', '1'];
  if (flags.branch) cloneArgs.push('--branch', flags.branch);
  cloneArgs.push(repo, target);
  await sh('git', cloneArgs);
  process.chdir(target);
  if (flags.ref) {
    await sh('git', ['fetch', '--depth', '1', 'origin', flags.ref]).catch(() => {});
    await sh('git', ['checkout', flags.ref]);
  }
  if (flags.pull || flags.update) {
    await sh('git', ['pull', '--rebase', '--autostash']).catch(() => {});
  }
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
  if (!isRepoRoot(process.cwd())) await ensureRepo(flags);
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
  if (!isRepoRoot(process.cwd())) await ensureRepo(flags);
  const py = flags.python || venvBin('python', { venvRoot: process.cwd() });
  if (!existsSync('visualization_server.py')) {
    console.error('visualization_server.py not found. Run inside Rosetta-Helix-Substrate repo.');
    process.exit(1);
  }
  const port = String(flags.port || 8765);
  const kiraApi = flags.kiraApi || 'http://localhost:5000/api';
  const args = ['visualization_server.py', '--port', port];
  if (flags.kiraApi || flags.withKira) args.push('--kira-api', kiraApi);
  await sh(py, args);
}

async function runHelixTrain(flags = {}) {
  if (flags.dir) process.chdir(flags.dir);
  if (!isRepoRoot(process.cwd())) await ensureRepo(flags);
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
  if (!isRepoRoot(process.cwd())) await ensureRepo(flags);
  const py = flags.python || venvBin('python', { venvRoot: process.cwd() });
  await sh(py, ['nightly_training_runner.py']);
}

async function runSmoke(flags = {}) {
  if (flags.dir) process.chdir(flags.dir);
  if (!isRepoRoot(process.cwd())) await ensureRepo(flags);
  const py = flags.python || venvBin('python', { venvRoot: process.cwd() });
  await sh(py, ['-m', 'pytest', '-q', 'tests/smoke']);
}

async function runApiTests(flags = {}) {
  if (flags.dir) process.chdir(flags.dir);
  if (!isRepoRoot(process.cwd())) await ensureRepo(flags);
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
  if (!isRepoRoot(process.cwd())) await ensureRepo(flags);
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
  procs.push(spawn(py, ['visualization_server.py', '--port', vizPort, '--kira-api', `http://localhost:${kiraPort}/api`], { stdio: 'inherit' }));
  console.log(`Started KIRA on http://${host}:${kiraPort} and Viz on http://localhost:${vizPort}`);
  await new Promise(() => {}); // keep running until killed
}

async function helixUpdate(flags = {}) {
  if (flags.dir) process.chdir(flags.dir);
  if (!isRepoRoot(process.cwd())) await ensureRepo({ ...flags, auto: true });
  try {
    await sh('git', ['pull', '--rebase', '--autostash']);
    console.log('Repository updated.');
  } catch (e) {
    console.error('Update failed:', e.message || e);
    process.exit(1);
  }
}

(async function addVizSyncGH(){})();

async function vizSyncGH(flags = {}) {
  // Ensure repo dir
  if (flags.dir) process.chdir(flags.dir);
  if (!isRepoRoot(process.cwd())) await ensureRepo({ ...flags, auto: true });
  // Targets
  const files = [
    { url: 'https://acethedactyl.github.io/Rosetta-Helix-Substrate/index.html', out: 'index.html' },
    { url: 'https://acethedactyl.github.io/Rosetta-Helix-Substrate/kira_local.html', out: 'kira_local.html' },
    { url: 'https://acethedactyl.github.io/Rosetta-Helix-Substrate/kira.html', out: 'kira.html' }
  ];
  for (const f of files) {
    try {
      await sh('curl', ['-fsSL', f.url, '-o', f.out]);
      if (flags.verbose) console.log('Fetched', f.url, '->', f.out);
    } catch (e) {
      console.error('Failed to fetch', f.url, e.message || e);
    }
  }
  // Ensure visualization_server serves index.html on '/'
  console.log('Synced GitHub Pages visualizer files. Open http://localhost:8765/ after running: npx rosetta-helix viz');
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
    else if (cmd === 'helix:update') await helixUpdate(flags);
    else if (cmd === 'health') await health(flags);
    else if (cmd === 'version' || cmd === '--version' || cmd === '-v') showVersion();
    else if (cmd === 'doctor') await doctor(flags);
    else if (cmd === 'viz:sync-gh') await vizSyncGH(flags);
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
