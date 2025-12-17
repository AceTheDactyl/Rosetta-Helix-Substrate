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
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('   K.I.R.A. Unified Backend Server');
  console.log('   All modules integrated');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');
  console.log('   Starting server at http://localhost:5000');
  console.log('   Open http://localhost:5000/kira/ in browser');
  console.log('');
  console.log('   Commands: /state /train /evolve /grammar /coherence');
  console.log('             /emit /tokens /triad /hit_it /reset /save /help');
  console.log('');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');
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
  const https = require('https');

  console.log('═══════════════════════════════════════════════════════════════');
  console.log('   Syncing interfaces from GitHub Pages');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');

  const baseUrl = 'https://acethedactyl.github.io/Rosetta-Helix-Substrate';

  // Files to sync from GitHub Pages
  // Note: Only syncing files that are actually deployed to GitHub Pages
  const filesToSync = [
    { url: `${baseUrl}/kira.html`, dest: 'docs/kira/index.html', name: 'KIRA Interface' },
    { url: `${baseUrl}/index.html`, dest: 'docs/index.html', name: 'Landing Page' }
  ];

  // Optional files - currently none are deployed but keeping structure for future
  const optionalFiles = [];

  let syncedCount = 0;
  let skippedCount = 0;

  // Function to sync a file
  async function syncFile(file, isOptional = false) {
    try {
      if (!isOptional) console.log(`Fetching ${file.name}...`);

      await new Promise((resolve, reject) => {
        https.get(file.url, (res) => {
          if (res.statusCode === 404) {
            if (!isOptional) {
              console.log(`  ⚠ ${file.name}: Not found on GitHub Pages (using local)`);
            }
            skippedCount++;
            resolve();
            return;
          } else if (res.statusCode !== 200) {
            if (!isOptional) {
              console.log(`  ⚠ ${file.name}: HTTP ${res.statusCode} (using local)`);
            }
            skippedCount++;
            resolve();
            return;
          }

          // Only show message for optional files if they exist
          if (isOptional) {
            console.log(`Fetching ${file.name}...`);
          }

          // Ensure destination directory exists
          const destPath = path.join(process.cwd(), file.dest);
          const destDir = path.dirname(destPath);
          if (!fs.existsSync(destDir)) {
            fs.mkdirSync(destDir, { recursive: true });
          }

          const writer = fs.createWriteStream(destPath);
          res.pipe(writer);
          writer.on('finish', () => {
            console.log(`  ✓ ${file.name} synced`);
            syncedCount++;
            resolve();
          });
          writer.on('error', reject);
        }).on('error', (e) => {
          if (!isOptional) {
            console.log(`  ⚠ ${file.name}: Network error (${e.message})`);
          }
          resolve();
        });
      });
    } catch (e) {
      if (!isOptional) {
        console.log(`  ✗ ${file.name}: ${e.message}`);
      }
    }
  }

  // Sync required files
  for (const file of filesToSync) {
    await syncFile(file, false);
  }

  // Silently check optional files
  for (const file of optionalFiles) {
    await syncFile(file, true);
  }

  console.log('');
  if (syncedCount > 0) {
    console.log(`✓ Synced ${syncedCount} file(s) from GitHub Pages`);
  } else {
    console.log('⚠ No files synced - using local versions');
  }

  // Check for artifacts directory
  const artifactsDir = path.join(process.cwd(), 'artifacts');
  if (!fs.existsSync(artifactsDir)) {
    fs.mkdirSync(artifactsDir, { recursive: true });
  }

  console.log('');
  console.log('Web Interfaces:');
  console.log(`  KIRA UI: http://localhost:5000/kira/ (after running start)`);
  console.log(`  Landing: docs/index.html`);
  console.log('');
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
  // Auto-sync from GitHub Pages before starting
  console.log('Checking for updates from GitHub Pages...');
  await runVizSync();

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
    else if (cmd === 'viz:sync' || cmd === 'viz:sync-gh') await runVizSync();
    else if (cmd === 'start' || cmd === 'star') await runStart();
    else if (cmd === 'health') await runHealth();
    else if (cmd === 'doctor') await runDoctor();
    else if (cmd === 'helix:train') await runHelixTrain();
    else if (cmd === 'helix:nightly') await runHelixNightly();
    else if (cmd === 'smoke') await runSmoke();
    else if (cmd === 'api:test') await runApiTests();
    else if (cmd.startsWith('docker:')) await runCompose(cmd);
    else {
      console.log('═══════════════════════════════════════════════════════════════');
      console.log('   Rosetta Helix CLI - Unified Consciousness Framework');
      console.log('═══════════════════════════════════════════════════════════════');
      console.log('');
      console.log('Usage: npx rosetta-helix <command>');
      console.log('');
      console.log('🚀 Quick Start:');
      console.log('  start           Start KIRA server with full UCF integration');
      console.log('  viz:sync        Check training data and show interfaces');
      console.log('');
      console.log('🔧 Setup & Configuration:');
      console.log('  setup           Create .venv and install dependencies');
      console.log('  doctor          Run environment checks');
      console.log('  health          Check service health endpoints');
      console.log('');
      console.log('🧬 Training & Testing:');
      console.log('  helix:train     Run helix training');
      console.log('  helix:nightly   Run nightly training');
      console.log('  smoke           Run smoke tests');
      console.log('  api:test        Run API contract tests');
      console.log('');
      console.log('🔗 Aliases:');
      console.log('  kira            Start KIRA server (alias for start)');
      console.log('  unified         Start KIRA server (legacy alias)');
      console.log('  viz             Show local interface paths');
      console.log('  viz:sync-gh     Sync visualization data (alias for viz:sync)');
      console.log('');
      console.log('🐳 Docker:');
      console.log('  docker:build    Build Docker images');
      console.log('  docker:up       Start services');
      console.log('  docker:down     Stop services');
      console.log('  docker:logs     View logs');
      console.log('');
      console.log('📍 Web Interfaces:');
      console.log('  KIRA UI:  http://localhost:5000/kira/ (after running start)');
      console.log('  Landing:  docs/index.html (static file)');
      console.log('');
      console.log('═══════════════════════════════════════════════════════════════');
      process.exit(1);
    }
  } catch (err) {
    console.error(err.message || err);
    process.exit(1);
  }
})();
