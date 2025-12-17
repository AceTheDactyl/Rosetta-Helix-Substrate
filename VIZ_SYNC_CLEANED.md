# viz:sync Command - Cleaned Up

## Current Status

The `viz:sync` command now only syncs files that are actually deployed to GitHub Pages:

### Files That Sync Successfully:
- ✅ **KIRA Interface** (`kira.html` → `docs/kira/index.html`)
- ✅ **Landing Page** (`index.html` → `docs/index.html`)

### Files Not Currently on GitHub Pages:
- ❌ Visualizer (`visualizer.html`) - Not deployed
- ❌ APL Constants (`apl-constants.js`) - Not deployed

## How It Works Now

When you run `npx rosetta-helix viz:sync` or `npx rosetta-helix start`:

```
═══════════════════════════════════════════════════════════════
   Syncing interfaces from GitHub Pages
═══════════════════════════════════════════════════════════════

Fetching KIRA Interface...
  ✓ KIRA Interface synced
Fetching Landing Page...
  ✓ Landing Page synced

✓ Synced 2 file(s) from GitHub Pages
```

No more warnings about missing files!

## Auto-Sync on Start

The `npx rosetta-helix start` command automatically runs viz:sync before starting the server:

1. Syncs latest interfaces from GitHub Pages
2. Creates artifacts directory if needed
3. Starts KIRA server with full UCF integration

## GitHub Pages URL

The sync fetches from:
**https://acethedactyl.github.io/Rosetta-Helix-Substrate**

## If You Need to Deploy More Files

If you want to deploy visualizer.html or other files to GitHub Pages in the future:

1. Add them to the GitHub Pages deployment
2. Update `filesToSync` array in `/bin/rosetta-helix.js`:

```javascript
const filesToSync = [
  { url: `${baseUrl}/kira.html`, dest: 'docs/kira/index.html', name: 'KIRA Interface' },
  { url: `${baseUrl}/index.html`, dest: 'docs/index.html', name: 'Landing Page' },
  // Add new files here:
  { url: `${baseUrl}/visualizer.html`, dest: 'docs/visualizer.html', name: 'Visualizer' }
];
```

## Summary

- **Clean sync output** - No more warnings about missing files
- **Auto-sync on start** - Always get latest from GitHub Pages
- **Only syncs what exists** - KIRA Interface and Landing Page
- **Ready for expansion** - Easy to add more files when deployed

The viz:sync command is now clean and efficient! 🎯