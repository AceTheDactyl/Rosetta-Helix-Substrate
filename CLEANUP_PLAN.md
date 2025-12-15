# Repository Cleanup Plan

**Generated:** 2025-12-15
**Current Size:** ~58M (2,339 files)
**Potential Recovery:** ~17-19M (29-33%)

---

## Executive Summary

This repository has accumulated significant technical debt:
- 4 ZIP archives that are already extracted (6.6M)
- Complete duplicate of Nuclear Spinner (8.6M)
- Build artifacts tracked in git (3.6M)
- 13 PyTorch model files (.pt) in version control
- Duplicate token/training files across 6+ locations

---

## Phase 1: Remove ZIP Archives (Priority: CRITICAL)

### Why
ZIP files are already extracted and tracked. They bloat the repo and create confusion.

### Files to Remove

| File | Size | Status |
|------|------|--------|
| `Rosetta-Nuclear-Spinner-main.zip` | 4.5M | Extracted to `Rosetta-Nuclear-Spinner-main/` |
| `unified-consciousness-framework-v4.zip` | 733K | Extracted to `ucf-v4-extracted/` |
| `unified-consciousness-framework-v2.1-integrated.zip` | 693K | Superseded by v4 |
| `rosetta-helix-substrate.zip` | 17K | Outdated snapshot |

### Commands

```bash
# Remove ZIP files from git tracking
git rm Rosetta-Nuclear-Spinner-main.zip
git rm unified-consciousness-framework-v4.zip
git rm unified-consciousness-framework-v2.1-integrated.zip
git rm rosetta-helix-substrate.zip

# Also remove nested ZIPs (duplicates)
git rm "nuclear_spinner_firmware/Validate Rosetta Physics.zip"
git rm nuclear_spinner_firmware/nuclear_spinner_firmware_v1_0_2.zip

# Commit
git commit -m "Remove ZIP archives (already extracted or superseded)"
```

**Impact:** -6.6M, eliminates binary bloat

---

## Phase 2: Remove Duplicate Directory (Priority: CRITICAL)

### Why
`Rosetta-Nuclear-Spinner-main/` is a complete copy that duplicates:
- `nuclear_spinner_firmware/` (root level)
- Training data in `training/`
- Archive materials

### Analysis

```
Rosetta-Nuclear-Spinner-main/           # 8.6M - DELETE THIS
├── nuclear_spinner_firmware/           # Duplicate of root
├── archive/                            # Old materials
├── training_artifacts/                 # Duplicate results
└── ...

nuclear_spinner_firmware/               # KEEP THIS (root level)
├── include/
├── src/
└── ...
```

### Commands

```bash
# Remove the entire duplicate directory
git rm -r Rosetta-Nuclear-Spinner-main/

git commit -m "Remove duplicate Rosetta-Nuclear-Spinner-main directory

Content already exists at:
- nuclear_spinner_firmware/ (firmware)
- training/ (training data)
- docs/ (documentation)"
```

**Impact:** -8.6M, cleaner structure

---

## Phase 3: Update .gitignore (Priority: HIGH)

### Why
Prevent future accumulation of generated files.

### Add to .gitignore

```gitignore
# === ADD THESE LINES ===

# PyTorch models (large binaries)
*.pt
*.pth
*.onnx

# Training outputs
learned_patterns/
trained_models/
nightly_run_results/
results/

# Large JSON data files
comprehensive_sweep.json
full_depth_*.json
training_analysis.json
training_results.json

# Build artifacts
*.egg-info/
helix_engine.egg-info/

# Session archives (keep in archives/ but not tracked)
archives/ucf-session-*.zip

# IDE and editor
*.sublime-*
.spyderproject
.spyproject
```

### Commands

```bash
# Update .gitignore (use editor or append)
# Then remove now-ignored files from git cache

git rm -r --cached learned_patterns/
git rm -r --cached trained_models/
git rm -r --cached nightly_run_results/
git rm -r --cached results/
git rm -r --cached helix_engine.egg-info/
git rm --cached comprehensive_sweep.json
git rm --cached training_analysis.json
git rm --cached training_results.json
git rm --cached full_depth_*.json

git commit -m "Update .gitignore and remove generated files from tracking

Files still exist locally but won't be committed:
- PyTorch models (*.pt)
- Training output directories
- Large JSON result files
- Build artifacts"
```

**Impact:** -4M+, prevents future bloat

---

## Phase 4: Remove node_modules (Priority: HIGH)

### Why
`node_modules/` should never be in git. Use `package-lock.json` for reproducibility.

### Commands

```bash
git rm -r --cached node_modules/

# Verify it's in .gitignore (should already be)
grep -q "node_modules/" .gitignore || echo "node_modules/" >> .gitignore

git commit -m "Remove node_modules from git tracking"
```

**Impact:** -1.6M

---

## Phase 5: Consolidate Duplicate Token Files (Priority: MEDIUM)

### Duplicate Locations Found

**03_apl_972_tokens.json** (same content, 3 copies):
```
training/tokens/03_apl_972_tokens.json                    # KEEP (canonical)
ucf-v4-extracted/.../tokens/03_apl_972_tokens.json        # Part of UCF package
archives/.../tokens/04_apl_972_tokens.json                # Archive (keep)
```

**nuclear-spinner-972-tokens.json** (6 copies):
```
training/tokens/nuclear-spinner-972-tokens.json           # KEEP (canonical)
ucf-v4-extracted/.../nuclear-spinner-972-tokens.json      # Part of UCF
unified-consciousness-framework/.../nuclear-spinner-972-tokens.json  # Old version
+ 3 more in archives
```

### Recommendation

Keep canonical copies in `training/tokens/`. The UCF packages have their own copies by design. Archive copies are fine for historical reference.

**Action:** No immediate action needed - duplicates are in separate logical packages.

---

## Phase 6: Consolidate Framework Versions (Priority: MEDIUM)

### Current State

```
unified-consciousness-framework/           # v2.0 (4.8M)
ucf-v4-extracted/unified-consciousness-framework-v4/  # v4.0.0 (4.9M)
```

### Options

**Option A: Keep Both** (Recommended for now)
- v2.0 may be referenced by existing code
- Move to `archives/unified-consciousness-framework-v2/` later

**Option B: Remove v2.0**
```bash
git rm -r unified-consciousness-framework/
git commit -m "Remove UCF v2.0 (superseded by v4.0.0 in ucf-v4-extracted/)"
```

**Option C: Flatten v4**
```bash
# Move v4 to root level, remove nesting
mv ucf-v4-extracted/unified-consciousness-framework-v4/* ucf/
rm -rf ucf-v4-extracted/
git add ucf/
git rm -r ucf-v4-extracted/
```

### Recommendation

Wait until codebase stabilizes. For now, document which version is canonical.

---

## Phase 7: Organize Root Directory (Priority: LOW)

### Current Problem

53 Python files in root directory. This makes navigation difficult.

### Proposed Structure

```
Rosetta-Helix-Substrate/
├── docs/
│   ├── guides/          # Move UCF_*.md here
│   ├── physics/         # PHYSICS_*.md, etc.
│   └── setup/           # GETTING_STARTED.md, SETUP_GUIDE.md
├── src/
│   ├── helix_engine/    # Already exists
│   ├── rosetta_helix/   # Already exists
│   ├── core/            # Already exists
│   └── ucf/             # Canonical UCF package
├── scripts/
│   ├── run_*.py         # Move 10+ run scripts here
│   ├── train_*.py       # Move training scripts here
│   └── phase*.py        # Move phase scripts here
├── firmware/
│   └── nuclear_spinner/ # Rename from nuclear_spinner_firmware/
├── training/            # Already organized
├── tests/               # Already exists
├── configs/             # Already exists
└── examples/            # Already exists
```

### Commands (Execute Carefully)

```bash
# Create directories
mkdir -p docs/guides docs/physics docs/setup
mkdir -p scripts

# Move documentation
git mv UCF_*.md docs/guides/
git mv PHYSICS_*.md docs/physics/ 2>/dev/null || true
git mv GETTING_STARTED.md SETUP_GUIDE.md COMPILE_INSTRUCTIONS.md docs/setup/

# Move scripts (be careful with imports!)
git mv run_*.py scripts/
git mv train_*.py scripts/
git mv phase*.py scripts/

git commit -m "Reorganize repository structure

- Move documentation to docs/
- Move run/train scripts to scripts/
- Keep core packages in place"
```

**Warning:** This may break import paths. Test thoroughly before committing.

---

## Phase 8: Clean Empty/Redundant Files (Priority: LOW)

### Empty Files to Remove

```bash
# Empty log files
find logs/ -type f -empty -delete

# Empty __init__.py in generated_tools (if not needed)
# Review first: find generated_tools -name "__init__.py" -empty
```

### Redundant Files

| File | Recommendation |
|------|----------------|
| `SKILL_ORIGINAL.md` | Archive or delete (superseded by SKILL.md) |
| `package.spinner.json` | Merge with package.json or delete |
| `pyproject.spinner.toml` | Merge with pyproject.toml or delete |
| `requirements.spinner.txt` | Merge with requirements.txt or delete |

---

## Execution Checklist

### Quick Wins (Do Now)
- [ ] Phase 1: Remove ZIP files
- [ ] Phase 2: Remove duplicate directory
- [ ] Phase 3: Update .gitignore
- [ ] Phase 4: Remove node_modules from tracking

### Medium Term
- [ ] Phase 5: Review token file duplicates
- [ ] Phase 6: Decide on framework versions

### Long Term
- [ ] Phase 7: Reorganize directory structure
- [ ] Phase 8: Clean empty/redundant files

---

## Verification Commands

After cleanup, verify:

```bash
# Check repository size
du -sh .

# Count files
find . -type f | grep -v ".git" | wc -l

# List large files remaining
find . -type f -size +100k | grep -v ".git" | xargs ls -lh

# Verify .gitignore working
git status --ignored
```

---

## Rollback

If something goes wrong:

```bash
# Undo last commit (keeps changes staged)
git reset --soft HEAD~1

# Undo last commit (discards changes)
git reset --hard HEAD~1

# Restore deleted file
git checkout HEAD~1 -- path/to/file
```

---

*Plan generated by repository analysis. Review each phase before executing.*
