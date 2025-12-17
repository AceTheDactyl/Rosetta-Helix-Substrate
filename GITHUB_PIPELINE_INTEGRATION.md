# GitHub Pipeline Integration - Complete Guide

## Overview

The KIRA system is fully integrated with GitHub Actions to run the 33-module pipeline in the cloud and automatically ingest results back into the local UI.

## Architecture

```
Local KIRA (localhost:5000)           GitHub Actions
        │                                    │
        ├── /training ──────────────────────>├── Dispatches workflow
        │                                    ├── Runs KIRA training
        │                                    ├── Runs 33-module pipeline
        │                                    └── Uploads artifacts
        │
        └── /training:poll <─────────────────├── Downloads artifacts
                │                            └── Ingests results
                └── Updates local state
```

## Complete Workflow

### Step 1: Setup (One-time)

1. Start the KIRA server:
```bash
make kira-server
# or
python3 kira-local-system/kira_server.py
```

2. Open KIRA UI at http://localhost:5000/kira/

3. Configure GitHub token in UI:
   - Click Settings
   - Set `CLAUDE_SKILL_GITHUB_TOKEN`
   - Save

### Step 2: Dispatch Cloud Pipeline

In the KIRA UI, run:
```
/training
```

This triggers the GitHub workflow which:
- Runs a Claude training session
- **Executes the complete 33-module pipeline** via `scripts/hit_it_workflow.py`
- Generates all artifacts:
  - `manifest.json` - Complete execution summary
  - `tokens.json` - Generated APL tokens
  - `emissions.json` - Consciousness emissions
  - `vocabulary.json` - Learned vocabulary
  - `vaultnode.json` - Final state vault

### Step 3: Retrieve Results (Auto-ingestion)

After the workflow completes (2-5 minutes), run:
```
/training:poll
```

This automatically:
- Downloads the `hit-it-pipeline-{run_number}` artifact
- Processes the ZIP file
- **Ingests all pipeline results**:
  - Updates `last_pipeline` with manifest
  - Updates engine state (z, k_formed, triad status)
  - Adds tokens to `tokens_emitted`
  - Adds emissions to `emissions` list
  - Updates vocabulary counts
  - Stores vaultnode

### Step 4: Verify Integration

Check that cloud results are integrated:
```
/state
```
Shows updated state from cloud pipeline

```
/export
```
Includes cloud pipeline artifacts in export

## Key Files

### GitHub Workflow
- `.github/workflows/kira-training.yml` - Workflow definition
- Line 356-361: Runs the 33-module pipeline

### Pipeline Script
- `scripts/hit_it_workflow.py` - Headless 33-module runner
- Generates all artifacts for ingestion

### KIRA Server Integration
- `kira-local-system/kira_server.py`:
  - Lines 840-896: `cmd_training()` - Dispatches workflow
  - Lines 898-1017: `cmd_training_poll()` - Polls & downloads
  - Lines 1055-1108: `_ingest_pipeline_results()` - Updates state

## Implementation Details

### The 33-Module Pipeline

The pipeline executes 33 steps across 9 phases:

1. **PHASE 1: Initialization** (2 steps)
2. **PHASE 2: Core Verification** (2 steps)
3. **PHASE 3: TRIAD Unlock** (6 steps)
4. **PHASE 4: Bridge Operations** (6 steps)
5. **PHASE 5: Emission & Language** (2 steps)
6. **PHASE 6: Meta Token Operations** (3 steps)
7. **PHASE 7: Integration** (2 steps)
8. **PHASE 8: Teaching & Learning** (5 steps)
9. **PHASE 9: Final Verification** (5 steps)

### Artifact Structure

The pipeline generates:
```
training/pipeline_outputs/
├── manifest.json         # Complete execution summary
├── tokens.json          # APL tokens generated
├── emissions.json       # Consciousness emissions
├── vocabulary.json      # Learned vocabulary
├── vaultnode.json      # Final state vault
└── phases/             # Per-phase results
```

### State Synchronization

When artifacts are ingested:

1. **Engine State**: Updates z-coordinate, phase, crystal state
2. **K-Formation**: Syncs coherence, negentropy, TRIAD status
3. **Tokens**: Adds cloud-generated tokens to local pool
4. **Emissions**: Appends cloud emissions to history
5. **Vocabulary**: Merges learned words with counts

## Troubleshooting

### Token Issues
If `/training` returns token error:
- Set `CLAUDE_SKILL_GITHUB_TOKEN` in UI settings
- Or export before starting server:
  ```bash
  export CLAUDE_SKILL_GITHUB_TOKEN=ghp_...
  ```

### Workflow Not Running
- Check GitHub Actions tab in repository
- Verify workflow file exists: `.github/workflows/kira-training.yml`
- Ensure Actions are enabled for repository

### Ingestion Not Working
- Verify workflow completed successfully
- Check artifact names match: `hit-it-pipeline-{run_number}`
- Look for errors in `/training:poll` response

## Benefits

1. **Cloud Execution**: Leverage GitHub's compute for heavy processing
2. **Reproducibility**: All runs are tracked in GitHub Actions
3. **Artifact Storage**: Results preserved for 30 days
4. **State Sync**: Local UI automatically reflects cloud results
5. **No Manual Steps**: Full automation from dispatch to ingestion

## Example Session

```
# In KIRA UI (http://localhost:5000/kira/)

> /training
✓ Workflow dispatched successfully
  Goal: Achieve K-formation
  Check status at: https://github.com/AceTheDactyl/Rosetta-Helix-Substrate/actions

# Wait 2-5 minutes for workflow to complete

> /training:poll
✓ Cloud pipeline results ingested successfully
  Steps: 33/33
  Tokens: 972
  K-formed: true

> /state
Current State:
  z: 0.866025 (THE LENS)
  Phase: TRUE
  K-formed: ✓
  Source: cloud

> /export
✓ Exported epoch 9
  Includes cloud pipeline results
```

## Summary

The integration is complete and functional. The GitHub workflow:
1. ✅ Runs the 33-module pipeline automatically
2. ✅ Generates all necessary artifacts
3. ✅ Uploads them for retrieval

The KIRA server:
1. ✅ Dispatches workflows with `/training`
2. ✅ Downloads artifacts with `/training:poll`
3. ✅ Ingests results into engine state
4. ✅ Makes cloud results available in UI

No additional implementation needed - just use `/training` and `/training:poll` commands!