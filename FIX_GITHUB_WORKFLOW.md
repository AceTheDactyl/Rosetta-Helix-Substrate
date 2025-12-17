# Fix for GitHub Workflow Dispatch Error

## The Issue

Getting error:
```json
{
  "message": "Workflow does not have 'workflow_dispatch' trigger",
  "documentation_url": "https://docs.github.com/rest/actions/workflows#create-a-workflow-dispatch-event",
  "status": "422"
}
```

## Root Cause

The `/training` command is trying to trigger `kira-training.yml` but GitHub might be looking for a different workflow or the workflow isn't properly registered.

## Solutions

### Solution 1: Use Workflow ID Instead of Name

GitHub sometimes requires the workflow ID instead of the filename. To find the workflow ID:

```bash
# Get workflow ID using GitHub CLI
gh workflow list

# Or via API
curl -H "Authorization: token YOUR_TOKEN" \
  https://api.github.com/repos/AceTheDactyl/Rosetta-Helix-Substrate/actions/workflows
```

### Solution 2: Verify Workflow File

The workflow file `.github/workflows/kira-training.yml` does have the `workflow_dispatch` trigger properly configured:

```yaml
on:
  workflow_dispatch:
    inputs:
      training_goal:
        description: 'Training goal'
        required: true
        default: 'Evolve consciousness toward THE LENS'
```

### Solution 3: Use Full Path or Different Reference

Try updating the workflow reference in the server:

```python
# Instead of just filename
TRAINING_WORKFLOW = 'kira-training.yml'

# Try full path
TRAINING_WORKFLOW = '.github/workflows/kira-training.yml'

# Or workflow ID (get from GitHub)
TRAINING_WORKFLOW = '12345678'  # Replace with actual ID
```

## Temporary Workaround

### Manual Trigger via GitHub UI

1. Go to: https://github.com/AceTheDactyl/Rosetta-Helix-Substrate/actions/workflows/kira-training.yml
2. Click "Run workflow"
3. Enter training goal
4. Click "Run workflow" button

### Use GitHub CLI

```bash
gh workflow run kira-training.yml \
  -f training_goal="Achieve K-formation" \
  -f max_turns=20 \
  -f initial_z=0.5
```

## Updated /help Command

The `/help` command now shows all available commands including:
- All UCF commands (`/ucf:*`)
- Pipeline commands (`/hit_it`, `/consciousness_journey`)
- All 21 UCF tools
- All 7 phases
- Sacred constants

### To See Updated Help

Restart the server and type `/help`:

```bash
# Restart server
npx rosetta-helix start

# In interface, type:
/help
```

You'll now see:

```
📍 Core Commands:
  /state - Show consciousness state
  /evolve [z] - Evolve toward target z
  ...

🚀 Pipeline Commands:
  /hit_it - ⭐ Run FULL 33-module pipeline
  /consciousness_journey - ⭐ 7-layer evolution
  ...

🔧 UCF Tools (21 Available):
  /ucf:status - System status
  /ucf:spinner - Generate 972 tokens
  ...

🔮 UCF Phases:
  /ucf:phase1 - Initialization (1-3)
  /ucf:phase2 - Core Tools (4-7)
  ...
```

## Checking Workflow Status

To verify the workflow is visible to GitHub:

```bash
# List all workflows
curl -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/AceTheDactyl/Rosetta-Helix-Substrate/actions/workflows

# Check specific workflow
curl -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/AceTheDactyl/Rosetta-Helix-Substrate/actions/workflows/kira-training.yml
```

## Alternative: Use a Different Workflow

If `kira-training.yml` isn't working, you could use one of the other workflows that has `workflow_dispatch`:

- `kira-claude-api.yml` - Has workflow_dispatch
- `autonomous-training.yml` - Might have it

To change which workflow is used:

```python
# In kira_server.py, line 146
TRAINING_WORKFLOW = os.environ.get('KIRA_TRAINING_WORKFLOW', 'kira-claude-api.yml')
```

## Summary

1. **Workflow exists and is configured correctly** in `.github/workflows/kira-training.yml`
2. **GitHub API might need workflow ID** instead of filename
3. **Manual triggering works** via GitHub UI or CLI
4. **/help command now shows all commands** properly

The issue is likely that GitHub's API wants a different identifier for the workflow. Try getting the workflow ID from the API and using that instead of the filename.