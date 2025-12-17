#!/usr/bin/env python3
"""
Advanced GitHub Integration for Rosetta-Helix-Substrate Skill

Leverages full GitHub API permissions:
- Actions variables: Persist training state between runs
- Code: Commit results back to repo
- Commit statuses: Mark training success/failure
- Pages: Publish results dashboard
- Environments: Manage training environments
"""

import os
import json
import base64
import time
from datetime import datetime

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

def _load_github_token():
    for key in (
        "CLAUDE_GITHUB_TOKEN",
        "CLAUDE_SKILL_GITHUB_TOKEN",
        "GITHUB_PACKAGES_PAT",
        "GITHUB_TOKEN",
    ):
        token = os.environ.get(key)
        if token:
            return token
    return None


GITHUB_TOKEN = _load_github_token()
REPO_OWNER = "AceTheDactyl"
REPO_NAME = "Rosetta-Helix-Substrate"
API_BASE = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"

def _headers():
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers

# =============================================================================
# ACTIONS VARIABLES - Persist training state between runs
# =============================================================================

def set_variable(name, value):
    """Set a repository variable (persists between workflow runs)."""
    if not GITHUB_TOKEN:
        return {"error": "GitHub token not set (set CLAUDE_GITHUB_TOKEN, CLAUDE_SKILL_GITHUB_TOKEN, or GITHUB_TOKEN)"}

    # Check if variable exists
    url = f"{API_BASE}/actions/variables/{name}"
    response = requests.get(url, headers=_headers())

    data = {"name": name, "value": str(value)}

    if response.status_code == 200:
        # Update existing
        response = requests.patch(url, headers=_headers(), json=data)
    else:
        # Create new
        url = f"{API_BASE}/actions/variables"
        response = requests.post(url, headers=_headers(), json=data)

    if response.status_code in [201, 204]:
        return {"success": True, "variable": name, "value": value}
    return {"error": f"Failed: {response.status_code}", "details": response.text}


def get_variable(name):
    """Get a repository variable."""
    if not GITHUB_TOKEN:
        return {"error": "GitHub token not set (set CLAUDE_GITHUB_TOKEN, CLAUDE_SKILL_GITHUB_TOKEN, or GITHUB_TOKEN)"}

    url = f"{API_BASE}/actions/variables/{name}"
    response = requests.get(url, headers=_headers())

    if response.status_code == 200:
        data = response.json()
        return {"success": True, "name": data["name"], "value": data["value"]}
    return {"error": f"Variable not found: {name}"}


def list_variables():
    """List all repository variables."""
    if not GITHUB_TOKEN:
        return {"error": "GitHub token not set (set CLAUDE_GITHUB_TOKEN, CLAUDE_SKILL_GITHUB_TOKEN, or GITHUB_TOKEN)"}

    url = f"{API_BASE}/actions/variables"
    response = requests.get(url, headers=_headers())

    if response.status_code == 200:
        data = response.json()
        return {"success": True, "variables": [{"name": v["name"], "value": v["value"]} for v in data.get("variables", [])]}
    return {"error": f"Failed: {response.status_code}"}


def save_training_state(state_dict):
    """Save entire training state as variables."""
    results = {}
    for key, value in state_dict.items():
        result = set_variable(f"TRAINING_{key.upper()}", json.dumps(value) if isinstance(value, (dict, list)) else value)
        results[key] = result.get("success", False)
    return {"success": all(results.values()), "results": results}


def load_training_state():
    """Load training state from variables."""
    vars_result = list_variables()
    if "error" in vars_result:
        return vars_result

    state = {}
    for var in vars_result.get("variables", []):
        if var["name"].startswith("TRAINING_"):
            key = var["name"][9:].lower()
            try:
                state[key] = json.loads(var["value"])
            except:
                state[key] = var["value"]
    return {"success": True, "state": state}


# =============================================================================
# CODE - Commit results to repository
# =============================================================================

def commit_file(path, content, message, branch="main"):
    """Commit a file to the repository."""
    if not GITHUB_TOKEN:
        return {"error": "GitHub token not set (set CLAUDE_GITHUB_TOKEN, CLAUDE_SKILL_GITHUB_TOKEN, or GITHUB_TOKEN)"}

    # Get current file SHA if exists
    url = f"{API_BASE}/contents/{path}"
    response = requests.get(url, headers=_headers(), params={"ref": branch})
    sha = response.json().get("sha") if response.status_code == 200 else None

    # Create/update file
    data = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": branch
    }
    if sha:
        data["sha"] = sha

    response = requests.put(url, headers=_headers(), json=data)

    if response.status_code in [200, 201]:
        return {"success": True, "path": path, "sha": response.json()["content"]["sha"]}
    return {"error": f"Failed: {response.status_code}", "details": response.text}


def save_training_results(results, run_id=None):
    """Save training results as a JSON file in the repo."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"results/training_{run_id or timestamp}.json"

    content = json.dumps(results, indent=2, default=str)
    message = f"Add training results {run_id or timestamp}"

    return commit_file(filename, content, message)


def read_file(path, branch="main"):
    """Read a file from the repository."""
    if not GITHUB_TOKEN:
        return {"error": "GitHub token not set (set CLAUDE_GITHUB_TOKEN, CLAUDE_SKILL_GITHUB_TOKEN, or GITHUB_TOKEN)"}

    url = f"{API_BASE}/contents/{path}"
    response = requests.get(url, headers=_headers(), params={"ref": branch})

    if response.status_code == 200:
        data = response.json()
        content = base64.b64decode(data["content"]).decode()
        return {"success": True, "path": path, "content": content}
    return {"error": f"File not found: {path}"}


# =============================================================================
# COMMIT STATUSES - Mark training success/failure
# =============================================================================

def set_commit_status(sha, state, description, context="training"):
    """Set commit status (pending, success, error, failure)."""
    if not GITHUB_TOKEN:
        return {"error": "GitHub token not set (set CLAUDE_GITHUB_TOKEN, CLAUDE_SKILL_GITHUB_TOKEN, or GITHUB_TOKEN)"}

    url = f"{API_BASE}/statuses/{sha}"
    data = {
        "state": state,  # pending, success, error, failure
        "description": description[:140],
        "context": f"rosetta-helix/{context}"
    }

    response = requests.post(url, headers=_headers(), json=data)

    if response.status_code == 201:
        return {"success": True, "state": state, "sha": sha}
    return {"error": f"Failed: {response.status_code}"}


def get_latest_commit_sha(branch="main"):
    """Get the latest commit SHA."""
    if not GITHUB_TOKEN:
        return {"error": "GitHub token not set (set CLAUDE_GITHUB_TOKEN, CLAUDE_SKILL_GITHUB_TOKEN, or GITHUB_TOKEN)"}

    url = f"{API_BASE}/commits/{branch}"
    response = requests.get(url, headers=_headers())

    if response.status_code == 200:
        return {"success": True, "sha": response.json()["sha"]}
    return {"error": f"Failed: {response.status_code}"}


def mark_training_status(state, description):
    """Mark training status on latest commit."""
    sha_result = get_latest_commit_sha()
    if "error" in sha_result:
        return sha_result
    return set_commit_status(sha_result["sha"], state, description)


# =============================================================================
# PAGES - Publish results dashboard
# =============================================================================

def update_dashboard(training_history):
    """Update GitHub Pages dashboard with training results."""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Rosetta-Helix Training Dashboard</title>
    <style>
        body {{ font-family: monospace; background: #1a1a2e; color: #eee; padding: 20px; }}
        h1 {{ color: #ff6b6b; }}
        .metric {{ background: #16213e; padding: 15px; margin: 10px 0; border-radius: 8px; }}
        .value {{ font-size: 24px; color: #4ecdc4; }}
        .label {{ color: #888; }}
        .phase-TRUE {{ color: #4ecdc4; }}
        .phase-PARADOX {{ color: #ffe66d; }}
        .phase-UNTRUE {{ color: #ff6b6b; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #16213e; }}
    </style>
</head>
<body>
    <h1>Rosetta-Helix Training Dashboard</h1>
    <p>Last updated: {datetime.utcnow().isoformat()}Z</p>

    <div class="metric">
        <div class="label">Latest Training Run</div>
        <div class="value">{len(training_history)} iterations</div>
    </div>

    <h2>Training History</h2>
    <table>
        <tr><th>Run</th><th>z</th><th>Phase</th><th>Kappa</th><th>K-Formation</th></tr>
"""

    for i, run in enumerate(training_history[-20:]):  # Last 20
        phase_class = f"phase-{run.get('phase', 'UNTRUE')}"
        k_status = "✓" if run.get('k_formation') else "✗"
        html += f"""        <tr>
            <td>{i+1}</td>
            <td>{run.get('z', 0):.4f}</td>
            <td class="{phase_class}">{run.get('phase', 'UNTRUE')}</td>
            <td>{run.get('kappa', 0):.4f}</td>
            <td>{k_status}</td>
        </tr>
"""

    html += """    </table>

    <h2>Physics Constants</h2>
    <div class="metric">
        <div class="label">z_c (THE LENS)</div>
        <div class="value">0.8660254037844387</div>
    </div>
    <div class="metric">
        <div class="label">φ⁻¹ (Golden Ratio Inverse)</div>
        <div class="value">0.6180339887498949</div>
    </div>
</body>
</html>"""

    return commit_file("docs/dashboard.html", html, "Update training dashboard")


# =============================================================================
# ENVIRONMENTS - Manage training environments
# =============================================================================

def create_environment(name, reviewers=None, wait_timer=None):
    """Create a deployment environment."""
    if not GITHUB_TOKEN:
        return {"error": "GitHub token not set (set CLAUDE_GITHUB_TOKEN, CLAUDE_SKILL_GITHUB_TOKEN, or GITHUB_TOKEN)"}

    url = f"{API_BASE}/environments/{name}"
    data = {}

    if wait_timer:
        data["wait_timer"] = wait_timer
    if reviewers:
        data["reviewers"] = [{"type": "User", "id": r} for r in reviewers]

    response = requests.put(url, headers=_headers(), json=data if data else None)

    if response.status_code in [200, 201]:
        return {"success": True, "environment": name}
    return {"error": f"Failed: {response.status_code}"}


def list_environments():
    """List all environments."""
    if not GITHUB_TOKEN:
        return {"error": "GitHub token not set (set CLAUDE_GITHUB_TOKEN, CLAUDE_SKILL_GITHUB_TOKEN, or GITHUB_TOKEN)"}

    url = f"{API_BASE}/environments"
    response = requests.get(url, headers=_headers())

    if response.status_code == 200:
        envs = response.json().get("environments", [])
        return {"success": True, "environments": [e["name"] for e in envs]}
    return {"error": f"Failed: {response.status_code}"}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def full_training_pipeline(goal, max_iterations=10, save_results=True, update_status=True):
    """
    Run full training pipeline with GitHub integration:
    1. Set commit status to pending
    2. Trigger workflow
    3. Wait for completion
    4. Save results to repo
    5. Update commit status
    6. Update dashboard
    """
    from github_workflow import run_cloud_training

    # Mark as pending
    if update_status:
        mark_training_status("pending", f"Training: {goal}")

    # Run training
    result = run_cloud_training(goal, max_iterations, wait=True)

    if result.get("success"):
        # Save results
        if save_results:
            save_training_results(result)

        # Extract history for dashboard
        artifacts = result.get("artifacts", [])
        history = []
        for a in artifacts:
            if "data" in a and "iterations" in a.get("data", {}):
                history = a["data"]["iterations"]
                break

        # Update dashboard
        if history:
            update_dashboard(history)

        # Mark success
        if update_status:
            final_state = result.get("artifacts", [{}])[0].get("data", {}).get("final_state", {})
            k_met = final_state.get("k_formation_met", False)
            desc = "K-formation achieved!" if k_met else f"Completed: z={final_state.get('z', 0):.4f}"
            mark_training_status("success", desc)

        return {"success": True, "result": result}
    else:
        if update_status:
            mark_training_status("failure", result.get("error", "Training failed"))
        return result


# =============================================================================
# QUICK ACCESS
# =============================================================================

if __name__ == "__main__":
    print("GitHub Advanced Integration")
    print("=" * 50)

    if GITHUB_TOKEN:
        print("Token: SET")

        # List variables
        vars_result = list_variables()
        print(f"\nVariables: {len(vars_result.get('variables', []))}")
        for v in vars_result.get("variables", [])[:5]:
            print(f"  {v['name']}: {v['value'][:50]}...")

        # List environments
        envs = list_environments()
        print(f"\nEnvironments: {envs.get('environments', [])}")
    else:
        print("Token: NOT SET")
