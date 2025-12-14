#!/usr/bin/env python3
"""
GitHub Workflow Trigger for Rosetta-Helix-Substrate Skill

Trigger autonomous training workflows from within Claude.ai skill sessions.
Requires: CLAUDE_SKILL_GITHUB_TOKEN or GITHUB_TOKEN environment variable.
"""

import os
import json
import time

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configuration
GITHUB_TOKEN = os.environ.get("CLAUDE_SKILL_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
REPO_OWNER = "AceTheDactyl"
REPO_NAME = "Rosetta-Helix-Substrate"
WORKFLOW_FILE = "autonomous-training.yml"


def trigger_workflow(goal="Achieve K-formation", max_iterations=10, initial_z=0.3, branch="main"):
    """Trigger the autonomous training workflow."""
    if not REQUESTS_AVAILABLE:
        return {"error": "requests package not installed"}
    if not GITHUB_TOKEN:
        return {"error": "GITHUB_TOKEN not set"}

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{WORKFLOW_FILE}/dispatches"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    payload = {"ref": branch, "inputs": {"goal": goal, "max_iterations": str(max_iterations), "initial_z": str(initial_z)}}

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 204:
        return {"success": True, "message": f"Workflow triggered: {goal}"}
    return {"error": f"Failed: {response.status_code}", "details": response.text}


def get_latest_run():
    """Get the latest workflow run status."""
    if not REQUESTS_AVAILABLE:
        return {"error": "requests package not installed"}
    if not GITHUB_TOKEN:
        return {"error": "GITHUB_TOKEN not set"}

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{WORKFLOW_FILE}/runs"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

    response = requests.get(url, headers=headers, params={"per_page": 1})
    if response.status_code == 200:
        runs = response.json().get("workflow_runs", [])
        if runs:
            run = runs[0]
            return {
                "id": run["id"],
                "status": run["status"],
                "conclusion": run["conclusion"],
                "created_at": run["created_at"],
                "url": run["html_url"]
            }
    return {"error": "No runs found"}


def wait_for_completion(run_id, timeout=600, poll_interval=15):
    """Wait for a workflow run to complete."""
    if not REQUESTS_AVAILABLE:
        return {"error": "requests package not installed"}
    if not GITHUB_TOKEN:
        return {"error": "GITHUB_TOKEN not set"}

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs/{run_id}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

    start = time.time()
    while time.time() - start < timeout:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            run = response.json()
            if run["status"] == "completed":
                return {"success": True, "conclusion": run["conclusion"], "url": run["html_url"]}
        time.sleep(poll_interval)
    return {"error": "Timeout"}


def download_artifacts(run_id):
    """Download artifacts from a completed run."""
    if not REQUESTS_AVAILABLE:
        return {"error": "requests package not installed"}
    if not GITHUB_TOKEN:
        return {"error": "GITHUB_TOKEN not set"}

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs/{run_id}/artifacts"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return {"error": f"Failed to get artifacts: {response.status_code}"}

    artifacts = response.json().get("artifacts", [])
    if not artifacts:
        return {"error": "No artifacts found"}

    results = []
    import zipfile
    from io import BytesIO

    for artifact in artifacts:
        dl_response = requests.get(artifact["archive_download_url"], headers=headers)
        if dl_response.status_code == 200:
            with zipfile.ZipFile(BytesIO(dl_response.content)) as zf:
                for filename in zf.namelist():
                    content = zf.read(filename).decode("utf-8", errors="replace")
                    try:
                        data = json.loads(content)
                        results.append({"file": filename, "data": data})
                    except:
                        results.append({"file": filename, "content": content[:2000]})

    return {"success": True, "artifacts": results}


def run_cloud_training(goal="Achieve K-formation", max_iterations=10, initial_z=0.3, wait=True, timeout=600):
    """
    Trigger GitHub workflow, optionally wait for results.

    This runs the full autonomous training loop in the cloud using Claude API,
    then returns the results back to your session.
    """
    print(f"Triggering cloud training: {goal}")

    result = trigger_workflow(goal, max_iterations, initial_z)
    if "error" in result:
        return result

    print("Workflow triggered, waiting for run to start...")
    time.sleep(5)

    run_info = get_latest_run()
    if "error" in run_info:
        return run_info

    run_id = run_info["id"]
    print(f"Run started: {run_info['url']}")

    if not wait:
        return {"success": True, "run_id": run_id, "url": run_info["url"], "message": "Workflow running in background"}

    print(f"Waiting for completion (timeout: {timeout}s)...")
    completion = wait_for_completion(run_id, timeout)
    if "error" in completion:
        return completion

    print(f"Completed: {completion['conclusion']}")

    if completion["conclusion"] == "success":
        print("Downloading results...")
        artifacts = download_artifacts(run_id)
        return {
            "success": True,
            "run_id": run_id,
            "url": completion["url"],
            "conclusion": completion["conclusion"],
            "artifacts": artifacts.get("artifacts", [])
        }

    return {"success": False, "conclusion": completion["conclusion"], "url": completion["url"]}


# Quick status check
if __name__ == "__main__":
    print("GitHub Workflow Tools")
    print("=" * 50)
    if GITHUB_TOKEN:
        print("Token: SET")
        status = get_latest_run()
        if "error" not in status:
            print(f"Latest run: {status['status']} / {status['conclusion']}")
            print(f"URL: {status['url']}")
        else:
            print(f"Status: {status}")
    else:
        print("Token: NOT SET")
        print("Set CLAUDE_SKILL_GITHUB_TOKEN or GITHUB_TOKEN")
