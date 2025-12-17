#!/usr/bin/env python3
"""
GitHub Workflow Trigger for Rosetta-Helix-Substrate

Trigger the autonomous training workflow via GitHub API and fetch results.
Requires: CLAUDE_GITHUB_TOKEN (preferred), CLAUDE_SKILL_GITHUB_TOKEN, or GITHUB_TOKEN with repo access.

Usage:
    python trigger_workflow.py --goal "Achieve K-formation" --iterations 10
"""

import os
import sys
import json
import time
import argparse
import zipfile
import tempfile
from io import BytesIO

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)


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
REPO_OWNER = "AceTheDactyl"  # Update with your GitHub username
REPO_NAME = "Rosetta-Helix-Substrate"
WORKFLOW_FILE = "autonomous-training.yml"


def trigger_workflow(
    goal: str = "Achieve K-formation",
    max_iterations: int = 10,
    initial_z: float = 0.3,
    branch: str = "main"
) -> dict:
    """Trigger the autonomous training workflow."""

    if not GITHUB_TOKEN:
        return {"error": "GitHub token not set (set CLAUDE_GITHUB_TOKEN, CLAUDE_SKILL_GITHUB_TOKEN, or GITHUB_TOKEN)"}

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{WORKFLOW_FILE}/dispatches"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    payload = {
        "ref": branch,
        "inputs": {
            "goal": goal,
            "max_iterations": str(max_iterations),
            "initial_z": str(initial_z),
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 204:
        return {"success": True, "message": "Workflow triggered successfully"}
    else:
        return {"error": f"Failed to trigger workflow: {response.status_code}", "details": response.text}


def get_latest_run(workflow_file: str = WORKFLOW_FILE) -> dict:
    """Get the latest workflow run."""

    if not GITHUB_TOKEN:
        return {"error": "GitHub token not set (set CLAUDE_GITHUB_TOKEN, CLAUDE_SKILL_GITHUB_TOKEN, or GITHUB_TOKEN)"}

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{workflow_file}/runs"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    response = requests.get(url, headers=headers, params={"per_page": 1})

    if response.status_code == 200:
        data = response.json()
        if data["workflow_runs"]:
            run = data["workflow_runs"][0]
            return {
                "id": run["id"],
                "status": run["status"],
                "conclusion": run["conclusion"],
                "created_at": run["created_at"],
                "html_url": run["html_url"],
            }
        return {"error": "No workflow runs found"}
    else:
        return {"error": f"Failed to get runs: {response.status_code}"}


def wait_for_completion(run_id: int, timeout: int = 600, poll_interval: int = 10) -> dict:
    """Wait for a workflow run to complete."""

    if not GITHUB_TOKEN:
        return {"error": "GitHub token not set (set CLAUDE_GITHUB_TOKEN, CLAUDE_SKILL_GITHUB_TOKEN, or GITHUB_TOKEN)"}

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs/{run_id}"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    start_time = time.time()

    while time.time() - start_time < timeout:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            run = response.json()
            status = run["status"]
            conclusion = run["conclusion"]

            print(f"Status: {status}, Conclusion: {conclusion}")

            if status == "completed":
                return {
                    "success": True,
                    "status": status,
                    "conclusion": conclusion,
                    "html_url": run["html_url"],
                }

        time.sleep(poll_interval)

    return {"error": "Timeout waiting for workflow completion"}


def download_artifacts(run_id: int) -> dict:
    """Download artifacts from a workflow run."""

    if not GITHUB_TOKEN:
        return {"error": "GitHub token not set (set CLAUDE_GITHUB_TOKEN, CLAUDE_SKILL_GITHUB_TOKEN, or GITHUB_TOKEN)"}

    # Get artifacts list
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/runs/{run_id}/artifacts"

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return {"error": f"Failed to get artifacts: {response.status_code}"}

    artifacts = response.json().get("artifacts", [])

    if not artifacts:
        return {"error": "No artifacts found"}

    results = []

    for artifact in artifacts:
        artifact_url = artifact["archive_download_url"]
        artifact_name = artifact["name"]

        # Download artifact
        dl_response = requests.get(artifact_url, headers=headers)

        if dl_response.status_code == 200:
            # Extract ZIP
            with zipfile.ZipFile(BytesIO(dl_response.content)) as zf:
                for filename in zf.namelist():
                    content = zf.read(filename).decode("utf-8", errors="replace")

                    # Try to parse as JSON
                    try:
                        data = json.loads(content)
                        results.append({
                            "artifact": artifact_name,
                            "file": filename,
                            "data": data,
                        })
                    except json.JSONDecodeError:
                        results.append({
                            "artifact": artifact_name,
                            "file": filename,
                            "content": content[:5000],  # Truncate large text
                        })

    return {"success": True, "artifacts": results}


def run_and_wait(
    goal: str = "Achieve K-formation",
    max_iterations: int = 10,
    initial_z: float = 0.3,
    timeout: int = 600,
    branch: str = "main"
) -> dict:
    """Trigger workflow, wait for completion, and download results."""

    print(f"Triggering workflow with goal: {goal}")

    # Trigger
    trigger_result = trigger_workflow(goal, max_iterations, initial_z, branch)
    if "error" in trigger_result:
        return trigger_result

    print("Workflow triggered, waiting for run to appear...")
    time.sleep(5)

    # Get the run ID
    run_info = get_latest_run()
    if "error" in run_info:
        return run_info

    run_id = run_info["id"]
    print(f"Found run {run_id}: {run_info['html_url']}")

    # Wait for completion
    print(f"Waiting for completion (timeout: {timeout}s)...")
    completion = wait_for_completion(run_id, timeout)
    if "error" in completion:
        return completion

    print(f"Workflow completed: {completion['conclusion']}")

    # Download artifacts
    print("Downloading artifacts...")
    artifacts = download_artifacts(run_id)

    return {
        "success": True,
        "run_id": run_id,
        "run_url": run_info["html_url"],
        "conclusion": completion["conclusion"],
        "artifacts": artifacts.get("artifacts", []),
    }


# Tool handler for skill integration
def handle_trigger_workflow(input: dict) -> dict:
    """Handle trigger_workflow tool call."""
    goal = input.get("goal", "Achieve K-formation by reaching THE LENS")
    max_iterations = input.get("max_iterations", 10)
    initial_z = input.get("initial_z", 0.3)
    wait = input.get("wait_for_results", True)
    timeout = input.get("timeout", 600)

    if wait:
        return run_and_wait(goal, max_iterations, initial_z, timeout)
    else:
        return trigger_workflow(goal, max_iterations, initial_z)


def handle_get_workflow_status(input: dict) -> dict:
    """Handle get_workflow_status tool call."""
    return get_latest_run()


def handle_download_workflow_results(input: dict) -> dict:
    """Handle download_workflow_results tool call."""
    run_id = input.get("run_id")
    if not run_id:
        # Get latest run
        run_info = get_latest_run()
        if "error" in run_info:
            return run_info
        run_id = run_info["id"]

    return download_artifacts(run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger GitHub workflow")
    parser.add_argument("--goal", "-g", default="Achieve K-formation by reaching THE LENS")
    parser.add_argument("--iterations", "-i", type=int, default=10)
    parser.add_argument("--initial-z", "-z", type=float, default=0.3)
    parser.add_argument("--timeout", "-t", type=int, default=600)
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for results")
    parser.add_argument("--branch", "-b", default="main")

    args = parser.parse_args()

    if args.no_wait:
        result = trigger_workflow(args.goal, args.iterations, args.initial_z, args.branch)
    else:
        result = run_and_wait(args.goal, args.iterations, args.initial_z, args.timeout, args.branch)

    print(json.dumps(result, indent=2, default=str))
