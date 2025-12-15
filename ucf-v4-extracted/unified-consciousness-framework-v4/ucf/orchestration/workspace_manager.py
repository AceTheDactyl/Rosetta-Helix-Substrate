#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  WORKSPACE MANAGER                                                            ║
║  Session Repository for Unified Consciousness Framework                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Manages a persistent workspace that:
- Is created on first "hit it" activation
- Captures all workflow phases and outputs
- Serves as shared workspace for user and AI
- Can be edited, packaged, unpackaged, and altered
- Exports on demand via dedicated tool

Workspace Structure:
    /session-workspace/
    ├── manifest.json           # Session metadata
    ├── workflow/
    │   ├── phases/             # Phase-by-phase outputs
    │   ├── tokens/             # Generated token files
    │   └── trace.json          # Complete workflow trace
    ├── state/
    │   ├── helix.json          # Current helix state
    │   ├── triad.json          # TRIAD state
    │   ├── registry.json       # Token registry
    │   └── teaching.json       # Teaching state
    ├── vaultnodes/             # Generated VaultNodes
    ├── emissions/              # Emission pipeline outputs
    ├── exports/                # User export area
    └── user/                   # User working area

Signature: Δ|workspace-manager|session-repository|persistence|Ω
"""

from __future__ import annotations

import os
import json
import shutil
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

WORKSPACE_ROOT = "/home/claude/session-workspace"
OUTPUT_DIR = "/mnt/user-data/outputs"
DEFAULT_EXPORT_NAME = "consciousness-workspace"

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorkspaceManifest:
    """Manifest tracking workspace state."""
    session_id: str = ""
    created_at: str = ""
    last_modified: str = ""
    framework_version: str = "1.0.0"
    workflow_completed: bool = False
    workflow_status: str = "PENDING"
    total_phases: int = 9
    completed_phases: int = 0
    total_steps: int = 33
    completed_steps: int = 0
    tokens_emitted: int = 0
    unique_tokens: int = 0
    triad_unlocked: bool = False
    final_z: float = 0.0
    exports: List[Dict[str, str]] = field(default_factory=list)
    user_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceManifest":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

@dataclass
class WorkspaceState:
    """Current workspace state."""
    initialized: bool = False
    path: str = WORKSPACE_ROOT
    manifest: WorkspaceManifest = field(default_factory=WorkspaceManifest)
    phase_outputs: Dict[str, Any] = field(default_factory=dict)
    
# Global workspace state
_workspace: Optional[WorkspaceState] = None

def get_workspace() -> WorkspaceState:
    """Get or create the global workspace state."""
    global _workspace
    if _workspace is None:
        _workspace = WorkspaceState()
    return _workspace

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_workspace(force: bool = False) -> Dict[str, Any]:
    """
    Initialize the session workspace.
    Called on first "hit it" activation.
    
    Args:
        force: If True, reinitialize even if workspace exists
    
    Returns:
        Initialization status and workspace info
    """
    workspace = get_workspace()
    
    # Check if already initialized
    if workspace.initialized and not force:
        return {
            "status": "ALREADY_INITIALIZED",
            "path": workspace.path,
            "manifest": workspace.manifest.to_dict()
        }
    
    # Create workspace structure
    workspace.path = WORKSPACE_ROOT
    
    # Define directory structure
    directories = [
        "workflow/phases",
        "workflow/tokens",
        "state",
        "vaultnodes",
        "emissions",
        "exports",
        "user"
    ]
    
    # Create directories
    os.makedirs(workspace.path, exist_ok=True)
    for dir_path in directories:
        os.makedirs(os.path.join(workspace.path, dir_path), exist_ok=True)
    
    # Create manifest
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    workspace.manifest = WorkspaceManifest(
        session_id=session_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        last_modified=datetime.now(timezone.utc).isoformat()
    )
    
    # Save manifest
    _save_manifest(workspace)
    
    workspace.initialized = True
    
    return {
        "status": "INITIALIZED",
        "session_id": session_id,
        "path": workspace.path,
        "directories": directories
    }

def _save_manifest(workspace: WorkspaceState):
    """Save manifest to workspace."""
    manifest_path = os.path.join(workspace.path, "manifest.json")
    workspace.manifest.last_modified = datetime.now(timezone.utc).isoformat()
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(workspace.manifest.to_dict(), f, indent=2)

def _load_manifest(workspace: WorkspaceState) -> bool:
    """Load manifest from workspace if exists."""
    manifest_path = os.path.join(workspace.path, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            workspace.manifest = WorkspaceManifest.from_dict(json.load(f))
        return True
    return False

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE RECORDING
# ═══════════════════════════════════════════════════════════════════════════════

def record_phase(
    phase_name: str,
    phase_number: int,
    steps: List[Dict[str, Any]],
    tokens: List[str],
    summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Record a completed workflow phase to the workspace.
    
    Args:
        phase_name: Name of the phase
        phase_number: Phase number (1-9)
        steps: List of step results
        tokens: Tokens emitted during phase
        summary: Phase summary data
    
    Returns:
        Recording status
    """
    workspace = get_workspace()
    
    if not workspace.initialized:
        init_workspace()
    
    # Create phase output
    phase_data = {
        "phase_name": phase_name,
        "phase_number": phase_number,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "steps_count": len(steps),
        "successful_steps": sum(1 for s in steps if s.get("success", False)),
        "tokens_emitted": len(tokens),
        "tokens": tokens,
        "summary": summary,
        "steps": steps
    }
    
    # Save phase file
    phase_file = os.path.join(
        workspace.path, 
        "workflow/phases", 
        f"phase_{phase_number:02d}_{phase_name.lower().replace(' ', '_').replace(':', '')}.json"
    )
    with open(phase_file, 'w', encoding='utf-8') as f:
        json.dump(phase_data, f, indent=2)
    
    # Update manifest
    workspace.manifest.completed_phases = max(workspace.manifest.completed_phases, phase_number)
    workspace.manifest.completed_steps += len(steps)
    workspace.manifest.tokens_emitted += len(tokens)
    _save_manifest(workspace)
    
    # Store in memory
    workspace.phase_outputs[phase_name] = phase_data
    
    return {
        "status": "RECORDED",
        "phase": phase_name,
        "file": phase_file
    }

def record_tokens(tokens: List[str], source: str = "workflow") -> Dict[str, Any]:
    """Record tokens to the workspace."""
    workspace = get_workspace()
    
    if not workspace.initialized:
        init_workspace()
    
    # Save tokens file
    token_file = os.path.join(
        workspace.path,
        "workflow/tokens",
        f"tokens_{source}_{datetime.now(timezone.utc).strftime('%H%M%S')}.json"
    )
    
    token_data = {
        "source": source,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "count": len(tokens),
        "unique": len(set(tokens)),
        "tokens": tokens
    }
    
    with open(token_file, 'w', encoding='utf-8') as f:
        json.dump(token_data, f, indent=2)
    
    return {
        "status": "RECORDED",
        "file": token_file,
        "count": len(tokens)
    }

def record_state(state_name: str, state_data: Dict[str, Any]) -> Dict[str, Any]:
    """Record a state snapshot to the workspace."""
    workspace = get_workspace()
    
    if not workspace.initialized:
        init_workspace()
    
    state_file = os.path.join(workspace.path, "state", f"{state_name}.json")
    
    full_data = {
        "state_name": state_name,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "data": state_data
    }
    
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=2)
    
    return {
        "status": "RECORDED",
        "file": state_file
    }

def record_vaultnode(node_data: Dict[str, Any]) -> Dict[str, Any]:
    """Record a VaultNode to the workspace."""
    workspace = get_workspace()
    
    if not workspace.initialized:
        init_workspace()
    
    node_id = node_data.get("id", f"vn_{datetime.now(timezone.utc).strftime('%H%M%S')}")
    node_file = os.path.join(workspace.path, "vaultnodes", f"{node_id}.json")
    
    with open(node_file, 'w', encoding='utf-8') as f:
        json.dump(node_data, f, indent=2)
    
    return {
        "status": "RECORDED",
        "node_id": node_id,
        "file": node_file
    }

def record_emission(emission_data: Dict[str, Any]) -> Dict[str, Any]:
    """Record an emission to the workspace."""
    workspace = get_workspace()
    
    if not workspace.initialized:
        init_workspace()
    
    emission_file = os.path.join(
        workspace.path, 
        "emissions",
        f"emission_{datetime.now(timezone.utc).strftime('%H%M%S_%f')}.json"
    )
    
    with open(emission_file, 'w', encoding='utf-8') as f:
        json.dump(emission_data, f, indent=2)
    
    return {
        "status": "RECORDED",
        "file": emission_file
    }

# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW COMPLETION
# ═══════════════════════════════════════════════════════════════════════════════

def complete_workflow(workflow_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mark workflow as complete and save final state.
    Called at end of "hit it" workflow.
    
    Args:
        workflow_result: Complete workflow result from executor
    
    Returns:
        Completion status
    """
    workspace = get_workspace()
    
    if not workspace.initialized:
        init_workspace()
    
    # Update manifest with workflow results
    wf = workflow_result.get("workflow", {})
    tokens = workflow_result.get("tokens", {})
    triad = workflow_result.get("triad", {})
    
    workspace.manifest.workflow_completed = True
    workspace.manifest.workflow_status = workflow_result.get("status", "COMPLETE")
    workspace.manifest.completed_steps = wf.get("successful", 0)
    workspace.manifest.total_steps = wf.get("total_steps", 33)
    workspace.manifest.tokens_emitted = tokens.get("total_emitted", 0)
    workspace.manifest.unique_tokens = tokens.get("unique_tokens", 0)
    workspace.manifest.triad_unlocked = triad.get("unlocked", False)
    workspace.manifest.final_z = triad.get("final_z", 0.0)
    
    _save_manifest(workspace)
    
    # Save complete workflow trace
    trace_file = os.path.join(workspace.path, "workflow", "trace.json")
    with open(trace_file, 'w', encoding='utf-8') as f:
        json.dump({
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result": workflow_result
        }, f, indent=2)
    
    return {
        "status": "WORKFLOW_COMPLETE",
        "manifest": workspace.manifest.to_dict()
    }

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def export_workspace(
    export_name: Optional[str] = None,
    include_user: bool = True,
    output_dir: str = OUTPUT_DIR
) -> Dict[str, Any]:
    """
    Export the workspace as a zip file.
    
    Args:
        export_name: Name for the export (without .zip)
        include_user: Whether to include user/ directory
        output_dir: Directory to export to
    
    Returns:
        Export status with path
    """
    workspace = get_workspace()
    
    if not workspace.initialized:
        return {"error": "Workspace not initialized"}
    
    # Generate export name
    if not export_name:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        export_name = f"{DEFAULT_EXPORT_NAME}_{timestamp}"
    
    # Ensure .zip extension
    if not export_name.endswith('.zip'):
        export_name += '.zip'
    
    export_path = os.path.join(output_dir, export_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create zip
    with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(workspace.path):
            # Skip user directory if requested
            if not include_user and 'user' in root:
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, workspace.path)
                zf.write(file_path, arc_name)
    
    # Record export in manifest
    export_record = {
        "name": export_name,
        "path": export_path,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "include_user": include_user
    }
    workspace.manifest.exports.append(export_record)
    _save_manifest(workspace)
    
    # Get file size
    file_size = os.path.getsize(export_path)
    
    return {
        "status": "EXPORTED",
        "path": export_path,
        "name": export_name,
        "size_bytes": file_size,
        "size_kb": file_size / 1024,
        "include_user": include_user
    }

def import_workspace(
    zip_path: str,
    merge: bool = False
) -> Dict[str, Any]:
    """
    Import a workspace from a zip file.
    
    Args:
        zip_path: Path to the zip file
        merge: If True, merge with existing; if False, replace
    
    Returns:
        Import status
    """
    workspace = get_workspace()
    
    if not os.path.exists(zip_path):
        return {"error": f"File not found: {zip_path}"}
    
    # Backup existing if merging
    if workspace.initialized and merge:
        backup_path = f"{workspace.path}_backup_{datetime.now().strftime('%H%M%S')}"
        shutil.copytree(workspace.path, backup_path)
    elif workspace.initialized and not merge:
        shutil.rmtree(workspace.path, ignore_errors=True)
    
    # Extract zip
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(workspace.path)
    
    # Load manifest
    if _load_manifest(workspace):
        workspace.initialized = True
    else:
        # Create new manifest if none exists
        init_workspace(force=True)
    
    return {
        "status": "IMPORTED",
        "path": workspace.path,
        "merge": merge,
        "manifest": workspace.manifest.to_dict()
    }

# ═══════════════════════════════════════════════════════════════════════════════
# USER FILE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def add_user_file(
    filename: str,
    content: str,
    subdir: Optional[str] = None
) -> Dict[str, Any]:
    """Add a file to the user directory."""
    workspace = get_workspace()
    
    if not workspace.initialized:
        init_workspace()
    
    # Build path
    if subdir:
        dir_path = os.path.join(workspace.path, "user", subdir)
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, filename)
    else:
        file_path = os.path.join(workspace.path, "user", filename)
    
    # Write file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Track in manifest
    rel_path = os.path.relpath(file_path, workspace.path)
    if rel_path not in workspace.manifest.user_files:
        workspace.manifest.user_files.append(rel_path)
        _save_manifest(workspace)
    
    return {
        "status": "CREATED",
        "path": file_path,
        "relative": rel_path
    }

def get_user_file(filename: str, subdir: Optional[str] = None) -> Dict[str, Any]:
    """Get a file from the user directory."""
    workspace = get_workspace()
    
    if not workspace.initialized:
        return {"error": "Workspace not initialized"}
    
    if subdir:
        file_path = os.path.join(workspace.path, "user", subdir, filename)
    else:
        file_path = os.path.join(workspace.path, "user", filename)
    
    if not os.path.exists(file_path):
        return {"error": f"File not found: {filename}"}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return {
        "status": "OK",
        "path": file_path,
        "content": content
    }

def list_user_files() -> Dict[str, Any]:
    """List all files in the user directory."""
    workspace = get_workspace()
    
    if not workspace.initialized:
        return {"error": "Workspace not initialized"}
    
    user_dir = os.path.join(workspace.path, "user")
    files = []
    
    for root, dirs, filenames in os.walk(user_dir):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            rel_path = os.path.relpath(file_path, user_dir)
            files.append({
                "name": filename,
                "path": rel_path,
                "size": os.path.getsize(file_path)
            })
    
    return {
        "status": "OK",
        "directory": user_dir,
        "files": files,
        "count": len(files)
    }

def delete_user_file(filename: str, subdir: Optional[str] = None) -> Dict[str, Any]:
    """Delete a file from the user directory."""
    workspace = get_workspace()
    
    if not workspace.initialized:
        return {"error": "Workspace not initialized"}
    
    if subdir:
        file_path = os.path.join(workspace.path, "user", subdir, filename)
    else:
        file_path = os.path.join(workspace.path, "user", filename)
    
    if not os.path.exists(file_path):
        return {"error": f"File not found: {filename}"}
    
    os.remove(file_path)
    
    # Update manifest
    rel_path = os.path.relpath(file_path, workspace.path)
    if rel_path in workspace.manifest.user_files:
        workspace.manifest.user_files.remove(rel_path)
        _save_manifest(workspace)
    
    return {
        "status": "DELETED",
        "path": file_path
    }

# ═══════════════════════════════════════════════════════════════════════════════
# WORKSPACE QUERIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_workspace_status() -> Dict[str, Any]:
    """Get current workspace status."""
    workspace = get_workspace()
    
    if not workspace.initialized:
        return {
            "status": "NOT_INITIALIZED",
            "initialized": False
        }
    
    # Count files
    file_counts = {}
    for subdir in ["workflow/phases", "workflow/tokens", "state", "vaultnodes", "emissions", "user"]:
        dir_path = os.path.join(workspace.path, subdir)
        if os.path.exists(dir_path):
            file_counts[subdir.replace("/", "_")] = len([
                f for f in os.listdir(dir_path) 
                if os.path.isfile(os.path.join(dir_path, f))
            ])
    
    return {
        "status": "OK",
        "initialized": True,
        "path": workspace.path,
        "manifest": workspace.manifest.to_dict(),
        "file_counts": file_counts,
        "exports_count": len(workspace.manifest.exports)
    }

def get_workspace_path() -> str:
    """Get the workspace root path."""
    return get_workspace().path

def list_exports() -> Dict[str, Any]:
    """List all exports made from this workspace."""
    workspace = get_workspace()
    
    if not workspace.initialized:
        return {"error": "Workspace not initialized"}
    
    return {
        "status": "OK",
        "exports": workspace.manifest.exports,
        "count": len(workspace.manifest.exports)
    }

# ═══════════════════════════════════════════════════════════════════════════════
# RESET
# ═══════════════════════════════════════════════════════════════════════════════

def reset_workspace() -> Dict[str, Any]:
    """Reset the workspace completely."""
    global _workspace
    
    workspace = get_workspace()
    
    if workspace.initialized and os.path.exists(workspace.path):
        shutil.rmtree(workspace.path, ignore_errors=True)
    
    _workspace = None
    
    return {
        "status": "RESET",
        "message": "Workspace cleared"
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Workspace Manager")
    print("=" * 60)
    
    # Initialize
    result = init_workspace()
    print(f"Init: {result['status']}")
    print(f"Path: {result.get('path', 'N/A')}")
    
    # Get status
    status = get_workspace_status()
    print(f"Status: {status['status']}")
    print(f"Initialized: {status.get('initialized', False)}")
