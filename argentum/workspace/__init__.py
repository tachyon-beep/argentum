"""Persistent project workspace utilities."""

from pathlib import Path

from argentum.knowledge.document_store import DocumentStore
from argentum.workspace.manager import ProjectWorkspace, WorkspaceManager
from argentum.workspace.profiles import apply_speech_defaults, load_agent_profile, save_agent_profile


def get_default_workspace_root() -> Path:
    """Return the default root directory for project workspaces."""
    return WorkspaceManager().base_path


__all__ = [
    "ProjectWorkspace",
    "WorkspaceManager",
    "DocumentStore",
    "get_default_workspace_root",
    "load_agent_profile",
    "save_agent_profile",
    "apply_speech_defaults",
]
