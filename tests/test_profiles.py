"""Tests for workspace agent profiles."""

import json
from pathlib import Path

from argentum.agents.base import AgentConfig, Role
from argentum.workspace import WorkspaceManager, load_agent_profile


def test_load_agent_profile_creates_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARGENTUM_WORKSPACES_DIR", str(tmp_path / "projects"))
    manager = WorkspaceManager()
    workspace = manager.create_project("panel", title="Advisory Panel")

    fallback = AgentConfig(
        name="Chief Security Officer",
        role=Role.ADVISOR,
        persona="Default persona",
        model="gpt-4",
        temperature=0.7,
        max_tokens=400,
        metadata={"slug": "security"},
    )

    config = load_agent_profile(workspace, "security", fallback)
    assert config == fallback

    profile_path = workspace.root / "agents" / "security" / "profile.json"
    assert profile_path.exists()

    # Override persona and temperature in profile
    overrides = json.loads(profile_path.read_text(encoding="utf-8"))
    overrides["persona"] = "Custom security persona"
    overrides["temperature"] = 0.3
    profile_path.write_text(json.dumps(overrides, indent=2), encoding="utf-8")

    updated = load_agent_profile(workspace, "security", fallback)
    assert updated.persona == "Custom security persona"
    assert updated.temperature == 0.3
    assert updated.metadata.get("slug") == "security"
