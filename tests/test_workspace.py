"""Tests for project workspace scaffolding."""

import json
from pathlib import Path

import pytest

from argentum.workspace import WorkspaceManager


def _manager(tmp_path: Path) -> WorkspaceManager:
    base = tmp_path / "projects"
    user_templates = tmp_path / "user_templates"
    system_templates = tmp_path / "system_templates"
    system_templates.mkdir(parents=True, exist_ok=True)
    return WorkspaceManager(
        base_path=base,
        user_templates_path=user_templates,
        system_templates_path=system_templates,
    )


def test_workspace_creation_default_structure(tmp_path) -> None:
    manager = _manager(tmp_path)

    workspace = manager.create_project("knit-cast", title="AI Knitting Circle")

    assert workspace.root.exists()
    for rel in ("agents", "sessions", "cache/hot", "cache/warm", "cache/cold"):
        assert (workspace.root / rel).exists()

    manifest = workspace.load_manifest()
    assert manifest["project_id"] == "knit-cast"
    assert manifest["display_name"] == "AI Knitting Circle"
    assert manifest["storage"]["base_path"] == str(workspace.root)


def test_workspace_creation_with_template(tmp_path) -> None:
    manager = _manager(tmp_path)

    template_dir = manager.system_templates_path / "podcast"
    (template_dir / "agents").mkdir(parents=True, exist_ok=True)
    (template_dir / "agents" / "README.md").write_text("agents go here", encoding="utf-8")
    template_manifest = {
        "display_name": "Template Title",
        "description": "Template description",
        "default_agents": ["host", "guest"],
    }
    (template_dir / "project.json").write_text(json.dumps(template_manifest), encoding="utf-8")

    workspace = manager.create_project("pod-show", template="podcast", title="Actual Title")

    assert (workspace.root / "agents" / "README.md").exists()
    manifest = workspace.load_manifest()
    assert manifest["project_id"] == "pod-show"
    assert manifest["display_name"] == "Actual Title"  # CLI override wins
    assert manifest["default_agents"] == ["host", "guest"]
    assert manifest["storage"]["base_path"] == str(workspace.root)


def test_get_project_missing(tmp_path) -> None:
    manager = _manager(tmp_path)
    with pytest.raises(FileNotFoundError):
        manager.get_project("unknown")
