"""Tests for CLI knowledge search with cold transcript fallback."""

import json
from pathlib import Path

from click.testing import CliRunner

from argentum.cli import cli
from argentum.workspace import WorkspaceManager


def test_knowledge_search_falls_back_to_cold_transcripts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARGENTUM_WORKSPACES_DIR", str(tmp_path / "projects"))

    manager = WorkspaceManager()
    workspace = manager.create_project("policy-show", title="Policy Show")

    session_dir = workspace.root / "sessions" / "session-1"
    session_dir.mkdir(parents=True, exist_ok=True)
    transcript = {
        "messages": [
            {"type": "user", "sender": "orchestrator", "content": "Let's discuss recycling policy."},
            {"type": "assistant", "sender": "Agent", "content": "Recycling policy should include incentives."},
        ]
    }
    (session_dir / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "project",
            "knowledge",
            "policy-show",
            "--search",
            "recycling",
        ],
    )

    assert result.exit_code == 0
    assert "Cold transcript search" in result.output
    assert "session-1" in result.output
    assert "Recycling policy should include" in result.output
