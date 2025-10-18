"""Tests for workspace compaction and purge commands."""

import json
from datetime import UTC, datetime, timedelta

from click.testing import CliRunner

from argentum.cli import cli
from argentum.workspace import WorkspaceManager
from argentum.workspace.knowledge import KnowledgeGraph, update_knowledge_graph
from argentum.workspace.warm_store import WarmCacheStore


def _create_highlight(session_dir, session_id: str, days_ago: int) -> None:
    timestamp = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    data = {
        "session_id": session_id,
        "timestamp": timestamp,
        "command": "debate",
        "project_id": "project",
        "topic": "Budget",
        "agents": ["Agent"],
        "quotes": [
            {"agent": "Agent", "content": "We should invest in recycling."}
        ],
        "items": [
            {
                "type": "statement",
                "agent": "Agent",
                "text": "We should invest in recycling.",
                "metadata": {},
            }
        ],
    }
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "highlights.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def test_project_compact_removes_old_warm_entries(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARGENTUM_WORKSPACES_DIR", str(tmp_path / "projects"))
    manager = WorkspaceManager()
    workspace = manager.create_project("compact", title="Compact Test")

    session_dir = workspace.root / "sessions" / "session-old"
    _create_highlight(session_dir, "session-old", days_ago=120)

    store = WarmCacheStore(workspace.root / "cache" / "warm" / "highlights.db")
    store.replace_session_entries("session-old", [
        {"type": "statement", "agent": "Agent", "text": "We should invest in recycling.", "metadata": {}}
    ])

    runner = CliRunner()
    result = runner.invoke(cli, ["project", "compact", "compact", "--days", "30"], catch_exceptions=False)
    assert result.exit_code == 0

    assert store.search("recycling") == []
    metadata = json.loads((session_dir / "highlights.json").read_text(encoding="utf-8")).get("metadata", {})
    assert "warm_cache_compacted_at" in metadata


def test_project_purge_removes_session(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARGENTUM_WORKSPACES_DIR", str(tmp_path / "projects"))
    manager = WorkspaceManager()
    workspace = manager.create_project("cleanup", title="Cleanup")

    session_id = "session-1"
    session_dir = workspace.root / "sessions" / session_id
    _create_highlight(session_dir, session_id, days_ago=10)
    transcript = {
        "messages": [
            {"type": "user", "sender": "orchestrator", "content": "Discuss recycling."},
            {"type": "assistant", "sender": "Agent", "content": "Recycling policy should include incentives."},
        ]
    }
    (session_dir / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")

    store = WarmCacheStore(workspace.root / "cache" / "warm" / "highlights.db")
    store.replace_session_entries(session_id, [
        {"type": "statement", "agent": "Agent", "text": "Recycling policy should include incentives.", "metadata": {}}
    ])

    highlights = json.loads((session_dir / "highlights.json").read_text(encoding="utf-8"))
    update_knowledge_graph(workspace, highlights)

    timeline_path = workspace.root / "timeline.jsonl"
    timeline_path.write_text(json.dumps({"session_id": session_id}) + "\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["project", "purge", "cleanup", "--session", session_id, "--force"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    assert not session_dir.exists()
    assert store.search("incentives") == []

    nodes = KnowledgeGraph(workspace.root / "knowledge").list_nodes()
    assert all(session_id not in (node.get("id") or "") for node in nodes)

    assert session_id not in timeline_path.read_text(encoding="utf-8")


def test_project_warm_rebuild(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARGENTUM_WORKSPACES_DIR", str(tmp_path / "projects"))
    manager = WorkspaceManager()
    workspace = manager.create_project("rebuild", title="Rebuild Warm Cache")

    for idx in range(2):
        session_id = f"session-{idx}"
        session_dir = workspace.root / "sessions" / session_id
        _create_highlight(session_dir, session_id, days_ago=idx)

    store = WarmCacheStore(workspace.root / "cache" / "warm" / "highlights.db")
    store.clear()
    assert store.count_entries() == 0

    runner = CliRunner()
    result = runner.invoke(cli, ["project", "warm-rebuild", "rebuild"], catch_exceptions=False)
    assert result.exit_code == 0
    assert store.count_entries() > 0


def test_project_timeline_command(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARGENTUM_WORKSPACES_DIR", str(tmp_path / "projects"))
    manager = WorkspaceManager()
    workspace = manager.create_project("timeline", title="Timeline")

    timeline_path = workspace.root / "timeline.jsonl"
    entries = [
        {"timestamp": "2024-01-01T10:00:00+00:00", "session_id": "one", "command": "debate", "topic": "Energy", "duration_seconds": 12.3},
        {"timestamp": "2024-01-02T12:00:00+00:00", "session_id": "two", "command": "advisory", "question": "Kubernetes?", "duration_seconds": 8.1},
    ]
    timeline_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["project", "timeline", "timeline", "--limit", "5"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "Timeline for timeline" in result.output
    assert "two" in result.output and "one" in result.output
