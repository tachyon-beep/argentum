"""Tests for shared session runner utilities in the CLI."""

import asyncio
from pathlib import Path

import pytest

from argentum.cli import _persist_session_result, _prepare_session_environment
from argentum.models import (
    AgentResponse,
    Message,
    MessageType,
    OrchestrationPattern,
    OrchestrationResult,
    TerminationReason,
)
from argentum.persistence import ConversationSession
from argentum.workspace import WorkspaceManager


@pytest.mark.asyncio
async def test_session_environment_and_persistence(tmp_path, monkeypatch) -> None:
    workspace_root = tmp_path / "projects"
    monkeypatch.setenv("ARGENTUM_WORKSPACES_DIR", str(workspace_root))

    manager = WorkspaceManager()
    workspace = manager.create_project("podcast", title="AI Podcast")

    env = _prepare_session_environment(
        command="debate",
        project="podcast",
        session_id=None,
        seed="Carbon Tax",
        summary_mode=None,
        summary_command=None,
    )

    assert env.workspace is not None
    assert env.workspace.slug == "podcast"
    assert env.session_dir == workspace.root / "sessions" / env.session_id
    assert env.store.base_path == env.session_dir
    assert env.memory_store is not None
    assert env.summary_strategy.name == "heuristic"
    retrieval_meta = env.metadata.get("retrieval")
    assert retrieval_meta
    assert retrieval_meta["query"] == "Carbon Tax"
    assert retrieval_meta["limit"] == 3
    assert retrieval_meta["hit_count"] == 0
    assert env.metadata.get("retrieved_docs") == []
    assert env.metadata.get("retrieval_history") == []

    metadata = dict(env.metadata)
    metadata.update(
        {
            "topic": "Carbon Tax",
            "ministers": ["Agent 1"],
            "rounds": 3,
            "summary_mode": env.summary_strategy.name,
        }
    )

    session = ConversationSession(
        store=env.store,
        session_id=env.session_id,
        metadata=metadata,
    )

    messages = [
        Message(type=MessageType.USER, sender="orchestrator", content="Discuss the carbon tax policy."),
        Message(type=MessageType.ASSISTANT, sender="Agent 1", content="We should gradually increase it."),
    ]
    final_outputs = [
        AgentResponse(agent_name="Agent 1", content="We should gradually increase it.", confidence=0.9),
    ]
    result = OrchestrationResult(
        pattern=OrchestrationPattern.GROUP_CHAT,
        messages=messages,
        final_outputs=final_outputs,
        consensus="Increase carbon tax gradually.",
        termination_reason=TerminationReason.CONSENSUS_REACHED,
        metadata={"agent_names": ["Agent 1"]},
        duration_seconds=15.0,
    )

    transcript_path = await _persist_session_result(env, session, metadata, result)

    assert transcript_path == env.session_dir / "transcript.json"
    assert transcript_path.exists()

    highlights_path = env.session_dir / "highlights.json"
    assert highlights_path.exists()

    knowledge_nodes = workspace.root / "knowledge" / "nodes.jsonl"
    assert knowledge_nodes.exists()

    warm_cache_db = workspace.root / "cache" / "warm" / "highlights.db"
    assert warm_cache_db.exists()

    timeline = workspace.root / "timeline.jsonl"
    assert timeline.exists()
    assert timeline.read_text().strip() != ""


def test_session_environment_requires_project(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARGENTUM_WORKSPACES_DIR", str(tmp_path / "projects"))
    with pytest.raises(FileNotFoundError):
        _prepare_session_environment(
            command="debate",
            project="missing",
            session_id=None,
            seed="Topic",
            summary_mode=None,
            summary_command=None,
        )
