"""Tests for workspace knowledge graph and highlights."""

from pathlib import Path

from argentum.models import (
    AgentResponse,
    Message,
    MessageType,
    OrchestrationPattern,
    OrchestrationResult,
    TerminationReason,
)
from argentum.workspace import WorkspaceManager
from argentum.workspace.knowledge import (
    KnowledgeGraph,
    build_tts_markdown,
    build_tts_script,
    build_session_highlights,
    get_agent_activity,
    get_sessions_for_topic,
    index_highlights_in_warm_store,
    resolve_retrieval_config,
    save_session_highlights,
    update_knowledge_graph,
)
from argentum.workspace.warm_store import WarmCacheStore


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


def test_highlights_and_knowledge_graph(tmp_path) -> None:
    manager = _manager(tmp_path)
    workspace = manager.create_project("ai-show", title="AI Show")

    metadata = {
        "command": "debate",
        "topic": "Carbon Tax",
        "project_id": workspace.slug,
    }
    messages = [
        Message(type=MessageType.USER, sender="orchestrator", content="Discuss carbon tax."),
        Message(type=MessageType.ASSISTANT, sender="Agent 1", content="We should increase it."),
    ]
    final_outputs = [
        AgentResponse(agent_name="Agent 1", content="We should increase it.", confidence=0.9),
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

    highlights = build_session_highlights("session-1", metadata, result)
    saved_path = save_session_highlights(workspace, highlights)
    assert saved_path.exists()
    assert highlights["summary"]
    assert highlights["action_items"]

    update_knowledge_graph(workspace, highlights)
    index_highlights_in_warm_store(workspace, highlights)

    graph = KnowledgeGraph(workspace.root / "knowledge")
    nodes = graph.list_nodes()
    edges = graph.list_edges()

    assert any(node["type"] == "session" for node in nodes)
    assert any(node["type"] == "topic" for node in nodes)
    assert any(edge["type"] == "PARTICIPATED_IN" for edge in edges)

    store = WarmCacheStore(workspace.root / "cache" / "warm" / "highlights.db")
    results = store.search("carbon", limit=5)
    assert results
    assert results[0]["session_id"] == "session-1"

    topic_sessions = get_sessions_for_topic(graph, "Carbon Tax")
    assert "session-1" in topic_sessions

    activity = get_agent_activity(graph, "Agent 1")
    assert "session-1" in activity["sessions"]
    assert any("increase" in (statement["text"] or "").lower() for statement in activity["statements"])

    tts_markdown = build_tts_markdown(highlights)
    assert "Carbon Tax" in tts_markdown

    tts_script = build_tts_script(highlights)
    assert tts_script["segments"]
    assert tts_script["segments"][0]["speaker"] == "Narrator"
    assert "voice_profiles" in tts_script


def test_resolve_retrieval_config_defaults_and_overrides() -> None:
    manifest = {
        "retrieval": {
            "enabled": False,
            "limit": "5",
            "min_score": "0.25",
            "prefetch": False,
        }
    }

    config = resolve_retrieval_config(manifest)
    assert config["enabled"] is False
    assert config["prefetch"] is False
    assert config["limit"] == 5
    assert config["min_score"] == 0.25

    fallback_config = resolve_retrieval_config({"retrieval": {"limit": "not-a-number"}})
    assert fallback_config["limit"] == 3
    assert fallback_config["min_score"] == 0.0

    default_config = resolve_retrieval_config(None)
    assert default_config["enabled"] is True
    assert default_config["limit"] == 3
