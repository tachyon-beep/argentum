"""Tests for persistence utilities."""

from datetime import UTC, datetime

import pytest

from argentum.models import (
    AgentResponse,
    Message,
    MessageType,
    OrchestrationPattern,
    OrchestrationResult,
    TerminationReason,
)
from argentum.persistence import ConversationSerializer, ConversationSession, JSONFileStore


def _sample_result() -> OrchestrationResult:
    message_system = Message(
        type=MessageType.SYSTEM,
        sender="orchestrator",
        content="Discuss renewable energy adoption.",
    )
    message_agent = Message(
        type=MessageType.ASSISTANT,
        sender="Analyst",
        content="We should invest in solar and wind infrastructure.",
        metadata={"agent_role": "advisor"},
    )
    response = AgentResponse(
        agent_name="Analyst",
        content="We should invest in solar and wind infrastructure.",
        confidence=0.82,
        citations=["energy_report.pdf"],
        metadata={"agent_role": "advisor"},
    )

    return OrchestrationResult(
        pattern=OrchestrationPattern.SEQUENTIAL,
        messages=[message_system, message_agent],
        final_outputs=[response],
        consensus="We should invest in solar and wind infrastructure.",
        termination_reason=TerminationReason.MAX_TURNS_REACHED,
        metadata={"topic": "renewables"},
        duration_seconds=3.5,
    )


def test_serializer_roundtrip() -> None:
    result = _sample_result()
    payload = ConversationSerializer.serialize_result(result)
    restored = ConversationSerializer.deserialize_result(payload)

    assert restored.pattern == result.pattern
    assert restored.consensus == result.consensus
    assert restored.termination_reason == result.termination_reason
    assert restored.metadata == result.metadata
    assert restored.duration_seconds == result.duration_seconds
    assert len(restored.messages) == len(result.messages)
    assert restored.messages[1].content == result.messages[1].content
    assert restored.final_outputs[0].metadata == result.final_outputs[0].metadata


@pytest.mark.asyncio
async def test_json_file_store_roundtrip(tmp_path) -> None:
    store = JSONFileStore(base_path=tmp_path)
    payload = {
        "session_id": "session-1",
        "metadata": {"topic": "energy"},
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "results": [],
    }

    await store.save_conversation("session-1", payload)
    loaded = await store.load_conversation("session-1")

    assert loaded["session_id"] == "session-1"
    assert loaded["metadata"] == {"topic": "energy"}

    await store.save_conversation("session-2", payload)
    sessions = await store.list_conversations()
    assert sessions == ["session-1", "session-2"]

    await store.delete_conversation("session-1")
    sessions_after_delete = await store.list_conversations()
    assert sessions_after_delete == ["session-2"]


@pytest.mark.asyncio
async def test_conversation_session_save_and_load(tmp_path) -> None:
    store = JSONFileStore(base_path=tmp_path)
    session = ConversationSession(store=store, metadata={"scenario": "debate"})

    result = _sample_result()
    await session.save(result)

    reloaded_session = ConversationSession(store=store, session_id=session.session_id)
    await reloaded_session.load()

    assert reloaded_session.metadata == {"scenario": "debate"}
    assert len(reloaded_session.history) == 1
    assert "Analyst" in reloaded_session.get_full_transcript()

    # Ensure append works without losing previous data.
    await reloaded_session.save(_sample_result())
    assert len(reloaded_session.history) == 2

    # Load again to confirm both entries persisted.
    fresh_session = ConversationSession(store=store, session_id=session.session_id)
    await fresh_session.load()
    assert len(fresh_session.history) == 2
