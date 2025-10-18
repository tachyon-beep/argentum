"""Demonstration of two agents retrieving shared project documents mid-session."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shutil
from typing import Any

from argentum.agents.base import AgentConfig
from argentum.agents.llm_agent import LLMAgent
from argentum.llm.provider import LLMProvider
from argentum.models import (
    Message,
    MessageType,
    OrchestrationPattern,
    OrchestrationResult,
    Role,
    Task,
    TerminationReason,
)
from argentum.persistence import ConversationSession, JSONFileStore
from argentum.workspace import WorkspaceManager
from argentum.workspace.knowledge import (
    SessionRetriever,
    WorkspaceDocumentIndex,
    build_session_highlights,
)


class ScriptedProvider(LLMProvider):
    """Deterministic provider used for the demo."""

    def __init__(self, name: str, *, trigger_retrieval: bool) -> None:
        self.name = name
        self.trigger_retrieval = trigger_retrieval
        self.calls = 0

    def count_tokens(self, messages: list[Message]) -> int:  # pragma: no cover - trivial
        return sum(len(message.content) for message in messages)

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls += 1
        if self.calls == 1 and self.trigger_retrieval:
            return "<<retrieve: carbon dividend market data>>"
        if self.name == "Minister of Finance":
            return (
                "Building on [Doc 1], we should move in $5 increments so the dividend cycle "
                "and community solar funds stay balanced."
            )
        return (
            "[Doc 1] shows clean-tech jobs jumped 12% when rebates accompanied pricing, "
            "so let's pair the increase with community investment messaging."
        )

    async def generate_with_tools(  # pragma: no cover - unused in demo
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    def get_model_name(self) -> str:  # pragma: no cover - simple
        return f"scripted-{self.name}"


async def main() -> None:
    base_path = Path(__file__).resolve().parent / "demo_workspace"
    os.environ["ARGENTUM_WORKSPACES_DIR"] = str(base_path)

    manager = WorkspaceManager(base_path=base_path)
    project_root = base_path / "two-actor-demo"
    if project_root.exists():
        shutil.rmtree(project_root)
    project = manager.create_project("two-actor-demo", title="Two Actor Retrieval Demo")

    doc_path = project.root / "docs" / "carbon-briefing.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        "\n".join(
            [
                "Carbon Pricing Briefing",
                "",
                "- Incremental increases of $5 per ton keep exporters competitive.",
                "- Recycling revenue into community solar programmes builds durable support.",
                "- 2025 market survey: clean-tech employment rose 12% alongside dividend rebates.",
            ]
        ),
        encoding="utf-8",
    )

    index = WorkspaceDocumentIndex(project)
    index.ingest_files([doc_path], tags=("policy", "demo"))

    retriever = SessionRetriever(index, default_limit=3, default_min_score=0.0)

    topic = "Should we increase the carbon tax this quarter?"
    session_id = "two-actor-demo"

    finance_agent = LLMAgent(
        config=AgentConfig(
            name="Minister of Finance",
            role=Role.ADVISOR,
            persona="Balance fiscal sustainability with market stability.",
            model="scripted-finance",
            temperature=0.0,
            max_tokens=120,
            speaking_style="boardroom",
            speech_tags=["measured", "formal"],
            tts_voice="boardroom_male",
        ),
        provider=ScriptedProvider("Minister of Finance", trigger_retrieval=True),
        retriever=retriever,
    )
    environment_agent = LLMAgent(
        config=AgentConfig(
            name="Minister of Environment",
            role=Role.ADVISOR,
            persona="Promote sustainable growth and community outcomes.",
            model="scripted-environment",
            temperature=0.0,
            max_tokens=120,
            speaking_style="podcast",
            speech_tags=["warm", "optimistic"],
            tts_voice="podcast_female",
        ),
        provider=ScriptedProvider("Minister of Environment", trigger_retrieval=False),
        retriever=retriever,
    )

    context: dict[str, Any] = {
        "topic": topic,
        "knowledge_documents": [],
        "retrieval_history": [],
    }

    task_description = (
        f"Debate: {topic}\n"
        "1. Weigh fiscal + environmental considerations\n"
        "2. Reference evidence and suggest next steps\n"
        "3. Ask for new evidence with <<retrieve: keywords>> when needed."
    )

    messages = [Message(type=MessageType.USER, sender="orchestrator", content=task_description)]

    finance_response = await finance_agent.generate_response(messages=messages, context=context)
    messages.append(
        Message(
            type=MessageType.ASSISTANT,
            sender=finance_agent.name,
            content=finance_response.content,
            metadata={"agent_role": finance_agent.role.value, **finance_response.metadata},
        )
    )

    environment_response = await environment_agent.generate_response(messages=messages, context=context)
    messages.append(
        Message(
            type=MessageType.ASSISTANT,
            sender=environment_agent.name,
            content=environment_response.content,
            metadata={"agent_role": environment_agent.role.value, **environment_response.metadata},
        )
    )

    retrieved_docs = context["knowledge_documents"]
    metadata = {
        "command": "demo",
        "session_id": session_id,
        "project_id": project.slug,
        "project_display_name": "Two Actor Retrieval Demo",
        "topic": topic,
        "retrieval": {
            "enabled": True,
            "prefetch": False,
            "limit": retriever.default_limit,
            "min_score": retriever.default_min_score,
            "query": topic,
            "hit_count": len(retrieved_docs),
        },
        "retrieved_docs": retrieved_docs,
        "retrieval_history": retriever.history,
    }

    result = OrchestrationResult(
        pattern=OrchestrationPattern.GROUP_CHAT,
        messages=messages,
        final_outputs=[finance_response, environment_response],
        consensus=environment_response.content,
        termination_reason=TerminationReason.CONSENSUS_REACHED,
        metadata={
            "agent_names": [finance_agent.name, environment_agent.name],
            "num_agents": 2,
            "statistics": {"total_turns": 2},
        },
        duration_seconds=None,
    )

    session_dir = project.root / "sessions" / session_id
    store = JSONFileStore(base_path=session_dir)
    session = ConversationSession(store=store, session_id=session_id, metadata=metadata)
    await session.save(result)

    highlights = build_session_highlights(session_id, metadata, result)
    (session_dir / "highlights.json").write_text(highlights_to_pretty_json(highlights), encoding="utf-8")

    print("=== Conversation ===")
    for message in messages:
        if message.type != MessageType.ASSISTANT:
            continue
        citations = message.metadata.get("citations")
        citation_str = f" (citations: {', '.join(c['label'] for c in citations)})" if citations else ""
        print(f"[{message.sender}]{citation_str}: {message.content}")

    print("\n=== Retrieval History ===")
    history = highlights.get("retrieval_history") or []
    if not history:
        print("(none)")
    else:
        for event in history:
            labels = ", ".join(event.get("labels") or [])
            print(f"- {event.get('query')}: {labels or 'no results'}")

    print("\n=== Summary ===")
    print(highlights["summary"])

    print("\nHighlights written to:", session_dir / "highlights.json")


def highlights_to_pretty_json(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
