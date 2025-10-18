"""LLM-powered agent implementation."""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from argentum.agents.base import Agent, AgentConfig
from argentum.memory.agent_memory import AgentMemoryStore
from argentum.llm.provider import LLMProvider
from argentum.models import AgentResponse, Message, MessageType

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from argentum.workspace.knowledge import SessionRetriever


class LLMAgent(Agent):
    """An agent powered by a Large Language Model."""

    RETRIEVE_PATTERN = re.compile(r"<<\s*retrieve\s*:(.*?)>>", re.IGNORECASE | re.DOTALL)

    def __init__(
        self,
        config: AgentConfig,
        provider: LLMProvider,
        *,
        memory_store: AgentMemoryStore | None = None,
        retriever: "SessionRetriever | None" = None,
    ):
        """Initialize the LLM agent.

        Args:
            config: Agent configuration
            provider: LLM provider instance
            memory_store: Optional persistent memory store
        """
        super().__init__(config)
        self.provider = provider
        self.memory_store = memory_store
        self.retriever = retriever

    async def generate_response(
        self,
        messages: list[Message],
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Generate a response using the LLM provider with optional retrieval."""

        context = context or {}
        if not isinstance(context, dict):  # defensive: ensure we can mutate
            context = dict(context)

        knowledge_documents = context.get("knowledge_documents")
        if not isinstance(knowledge_documents, list):
            knowledge_documents = []
            context["knowledge_documents"] = knowledge_documents

        if self.retriever:
            self.retriever.register_documents(knowledge_documents)

        retrieval_history = context.get("retrieval_history")
        if not isinstance(retrieval_history, list):
            retrieval_history = self.retriever.history if self.retriever else []
            context["retrieval_history"] = retrieval_history

        topic = self._detect_topic(context)

        attempts = 0
        max_attempts = 2
        final_content = ""
        final_citations: list[dict[str, Any]] = []
        retrieval_events: list[dict[str, Any]] = []

        # Main generation loop (allows a single retrieval cycle)
        while True:
            attempts += 1

            system_prompt, candidate_citations = self._compose_system_prompt(
                topic=topic,
                context=context,
                knowledge_documents=knowledge_documents,
            )

            system_message = Message(
                type=MessageType.SYSTEM,
                sender="system",
                content=system_prompt,
            )

            all_messages = [system_message, *messages]
            provider_messages = [{"role": msg.type.value, "content": msg.content} for msg in all_messages]

            response_content = await self.provider.generate(
                messages=provider_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            retrieval_query = self._extract_retrieval_query(response_content)
            if (
                retrieval_query
                and self.retriever
                and attempts <= max_attempts
            ):
                new_docs = self.retriever.search(retrieval_query)
                if new_docs:
                    # Append new documents and continue the loop to regenerate response with updated context
                    knowledge_documents.extend(new_docs)
                    context["knowledge_documents"] = knowledge_documents
                    context.setdefault("retrieval_history", self.retriever.history)
                    labels = [doc.get("label") for doc in new_docs if doc.get("label")]
                    retrieval_events.append(
                        {
                            "query": retrieval_query,
                            "labels": labels,
                            "count": len(new_docs),
                        }
                    )
                    # Remove retrieval directive before the next attempt
                    if labels:
                        retrieval_notice = f"Retrieved documents: {', '.join(labels)} for query '{retrieval_query}'."
                    else:
                        retrieval_notice = f"Retrieved {len(new_docs)} document chunk(s) for query '{retrieval_query}'."
                    messages = [
                        *messages,
                        Message(
                            type=MessageType.SYSTEM,
                            sender="orchestrator",
                            content=retrieval_notice,
                            metadata={"retrieval": {"query": retrieval_query, "labels": labels}},
                        ),
                    ]
                    continue

            final_content = self._strip_retrieval_markers(response_content)
            final_citations = candidate_citations
            break

        if self.memory_store:
            memory_metadata = {"role": self.role.value}
            if final_citations:
                memory_metadata["citations"] = final_citations
            if retrieval_events:
                memory_metadata["retrieval_events"] = retrieval_events
            self.memory_store.record_statement(
                self.name,
                final_content,
                topic=topic,
                metadata=memory_metadata,
            )

        response_metadata = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "role": self.role.value,
        }
        if final_citations:
            response_metadata["citations"] = final_citations
        if retrieval_events:
            response_metadata["retrieval_events"] = retrieval_events
        if self.config.speaking_style:
            response_metadata.setdefault("speaking_style", self.config.speaking_style)
        if self.config.speech_tags:
            response_metadata.setdefault("speech_tags", self.config.speech_tags)
        if self.config.tts_voice:
            response_metadata.setdefault("tts_voice", self.config.tts_voice)

        return AgentResponse(
            agent_name=self.name,
            content=final_content,
            metadata=response_metadata,
        )

    def _compose_system_prompt(
        self,
        *,
        topic: str | None,
        context: dict[str, Any],
        knowledge_documents: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Build the system prompt and return candidate citation metadata."""

        system_prompt = self.get_system_prompt()

        if self.memory_store:
            history_snippet = self.memory_store.get_history_prompt(
                self.name,
                topic=topic,
            )
            if history_snippet:
                system_prompt += f"\n\n{history_snippet}"

        additional_lines: list[str] = []
        for key, value in context.items():
            if key in {"knowledge_documents", "retrieval_history"}:
                continue
            additional_lines.append(f"- {key}: {value}")
        if additional_lines:
            system_prompt += "\n\nAdditional Context:\n" + "\n".join(additional_lines)

        retrieval_history = context.get("retrieval_history")
        if isinstance(retrieval_history, list) and retrieval_history:
            history_lines = []
            for event in retrieval_history[-5:]:
                if not isinstance(event, dict):
                    continue
                query = event.get("query") or "?"
                labels = ", ".join(str(label) for label in event.get("labels") or [])
                history_lines.append(f"- {query}: {labels or 'no results'}")
            if history_lines:
                system_prompt += "\n\nRecent Retrievals:\n" + "\n".join(history_lines)

        citations: list[dict[str, Any]] = []
        if knowledge_documents:
            formatted_docs: list[str] = []
            for idx, doc in enumerate(knowledge_documents, start=1):
                doc_id = str(doc.get("doc_id") or f"doc-{idx}")
                chunk_id = str(doc.get("chunk_id") or f"{doc_id}::chunk")
                score = doc.get("score")
                text = (doc.get("text") or "").strip()
                if len(text) > 420:
                    text = text[:417] + "..."
                metadata = doc.get("metadata") or {}
                document_meta = metadata.get("document") or {}
                chunk_meta = metadata.get("chunk") or {}
                title = str(document_meta.get("title") or doc_id)
                tags = document_meta.get("tags") or []
                tag_str = f" | tags: {', '.join(tags)}" if tags else ""
                position = chunk_meta.get("position")
                position_str = f"chunk #{position}" if position else chunk_id
                score_str = ""
                if isinstance(score, (int, float)):
                    score_str = f"(score {score:.3f}) "
                label = str(doc.get("label") or document_meta.get("label") or f"Doc {idx}")
                formatted_docs.append(
                    f"[{label}] {title}{tag_str} [{position_str}]\n{score_str}{text}"
                )
                citations.append(
                    {
                        "label": label,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "title": title,
                        "score": score,
                        "position": position,
                        "tags": tags,
                    }
                )

            if formatted_docs:
                system_prompt += (
                    "\n\nRetrieved Knowledge Snippets:\n"
                    + "\n\n".join(formatted_docs)
                    + "\n\nWhen using information from these snippets, reference the label (e.g., [Doc 1]) so readers can trace the source."
                )

        return system_prompt, citations

    def _extract_retrieval_query(self, text: str) -> str | None:
        match = self.RETRIEVE_PATTERN.search(text)
        if not match:
            return None
        query = match.group(1).strip()
        return query or None

    @staticmethod
    def _strip_retrieval_markers(text: str) -> str:
        return LLMAgent.RETRIEVE_PATTERN.sub("", text).strip()

    def _detect_topic(self, context: dict[str, Any] | None) -> str | None:
        """Extract a topic string from the provided context."""
        if not context:
            return None

        for key in ("memory_topic", "topic", "question", "subject", "title"):
            value = context.get(key)
            if isinstance(value, str) and value:
                return value
        return None
