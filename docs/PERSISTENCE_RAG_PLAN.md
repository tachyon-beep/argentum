# Implementation Plan: Persistence + Identity + RAG

## Overview

This plan combines three interconnected features to create a comprehensive knowledge and memory system for Argentum:

1. **Conversation Persistence**: Store and reload transcripts
2. **Agent Identity & Continuity**: Agents remember their own statements
3. **Knowledge Base (RAG)**: Ground debates in factual documents

## Phase 1: Conversation Persistence (Week 1-2)

### Goal

Store conversation transcripts and enable multi-session continuity.

### Components

#### 1.1 Storage Backend (`argentum/persistence/storage.py`)

```python
from abc import ABC, abstractmethod
from pathlib import Path
import json
from datetime import datetime

class ConversationStore(ABC):
    """Abstract base for conversation storage."""
    
    @abstractmethod
    async def save_conversation(self, conversation_id: str, data: dict) -> None:
        """Save a conversation."""
    
    @abstractmethod
    async def load_conversation(self, conversation_id: str) -> dict:
        """Load a conversation."""
    
    @abstractmethod
    async def list_conversations(self, filters: dict = None) -> list[str]:
        """List stored conversations."""
    
    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation."""

class JSONFileStore(ConversationStore):
    """File-based conversation storage using JSON."""
    
    def __init__(self, base_path: Path = Path("./conversations")):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def save_conversation(self, conversation_id: str, data: dict) -> None:
        """Save conversation to JSON file."""
        file_path = self.base_path / f"{conversation_id}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    # ... other methods

class SQLiteStore(ConversationStore):
    """SQLite-based conversation storage for better querying."""
    # ... implementation
```

#### 1.2 Conversation Serialization (`argentum/persistence/serializer.py`)

```python
from argentum.models import OrchestrationResult, Context
from argentum.memory.context import ConversationHistory

class ConversationSerializer:
    """Serialize/deserialize conversations for storage."""
    
    @staticmethod
    def serialize_result(result: OrchestrationResult) -> dict:
        """Convert OrchestrationResult to storable dict."""
        return {
            "id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "pattern": result.pattern.value,
            "messages": [
                {
                    "type": msg.type.value,
                    "sender": msg.sender,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "metadata": msg.metadata,
                }
                for msg in result.messages
            ],
            "consensus": result.consensus,
            "termination_reason": result.termination_reason.value,
            "duration_seconds": result.duration_seconds,
            "metadata": result.metadata,
        }
    
    @staticmethod
    def deserialize_result(data: dict) -> OrchestrationResult:
        """Reconstruct OrchestrationResult from stored data."""
        # ... implementation
    
    @staticmethod
    def serialize_context(context: Context) -> dict:
        """Serialize context for storage."""
        # ... implementation
```

#### 1.3 Session Management (`argentum/persistence/session.py`)

```python
from uuid import uuid4

class ConversationSession:
    """Manages a conversation session with persistence."""
    
    def __init__(
        self,
        store: ConversationStore,
        session_id: str | None = None,
        metadata: dict | None = None,
    ):
        self.session_id = session_id or str(uuid4())
        self.store = store
        self.metadata = metadata or {}
        self.history: list[OrchestrationResult] = []
    
    async def save(self, result: OrchestrationResult) -> None:
        """Save a conversation result to this session."""
        self.history.append(result)
        
        data = {
            "session_id": self.session_id,
            "metadata": self.metadata,
            "results": [
                ConversationSerializer.serialize_result(r)
                for r in self.history
            ],
        }
        
        await self.store.save_conversation(self.session_id, data)
    
    async def load(self) -> None:
        """Load conversation history from storage."""
        data = await self.store.load_conversation(self.session_id)
        self.metadata = data["metadata"]
        self.history = [
            ConversationSerializer.deserialize_result(r)
            for r in data["results"]
        ]
    
    def get_full_transcript(self) -> str:
        """Get complete transcript of all conversations in session."""
        transcript_parts = []
        
        for i, result in enumerate(self.history, 1):
            transcript_parts.append(f"=== Session {i} ===")
            for msg in result.messages:
                if msg.sender != "orchestrator":
                    transcript_parts.append(f"[{msg.sender}]: {msg.content}")
        
        return "\n\n".join(transcript_parts)
```

### Usage Example

```python
from argentum.persistence import JSONFileStore, ConversationSession

# Create session
store = JSONFileStore(Path("./debate_transcripts"))
session = ConversationSession(
    store=store,
    metadata={
        "scenario": "government_debate",
        "topic": "Climate Policy",
        "date": datetime.now().isoformat(),
    }
)

# Run first debate
debate = GovernmentDebate(topic="Carbon tax proposal", session=session)
result = await debate.run()
await session.save(result)

# Later: Continue the conversation
session2 = ConversationSession(store=store, session_id=session.session_id)
await session2.load()

# Now ministers can reference previous debate
debate2 = GovernmentDebate(
    topic="Implementation timeline for carbon tax",
    session=session2,  # Pass existing session
    load_history=True,  # Include previous transcript in context
)
```

---

## Phase 2: Agent Identity & Memory (Week 2-3)

### Goal

Agents remember their own previous statements and maintain consistent personas across sessions.

### Components

#### 2.1 Agent Memory (`argentum/memory/agent_memory.py`)

```python
from argentum.models import Message
from collections import defaultdict

class AgentMemory:
    """Stores an agent's personal memory across conversations."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.my_statements: list[Message] = []
        self.key_points: dict[str, list[str]] = defaultdict(list)
        self.stances: dict[str, str] = {}  # topic -> stance
    
    def record_statement(self, message: Message, topic: str | None = None) -> None:
        """Record something this agent said."""
        self.my_statements.append(message)
        
        if topic:
            # Extract key points (could use LLM to summarize)
            self.key_points[topic].append(message.content)
    
    def get_my_history(self, topic: str | None = None) -> str:
        """Get formatted history of what I've said."""
        if topic and topic in self.key_points:
            statements = self.key_points[topic]
        else:
            statements = [msg.content for msg in self.my_statements]
        
        if not statements:
            return "I have not spoken about this before."
        
        return f"Previously, I have stated:\n" + "\n".join(
            f"- {stmt[:200]}..." if len(stmt) > 200 else f"- {stmt}"
            for stmt in statements[-5:]  # Last 5 statements
        )
    
    def set_stance(self, topic: str, stance: str) -> None:
        """Record a stance on a topic."""
        self.stances[topic] = stance
    
    def get_stance(self, topic: str) -> str | None:
        """Retrieve stance on a topic."""
        return self.stances.get(topic)
    
    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "agent_name": self.agent_name,
            "statements": [
                {
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                }
                for msg in self.my_statements
            ],
            "key_points": dict(self.key_points),
            "stances": self.stances,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgentMemory":
        """Deserialize from storage."""
        memory = cls(data["agent_name"])
        memory.key_points = defaultdict(list, data["key_points"])
        memory.stances = data["stances"]
        return memory

class AgentMemoryStore:
    """Persistent storage for agent memories."""
    
    def __init__(self, base_path: Path = Path("./agent_memories")):
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, AgentMemory] = {}
    
    async def get_memory(self, agent_name: str) -> AgentMemory:
        """Get or create agent memory."""
        if agent_name in self._cache:
            return self._cache[agent_name]
        
        file_path = self.base_path / f"{agent_name}.json"
        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)
            memory = AgentMemory.from_dict(data)
        else:
            memory = AgentMemory(agent_name)
        
        self._cache[agent_name] = memory
        return memory
    
    async def save_memory(self, memory: AgentMemory) -> None:
        """Persist agent memory."""
        file_path = self.base_path / f"{memory.agent_name}.json"
        with open(file_path, 'w') as f:
            json.dump(memory.to_dict(), f, indent=2)
```

#### 2.2 Enhanced LLM Agent (`argentum/agents/llm_agent.py` - additions)

```python
class LLMAgent(Agent):
    """LLM-powered agent with memory."""
    
    def __init__(
        self,
        config: AgentConfig,
        provider: LLMProvider,
        memory: AgentMemory | None = None,  # NEW
    ):
        super().__init__(config)
        self.provider = provider
        self.memory = memory  # NEW
    
    async def generate_response(
        self,
        messages: list[Message],
        context: dict[str, Any] | None = None,
        topic: str | None = None,  # NEW
    ) -> AgentResponse:
        """Generate response with memory integration."""
        
        # Build system prompt with memory
        system_prompt = self.config.persona
        
        # Add agent's own history if available
        if self.memory:
            my_history = self.memory.get_my_history(topic)
            system_prompt += f"\n\n## Your Previous Statements\n{my_history}"
            
            # Add stance if exists
            if topic and (stance := self.memory.get_stance(topic)):
                system_prompt += f"\n\nYour established stance: {stance}"
        
        # ... rest of generation logic
        
        response = await self.provider.generate(...)
        
        # Record this statement in memory
        if self.memory and topic:
            response_msg = Message(
                type=MessageType.ASSISTANT,
                sender=self.name,
                content=response,
            )
            self.memory.record_statement(response_msg, topic)
        
        return AgentResponse(agent_name=self.name, content=response)
```

#### 2.3 Session with Memory Integration

```python
class MemoryAwareSession(ConversationSession):
    """Session that maintains agent memories."""
    
    def __init__(
        self,
        store: ConversationStore,
        memory_store: AgentMemoryStore,
        session_id: str | None = None,
        metadata: dict | None = None,
    ):
        super().__init__(store, session_id, metadata)
        self.memory_store = memory_store
    
    async def create_agent(self, config: AgentConfig, provider: LLMProvider) -> LLMAgent:
        """Create agent with memory."""
        memory = await self.memory_store.get_memory(config.name)
        return LLMAgent(config=config, provider=provider, memory=memory)
    
    async def save(self, result: OrchestrationResult) -> None:
        """Save conversation and update agent memories."""
        await super().save(result)
        
        # Update and save agent memories
        for msg in result.messages:
            if msg.sender != "orchestrator":
                memory = await self.memory_store.get_memory(msg.sender)
                memory.record_statement(msg, self.metadata.get("topic"))
                await self.memory_store.save_memory(memory)
```

### Usage Example

```python
# First debate
memory_store = AgentMemoryStore()
session = MemoryAwareSession(
    store=JSONFileStore(),
    memory_store=memory_store,
    metadata={"topic": "carbon_tax"}
)

# Create agents with memory
finance_agent = await session.create_agent(
    AgentConfig(name="Minister of Finance", ...),
    provider
)

# Run debate
result = await orchestrator.execute(
    agents=[finance_agent, ...],
    task="Should we implement a carbon tax?"
)

await session.save(result)

# Next week: Same minister, new debate
session2 = MemoryAwareSession(
    store=JSONFileStore(),
    memory_store=memory_store,  # Same memory store!
    metadata={"topic": "carbon_tax_implementation"}
)

finance_agent2 = await session2.create_agent(
    AgentConfig(name="Minister of Finance", ...),  # Same name = same memory
    provider
)

# The agent will now say:
# "Previously, I have stated:
#  - We must balance environmental goals with economic stability
#  - A gradual phase-in is essential to avoid business disruption
# Based on my previous position, I propose..."
```

---

## Phase 3: Knowledge Base (RAG) (Week 4-6)

### Goal

Ground agent statements in factual documents with citations.

### Components

#### 3.1 Document Store (`argentum/knowledge/document_store.py`)

```python
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

class DocumentStore:
    """Vector-based document storage for RAG."""
    
    def __init__(self, persist_directory: Path = Path("./knowledge_base")):
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=self.embedding_fn
        )
    
    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict] | None = None,
        chunk_size: int = 500,
    ) -> None:
        """Add documents with automatic chunking."""
        chunks = []
        chunk_metas = []
        
        for i, doc in enumerate(documents):
            doc_chunks = self._chunk_document(doc, chunk_size)
            meta = metadatas[i] if metadatas else {}
            
            for j, chunk in enumerate(doc_chunks):
                chunks.append(chunk)
                chunk_metas.append({
                    **meta,
                    "doc_index": i,
                    "chunk_index": j,
                    "source": meta.get("source", f"doc_{i}"),
                })
        
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            metadatas=chunk_metas,
            ids=ids
        )
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        """Search for relevant document chunks."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict,
        )
        
        return [
            {
                "content": doc,
                "metadata": meta,
                "distance": dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
    
    @staticmethod
    def _chunk_document(text: str, chunk_size: int) -> list[str]:
        """Split document into overlapping chunks."""
        words = text.split()
        chunks = []
        overlap = 50
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
```

#### 3.2 RAG-Enhanced Agent (`argentum/agents/rag_agent.py`)

```python
class RAGAgent(LLMAgent):
    """Agent with document grounding capabilities."""
    
    def __init__(
        self,
        config: AgentConfig,
        provider: LLMProvider,
        knowledge_base: DocumentStore,
        memory: AgentMemory | None = None,
        require_citations: bool = False,
    ):
        super().__init__(config, provider, memory)
        self.knowledge_base = knowledge_base
        self.require_citations = require_citations
    
    async def generate_response(
        self,
        messages: list[Message],
        context: dict[str, Any] | None = None,
        topic: str | None = None,
    ) -> AgentResponse:
        """Generate response with document grounding."""
        
        # Get last message (the question/prompt)
        last_message = messages[-1].content if messages else ""
        
        # Retrieve relevant documents
        relevant_docs = self.knowledge_base.search(
            query=last_message,
            n_results=3,
        )
        
        # Build enhanced prompt
        system_prompt = self.config.persona
        
        if relevant_docs:
            doc_context = "\n\n".join([
                f"[Source: {doc['metadata']['source']}]\n{doc['content']}"
                for doc in relevant_docs
            ])
            system_prompt += f"\n\n## Relevant Documents\n{doc_context}"
        
        if self.require_citations:
            system_prompt += "\n\nIMPORTANT: Support your statements with citations. Reference sources like [Source: filename]."
        
        # Add memory
        if self.memory:
            my_history = self.memory.get_my_history(topic)
            system_prompt += f"\n\n## Your Previous Statements\n{my_history}"
        
        # Generate response
        response_content = await self.provider.generate(
            messages=[{"role": "system", "content": system_prompt}, ...],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        # Extract citations
        citations = self._extract_citations(response_content, relevant_docs)
        
        # Record in memory
        if self.memory and topic:
            response_msg = Message(
                type=MessageType.ASSISTANT,
                sender=self.name,
                content=response_content,
                metadata={"citations": citations},
            )
            self.memory.record_statement(response_msg, topic)
        
        return AgentResponse(
            agent_name=self.name,
            content=response_content,
            metadata={"citations": citations, "retrieved_docs": relevant_docs},
        )
    
    @staticmethod
    def _extract_citations(text: str, docs: list[dict]) -> list[str]:
        """Extract citations from response."""
        import re
        citations = []
        for doc in docs:
            source = doc['metadata']['source']
            if source in text or f"[Source: {source}]" in text:
                citations.append(source)
        return citations
```

### Complete Integration Example

```python
# Setup
from argentum.persistence import JSONFileStore, AgentMemoryStore, MemoryAwareSession
from argentum.knowledge import DocumentStore
from argentum.agents import RAGAgent
from argentum.scenarios.debate import GovernmentDebate

# Initialize stores
conv_store = JSONFileStore(Path("./debates"))
memory_store = AgentMemoryStore(Path("./minister_memories"))
knowledge_base = DocumentStore(Path("./government_docs"))

# Load documents
knowledge_base.add_documents([
    Path("2025_budget.pdf").read_text(),
    Path("climate_policy.pdf").read_text(),
    Path("defense_strategy.pdf").read_text(),
], metadatas=[
    {"source": "2025_budget.pdf", "type": "budget"},
    {"source": "climate_policy.pdf", "type": "policy"},
    {"source": "defense_strategy.pdf", "type": "strategy"},
])

# Create session
session = MemoryAwareSession(
    store=conv_store,
    memory_store=memory_store,
    metadata={"topic": "carbon_tax", "date": "2025-10-16"}
)

# Create RAG-enabled agents
finance_minister = RAGAgent(
    config=AgentConfig(
        name="Minister of Finance",
        role=Role.PARTICIPANT,
        persona="You are the Minister of Finance. Focus on fiscal responsibility and economic impact.",
    ),
    provider=OpenAIProvider(base_url="http://localhost:5000/v1"),
    knowledge_base=knowledge_base,
    memory=await memory_store.get_memory("Minister of Finance"),
    require_citations=True,
)

environment_minister = RAGAgent(
    config=AgentConfig(
        name="Minister of Environment",
        role=Role.PARTICIPANT,
        persona="You are the Minister of Environment. Focus on climate goals and sustainability.",
    ),
    provider=OpenAIProvider(base_url="http://localhost:5000/v1"),
    knowledge_base=knowledge_base,
    memory=await memory_store.get_memory("Minister of Environment"),
    require_citations=True,
)

# Run debate with full continuity
chat_manager = ChatManager(max_turns=8, min_turns=4)
orchestrator = GroupChatOrchestrator(chat_manager=chat_manager)

result = await orchestrator.execute(
    agents=[finance_minister, environment_minister],
    task="Should we increase the carbon tax from $50 to $80 per ton?"
)

# Save everything
await session.save(result)

# Output with citations
for msg in result.messages:
    if msg.sender != "orchestrator":
        print(f"\n[{msg.sender}]")
        print(msg.content)
        if cites := msg.metadata.get("citations"):
            print(f"📚 Sources: {', '.join(cites)}")

# Next session - ministers remember their positions
session2 = MemoryAwareSession(
    store=conv_store,
    memory_store=memory_store,  # Same memory!
    metadata={"topic": "carbon_tax_implementation", "date": "2025-10-23"}
)

# Ministers will reference: "In our previous discussion, I argued that..."
```

---

## Benefits of This Integrated Approach

### 1. **Continuity Across Sessions** 🔄

- Ministers remember their previous positions
- No contradictions between debates
- Consistent personas over time

### 2. **Fact-Based Arguments** 📊

- All claims backed by documents
- Citation tracking
- Verifiable statements

### 3. **Audit Trail** 📝

- Complete transcripts stored
- Who said what, when
- Traceable decision-making

### 4. **Realistic Debates** 🎭

- "As I stated in our budget meeting last week..."
- "According to the 2025 Climate Report [cite]..."
- "Building on Finance Minister's concern about costs..."

### 5. **Production Ready** 🏢

- Suitable for real advisory boards
- Compliance-friendly (audit trails)
- Grounded in company/government docs

---

## Dependencies

```toml
[project.optional-dependencies]
knowledge = [
    "chromadb>=0.4.0",        # Vector database
    "sentence-transformers>=2.2.0",  # Embeddings
    "pypdf>=3.0.0",           # PDF parsing
    "aiosqlite>=0.19.0",      # Async SQLite
]
```

---

## Timeline Summary

- **Week 1**: Conversation persistence (save/load transcripts)
- **Week 2**: Agent identity & memory (remember own statements)
- **Week 3**: Memory integration (continuity across sessions)
- **Week 4**: Document store & RAG basics (retrieval)
- **Week 5**: RAG agent integration (grounded responses)
- **Week 6**: Citation tracking & polish (production-ready)

**Total: ~6 weeks for complete system**

---

## Testing Strategy

```python
# Test 1: Persistence
async def test_conversation_persistence():
    session = ConversationSession(store=JSONFileStore())
    result = await run_debate()
    await session.save(result)
    
    session2 = ConversationSession(store=JSONFileStore(), session_id=session.session_id)
    await session2.load()
    assert len(session2.history) == 1

# Test 2: Agent Memory
async def test_agent_remembers_statements():
    memory = AgentMemory("Minister of Finance")
    memory.record_statement(Message(...), topic="budget")
    
    history = memory.get_my_history("budget")
    assert "Previously, I have stated" in history

# Test 3: RAG Citations
async def test_rag_citations():
    knowledge_base = DocumentStore()
    knowledge_base.add_documents([budget_doc], metadatas=[{"source": "budget.pdf"}])
    
    agent = RAGAgent(..., knowledge_base=knowledge_base, require_citations=True)
    response = await agent.generate_response(...)
    
    assert len(response.metadata["citations"]) > 0
    assert "budget.pdf" in response.metadata["citations"]
```

---

## Next Steps

Would you like me to:

1. **Start with Phase 1** (Persistence) - Get transcript storage working first
2. **Prototype the complete flow** - Quick POC of all three features
3. **Focus on memory first** - Agent continuity before documents
4. **Something else?**

Let me know and I'll begin implementation! 🚀
