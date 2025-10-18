# Quick Start Guide

## Installation

### Prerequisites

- Python 3.11 or higher
- OpenAI API key (or Azure OpenAI credentials)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/argentum.git
cd argentum

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Set Up Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_key_here
```

## Usage Examples

### 1. Command Line Interface

#### Government Debate

```bash
# Create a project workspace (stores data under workspace/<slug>/)
argentum project init knit-cast --title "AI Knitting Circle"

# Run a debate about climate policy and persist it to the workspace
argentum debate "National carbon tax policy" \
  --project knit-cast \
  --ministers finance environment health \
  --rounds 3 \
  --summary-mode frontier  # requires OPENAI_API_KEY

# Reuse the same workspace with a custom session id
argentum debate "Universal basic income" \
  --project knit-cast \
  --session-id ubi-episode
```

#### CTO Advisory Panel

```bash
# Get advice on a technical decision
argentum advisory "Should we adopt Kubernetes?" \
  --project knit-cast \
  --advisors security engineering operations \
  --rounds 2 \
  --summary-mode local \
  --summary-command "ollama run llama3.1"  # example local summariser

# Append to an existing consultation session
argentum advisory "Should we adopt Kubernetes?" \
  --project knit-cast \
  --session-id architecture-eval
```

#### List Available Roles

```bash
argentum list-roles
```

Transcripts, highlights, and knowledge graph updates are saved to `workspace/<project>/sessions/<session-id>/`.

Customize agent personas at any time:

```bash
# Inspect and edit stored profiles
argentum project agent show knit-cast host
argentum project agent update knit-cast host --persona "Charismatic moderator" --temperature 0.4

# Adjust tone and suggested TTS voice
argentum project agent update knit-cast host \
  --speaking-style podcast \
  --speech-tag playful --speech-tag friendly \
  --tts-voice podcast_female
```

#### Ingest & Retrieve Project Documents (RAG)

```bash
# Add background reading to the workspace
argentum project docs ingest knit-cast docs/policy-brief.md --tag policy --tag 2026-q1

# Inspect the indexed sources
argentum project docs list knit-cast

# Ad-hoc semantic search over the chunks
argentum project docs search knit-cast --query "carbon dividend" --min-score 0.15

# Maintenance helpers
argentum project docs purge knit-cast policy-brief-20250101-120000
argentum project docs rebuild knit-cast
```

When you start a session the top matching chunks are prefetched automatically. During the debate/advisory, any agent can ask for more context by typing `<<retrieve: your search terms>>`. The orchestrator will pause, ingest the new snippets (labeled `[Doc n]`), and the agents will cite them in their replies and in the session highlights.

Try the self-contained demo:

```bash
python examples/two_actor_retrieval_demo.py
```

If you have a local OpenAI-compatible server (e.g. `http://localhost:5000/v1`) running a quantised 70B model, point Argentum at it with:

```bash
export OPENAI_API_KEY=sk-local-test
export OPENAI_API_BASE=http://localhost:5000/v1
argentum debate "Should we increase the carbon tax?" \
  --project knit-cast \
  --ministers finance --ministers environment \
  --rounds 1
```

Generate ready-to-narrate scripts from the captured highlights:

```bash
python - <<'PY'
import json, pathlib
from argentum.workspace.knowledge import build_tts_markdown, build_tts_script

highlights_path = pathlib.Path('workspace/knit-cast/sessions/<session-id>/highlights.json')
highlights = json.loads(highlights_path.read_text())

print("Markdown summary\n================")
print(build_tts_markdown(highlights))

print("JSON script\n============")
print(build_tts_script(highlights))
PY
```

### Project Workspaces

```bash
# Create a reusable workspace scaffold
argentum project init knit-cast --title "AI Knitting Circle"

# Inspect manifest and paths
argentum project info knit-cast

# Summarise knowledge captured so far
argentum project knowledge knit-cast --show summary

# Search warm cache highlights (FTS query)
argentum project knowledge knit-cast --search "carbon"

# Show retrieval history and citations for a session
cat workspace/knit-cast/sessions/<session-id>/highlights.json

# Inspect the project timeline
argentum project timeline knit-cast --limit 10

# List all known projects
argentum project list

# Compact warm-cache entries older than 60 days and purge a session
argentum project compact knit-cast --days 60
argentum project purge knit-cast --session ubi-episode --force
```

### 2. Python API

#### 2.1 Simple Group Chat

```python
import asyncio
from argentum import Agent, GroupChatOrchestrator, ChatManager
from argentum.agents.base import AgentConfig
from argentum.agents.llm_agent import LLMAgent
from argentum.llm import OpenAIProvider
from argentum.models import Role

async def main():
    # Create provider
    provider = OpenAIProvider(model="gpt-4")
    
    # Create agents
    agent1 = LLMAgent(
        config=AgentConfig(
            name="Alice",
            role=Role.PARTICIPANT,
            persona="You are optimistic and creative"
        ),
        provider=provider
    )
    
    agent2 = LLMAgent(
        config=AgentConfig(
            name="Bob",
            role=Role.PARTICIPANT,
            persona="You are analytical and cautious"
        ),
        provider=provider
    )
    
    # Create orchestrator
    chat_manager = ChatManager(max_turns=6)
    orchestrator = GroupChatOrchestrator(chat_manager)
    
    # Run conversation
    result = await orchestrator.execute(
        agents=[agent1, agent2],
        task="Discuss the pros and cons of remote work"
    )
    
    # Print results
    for msg in result.messages:
        if msg.sender != "orchestrator":
            print(f"\n[{msg.sender}]: {msg.content}")

asyncio.run(main())
```

#### 2.2 Sequential Pipeline

```python
import asyncio
from argentum.orchestration import SequentialOrchestrator
from argentum.agents.llm_agent import LLMAgent
from argentum.agents.base import AgentConfig
from argentum.llm import OpenAIProvider
from argentum.models import Role

async def main():
    provider = OpenAIProvider()
    
    # Create pipeline agents
    writer = LLMAgent(
        AgentConfig(name="Writer", role=Role.MAKER, 
                   persona="Draft content"),
        provider
    )
    editor = LLMAgent(
        AgentConfig(name="Editor", role=Role.CHECKER,
                   persona="Edit and improve"),
        provider
    )
    
    # Run pipeline
    pipeline = SequentialOrchestrator()
    result = await pipeline.execute(
        agents=[writer, editor],
        task="Write a product description for wireless headphones"
    )
    
    print(result.final_outputs[-1].content)

asyncio.run(main())
```

#### 2.3 Concurrent Analysis

```python
import asyncio
from argentum.orchestration import ConcurrentOrchestrator
from argentum.agents.llm_agent import LLMAgent
from argentum.agents.base import AgentConfig
from argentum.llm import OpenAIProvider
from argentum.models import Role

async def main():
    provider = OpenAIProvider()
    
    # Create parallel analysts
    financial = LLMAgent(
        AgentConfig(name="Financial Analyst", role=Role.ADVISOR,
                   persona="Analyze from financial perspective"),
        provider
    )
    technical = LLMAgent(
        AgentConfig(name="Technical Analyst", role=Role.ADVISOR,
                   persona="Analyze from technical perspective"),
        provider
    )
    
    # Run concurrent analysis
    concurrent = ConcurrentOrchestrator()
    result = await concurrent.execute(
        agents=[financial, technical],
        task="Evaluate investing in quantum computing startups"
    )
    
    print(result.consensus)

asyncio.run(main())
```

### 3. Pre-built Scenarios

#### 3.1 Government Debate

```python
import asyncio
from argentum.scenarios import GovernmentDebate

async def main():
    debate = GovernmentDebate(
        topic="Universal Basic Income implementation",
        ministers=["finance", "health", "education"],
        rounds=3
    )
    
    result = await debate.run()
    print(result.consensus)

asyncio.run(main())
```

#### 3.2 CTO Advisory Panel

```python
import asyncio
from argentum.scenarios import CTOAdvisoryPanel

async def main():
    panel = CTOAdvisoryPanel(
        question="Move to serverless architecture?",
        advisors=["security", "finance", "engineering"],
        rounds=2
    )
    
    result = await panel.consult()
    print(result.consensus)

asyncio.run(main())
```

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
ARGENTUM_DEFAULT_MODEL=gpt-4
ARGENTUM_MAX_TOKENS=4096
ARGENTUM_TEMPERATURE=0.7
ARGENTUM_MAX_TURNS=10
```

### Agent Configuration

```python
from argentum.agents.base import AgentConfig
from argentum.models import Role

config = AgentConfig(
    name="Expert Agent",
    role=Role.ADVISOR,
    persona="You are a domain expert in...",
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    tools=["web_search", "calculator"]
)
```

### Chat Manager Settings

```python
from argentum.coordination import ChatManager, SpeakerSelectionMode

chat_manager = ChatManager(
    max_turns=10,
    selection_mode=SpeakerSelectionMode.ROUND_ROBIN,
    allow_repeats=True,
    min_turns=2
)
```

## Next Steps

- Read the full documentation (coming soon)
- Check out more examples in the `examples/` directory
- Explore the API reference
- Join our community

## Troubleshooting

### API Key Issues

```bash
# Make sure your API key is set
echo $OPENAI_API_KEY

# Or set it temporarily
export OPENAI_API_KEY='your-key-here'
```

### Import Errors

```bash
# Reinstall in development mode
pip install -e .

# Check installation
python -c "import argentum; print(argentum.__version__)"
```

### Rate Limits

If you hit OpenAI rate limits:

- Reduce the number of rounds
- Use a smaller model (gpt-3.5-turbo)
- Add delays between requests
- Consider using Azure OpenAI for higher limits

## Getting Help

- GitHub Issues: <https://github.com/yourusername/argentum/issues>
- Documentation: <https://argentum.readthedocs.io>
- Examples: See `examples/` directory

### Example: Two-Actor Retrieval Demo

`examples/two_actor_retrieval_demo.py` spins up a lightweight workspace, ingests a background briefing, and runs two scripted ministers through a debate. The first agent requests additional evidence with `<<retrieve: ...>>`; both agents then cite `[Doc 1]` in their replies and the resulting `highlights.json` logs the retrieval history and sources.
