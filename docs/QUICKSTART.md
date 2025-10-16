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
# Run a debate about climate policy
argentum debate "National carbon tax policy" \
  --ministers finance environment health \
  --rounds 3
```

#### CTO Advisory Panel

```bash
# Get advice on a technical decision
argentum advisory "Should we adopt Kubernetes?" \
  --advisors security engineering operations \
  --rounds 2
```

#### List Available Roles

```bash
argentum list-roles
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
