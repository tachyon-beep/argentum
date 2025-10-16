# Argentum 🎭

<div align="center">

**A versatile multi-agent AI dialogue system for debates and advisory panels**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## 🌟 Overview

Argentum is a sophisticated multi-agent AI system that enables multiple specialized AI agents to collaborate, debate, and provide advisory insights. Built on cutting-edge research in multi-agent LLM systems, Argentum supports various orchestration patterns to tackle complex problems requiring diverse perspectives and collaborative intelligence.

### Key Features

- 🎯 **Multiple Orchestration Patterns**
  - Sequential pipelines for staged workflows
  - Concurrent processing for parallel perspectives
  - Group chat for interactive debates and collaboration

- 🤖 **Flexible Agent System**
  - Role-based agent specialization
  - LLM-powered and rule-based agents
  - Dynamic agent recruitment

- 💬 **Advanced Chat Management**
  - Turn-taking and moderation
  - Context-aware conversation flow
  - Termination criteria and consensus building

- 🧠 **Intelligent Memory**
  - Shared context management
  - Conversation history persistence
  - Summarization for long discussions

- 🔌 **Provider Agnostic**
  - OpenAI, Azure OpenAI support
  - Local model compatibility (Ollama, etc.)
  - Easy provider switching

- 👤 **Human-in-the-Loop**
  - Interactive oversight capabilities
  - Manual intervention options
  - Collaborative AI-human workflows

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/argentum.git
cd argentum

# Install dependencies
pip install -e .

# Or with all optional dependencies
pip install -e ".[all]"
```

### Basic Usage

```python
from argentum import Agent, GroupChatOrchestrator, ChatManager
from argentum.llm import OpenAIProvider

# Create specialized agents
finance_minister = Agent(
    name="Finance Minister",
    role="financial_advisor",
    persona="Focus on budget and economic impacts",
    provider=OpenAIProvider(model="gpt-4")
)

environment_minister = Agent(
    name="Environment Minister",
    role="environmental_advisor",
    persona="Focus on sustainability and environmental concerns",
    provider=OpenAIProvider(model="gpt-4")
)

# Set up debate
chat_manager = ChatManager(max_turns=10)
orchestrator = GroupChatOrchestrator(chat_manager=chat_manager)

# Run the debate
result = await orchestrator.execute(
    agents=[finance_minister, environment_minister],
    task="Should we implement a carbon tax?",
    context={"background": "Climate change urgency vs economic concerns"}
)

print(result.consensus)
```

## 📖 Use Cases

### 1. AI Debate Simulation

Simulate policy debates between AI government ministers with different perspectives:

```python
from argentum.scenarios import GovernmentDebate

debate = GovernmentDebate(
    ministers=["finance", "environment", "defense", "health"],
    topic="National Infrastructure Investment Plan",
    rounds=5
)

result = await debate.run()
```

### 2. Virtual CTO Advisory Panel

Consult a panel of AI experts for technical decision-making:

```python
from argentum.scenarios import CTOAdvisoryPanel

panel = CTOAdvisoryPanel(
    advisors=["security", "finance", "engineering", "product"],
    question="Should we adopt microservices architecture?",
)

recommendations = await panel.consult()
```

### 3. Sequential Pipeline

Chain multiple specialized agents for progressive refinement:

```python
from argentum import SequentialOrchestrator

pipeline = SequentialOrchestrator([
    draft_writer_agent,
    editor_agent,
    fact_checker_agent,
    compliance_agent
])

result = await pipeline.execute(task="Create a press release")
```

### 4. Concurrent Analysis

Get diverse perspectives in parallel:

```python
from argentum import ConcurrentOrchestrator

parallel = ConcurrentOrchestrator([
    logical_reasoner_agent,
    creative_thinker_agent,
    data_analyst_agent
])

result = await parallel.execute(task="Analyze market opportunity")
```

## 🏗️ Architecture

```
argentum/
├── agents/              # Agent abstractions and implementations
│   ├── base.py         # Base Agent class
│   ├── llm_agent.py    # LLM-powered agent
│   └── tool_agent.py   # Rule-based/tool agent
├── orchestration/       # Orchestration patterns
│   ├── sequential.py   # Pipeline pattern
│   ├── concurrent.py   # Parallel pattern
│   └── group_chat.py   # Debate pattern
├── coordination/        # Coordination logic
│   ├── chat_manager.py # Conversation moderator
│   └── aggregator.py   # Result aggregation
├── memory/             # Context and persistence
│   ├── context.py      # Shared context
│   └── persistence.py  # Storage backend
├── llm/                # LLM provider abstractions
│   ├── provider.py     # Provider interface
│   └── models.py       # Model configurations
└── scenarios/          # Pre-built scenarios
    ├── debate.py       # Government debate
    └── advisory.py     # CTO advisory panel
```

## 🎯 Core Concepts

### Orchestration Patterns

**Sequential**: Agents work in a fixed linear sequence, each building on the previous agent's output.

```python
# Draft → Edit → Review → Publish
pipeline = SequentialOrchestrator([writer, editor, reviewer, publisher])
```

**Concurrent**: Multiple agents work in parallel, providing independent analyses.

```python
# All experts analyze simultaneously
parallel = ConcurrentOrchestrator([expert1, expert2, expert3])
```

**Group Chat**: Agents interact in a shared conversation, debating and collaborating.

```python
# Interactive debate with moderation
debate = GroupChatOrchestrator(agents, chat_manager)
```

### Agent Roles

Agents can be assigned specialized roles:

- **Maker**: Creates initial outputs
- **Checker**: Reviews and critiques
- **Advisor**: Provides domain expertise
- **Judge**: Evaluates arguments
- **Moderator**: Manages conversation flow

### Chat Manager

The chat manager controls conversation flow:

- **Turn-taking**: Round-robin, priority-based, or dynamic
- **Moderation**: Keeps discussion on-topic
- **Termination**: Decides when to end the conversation
- **Intervention**: Allows human oversight

## ⚙️ Configuration

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_key

# Argentum settings
ARGENTUM_DEFAULT_MODEL=gpt-4
ARGENTUM_MAX_TOKENS=4096
ARGENTUM_TEMPERATURE=0.7
```

### Agent Configuration

```python
agent = Agent(
    name="Security Advisor",
    role="security_expert",
    persona="You are a cybersecurity expert focusing on threat analysis",
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    tools=["web_search", "database_query"]
)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=argentum --cov-report=html

# Run specific test file
pytest tests/test_orchestration.py
```

## 📚 Documentation

Full documentation is available at [argentum.readthedocs.io](https://argentum.readthedocs.io) (coming soon).

### Examples

Check the `examples/` directory for complete examples:

- `examples/minister_debate.py` - Government policy debate
- `examples/cto_panel.py` - Technical advisory panel
- `examples/pipeline_demo.py` - Sequential workflow
- `examples/concurrent_analysis.py` - Parallel processing

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Argentum is inspired by cutting-edge research in multi-agent systems:

- [MIT CSAIL Multi-Model Collaboration](https://news.mit.edu/2023/multi-ai-collaboration-helps-reasoning-factual-accuracy-language-models-0918)
- [Microsoft AutoGen](https://microsoft.github.io/autogen/)
- [MetaGPT Framework](https://github.com/geekan/MetaGPT)
- [DebateSim Research](https://openreview.net/forum?id=5yxNX2pmhJ)

## 🗺️ Roadmap

- [x] Core orchestration patterns
- [x] LLM provider abstraction
- [x] Basic memory system
- [ ] Advanced context summarization
- [ ] Web UI for debate visualization
- [ ] Agent marketplace/registry
- [ ] Quality metrics dashboard
- [ ] Multi-language support
- [ ] Cloud deployment templates

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub or reach out to the team.

---

<div align="center">

**Built with ❤️ by the Argentum Team**

</div>
