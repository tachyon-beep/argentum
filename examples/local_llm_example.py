"""Example: Using Argentum with a local OpenAI-compatible LLM server.

This example shows how to connect to a local LLM server (like llama.cpp, vLLM,
LocalAI, etc.) that implements the OpenAI API format.

Your local server on http://localhost:5000 should work perfectly!
"""

import asyncio

from argentum.agents.base import AgentConfig
from argentum.agents.llm_agent import LLMAgent
from argentum.coordination.chat_manager import ChatManager
from argentum.llm.provider import OpenAIProvider
from argentum.models import Role
from argentum.orchestration.group_chat import GroupChatOrchestrator


async def main() -> None:
    """Run a debate using a local LLM server."""

    # Configure the OpenAI provider to use your local server
    # The OpenAI client supports base_url parameter for custom endpoints
    local_provider = OpenAIProvider(
        model="local-model",  # Use whatever model name your server expects
        api_key="not-needed",  # Most local servers don't need a real API key
        base_url="http://localhost:5000/v1",  # Point to your local OpenAI-compatible server
    )

    print("🤖 Setting up debate with LOCAL LLM server at http://localhost:5000")
    print("=" * 80)

    # Create agents for a simple debate
    agents = [
        LLMAgent(
            config=AgentConfig(
                name="Optimist",
                role=Role.PARTICIPANT,
                persona="You are an optimistic person who sees the bright side of technology.",
                model="local-model",
                temperature=0.8,
            ),
            provider=local_provider,
        ),
        LLMAgent(
            config=AgentConfig(
                name="Skeptic",
                role=Role.PARTICIPANT,
                persona="You are a cautious skeptic who questions new technology trends.",
                model="local-model",
                temperature=0.8,
            ),
            provider=local_provider,
        ),
        LLMAgent(
            config=AgentConfig(
                name="Pragmatist",
                role=Role.PARTICIPANT,
                persona="You are a practical person who balances innovation with real-world constraints.",
                model="local-model",
                temperature=0.8,
            ),
            provider=local_provider,
        ),
    ]

    # Set up the debate
    chat_manager = ChatManager(
        max_turns=6,  # 2 turns per agent
        min_turns=3,  # At least one turn per agent
    )

    orchestrator = GroupChatOrchestrator(chat_manager=chat_manager)

    # Run the debate
    topic = "Should companies adopt AI assistants for all customer service?"

    print(f"\n📋 Topic: {topic}\n")
    print("Starting debate... (this might be slow with a local model)")
    print("=" * 80 + "\n")

    result = await orchestrator.execute(
        agents=agents,
        task=topic,
    )

    # Display results
    print("\n" + "=" * 80)
    print("💬 DEBATE TRANSCRIPT")
    print("=" * 80 + "\n")

    for msg in result.messages:
        if msg.sender != "orchestrator":
            print(f"[{msg.sender}]")
            print(msg.content)
            print()

    print("=" * 80)
    print("🎯 CONSENSUS")
    print("=" * 80)
    print(result.consensus)
    print()

    print(f"⏱️  Duration: {result.duration_seconds:.2f}s")
    print(f"✅ Termination: {result.termination_reason.value}")


if __name__ == "__main__":
    asyncio.run(main())
