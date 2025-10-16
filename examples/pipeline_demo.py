"""Example: Simple Sequential Pipeline."""

import asyncio
import os

from argentum.agents.base import AgentConfig
from argentum.agents.llm_agent import LLMAgent
from argentum.llm.provider import OpenAIProvider
from argentum.models import Role
from argentum.orchestration import SequentialOrchestrator


async def main() -> None:
    """Run a sequential pipeline example."""
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        return

    provider = OpenAIProvider(model="gpt-4")

    # Create agents for a content creation pipeline
    drafter = LLMAgent(
        config=AgentConfig(
            name="Content Drafter",
            role=Role.MAKER,
            persona="You draft initial content. Be creative and thorough.",
            max_tokens=300,
        ),
        provider=provider,
    )

    editor = LLMAgent(
        config=AgentConfig(
            name="Content Editor",
            role=Role.CHECKER,
            persona="You edit for clarity, grammar, and style. Improve the draft.",
            max_tokens=300,
        ),
        provider=provider,
    )

    fact_checker = LLMAgent(
        config=AgentConfig(
            name="Fact Checker",
            role=Role.CHECKER,
            persona="You verify factual accuracy and flag any unsupported claims.",
            max_tokens=200,
        ),
        provider=provider,
    )

    # Create pipeline
    pipeline = SequentialOrchestrator()

    print("=" * 80)
    print("SEQUENTIAL PIPELINE EXAMPLE")
    print("=" * 80)
    print("\nPipeline: Drafter → Editor → Fact Checker")
    print("\nRunning pipeline...\n")

    # Execute pipeline
    result = await pipeline.execute(
        agents=[drafter, editor, fact_checker],
        task="Write a brief paragraph about the benefits of renewable energy.",
    )

    # Display results
    print("\n" + "=" * 80)
    print("PIPELINE RESULTS")
    print("=" * 80 + "\n")

    for i, output in enumerate(result.final_outputs, 1):
        print(f"Stage {i}: {output.agent_name}")
        print("-" * 80)
        print(output.content)
        print()

    print("=" * 80)
    print(f"Duration: {result.duration_seconds:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
