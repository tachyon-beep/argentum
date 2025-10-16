"""Example: Government Minister Debate on Carbon Tax."""

import asyncio
import os

from argentum.scenarios import GovernmentDebate


async def main() -> None:
    """Run a government debate example."""
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return

    # Create a debate
    debate = GovernmentDebate(
        topic="Implementation of a National Carbon Tax to combat climate change",
        ministers=["finance", "environment", "defense", "health"],
        rounds=3,
    )

    print("=" * 80)
    print("GOVERNMENT MINISTER DEBATE SIMULATION")
    print("=" * 80)
    print(f"\nTopic: {debate.topic}")
    print(f"Participants: {', '.join(debate.ministers)}")
    print(f"Rounds: {debate.rounds}")
    print("\nStarting debate...\n")

    # Run the debate
    result = await debate.run()

    # Display results
    print("\n" + "=" * 80)
    print("DEBATE TRANSCRIPT")
    print("=" * 80 + "\n")

    for msg in result.messages:
        if msg.sender != "orchestrator":
            print(f"\n{'=' * 80}")
            print(f"[{msg.sender}]")
            print(f"{'=' * 80}")
            print(msg.content)

    print("\n" + "=" * 80)
    print("FINAL CONSENSUS")
    print("=" * 80)
    print(result.consensus)

    # Show statistics
    stats = result.metadata.get("statistics", {})
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Duration: {result.duration_seconds:.2f} seconds")
    print(f"Total turns: {stats.get('total_turns', 0)}")
    print(f"Unique speakers: {stats.get('unique_speakers', 0)}")
    print(f"Termination reason: {result.termination_reason.value}")


if __name__ == "__main__":
    asyncio.run(main())
