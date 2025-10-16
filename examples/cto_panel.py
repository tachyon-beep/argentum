"""Example: CTO Advisory Panel on Microservices Migration."""

import asyncio
import os

from argentum.scenarios import CTOAdvisoryPanel


async def main() -> None:
    """Run a CTO advisory panel example."""
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return

    # Create an advisory panel
    panel = CTOAdvisoryPanel(
        question="Should we migrate our monolithic e-commerce platform to microservices architecture?",
        advisors=["security", "finance", "engineering", "product"],
        rounds=2,
    )

    print("=" * 80)
    print("CTO ADVISORY PANEL CONSULTATION")
    print("=" * 80)
    print(f"\nQuestion: {panel.question}")
    print(f"Advisors: {', '.join(panel.advisors)}")
    print(f"Rounds: {panel.rounds}")
    print("\nStarting consultation...\n")

    # Run the consultation
    result = await panel.consult()

    # Display results
    print("\n" + "=" * 80)
    print("CONSULTATION TRANSCRIPT")
    print("=" * 80 + "\n")

    for msg in result.messages:
        if msg.sender != "orchestrator":
            print(f"\n{'=' * 80}")
            print(f"[{msg.sender}]")
            print(f"{'=' * 80}")
            print(msg.content)

    print("\n" + "=" * 80)
    print("SUMMARY")
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
