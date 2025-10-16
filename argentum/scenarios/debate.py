"""Government minister debate scenario."""

import asyncio

from argentum.agents.base import AgentConfig
from argentum.agents.llm_agent import LLMAgent
from argentum.coordination.chat_manager import ChatManager
from argentum.llm.provider import LLMProvider, OpenAIProvider
from argentum.models import OrchestrationResult, Role
from argentum.orchestration.group_chat import GroupChatOrchestrator


class GovernmentDebate:
    """Simulate a debate between government ministers on a policy topic."""

    def __init__(
        self,
        topic: str,
        ministers: list[str] | None = None,
        rounds: int = 5,
        provider: LLMProvider | None = None,
    ):
        """Initialize the government debate.

        Args:
            topic: Policy topic to debate
            ministers: List of minister roles (e.g., ["finance", "environment"])
            rounds: Number of debate rounds
            provider: LLM provider (defaults to OpenAI)
        """
        self.topic = topic
        self.rounds = rounds
        self.provider = provider or OpenAIProvider()

        # Default ministers if not specified
        if ministers is None:
            ministers = ["finance", "environment", "defense", "health"]

        self.minister_configs = {
            "finance": {
                "name": "Minister of Finance",
                "persona": (
                    "Focus on fiscal responsibility, economic growth, and budget impacts. You analyze costs and financial sustainability."
                ),
            },
            "environment": {
                "name": "Minister of Environment",
                "persona": ("Focus on environmental protection, sustainability, and climate impact. You advocate for green policies."),
            },
            "defense": {
                "name": "Minister of Defense",
                "persona": (
                    "Focus on national security, strategic implications, and protection of citizens. You prioritize safety and security."
                ),
            },
            "health": {
                "name": "Minister of Health",
                "persona": ("Focus on public health, healthcare access, and wellbeing of citizens. You prioritize health outcomes."),
            },
            "education": {
                "name": "Minister of Education",
                "persona": "Focus on educational outcomes, workforce development, and equal opportunity for all citizens.",
            },
            "infrastructure": {
                "name": "Minister of Infrastructure",
                "persona": "Focus on transportation, utilities, and physical infrastructure needs. You think about long-term development.",
            },
        }

        self.ministers = ministers
        self.agents: list[LLMAgent] = []

    def _create_agents(self) -> list[LLMAgent]:
        """Create agent instances for each minister.

        Returns:
            List of minister agents
        """
        agents = []

        for minister_key in self.ministers:
            if minister_key not in self.minister_configs:
                continue

            config_data = self.minister_configs[minister_key]

            config = AgentConfig(
                name=config_data["name"],
                role=Role.ADVISOR,
                persona=config_data["persona"],
                model="gpt-4",
                temperature=0.7,
                max_tokens=500,
            )

            agent = LLMAgent(config=config, provider=self.provider)
            agents.append(agent)

        return agents

    async def run(self) -> OrchestrationResult:
        """Run the government debate.

        Returns:
            Orchestration result with debate transcript
        """
        # Create agents
        self.agents = self._create_agents()

        # Set up chat manager
        chat_manager = ChatManager(
            max_turns=len(self.agents) * self.rounds,
            min_turns=len(self.agents),
        )

        # Create orchestrator
        orchestrator = GroupChatOrchestrator(chat_manager=chat_manager)

        # Run the debate
        task_description = f"""
Policy Debate Topic: {self.topic}

Instructions: Each minister should:
1. Present their department's perspective on this policy
2. Consider both benefits and concerns from your area
3. Cite specific impacts and considerations
4. Respond to points raised by other ministers
5. Work towards finding common ground or identifying key trade-offs

Let's begin the debate.
        """.strip()

        return await orchestrator.execute(
            agents=self.agents,
            task=task_description,
        )


async def main() -> None:
    """Example usage of the government debate scenario."""
    debate = GovernmentDebate(
        topic="Implementation of a National Carbon Tax",
        ministers=["finance", "environment", "defense", "health"],
        rounds=3,
    )

    print("Starting government minister debate...")
    print(f"Topic: {debate.topic}")
    print(f"Ministers: {', '.join(debate.ministers)}\n")

    result = await debate.run()

    print("\n" + "=" * 80)
    print("DEBATE TRANSCRIPT")
    print("=" * 80 + "\n")

    for msg in result.messages:
        if msg.sender != "orchestrator":
            print(f"[{msg.sender}]")
            print(msg.content)
            print()

    print("=" * 80)
    print("CONSENSUS")
    print("=" * 80)
    print(result.consensus)
    print()

    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Termination: {result.termination_reason.value}")


if __name__ == "__main__":
    asyncio.run(main())
