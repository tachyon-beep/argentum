"""Virtual CTO advisory panel scenario."""

import asyncio
from typing import TYPE_CHECKING, Any

from argentum.agents.base import AgentConfig
from argentum.agents.llm_agent import LLMAgent
from argentum.coordination.chat_manager import ChatManager
from argentum.llm.provider import LLMProvider, OpenAIProvider
from argentum.memory.agent_memory import AgentMemoryStore
from argentum.models import OrchestrationResult, Role, Task
from argentum.orchestration.group_chat import GroupChatOrchestrator
from argentum.workspace.profiles import apply_speech_defaults, load_agent_profile

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from argentum.workspace.knowledge import SessionRetriever
    from argentum.workspace.manager import ProjectWorkspace


class CTOAdvisoryPanel:
    """Simulate a CTO consulting with expert advisors on a technical question."""

    def __init__(
        self,
        question: str,
        advisors: list[str] | None = None,
        rounds: int = 3,
        provider: LLMProvider | None = None,
        memory_store: AgentMemoryStore | None = None,
        workspace: "ProjectWorkspace | None" = None,
        context_documents: list[dict[str, Any]] | None = None,
        retriever: "SessionRetriever | None" = None,
    ):
        """Initialize the CTO advisory panel.

        Args:
            question: Technical question or decision to discuss
            advisors: List of advisor roles
            rounds: Number of discussion rounds
            provider: LLM provider (defaults to OpenAI)
        """
        self.question = question
        self.rounds = rounds
        self.provider = provider or OpenAIProvider()
        self.memory_store = memory_store
        self.workspace = workspace
        self.context_documents = context_documents if context_documents is not None else []
        self.retriever = retriever
        self.manifest = workspace.load_manifest() if workspace else None

        # Default advisors if not specified
        if advisors is None:
            advisors = ["security", "finance", "engineering", "product"]

        self.advisor_configs = {
            "security": {
                "name": "Chief Security Officer",
                "persona": (
                    "Expert in cybersecurity, threat modeling, and compliance. You assess security risks and data protection concerns."
                ),
            },
            "finance": {
                "name": "Chief Financial Officer",
                "persona": ("Expert in cost analysis, ROI, and financial planning. You evaluate budget impacts and long-term costs."),
            },
            "engineering": {
                "name": "VP of Engineering",
                "persona": (
                    "Expert in technical architecture, scalability, and engineering best practices. "
                    "You assess technical feasibility and complexity."
                ),
            },
            "product": {
                "name": "Chief Product Officer",
                "persona": (
                    "Expert in user experience, product strategy, and market fit. You consider customer impact and competitive advantage."
                ),
            },
            "operations": {
                "name": "VP of Operations",
                "persona": "Expert in operational efficiency, deployment, and maintenance. You think about day-to-day running and support.",
            },
            "data": {
                "name": "Chief Data Officer",
                "persona": "Expert in data strategy, analytics, and AI/ML. You assess data implications and insights opportunities.",
            },
        }

        self.advisors = advisors
        self.agents: list[LLMAgent] = []

    def _create_agents(self) -> list[LLMAgent]:
        """Create agent instances for each advisor.

        Returns:
            List of advisor agents
        """
        agents = []

        for advisor_key in self.advisors:
            if advisor_key not in self.advisor_configs:
                continue

            config_data = self.advisor_configs[advisor_key]

            config = AgentConfig(
                name=config_data["name"],
                role=Role.ADVISOR,
                persona=config_data["persona"],
                model="gpt-4",
                temperature=0.7,
                max_tokens=400,
                metadata={"slug": advisor_key},
            )

            config = apply_speech_defaults(config, self.manifest, advisor_key)
            config = load_agent_profile(self.workspace, advisor_key, config)

            if self.memory_store:
                agent = LLMAgent(
                    config=config,
                    provider=self.provider,
                    memory_store=self.memory_store,
                    retriever=self.retriever,
                )
            else:
                agent = LLMAgent(config=config, provider=self.provider, retriever=self.retriever)
            agents.append(agent)

        return agents

    async def consult(self) -> OrchestrationResult:
        """Run the advisory panel consultation.

        Returns:
            Orchestration result with consultation transcript
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

        # Run the consultation
        task_description = f"""
CTO Question: {self.question}

        Instructions: Each advisor should:
1. Provide analysis from your domain expertise
2. Identify key risks, benefits, or considerations
3. Build on or respond to points raised by other advisors
4. Be specific with recommendations
5. Help the CTO make an informed decision
6. If you need additional evidence, reply with <<retrieve: your search terms>> and pause for the new documents before replying fully.

Let's begin the consultation.
        """.strip()

        task = Task(
            description=task_description,
            context={
                "question": self.question,
                "memory_topic": self.question,
                "knowledge_documents": self.context_documents,
                "retrieval_history": self.retriever.history if self.retriever else [],
            },
        )

        return await orchestrator.execute(
            agents=self.agents,
            task=task,
        )


async def main() -> None:
    """Example usage of the CTO advisory panel scenario."""
    panel = CTOAdvisoryPanel(
        question="Should we migrate our monolithic application to microservices architecture?",
        advisors=["security", "finance", "engineering", "product"],
        rounds=2,
    )

    print("Starting CTO advisory panel consultation...")
    print(f"Question: {panel.question}")
    print(f"Advisors: {', '.join(panel.advisors)}\n")

    result = await panel.consult()

    print("\n" + "=" * 80)
    print("CONSULTATION TRANSCRIPT")
    print("=" * 80 + "\n")

    for msg in result.messages:
        if msg.sender != "orchestrator":
            print(f"[{msg.sender}]")
            print(msg.content)
            print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(result.consensus)
    print()

    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Termination: {result.termination_reason.value}")


if __name__ == "__main__":
    asyncio.run(main())
