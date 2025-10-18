"""Group chat orchestration pattern - interactive multi-agent debate."""

import time
from collections.abc import Sequence

from argentum.agents.base import Agent
from argentum.coordination.chat_manager import ChatManager
from argentum.memory.context import Context
from argentum.models import (
    Message,
    MessageType,
    OrchestrationPattern,
    OrchestrationResult,
    Task,
)
from argentum.orchestration.base import Orchestrator


class GroupChatOrchestrator(Orchestrator):
    """Group chat orchestration - agents interact in a shared conversation.

    Multiple agents converse, debate, and collaborate in turns,
    managed by a chat manager that controls conversation flow.
    """

    def __init__(self, chat_manager: ChatManager | None = None):
        """Initialize the group chat orchestrator.

        Args:
            chat_manager: Chat manager instance (creates default if None)
        """
        self.chat_manager = chat_manager or ChatManager()

    async def execute(
        self,
        agents: Sequence[Agent],
        task: Task | str,
        context: Context | None = None,
    ) -> OrchestrationResult:
        """Execute group chat among agents.

        Args:
            agents: List of agents to participate
            task: Task to discuss
            context: Shared context

        Returns:
            Orchestration result
        """
        start_time = time.time()
        task_obj = self._prepare_task(task)
        ctx = self._prepare_context(context)

        # Reset chat manager
        self.chat_manager.reset()

        # Create initial task message
        task_message = Message(
            type=MessageType.USER,
            sender="orchestrator",
            content=task_obj.description,
        )
        ctx.add_message(task_message)

        # Main conversation loop
        final_outputs = []

        while True:
            # Check termination
            should_terminate, reason = self.chat_manager.should_terminate(ctx)
            if should_terminate:
                break

            # Select next speaker
            next_speaker = self.chat_manager.select_next_speaker(agents, ctx)
            if next_speaker is None:
                break

            # Get agent's response
            try:
                response = await next_speaker.generate_response(
                    messages=ctx.get_messages(),
                    context=task_obj.context,
                )

                # Add to context
                metadata = {
                    "agent_role": next_speaker.role.value,
                    "turn": self.chat_manager.get_turn_count() + 1,
                }
                if isinstance(response.metadata, dict):
                    metadata.update(response.metadata)

                agent_message = Message(
                    type=MessageType.ASSISTANT,
                    sender=next_speaker.name,
                    content=response.content,
                    metadata=metadata,
                )
                ctx.add_message(agent_message)

                # Store response
                final_outputs.append(response)

                # Notify agent
                next_speaker.receive_message(agent_message)

                # Record turn
                self.chat_manager.record_turn(next_speaker)

            except (RuntimeError, ValueError, TypeError) as e:
                # Handle agent error
                error_message = Message(
                    type=MessageType.SYSTEM,
                    sender="orchestrator",
                    content=f"Error from {next_speaker.name}: {e!s}",
                    metadata={"error": True, "agent": next_speaker.name},
                )
                ctx.add_message(error_message)
                # Continue to next turn
                continue

        # Generate consensus
        consensus = self._generate_consensus(final_outputs)

        duration = time.time() - start_time

        return OrchestrationResult(
            pattern=OrchestrationPattern.GROUP_CHAT,
            messages=ctx.get_messages(),
            final_outputs=final_outputs,
            consensus=consensus,
            termination_reason=reason,
            metadata={
                "num_agents": len(agents),
                "agent_names": [a.name for a in agents],
                "statistics": self.chat_manager.get_statistics(),
            },
            duration_seconds=duration,
        )

    def _generate_consensus(self, responses: list) -> str:
        """Generate a consensus from all responses.

        Args:
            responses: List of agent responses

        Returns:
            Consensus summary
        """
        if not responses:
            return "No consensus reached - no responses."

        # Simple consensus - get the last response as the summary
        # In a more sophisticated version, this could use an LLM to synthesize
        return f"Debate concluded with {len(responses)} contributions. Final perspective: {responses[-1].content}"
