"""Sequential orchestration pattern - pipeline of agents."""

import time
from collections.abc import Sequence

from argentum.agents.base import Agent
from argentum.memory.context import Context
from argentum.models import (
    Message,
    MessageType,
    OrchestrationPattern,
    OrchestrationResult,
    Task,
    TerminationReason,
)
from argentum.orchestration.base import Orchestrator


class SequentialOrchestrator(Orchestrator):
    """Sequential orchestration - agents execute in a fixed linear order.

    Each agent's output becomes input for the next agent, creating a pipeline.
    Useful for workflows with clear stages or progressive refinement.
    """

    async def execute(
        self,
        agents: Sequence[Agent],
        task: Task | str,
        context: Context | None = None,
    ) -> OrchestrationResult:
        """Execute agents sequentially in order.

        Args:
            agents: List of agents (order matters)
            task: Task to execute
            context: Shared context

        Returns:
            Orchestration result
        """
        start_time = time.time()
        task_obj = self._prepare_task(task)
        ctx = self._prepare_context(context)

        # Create initial task message
        task_message = Message(
            type=MessageType.USER,
            sender="orchestrator",
            content=task_obj.description,
        )
        ctx.add_message(task_message)

        # Execute agents in sequence
        final_outputs = []

        for i, agent in enumerate(agents):
            # Get conversation history for this agent
            messages = ctx.get_messages()

            # Generate response
            response = await agent.generate_response(
                messages=messages,
                context=task_obj.context,
            )

            # Add agent's response to context
            metadata = {
                "agent_role": agent.role.value,
                "stage": i + 1,
                "total_stages": len(agents),
            }
            if isinstance(response.metadata, dict):
                metadata.update(response.metadata)

            agent_message = Message(
                type=MessageType.ASSISTANT,
                sender=agent.name,
                content=response.content,
                metadata=metadata,
            )
            ctx.add_message(agent_message)

            # Store the response
            final_outputs.append(response)

            # Notify agent of the message
            agent.receive_message(agent_message)

        duration = time.time() - start_time

        return OrchestrationResult(
            pattern=OrchestrationPattern.SEQUENTIAL,
            messages=ctx.get_messages(),
            final_outputs=final_outputs,
            consensus=final_outputs[-1].content if final_outputs else None,
            termination_reason=TerminationReason.MAX_TURNS_REACHED,
            metadata={
                "num_agents": len(agents),
                "agent_names": [a.name for a in agents],
            },
            duration_seconds=duration,
        )
