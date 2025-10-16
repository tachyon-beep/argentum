"""Concurrent orchestration pattern - parallel agent execution."""

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from argentum.agents.base import Agent
from argentum.memory.context import Context
from argentum.models import (
    AgentResponse,
    Message,
    MessageType,
    OrchestrationPattern,
    OrchestrationResult,
    Task,
    TerminationReason,
)
from argentum.orchestration.base import Orchestrator


class ConcurrentOrchestrator(Orchestrator):
    """Concurrent orchestration - agents execute in parallel.

    Multiple agents work simultaneously on the same task, providing
    independent analyses from different perspectives.
    """

    async def execute(
        self,
        agents: Sequence[Agent],
        task: Task | str,
        context: Context | None = None,
    ) -> OrchestrationResult:
        """Execute agents concurrently.

        Args:
            agents: List of agents (all execute in parallel)
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

        # Execute all agents in parallel
        agent_tasks = [self._execute_agent(agent, ctx.get_messages(), task_obj.context) for agent in agents]

        responses = await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Process responses and add to context
        final_outputs: list[AgentResponse] = []
        for agent, response in zip(agents, responses, strict=True):
            if isinstance(response, Exception):
                # Handle agent execution error
                error_message = Message(
                    type=MessageType.SYSTEM,
                    sender="orchestrator",
                    content=f"Agent {agent.name} failed: {response!s}",
                    metadata={"error": True},
                )
                ctx.add_message(error_message)
                continue

            # Response is AgentResponse here, not an exception
            agent_response: AgentResponse = response  # type: ignore[assignment]

            # Add agent's response to context
            agent_message = Message(
                type=MessageType.ASSISTANT,
                sender=agent.name,
                content=agent_response.content,
                metadata={"agent_role": agent.role.value},
            )
            ctx.add_message(agent_message)

            # Store the response
            final_outputs.append(agent_response)

        # Aggregate results (simple concatenation for now)
        consensus = self._aggregate_responses(final_outputs)

        duration = time.time() - start_time

        return OrchestrationResult(
            pattern=OrchestrationPattern.CONCURRENT,
            messages=ctx.get_messages(),
            final_outputs=final_outputs,
            consensus=consensus,
            termination_reason=TerminationReason.MAX_TURNS_REACHED,
            metadata={
                "num_agents": len(agents),
                "agent_names": [a.name for a in agents],
                "successful_agents": len(final_outputs),
            },
            duration_seconds=duration,
        )

    async def _execute_agent(
        self,
        agent: Agent,
        messages: list[Message],
        context: dict,
    ) -> Any:
        """Execute a single agent.

        Args:
            agent: Agent to execute
            messages: Conversation history
            context: Task context

        Returns:
            Agent response
        """
        return await agent.generate_response(messages=messages, context=context)

    def _aggregate_responses(self, responses: list) -> str:
        """Aggregate responses from multiple agents.

        Args:
            responses: List of agent responses

        Returns:
            Aggregated consensus
        """
        if not responses:
            return "No responses received."

        # Simple aggregation - concatenate all responses
        parts = ["Summary of all agent responses:\n"]
        for i, response in enumerate(responses, 1):
            parts.append(f"\n{i}. {response.agent_name}:")
            parts.append(f"{response.content}\n")

        return "\n".join(parts)
