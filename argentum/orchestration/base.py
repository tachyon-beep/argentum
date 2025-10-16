"""Base orchestration classes."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from argentum.agents.base import Agent
from argentum.memory.context import Context
from argentum.models import OrchestrationResult, Task


class Orchestrator(ABC):
    """Base class for all orchestration patterns."""

    @abstractmethod
    async def execute(
        self,
        agents: Sequence[Agent],
        task: Task | str,
        context: Context | None = None,
    ) -> OrchestrationResult:
        """Execute the orchestration pattern.

        Args:
            agents: List of agents to orchestrate
            task: Task to execute (or simple string description)
            context: Optional shared context

        Returns:
            Orchestration result
        """
        ...

    def _prepare_task(self, task: Task | str) -> Task:
        """Convert task to Task object if it's a string.

        Args:
            task: Task or string description

        Returns:
            Task object
        """
        if isinstance(task, str):
            return Task(description=task)
        return task

    def _prepare_context(self, context: Context | None) -> Context:
        """Prepare context for orchestration.

        Args:
            context: Optional context

        Returns:
            Context object
        """
        if context is None:
            return Context()
        return context
