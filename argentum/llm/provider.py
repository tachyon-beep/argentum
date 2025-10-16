"""LLM provider abstractions."""
# pylint: disable=import-outside-toplevel  # Lazy imports to avoid circular dependencies and heavy overhead

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from argentum.models import AgentResponse, Message


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def count_tokens(self, messages: list["Message"]) -> int:
        """Count tokens in a message list."""

    @abstractmethod
    async def generate_with_tools(self, messages: list["Message"], tools: list[dict[str, Any]], **kwargs: Any) -> "AgentResponse":
        """Generate with tool support."""

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name."""

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Generate a response from the model."""


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider.

    Supports both OpenAI's official API and OpenAI-compatible local servers
    (like llama.cpp, vLLM, LocalAI, etc.) via the base_url parameter.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize OpenAI provider.

        Args:
            model: Model name to use
            api_key: OpenAI API key (or dummy value for local servers)
            organization: OpenAI organization ID (optional)
            base_url: Custom base URL for OpenAI-compatible servers
                     (e.g., "http://localhost:5000/v1" for local LLMs)
        """
        self.model = model
        self.api_key = api_key
        self.organization = organization
        self.base_url = base_url
        self._client: Any | None = None

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> str:
        """Generate a response using OpenAI's API or compatible server."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
            )

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return response.choices[0].message.content or ""

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model

    def count_tokens(self, messages: list["Message"]) -> int:
        """Count tokens in messages (rough estimate)."""
        return sum(len(str(m)) for m in messages) // 4

    async def generate_with_tools(self, messages: list["Message"], tools: list[dict[str, Any]], **kwargs: Any) -> "AgentResponse":
        """Generate response with tool support."""
        from argentum.models import AgentResponse

        msg_dicts = [{"role": m.type.value, "content": m.content} for m in messages]
        response_content = await self.generate(messages=msg_dicts, tools=tools, **kwargs)
        return AgentResponse(agent_name=self.model, content=response_content)


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI LLM provider."""

    def __init__(
        self,
        deployment_name: str,
        api_key: str | None = None,
        api_version: str = "2024-02-15-preview",
        azure_endpoint: str | None = None,
    ):
        """Initialize Azure OpenAI provider."""
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self._client: Any | None = None

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> str:
        """Generate a response using Azure OpenAI's API."""
        if self._client is None:
            from openai import AsyncAzureOpenAI

            self._client = AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,  # type: ignore[arg-type]
            )

        response = await self._client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return response.choices[0].message.content or ""

    def get_model_name(self) -> str:
        """Get the deployment name."""
        return self.deployment_name

    @property
    def model(self) -> str:
        """Get model name alias."""
        return self.deployment_name

    def count_tokens(self, messages: list["Message"]) -> int:
        """Count tokens in messages (rough estimate)."""
        return sum(len(str(m)) for m in messages) // 4

    async def generate_with_tools(self, messages: list["Message"], tools: list[dict[str, Any]], **kwargs: Any) -> "AgentResponse":
        """Generate response with tool support."""
        from argentum.models import AgentResponse

        msg_dicts = [{"role": m.type.value, "content": m.content} for m in messages]
        response_content = await self.generate(messages=msg_dicts, tools=tools, **kwargs)
        return AgentResponse(agent_name=self.deployment_name, content=response_content)
