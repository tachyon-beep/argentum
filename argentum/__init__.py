"""
Argentum: A versatile multi-agent AI dialogue system.

Argentum enables multiple specialized AI agents to collaborate, debate, and provide
advisory insights through various orchestration patterns.
"""

__version__ = "0.1.0"
__author__ = "Argentum Team"

from argentum.agents.base import Agent, AgentConfig
from argentum.coordination.chat_manager import ChatManager
from argentum.llm.provider import LLMProvider
from argentum.memory.context import Context, Message
from argentum.orchestration.concurrent import ConcurrentOrchestrator
from argentum.orchestration.group_chat import GroupChatOrchestrator
from argentum.orchestration.sequential import SequentialOrchestrator

__all__ = [
    "Agent",
    "AgentConfig",
    "ChatManager",
    "ConcurrentOrchestrator",
    "Context",
    "GroupChatOrchestrator",
    "LLMProvider",
    "Message",
    "SequentialOrchestrator",
]
