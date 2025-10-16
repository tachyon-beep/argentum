"""Tests for memory and context management."""

import pytest

from argentum.memory.context import Context, ConversationHistory
from argentum.models import Message, MessageType


class TestContext:
    """Test context management."""

    def test_context_creation(self):
        """Test creating a context."""
        context = Context()

        assert context.id is not None
        assert context.created_at is not None
        assert len(context.messages) == 0

    def test_add_message(self):
        """Test adding messages to context."""
        context = Context()
        msg = Message(type=MessageType.USER, sender="user", content="Hello")

        context.add_message(msg)

        assert len(context.messages) == 1
        assert context.messages[0].content == "Hello"

    def test_get_messages(self):
        """Test retrieving messages."""
        context = Context()
        msg1 = Message(type=MessageType.USER, sender="user", content="First")
        msg2 = Message(type=MessageType.ASSISTANT, sender="agent", content="Second")

        context.add_message(msg1)
        context.add_message(msg2)

        messages = context.get_messages()
        assert len(messages) == 2

    def test_get_messages_with_limit(self):
        """Test retrieving limited number of messages."""
        context = Context()

        for i in range(10):
            msg = Message(type=MessageType.USER, sender="user", content=f"Message {i}")
            context.add_message(msg)

        recent = context.get_messages(limit=3)
        assert len(recent) == 3
        assert recent[0].content == "Message 7"

    def test_get_messages_by_sender(self):
        """Test filtering messages by sender."""
        context = Context()

        msg1 = Message(type=MessageType.USER, sender="user", content="User message")
        msg2 = Message(type=MessageType.ASSISTANT, sender="agent1", content="Agent 1 message")
        msg3 = Message(type=MessageType.ASSISTANT, sender="agent2", content="Agent 2 message")

        context.add_message(msg1)
        context.add_message(msg2)
        context.add_message(msg3)

        agent1_msgs = context.get_messages_by_sender("agent1")
        assert len(agent1_msgs) == 1
        assert agent1_msgs[0].content == "Agent 1 message"

    def test_clear(self):
        """Test clearing context."""
        context = Context()
        msg = Message(type=MessageType.USER, sender="user", content="Test")
        context.add_message(msg)

        assert len(context.messages) == 1

        context.clear()

        assert len(context.messages) == 0
        assert context.summary is None

    def test_conversation_length(self):
        """Test getting conversation length."""
        context = Context()

        assert context.get_conversation_length() == 0

        for i in range(5):
            msg = Message(type=MessageType.USER, sender="user", content=f"Message {i}")
            context.add_message(msg)

        assert context.get_conversation_length() == 5

    def test_get_participants(self):
        """Test getting unique participants."""
        context = Context()

        msg1 = Message(type=MessageType.USER, sender="user", content="Hi")
        msg2 = Message(type=MessageType.ASSISTANT, sender="agent1", content="Hello")
        msg3 = Message(type=MessageType.ASSISTANT, sender="agent2", content="Hey")
        msg4 = Message(type=MessageType.USER, sender="user", content="Thanks")

        context.add_message(msg1)
        context.add_message(msg2)
        context.add_message(msg3)
        context.add_message(msg4)

        participants = context.get_participants()
        assert len(participants) == 3
        assert "user" in participants
        assert "agent1" in participants
        assert "agent2" in participants

    @pytest.mark.asyncio
    async def test_summarize(self):
        """Test context summarization."""
        context = Context()

        msg1 = Message(type=MessageType.USER, sender="user", content="Question")
        msg2 = Message(type=MessageType.ASSISTANT, sender="agent", content="Answer")

        context.add_message(msg1)
        context.add_message(msg2)

        summary = await context.summarize()

        assert "2 participants" in summary
        assert "Total messages: 2" in summary
        assert context.summary is not None

    @pytest.mark.asyncio
    async def test_summarize_empty_context(self):
        """Test summarizing an empty context."""
        context = Context()

        summary = await context.summarize()

        assert "No messages" in summary


class TestConversationHistory:
    """Test conversation history."""

    def test_conversation_history_creation(self):
        """Test creating conversation history."""
        context = Context()
        history = ConversationHistory(context=context)

        assert history.context == context
        assert len(history.agent_states) == 0

    def test_add_agent_state(self):
        """Test storing agent state."""
        context = Context()
        history = ConversationHistory(context=context)

        history.add_agent_state("agent1", {"key": "value"})

        assert "agent1" in history.agent_states
        assert history.agent_states["agent1"]["key"] == "value"

    def test_get_agent_state(self):
        """Test retrieving agent state."""
        context = Context()
        history = ConversationHistory(context=context)

        history.add_agent_state("agent1", {"data": "test"})

        # Get state
        state = history.get_agent_state("agent1")
        assert state is not None
        assert isinstance(state, dict)
        assert state["data"] == "test"

    def test_get_nonexistent_agent_state(self):
        """Test retrieving nonexistent agent state."""
        context = Context()
        history = ConversationHistory(context=context)

        state = history.get_agent_state("nonexistent")
        assert state is None
