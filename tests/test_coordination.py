"""Tests for chat manager and coordination."""

from conftest import MockLLMProvider

from argentum.agents.llm_agent import LLMAgent
from argentum.coordination.chat_manager import ChatManager, SpeakerSelectionMode
from argentum.memory.context import Context
from argentum.models import TerminationReason


class TestChatManager:
    """Test chat manager functionality."""

    def test_chat_manager_creation(self):
        """Test creating a chat manager."""
        manager = ChatManager(max_turns=10, selection_mode=SpeakerSelectionMode.ROUND_ROBIN)

        assert manager.max_turns == 10
        assert manager.selection_mode == SpeakerSelectionMode.ROUND_ROBIN

    def test_chat_manager_defaults(self):
        """Test default chat manager settings."""
        manager = ChatManager()

        assert manager.max_turns == 10
        assert manager.selection_mode == SpeakerSelectionMode.ROUND_ROBIN
        assert manager.allow_repeats is True
        assert manager.min_turns == 2

    def test_round_robin_selection(self, multiple_agent_configs):
        """Test round-robin speaker selection."""
        provider = MockLLMProvider()
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        manager = ChatManager(selection_mode=SpeakerSelectionMode.ROUND_ROBIN)
        context = Context()

        # First turn
        speaker1 = manager.select_next_speaker(agents, context)
        assert speaker1 == agents[0]

        assert speaker1 is not None
        manager.record_turn(speaker1)

        # Second turn
        speaker2 = manager.select_next_speaker(agents, context)
        assert speaker2 == agents[1]

        assert speaker2 is not None
        manager.record_turn(speaker2)

        # Third turn
        speaker3 = manager.select_next_speaker(agents, context)
        assert speaker3 == agents[2]

    def test_max_turns_termination(self):
        """Test termination at max turns."""
        manager = ChatManager(max_turns=3)
        context = Context()

        # Before max turns
        should_term, reason = manager.should_terminate(context)
        assert should_term is False

        # Simulate turns
        manager._current_turn = 3

        # At max turns
        should_term, reason = manager.should_terminate(context)
        assert should_term is True
        assert reason == TerminationReason.MAX_TURNS_REACHED

    def test_min_turns_requirement(self):
        """Test minimum turns requirement."""
        manager = ChatManager(max_turns=10, min_turns=3)
        context = Context()

        # Before min turns
        manager._current_turn = 2
        should_term, _ = manager.should_terminate(context)
        assert should_term is False

    def test_record_turn(self, agent_config):
        """Test recording agent turns."""
        provider = MockLLMProvider()
        agent = LLMAgent(config=agent_config, provider=provider)

        manager = ChatManager()
        assert manager.get_turn_count() == 0

        manager.record_turn(agent)
        assert manager.get_turn_count() == 1

        manager.record_turn(agent)
        assert manager.get_turn_count() == 2

    def test_speaker_history(self, multiple_agent_configs):
        """Test speaker history tracking."""
        provider = MockLLMProvider()
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        manager = ChatManager()

        manager.record_turn(agents[0])
        manager.record_turn(agents[1])
        manager.record_turn(agents[0])

        history = manager.get_speaker_history()
        assert len(history) == 3
        assert history[0] == "Agent 1"
        assert history[1] == "Agent 2"
        assert history[2] == "Agent 1"

    def test_reset(self, agent_config):
        """Test resetting chat manager state."""
        provider = MockLLMProvider()
        agent = LLMAgent(config=agent_config, provider=provider)

        manager = ChatManager()
        manager.record_turn(agent)
        manager.record_turn(agent)

        assert manager.get_turn_count() == 2

        manager.reset()

        assert manager.get_turn_count() == 0
        assert len(manager.get_speaker_history()) == 0

    def test_statistics(self, multiple_agent_configs):
        """Test conversation statistics."""
        provider = MockLLMProvider()
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        manager = ChatManager(max_turns=10)

        manager.record_turn(agents[0])
        manager.record_turn(agents[1])
        manager.record_turn(agents[0])

        stats = manager.get_statistics()

        assert stats["total_turns"] == 3
        assert stats["unique_speakers"] == 2
        assert stats["speaker_counts"]["Agent 1"] == 2
        assert stats["speaker_counts"]["Agent 2"] == 1
        assert stats["turns_remaining"] == 7

    def test_random_selection(self, multiple_agent_configs):
        """Test random speaker selection."""
        provider = MockLLMProvider()
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        manager = ChatManager(selection_mode=SpeakerSelectionMode.RANDOM)
        context = Context()

        # Should select an agent (can't predict which)
        speaker = manager.select_next_speaker(agents, context)
        assert speaker in agents

    def test_selection_returns_none_at_max_turns(self, multiple_agent_configs):
        """Test speaker selection returns None when max turns reached."""
        provider = MockLLMProvider()
        agents = [LLMAgent(config=cfg, provider=provider) for cfg in multiple_agent_configs]

        manager = ChatManager(max_turns=2)
        manager._current_turn = 2

        context = Context()
        speaker = manager.select_next_speaker(agents, context)

        assert speaker is None
