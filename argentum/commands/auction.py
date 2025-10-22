from __future__ import annotations

import asyncio
import sys

import click

from argentum.cli_utils import (
    console,
    _default_agent_config,
    _prepare_session_environment,
    _persist_session_result,
)
from argentum.memory.context import Context
from argentum.agents.llm_agent import LLMAgent
from argentum.llm.provider import OpenAIProvider
from argentum.orchestration.auction_chat import AuctionGroupChatOrchestrator
from argentum.coordination.auction_manager import create_default_auction_manager
from argentum.persistence import ConversationSession
from argentum.audio.factory import get_audio_controller


@click.command()
@click.argument("prompt")
@click.option("--agents", "-a", multiple=True, help="Agents to include (defaults to manifest default_agents)")
@click.option("--micro-turns", "-t", default=4, help="Number of micro-turns to run")
@click.option("--emotion-control", type=click.Choice(["heuristic", "llm", "hybrid"], case_sensitive=False))
@click.option("--interjection-min-importance", type=float)
@click.option("--session-id", "-s")
@click.option("--project", "-p", required=True)
@click.option("--summary-mode", type=click.Choice(["heuristic", "frontier", "local", "none"], case_sensitive=False))
@click.option("--summary-command")
@click.option("--emit-provisional-interrupt", is_flag=True, help="Emit provisional interrupt events for observability.")
@click.option("--tts-provider", type=click.Choice(["sim", "elevenlabs"], case_sensitive=False), help="Override TTS provider for this run.")
@click.option("--tts-voice", help="Override TTS voice (provider-specific).")
@click.option("--tts-latency", type=click.Choice(["low", "balanced", "high"], case_sensitive=False), help="TTS latency mode (provider-specific).")
def auction(prompt, agents, micro_turns, emotion_control, interjection_min_importance, session_id, project, summary_mode, summary_command, emit_provisional_interrupt, tts_provider, tts_voice, tts_latency):
    """Run an auction-based group chat with interjections and interrupts."""
    env = _prepare_session_environment(command="auction", project=project, session_id=session_id, seed=prompt, summary_mode=summary_mode, summary_command=summary_command)
    manifest = env.manifest or {}
    auction_cfg = ((manifest.get("conversation") or {}).get("auction") or {}).copy()
    if emotion_control:
        auction_cfg["emotion_control"] = emotion_control
    if interjection_min_importance is not None:
        auction_cfg["interjection_min_importance"] = float(interjection_min_importance)

    manager = create_default_auction_manager(auction_cfg)

    # Optional TTS overrides: rebuild audio controller if flags provided
    if any([tts_provider, tts_voice, tts_latency]):
        m = dict(manifest)
        convo = dict((m.get("conversation") or {}))
        tts_cfg = dict((convo.get("tts") or {}))
        if tts_provider:
            tts_cfg["provider"] = "elevenlabs" if tts_provider.lower() == "elevenlabs" else "sim"
        if tts_voice is not None:
            tts_cfg["voice"] = tts_voice
        if tts_latency is not None:
            tts_cfg["latency_mode"] = tts_latency
        convo["tts"] = tts_cfg
        m["conversation"] = convo
        try:
            env.audio_controller = get_audio_controller(m)
        except Exception:
            pass


    agent_keys = list(agents) if agents else []
    if not agent_keys:
        defaults = manifest.get("default_agents") or []
        if isinstance(defaults, list):
            agent_keys = [str(x) for x in defaults]
    if not agent_keys:
        agent_keys = ["agent_a", "agent_b"]

    provider = OpenAIProvider(model="gpt-4")
    agent_objs = [LLMAgent(config=_default_agent_config(k), provider=provider) for k in agent_keys]

    orch = AuctionGroupChatOrchestrator(manager=manager, audio=env.audio_controller, micro_turns=micro_turns, enable_interjections=True, enable_interrupts=True, interjection_min_importance=float(auction_cfg.get("interjection_min_importance", 0.5)), emit_provisional_interrupt=bool(emit_provisional_interrupt))

    session = ConversationSession(store=env.store, session_id=env.session_id, metadata=dict(env.metadata))

    async def run_auction():
        console.print("\n[bold]Prompt:[/bold] " + prompt)
        console.print("[bold]Agents:[/bold] " + ", ".join(a.name for a in agent_objs) + "\n")
        with console.status("[bold green]Running auction chat..."):
            result = await orch.execute(agent_objs, task=prompt, context=Context())
        console.print("\n" + "=" * 80)
        console.print("[bold]TRANSCRIPT[/bold]", justify="center")
        console.print("=" * 80 + "\n")
        for msg in result.messages:
            if msg.sender != "orchestrator":
                console.print("\n[bold cyan][" + msg.sender + "][/bold cyan]")
                console.print(msg.content)
        console.print("\n" + "=" * 80)
        console.print("[bold]CONSENSUS[/bold]", justify="center")
        console.print("=" * 80)
        console.print(result.consensus or "")
        try:
            transcript_path = await _persist_session_result(env, session, env.metadata, result)
            console.print("[dim]Transcript saved to: " + str(transcript_path) + "[/dim]")
        except (OSError, ValueError) as error:
            console.print("[yellow]Warning: failed to persist transcript (" + str(error) + ").[/yellow]")

    try:
        asyncio.run(run_auction())
    except KeyboardInterrupt:
        console.print("\n[yellow]Auction chat interrupted by user[/yellow]")
        sys.exit(0)
    except (RuntimeError, ValueError, OSError) as e:
        console.print("\n[red]Error: " + str(e) + "[/red]")
        sys.exit(1)
