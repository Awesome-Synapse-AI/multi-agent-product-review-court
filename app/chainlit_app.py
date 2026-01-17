from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Tuple

import chainlit as cl

from app.graph import run_workflow

logger = logging.getLogger(__name__)

WELCOME_TEXT = (
    "Welcome to Product Review Court (Chainlit). "
    "Ask about a single product or compare a few—I'll pull reviews, run the Complaints "
    "and Fanboy agents, and let the Judge summarize."
)


def _ensure_session_id() -> str:
    """
    Persist a per-chat session id so LangGraph memory can track the conversation.
    """
    session_id = cl.user_session.get("session_id")  # type: ignore[arg-type]
    if not session_id:
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)
    return session_id


def _extract_display_fields(result: Any) -> Tuple[str, Dict[str, Any], str]:
    """
    Normalize LangGraph output for Chainlit rendering.
    """
    def _format_judge(out: Dict[str, Any]) -> str:
        verdict = out.get("verdict", {}) if isinstance(out, dict) else {}
        headline = out.get("headline") or verdict.get("headline") or ""
        summary = out.get("summary") or verdict.get("summary") or ""
        winner = verdict.get("winner_product_key") or out.get("winner_product_key")
        pros = out.get("pros") or verdict.get("pros") or []
        cons = out.get("cons") or verdict.get("cons") or []
        evidence = out.get("evidence") or verdict.get("evidence") or []
        uncertainties = out.get("uncertainties") or verdict.get("uncertainties") or []

        sections: list[str] = []
        headline_line = ""
        if headline:
            headline_line = f"**{headline}**"
        elif winner:
            headline_line = f"**Winner:** {winner}"
        if headline_line:
            sections.append(headline_line)
        if summary:
            sections.append(summary)
        if pros:
            pros_block = ["**Pros:**"]
            pros_block.extend(f"- {p}" for p in pros if p)
            sections.append("\n".join(pros_block))
        if cons:
            cons_block = ["**Cons:**"]
            cons_block.extend(f"- {c}" for c in cons if c)
            sections.append("\n".join(cons_block))
        if evidence:
            ev_block = ["**Evidence:**"]
            for ev in evidence:
                if isinstance(ev, dict):
                    quote = ev.get("quote")
                    source = ev.get("source_url") or ev.get("url")
                    parts = []
                    if quote:
                        parts.append(quote)
                    if source:
                        parts.append(f"(source: {source})")
                    if parts:
                        ev_block.append(f"- {' '.join(parts)}")
                elif isinstance(ev, str):
                    ev_block.append(f"- {ev}")
            if len(ev_block) > 1:
                sections.append("\n".join(ev_block))
        if uncertainties:
            unc_block = ["**Uncertainties:**"]
            unc_block.extend(f"- {u}" for u in uncertainties if u)
            sections.append("\n".join(unc_block))
        return "\n\n".join([s for s in sections if s])

    if isinstance(result, dict):
        judge_output: Dict[str, Any] = result.get("judge_output") or {}
        mode = result.get("mode", "single")
        tokens = result.get("final_response_tokens") or []
        final_text = ""

        # Prefer the formatted judge view if we have any judge output at all.
        formatted = _format_judge(judge_output) if judge_output else ""
        if formatted:
            final_text = formatted

        # Otherwise fall back to model-provided text or streamed tokens.
        if not final_text:
            final_text = result.get("final_response_text") or ""
        if not final_text and isinstance(tokens, list) and tokens:
            final_text = "".join(tokens)
        if not final_text and isinstance(judge_output, dict) and "message" in judge_output:
            msg_val = judge_output.get("message")
            if isinstance(msg_val, str):
                final_text = msg_val
        if not final_text and judge_output and not formatted:
            final_text = json.dumps(judge_output, indent=2)

        if not isinstance(judge_output, dict):
            judge_output = {"raw_output": judge_output}
        return final_text, judge_output, mode

    return str(result), {}, "single"


@cl.on_chat_start
async def on_chat_start() -> None:
    _ensure_session_id()
    await cl.Message(content=WELCOME_TEXT).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    session_id = _ensure_session_id()
    user_text = message.content if hasattr(message, "content") else str(message)
    msg = cl.Message(content="", author="assistant")
    await msg.send()

    streamed_any = False

    def on_stream(token: str) -> None:
        nonlocal streamed_any
        streamed_any = True
        if token:
            cl.run_sync(msg.stream_token(token))

    try:
        result = await cl.make_async(run_workflow)(
            user_text,
            session_id=session_id,
            stream_callback=on_stream,
        )
    except Exception as exc:
        logger.exception("Chainlit workflow failed")
        msg.content = f"Something went wrong: {exc}"
        await msg.update()
        return

    final_text, judge_output, mode = _extract_display_fields(result)
    if not final_text:
        final_text = "No response generated."

    # Ensure markdown renders (update after streaming to replace flat token stream).
    msg.content = final_text
    msg.elements = []  # force re-render as message body
    await msg.update()
