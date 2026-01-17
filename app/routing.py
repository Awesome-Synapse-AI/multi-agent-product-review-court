from __future__ import annotations

import json
import re
from typing import Any, Dict, Literal, Optional

import logging

from app.llm import build_ollama_client

RouteType = Literal["greeting", "followup", "product_review"]

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|yo|sup|howdy|greetings|hola|good\s+(morning|afternoon|evening))\b",
    re.IGNORECASE,
)
_FOLLOWUP_STRONG_RE = re.compile(
    r"\b(previous|last|earlier|above|prior|from before|as before|previous output|follow[-\s]?up)\b",
    re.IGNORECASE,
)
_FOLLOWUP_ACTION_RE = re.compile(
    r"\b(compare|summarize|explain|clarify|expand|detail|break down|why|which one|rank)\b",
    re.IGNORECASE,
)
_FOLLOWUP_PRONOUN_RE = re.compile(r"\b(it|that|those|them|these)\b", re.IGNORECASE)
_PRODUCT_HINT_RE = re.compile(
    r"\b(review|compare|vs|versus|recommend|best|top|rating|pros|cons|worth|buy|purchase|price|specs|spec|battery|camera|performance)\b",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


def route_message(message: str, has_previous: bool, llm: Any = None) -> RouteType:
    msg = (message or "").strip()
    logger.info("route_message start message_len=%s has_previous=%s", len(msg), has_previous)
    if not msg:
        logger.info("route_message empty_message -> greeting")
        return "greeting"

    label = _llm_route(msg, has_previous, llm)
    logger.info("route_message label=%s has_previous=%s", label, has_previous)
    if label:
        if label == "followup" and not has_previous:
            logger.info("route_message followup_without_history -> greeting")
            return "greeting"
        return label

    normalized = msg.lower()
    if _GREETING_RE.search(normalized) and not _PRODUCT_HINT_RE.search(normalized):
        logger.info("route_message heuristic=greeting")
        return "greeting"
    if has_previous and _FOLLOWUP_PRONOUN_RE.search(normalized):
        logger.info("route_message heuristic=followup")
        return "followup"
    logger.info("route_message heuristic=product_review")
    return "product_review"


def build_greeting_response(message: str) -> Dict[str, str]:
    logger.info("build_greeting_response")
    return {
        "message": "Hi! Tell me which product you'd like reviewed or compared, and I'll help."
    }


def build_followup_response(
    message: str, last_result: Optional[Dict[str, Any]], llm: Any = None
) -> Dict[str, str]:
    logger.info("build_followup_response has_last_result=%s", bool(last_result))
    if not last_result:
        return {
            "message": "I don't have previous results yet. Ask for a product review first."
        }

    model = llm or build_ollama_client()
    logger.info("build_followup_response using_model=%s", type(model).__name__)
    prompt = (
        "You are a product review assistant. The user asked a follow-up question.\n"
        "Use ONLY the previous result JSON below; do not invent new products or data.\n"
        "Respond in plain text, concise and helpful.\n\n"
        f"Previous result JSON:\n{json.dumps(last_result, indent=2, default=str)}\n\n"
        f"User message: \"{message}\"\n"
    )

    try:
        logger.info("build_followup_response invoking_model prompt_len=%s", len(prompt))
        response = model.invoke(prompt)
        content = getattr(response, "content", str(response)).strip()
        if not content:
            content = "I can help with that. What would you like to compare in the previous results?"
        logger.info("build_followup_response completed response_len=%s", len(content))
        return {"message": content}
    except Exception as exc:
        logger.exception("build_followup_response failed")
        return {"message": f"Unable to answer follow-up: {exc}"}


def _is_followup(normalized: str) -> bool:
    if _FOLLOWUP_STRONG_RE.search(normalized):
        return True
    if _FOLLOWUP_ACTION_RE.search(normalized) and _FOLLOWUP_PRONOUN_RE.search(normalized):
        return True
    return False


def _llm_route(
    normalized: str, has_previous: bool, llm: Any = None
) -> Optional[RouteType]:
    logger.info("_llm_route start message_len=%s has_previous=%s", len(normalized), has_previous)
    model = llm or build_ollama_client()
    prompt = (
        "Classify the user's message into one label: greeting, followup, product_review.\n"
        "Definitions:\n"
        "- greeting: small talk/hello, no product review request.\n"
        "- followup: refers to earlier results or asks to compare/summarize previous output.\n"
        "- product_review: asks for reviews, comparisons, recommendations, or product research.\n"
        f"Previous result available: {has_previous}\n"
        f"User message: \"{normalized}\"\n"
        "Return only the label."
    )
    try:
        logger.info("_llm_route invoking_model prompt_len=%s", len(prompt))
        response = model.invoke(prompt)
        content = getattr(response, "content", str(response)).strip().lower()
        cleaned = re.sub(r"[^a-z_]", "", content)
        if cleaned in {"greeting", "followup", "product_review"}:
            return cleaned  # type: ignore[return-value]
        logger.info("_llm_route invalid_label cleaned=%s", cleaned)
    except Exception:
        logger.exception("_llm_route failed")
        return None
    return None
