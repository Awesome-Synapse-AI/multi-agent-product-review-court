from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.graph import run_workflow
from app.config import get_settings

logger = logging.getLogger(__name__)

def _configure_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


_configure_logging()

def _normalize_chat_result(result: Any) -> Tuple[str, List[Dict], Dict, str, List[str]]:
    if not isinstance(result, dict):
        return "single", [], {"raw_output": result}, "", []
    mode = result.get("mode", "single")
    products = result.get("products", [])
    judge_output = result.get("judge_output", {})
    if not isinstance(products, list):
        products = []
    if not isinstance(judge_output, dict):
        judge_output = {"raw_output": judge_output}
    final_text = result.get("final_response_text", "") if isinstance(result, dict) else ""
    final_tokens = result.get("final_response_tokens", []) if isinstance(result, dict) else []
    if not isinstance(final_tokens, list):
        final_tokens = []
    return mode, products, judge_output, final_text, final_tokens

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    mode: str
    products: List[Dict]
    judge_output: Dict
    final_response_text: str = ""
    final_response_tokens: List[str] = Field(default_factory=list)

app = FastAPI(title="Product Review Court", version="0.2.0")

@app.get("/")
def root():
    """
    Lightweight health/info endpoint.
    """
    return {
        "status": "ok",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "chat_endpoint": "/api/chat",
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """
    Main endpoint for Product Review Court.
    Accepts a user message and returns a structured verdict.
    """
    try:
        logger.info("POST /api/chat session_id=%s message_len=%s", payload.session_id, len(payload.message))
        result = run_workflow(payload.message, session_id=payload.session_id)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        mode, products, judge_output, final_text, final_tokens = _normalize_chat_result(result)
        return ChatResponse(
            mode=mode,
            products=products,
            judge_output=judge_output,
            final_response_text=final_text,
            final_response_tokens=final_tokens,
        )
    except Exception as e:
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))
