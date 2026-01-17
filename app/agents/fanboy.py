from __future__ import annotations

from typing import Any, List

from .base import build_tool_agent
from ..tools.review_tools import (
    cluster_review_topics,
    get_review_stats,
    search_reviews,
)


FANBOY_SYSTEM_PROMPT = """
You are the Fanboy Agent (the Defense) in a Product Review Court.

Your ONLY job is to build a clear and honest case FOR the given product, based on the provided review snapshot and analytic tools.

Rules:
- Cite evidence using short quoted snippets and include source_url for every snippet.
- Do not invent numbers; mark estimates as approximate.
Return ONLY valid JSON in this schema:
{
  "product_key": "string",
  "top_praises": [
    {
      "strength": "string",
      "importance": "low|medium|high",
      "approx_frequency": "string",
      "snippets": [
        {
          "evidence_id": "string",
          "source_name": "string",
          "source_url": "string",
          "rating": 0,
          "text": "string",
          "review_time": "string|null"
        }
      ]
    }
  ],
  "overall_value_comment": "string"
}
"""


def build_fanboy_agent(model: Any):
    """Return runnable bound to the fanboy prompt and tools."""
    tools: List[Any] = [
        get_review_stats,
        search_reviews,
        cluster_review_topics,
    ]
    runnable = build_tool_agent(FANBOY_SYSTEM_PROMPT, tools)
    return runnable
