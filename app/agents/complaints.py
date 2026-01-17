from __future__ import annotations

from typing import Any, List

from .base import build_tool_agent
from ..tools.review_tools import (
    cluster_review_topics,
    compute_long_term_reliability,
    get_review_stats,
    search_reviews,
)


COMPLAINTS_SYSTEM_PROMPT = """
You are the Complaints Agent (the Prosecutor) in a Product Review Court.

Your ONLY job is to build a clear and honest case AGAINST the given product, based on the provided review snapshot and analytic tools.

You will be given:
- product_query, product_key
- review_snapshot_id and evidence_sources

Rules:
- Cite evidence using short quoted snippets and include source_url for every snippet.
- Do not invent numbers; mark estimates as approximate.
Return ONLY valid JSON in this schema:
{
  "product_key": "string",
  "top_complaints": [
    {
      "issue": "string",
      "severity": "low|medium|high",
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
  "overall_risk_comment": "string"
}
"""


def build_complaints_agent(model: Any):
    """Return runnable bound to the complaints prompt and tools."""
    tools: List[Any] = [
        get_review_stats,
        search_reviews,
        cluster_review_topics,
        compute_long_term_reliability,
    ]
    # Note: The base agent builder handles tool binding.
    # We might need to ensure the schema is enforced via structured output or prompt instructions.
    # For now, we rely on the prompt and the tool definitions.
    runnable = build_tool_agent(COMPLAINTS_SYSTEM_PROMPT, tools)
    return runnable
