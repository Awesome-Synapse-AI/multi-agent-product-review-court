from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.llm import build_ollama_client


class ProductCandidate(BaseModel):
    product_query: str
    product_key: str
    brand_hint: Optional[str] = None


class RequestConstraints(BaseModel):
    top_priorities: List[str] = Field(default_factory=list)
    budget_range: Optional[str] = None
    use_case: Optional[str] = None
    spec_filters: Dict[str, Any] = Field(default_factory=dict)


class IntentOutput(BaseModel):
    mode: Literal["single", "compare", "discover_then_compare"]
    product_candidates: List[ProductCandidate] = Field(default_factory=list)
    request_constraints: RequestConstraints = Field(default_factory=RequestConstraints)


def parse_intent(query: str, llm: Any = None) -> IntentOutput:
    """
    Parse user query into structured intent: mode, candidates, constraints.
    """
    model = llm or build_ollama_client()
    
    base_prompt = f"""
    You are an expert intent parser for a product review assistant.
    Analyze the user's message and extract the following:
    
    1. mode: 
       - "single": if the user asks about one specific product.
       - "compare": if the user asks to compare specific products (e.g. "A vs B").
       - "discover_then_compare": if the user asks to compare a category/brand but doesn't name specific models (e.g. "best 20-inch monitors", "Sony vs Bose headphones").
       
    2. product_candidates: list of objects with:
       - product_query: the text referring to the product.
       - product_key: a normalized, lowercase version of the product name.
       - brand_hint: if a brand is mentioned.
       
    3. request_constraints:
       - top_priorities: list of features/aspects the user cares about.
       - budget_range: price range if mentioned.
       - use_case: specific use case (e.g. "gaming", "office").
       - spec_filters: dictionary of specific specs (e.g. {{"size_inch": 20}}).

    User Message: "{query}"
    
    Return ONLY valid JSON matching this schema:
    {{
      "mode": "single|compare|discover_then_compare",
      "product_candidates": [
        {{ "product_query": "string", "product_key": "string", "brand_hint": "string|null" }}
      ],
      "request_constraints": {{
        "top_priorities": ["string"],
        "budget_range": "string|null",
        "use_case": "string|null",
        "spec_filters": {{ }}
      }}
    }}
    """
    
    max_retries = 5
    last_error = None
    
    for _ in range(max_retries):
        try:
            # prompt = base_prompt.format(query=query)
            prompt = base_prompt
            if last_error:
                prompt += f"\n\nOn the previous attempt, I got this error:\n{last_error}\n\nPlease correct the output and try again."
                last_error = None

            response = model.invoke(prompt)
            content = getattr(response, "content", str(response))
            # clean markdown code blocks
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```", "", content)
            
            data = json.loads(content)
            return IntentOutput(**data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error parsing intent, retrying: {e}")
            last_error = e

    # Fallback for simple cases or error
    if last_error:
        print(f"All retries failed. Falling back to simple intent. Last error: {last_error}")
    else:
        print("All retries failed. Falling back to simple intent.")
    return IntentOutput(
        mode="single",
        product_candidates=[
            ProductCandidate(product_query=query, product_key=query.lower().strip())
        ]
    )
