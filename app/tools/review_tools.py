from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import re
from urllib.parse import urlparse

from app.llm import build_ollama_client

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _call_tool(tool: Any, kwargs: Dict[str, Any]) -> Any:
    if hasattr(tool, "invoke"):
        return tool.invoke(kwargs)
    if hasattr(tool, "run"):
        return tool.run(**kwargs)
    if callable(tool):
        return tool(**kwargs)
    raise TypeError(f"Tool is not callable: {tool}")


def _google_search(
    query: str,
    num: int = 10,
    start: int = 1,
    site_restrict: Optional[str] = None,
    date_restrict: Optional[str] = None,
    gl: Optional[str] = None,
    lr: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Helper to call Google Custom Search JSON API.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": settings.google_api_key,
        "cx": settings.google_cse_id,
        "q": query,
        "num": min(num, 10),  # API limit is 10
        "start": start,
    }
    if site_restrict:
        params["q"] += f" site:{site_restrict}"
    if date_restrict:
        params["dateRestrict"] = date_restrict
    if gl:
        params["gl"] = gl
    if lr:
        params["lr"] = lr

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Google Search API error: {e}")
        return {"items": []}


@tool("web_search_review_pages")
def web_search_review_pages(
    product_query: str,
    sites: Optional[List[str]] = None,
    k: int = 10,
    region: Optional[str] = None,
    recency_days: Optional[int] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Find URLs likely containing reviews for a product by querying Google CSE.
    """
    results = []
    start = 1
    date_restrict = f"d{recency_days}" if recency_days else None
    
    # If sites are provided, we might need to iterate or join them. 
    # For simplicity, if multiple sites, we might do a broad search or join with OR if supported,
    # but site: operator usually takes one. We can try `site:a.com OR site:b.com`.
    site_str = None
    if sites:
        site_str = " OR ".join(sites) # This might be too long for query, but let's try.

    while len(results) < k:
        num_to_fetch = min(k - len(results), 10)
        data = _google_search(
            query=product_query + " reviews",
            num=num_to_fetch,
            start=start,
            site_restrict=site_str,
            date_restrict=date_restrict,
            gl=region,
            lr=language,
        )
        
        items = data.get("items", [])
        if not items:
            break
            
        for item in items:
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "source_name": urlparse(item.get("link")).netloc,
            })
            
        start += len(items)
        if start > 100: # Google CSE limit
            break
            
    return {"results": results[:k]}


@tool("discover_products")
def discover_products(
    category_query: str,
    brands: List[str],
    spec_filters: Optional[Dict[str, Any]] = None,
    max_per_brand: int = 3,
    region: Optional[str] = None,
    recency_days: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Given a category/brand/spec prompt, discover concrete product models suitable for comparison.
    """
    candidates = []
    for brand in brands:
        query = f'{brand} "{category_query}" (review OR "best" OR "top")'
        # Add spec filters to query if simple enough, else ignore for discovery
        if spec_filters:
            for key, val in spec_filters.items():
                query += f" {val}"
        
        # Search
        search_res = _call_tool(
            web_search_review_pages,
            {
                "product_query": query,
                "k": max_per_brand * 2,  # Fetch a few more to filter
                "region": region,
                "recency_days": recency_days,
            },
        )
        
        # Simple heuristic extraction (in reality, this needs an LLM or regex to extract model names)
        # For this tool, we will return the titles as proxies for product queries if we can't extract better.
        # Ideally, we'd pass these snippets to an LLM to extract the model name.
        # Since this is a tool, we might just return the search results as "candidates" 
        # or do some basic string manipulation.
        
        # Let's just take the top results and assume the title contains the product name.
        for item in search_res["results"][:max_per_brand]:
            candidates.append({
                "brand": brand,
                "product_query": item["title"], # This is a weak assumption but fits the tool interface
                "rationale": f"Found in search result: {item['title']}",
                "source_url": item["url"]
            })
            
    return {"candidates": candidates}


@tool("fetch_and_extract_reviews")
def fetch_and_extract_reviews(url: str, max_reviews: int = 50) -> Dict[str, Any]:
    """
    Fetch a page and use an LLM to extract reviews into ReviewEvidence[] if available.
    Otherwise, returns an empty list.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        # Get clean text, remove excessive whitespace
        page_text = " ".join(soup.get_text().split())

        # Limit text to avoid exceeding token limits (e.g., first 15k chars)
        page_text = page_text[:15000]

        if len(page_text) < 100:
            return {"reviews": [], "page_metadata": {"error": "Not enough content."}}

        llm = build_ollama_client()
        prompt = f"""
        You are an expert data extractor. From the following webpage text, extract all user reviews.

        Return ONLY a valid JSON object with a single key "reviews" containing a list of review objects.
        Each review object in the list must have this exact schema:
        {{
          "text": "<The full text of the review.>",
          "rating": "<The rating of the review. Use value 0 if not available.>"
        }}

        So the final output must have format like this

        reviews = [
            {{
                "text": "<The full text of the review.>",
                "rating": "<The rating of the review. Use value 0 if not available.>"
            }},
            {{
                "text": "<The full text of the review.>",
                "rating": "<The rating of the review. Use value 0 if not available.>"
            }},
            ...
        ]

        If a field is not available in the text, set its value to null.
        If no reviews are found on the page, return an empty list inside the "reviews" key, like this: {{"reviews": []}}.
        Do not include any non-review content like advertisements or product descriptions.

        Webpage Text:
        "{page_text}"

        JSON Output:
        """

        max_retries = 5
        last_error = None

        for _ in range(max_retries):
            try:
                # prompt = base_prompt
                if last_error:
                    prompt += f"\n\nOn the previous attempt, I got this error:\n{last_error}\n\nPlease correct the output and try again."
                    last_error = None

                response = llm.invoke(prompt)
                content = getattr(response, "content", str(response))
                # Clean markdown fences
                content = re.sub(r"```json\s*", "", content)
                content = re.sub(r"```", "", content)
                extracted_data = json.loads(content)
                break

            except (json.JSONDecodeError, Exception) as e:
                print(f"Error parsing intent, retrying: {e}")
                last_error = e

        if last_error:
            raise last_error

        # Now we expect a dictionary {"reviews": [...]}.
        
        review_list = extracted_data.get("reviews", [])

        reviews = []
        for i, item in enumerate(review_list[:max_reviews]):
            reviews.append(
                {
                    "evidence_id": f"{url}#{i}",
                    "source_name": urlparse(url).netloc,
                    "source_url": url,
                    "product_hint": None,
                    "rating": item.get("rating"),
                    "title": None,  # Not extracting title for simplicity
                    "text": item.get("text"),
                    "author": None,
                    "review_time": None,  # Could be another extraction field
                    "helpful_votes": None,
                    "verified": None,
                    "language": None,
                }
            )

        return {
            "reviews": reviews,
            "page_metadata": {
                "fetched_at": datetime.utcnow().isoformat() + "Z",
                "http_status": resp.status_code,
            },
        }
    except Exception as e:
        logger.error(f"Failed to fetch or process {url}: {e}")
        return {"reviews": [], "page_metadata": {"error": str(e)}}


@tool("create_review_snapshot")
def create_review_snapshot(
    product_key: str,
    product_query: str,
    source_urls: List[str],
    reviews: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build a snapshot from a set of URLs.
    """
    snapshot_id = f"snap_{product_key}_{int(time.time())}"
    # In a real app, we would save this to the database/blob storage here.
    # For now, we just return the ID and count.
    # The caller (graph) is responsible for persisting it to global memory.
    
    return {
        "review_snapshot_id": snapshot_id,
        "review_count": len(reviews),
        "snapshot_data": { # Return data so it can be stored in memory
            "snapshot_id": snapshot_id,
            "created_at": datetime.now().isoformat(),
            "product_key": product_key,
            "product_query": product_query,
            "source_urls": source_urls,
            "reviews": reviews
        }
    }


@tool("get_review_stats")
def get_review_stats(review_snapshot_id: str, snapshot_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    High-level stats for a review snapshot.
    """
    # In a real system, we'd load snapshot_data from DB using review_snapshot_id.
    # Here we assume it's passed in or we can't do much. 
    # For the sake of the tool interface, we'll assume the agent passes the data or we have a way to retrieve it.
    # Since we don't have a DB connected in this tool file, we rely on the agent passing it or mocking it.
    
    if not snapshot_data:
        return {"error": "Snapshot data not found"}
        
    reviews = snapshot_data.get("reviews", [])
    if not reviews:
        return {"review_count": 0}
        
    return {
        "review_snapshot_id": review_snapshot_id,
        "review_count": len(reviews),
        "avg_rating": 0, # We didn't extract ratings reliably
        "sources": list(set(r["source_name"] for r in reviews)),
        "first_review_date": None,
        "last_review_date": None
    }


@tool("search_reviews")
def search_reviews(
    review_snapshot_id: str,
    query: str,
    k: int = 20,
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    snapshot_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Search within a fetched snapshot (lexical).
    """
    if not snapshot_data:
        return {"error": "Snapshot data not found"}
        
    reviews = snapshot_data.get("reviews", [])
    results = []
    
    query_lower = query.lower()
    for r in reviews:
        text = r.get("text", "").lower()
        if query_lower in text:
            results.append(r)
            
    return {"reviews": results[:k]}


@tool("cluster_review_topics")
def cluster_review_topics(
    review_snapshot_id: str, 
    num_topics: int = 6,
    snapshot_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Cluster reviews into topics using LLM analysis of the snapshot.
    """
    if not snapshot_data:
        return {"error": "Snapshot data not found"}
        
    reviews = snapshot_data.get("reviews", [])
    if not reviews:
        return {"topics": []}
        
    # Prepare a sample of review texts
    sample_texts = []
    total_chars = 0
    
    # Shuffle to get a random sample
    shuffled = list(reviews)
    random.shuffle(shuffled)
    
    for r in shuffled:
        text = r.get("text", "")
        if len(text) > 40: 
            # Include a bit of context
            snippet = text[:300].replace("\n", " ")
            sample_texts.append(f"- {snippet}")
            total_chars += len(snippet)
            if total_chars > 6000: # Limit context
                break
    
    if not sample_texts:
        return {"topics": []}
        
    combined_text = "\n".join(sample_texts)
    
    llm = build_ollama_client()
    
    prompt = f"""
    Analyze the following product reviews and identify the top {num_topics} distinct topics or themes (e.g., "Battery Life", "Build Quality", "Price").
    For each topic, estimate the average sentiment (1.0 to 5.0) based on the reviews.
    
    Reviews:
    {combined_text}
    
    Return ONLY a JSON object with this schema:
    {{
      "topics": [
        {{
          "topic_id": 0,
          "label": "Short Label",
          "top_keywords": ["keyword1", "keyword2"],
          "avg_rating": 4.5
        }}
      ]
    }}
    """
    
    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        # Clean markdown
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```", "", content)
        data = json.loads(content)
        
        # Ensure topic_ids are correct
        for i, topic in enumerate(data.get("topics", [])):
            topic["topic_id"] = i
            
        return data
    except Exception as e:
        logger.error(f"Topic clustering failed: {e}")
        return {
            "topics": [
                {"topic_id": i, "label": f"Topic {i}", "top_keywords": [], "avg_rating": 3.0}
                for i in range(num_topics)
            ]
        }


@tool("compute_long_term_reliability")
def compute_long_term_reliability(
    review_snapshot_id: str,
    failure_keywords: Optional[List[str]] = None,
    snapshot_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Estimate failure mentions over time using snapshot reviews.
    """
    if not snapshot_data:
        return {"error": "Snapshot data not found"}
        
    reviews = snapshot_data.get("reviews", [])
    failure_keywords = failure_keywords or ["broke", "died", "stopped working", "failed", "dead"]
    
    mentions = 0
    for r in reviews:
        text = r.get("text", "").lower()
        if any(fk in text for fk in failure_keywords):
            mentions += 1
            
    return {
        "n_failure_mentions": mentions,
        "total_reviews_analyzed": len(reviews),
        "failure_rate_estimate": mentions / len(reviews) if reviews else 0
    }

def get_all_tools():
    return [
        web_search_review_pages,
        discover_products,
        fetch_and_extract_reviews,
        create_review_snapshot,
        get_review_stats,
        search_reviews,
        cluster_review_topics,
        compute_long_term_reliability
    ]
