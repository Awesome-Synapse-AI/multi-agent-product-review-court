from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.memory import get_global_memory

try:
    from langgraph.graph import StateGraph, START, END
except ImportError:
    StateGraph = None
    START = "START"
    END = "END"

from app.agents.complaints import COMPLAINTS_SYSTEM_PROMPT
from app.agents.fanboy import FANBOY_SYSTEM_PROMPT
from app.agents.judge import JUDGE_SYSTEM_PROMPT
    

from app.agents.complaints import build_complaints_agent
from app.agents.fanboy import build_fanboy_agent
from app.agents.judge import build_judge_agent
from app.config import get_settings
from app.llm import build_ollama_client
from app.routing import build_followup_response, build_greeting_response, route_message
from app.parsing import parse_intent, IntentOutput, ProductCandidate, RequestConstraints
from app.tools.review_tools import (
    web_search_review_pages,
    discover_products,
    fetch_and_extract_reviews,
    create_review_snapshot,
    get_review_stats,
    search_reviews,
    cluster_review_topics,
    compute_long_term_reliability,
)

logger = logging.getLogger(__name__)

# --- State Definitions ---

class ProductState(TypedDict):
    product_query: str
    product_key: str
    brand_hint: Optional[str]
    review_snapshot_id: Optional[str]
    evidence_sources: List[Dict[str, str]]
    complaints_case: Optional[Dict[str, Any]]
    fanboy_case: Optional[Dict[str, Any]]
    snapshot_data: Optional[Dict[str, Any]]

class GraphState(TypedDict):
    message: str
    mode: str
    request_constraints: Dict[str, Any]
    products: List[ProductState]
    judge_output: Dict[str, Any]
    final_response_text: str
    final_response_tokens: List[str]
    stream_callback: Optional[Any]
    progress_callback: Optional[Any]

# --- Helpers ---

def _call_tool(tool: Any, kwargs: Dict[str, Any]) -> Any:
    if hasattr(tool, "invoke"):
        return tool.invoke(kwargs)
    if hasattr(tool, "run"):
        return tool.run(**kwargs)
    if callable(tool):
        return tool(**kwargs)
    raise TypeError(f"Tool is not callable: {tool}")


def _strip_code_fences(text: str) -> str:
    cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def _tokenize_for_stream(text: str) -> List[str]:
    # Preserve whitespace by keeping trailing spaces with each token
    return re.findall(r"\S+\s*", text) if text else []

def _strip_disallowed_lines(text: str, mode: str) -> str:
    """
    Remove lines starting with 'Verdict' for all modes, and 'Winner' when in single mode.
    This keeps the Judge output focused on the rationale without redundant headers.
    """
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        if re.match(r"\s*verdict\b", line, re.IGNORECASE):
            continue
        if mode == "single" and re.match(r"\s*winner\b", line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def _emit_progress(state: Dict[str, Any], node_name: str) -> None:
    cb = state.get("progress_callback")
    if callable(cb):
        try:
            cb(node_name)
        except Exception:
            logger.debug("Progress callback failed for node=%s", node_name, exc_info=True)


def _execute_agent_loop(
    agent_runnable,
    input_text: str,
    tools_map: Dict[str, Any],
    model: Any,
    tool_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a ReAct-style loop for an agent.
    """
    
    system_prompt = ""
    
    if "Complaints" in str(agent_runnable): 
        system_prompt = COMPLAINTS_SYSTEM_PROMPT
    elif "Fanboy" in str(agent_runnable): 
        system_prompt = FANBOY_SYSTEM_PROMPT
    elif "Judge" in str(agent_runnable): 
        system_prompt = JUDGE_SYSTEM_PROMPT
    
    logger.info("Agent start name=%s input_len=%s tools=%s", agent_runnable, len(input_text), len(tools_map))
    tools_enabled = bool(tools_map)
    if tools_enabled:
        try:
            bound_model = model.bind_tools(list(tools_map.values()))
        except NotImplementedError:
            tools_enabled = False
            bound_model = model
            system_prompt = (
                system_prompt
                + "\n\nTool calling is unavailable. Use only the provided input and return JSON."
            )
            logger.warning("Tools not supported; running tool-less mode agent=%s", agent_runnable)
    else:
        bound_model = model
        system_prompt = (
            system_prompt
            + "\n\nTool calling is unavailable. Use only the provided input and return JSON."
        )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=input_text)]
    
    for turn in range(5): # Max turns
        logger.info("Agent turn=%s name=%s tools_enabled=%s", turn, agent_runnable, tools_enabled)
        try:
            response = bound_model.invoke(messages)
        except Exception as exc:
            logger.exception("Agent invoke failed agent=%s", agent_runnable)
            return {"error": f"LLM invoke failed: {exc}"}
        messages.append(response)
        
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            additional_kwargs = getattr(response, "additional_kwargs", None) or {}
            tool_calls = additional_kwargs.get("tool_calls")
        if not tool_calls:
            content = getattr(response, "content", None)
            if content is None:
                content = str(response)
            try:
                if isinstance(content, str):
                    content = _strip_code_fences(content)
                logger.info("Agent completed name=%s response_len=%s", agent_runnable, len(content or ""))
                return json.loads(content)
            except Exception:
                logger.info("Agent completed name=%s response_is_raw=True", agent_runnable)
                return {"raw_output": content}
        
        logger.info("Agent tool_calls name=%s count=%s", agent_runnable, len(tool_calls))
        for tool_call in tool_calls:
            tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
            tool_args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", None)
            tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {}
            if tool_args is None or not isinstance(tool_args, dict):
                tool_args = {}
            tool_func = tools_map.get(tool_name)
            
            if tool_func:
                try:
                    call_args = dict(tool_context or {})
                    call_args.update(tool_args)
                    logger.info("Agent tool_invoke name=%s tool=%s", agent_runnable, tool_name)
                    tool_output = _call_tool(tool_func, call_args)
                except Exception as e:
                    logger.exception("Tool failed agent=%s tool=%s turn=%s", agent_runnable, tool_name, turn)
                    tool_output = str(e)
            else:
                logger.warning("Agent tool_missing name=%s tool=%s", agent_runnable, tool_name)
                tool_output = "Tool not found"
                
            messages.append(
                ToolMessage(
                    content=json.dumps(tool_output, default=str),
                    tool_call_id=tool_call_id or ""
                )
            )

    return {"error": "Max turns reached"}

# --- Nodes ---

def parse_intent_node(state: GraphState) -> GraphState:
    _emit_progress(state, "ParseIntent")
    logger.info("Node: ParseIntent")
    intent: IntentOutput = parse_intent(state["message"])
    logger.info("ParseIntent mode=%s products=%s", intent.mode, len(intent.product_candidates))
    
    products = []
    for cand in intent.product_candidates:
        products.append({
            "product_query": cand.product_query,
            "product_key": cand.product_key,
            "brand_hint": cand.brand_hint,
            "review_snapshot_id": None,
            "evidence_sources": [],
            "complaints_case": None,
            "fanboy_case": None,
            "snapshot_data": None
        })
        
    return {
        "mode": intent.mode,
        "request_constraints": intent.request_constraints.dict(),
        "products": products
    }

def product_discovery_node(state: GraphState) -> GraphState:
    _emit_progress(state, "ProductDiscovery")
    logger.info("Node: ProductDiscovery")
    if state["mode"] != "discover_then_compare":
        logger.info("ProductDiscovery skipped mode=%s", state["mode"])
        return state
        
    new_products = []
    for prod in state["products"]:
        res = _call_tool(
            discover_products,
            {
                "category_query": prod["product_query"],
                "brands": [prod.get("brand_hint") or prod["product_key"]],
                "spec_filters": state.get("request_constraints", {}).get("spec_filters"),
            },
        )
        for cand in res.get("candidates", []):
            new_products.append({
                "product_query": cand["product_query"],
                "product_key": cand["product_query"].lower(),
                "brand_hint": cand.get("brand"),
                "review_snapshot_id": None,
                "evidence_sources": [{"source_name": "discovery", "url": cand["source_url"]}],
                "complaints_case": None,
                "fanboy_case": None,
                "snapshot_data": None
            })
            
    if new_products:
        return {"products": new_products}
    return state

def evidence_collector_node(state: GraphState) -> GraphState:
    _emit_progress(state, "EvidenceCollector")
    logger.info("Node: EvidenceCollector")
    memory = get_global_memory("data/global_memory.json")
    updated_products = []
    
    for prod in state["products"]:
        # Check memory first
        cached_prod = memory.get_product(prod["product_key"])
        cached_snapshot_id = cached_prod.get("latest_review_snapshot_id")
        if cached_snapshot_id:
            logger.info("EvidenceCollector cache_hit product_key=%s", prod["product_key"])
            # Load from memory if fresh enough (omitted freshness check for brevity)
            prod["review_snapshot_id"] = cached_snapshot_id
            # We would need to load the snapshot data too if we want to use it.
            # Assuming memory stores it or we can fetch it.
            # For this implementation, let's assume we re-fetch if not in state, 
            # or we just use the ID and hope tools can handle it (they can't without data).
            # So we should probably store the snapshot data in memory too.
            snapshots = cached_prod.get("snapshots", {})
            cached_snapshot = snapshots.get(cached_snapshot_id)
            if cached_snapshot:
                prod["snapshot_data"] = cached_snapshot
                updated_products.append(prod)
                continue
            # Snapshot data missing: clear ID so we re-fetch below.
            prod["review_snapshot_id"] = None
            prod["snapshot_data"] = None

        if prod["review_snapshot_id"]:
            updated_products.append(prod)
            continue
            
        # 1. Search
        logger.info("EvidenceCollector search_start product_key=%s", prod["product_key"])
        search_res = _call_tool(
            web_search_review_pages,
            {"product_query": prod["product_query"], "k": 5},
        )
        results = search_res.get("results", [])
        urls = [r["url"] for r in results if r.get("url")]
        logger.info("EvidenceCollector found_urls=%s", len(urls))
        
        # 2. Fetch & Extract
        reviews = []
        for url in urls[:3]: # Limit to 3 for speed
            extract_res = _call_tool(fetch_and_extract_reviews, {"url": url})
            reviews.extend(extract_res.get("reviews", []))
            
        # 3. Snapshot
        logger.info("EvidenceCollector snapshot_start product_key=%s reviews=%s", prod["product_key"], len(reviews))
        snapshot_res = _call_tool(
            create_review_snapshot,
            {
                "product_key": prod["product_key"],
                "product_query": prod["product_query"],
                "source_urls": urls,
                "reviews": reviews,
            },
        )
        
        prod["review_snapshot_id"] = snapshot_res["review_snapshot_id"]
        prod["snapshot_data"] = snapshot_res.get("snapshot_data")
        prod["evidence_sources"] = [
            {"source_name": r.get("source_name"), "url": r.get("url")}
            for r in results
            if r.get("url")
        ]
        
        # Persist to memory
        memory.update_product(prod["product_key"], {
            "latest_review_snapshot_id": prod["review_snapshot_id"],
            "snapshots": {
                prod["review_snapshot_id"]: prod["snapshot_data"]
            }
        })
        
        updated_products.append(prod)
        
    return {"products": updated_products}

def complaints_node(state: GraphState) -> GraphState:
    _emit_progress(state, "Complaints")
    logger.info("Node: Complaints")
    memory = get_global_memory("data/global_memory.json")
    model = build_ollama_client()
    tools_map = {
        "get_review_stats": get_review_stats,
        "search_reviews": search_reviews,
        "cluster_review_topics": cluster_review_topics,
        "compute_long_term_reliability": compute_long_term_reliability
    }
    
    updated_products = []
    for prod in state["products"]:
        snapshot_data = prod.get("snapshot_data")

        input_data = {
            "product_query": prod["product_query"],
            "product_key": prod["product_key"],
            "review_snapshot_id": prod["review_snapshot_id"],
            "evidence_sources": prod["evidence_sources"]
        }
        
        output = _execute_agent_loop(
            "Complaints",
            json.dumps(input_data),
            tools_map,
            model,
            tool_context={"snapshot_data": snapshot_data},
        )
        prod["complaints_case"] = output
        
        # Persist
        current = memory.get_product(prod["product_key"])
        last_outputs = current.get("last_outputs", {})
        last_outputs["complaints_case"] = output
        memory.update_product(prod["product_key"], {"last_outputs": last_outputs})
        
        updated_products.append(prod)
        
    return {"products": updated_products}

def fanboy_node(state: GraphState) -> GraphState:
    _emit_progress(state, "Fanboy")
    logger.info("Node: Fanboy")
    memory = get_global_memory("data/global_memory.json")
    model = build_ollama_client()
    tools_map = {
        "get_review_stats": get_review_stats,
        "search_reviews": search_reviews,
        "cluster_review_topics": cluster_review_topics
    }
    
    updated_products = []
    for prod in state["products"]:
        snapshot_data = prod.get("snapshot_data")

        input_data = {
            "product_query": prod["product_query"],
            "product_key": prod["product_key"],
            "review_snapshot_id": prod["review_snapshot_id"],
            "evidence_sources": prod["evidence_sources"]
        }
        
        output = _execute_agent_loop(
            "Fanboy",
            json.dumps(input_data),
            tools_map,
            model,
            tool_context={"snapshot_data": snapshot_data},
        )
        prod["fanboy_case"] = output
        
        # Persist
        # Note: we need to merge with existing last_outputs if possible, but update_product merges top level keys.
        # We should probably read-modify-write for deep keys or just overwrite "last_outputs" if we have both.
        # Here we might overwrite complaints if we are not careful.
        # Better to have separate keys or update carefully.
        # memory.update_product merges dicts at top level.
        # Let's just use separate keys for simplicity in memory structure or fetch first.
        current = memory.get_product(prod["product_key"])
        last_outputs = current.get("last_outputs", {})
        last_outputs["fanboy_case"] = output
        memory.update_product(prod["product_key"], {"last_outputs": last_outputs})
        
        updated_products.append(prod)
        
    return {"products": updated_products}

def judge_node(state: GraphState) -> GraphState:
    _emit_progress(state, "Judge")
    logger.info("Node: Judge")
    settings = get_settings()
    model = build_ollama_client(
        temperature=settings.judge_temperature,
        num_predict=settings.judge_max_output_tokens,
    )
    stream_callback = state.get("stream_callback")
    mode = state.get("mode", "single")

    input_data = {
        "mode": mode,
        "products": state["products"],
    }
    product_labels = [
        p.get("product_query") or p.get("product_key")
        for p in state.get("products", [])
        if p.get("product_query") or p.get("product_key")
    ]

    prompt = (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        f"Mode: {mode}\n"
        f"Products: {', '.join(product_labels) or 'N/A'}\n"
        f"Input JSON:\n{json.dumps(input_data, indent=2, default=str)}\n\n"
    )

    final_text = ""
    tokens: List[str] = []
    streamed_len = 0
    raw_text = ""
    try:
        if hasattr(model, "stream"):
            logger.info("Judge streaming response start")
            for chunk in model.stream(prompt):
                chunk_content = getattr(chunk, "content", None) or getattr(chunk, "text", None)
                if chunk_content is None:
                    chunk_content = str(chunk) if chunk is not None else ""
                elif not isinstance(chunk_content, str):
                    chunk_content = str(chunk_content)
                if raw_text and chunk_content.startswith(raw_text):
                    # LLM may send cumulative content; keep the latest cumulative.
                    raw_text = chunk_content
                else:
                    raw_text += chunk_content

                filtered_text = _strip_disallowed_lines(raw_text, mode)
                final_text = filtered_text
                new_delta = filtered_text[streamed_len:]
                if new_delta:
                    tokens.append(new_delta)
                    if callable(stream_callback):
                        stream_callback(new_delta)
                    streamed_len = len(filtered_text)
        else:
            logger.info("Judge streaming unavailable; using invoke()")
            resp = model.invoke(prompt)
            content = getattr(resp, "content", str(resp)) or ""
            final_text = _strip_code_fences(content.strip())
            final_text = _strip_disallowed_lines(final_text, mode)
            if final_text:
                tokens = [final_text]
                if callable(stream_callback):
                    stream_callback(final_text)
    except Exception as exc:
        logger.exception("Judge LLM failed")
        final_text = f"Unable to generate final response: {exc}"
        final_text = _strip_disallowed_lines(final_text, mode)
        if final_text:
            tokens = [final_text]
            if callable(stream_callback):
                stream_callback(final_text)

    if final_text:
        final_text = _strip_code_fences(final_text).strip()
    if not tokens and final_text:
        tokens = [final_text]

    return {
        "judge_output": {"message": final_text} if final_text else {},
        "final_response_text": final_text,
        "final_response_tokens": tokens,
    }

def synthesizer_node(state: GraphState) -> GraphState:
    _emit_progress(state, "Synthesizer")
    logger.info("Node: Synthesizer")
    judge_output = state.get("judge_output") or {}
    if isinstance(judge_output, str):
        try:
            judge_output = json.loads(judge_output)
        except Exception:
            logger.warning("Synthesizer could not parse judge_output string; using raw text")
            judge_output = {"raw_output": judge_output}

    verdict = judge_output.get("verdict") or {}
    stream_callback = state.get("stream_callback")

    if not verdict:
        final_text = "No verdict available. Please try again."
        tokens = _tokenize_for_stream(final_text)
        return {"final_response_text": final_text, "final_response_tokens": tokens}

    mode = verdict.get("mode") or state.get("mode", "single")
    product_labels = [p.get("product_query") or p.get("product_key") for p in state.get("products", [])]
    product_labels = [p for p in product_labels if p]

    prompt = (
        "You are the Final Synthesizer for Product Review Court.\n"
        "Write a concise, user-facing response that reflects ONLY the judge's verdict.\n"
        "- Include a headline verdict, a short rationale, and bullet Pros/Cons if available.\n"
        "- Keep it factual and avoid new claims beyond the verdict.\n"
        "- Use plain text (no JSON, no markdown tables).\n\n"
        f"Mode: {mode}\n"
        f"Products: {', '.join(product_labels) or 'N/A'}\n"
        f"Judge Verdict JSON:\n{json.dumps(verdict, indent=2, default=str)}\n\n"
        "Return the final response text only."
    )

    final_text = ""
    tokens: List[str] = []
    try:
        model = build_ollama_client()
        if hasattr(model, "stream"):
            logger.info("Synthesizer streaming response start")
            for chunk in model.stream(prompt):
                chunk_content = getattr(chunk, "content", None) or getattr(chunk, "text", None)
                if chunk_content is None:
                    chunk_content = str(chunk) if chunk is not None else ""
                # Stream raw chunk content to preserve spacing/newlines as produced by the model.
                final_text += chunk_content
                if chunk_content:
                    tokens.append(chunk_content)
                    if callable(stream_callback):
                        stream_callback(chunk_content)
        else:
            logger.info("Synthesizer streaming unavailable; using invoke()")
            resp = model.invoke(prompt)
            content = getattr(resp, "content", str(resp)) or ""
            final_text = _strip_code_fences(content.strip())
            tokens = _tokenize_for_stream(final_text)
            if callable(stream_callback) and final_text:
                stream_callback(final_text)
    except Exception as exc:
        logger.exception("Synthesizer LLM failed")
        final_text = f"Unable to generate final response: {exc}"
        tokens = _tokenize_for_stream(final_text)
        if callable(stream_callback) and final_text:
            stream_callback(final_text)
        
    if final_text:
        final_text = _strip_code_fences(final_text).strip()
    if not tokens and final_text:
        tokens = _tokenize_for_stream(final_text)
        
    return {
        "final_response_text": final_text,
        "final_response_tokens": tokens,
        "judge_output": judge_output,
    }

# --- Graph Construction ---

def build_graph():
    if StateGraph is None:
        return None
        
    workflow = StateGraph(GraphState)
    
    workflow.add_node("ParseIntent", parse_intent_node)
    workflow.add_node("ProductDiscovery", product_discovery_node)
    workflow.add_node("EvidenceCollector", evidence_collector_node)
    workflow.add_node("Complaints", complaints_node)
    workflow.add_node("Fanboy", fanboy_node)
    workflow.add_node("Judge", judge_node)
    
    workflow.add_edge(START, "ParseIntent")
    
    def route_discovery(state):
        if state["mode"] == "discover_then_compare":
            return "ProductDiscovery"
        return "EvidenceCollector"
        
    workflow.add_conditional_edges("ParseIntent", route_discovery)
    workflow.add_edge("ProductDiscovery", "EvidenceCollector")
    
    # Ensure Judge waits for both Complaints and Fanboy by running sequentially.
    workflow.add_edge("EvidenceCollector", "Complaints")
    workflow.add_edge("Complaints", "Fanboy")
    workflow.add_edge("Fanboy", "Judge")
    workflow.add_edge("Judge", END)
    
    return workflow.compile()

def _build_last_result(result: Dict[str, Any]) -> Dict[str, Any]:
    products_brief = []
    for prod in result.get("products", []) or []:
        products_brief.append(
            {
                "product_key": prod.get("product_key"),
                "product_query": prod.get("product_query"),
                "brand_hint": prod.get("brand_hint"),
            }
        )
    return {
        "mode": result.get("mode"),
        "products": products_brief,
        "judge_output": result.get("judge_output", {}),
        "final_response_text": result.get("final_response_text", ""),
        "final_response_tokens": result.get("final_response_tokens", []),
    }


def run_workflow(
    message: str,
    session_id: Optional[str] = None,
    stream_callback: Optional[Any] = None,
    progress_callback: Optional[Any] = None,
    callbacks: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    settings = get_settings()
    memory = get_global_memory(settings.memory_path)
    user_id = session_id or "default"
    user_state = memory.get_user(user_id)
    last_result = user_state.get("last_result")
    logger.info("run_workflow start session_id=%s message_len=%s has_last_result=%s", session_id, len(message or ""), bool(last_result))

    route = route_message(message, has_previous=bool(last_result))
    logger.info("run_workflow route=%s", route)
    if route == "greeting":
        response = {
            "mode": "chat",
            "products": [],
            "judge_output": build_greeting_response(message),
        }
        memory.update_user(user_id, {"last_route": route})
        logger.info("run_workflow completed route=greeting")
        return response

    if route == "followup":
        response = {
            "mode": "followup",
            "products": [],
            "judge_output": build_followup_response(message, last_result),
        }
        memory.update_user(user_id, {"last_route": route})
        logger.info("run_workflow completed route=followup")
        return response

    app = build_graph()
    if not app:
        logger.error("run_workflow build_graph_failed")
        return {"error": "LangGraph not available"}

    initial_state = {
        "message": message,
        "mode": "single",  # default
        "request_constraints": {},
        "products": [],
        "judge_output": {},
        "final_response_text": "",
        "final_response_tokens": [],
        "stream_callback": stream_callback,
        "progress_callback": progress_callback,
    }

    logger.info("run_workflow graph_invoke_start")
    invoke_kwargs = {}
    if callbacks:
        invoke_kwargs["config"] = {"callbacks": callbacks}
    result = app.invoke(initial_state, **invoke_kwargs)
    logger.info("run_workflow graph_invoke_complete keys=%s", list(result.keys()))

    if "error" not in result:
        memory.update_user(
            user_id,
            {"last_result": _build_last_result(result), "last_route": route},
        )
        logger.info("run_workflow memory_updated")
    return result
