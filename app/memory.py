from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, List

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # pragma: no cover
    SqliteSaver = None  # type: ignore


class GlobalMemory:
    """
    Persistent cache for product/tool data (keeps previous analyses).
    Stored as JSON on disk.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._state: Dict[str, Any] = {"products": {}, "users": {}}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._state = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._state = {"products": {}, "users": {}}

    def _persist(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, sort_keys=True)

    def get_product(self, product_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._state.setdefault("products", {}).setdefault(product_id, {})

    def update_product(self, product_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            product_entry = self._state.setdefault("products", {}).setdefault(
                product_id, {}
            )
            product_entry.update(data)
            self._persist()
            return product_entry

    def get_user(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._state.setdefault("users", {}).setdefault(user_id, {})

    def update_user(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            user_entry = self._state.setdefault("users", {}).setdefault(user_id, {})
            user_entry.update(data)
            self._persist()
            return user_entry

    def cache_tool_result(self, product_id: str, key: str, value: Any) -> Dict[str, Any]:
        with self._lock:
            product_entry = self.get_product(product_id)
            cache = product_entry.setdefault("tool_cache", {})
            cache[key] = value
            self._persist()
            return cache

    def get_tool_cache(self, product_id: str, key: str) -> Optional[Any]:
        with self._lock:
            product_entry = self._state.get("products", {}).get(product_id, {})
            return product_entry.get("tool_cache", {}).get(key)


def get_global_memory(path: str | Path) -> GlobalMemory:
    return GlobalMemory(Path(path))


class ChatHistory:
    """
    LangGraph-powered chat history storage so LLMs can refer to prior messages.
    Backed by SqliteSaver on disk to persist across sessions.
    """

    def __init__(self, path: Path):
        if SqliteSaver is None:
            raise RuntimeError("langgraph is required for chat history storage")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.saver = SqliteSaver.from_conn_string(str(path))
        self.namespace = "chat_history"

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        checkpoint = self.saver.get(
            {"thread_id": session_id, "checkpoint_ns": self.namespace}
        )
        if checkpoint and checkpoint.checkpoint:
            return checkpoint.checkpoint.get("messages", [])
        return []

    def save_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self.saver.put(
            {"messages": messages},
            {"thread_id": session_id, "checkpoint_ns": self.namespace},
        )

    def append_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        current = self.get_messages(session_id)
        merged = current + messages
        self.save_messages(session_id, merged)


def get_chat_history(path: str | Path) -> ChatHistory:
    return ChatHistory(Path(path))
