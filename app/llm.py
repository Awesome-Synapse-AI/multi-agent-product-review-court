from __future__ import annotations

from typing import Any, Optional

from .config import get_settings

try:
    from langchain_ollama import ChatOllama
except ImportError:  # pragma: no cover - fallback for older installs
    try:
        from langchain_community.chat_models import ChatOllama  # type: ignore
    except ImportError:  # pragma: no cover - allow import when LangChain is absent
        class ChatOllama:  # type: ignore
            """Lightweight stub for environments without langchain_ollama installed."""

            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

            def bind_tools(self, tools: Any) -> "ChatOllama":
                return self

            def invoke(self, prompt: Any) -> Any:
                # Stub response; real model will return a message object
                class _Resp:
                    content = "[]"

                    def __str__(self) -> str:
                        return self.content

                return _Resp()

            def stream(self, prompt: Any):
                """Yield a single stub chunk so streaming code paths continue to work."""
                class _Resp:
                    content = "[]"

                    def __str__(self) -> str:
                        return self.content

                yield _Resp()

            def __repr__(self) -> str:
                return f"ChatOllamaStub({self.kwargs})"


def build_ollama_client(
    temperature: Optional[float] = None,
    num_ctx: Optional[int] = None,
    num_predict: Optional[int] = None,
) -> Any:
    """Return a configured ChatOllama client with optional overrides."""
    settings = get_settings()
    model_kwargs = {
        "model": settings.ollama_model,
        "temperature": (
            temperature if temperature is not None else settings.ollama_temperature
        ),
        "base_url": settings.ollama_base_url,
    }

    ctx = settings.ollama_num_ctx if num_ctx is None else num_ctx
    if ctx is not None:
        model_kwargs["num_ctx"] = ctx
    if num_predict is not None:
        model_kwargs["num_predict"] = num_predict

    return ChatOllama(**model_kwargs)
