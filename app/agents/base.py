from __future__ import annotations

from typing import Any, Callable, List

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError:  # pragma: no cover - minimal shims to keep module importable
    class Runnable:  # type: ignore
        def __init__(self, fn=None):
            self.fn = fn

        def __or__(self, other):
            return self

        def invoke(self, inp):
            if callable(self.fn):
                return self.fn(inp)
            return None

    class RunnablePassthrough(Runnable):  # type: ignore
        pass

    class ChatPromptTemplate:  # type: ignore
        @classmethod
        def from_messages(cls, messages):
            return cls()

        def __or__(self, other):
            return self

    class StrOutputParser(Runnable):  # type: ignore
        def __or__(self, other):
            return self


def build_tool_agent(system_prompt: str, tools: List[Callable]) -> Runnable:
    """
    Build a simple LangChain runnable that wraps a system prompt and tools.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    def _bind_model(model: Any) -> Runnable:
        # Bind tools to the model
        bound = model.bind_tools(tools)
        return prompt | bound

    # Returning a factory runnable that expects {"input": "...", "model": model}
    return (
        {
            "input": RunnablePassthrough(),
            "model": lambda x: x["model"],
        }
        | (lambda d: _bind_model(d["model"]).invoke({"input": d["input"]}))
    )
