"""ConversationManager — sliding / summarize / hybrid strategies."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from membrane.compiler import CompileResult, ContextCompiler
from membrane.conversation.summarizer import summarize_turns
from membrane.conversation.window import sliding_turns


class ConversationManager:
    """
    Maintains turn history and compiles with :class:`ContextCompiler`.

    ``window_strategy``:
    - ``sliding``: keep last ``max_turns`` only
    - ``summarize``: when len(turns) > ``summary_trigger``, compress older turns
    - ``hybrid``: sliding recent + summary block for older
    """

    def __init__(
        self,
        compiler: ContextCompiler,
        *,
        max_turns: int = 50,
        summarization_fn: Optional[Callable[[str], str]] = None,
        window_strategy: Literal["sliding", "summarize", "hybrid"] = "hybrid",
        summary_trigger: int = 10,
        topic_aware: bool = True,
    ) -> None:
        self._compiler = compiler
        self._max_turns = max_turns
        self._summarization_fn = summarization_fn
        self._window_strategy = window_strategy
        self._summary_trigger = summary_trigger
        self._topic_aware = topic_aware
        self._turns: List[Tuple[str, str]] = []
        self._summary: Optional[str] = None

    def add_turn(self, role: str, content: str) -> None:
        self._turns.append((role, content))
        self._maybe_summarize()

    def _maybe_summarize(self) -> None:
        if self._window_strategy == "sliding":
            self._turns = sliding_turns(self._turns, max_turns=self._max_turns)
            return
        if len(self._turns) <= self._summary_trigger:
            return
        if self._window_strategy in ("summarize", "hybrid"):
            old = self._turns[:- self._max_turns // 2] if self._window_strategy == "hybrid" else self._turns[:-3]
            recent = self._turns[-3:] if self._window_strategy == "hybrid" else []
            self._summary = summarize_turns(old, self._summarization_fn)
            self._turns = recent if self._window_strategy == "hybrid" else self._turns[-self._max_turns :]

    def get_summary(self) -> Optional[str]:
        return self._summary

    def compile(
        self,
        *,
        memories: Sequence[Any],
        user_message: str,
        **kwargs: Any,
    ) -> CompileResult:
        """Compile with optional ``conversation_messages`` built from history + summary."""
        conv: List[Dict[str, str]] = []
        if self._summary:
            conv.append(
                {
                    "role": "system",
                    "content": f"--- Conversation summary ---\n{self._summary}",
                }
            )
        for role, content in self._turns:
            conv.append({"role": role, "content": content})
        merged = dict(kwargs)
        merged.setdefault("conversation_messages", conv)
        return self._compiler.compile(
            memories=memories,
            user_message=user_message,
            **merged,
        )

    def serialize(self) -> Dict[str, Any]:
        return {
            "turns": self._turns,
            "summary": self._summary,
            "strategy": self._window_strategy,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any], compiler: ContextCompiler) -> ConversationManager:
        m = cls(
            compiler,
            window_strategy=data.get("strategy", "hybrid"),
        )
        m._turns = [(t[0], t[1]) for t in data.get("turns", [])]
        m._summary = data.get("summary")
        return m
