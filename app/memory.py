"""Per-session conversation history (in-process, thread-safe, bounded)."""
from __future__ import annotations

import threading
from collections import deque

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ConversationStore:
    """Keeps the last ``max_turns`` user/assistant exchanges for each session id.

    In-process storage is enough for a single API container. To scale out to several
    replicas, swap this class for a Redis-backed implementation with the same interface.
    """

    def __init__(self, max_turns: int = 6):
        self._max_messages = max_turns * 2
        self._sessions: dict[str, deque[BaseMessage]] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[BaseMessage]:
        with self._lock:
            return list(self._sessions.get(session_id, ()))

    def append(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            history = self._sessions.setdefault(session_id, deque(maxlen=self._max_messages))
            history.append(HumanMessage(content=question))
            history.append(AIMessage(content=answer))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
