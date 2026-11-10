"""Append-only JSONL logs for answered questions and user feedback.

Every answer gets a ``message_id``; feedback refers to it, so each thumbs-up/down can be
joined back to the question, the answer and the sources that produced it.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with _lock, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_interaction(path: Path, record: dict) -> None:
    _append(path, record)


def save_feedback(
    path: Path, *, session_id: str, message_id: str, rating: str, comment: str | None = None
) -> dict:
    if rating not in {"up", "down"}:
        raise ValueError("rating must be 'up' or 'down'")
    record = {
        "session_id": session_id,
        "message_id": message_id,
        "rating": rating,
        "comment": (comment or "").strip(),
    }
    _append(path, record)
    return record


def feedback_summary(path: Path) -> dict:
    up = down = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rating = json.loads(line).get("rating")
            up += rating == "up"
            down += rating == "down"
    return {"up": up, "down": down, "total": up + down}
