"""Interactive command-line RAG chatbot.

Usage:
    python chatbot.py                  # interactive chat
    python chatbot.py --show-retrieval # also print the standalone query and scores

Commands inside the chat: /reset (new conversation), /quit
"""
from __future__ import annotations

import argparse
import logging
import uuid

from app.config import get_settings
from app.rag import ChatResult, build_assistant

logging.basicConfig(level=logging.WARNING)


def format_result(result: ChatResult, show_retrieval: bool = False) -> str:
    lines = [f"Bot: {result.answer}"]
    if show_retrieval:
        r = result.retrieval
        lines.append(f"     (search query: {result.standalone_question!r})")
        lines.append(
            f"     (reranker={r.get('reranker')}: {r.get('candidates', 0)} candidates -> "
            f"{r.get('kept', 0)} kept; top rerank scores {r.get('top_rerank_scores')})"
        )
    if result.sources:
        lines.append("     Sources:")
        for s in result.sources:
            page = f", p.{s.page}" if s.page else ""
            lines.append(f"       [{s.ref}] {s.title}{page}  (rerank {s.rerank_score:.2f}, vector {s.score:.2f})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="E-commerce support assistant (CLI)")
    parser.add_argument("--show-retrieval", action="store_true")
    args = parser.parse_args()

    assistant = build_assistant(get_settings())
    session_id = str(uuid.uuid4())
    print("E-commerce Support Assistant. Ask about returns, refunds, shipping, warranties or the product manual.")
    print("Type /reset for a new conversation, /quit to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"/quit", "/exit"}:
            break
        if question.lower() == "/reset":
            assistant.reset(session_id)
            session_id = str(uuid.uuid4())
            print("(conversation reset)\n")
            continue
        print(format_result(assistant.ask(question, session_id), args.show_retrieval) + "\n")


if __name__ == "__main__":
    main()
