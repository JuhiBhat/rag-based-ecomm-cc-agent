"""Run scripted multi-turn conversations through the real pipeline and save a log.

Produces logs/sample_conversation.md (human-readable) and logs/sample_conversation.json.
Requires OPENAI_API_KEY and an ingested collection (python ingest.py).

Usage:
    python scripts/run_sample_conversation.py
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, get_settings  # noqa: E402
from app.rag import build_assistant  # noqa: E402

CONVERSATIONS = {
    "Meesho returns (follow-ups)": [
        "What is Meesho's return and exchange policy?",
        "What about eligibility - are there conditions the product must meet?",
        "Does this apply to discounted items?",
        "Who pays the return shipping fee?",
    ],
    "Meesho seller side": [
        "What happens if a seller does not dispatch a Meesho order on time?",
        "Is there a penalty if the seller cancels the order?",
    ],
    "E-commerce returns and grievances in India": [
        "What return and refund information must a marketplace show to consumers?",
        "What about complaints - how fast must they be handled?",
        "Does this apply to someone selling their old phone in a personal capacity?",
    ],
    "Consumer rights under Indian law": [
        "What rights does a consumer have under the Consumer Protection Act, 2019?",
        "Where can I file a complaint if a defective product is not refunded?",
    ],
    "Product manual and warranty": [
        "How often should I replace the filters in the Havells Siphon water purifier?",
        "What does the Bajaj mixer grinder warranty cover?",
        "Does it cover the blades?",
    ],
    "Out-of-scope questions (should refuse)": [
        "What is Flipkart's return window for mobile phones?",
        "Who won the cricket world cup?",
    ],
}


def main() -> None:
    assistant = build_assistant(get_settings())
    out_dir = PROJECT_ROOT / "logs"
    out_dir.mkdir(exist_ok=True)
    records, md = [], [f"# Sample conversations\n\n_Generated {datetime.now():%Y-%m-%d %H:%M}_\n"]

    for title, turns in CONVERSATIONS.items():
        session_id = str(uuid.uuid4())
        md.append(f"\n## {title}\n")
        for question in turns:
            result = assistant.ask(question, session_id)
            records.append({"conversation": title, **result.to_dict()})
            md.append(f"**User:** {question}\n")
            if result.standalone_question != question:
                md.append(f"> _rewritten search query:_ {result.standalone_question}\n")
            r = result.retrieval
            md.append(
                f"> _retrieval:_ {r.get('candidates', 0)} candidates → reranked ({r.get('reranker')}) "
                f"→ {r.get('kept', 0)} kept\n"
            )
            md.append(f"**Bot:** {result.answer}\n")
            for s in result.sources:
                page = f", p.{s.page}" if s.page else ""
                md.append(f"- [{s.ref}] {s.title}{page} (rerank {s.rerank_score:.2f}, vector {s.score:.2f})")
            md.append("")
            print(f"Q: {question}\nA: {result.answer}\n")

    (out_dir / "sample_conversation.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "sample_conversation.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved logs to {out_dir}")


if __name__ == "__main__":
    main()
