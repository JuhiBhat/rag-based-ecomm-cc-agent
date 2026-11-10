"""Container start-up helper: ingest documents if the Qdrant collection is empty.

Set AUTO_INGEST=false to skip, or FORCE_REINGEST=true to rebuild on every start.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.vectorstore import count_points  # noqa: E402


def wait_for_qdrant(settings, attempts: int = 30) -> None:
    for _ in range(attempts):
        try:
            count_points(settings)
            return
        except Exception:  # server still starting
            time.sleep(2)
    raise RuntimeError(f"Qdrant not reachable at {settings.qdrant_url}")


def main() -> None:
    if os.getenv("AUTO_INGEST", "true").lower() != "true":
        return
    settings = get_settings()
    if settings.qdrant_url:
        wait_for_qdrant(settings)
    force = os.getenv("FORCE_REINGEST", "false").lower() == "true"
    if force or count_points(settings) == 0:
        import ingest

        try:
            ingest.run()
        except (FileNotFoundError, ValueError) as exc:
            # Start the API anyway; /health will report that the index is missing.
            print(f"WARNING: ingestion skipped - {exc}")
    else:
        print(f"Collection '{settings.collection}' already indexed - skipping ingestion.")


if __name__ == "__main__":
    main()
