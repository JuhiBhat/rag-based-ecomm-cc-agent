"""Download the public knowledge-base documents listed in data/sources.json.

HTML pages are converted to clean TXT (main content only); PDFs are saved as-is.
Files already present are skipped unless --force is given.

Usage:
    python scripts/download_documents.py [--force]

If a site blocks the download (some government sites reject scripted requests), open the URL
in a browser, save the page/PDF into data/documents/ using the filename shown, and re-run
ingest.py. Any other PDF/TXT you drop into that folder is ingested too.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "sources.json"
DOCS_DIR = ROOT / "data" / "documents"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def html_to_text(html: str, title: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    lines = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "dt", "dd"]):
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue
        if el.name in {"h1", "h2", "h3", "h4"}:
            lines.append(f"\n{text}\n")
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    body = "\n".join(lines).strip()
    return f"{title}\nSource: {url}\n\n{body}\n"


def download(entry: dict, force: bool) -> bool:
    target = DOCS_DIR / entry["filename"]
    if target.exists() and not force:
        print(f"skip   {target.name} (exists)")
        return True
    try:
        resp = requests.get(entry["url"], headers=HEADERS, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"FAILED {target.name}: {exc}\n       -> download manually from {entry['url']}")
        return False

    if entry.get("format") == "pdf":
        if not resp.content.startswith(b"%PDF"):
            print(f"FAILED {target.name}: response is not a PDF -> download manually from {entry['url']}")
            return False
        target.write_bytes(resp.content)
    else:
        text = html_to_text(resp.text, entry["title"], entry["url"])
        if len(text) < 600:  # app-style pages rendered by JavaScript come back nearly empty
            print(f"FAILED {target.name}: page text too short (blocked?) -> save it manually")
            return False
        target.write_text(text, encoding="utf-8")
    print(f"saved  {target.name} ({target.stat().st_size // 1024} KB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    entries = json.loads(SOURCES.read_text(encoding="utf-8"))["documents"]
    ok = sum(download(e, args.force) for e in entries)
    print(f"\n{ok}/{len(entries)} documents ready in {DOCS_DIR}")
    return 0 if ok == len(entries) else 1


if __name__ == "__main__":
    sys.exit(main())
