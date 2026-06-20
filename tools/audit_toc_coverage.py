from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.collector_core.html_utils import fetch_page_data


UUIDISH = re.compile(r"/[0-9a-f]{8,}[-0-9a-f]*", re.I)


def norm(url: str) -> str:
    return urldefrag(url)[0].rstrip("/")


def get_link_url(link) -> str:
    if isinstance(link, dict):
        return link.get("url") or link.get("href") or ""
    return getattr(link, "url", "") or getattr(link, "href", "") or ""


def get_link_text(link) -> str:
    if isinstance(link, dict):
        return link.get("text") or link.get("title") or ""
    return getattr(link, "text", "") or getattr(link, "title", "") or ""


def classify_link(root_url: str, url: str, text: str) -> str:
    root = urlparse(root_url)
    parsed = urlparse(url)

    if parsed.netloc != root.netloc:
        return "external"

    path = parsed.path.rstrip("/")
    lower_text = (text or "").strip().lower()

    if norm(url) == norm(root_url):
        return "self"

    if path in {"", "/"}:
        return "nav"

    if path in {"/devlog", "/about"}:
        return "nav"

    if "backend engineer" in lower_text:
        return "nav"

    if "김영찬" in text:
        return "nav"

    if UUIDISH.search(path):
        return "toc_candidate"

    return "same_domain_other"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python tools/audit_toc_coverage.py data/source_runs/<run_id>")
        return 2

    run_dir = Path(sys.argv[1])
    graph_path = run_dir / "source_graph.json"

    if not graph_path.exists():
        print("source_graph.json not found:", graph_path)
        return 1

    graph = json.loads(graph_path.read_text())
    input_url = graph.get("input_url") or ""
    visited_nodes = graph.get("nodes", [])

    visited = {}
    for n in visited_nodes:
        url = n.get("url") or ""
        if url:
            visited[norm(url)] = {
                "title": n.get("title") or "",
                "url": url,
                "text_chars": len(n.get("text") or ""),
                "depth": (n.get("meta") or {}).get("depth"),
            }

    print("== TOC COVERAGE AUDIT ==")
    print("run_dir:", run_dir)
    print("input_url:", input_url)
    print("visited_pages:", len(visited))
    print()

    page = fetch_page_data(input_url, render=True, visible=False, timeout=45)

    raw_links = []
    seen = set()

    for link in getattr(page, "links", []):
        href = get_link_url(link)
        text = " ".join(get_link_text(link).split())

        if not href:
            continue

        abs_url = norm(urljoin(input_url, href))
        if abs_url in seen:
            continue
        seen.add(abs_url)

        kind = classify_link(input_url, abs_url, text)
        raw_links.append({
            "kind": kind,
            "title": text or abs_url,
            "url": abs_url,
        })

    toc = [x for x in raw_links if x["kind"] == "toc_candidate"]
    same_domain_other = [x for x in raw_links if x["kind"] == "same_domain_other"]
    nav = [x for x in raw_links if x["kind"] == "nav"]

    toc_urls = {norm(x["url"]) for x in toc}
    collected_toc = [x for x in toc if norm(x["url"]) in visited]
    missing_toc = [x for x in toc if norm(x["url"]) not in visited]

    extra = []
    for url, meta in visited.items():
        if url == norm(input_url):
            continue
        if url not in toc_urls:
            extra.append(meta)

    coverage = (len(collected_toc) / len(toc) * 100) if toc else 0.0

    print("root_title:", getattr(page, "title", ""))
    print("root_links_total:", len(raw_links))
    print("toc_candidates:", len(toc))
    print("toc_collected:", len(collected_toc))
    print("toc_missing:", len(missing_toc))
    print("toc_coverage:", f"{coverage:.1f}%")
    print("extra_collected_outside_toc:", len(extra))
    print()

    print("== TOC CANDIDATES ==")
    for i, x in enumerate(toc, 1):
        mark = "OK" if norm(x["url"]) in visited else "MISS"
        print(f"{i:02d}. [{mark}] {x['title']}")
        print("    ", x["url"])

    print()
    print("== MISSING TOC ITEMS ==")
    for i, x in enumerate(missing_toc, 1):
        print(f"{i:02d}. {x['title']}")
        print("    ", x["url"])

    print()
    print("== EXTRA COLLECTED OUTSIDE ROOT TOC ==")
    for i, x in enumerate(extra, 1):
        print(f"{i:02d}. {x['title']} | depth={x['depth']} | chars={x['text_chars']}")
        print("    ", x["url"])

    print()
    print("== SAME DOMAIN NON-TOC LINKS FOUND ON ROOT ==")
    for i, x in enumerate(same_domain_other[:50], 1):
        print(f"{i:02d}. {x['title']}")
        print("    ", x["url"])

    print()
    print("== NAV LINKS FOUND ON ROOT ==")
    for i, x in enumerate(nav[:50], 1):
        print(f"{i:02d}. {x['title']}")
        print("    ", x["url"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
