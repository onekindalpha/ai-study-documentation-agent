from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from tools.collector_core.html_utils import fetch_page_data, same_domain
from tools.collector_core.schema import make_graph, make_node


def collect(url: str, *, run_dir, trace, plan, max_pages: int = 20, visible: bool = False, max_depth: int = 1, timeout: int = 30, **kwargs) -> dict[str, Any]:
    book = fetch_page_data(url, timeout=timeout, render=False)
    trace.event("wikidocs_book_page_fetched", title=book.title, links=len(book.links), text_chars=len(book.text))
    chapter_links = []
    seen = set()
    for link in book.links:
        href = link.get("url") or ""
        path = urlparse(href).path.strip("/")
        if not same_domain(href, url) or href in seen:
            continue
        if path.isdigit():
            seen.add(href)
            chapter_links.append({"url": href, "title": link.get("text") or f"Chapter {len(chapter_links)+1}"})
    chapter_links = chapter_links[:max_pages]
    trace.event("wikidocs_chapter_links_extracted", chapter_count=len(chapter_links))

    graph = make_graph(
        input_url=url,
        url_type="wikidocs",
        site_hint="wikidocs",
        content_shape=plan.content_shape,
        navigation_shape=plan.navigation_shape,
        access_level=plan.access_level,
        evidence_targets=plan.evidence_targets,
        title=book.title or "WikiDocs book",
    )
    book_node = make_node(node_type="book", title=book.title or "WikiDocs book", url=url, order=1, text=book.text[:12000])
    graph["quality"]["pages_collected"] = 1
    children = []
    for idx, item in enumerate(chapter_links, start=1):
        try:
            page = fetch_page_data(item["url"], timeout=timeout, render=False)
            trace.event("wikidocs_chapter_collected", order=idx, title=page.title or item["title"], url=item["url"], text_chars=len(page.text), code_blocks=len(page.code_blocks))
            evidence = [{"type": "heading", "text": h} for h in page.headings[:20]]
            evidence += [{"type": "code", "text": c[:2000]} for c in page.code_blocks[:15]]
            children.append(make_node(node_type="chapter", title=page.title or item["title"], url=item["url"], order=idx, text=page.text[:16000], evidence=evidence))
            graph["quality"]["pages_collected"] += 1
        except Exception as exc:
            trace.warning("wikidocs_chapter_failed", url=item["url"], error=str(exc))
            graph["quality"]["missing"].append(f"failed chapter: {item['url']}")
    book_node["children"] = children
    graph["nodes"].append(book_node)
    return graph
