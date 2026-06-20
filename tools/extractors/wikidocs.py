from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin, urlparse

from tools.collector_core.html_utils import fetch_html, fetch_page_data, same_domain
from tools.collector_core.schema import make_graph, make_node


def is_probable_book_chapter(link_text: str, book_title: str) -> bool:
    text = f"{link_text}\n{book_title}".lower()
    if any(bad in text for bad in ["이벤트", "증정", "출판사", "광고", "최근 변경", "댓글", "문의"]):
        return False
    chapter_markers = [
        "문제",
        "마당",
        "장 ",
        "chapter",
        "코딩 테스트",
        "파이썬",
        "자료구조",
        "스택",
        "큐",
        "해시",
        "트리",
        "그래프",
        "정렬",
        "탐색",
    ]
    return any(marker in text for marker in chapter_markers)


def extract_book_toc_links(book_url: str, raw_html: str) -> list[dict[str, Any]]:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return []

    soup = BeautifulSoup(raw_html, "html.parser")
    toc_root = soup.select_one(".toc") or soup.select_one(".list-group")
    if not toc_root:
        return []

    links: list[dict[str, Any]] = []
    seen: set[str] = set()
    for order, a in enumerate(toc_root.find_all("a"), start=1):
        text = re.sub(r"\s+", " ", a.get_text(" ", strip=True)).strip()
        href = str(a.get("href") or "").strip()
        if not text or not href:
            continue
        if href.startswith("/book/"):
            continue
        page_id = ""
        match = re.search(r"page\((\d+)\)", href)
        if match:
            page_id = match.group(1)
        elif re.fullmatch(r"/?\d+", href):
            page_id = href.strip("/")
        if not page_id:
            continue
        page_url = urljoin(book_url, f"/{page_id}")
        if page_url in seen:
            continue
        seen.add(page_url)
        depth = 0
        parent = a.find_parent(["li", "div"])
        class_text = " ".join(parent.get("class", [])) if parent else ""
        depth_match = re.search(r"(?:level|depth|indent)-?(\d+)", class_text)
        if depth_match:
            depth = int(depth_match.group(1))
        elif re.match(r"^\d+\s+", text):
            depth = 0
        elif re.match(r"^\d{2}-\d+", text):
            depth = 2
        elif re.match(r"^\d{2}\s+", text):
            depth = 1
        elif text.startswith("문제"):
            depth = 3
        links.append({"url": page_url, "title": text, "page_id": page_id, "order": order, "depth": depth})
    return links


def collect(url: str, *, run_dir, trace, plan, max_pages: int = 20, visible: bool = False, max_depth: int = 1, timeout: int = 30, **kwargs) -> dict[str, Any]:
    raw_html = fetch_html(url, timeout=timeout)
    book = fetch_page_data(url, timeout=timeout, render=False)
    trace.event("wikidocs_book_page_fetched", title=book.title, links=len(book.links), text_chars=len(book.text))
    chapter_links = extract_book_toc_links(url, raw_html)
    if chapter_links:
        trace.event("wikidocs_toc_links_extracted_from_sidebar", toc_count=len(chapter_links))
    fallback_links = []
    seen = set()
    if not chapter_links:
        for link in book.links:
            href = link.get("url") or ""
            path = urlparse(href).path.strip("/")
            if not same_domain(href, url) or href in seen:
                continue
            if path.isdigit():
                title = link.get("text") or f"Chapter {len(fallback_links)+1}"
                if not is_probable_book_chapter(title, book.title):
                    continue
                seen.add(href)
                fallback_links.append({"url": href, "title": title, "depth": 0, "page_id": path})
        chapter_links = fallback_links
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
    graph["toc_tree"] = chapter_links
    graph["summary_hint"] = "WikiDocs book TOC is extracted from the left sidebar/list-group order. Each javascript:page(id) item is converted to /id and collected in TOC order."
    book_node = make_node(node_type="book", title=book.title or "WikiDocs book", url=url, order=1, text=book.text[:12000])
    graph["quality"]["pages_collected"] = 1
    graph["quality"]["toc_candidates"] = len(chapter_links)
    children = []
    for idx, item in enumerate(chapter_links, start=1):
        try:
            page = fetch_page_data(item["url"], timeout=timeout, render=False)
            trace.event("wikidocs_chapter_collected", order=idx, title=page.title or item["title"], url=item["url"], text_chars=len(page.text), code_blocks=len(page.code_blocks))
            evidence = [{"type": "heading", "text": h} for h in page.headings[:20]]
            evidence += [{"type": "code", "text": c[:2000]} for c in page.code_blocks[:15]]
            children.append(make_node(
                node_type="chapter",
                title=page.title or item["title"],
                url=item["url"],
                order=idx,
                text=page.text[:16000],
                evidence=evidence,
                meta={"toc_title": item.get("title", ""), "toc_depth": item.get("depth", 0), "page_id": item.get("page_id", "")},
            ))
            graph["quality"]["pages_collected"] += 1
        except Exception as exc:
            trace.warning("wikidocs_chapter_failed", url=item["url"], error=str(exc))
            graph["quality"]["missing"].append(f"failed chapter: {item['url']}")
    graph["quality"]["toc_collected"] = len(children)
    graph["quality"]["toc_missing"] = max(0, len(chapter_links) - len(children))
    graph["quality"]["toc_coverage"] = round(len(children) / len(chapter_links), 4) if chapter_links else 0.0
    book_node["children"] = children
    graph["nodes"].append(book_node)
    return graph
