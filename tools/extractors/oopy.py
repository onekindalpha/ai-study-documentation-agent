from __future__ import annotations

import re
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

from tools.collector_core.html_utils import fetch_page_data
from tools.collector_core.schema import make_graph, make_node


UUIDISH = re.compile(r"/[0-9a-f]{8,}[-0-9a-f]*", re.I)


def norm(url: str) -> str:
    return urldefrag(url)[0].rstrip("/")


def link_url(link: Any) -> str:
    if isinstance(link, dict):
        return link.get("url") or link.get("href") or ""
    return getattr(link, "url", "") or getattr(link, "href", "") or ""


def link_text(link: Any) -> str:
    if isinstance(link, dict):
        return link.get("text") or link.get("title") or ""
    return getattr(link, "text", "") or getattr(link, "title", "") or ""


def is_nav_or_profile(root_url: str, url: str, text: str) -> bool:
    root = urlparse(root_url)
    parsed = urlparse(url)
    label = " ".join((text or "").split()).lower()
    path = parsed.path.rstrip("/")

    if parsed.netloc != root.netloc:
        return True

    if norm(url) == norm(root_url):
        return True

    if path in {"", "/", "/devlog", "/about"}:
        return True

    if label in {"home", "devlog", "about"}:
        return True

    if "backend engineer" in label:
        return True

    if "김영찬" in text:
        return True

    return False


def extract_root_toc_links(root_url: str, page) -> list[dict[str, str]]:
    seen: set[str] = set()
    toc: list[dict[str, str]] = []

    for link in getattr(page, "links", []) or []:
        href = link_url(link)
        text = " ".join(link_text(link).split())

        if not href:
            continue

        abs_url = norm(urljoin(root_url, href))

        if abs_url in seen:
            continue
        seen.add(abs_url)

        parsed = urlparse(abs_url)

        if is_nav_or_profile(root_url, abs_url, text):
            continue

        if not UUIDISH.search(parsed.path):
            continue

        toc.append({"url": abs_url, "link_text": text or abs_url})

    return toc


def page_node(page, *, url: str, order: int, depth: int, node_type: str, meta_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    node = make_node(
        node_type=node_type,
        title=getattr(page, "title", "") or url,
        url=url,
        order=order,
        text=getattr(page, "text", "") or "",
        meta={"depth": depth, **(meta_extra or {})},
    )
    return node


def collect(
    url: str,
    *,
    run_dir,
    trace,
    plan,
    max_pages: int = 30,
    max_depth: int = 1,
    visible: bool = False,
    timeout: int = 45,
    **kwargs,
) -> dict[str, Any]:
    root_url = norm(url)
    screenshot_dir = run_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    root_page = fetch_page_data(
        root_url,
        render=True,
        visible=visible,
        timeout=timeout,
        screenshot_path=screenshot_dir / "oopy_root.png",
    )

    graph = make_graph(
        input_url=root_url,
        url_type="oopy",
        site_hint="oopy",
        content_shape=plan.content_shape,
        navigation_shape=plan.navigation_shape,
        access_level=plan.access_level,
        evidence_targets=plan.evidence_targets,
        title=getattr(root_page, "title", "") or root_url,
    )

    root_node = page_node(root_page, url=root_url, order=1, depth=0, node_type="oopy_root")
    graph["nodes"].append(root_node)

    toc_links = extract_root_toc_links(root_url, root_page)
    trace.event(
        "oopy_root_toc_detected",
        root_title=getattr(root_page, "title", "") or "",
        toc_candidates=len(toc_links),
        root_links=len(getattr(root_page, "links", []) or []),
    )

    max_toc_pages = max(0, max_pages - 1)
    selected_toc_links = toc_links[:max_toc_pages]

    collected = 0
    failed = 0

    for idx, item in enumerate(selected_toc_links, start=2):
        child_url = item["url"]

        try:
            page = fetch_page_data(
                child_url,
                render=True,
                visible=False,
                timeout=timeout,
            )

            node = page_node(
                page,
                url=child_url,
                order=idx,
                depth=1,
                node_type="oopy_toc_page",
                meta_extra={"root_toc_link_text": item.get("link_text", "")},
            )

            graph["nodes"].append(node)
            collected += 1

            trace.event(
                "oopy_toc_page_collected",
                order=idx - 1,
                title=node.get("title"),
                url=child_url,
                text_chars=len(node.get("text") or ""),
            )

        except Exception as exc:
            failed += 1
            trace.warning(
                "oopy_toc_page_failed",
                url=child_url,
                link_text=item.get("link_text", ""),
                error=str(exc),
            )

    toc_total = len(toc_links)
    toc_missing = max(0, toc_total - collected)
    toc_coverage = collected / toc_total if toc_total else 0.0

    graph["quality"]["pages_collected"] = len(graph["nodes"])
    graph["quality"]["text_chars"] = sum(len(n.get("text") or "") for n in graph["nodes"])
    graph["quality"]["images_collected"] = len(getattr(root_page, "images", []) or [])
    graph["quality"]["child_links_collected"] = collected
    graph["quality"]["toc_candidates"] = toc_total
    graph["quality"]["toc_collected"] = collected
    graph["quality"]["toc_missing"] = toc_missing
    graph["quality"]["toc_coverage"] = round(toc_coverage, 4)
    graph["quality"]["extra_collected_outside_toc"] = 0

    if toc_total == 0:
        graph["quality"]["missing"].append("root toc links not detected")

    if toc_total > 0 and toc_coverage < 0.8:
        graph["quality"]["warnings"].append("toc coverage below 80 percent")

    if failed:
        graph["quality"]["warnings"].append(f"{failed} toc pages failed to collect")

    trace.event(
        "oopy_toc_collection_finished",
        toc_candidates=toc_total,
        toc_collected=collected,
        toc_missing=toc_missing,
        toc_coverage=round(toc_coverage, 4),
        pages_collected=len(graph["nodes"]),
        extra_collected_outside_toc=0,
    )

    return graph
