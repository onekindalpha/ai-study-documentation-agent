from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from tools.collector_core.html_utils import fetch_page_data, normalize_url
from tools.collector_core.schema import make_graph, make_node


def is_agent_academy_url(url: str) -> bool:
    p = urlparse(url)
    return p.netloc.lower() == "microsoft.github.io" and p.path.startswith("/agent-academy")


def find_course_entry(root_page, root_url: str) -> str:
    candidates = []
    for link in root_page.links:
        text = (link.get("text") or "").lower()
        href = link.get("url") or ""
        if not is_agent_academy_url(href):
            continue
        score = 0
        if "get started" in text:
            score += 5
        if "recruit" in text or "/recruit" in href:
            score += 4
        if href.rstrip("/").endswith("/agent-academy/recruit") or href.rstrip("/").endswith("/agent-academy/recruit"):
            score += 4
        if score:
            candidates.append((score, href))
    if candidates:
        return sorted(candidates, reverse=True)[0][1]
    return normalize_url(root_url, "/agent-academy/recruit/")


def extract_lesson_links(course_page, course_url: str, max_pages: int) -> list[dict]:
    result = []
    seen = set()
    for link in course_page.links:
        href = link.get("url") or ""
        text = (link.get("text") or "").strip()
        if not href or href in seen or not is_agent_academy_url(href):
            continue
        path = urlparse(href).path.strip("/")
        if not path.startswith("agent-academy/recruit"):
            continue
        if href.rstrip("/") == course_url.rstrip("/"):
            continue
        if any(bad in path for bad in ["assets", "img", "images"]):
            continue
        seen.add(href)
        result.append({"url": href, "title": text or path.split("/")[-1].replace("-", " ")})
    return result[:max_pages]


def collect(url: str, *, run_dir, trace, plan, max_pages: int = 20, visible: bool = False, max_depth: int = 1, timeout: int = 30, **kwargs) -> dict[str, Any]:
    root = fetch_page_data(url, timeout=timeout, render=False)
    trace.event("agent_academy_root_fetched", title=root.title, links=len(root.links), text_chars=len(root.text))
    course_url = find_course_entry(root, url)
    trace.event("agent_academy_course_entry_selected", course_url=course_url)
    course = fetch_page_data(course_url, timeout=timeout, render=False)
    lesson_links = extract_lesson_links(course, course_url, max_pages=max_pages)
    trace.event("agent_academy_lesson_links_extracted", lesson_count=len(lesson_links))

    graph = make_graph(
        input_url=url,
        url_type="agent_academy",
        site_hint="agent_academy",
        content_shape=plan.content_shape,
        navigation_shape=plan.navigation_shape,
        access_level=plan.access_level,
        evidence_targets=plan.evidence_targets,
        title=course.title or root.title or "Microsoft Agent Academy",
    )

    course_node = make_node(
        node_type="course",
        title=course.title or "Agent Academy course",
        url=course_url,
        order=1,
        text=course.text[:12000],
        evidence=[{"type": "heading", "text": h} for h in course.headings[:20]],
    )
    graph["quality"]["pages_collected"] = 1
    graph["quality"]["images_collected"] += len(course.images)
    graph["assets"].extend({"type": "image", **img} for img in course.images[:30])

    children = []
    for idx, item in enumerate(lesson_links, start=1):
        try:
            page = fetch_page_data(item["url"], timeout=timeout, render=False)
            trace.event("agent_academy_lesson_collected", order=idx, title=page.title or item["title"], url=item["url"], text_chars=len(page.text))
            evidence = [{"type": "heading", "text": h} for h in page.headings[:20]]
            evidence += [{"type": "code", "text": c[:2000]} for c in page.code_blocks[:10]]
            lab_links = [l for l in page.links if any(k in ((l.get("text") or "") + " " + (l.get("url") or "")).lower() for k in ["lab", "exercise", "github", "learn"])]
            evidence += [{"type": "external_link", "text": l.get("text") or l.get("url"), "url": l.get("url")} for l in lab_links[:20]]
            children.append(make_node(node_type="lesson", title=page.title or item["title"], url=item["url"], order=idx, text=page.text[:16000], evidence=evidence))
            graph["quality"]["pages_collected"] += 1
            graph["quality"]["images_collected"] += len(page.images)
            graph["assets"].extend({"type": "image", **img} for img in page.images[:10])
        except Exception as exc:
            trace.warning("agent_academy_lesson_failed", url=item["url"], error=str(exc))
            graph["quality"]["missing"].append(f"failed lesson: {item['url']}")
    course_node["children"] = children
    graph["nodes"].append(course_node)
    return graph
