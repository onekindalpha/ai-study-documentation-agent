from __future__ import annotations

from typing import Any

from tools.collector_core.html_utils import fetch_page_data
from tools.collector_core.schema import make_graph, make_node


def collect(url: str, *, run_dir, trace, plan, max_pages: int = 1, visible: bool = False, max_depth: int = 1, timeout: int = 30, **kwargs) -> dict[str, Any]:
    graph = make_graph(
        input_url=url,
        url_type="generic_web",
        site_hint=plan.site_hint,
        content_shape=plan.content_shape,
        navigation_shape=plan.navigation_shape,
        access_level=plan.access_level,
        evidence_targets=plan.evidence_targets,
        title="Generic web source",
    )
    try:
        page = fetch_page_data(url, timeout=timeout, render=False)
        trace.event("generic_page_collected", title=page.title, text_chars=len(page.text), links=len(page.links))
        evidence = [{"type": "heading", "text": h} for h in page.headings[:20]]
        evidence += [{"type": "link", "text": l.get("text") or l.get("url"), "url": l.get("url")} for l in page.links[:50]]
        graph["title"] = page.title or url
        graph["nodes"].append(make_node(node_type="page", title=page.title or url, url=url, order=1, text=page.text[:16000], evidence=evidence))
        graph["quality"]["pages_collected"] = 1
        graph["quality"]["images_collected"] = len(page.images)
        graph["assets"].extend({"type": "image", **img} for img in page.images[:30])
    except Exception as exc:
        trace.warning("generic_page_failed", error=str(exc))
        graph["quality"]["missing"].append(f"generic page fetch failed: {exc}")
    return graph
