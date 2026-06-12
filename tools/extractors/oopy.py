from __future__ import annotations

from collections import deque
from typing import Any

from tools.collector_core.html_utils import fetch_page_data, same_domain
from tools.collector_core.schema import make_graph, make_node


def collect(url: str, *, run_dir, trace, plan, max_pages: int = 12, max_depth: int = 1, visible: bool = False, timeout: int = 30, **kwargs) -> dict[str, Any]:
    graph = make_graph(
        input_url=url,
        url_type="oopy",
        site_hint="oopy",
        content_shape=plan.content_shape,
        navigation_shape=plan.navigation_shape,
        access_level=plan.access_level,
        evidence_targets=plan.evidence_targets,
        title="Oopy source",
    )
    queue = deque([(url, 0)])
    seen = set()
    nodes = []
    order = 0
    while queue and len(nodes) < max_pages:
        current, depth = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        try:
            screenshot = run_dir / "screenshots" / f"oopy_{len(nodes)+1:03d}.png"
            page = fetch_page_data(current, timeout=timeout, render=True, visible=visible, screenshot_path=screenshot)
            order += 1
            trace.event("oopy_page_collected", order=order, depth=depth, title=page.title, url=current, text_chars=len(page.text), links=len(page.links))
            evidence = [{"type": "heading", "text": h} for h in page.headings[:20]]
            node = make_node(node_type="page", title=page.title or f"Oopy page {order}", url=current, order=order, text=page.text[:16000], evidence=evidence, meta={"depth": depth})
            nodes.append(node)
            graph["quality"]["pages_collected"] += 1
            graph["quality"]["images_collected"] += len(page.images)
            graph["assets"].extend({"type": "image", **img} for img in page.images[:15])
            if depth < max_depth:
                for link in page.links:
                    href = link.get("url") or ""
                    if href and same_domain(href, url) and href not in seen:
                        queue.append((href, depth + 1))
            graph["quality"]["child_links_collected"] = len(seen)
        except Exception as exc:
            trace.warning("oopy_page_failed", url=current, error=str(exc))
            graph["quality"]["missing"].append(f"failed oopy page: {current}")
    graph["nodes"] = nodes
    if nodes:
        graph["title"] = nodes[0].get("title") or "Oopy source"
    return graph
