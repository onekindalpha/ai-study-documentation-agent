from __future__ import annotations

from typing import Any

from tools.collector_core.html_utils import fetch_page_data
from tools.collector_core.schema import make_graph, make_node


def collect(url: str, *, run_dir, trace, plan, max_pages: int = 3, visible: bool = False, max_depth: int = 1, timeout: int = 30, **kwargs) -> dict[str, Any]:
    graph = make_graph(
        input_url=url,
        url_type="protected_course",
        site_hint=plan.site_hint,
        content_shape=plan.content_shape,
        navigation_shape=plan.navigation_shape,
        access_level=plan.access_level,
        evidence_targets=plan.evidence_targets,
        title=f"Protected course page: {plan.site_hint}",
    )
    try:
        screenshot = run_dir / "screenshots" / "protected_course_visible_page.png"
        page = fetch_page_data(url, timeout=timeout, render=True, visible=visible, screenshot_path=screenshot)
        trace.event("protected_course_visible_page_collected", title=page.title, text_chars=len(page.text), links=len(page.links))
        curriculum_candidates = []
        for link in page.links:
            text = (link.get("text") or "").strip()
            if text and len(text) < 180:
                curriculum_candidates.append(text)
        evidence = [{"type": "heading", "text": h} for h in page.headings[:20]]
        evidence += [{"type": "curriculum_candidate", "text": t} for t in curriculum_candidates[:80]]
        graph["nodes"].append(make_node(node_type="current_lecture", title=page.title or "Visible course page", url=url, order=1, text=page.text[:16000], evidence=evidence))
        graph["quality"]["pages_collected"] = 1
        graph["quality"]["images_collected"] = len(page.images)
        graph["quality"]["missing"].extend([
            "video transcript may require login, enrollment, or caption access",
            "DRM/protected video content is not collected in v0",
        ])
    except Exception as exc:
        trace.warning("protected_course_render_failed", error=str(exc))
        graph["quality"]["missing"].append(f"protected course render failed: {exc}")
    return graph
