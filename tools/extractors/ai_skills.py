from __future__ import annotations

from typing import Any

from tools.collector_core.schema import make_graph, make_node


def collect(url: str, *, run_dir, trace, plan, max_pages: int = 20, visible: bool = False, max_depth: int = 1, timeout: int = 30, **kwargs) -> dict[str, Any]:
    graph = make_graph(
        input_url=url,
        url_type="ai_skills",
        site_hint="ai_skills_navigator",
        content_shape=plan.content_shape,
        navigation_shape=plan.navigation_shape,
        access_level=plan.access_level,
        evidence_targets=plan.evidence_targets,
        title="AI Skills Navigator source",
    )
    trace.warning("ai_skills_adapter_placeholder", message="Existing AI Skills collector should be wrapped here later.")
    graph["nodes"].append(
        make_node(
            node_type="course",
            title="AI Skills Navigator adapter placeholder",
            url=url,
            order=1,
            text="AI Skills Navigator collection is preserved in the dedicated branch. Universal adapter wiring is pending.",
        )
    )
    graph["quality"]["missing"].append("AI Skills adapter not wired in universal collector v0")
    return graph
