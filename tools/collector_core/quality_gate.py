from __future__ import annotations

from typing import Any

from .schema import update_quality_from_graph


def apply_quality_gate(graph: dict[str, Any]) -> dict[str, Any]:
    update_quality_from_graph(graph)
    q = graph.setdefault("quality", {})
    missing = q.setdefault("missing", [])
    warnings = q.setdefault("warnings", [])

    url_type = graph.get("url_type") or "generic_web"
    text_chars = int(q.get("text_chars") or 0)
    pages = int(q.get("pages_collected") or 0)
    transcripts = int(q.get("transcript_segments") or 0)

    status = "partial"
    if url_type == "youtube":
        if transcripts > 0 and text_chars >= 500:
            status = "pass"
        else:
            status = "partial"
            if "video transcript not accessible" not in missing:
                missing.append("video transcript not accessible")
    elif url_type in {"agent_academy", "wikidocs", "oopy"}:
        if pages >= 2 and text_chars >= 1500:
            status = "pass"
        elif text_chars >= 500:
            status = "partial"
            warnings.append("collection is thin for a multi-page learning source")
        else:
            status = "fail"
            missing.append("not enough learning text collected")
    elif url_type == "protected_course":
        if text_chars >= 1000 or pages >= 1:
            status = "partial"
            missing.append("protected course content may require login, enrollment, or visible transcript access")
        else:
            status = "fail"
            missing.append("protected course page did not expose enough visible content")
    else:
        if text_chars >= 1200:
            status = "partial"
            warnings.append("generic web page collected; learning structure is not confirmed")
        else:
            status = "fail"
            missing.append("URL does not expose enough learning content")

    graph["status"] = status
    q["quality_status"] = status
    q["can_generate_article"] = status == "pass"
    return graph
