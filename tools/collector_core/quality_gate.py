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
    usable_units = int(q.get("usable_units") or q.get("usable_text_units_count") or 0)
    usable_chars = int(q.get("usable_text_chars") or 0)
    high_value_units = int(q.get("high_value_units") or 0)

    def enough_usable() -> bool:
        return usable_units >= 5 or usable_chars >= 800 or high_value_units >= 3

    status = "partial"
    if url_type == "youtube":
        if transcripts > 0 and text_chars >= 500:
            status = "pass"
        else:
            status = "partial"
            if "video transcript not accessible" not in missing:
                missing.append("video transcript not accessible")
    elif url_type == "agent_academy_videos":
        video_candidates = int(q.get("video_candidates") or 0)
        videos_collected = int(q.get("videos_transcript_collected") or 0)
        if video_candidates > 0 and videos_collected > 0 and transcripts > 0:
            status = "pass"
        elif video_candidates > 0:
            status = "partial"
            warnings.append("video candidates detected but transcript coverage is incomplete")
        else:
            status = "fail"
            missing.append("Agent Academy video cards not detected")
    elif url_type in {"agent_academy", "wikidocs", "oopy"}:
        if url_type == "oopy" and (enough_usable() or (pages >= 1 and text_chars >= 900)):
            status = "pass"
            if pages <= 1:
                warnings.append("Oopy single-page learning material accepted; child pages were not required")
        elif pages >= 2 and text_chars >= 1500:
            status = "pass"
        elif text_chars >= 500:
            status = "partial"
            warnings.append("collection is thin for a multi-page learning source")
        else:
            status = "fail"
            missing.append("not enough learning text collected")
    elif url_type == "notion_page":
        if enough_usable():
            status = "pass"
        else:
            status = "fail"
            if "notion page adapter needed or no usable notion blocks" not in missing:
                missing.append("notion page adapter needed or no usable notion blocks")
            if "provide Notion export markdown/html or pasted text" not in warnings:
                warnings.append("provide Notion export markdown/html or pasted text")
    elif url_type == "protected_course":
        if text_chars >= 1000 or pages >= 1:
            status = "partial"
            missing.append("protected course content may require login, enrollment, or visible transcript access")
        else:
            status = "fail"
            missing.append("protected course page did not expose enough visible content")
    else:
        if enough_usable() or text_chars >= 1200:
            status = "partial"
            warnings.append("generic web page collected; learning structure is not confirmed")
        else:
            status = "fail"
            missing.append("URL does not expose enough learning content")

    graph["status"] = status
    q["quality_status"] = status
    # pass is enough for direct generation; partial may still be accepted later by
    # the app when usable_text_units are high-quality and source-specific checks pass.
    q["can_generate_article"] = status == "pass"
    return graph
