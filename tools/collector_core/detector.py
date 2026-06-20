from __future__ import annotations

from dataclasses import dataclass, asdict
from urllib.parse import urlparse


@dataclass
class CollectionPlan:
    input_url: str
    extractor: str
    site_hint: str
    content_shape: list[str]
    navigation_shape: list[str]
    access_level: str
    evidence_targets: list[str]
    plan: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def detect_url(url: str) -> CollectionPlan:
    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")
    path = parsed.path.lower()

    if host in {"youtu.be", "youtube.com", "m.youtube.com", "music.youtube.com"} or "youtube.com" in host:
        return CollectionPlan(
            input_url=url,
            extractor="youtube",
            site_hint="youtube",
            content_shape=["video_only"],
            navigation_shape=["single_page"],
            access_level="public_or_partial",
            evidence_targets=["title", "description", "transcript", "chapters"],
            plan=["extract_video_id", "fetch_transcript", "group_transcript", "write_source_graph"],
        )

    if host == "microsoft.github.io" and path.startswith("/agent-academy"):
        return CollectionPlan(
            input_url=url,
            extractor="agent_academy",
            site_hint="agent_academy",
            content_shape=["text_video_mixed", "lab_or_exercise"],
            navigation_shape=["sidebar_course", "next_prev_sequence"],
            access_level="public",
            evidence_targets=["main_text", "toc", "headings", "links", "images", "code_blocks", "lab_steps"],
            plan=["render_or_fetch_page", "find_course_entry", "extract_lesson_links", "crawl_lessons_in_order", "write_source_graph"],
        )

    if host == "wikidocs.net" and path.startswith("/book/"):
        return CollectionPlan(
            input_url=url,
            extractor="wikidocs",
            site_hint="wikidocs",
            content_shape=["text_only", "code_heavy"],
            navigation_shape=["toc_based"],
            access_level="public",
            evidence_targets=["main_text", "toc", "headings", "code_blocks", "tables"],
            plan=["fetch_book_page", "extract_toc", "crawl_chapters_in_order", "write_source_graph"],
        )

    if host.endswith("oopy.io"):
        return CollectionPlan(
            input_url=url,
            extractor="oopy",
            site_hint="oopy",
            content_shape=["text_only", "image_or_slide_heavy"],
            navigation_shape=["child_pages"],
            access_level="public_or_partial",
            evidence_targets=["main_text", "child_links", "images", "headings"],
            plan=["render_root_page", "extract_same_domain_links", "crawl_child_pages_with_limits", "write_source_graph"],
        )

    if "inflearn.com" in host or "udemy.com" in host:
        return CollectionPlan(
            input_url=url,
            extractor="protected_course",
            site_hint="inflearn" if "inflearn.com" in host else "udemy",
            content_shape=["video_only", "text_video_mixed"],
            navigation_shape=["dynamic_app"],
            access_level="login_or_enrollment_may_be_required",
            evidence_targets=["visible_text", "curriculum", "current_lecture", "transcript_if_visible"],
            plan=["render_visible_page", "extract_visible_dom", "detect_curriculum", "detect_transcript", "write_partial_report"],
        )


    if "aiskillsnavigator.microsoft.com" in host:
        return CollectionPlan(
            input_url=url,
            extractor="ai_skills",
            site_hint="ai_skills_navigator",
            content_shape=["text_video_mixed", "lab_or_exercise"],
            navigation_shape=["dynamic_app"],
            access_level="login_may_be_required",
            evidence_targets=["lesson_tree", "main_text", "video_links", "lab_links", "external_references"],
            plan=["run_existing_ai_skills_collector_adapter", "write_source_graph"],
        )

    return CollectionPlan(
        input_url=url,
        extractor="generic_web",
        site_hint=host or "generic_web",
        content_shape=["unknown"],
        navigation_shape=["single_page"],
        access_level="public_or_partial",
        evidence_targets=["main_text", "headings", "links", "images"],
        plan=["fetch_page", "extract_main_text", "write_source_graph"],
    )
