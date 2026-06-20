from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from tools.collector_core.html_utils import fetch_page_data, normalize_url
from tools.collector_core.schema import make_graph, make_node


def is_agent_academy_url(url: str) -> bool:
    p = urlparse(url)
    return p.netloc.lower() == "microsoft.github.io" and p.path.startswith("/agent-academy")


def is_agent_academy_videos_url(url: str) -> bool:
    p = urlparse(url)
    return p.netloc.lower() == "microsoft.github.io" and p.path.rstrip("/").endswith("/agent-academy/videos")


def clean_video_title(value: str) -> str:
    title = re.sub(r"\s+", " ", str(value or "")).strip()
    title = re.sub(r"^play\s+", "", title, flags=re.I).strip()
    return title


def extract_video_cards(page, page_url: str) -> list[dict[str, str]]:
    """Extract Agent Academy video cards from thumbnail-backed HTML.

    The videos page stores YouTube IDs in img.youtube.com thumbnail URLs rather
    than normal watch links, so regular anchor extraction misses them.
    """
    html = getattr(page, "html", "") or ""
    cards: list[dict[str, str]] = []
    seen: set[str] = set()

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or ""
            match = re.search(r"img\.youtube\.com/vi/([A-Za-z0-9_-]{6,})/", src)
            if not match:
                continue
            video_id = match.group(1)
            if video_id in seen:
                continue
            seen.add(video_id)

            parent = img
            for _ in range(6):
                if parent and parent.find(class_=re.compile(r"card-title|duration|track-badge|card-date")):
                    break
                parent = parent.parent if parent else None
            scope = parent or img.parent or img

            def text_of(selector: str) -> str:
                found = scope.select_one(selector) if hasattr(scope, "select_one") else None
                return clean_video_title(found.get_text(" ", strip=True)) if found else ""

            title = (
                text_of(".card-title")
                or clean_video_title(img.get("alt") or "")
                or clean_video_title(scope.get("aria-label") if hasattr(scope, "get") else "")
                or f"YouTube video {video_id}"
            )
            cards.append({
                "video_id": video_id,
                "title": title,
                "youtube_url": f"https://youtu.be/{video_id}",
                "thumbnail_url": normalize_url(page_url, src),
                "duration": text_of(".duration"),
                "track": text_of(".track-badge"),
                "date": text_of(".card-date"),
            })
    except Exception:
        pass

    if cards:
        return cards

    for match in re.finditer(r"img\.youtube\.com/vi/([A-Za-z0-9_-]{6,})/", html):
        video_id = match.group(1)
        if video_id in seen:
            continue
        seen.add(video_id)
        window = html[max(0, match.start() - 800): match.end() + 1200]
        title_match = re.search(r'class=["\']card-title["\'][^>]*>(.*?)</', window, re.S)
        alt_match = re.search(r'alt=["\']([^"\']+)["\']', window, re.S)
        title = clean_video_title(
            re.sub(r"<[^>]+>", " ", title_match.group(1)) if title_match else (alt_match.group(1) if alt_match else "")
        )
        cards.append({
            "video_id": video_id,
            "title": title or f"YouTube video {video_id}",
            "youtube_url": f"https://youtu.be/{video_id}",
            "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
            "duration": "",
            "track": "",
            "date": "",
        })
    return cards


def collect_video_index(url: str, *, root, trace, plan, max_pages: int, timeout: int) -> dict[str, Any]:
    from tools.extractors.youtube import fetch_oembed_title, fetch_transcript, group_segments

    cards = extract_video_cards(root, url)
    trace.event("agent_academy_video_cards_extracted", video_count=len(cards))

    graph = make_graph(
        input_url=url,
        url_type="agent_academy_videos",
        site_hint="agent_academy",
        content_shape=["video_index", "video_transcript"],
        navigation_shape=["video_cards"],
        access_level=plan.access_level,
        evidence_targets=["video_cards", "youtube_ids", "transcripts"],
        title=root.title or "Agent Academy Videos",
    )
    graph["summary_hint"] = "Agent Academy videos index: video cards are extracted from YouTube thumbnail IDs, then transcript evidence is collected for each accessible video within max-pages."
    graph["video_index"] = cards

    index_lines = [
        f"Video candidates detected: {len(cards)}",
        f"Transcript collection limit: {max_pages}",
    ]
    for idx, card in enumerate(cards, start=1):
        meta = " / ".join(part for part in [card.get("track"), card.get("duration"), card.get("date")] if part)
        index_lines.append(f"{idx}. {card.get('title')} - {card.get('youtube_url')}" + (f" ({meta})" if meta else ""))

    index_node = make_node(
        node_type="video_index",
        title=root.title or "Agent Academy Videos",
        url=url,
        order=1,
        text="\n".join(index_lines),
        evidence=[
            {"type": "video_candidate", "text": card.get("title"), "url": card.get("youtube_url")}
            for card in cards[:80]
        ],
    )

    collected = 0
    transcript_segments = 0
    missing = 0
    children: list[dict[str, Any]] = []

    for idx, card in enumerate(cards[:max_pages], start=1):
        youtube_url = card["youtube_url"]
        video_id = card["video_id"]
        title = card.get("title") or fetch_oembed_title(youtube_url) or f"YouTube video {video_id}"
        video_node = make_node(
            node_type="video",
            title=title,
            url=youtube_url,
            order=idx,
            text="",
            meta={
                "video_id": video_id,
                "duration": card.get("duration", ""),
                "track": card.get("track", ""),
                "date": card.get("date", ""),
                "thumbnail_url": card.get("thumbnail_url", ""),
            },
        )
        try:
            raw_segments = fetch_transcript(video_id)
            grouped = group_segments(raw_segments)
            video_node["text"] = (
                f"Transcript collected for {title}. "
                f"Raw segments: {len(raw_segments)}. Grouped segments: {len(grouped)}."
            )
            video_node["children"] = [
                make_node(
                    node_type="transcript_segment",
                    title=f"{title} transcript {seg_idx:03d}",
                    url=youtube_url,
                    order=seg_idx,
                    text=item["text"],
                    meta={"start": item["start"], "end": item["end"]},
                )
                for seg_idx, item in enumerate(grouped, start=1)
            ]
            transcript_segments += len(raw_segments)
            collected += 1
            trace.event(
                "agent_academy_video_transcript_collected",
                order=idx,
                video_id=video_id,
                title=title,
                raw_segments=len(raw_segments),
                grouped_segments=len(grouped),
            )
        except Exception as exc:
            missing += 1
            video_node["text"] = f"Transcript was not accessible for this video. Title: {title}."
            video_node["evidence"] = [{"type": "transcript_missing", "text": str(exc)[:500]}]
            trace.warning("agent_academy_video_transcript_missing", order=idx, video_id=video_id, title=title, error=str(exc))
        children.append(video_node)

    index_node["children"] = children
    graph["nodes"].append(index_node)
    graph["assets"].extend({"type": "image", "url": card.get("thumbnail_url"), "alt": card.get("title")} for card in cards[:80] if card.get("thumbnail_url"))
    graph["quality"]["pages_collected"] = 1 + len(children)
    graph["quality"]["video_candidates"] = len(cards)
    graph["quality"]["videos_transcript_collected"] = collected
    graph["quality"]["videos_transcript_missing"] = missing
    graph["quality"]["transcript_segments"] = transcript_segments
    graph["quality"]["images_collected"] = len(graph["assets"])
    graph["quality"]["text_chars"] = len(index_node["text"]) + sum(
        len(node.get("text") or "") + sum(len(child.get("text") or "") for child in node.get("children") or [])
        for node in children
    )
    if not cards:
        graph["quality"]["missing"].append("Agent Academy video cards not detected")
    if cards and collected == 0:
        graph["quality"]["warnings"].append("video candidates detected but transcripts were not collected")
    return graph


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


def extract_curriculum_overview(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    try:
        start = next(i for i, line in enumerate(lines) if "Curriculum Overview" in line)
    except StopIteration:
        return []

    overview: list[dict[str, str]] = []
    i = start + 1
    while i < len(lines):
        line = lines[i]
        if line in {"NOTE", "Evidence:"} or line.startswith("✅"):
            break
        if line.isdigit() and len(line) <= 2:
            number = line.zfill(2)
            j = i + 1
            while j < len(lines) and len(lines[j]) <= 3:
                j += 1
            title = lines[j] if j < len(lines) else f"Lesson {number}"
            briefing = lines[j + 1] if j + 1 < len(lines) else ""
            if title.lower() not in {"lesson", "title", "mission briefing"}:
                overview.append({"lesson": number, "title": title, "briefing": briefing})
            i = j + 2
            continue
        i += 1
    return overview[:20]


def collect(url: str, *, run_dir, trace, plan, max_pages: int = 20, visible: bool = False, max_depth: int = 1, timeout: int = 30, **kwargs) -> dict[str, Any]:
    root = fetch_page_data(url, timeout=timeout, render=False)
    trace.event("agent_academy_root_fetched", title=root.title, links=len(root.links), text_chars=len(root.text))

    if is_agent_academy_videos_url(url):
        return collect_video_index(url, root=root, trace=trace, plan=plan, max_pages=max_pages, timeout=timeout)

    course_url = find_course_entry(root, url)
    trace.event("agent_academy_course_entry_selected", course_url=course_url)
    course = fetch_page_data(course_url, timeout=timeout, render=False)
    lesson_links = extract_lesson_links(course, course_url, max_pages=max_pages)
    curriculum = extract_curriculum_overview(course.text)
    trace.event("agent_academy_lesson_links_extracted", lesson_count=len(lesson_links))
    trace.event("agent_academy_curriculum_overview_extracted", item_count=len(curriculum))

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
    graph["curriculum_overview"] = curriculum
    graph["summary_hint"] = "Agent Academy Recruit curriculum: Copilot Studio setup, declarative/custom agents, Topics, Adaptive Cards, Agent Flows, publishing, licensing, and badge validation."

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
            overview_item = next((row for row in curriculum if row.get("lesson") == f"{idx - 1:02d}"), {})
            children.append(make_node(
                node_type="lesson",
                title=page.title or item["title"],
                url=item["url"],
                order=idx,
                text=page.text[:16000],
                evidence=evidence,
                meta={
                    "curriculum_lesson": overview_item.get("lesson", ""),
                    "mission_briefing": overview_item.get("briefing", ""),
                },
            ))
            graph["quality"]["pages_collected"] += 1
            graph["quality"]["images_collected"] += len(page.images)
            graph["assets"].extend({"type": "image", **img} for img in page.images[:10])
        except Exception as exc:
            trace.warning("agent_academy_lesson_failed", url=item["url"], error=str(exc))
            graph["quality"]["missing"].append(f"failed lesson: {item['url']}")
    course_node["children"] = children
    graph["nodes"].append(course_node)
    return graph
