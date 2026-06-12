from __future__ import annotations

import json
import re
import urllib.request
from urllib.parse import parse_qs, quote, urlparse
from typing import Any

from tools.collector_core.schema import make_graph, make_node


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")

    if host == "youtu.be":
        return parsed.path.strip("/").split("/")[0]

    qs = parse_qs(parsed.query)
    if qs.get("v"):
        return qs["v"][0]

    match = re.search(r"/(shorts|embed|live)/([A-Za-z0-9_-]{6,})", parsed.path)
    if match:
        return match.group(2)

    raise ValueError(f"Could not extract YouTube video id from {url}")


def fetch_oembed_title(url: str) -> str:
    api = "https://www.youtube.com/oembed?format=json&url=" + quote(url, safe="")

    try:
        import certifi
        import requests

        resp = requests.get(api, timeout=10, verify=certifi.where())
        resp.raise_for_status()
        data = resp.json()
        return data.get("title") or ""
    except Exception:
        try:
            with urllib.request.urlopen(api, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            return data.get("title") or ""
        except Exception:
            return ""


def fetch_transcript(video_id: str, languages: list[str] | None = None) -> list[dict[str, Any]]:
    languages = languages or ["ko", "en", "en-US", "ja"]

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except Exception as exc:
        raise RuntimeError(f"youtube-transcript-api import failed: {exc}") from exc

    def normalize_fetched(fetched) -> list[dict[str, Any]]:
        if hasattr(fetched, "to_raw_data"):
            return list(fetched.to_raw_data())

        out: list[dict[str, Any]] = []
        for item in fetched:
            if isinstance(item, dict):
                out.append(item)
            else:
                out.append(
                    {
                        "text": getattr(item, "text", ""),
                        "start": getattr(item, "start", 0.0),
                        "duration": getattr(item, "duration", 0.0),
                    }
                )
        return out

    errors: list[str] = []
    api = YouTubeTranscriptApi()

    try:
        fetched = api.fetch(video_id, languages=languages)
        return normalize_fetched(fetched)
    except Exception as exc:
        errors.append(f"api.fetch={type(exc).__name__}: {exc}")

    try:
        transcript_list = api.list(video_id)

        for lang in languages:
            try:
                transcript = transcript_list.find_transcript([lang])
                return normalize_fetched(transcript.fetch())
            except Exception as exc:
                errors.append(f"find_transcript[{lang}]={type(exc).__name__}: {exc}")

        try:
            transcript = transcript_list.find_generated_transcript(languages)
            return normalize_fetched(transcript.fetch())
        except Exception as exc:
            errors.append(f"find_generated_transcript={type(exc).__name__}: {exc}")

    except Exception as exc:
        errors.append(f"api.list={type(exc).__name__}: {exc}")

    raise RuntimeError("transcript not available: " + " | ".join(errors))


def is_noise_segment(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    return bool(
        re.fullmatch(
            r"\[(음악|박수|웃음|Music|Applause|Laughter|music|applause|laughter)\]",
            value,
        )
    )


def group_segments(segments: list[dict[str, Any]], max_chars: int = 1600) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current: list[str] = []
    start = 0.0
    end = 0.0

    for seg in segments:
        text = (seg.get("text") or "").replace("\n", " ").strip()
        if is_noise_segment(text):
            continue

        if not current:
            start = float(seg.get("start") or 0.0)

        end = float(seg.get("start") or 0.0) + float(seg.get("duration") or 0.0)
        current.append(text)

        if len(" ".join(current)) >= max_chars:
            groups.append({"start": start, "end": end, "text": " ".join(current)})
            current = []

    if current:
        groups.append({"start": start, "end": end, "text": " ".join(current)})

    return groups


def collect(
    url: str,
    *,
    run_dir,
    trace,
    plan,
    max_pages: int = 20,
    visible: bool = False,
    max_depth: int = 1,
    timeout: int = 30,
    **kwargs,
) -> dict[str, Any]:
    video_id = extract_video_id(url)
    trace.event("youtube_video_id_extracted", video_id=video_id)

    title = fetch_oembed_title(url) or f"YouTube video {video_id}"

    graph = make_graph(
        input_url=url,
        url_type="youtube",
        site_hint="youtube",
        content_shape=plan.content_shape,
        navigation_shape=plan.navigation_shape,
        access_level=plan.access_level,
        evidence_targets=plan.evidence_targets,
        title=title,
    )

    video_node = make_node(
        node_type="video",
        title=title,
        url=url,
        order=1,
        text="",
        meta={"video_id": video_id},
    )

    try:
        raw_segments = fetch_transcript(video_id)
        trace.event("youtube_transcript_collected", raw_segments=len(raw_segments))

        grouped = group_segments(raw_segments)
        trace.event("youtube_transcript_grouped", grouped_segments=len(grouped))

        children = []
        for idx, item in enumerate(grouped, start=1):
            children.append(
                make_node(
                    node_type="transcript_segment",
                    title=f"Transcript segment {idx:03d}",
                    url=url,
                    order=idx,
                    text=item["text"],
                    meta={"start": item["start"], "end": item["end"]},
                )
            )

        video_node["children"] = children
        video_node["text"] = f"Transcript collected for YouTube video {video_id}."
        graph["quality"]["transcript_segments"] = len(raw_segments)

    except Exception as exc:
        trace.warning("youtube_transcript_missing", error=str(exc))
        graph["quality"]["missing"].append("video transcript not accessible")
        video_node["text"] = "Transcript was not accessible for this video."

    graph["nodes"].append(video_node)
    graph["quality"]["pages_collected"] = 1
    graph["quality"]["text_chars"] = len(video_node.get("text", "")) + sum(
        len(c.get("text", "")) for c in video_node.get("children", [])
    )

    return graph
