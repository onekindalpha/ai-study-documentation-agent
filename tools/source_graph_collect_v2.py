from __future__ import annotations

import argparse
import asyncio
import html as html_lib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

USER_AGENT = "Mozilla/5.0 StudyCaptureCopilot/2.0 (+source-graph-collector)"
MEDIA_EXT_RE = re.compile(r"\.(png|jpe?g|gif|webp|svg|ico|css|js|woff2?|ttf|map|pdf|zip)(?:$|[?#])", re.I)
LEARNING_LINK_RE = re.compile(
    r"(chapter|lesson|lab|exercise|instructions|module|learn|summary|transcript|curriculum|강의|실습|목차|단원|학습|문제|정리|article|docs|notion|oopy)",
    re.I,
)
YOUTUBE_RE = re.compile(r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)([A-Za-z0-9_-]{6,})")


def clean_text(text: str) -> str:
    text = html_lib.unescape(text or "")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_name(text: str, default: str = "source_pack") -> str:
    text = re.sub(r"[^a-zA-Z0-9가-힣_-]+", "_", text or default).strip("_")
    return (text or default)[:70]


def normalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc, p.path.rstrip("/") or "/", "", p.query, ""))
    except Exception:
        return url


def url_domain(url: str) -> str:
    return (urlparse(url or "").netloc or "").lower().removeprefix("www.")


def same_origin(a: str, b: str) -> bool:
    return url_domain(a) == url_domain(b)


def video_id(url: str) -> str:
    m = YOUTUBE_RE.search(url or "")
    if m:
        return m.group(1)
    p = urlparse(url or "")
    qs = parse_qs(p.query or "")
    return (qs.get("v") or [""])[0]


def fetch_bytes(url: str, timeout: int = 18, limit: int = 3_000_000) -> tuple[bytes, str, str]:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read(limit)
        ctype = resp.headers.get("Content-Type", "")
        final_url = resp.geturl()
    return data, ctype, final_url


def fetch_text(url: str, timeout: int = 18, limit: int = 3_000_000) -> tuple[str, str, str]:
    data, ctype, final_url = fetch_bytes(url, timeout=timeout, limit=limit)
    return data.decode("utf-8", errors="replace"), ctype, final_url


class HTMLExtract(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.title = ""
        self.parts: list[str] = []
        self.links: list[dict[str, str]] = []
        self.headings: list[dict[str, Any]] = []
        self._tag_stack: list[str] = []
        self._current_link: str | None = None
        self._current_link_text: list[str] = []
        self._current_heading_level: int | None = None
        self._current_heading_text: list[str] = []
        self._in_script = False
        self.scripts: list[str] = []
        self._script_buf: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        attrs_d = {k.lower(): (v or "") for k, v in attrs}
        tag = tag.lower()
        self._tag_stack.append(tag)
        if tag in {"script", "style", "noscript"}:
            if tag == "script":
                self._in_script = True
                self._script_buf = []
            else:
                self._skip_depth += 1
        if tag == "a" and attrs_d.get("href"):
            self._current_link = urljoin(self.base_url, attrs_d.get("href") or "")
            self._current_link_text = []
        if re.fullmatch(r"h[1-6]", tag):
            self._current_heading_level = int(tag[1])
            self._current_heading_text = []
        if tag in {"p", "div", "section", "article", "li", "br", "tr", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag == "script" and self._in_script:
            self._in_script = False
            text = "".join(self._script_buf).strip()
            if text:
                self.scripts.append(text)
            self._script_buf = []
        elif tag in {"style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "a" and self._current_link:
            text = clean_text("".join(self._current_link_text))[:200]
            self.links.append({"url": self._current_link, "text": text})
            self._current_link = None
            self._current_link_text = []
        if re.fullmatch(r"h[1-6]", tag) and self._current_heading_level is not None:
            text = clean_text("".join(self._current_heading_text))[:200]
            if text:
                self.headings.append({"level": self._current_heading_level, "text": text})
            self._current_heading_level = None
            self._current_heading_text = []
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str):
        if self._in_script:
            self._script_buf.append(data)
            return
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._tag_stack and self._tag_stack[-1] == "title":
            self.title += text + " "
        self.parts.append(text + " ")
        if self._current_link is not None:
            self._current_link_text.append(text + " ")
        if self._current_heading_level is not None:
            self._current_heading_text.append(text + " ")

    def body_text(self) -> str:
        return clean_text("".join(self.parts))


def extract_urls_from_scripts(scripts: list[str], base_url: str) -> list[str]:
    urls: list[str] = []
    for script in scripts:
        # Full URLs in Next.js/Oopy data.
        for raw in re.findall(r"https?://[^\"'<>\\\s]+", script):
            urls.append(html_lib.unescape(raw).replace("\\/", "/"))
        # Oopy/Notion often stores paths/IDs rather than href anchors.
        for raw in re.findall(r"[\"'](/[^\"'<>\\\s]{4,})[\"']", script):
            urls.append(urljoin(base_url, raw.replace("\\/", "/")))
    return urls


def trafilatura_extract(html: str, url: str) -> str:
    try:
        import trafilatura
        extracted = trafilatura.extract(html, url=url, include_links=True, include_tables=True, include_formatting=True)
        return clean_text(extracted or "")
    except Exception:
        return ""


async def crawl4ai_extract(url: str) -> tuple[str, str, list[dict[str, str]], list[dict[str, Any]]]:
    """Optional Crawl4AI extraction. Falls back silently when unavailable/broken."""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    except Exception:
        return "", "", [], []
    try:
        browser_config = BrowserConfig(headless=True, verbose=False)
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
        markdown = clean_text(getattr(result, "markdown", "") or "")
        title = ""
        try:
            title = str((getattr(result, "metadata", {}) or {}).get("title") or "")
        except Exception:
            title = ""
        links: list[dict[str, str]] = []
        raw_links = getattr(result, "links", None)
        if isinstance(raw_links, dict):
            for group in raw_links.values():
                if isinstance(group, list):
                    for item in group:
                        if isinstance(item, dict) and item.get("href"):
                            links.append({"url": urljoin(url, str(item.get("href"))), "text": str(item.get("text") or "")[:200]})
        headings = [{"level": int(m.group(1)), "text": clean_text(m.group(2))[:200]} for m in re.finditer(r"^(#{1,6})\s+(.+)$", markdown, flags=re.M)]
        return title, markdown, links, headings
    except Exception:
        return "", "", [], []


@dataclass
class Node:
    label: str
    type: str
    title: str
    url: str
    text: str
    headings: list[dict[str, Any]]
    links: list[dict[str, str]]
    error: str = ""


def fetch_page_node(url: str, label: str, node_type: str = "page", use_crawl4ai: bool = True) -> Node:
    title = ""
    text = ""
    links: list[dict[str, str]] = []
    headings: list[dict[str, Any]] = []
    error = ""
    if use_crawl4ai:
        try:
            title_c4, md_c4, links_c4, headings_c4 = asyncio.run(crawl4ai_extract(url))
            if len(md_c4) > 600:
                return Node(label, node_type, title_c4, url, md_c4, headings_c4, links_c4)
        except Exception:
            pass
    try:
        html, ctype, final_url = fetch_text(url)
        parser = HTMLExtract(final_url)
        parser.feed(html)
        title = clean_text(parser.title)
        body = trafilatura_extract(html, final_url) or parser.body_text()
        links = parser.links
        links.extend({"url": u, "text": "structured_url"} for u in extract_urls_from_scripts(parser.scripts, final_url))
        headings = parser.headings
        text = body
        url = final_url
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return Node(label, node_type, title, url, text, headings, links, error=error)


def should_follow(seed_url: str, candidate: str, text: str, source_type: str) -> bool:
    if not candidate.startswith(("http://", "https://")):
        return False
    if MEDIA_EXT_RE.search(candidate):
        return False
    c_host = url_domain(candidate)
    s_host = url_domain(seed_url)
    hay = f"{candidate} {text}"
    if source_type in {"oopy", "wikidocs"} and c_host == s_host:
        return True
    if source_type == "ai_skills" and (c_host.endswith("microsoftlearning.github.io") or c_host.endswith("learn.microsoft.com") or c_host in {"youtube.com", "youtu.be", "www.youtube.com"}):
        return True
    if LEARNING_LINK_RE.search(hay) and (c_host == s_host or c_host.endswith("learn.microsoft.com") or c_host.endswith("microsoftlearning.github.io")):
        return True
    return False


def detect_source_type(url: str) -> str:
    host = url_domain(url)
    if "aiskillsnavigator.microsoft.com" in host:
        return "ai_skills"
    if "youtu.be" in host or "youtube.com" in host:
        return "youtube"
    if "oopy.io" in host or "notion.site" in host:
        return "oopy"
    if "wikidocs.net" in host:
        return "wikidocs"
    if "inflearn.com" in host or "udemy.com" in host:
        return "protected_course"
    return "web"


def collect_web_graph(seed_url: str, crawl_limit: int = 16, use_crawl4ai: bool = True) -> tuple[list[Node], dict[str, Any]]:
    source_type = detect_source_type(seed_url)
    nodes: list[Node] = []
    seen: set[str] = set()
    queue: list[tuple[str, str, str]] = [(seed_url, "seed", source_type)]
    while queue and len(nodes) < max(1, crawl_limit + 1):
        url, label, typ = queue.pop(0)
        norm = normalize_url(url)
        if norm in seen:
            continue
        seen.add(norm)
        print(f"Collecting {label}: {url}")
        node = fetch_page_node(url, label=label, node_type=typ, use_crawl4ai=use_crawl4ai)
        nodes.append(node)
        for link in node.links[:160]:
            u = normalize_url(link.get("url") or "")
            if u in seen:
                continue
            if should_follow(seed_url, u, link.get("text") or "", source_type):
                kind = "linked_lab" if re.search(r"lab|exercise|instructions", f"{u} {link.get('text','')}", re.I) else "linked_source"
                queue.append((u, f"{kind}_{len(queue)+1}", "lab" if kind == "linked_lab" else "page"))
            if len(queue) > crawl_limit * 4:
                queue = queue[: crawl_limit * 4]
    return nodes, {"source_type": source_type}


def transcript_with_youtube_api(vid: str, languages: list[str] | None = None) -> tuple[str, str]:
    languages = languages or ["ko", "en"]
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript_items = YouTubeTranscriptApi.get_transcript(vid, languages=languages)
        lines = []
        for item in transcript_items:
            start = float(item.get("start") or 0.0)
            mm = int(start // 60)
            ss = int(start % 60)
            lines.append(f"[{mm:02d}:{ss:02d}] {clean_text(str(item.get('text') or ''))}")
        return "\n".join(lines), "youtube-transcript-api"
    except Exception as exc:
        return "", f"youtube-transcript-api failed: {type(exc).__name__}: {exc}"


def ytdlp_metadata(url: str) -> tuple[dict[str, Any], str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--dump-json", "--skip-download", url],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.splitlines()[-1]), "yt-dlp"
        return {}, proc.stderr[-1000:]
    except Exception as exc:
        return {}, f"yt-dlp failed: {type(exc).__name__}: {exc}"


def collect_youtube(url: str) -> tuple[list[Node], dict[str, Any]]:
    vid = video_id(url)
    metadata, meta_status = ytdlp_metadata(url)
    title = str(metadata.get("title") or "")
    description = str(metadata.get("description") or "")
    chapters = metadata.get("chapters") if isinstance(metadata.get("chapters"), list) else []
    transcript = ""
    transcript_status = ""
    if vid:
        transcript, transcript_status = transcript_with_youtube_api(vid)
    if not transcript and isinstance(metadata.get("subtitles"), dict):
        transcript_status += "\nyt-dlp subtitles are present but not downloaded in this fast path."
    lines = ["[VIDEO_SOURCE]", f"url: {url}", f"video_id: {vid}", f"title: {title}", f"metadata_status: {meta_status}", f"transcript_status: {transcript_status}"]
    if description:
        lines.extend(["", "## Description", description[:6000]])
    if chapters:
        lines.extend(["", "## Chapters"])
        for ch in chapters[:80]:
            lines.append(f"- {ch.get('start_time')}: {ch.get('title')}")
    if transcript:
        lines.extend(["", "## Transcript", transcript[:80_000]])
    else:
        lines.extend(["", "## Transcript Missing", "자막/스크립트가 확보되지 않아 영상 내용 기반 글 생성은 quality gate에서 제한되어야 합니다."])
    node = Node("video_seed", "video", title, url, "\n".join(lines), [], [], error="" if transcript else transcript_status)
    return [node], {"source_type": "youtube", "video_id": vid, "transcript_chars": len(transcript)}


def collect_ai_skills_via_legacy(seed_url: str, out_dir: Path, args: argparse.Namespace) -> tuple[list[Node], dict[str, Any]]:
    legacy_script = Path(__file__).with_name("collect_source_pack.py")
    if not legacy_script.exists():
        return collect_web_graph(seed_url, crawl_limit=args.crawl_limit, use_crawl4ai=True)
    legacy_out = out_dir / "_legacy_collect"
    legacy_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(legacy_script),
        seed_url,
        "--headless",
        "--out",
        str(legacy_out),
        "--no-manual-pause",
        "--follow-labs",
        "--follow-limit",
        str(min(args.follow_limit, 8)),
        "--crawl-limit",
        str(min(args.crawl_limit, 12)),
        "--tree-limit",
        str(min(args.tree_limit, 24)),
        "--auto-login-wait",
        str(getattr(args, "auto_login_wait", 0) or 0),
    ]
    if args.user_data_dir:
        cmd.extend(["--user-data-dir", str(Path(args.user_data_dir) / "legacy")])
    print("Running AI Skills legacy profile as extractor fallback...")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max(180, args.timeout_seconds), check=False)
    md_files = sorted([p for p in legacy_out.glob("*.md") if not p.name.endswith(".report.md")], key=lambda p: p.stat().st_mtime, reverse=True)
    json_files = sorted(legacy_out.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if proc.returncode != 0 or not md_files:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:], file=sys.stderr)
        # If legacy fails, still try public/API/HTML graph.
        nodes, meta = collect_web_graph(seed_url, crawl_limit=args.crawl_limit, use_crawl4ai=True)
        meta.update({"legacy_status": "failed", "legacy_stderr_tail": proc.stderr[-2000:], "legacy_stdout_tail": proc.stdout[-2000:]})
        return nodes, meta
    legacy_payload: dict[str, Any] = {}
    if json_files:
        try:
            legacy_payload = json.loads(json_files[0].read_text(encoding="utf-8", errors="replace"))
        except Exception:
            legacy_payload = {}
    snapshots = legacy_payload.get("snapshots") if isinstance(legacy_payload.get("snapshots"), list) else []
    nodes: list[Node] = []
    for idx, snap in enumerate(snapshots[:80], start=1):
        if not isinstance(snap, dict):
            continue
        text = clean_text(str(snap.get("visible_text") or ""))[:35_000]
        headings = snap.get("headings") if isinstance(snap.get("headings"), list) else []
        links = snap.get("links") if isinstance(snap.get("links"), list) else []
        nodes.append(Node(
            label=str(snap.get("label") or f"ai_skills_node_{idx}"),
            type=str(snap.get("label") or "ai_skills"),
            title=str(snap.get("title") or "")[:300],
            url=str(snap.get("url") or ""),
            text=text,
            headings=headings[:30],
            links=links[:120],
            error=str(snap.get("error") or ""),
        ))
    # Free disk: compact output will be written by this v2 collector.
    try:
        shutil.rmtree(legacy_out)
    except Exception:
        pass
    meta = {
        "source_type": "ai_skills",
        "legacy_status": "ok",
        "legacy_stats": legacy_payload.get("stats", {}),
        "tree_items": legacy_payload.get("tree_items", []),
        "lab_url_candidates": legacy_payload.get("lab_url_candidates", []),
        "video_url_candidates": legacy_payload.get("video_url_candidates", []),
        "lesson_url_candidates": legacy_payload.get("lesson_url_candidates", []),
        "current_url": legacy_payload.get("current_url") or seed_url,
        "title": legacy_payload.get("title") or "AI Skills Navigator",
    }
    return nodes, meta


def build_quality(seed_url: str, nodes: list[Node], meta: dict[str, Any]) -> dict[str, Any]:
    text_chars = sum(len(n.text or "") for n in nodes)
    headings_count = sum(len(n.headings or []) for n in nodes)
    page_count = len(nodes)
    lab_count = sum(1 for n in nodes if re.search(r"lab|exercise|instructions|microsoftlearning", f"{n.label} {n.type} {n.url}", re.I))
    video_count = sum(1 for n in nodes if n.type == "video" or YOUTUBE_RE.search(n.url or "") or "youtube" in (n.text or "").lower())
    code_blocks = len(re.findall(r"```|\bdef\s+\w+\(|\bclass\s+\w+|\bSELECT\b|\bDAX\b|\bPowerShell\b", "\n".join(n.text for n in nodes), re.I))
    warnings: list[str] = []
    source_type = meta.get("source_type") or detect_source_type(seed_url)
    if text_chars < 3000:
        warnings.append("low_text_chars")
    if source_type in {"oopy", "wikidocs", "web"} and page_count < 2 and text_chars < 8000:
        warnings.append("shallow_web_collection")
    if source_type == "youtube" and int(meta.get("transcript_chars") or 0) < 2000:
        warnings.append("missing_or_short_video_transcript")
    if source_type == "ai_skills" and lab_count == 0 and headings_count < 6:
        warnings.append("ai_skills_missing_labs_or_lessons")
    return {
        "usable_text_chars": text_chars,
        "pages_collected": page_count,
        "headings_count": headings_count,
        "lab_count": lab_count,
        "video_count": video_count,
        "code_block_count": code_blocks,
        "warnings": warnings,
    }


def write_outputs(seed_url: str, nodes: list[Node], meta: dict[str, Any], out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    title = meta.get("title") or (nodes[0].title if nodes else "source_graph_v2") or "source_graph_v2"
    base = f"{ts}_{safe_name(title)}"
    md_path = out_dir / f"{base}.md"
    json_path = out_dir / f"{base}.json"
    report_path = out_dir / f"{base}.report.md"
    quality = build_quality(seed_url, nodes, meta)
    all_links: list[dict[str, str]] = []
    all_headings: list[dict[str, Any]] = []
    video_urls: list[str] = list(meta.get("video_url_candidates") or [])
    lab_urls: list[str] = list(meta.get("lab_url_candidates") or [])
    lesson_urls: list[str] = list(meta.get("lesson_url_candidates") or [])
    for n in nodes:
        all_links.extend(n.links[:120])
        all_headings.extend(n.headings[:40])
        for link in n.links:
            u = link.get("url") or ""
            if YOUTUBE_RE.search(u) and u not in video_urls:
                video_urls.append(u)
            if re.search(r"lab|exercise|instructions|microsoftlearning", f"{u} {link.get('text','')}", re.I) and u not in lab_urls:
                lab_urls.append(u)
            if re.search(r"lesson|module|learn", f"{u} {link.get('text','')}", re.I) and u not in lesson_urls:
                lesson_urls.append(u)
    stats = {
        "collector": "source_graph_collect_v2",
        "source_type": meta.get("source_type") or detect_source_type(seed_url),
        "page_count": len(nodes),
        "visible_text_chars": quality["usable_text_chars"],
        "link_count": len(all_links),
        "video_candidate_count": len(video_urls),
        "lesson_candidate_count": len(lesson_urls),
        "lab_candidate_count": len(lab_urls),
        "heading_count": len(all_headings),
    }
    payload = {
        "collector": "source_graph_collect_v2",
        "title": title,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "seed_url": seed_url,
        "current_url": meta.get("current_url") or (nodes[0].url if nodes else seed_url),
        "stats": stats,
        "quality": quality,
        "snapshots": [
            {
                "label": n.label,
                "type": n.type,
                "title": n.title,
                "url": n.url,
                "visible_text": n.text[:60_000],
                "headings": n.headings[:40],
                "links": n.links[:120],
                "error": n.error,
            }
            for n in nodes
        ],
        "lab_url_candidates": lab_urls[:120],
        "video_url_candidates": video_urls[:60],
        "lesson_url_candidates": lesson_urls[:120],
        "tree_items": meta.get("tree_items", [])[:120] if isinstance(meta.get("tree_items"), list) else [],
        "meta": meta,
    }
    lines = [
        f"# Source Pack: {title}",
        "",
        f"- Collector: source_graph_collect_v2",
        f"- Seed URL: {seed_url}",
        f"- Current URL: {payload['current_url']}",
        "",
        "## Collection Stats",
    ]
    for k, v in stats.items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Quality"])
    for k, v in quality.items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Learning Flow / Evidence Nodes"])
    for idx, n in enumerate(nodes, start=1):
        lines.extend([
            "",
            f"### {idx}. {n.title or n.label}",
            f"- type: {n.type}",
            f"- url: {n.url}",
            f"- text_chars: {len(n.text or '')}",
        ])
        if n.headings:
            lines.append("- headings: " + " | ".join(str(h.get("text") or h) for h in n.headings[:12]))
        if n.error:
            lines.append(f"- error: {n.error}")
        lines.extend(["", (n.text or "")[:45_000]])
    if lab_urls:
        lines.extend(["", "## Lab / Exercise URL Candidates"])
        lines.extend(f"- {u}" for u in lab_urls[:80])
    if video_urls:
        lines.extend(["", "## Video URL Candidates"])
        lines.extend(f"- {u}" for u in video_urls[:40])
    md_path.write_text("\n".join(lines).strip(), encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = [
        "# Source Graph Collector V2 Report",
        "",
        f"- seed_url: {seed_url}",
        f"- source_type: {stats['source_type']}",
        f"- pages_collected: {len(nodes)}",
        f"- usable_text_chars: {quality['usable_text_chars']}",
        f"- warnings: {', '.join(quality.get('warnings') or []) or 'none'}",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return md_path, json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--out", default="data/source_packs")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-manual-pause", action="store_true")
    parser.add_argument("--follow-labs", action="store_true")
    parser.add_argument("--follow-limit", type=int, default=8)
    parser.add_argument("--crawl-limit", type=int, default=16)
    parser.add_argument("--tree-limit", type=int, default=24)
    parser.add_argument("--user-data-dir", default="")
    parser.add_argument("--auto-login-wait", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=260)
    parser.add_argument("--no-crawl4ai", action="store_true")
    args, _unknown = parser.parse_known_args()

    seed_url = args.url
    out_dir = Path(args.out)
    source_type = detect_source_type(seed_url)
    started = time.perf_counter()
    if source_type == "protected_course":
        raise SystemExit("Protected course URL detected. Use browser-assisted capture mode rather than headless URL-only collection.")
    if source_type == "youtube":
        nodes, meta = collect_youtube(seed_url)
    elif source_type == "ai_skills":
        nodes, meta = collect_ai_skills_via_legacy(seed_url, out_dir, args)
    else:
        nodes, meta = collect_web_graph(seed_url, crawl_limit=args.crawl_limit, use_crawl4ai=not args.no_crawl4ai)
    meta["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    md_path, json_path, report_path = write_outputs(seed_url, nodes, meta, out_dir)
    print("\n수집 완료")
    print(f"Markdown: {md_path}")
    print(f"JSON: {json_path}")
    print(f"Report: {report_path}")
    q = build_quality(seed_url, nodes, meta)
    print(f"Pages collected: {len(nodes)}")
    print(f"Visible text chars: {q['usable_text_chars']}")
    print(f"Warnings: {q.get('warnings') or []}")


if __name__ == "__main__":
    main()
