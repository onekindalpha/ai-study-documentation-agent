#!/usr/bin/env python3
"""
Study Capture Copilot - Source Graph Collector v2

Goal:
- Do not write Medium drafts.
- Collect URL evidence into a compact source graph markdown + json.
- Prefer general extractor stack over hand-built site scraping.
- Let AI Skills Navigator continue to use the existing specialized collector.

Supported first-stage extractors:
- YouTube: youtube-transcript-api, yt-dlp metadata fallback
- General web/Oopy/WikiDocs/docs/labs: Crawl4AI if available, then Trafilatura, then BeautifulSoup
- Link following: same-origin doc links + known lab/doc/video links, with ordering and caps
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, parse_qs

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 StudyCaptureCopilot/2.0"
MAX_TEXT_PER_NODE = 14000
MAX_JSON_TEXT_PER_NODE = 4000


@dataclass
class Node:
    order: int
    type: str
    title: str
    url: str
    text: str
    headings: list[str]
    links: list[str]
    extractor: str
    quality: dict[str, Any]


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"https?://", "", text or "")
    text = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", text).strip("_")
    return (text[:max_len] or "source_graph")


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def domain(url: str) -> str:
    return (urlparse(url).netloc or "").lower().replace("www.", "")


def normalize_url(base: str, href: str) -> str:
    if not href:
        return ""
    href = href.strip()
    if href.startswith(("mailto:", "tel:", "javascript:", "data:")):
        return ""
    u = urljoin(base, href)
    p = urlparse(u)
    if not p.scheme.startswith("http"):
        return ""
    # Keep anchors for lab steps, but drop common tracking query params.
    return u.split("#", 1)[0] + (("#" + p.fragment) if p.fragment else "")


def is_youtube_url(url: str) -> bool:
    h = domain(url)
    return h in {"youtube.com", "youtu.be", "m.youtube.com"} or "youtube.com/embed/" in url


def youtube_video_id(url: str) -> str:
    p = urlparse(url)
    h = domain(url)
    if h == "youtu.be":
        return p.path.strip("/").split("/")[0]
    if "/embed/" in p.path:
        return p.path.split("/embed/", 1)[1].split("/", 1)[0]
    qs = parse_qs(p.query)
    return (qs.get("v") or [""])[0]


def is_ai_skills_url(url: str) -> bool:
    return "aiskillsnavigator.microsoft.com" in domain(url)


def is_known_lab_or_doc(url: str) -> bool:
    d = domain(url)
    l = url.lower()
    return (
        "microsoftlearning.github.io" in d
        or "learn.microsoft.com" in d
        or "/instructions/labs/" in l
        or "/labs/" in l
        or "exercise" in l
        or "lab" in l
        or "wikidocs.net" in d
        or "oopy.io" in d
        or "notion.site" in d
    )


def link_score(seed_url: str, link: str, anchor_text: str = "") -> int:
    sd = domain(seed_url)
    d = domain(link)
    l = link.lower()
    a = (anchor_text or "").lower()
    score = 0
    if d == sd:
        score += 8
    if is_known_lab_or_doc(link):
        score += 12
    if is_youtube_url(link):
        score += 12
    for kw in ["chapter", "lesson", "module", "lab", "exercise", "instructions", "summary", "transcript", "curriculum", "notion", "oopy", "wikidocs"]:
        if kw in l or kw in a:
            score += 4
    for bad in ["login", "signin", "privacy", "terms", "pricing", "share", "facebook", "twitter", "linkedin", "mailto"]:
        if bad in l:
            score -= 20
    if link.endswith(('.png','.jpg','.jpeg','.gif','.svg','.webp','.css','.js','.ico')):
        score -= 25
    return score


def fetch_html(url: str, timeout: int = 30) -> tuple[str, str]:
    if requests is None:
        raise RuntimeError("requests is not installed")
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.url, r.text


async def crawl4ai_markdown(url: str, timeout: int = 45) -> tuple[str, str, list[str], list[str]] | None:
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig  # type: ignore
    except Exception:
        return None
    try:
        browser_config = BrowserConfig(headless=True, verbose=False)
        run_config = CrawlerRunConfig(page_timeout=timeout * 1000)
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
        md = getattr(result, "markdown", None) or getattr(result, "cleaned_html", "") or ""
        final_url = getattr(result, "url", None) or url
        links: list[str] = []
        raw_links = getattr(result, "links", None)
        if isinstance(raw_links, dict):
            for group in raw_links.values():
                if isinstance(group, list):
                    for item in group:
                        href = item.get("href") if isinstance(item, dict) else str(item)
                        u = normalize_url(final_url, href)
                        if u:
                            links.append(u)
        headings = extract_headings_from_markdown(md)
        return final_url, clean_text(md), headings, sorted(set(links))
    except Exception:
        return None


def extract_headings_from_markdown(md: str) -> list[str]:
    heads = []
    for line in (md or "").splitlines():
        m = re.match(r"^#{1,4}\s+(.+)", line.strip())
        if m:
            heads.append(clean_text(m.group(1))[:180])
    return heads[:60]


def extract_with_trafilatura(url: str, html: str) -> tuple[str, list[str]] | None:
    try:
        import trafilatura  # type: ignore
    except Exception:
        return None
    try:
        text = trafilatura.extract(html, url=url, include_links=True, include_tables=True, output_format="markdown")
        if not text:
            return None
        return clean_text(text), extract_headings_from_markdown(text)
    except Exception:
        return None


def extract_with_bs4(url: str, html: str) -> tuple[str, str, list[str], list[tuple[str, str]]]:
    if BeautifulSoup is None:
        # Crude fallback
        title = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I|re.S)
        txt = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.I|re.S)
        txt = re.sub(r"<[^>]+>", " ", txt)
        return clean_text(title.group(1) if title else url), clean_text(txt), [], []
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "footer", "header", "nav"]):
        tag.decompose()
    title = clean_text((soup.title.get_text(" ", strip=True) if soup.title else "") or url)
    heads = [clean_text(h.get_text(" ", strip=True)) for h in soup.find_all(["h1", "h2", "h3", "h4"])]
    links: list[tuple[str, str]] = []
    for a in soup.find_all("a"):
        href = normalize_url(url, a.get("href") or "")
        if href:
            links.append((href, clean_text(a.get_text(" ", strip=True))))
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = clean_text(main.get_text("\n", strip=True))
    return title, text, heads[:80], links


def extract_next_data_text_and_links(base_url: str, html: str) -> tuple[str, list[str]]:
    """Oopy/Notion/Next.js often hides useful records in JSON scripts."""
    texts: list[str] = []
    links: list[str] = []
    if BeautifulSoup is None:
        scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.S|re.I)
    else:
        soup = BeautifulSoup(html, "html.parser")
        scripts = [s.get_text("", strip=False) for s in soup.find_all("script")]
    for s in scripts:
        if not s or ("__NEXT_DATA__" not in s and "recordMap" not in s and "notion" not in s.lower() and "oopy" not in s.lower()):
            continue
        # URLs hidden inside JSON/scripts
        for u in re.findall(r"https?://[^\"'\\\s<>]+", s):
            u = normalize_url(base_url, u)
            if u:
                links.append(u)
        for rel in re.findall(r"[\"'](/[^\"'<> ]{4,})[\"']", s):
            u = normalize_url(base_url, rel)
            if u:
                links.append(u)
        # Try JSON parse for __NEXT_DATA__ style content, but keep capped.
        m = re.search(r"({.*})", s, flags=re.S)
        if not m:
            continue
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue
        values: list[str] = []
        def walk(x: Any):
            if len("\n".join(values)) > 25000:
                return
            if isinstance(x, dict):
                for k, v in x.items():
                    if k in {"title", "plain_text", "text", "value", "name", "description"} and isinstance(v, str):
                        if 2 <= len(v.strip()) <= 1200:
                            values.append(v.strip())
                    walk(v)
            elif isinstance(x, list):
                for it in x[:2000]:
                    walk(it)
        walk(obj)
        if values:
            texts.append(clean_text("\n".join(dict.fromkeys(values))))
    return clean_text("\n\n".join(texts)), sorted(set(links))


def collect_page(url: str, order: int, node_type: str = "page") -> Node:
    final_url = url
    title = url
    text = ""
    headings: list[str] = []
    links: list[str] = []
    extractor = ""
    errors: list[str] = []

    # Crawl4AI first for JS-heavy pages if installed.
    try:
        c4 = asyncio.run(crawl4ai_markdown(url))
    except RuntimeError:
        c4 = None
    except Exception as e:
        c4 = None
        errors.append(f"crawl4ai: {type(e).__name__}: {e}")
    if c4 and len(c4[1]) > 500:
        final_url, text, headings, links = c4
        title = headings[0] if headings else final_url
        extractor = "crawl4ai"
    else:
        html = ""
        try:
            final_url, html = fetch_html(url)
        except Exception as e:
            errors.append(f"requests: {type(e).__name__}: {e}")
        if html:
            try:
                title, bs_text, bs_heads, bs_links = extract_with_bs4(final_url, html)
                links = [u for u, _ in bs_links]
                headings = bs_heads
                tri = extract_with_trafilatura(final_url, html)
                if tri and len(tri[0]) >= max(700, len(bs_text) * 0.35):
                    text, tri_heads = tri
                    headings = tri_heads or headings
                    extractor = "trafilatura"
                else:
                    text = bs_text
                    extractor = "beautifulsoup"
                next_text, next_links = extract_next_data_text_and_links(final_url, html)
                if next_text and len(next_text) > len(text) * 0.2:
                    text = clean_text(text + "\n\n[Hidden Next/Oopy/Notion data]\n" + next_text)
                    extractor += "+next_json"
                links = sorted(set(links + next_links))
            except Exception as e:
                errors.append(f"extract: {type(e).__name__}: {e}")

    q = {"text_chars": len(text), "heading_count": len(headings), "link_count": len(links)}
    if errors:
        q["errors"] = errors[-5:]
    return Node(
        order=order,
        type=node_type,
        title=clean_text(title)[:220],
        url=final_url,
        text=clean_text(text)[:MAX_TEXT_PER_NODE],
        headings=headings[:60],
        links=links[:200],
        extractor=extractor or "none",
        quality=q,
    )


def collect_youtube(url: str, order: int) -> Node:
    vid = youtube_video_id(url)
    title = f"YouTube video {vid}" if vid else url
    transcript_text = ""
    desc = ""
    meta: dict[str, Any] = {}
    errors: list[str] = []
    if vid:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
            parts = YouTubeTranscriptApi.get_transcript(vid, languages=["ko", "en"])
            lines = []
            for p in parts:
                start = float(p.get("start", 0.0))
                mm = int(start // 60)
                ss = int(start % 60)
                lines.append(f"[{mm:02d}:{ss:02d}] {p.get('text','')}")
            transcript_text = clean_text("\n".join(lines))
        except Exception as e:
            errors.append(f"youtube_transcript_api: {type(e).__name__}: {e}")
        try:
            import yt_dlp  # type: ignore
            ydl_opts = {"quiet": True, "skip_download": True, "extract_flat": False}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                meta = ydl.extract_info(url, download=False) or {}
            title = meta.get("title") or title
            desc = meta.get("description") or ""
            chapters = meta.get("chapters") or []
            if chapters:
                ch_lines = ["[Chapters]"]
                for ch in chapters[:80]:
                    st = int(ch.get("start_time") or 0)
                    ch_lines.append(f"[{st//60:02d}:{st%60:02d}] {ch.get('title','')}")
                desc = clean_text(desc + "\n\n" + "\n".join(ch_lines))
        except Exception as e:
            errors.append(f"yt_dlp: {type(e).__name__}: {e}")
    text = clean_text("\n\n".join(x for x in [desc, "[Transcript]\n" + transcript_text if transcript_text else ""] if x))
    headings = [title]
    q = {
        "video_id": vid,
        "transcript_chars": len(transcript_text),
        "description_chars": len(desc),
        "text_chars": len(text),
        "errors": errors[-5:],
    }
    return Node(order=order, type="video", title=title[:220], url=url, text=text[:MAX_TEXT_PER_NODE], headings=headings, links=[], extractor="youtube", quality=q)


def should_follow(seed_url: str, link: str, anchor: str = "") -> bool:
    score = link_score(seed_url, link, anchor)
    return score >= 8


def collect_graph(seed_url: str, max_pages: int = 24, max_depth: int = 2) -> tuple[list[Node], dict[str, Any]]:
    if is_ai_skills_url(seed_url):
        raise RuntimeError("AI Skills Navigator should use the existing specialized Playwright/API collector first; v2 is for web/video/doc fallback.")

    seen: set[str] = set()
    queue: list[tuple[str, int, str]] = [(seed_url, 0, "seed_video" if is_youtube_url(seed_url) else "seed_page")]
    nodes: list[Node] = []
    order = 1
    while queue and len(nodes) < max_pages:
        url, depth, ntype = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            if is_youtube_url(url):
                node = collect_youtube(url, order)
            else:
                node = collect_page(url, order, ntype)
        except Exception as e:
            node = Node(order=order, type="error", title=url, url=url, text="", headings=[], links=[], extractor="error", quality={"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})
        nodes.append(node)
        order += 1
        if depth >= max_depth:
            continue
        # Follow links in score order.
        scored = []
        for link in node.links:
            if link in seen:
                continue
            if should_follow(seed_url, link):
                scored.append((link_score(seed_url, link), link))
        for _, link in sorted(scored, reverse=True)[:12]:
            if len(queue) + len(nodes) >= max_pages:
                break
            ntype2 = "linked_video" if is_youtube_url(link) else ("linked_lab" if is_known_lab_or_doc(link) else "linked_page")
            queue.append((link, depth + 1, ntype2))
    stats = build_stats(seed_url, nodes)
    return nodes, stats


def build_stats(seed_url: str, nodes: list[Node]) -> dict[str, Any]:
    all_text = "\n".join(n.text for n in nodes)
    return {
        "seed_url": seed_url,
        "source_type": detect_source_type(seed_url),
        "page_count": len([n for n in nodes if "page" in n.type or "lab" in n.type or "seed" in n.type]),
        "video_count": len([n for n in nodes if "video" in n.type]),
        "visible_text_chars": len(all_text),
        "heading_count": sum(len(n.headings) for n in nodes),
        "link_count": sum(len(n.links) for n in nodes),
        "lab_count": len([n for n in nodes if "lab" in n.type or "lab" in n.url.lower() or "exercise" in n.url.lower()]),
        "transcript_chars": sum(int(n.quality.get("transcript_chars") or 0) for n in nodes),
        "extractors": sorted(set(n.extractor for n in nodes)),
    }


def detect_source_type(url: str) -> str:
    d = domain(url)
    if is_youtube_url(url):
        return "youtube"
    if "oopy.io" in d or "notion.site" in d:
        return "oopy_notion"
    if "wikidocs.net" in d:
        return "wikidocs"
    if "microsoftlearning.github.io" in d or "learn.microsoft.com" in d:
        return "learn_lab_doc"
    return "web"


def markdown_report(seed_url: str, run_id: str, nodes: list[Node], stats: dict[str, Any]) -> str:
    lines = []
    lines.append("# Source Graph v2 Pack")
    lines.append("")
    lines.append(f"seed_url: {seed_url}")
    lines.append(f"run_id: {run_id}")
    lines.append(f"source_type: {stats.get('source_type')}")
    lines.append("")
    lines.append("## Source Graph Stats")
    for k, v in stats.items():
        if k != "seed_url":
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Learning Flow Nodes")
    for n in nodes:
        lines.append(f"\n### {n.order}. [{n.type}] {n.title}")
        lines.append(f"- url: {n.url}")
        lines.append(f"- extractor: {n.extractor}")
        lines.append(f"- text_chars: {len(n.text)}")
        if n.headings:
            lines.append("- headings: " + " | ".join(n.headings[:12]))
        lines.append("")
        if n.text:
            lines.append(n.text[:MAX_TEXT_PER_NODE])
        else:
            lines.append("(no usable text extracted)")
    return "\n".join(lines).strip() + "\n"


def compact_json(nodes: list[Node], stats: dict[str, Any], seed_url: str, run_id: str) -> dict[str, Any]:
    jnodes = []
    for n in nodes:
        d = asdict(n)
        d["text"] = d.get("text", "")[:MAX_JSON_TEXT_PER_NODE]
        d["links"] = d.get("links", [])[:80]
        jnodes.append(d)
    return {"ok": True, "collector": "source_graph_v2", "seed_url": seed_url, "run_id": run_id, "stats": stats, "nodes": jnodes}


def quality_ok(stats: dict[str, Any]) -> bool:
    stype = stats.get("source_type")
    if stype == "youtube":
        return int(stats.get("transcript_chars") or 0) >= 1200 or int(stats.get("visible_text_chars") or 0) >= 2500
    if stype in {"oopy_notion", "wikidocs", "web", "learn_lab_doc"}:
        return int(stats.get("visible_text_chars") or 0) >= 3000 and (int(stats.get("page_count") or 0) >= 1)
    return int(stats.get("visible_text_chars") or 0) >= 3000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default="data/source_packs")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--max-pages", type=int, default=24)
    ap.add_argument("--max-depth", type=int, default=2)
    args = ap.parse_args()

    seed_url = args.url.strip()
    run_id = args.run_id or f"run_{now_stamp()}"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    try:
        nodes, stats = collect_graph(seed_url, max_pages=args.max_pages, max_depth=args.max_depth)
        stats["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        stats["quality_ok"] = quality_ok(stats)
        title_slug = slugify(nodes[0].title if nodes else seed_url)
        base = out_dir / f"{now_stamp()}_{title_slug}_source_graph_v2"
        md_path = base.with_suffix(".md")
        json_path = base.with_suffix(".json")
        md = markdown_report(seed_url, run_id, nodes, stats)
        data = compact_json(nodes, stats, seed_url, run_id)
        md_path.write_text(md, encoding="utf-8")
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"SOURCE_GRAPH_V2_MD={md_path}")
        print(f"SOURCE_GRAPH_V2_JSON={json_path}")
        print(json.dumps(stats, ensure_ascii=False))
        if not stats["quality_ok"]:
            print("SOURCE_GRAPH_V2_QUALITY=insufficient", file=sys.stderr)
            sys.exit(4)
    except Exception as e:
        print(f"SOURCE_GRAPH_V2_ERROR={type(e).__name__}: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
