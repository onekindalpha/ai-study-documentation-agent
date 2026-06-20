from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse


VIDEO_PATTERNS = re.compile(
    r"(youtube\.com|youtu\.be|vimeo\.com|learn-video\.azurefd\.net|microsoftstream|stream|video|watch|player|embed|vod|m3u8|hls|inflearn\.com/courses/lecture|udemy\.com/.*/learn/lecture)",
    re.IGNORECASE,
)

LESSON_PATTERNS = re.compile(
    r"(lesson|lecture|curriculum|module|unit|unitId|course|courseId|playlist|player|learn|training|video|watch|lab|exercise|assignment|resources|instructions|microsoftlearning\.github\.io)",
    re.IGNORECASE,
)

LAB_PATTERNS = re.compile(
    r"(microsoftlearning\.github\.io|Instructions/(Labs|Exercises)|Launch Exercise|\b(lab|labs|exercise|exercises|instructions)\b|실습|랩)",
    re.IGNORECASE,
)

TREE_LAB_PATTERNS = re.compile(r"(lab|exercise|instructions|실습|랩)", re.IGNORECASE)

LAB_ACTION_PATTERNS = re.compile(
    r"(launch the exercise|launch exercise|launch lab|start exercise|start lab|open exercise|open lab|begin exercise|begin lab|instructions|view instructions|exercise|lab|실습 시작|실습|랩)",
    re.IGNORECASE,
)

VIDEO_ACTION_PATTERNS = re.compile(
    r"\b(start session|watch session|watch video|play video|start video|open session|view session)\b",
    re.IGNORECASE,
)

AI_SKILLS_PLAYER_HOST = "aiskillsnavigator.microsoft.com"
AI_SKILLS_PLAYER_PATH = "/player"

CONTENT_LINK_PATTERNS = re.compile(
    r"(article|blog|docs|learn|training|module|unit|unitId|lesson|lecture|curriculum|course|courseId|playlist|player|video|watch|lab|exercise|assignment|resources|instructions|quickstart|tutorial)",
    re.IGNORECASE,
)

AI_SKILLS_API_PATTERNS = re.compile(
    r"api\.projectorono\.microsoft\.com/(skillingplans|content/v1/LearningObjects/Catalog)",
    re.IGNORECASE,
)

EXPAND_BUTTON_PATTERNS = re.compile(
    r"^(summary|transcript|show summary|show transcript|show more|show details|details|overview|resources|expand|요약|자막|더 보기)$",
    re.IGNORECASE,
)


def load_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Python package 'playwright' is required for browser collection.\n"
            "Install once with:\n"
            "  python -m pip install playwright\n"
            "  python -m playwright install chromium\n"
            "Then rerun this collector from the final AI Skills Navigator player URL."
        ) from exc
    return sync_playwright


def clean_text(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def unique_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if not item:
            continue
        key = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def flatten_json_for_source_text(value, lines: list[str], label: str = "") -> None:
    if isinstance(value, dict):
        for key in ["title", "name", "description", "summary", "url", "sourceUrl", "contentUrl", "videoUrl"]:
            item = value.get(key)
            if isinstance(item, str) and clean_text(item):
                prefix = f"{label} {key}".strip()
                lines.append(f"{prefix}: {clean_text(item)}")
        for key, item in value.items():
            if key in {"title", "name", "description", "summary", "url", "sourceUrl", "contentUrl", "videoUrl"}:
                continue
            flatten_json_for_source_text(item, lines, key)
    elif isinstance(value, list):
        for item in value:
            flatten_json_for_source_text(item, lines, label)
    elif isinstance(value, str):
        text = clean_text(value)
        if text and (len(text) > 24 or re.search(r"https?://|module-|video-|exercise|lab|unit|lesson", text, re.IGNORECASE)):
            prefix = f"{label}: " if label else ""
            lines.append(f"{prefix}{text}")


def extract_urls_from_json(value) -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            urls.extend(extract_urls_from_json(item))
    elif isinstance(value, list):
        for item in value:
            urls.extend(extract_urls_from_json(item))
    elif isinstance(value, str):
        urls.extend(re.findall(r"https?://[^\s\)\]\}<>\"']+", value))
    return unique_keep_order(urls)


def extract_markdown_links(text: str) -> list[dict]:
    links: list[dict] = []
    image_link_pattern = re.compile(r"\[!\[([^\]]*)\]\((https?://[^)\s]+)\)\]\((https?://[^)\s]+)\)")
    consumed_spans: list[tuple[int, int]] = []
    for match in image_link_pattern.finditer(text or ""):
        label = clean_text(match.group(1))
        image_url = match.group(2)
        url = match.group(3)
        consumed_spans.append(match.span())
        links.append(
            {
                "text": label[:300],
                "url": url,
                "kind": classify_url(f"{url} {label}"),
            }
        )
        if not is_media_asset_url(image_url):
            links.append(
                {
                    "text": label[:300],
                    "url": image_url,
                    "kind": classify_url(f"{image_url} {label}"),
                }
            )

    pattern = re.compile(r"!?\[([^\]]*)\]\((https?://[^)\s]+)\)")
    for match in pattern.finditer(text or ""):
        if any(start <= match.start() < end for start, end in consumed_spans):
            continue
        label = clean_text(match.group(1))
        url = match.group(2)
        if is_media_asset_url(url):
            continue
        links.append(
            {
                "text": label[:300],
                "url": url,
                "kind": classify_url(f"{url} {label}"),
            }
        )
    return unique_keep_order(links)


def api_response_to_snapshot(url: str, payload) -> dict:
    lines: list[str] = []
    flatten_json_for_source_text(payload, lines)
    visible_text = clean_text("\n".join(unique_keep_order(lines)))
    all_urls = extract_urls_from_json(payload)
    markdown_links = extract_markdown_links(visible_text)
    all_urls = unique_keep_order(all_urls + [item["url"] for item in markdown_links])
    title = ""
    if isinstance(payload, dict):
        title = payload.get("title") or payload.get("name") or "AI Skills API response"
    return {
        "label": "ai_skills_api_response",
        "title": title,
        "url": url,
        "visible_text": visible_text,
        "links": unique_keep_order(markdown_links + [
            {"text": "", "url": item, "kind": classify_url(item)}
            for item in all_urls
            if not is_media_asset_url(item)
        ]),
        "iframe_urls": [],
        "video_src_urls": [],
        "script_url_candidates": [],
        "frame_texts": [],
        "all_url_candidates": all_urls,
        "headings": [],
        "expanded_sections": [],
        "video_items": [],
        "lesson_items": [],
        "lab_items": [],
    }


def attach_ai_skills_api_capture(page, snapshots: list[dict]) -> None:
    seen: set[str] = set()

    def capture_response(response) -> None:
        try:
            if response.status != 200 or not AI_SKILLS_API_PATTERNS.search(response.url):
                return
            if response.url in seen:
                return
            content_type = response.headers.get("content-type", "")
            if "json" not in content_type.lower():
                return
            text = response.text()
            payload = json.loads(text)
            snapshots.append(api_response_to_snapshot(response.url, payload))
            seen.add(response.url)
        except Exception:
            return

    page.on("response", capture_response)


def is_ai_skills_player_url(url: str) -> bool:
    parsed = urlparse(url)
    return AI_SKILLS_PLAYER_HOST in parsed.netloc and parsed.path.startswith(AI_SKILLS_PLAYER_PATH)


def same_origin(url_a: str, url_b: str) -> bool:
    a = urlparse(url_a)
    b = urlparse(url_b)
    return (a.scheme, a.netloc) == (b.scheme, b.netloc)


def normalize_url_for_visit(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()


def is_media_asset_url(url: str) -> bool:
    path = urlparse(url or "").path.lower()
    return bool(re.search(r"\.(png|jpe?g|gif|webp|svg|ico|css|woff2?|ttf|map)$", path))


def should_collect_link(seed_url: str, candidate_url: str, text: str = "") -> bool:
    if not candidate_url.startswith(("http://", "https://")):
        return False
    if is_media_asset_url(candidate_url):
        return False
    haystack = f"{candidate_url} {text}"
    if LAB_PATTERNS.search(haystack):
        return True
    host = (urlparse(seed_url).netloc or "").lower()
    if same_origin(seed_url, candidate_url) and any(token in host for token in ["oopy.io", "wikidocs.net"]):
        return True
    if same_origin(seed_url, candidate_url) and CONTENT_LINK_PATTERNS.search(haystack):
        return True
    return False


def looks_like_login_or_access_page(url: str, title: str = "", text: str = "") -> bool:
    lowered_url = (url or "").lower()
    lowered_title = (title or "").lower()
    lowered_text = clean_text(text or "").lower()
    auth_url_tokens = [
        "login",
        "signin",
        "sign-in",
        "microsoftonline",
        "oauth",
        "authorize",
        "auth",
        "accounts.google",
    ]
    if any(token in lowered_url for token in auth_url_tokens):
        return True
    title_tokens = ["sign in to your account", "login", "sign in", "로그인", "access denied"]
    if any(token in lowered_title for token in title_tokens):
        return True
    access_tokens = [
        "sign in to your account",
        "pick an account",
        "enter password",
        "로그인하세요",
        "로그인이 필요",
        "access denied",
        "permission denied",
        "권한이 필요",
    ]
    return any(token in lowered_text for token in access_tokens)


def maybe_pause_for_login_or_confirmation(page, seed_url: str, manual_pause: bool, auto_login_wait: int = 0) -> None:
    try:
        current_url = page.url
        title = page.title()
        body_text = clean_text(page.locator("body").inner_text(timeout=3000)).lower()
    except Exception:
        current_url = page.url
        title = ""
        body_text = ""

    loginish = looks_like_login_or_access_page(current_url, title, body_text)
    if loginish:
        print("\nManual login may be required.")
        print("- Complete login in the opened browser.")
        print("- Do not share credentials with this collector.")
        print("- If redirected away from the source, paste this original URL back into the address bar:")
        print(f"  {seed_url}")
        if manual_pause:
            input("After the source page is visible again, press Enter: ")
        else:
            print("- Non-interactive mode: continuing without waiting for manual login.")
            if auto_login_wait > 0:
                print(f"- Waiting {auto_login_wait}s for manual login/session restore...")
                page.wait_for_timeout(auto_login_wait * 1000)
                try:
                    if looks_like_login_or_access_page(page.url, page.title(), page.locator("body").inner_text(timeout=3000)):
                        page.goto(seed_url, wait_until="domcontentloaded", timeout=60000)
                        page.wait_for_timeout(2500)
                except Exception:
                    pass

    if manual_pause:
        print("\nManual confirmation step:")
        print("1) Confirm this is the representative source page you want to collect.")
        print("2) If the site requires login, complete it manually in the browser.")
        print("3) If needed, manually open any lesson/summary/curriculum area you want visible.")
        input("When ready to collect, press Enter: ")


def wait_for_source_ready(page, timeout_ms: int = 30000) -> None:
    deadline = time.time() + (timeout_ms / 1000)
    while time.time() < deadline:
        try:
            body = clean_text(page.locator("body").inner_text(timeout=1500))
            tree_count = page.locator("[role=treeitem]").count()
            if tree_count > 0:
                return
            if len(body) > 800 and "Loading content..." not in body[:120]:
                return
        except Exception:
            pass
        page.wait_for_timeout(750)


def expand_summary_sections(page) -> list[str]:
    clicked: list[str] = []
    selectors = [
        "button[aria-expanded]",
        "button[aria-controls]",
        "[role=button][aria-expanded]",
        "[role=button][aria-controls]",
        "summary",
        "details summary",
    ]
    for _round in range(3):
        changed = False
        for selector in selectors:
            try:
                handles = page.locator(selector).all()
            except Exception:
                continue
            for handle in handles[:80]:
                try:
                    text = clean_text(handle.inner_text(timeout=800))
                    if not text or not EXPAND_BUTTON_PATTERNS.search(text):
                        continue
                    aria_expanded = handle.get_attribute("aria-expanded", timeout=500)
                    if aria_expanded == "true" and "show more" not in text.lower() and "더 보기" not in text:
                        continue
                    if not handle.is_visible(timeout=500):
                        continue
                    handle.click(timeout=1200)
                    page.wait_for_timeout(350)
                    clicked.append(text[:120])
                    changed = True
                except Exception:
                    continue
        if not changed:
            break
    return unique_keep_order(clicked)


def extract_headings(page) -> list[dict]:
    try:
        return page.eval_on_selector_all(
            "h1,h2,h3,h4,h5,h6,[role=heading]",
            """
            els => els.map(el => ({
                level: el.tagName && /^H[1-6]$/.test(el.tagName) ? Number(el.tagName.slice(1)) : null,
                text: (el.innerText || el.textContent || '').trim()
            })).filter(x => x.text)
            """,
        )
    except Exception:
        return []


def extract_candidate_items(page, current_url: str) -> dict[str, list[dict]]:
    try:
        raw_items = page.eval_on_selector_all(
            "a[href], button, [role=button], [role=listitem], [role=treeitem], li",
            """
            els => els.map(el => ({
                tag: el.tagName,
                role: el.getAttribute('role') || '',
                text: (el.innerText || el.textContent || '').trim(),
                href: el.href || el.getAttribute('href') || ''
            })).filter(x => x.text || x.href)
            """,
        )
    except Exception:
        raw_items = []
    video_items: list[dict] = []
    lesson_items: list[dict] = []
    lab_items: list[dict] = []
    for item in raw_items:
        text = clean_text(item.get("text") or "")
        href = item.get("href") or ""
        if href:
            href = urljoin(current_url, href)
        haystack = f"{text} {href}"
        normalized = {
            "text": text[:300],
            "url": href,
            "tag": item.get("tag") or "",
            "role": item.get("role") or "",
        }
        if VIDEO_PATTERNS.search(haystack) or VIDEO_ACTION_PATTERNS.search(text):
            video_items.append(normalized)
        if LESSON_PATTERNS.search(haystack):
            lesson_items.append(normalized)
        if LAB_PATTERNS.search(haystack):
            if is_media_asset_url(href):
                continue
            lab_items.append(normalized)
    return {
        "video_items": unique_keep_order(video_items),
        "lesson_items": unique_keep_order(lesson_items),
        "lab_items": unique_keep_order(lab_items),
    }


def open_player_navigation(page) -> list[str]:
    clicked: list[str] = []
    for label in ["Show navigation", "Navigation", "목차", "탐색"]:
        try:
            target = page.get_by_role("button", name=label, exact=True)
            if target.count() != 1:
                continue
            expanded = target.get_attribute("aria-expanded", timeout=500)
            if expanded == "true":
                return clicked
            target.click(timeout=3000)
            page.wait_for_timeout(700)
            clicked.append(label)
            return clicked
        except Exception:
            continue
    return clicked


def get_tree_items(page) -> list[dict]:
    try:
        raw_items = page.eval_on_selector_all(
            "[role=treeitem]",
            """
            els => els.map((el, idx) => ({
                index: idx,
                text: (el.innerText || el.textContent || '').trim(),
                selected: el.getAttribute('aria-selected') || '',
                expanded: el.getAttribute('aria-expanded') || '',
                level: el.getAttribute('aria-level') || '',
                role: el.getAttribute('role') || ''
            })).filter(x => x.text)
            """,
        )
    except Exception:
        raw_items = []

    items = []
    for item in raw_items:
        text = clean_text(item.get("text") or "")
        if not text:
            continue
        item["text"] = text[:500]
        items.append(item)
    return unique_keep_order(items)


def expand_collapsed_tree_items(page, rounds: int = 4) -> list[str]:
    expanded: list[str] = []
    for _round in range(rounds):
        changed = False
        try:
            collapsed = page.locator('[role="treeitem"][aria-expanded="false"]')
            count = collapsed.count()
        except Exception:
            break
        if count < 1:
            break
        for idx in range(count):
            try:
                item = collapsed.nth(idx)
                text = clean_text(item.inner_text(timeout=800))
                item.scroll_into_view_if_needed(timeout=1500)
                item.focus(timeout=1500)
                page.keyboard.press("ArrowRight")
                page.wait_for_timeout(450)
                expanded.append(text[:160])
                changed = True
            except Exception:
                continue
        if not changed:
            break
    return unique_keep_order(expanded)


def click_tree_item_by_text(page, text: str) -> bool:
    text = clean_text(text)
    if not text:
        return False
    exact_pattern = re.compile(rf"^\s*{re.escape(text)}\s*$")
    try:
        candidates = page.locator("[role=treeitem]").filter(has_text=exact_pattern)
        for idx in range(candidates.count() - 1, -1, -1):
            candidate = candidates.nth(idx)
            if clean_text(candidate.inner_text(timeout=800)) != text:
                continue
            candidate.scroll_into_view_if_needed(timeout=1500)
            candidate.click(timeout=3000)
            return True
    except Exception:
        pass
    try:
        return bool(
            page.evaluate(
                """
                targetText => {
                    const normalize = value => (value || '').replace(/\\s+/g, ' ').trim();
                    const items = Array.from(document.querySelectorAll('[role="treeitem"]'));
                    const target = items.find(el => normalize(el.innerText || el.textContent) === targetText);
                    if (!target) return false;
                    target.scrollIntoView({ block: 'center', inline: 'nearest' });
                    const clickable = target.querySelector('a,button,[role="button"],[tabindex]') || target;
                    clickable.click();
                    return true;
                }
                """,
                text,
            )
        )
    except Exception:
        return False


def collect_tree_item_snapshots(context, page, limit: int = 24) -> tuple[list[dict], list[dict]]:
    snapshots: list[dict] = []
    collected_items: list[dict] = []
    seen_action_signatures: set[str] = set()
    open_player_navigation(page)
    expand_collapsed_tree_items(page)

    seen_texts: set[str] = set()
    for _round in range(3):
        expand_collapsed_tree_items(page)
        items = get_tree_items(page)
        new_items = [
            item for item in items
            if item.get("text") not in seen_texts
            and 0 < len(item.get("text") or "") <= 220
            and not item.get("expanded")
        ]
        if not new_items:
            break
        collected_items.extend(new_items)
        for item in new_items:
            text = item.get("text") or ""
            seen_texts.add(text)
            if len(snapshots) >= limit:
                break
            try:
                open_player_navigation(page)
                if not click_tree_item_by_text(page, text):
                    continue
                page.wait_for_timeout(1200)
                expanded_sections = expand_summary_sections(page)
                auto_scroll(page, steps=6, delay_ms=350)
                label = "tree_lab" if TREE_LAB_PATTERNS.search(text) else "tree_item"
                snapshot = collect_page_snapshot(
                    page,
                    label=f"{label}_{len(snapshots) + 1}",
                    expanded_sections=expanded_sections,
                )
                snapshot["tree_item_text"] = text
                snapshots.append(snapshot)
                if TREE_LAB_PATTERNS.search(text) and len(snapshots) < limit:
                    snapshots.extend(
                        collect_lab_action_snapshots(
                            context,
                            [page],
                            limit=max(0, limit - len(snapshots)),
                        )
                    )
                if len(snapshots) < limit:
                    snapshots.extend(
                        collect_video_action_snapshots(
                            context,
                            [page],
                            limit=max(0, limit - len(snapshots)),
                            seen_signatures=seen_action_signatures,
                        )
                    )
                if text.strip().lower() == "summary" and len(snapshots) < limit:
                    snapshots.extend(
                        collect_next_button_snapshots(
                            page,
                            limit=min(4, max(0, limit - len(snapshots))),
                        )
                    )
                open_player_navigation(page)
            except Exception:
                continue
        if len(snapshots) >= limit:
            break

    return snapshots, unique_keep_order(collected_items)


def collect_lab_action_snapshots(context, source_pages: list, limit: int = 8) -> list[dict]:
    snapshots: list[dict] = []
    seen_urls: set[str] = set()
    for page in source_pages:
        if len(snapshots) >= limit:
            break
        try:
            source_text = clean_text(page.locator("body").inner_text(timeout=3000))
        except Exception:
            source_text = ""
        if not TREE_LAB_PATTERNS.search(source_text):
            continue

        candidates = []
        for selector in ["a[href]", "button", "[role=button]"]:
            try:
                handles = page.locator(selector).all()
            except Exception:
                continue
            for handle in handles[:80]:
                try:
                    text = clean_text(handle.inner_text(timeout=500))
                    href = handle.get_attribute("href", timeout=500) or ""
                    if text and LAB_ACTION_PATTERNS.search(text):
                        candidates.append((selector, text, href, handle))
                    elif href and LAB_PATTERNS.search(href + " " + text):
                        candidates.append((selector, text, href, handle))
                except Exception:
                    continue

        for _selector, text, href, handle in candidates:
            if len(snapshots) >= limit:
                break
            try:
                before_pages = set(context.pages)
                before_url = page.url
                handle.scroll_into_view_if_needed(timeout=1500)
                handle.click(timeout=3000)
                page.wait_for_timeout(2500)
                new_pages = [candidate for candidate in context.pages if candidate not in before_pages]
                target_page = new_pages[-1] if new_pages else page
                if new_pages:
                    target_page.wait_for_load_state("domcontentloaded", timeout=30000)
                    target_page.wait_for_timeout(1500)
                normalized = normalize_url_for_visit(target_page.url)
                if normalized in seen_urls:
                    continue
                seen_urls.add(normalized)
                expanded_sections = expand_summary_sections(target_page)
                auto_scroll(target_page, steps=8, delay_ms=400)
                snapshot = collect_page_snapshot(
                    target_page,
                    label=f"lab_action_{len(snapshots) + 1}",
                    expanded_sections=expanded_sections,
                )
                snapshot["clicked_action_text"] = text
                snapshot["clicked_action_href"] = href
                snapshots.append(snapshot)
                if target_page == page and normalize_url_for_visit(page.url) != normalize_url_for_visit(before_url):
                    page.go_back(wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_timeout(1000)
            except Exception:
                continue
    return snapshots


def collect_video_action_snapshots(
    context,
    source_pages: list,
    limit: int = 8,
    seen_signatures: set[str] | None = None,
) -> list[dict]:
    snapshots: list[dict] = []
    if seen_signatures is None:
        seen_signatures = set()
    for page in source_pages:
        if len(snapshots) >= limit:
            break

        candidates = []
        for selector in ["a[href]", "button", "[role=button]"]:
            try:
                handles = page.locator(selector).all()
            except Exception:
                continue
            for handle in handles[:80]:
                try:
                    text = clean_text(handle.inner_text(timeout=500))
                    href = handle.get_attribute("href", timeout=500) or ""
                    aria = handle.get_attribute("aria-label", timeout=500) or ""
                    title = handle.get_attribute("title", timeout=500) or ""
                    action_text = clean_text(" ".join([text, aria, title]))
                    if len(action_text) > 180:
                        continue
                    if VIDEO_ACTION_PATTERNS.search(action_text) or (
                        href and VIDEO_PATTERNS.search(href)
                    ):
                        signature = f"{selector}|{action_text}|{href}"
                        if signature in seen_signatures:
                            continue
                        seen_signatures.add(signature)
                        candidates.append((selector, action_text or text, href, handle))
                except Exception:
                    continue

        for _selector, text, href, handle in candidates:
            if len(snapshots) >= limit:
                break
            try:
                before_pages = set(context.pages)
                before_url = page.url
                handle.scroll_into_view_if_needed(timeout=1500)
                handle.click(timeout=3000)
                page.wait_for_timeout(2500)
                new_pages = [candidate for candidate in context.pages if candidate not in before_pages]
                target_page = new_pages[-1] if new_pages else page
                if new_pages:
                    target_page.wait_for_load_state("domcontentloaded", timeout=30000)
                    target_page.wait_for_timeout(1500)
                expanded_sections = expand_summary_sections(target_page)
                auto_scroll(target_page, steps=8, delay_ms=400)
                snapshot = collect_page_snapshot(
                    target_page,
                    label=f"video_action_{len(snapshots) + 1}",
                    expanded_sections=expanded_sections,
                )
                snapshot["clicked_action_text"] = text
                snapshot["clicked_action_href"] = href
                snapshots.append(snapshot)
                if target_page == page and normalize_url_for_visit(page.url) != normalize_url_for_visit(before_url):
                    page.go_back(wait_until="domcontentloaded", timeout=30000)
                    page.wait_for_timeout(1000)
            except Exception:
                continue
    return snapshots


def collect_next_button_snapshots(page, limit: int = 12) -> list[dict]:
    snapshots: list[dict] = []
    seen_signatures: set[str] = set()
    for idx in range(limit):
        try:
            current_text = clean_text(page.locator("body").inner_text(timeout=3000))
            signature = current_text[:400]
            if signature in seen_signatures:
                break
            seen_signatures.add(signature)
            next_buttons = page.locator("button").filter(has_text=re.compile(r"^(Next|다음)$", re.IGNORECASE))
            count = next_buttons.count()
            if count < 1:
                break
            button = next_buttons.first()
            if not button.is_enabled(timeout=1000):
                break
            button.click(timeout=3000)
            page.wait_for_timeout(1600)
            expanded_sections = expand_summary_sections(page)
            auto_scroll(page, steps=6, delay_ms=300)
            snapshots.append(
                collect_page_snapshot(
                    page,
                    label=f"next_item_{idx + 1}",
                    expanded_sections=expanded_sections,
                )
            )
        except Exception:
            break
    return snapshots


def classify_url(url: str) -> str:
    if VIDEO_PATTERNS.search(url):
        return "video_or_player_candidate"
    if LAB_PATTERNS.search(url):
        return "lab_or_exercise_candidate"
    if LESSON_PATTERNS.search(url):
        return "lesson_or_course_candidate"
    return "link"


def extract_text_from_frames(page) -> list[dict]:
    frame_texts = []
    for idx, frame in enumerate(page.frames):
        try:
            txt = frame.locator("body").inner_text(timeout=3000)
            txt = clean_text(txt)
            if txt:
                frame_texts.append(
                    {
                        "frame_index": idx,
                        "frame_url": frame.url,
                        "text": txt,
                    }
                )
        except Exception:
            continue
    return frame_texts



def extract_structured_script_evidence(page, current_url: str, max_text_chars: int = 80000) -> tuple[str, list[str]]:
    """Extract JSON-backed content from Next/Notion/Oopy-style pages.

    Oopy/Notion exports often keep page blocks and child-page URLs inside
    __NEXT_DATA__ or other application/json scripts.  Anchor crawling alone can
    therefore capture only the title/nav shell.  This extractor treats those
    JSON scripts as evidence and also surfaces same-origin child page URLs for
    the deep crawler.
    """
    try:
        scripts = page.eval_on_selector_all(
            "script#__NEXT_DATA__, script[type='application/json']",
            "els => els.map(s => s.textContent || '').filter(Boolean)",
        )
    except Exception:
        scripts = []

    text_blocks: list[str] = []
    urls: list[str] = []
    seen_text: set[str] = set()
    for raw in (scripts or [])[:12]:
        if not raw or len(raw) < 80:
            continue
        urls.extend(re.findall(r"https?://[^\s\)\]\}<>\"']+", raw))
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        lines: list[str] = []
        flatten_json_for_source_text(payload, lines)
        urls.extend(extract_urls_from_json(payload))
        cleaned = clean_text("\n".join(unique_keep_order(lines)))
        if not cleaned:
            continue
        # Avoid dumping framework noise: keep meaningful medium-length strings.
        useful_lines = []
        for line in cleaned.splitlines():
            line = clean_text(line)
            if len(line) < 8:
                continue
            low = line.lower()
            if any(noise in low for noise in ["webpack", "__next", "buildid", "static/chunks", "google-analytics"]):
                continue
            useful_lines.append(line)
        useful = clean_text("\n".join(unique_keep_order(useful_lines)))
        if useful and useful not in seen_text:
            seen_text.add(useful)
            text_blocks.append(useful[:max_text_chars])

    normalized_urls = []
    seed_host = (urlparse(current_url).netloc or "").lower()
    for u in unique_keep_order(urls):
        if not isinstance(u, str):
            continue
        u = u.replace("\\u002F", "/").replace("\\/", "/")
        u = urljoin(current_url, u)
        if not u.startswith(("http://", "https://")) or is_media_asset_url(u):
            continue
        host = (urlparse(u).netloc or "").lower()
        if host == seed_host or host.endswith(".oopy.io") or host.endswith(".wikidocs.net"):
            normalized_urls.append(u)
    return clean_text("\n\n".join(text_blocks))[:max_text_chars], unique_keep_order(normalized_urls)


def collect_page_snapshot(page, label: str = "page", expanded_sections: list[str] | None = None) -> dict:
    current_url = page.url
    try:
        title = page.title()
    except Exception:
        title = ""

    try:
        visible_text = page.locator("body").inner_text(timeout=10000)
        visible_text = clean_text(visible_text)
    except Exception:
        visible_text = ""

    structured_text, structured_urls = extract_structured_script_evidence(page, current_url)
    if structured_text and structured_text not in visible_text:
        visible_text = clean_text(visible_text + "\n\n[Structured page data]\n" + structured_text)

    try:
        links = page.eval_on_selector_all(
            "a[href]",
            """
            els => els.map(a => ({
                text: (a.innerText || a.textContent || '').trim(),
                href: a.href
            }))
            """,
        )
    except Exception:
        links = []

    normalized_links = []
    for item in links:
        href = item.get("href") or ""
        text = clean_text(item.get("text") or "")
        if not href:
            continue
        href = urljoin(current_url, href)
        normalized_links.append(
            {
                "text": text[:300],
                "url": href,
                "kind": classify_url(href + " " + text),
            }
        )

    try:
        iframe_urls = page.eval_on_selector_all("iframe[src]", "els => els.map(x => x.src)")
    except Exception:
        iframe_urls = []

    try:
        video_src_urls = page.eval_on_selector_all(
            "video source[src], video[src]",
            "els => els.map(x => x.src || x.getAttribute('src')).filter(Boolean)",
        )
    except Exception:
        video_src_urls = []

    try:
        scripts_text = page.eval_on_selector_all(
            "script",
            "els => els.map(s => s.textContent || '').join('\\n')",
        )
    except Exception:
        scripts_text = ""

    script_url_candidates = re.findall(r"https?://[^\\s'\"<>]+", scripts_text or "")
    frame_texts = extract_text_from_frames(page)
    headings = extract_headings(page)
    candidate_items = extract_candidate_items(page, current_url)

    all_url_candidates = []
    all_url_candidates.extend([x["url"] for x in normalized_links])
    all_url_candidates.extend(iframe_urls or [])
    all_url_candidates.extend(video_src_urls or [])
    all_url_candidates.extend(script_url_candidates or [])
    all_url_candidates.extend(re.findall(r"https?://[^\s\)\]\}<>\"']+", visible_text or ""))
    all_url_candidates = unique_keep_order(all_url_candidates)

    return {
        "label": label,
        "title": title,
        "url": current_url,
        "visible_text": visible_text,
        "links": unique_keep_order(normalized_links),
        "iframe_urls": unique_keep_order(iframe_urls or []),
        "video_src_urls": unique_keep_order(video_src_urls or []),
        "script_url_candidates": unique_keep_order(script_url_candidates or []),
        "frame_texts": frame_texts,
        "all_url_candidates": all_url_candidates,
        "headings": headings,
        "expanded_sections": expanded_sections or [],
        **candidate_items,
    }


def auto_scroll(page, steps: int = 8, delay_ms: int = 700) -> None:
    try:
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(delay_ms)
        for i in range(steps):
            page.evaluate(
                "(args) => window.scrollTo(0, document.body.scrollHeight * ((args.i + 1) / args.steps))",
                {"i": i, "steps": steps},
            )
            page.wait_for_timeout(delay_ms)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(delay_ms)
    except Exception:
        pass


def open_and_collect(context, url: str, label: str) -> dict:
    page = context.new_page()
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2500)
        expanded_sections = expand_summary_sections(page)
        auto_scroll(page, steps=10, delay_ms=500)
        return collect_page_snapshot(page, label=label, expanded_sections=expanded_sections)
    except Exception as e:
        return {
            "label": label,
            "title": "",
            "url": url,
            "visible_text": "",
            "links": [],
            "iframe_urls": [],
            "video_src_urls": [],
            "script_url_candidates": [],
            "frame_texts": [],
            "all_url_candidates": [],
            "error": repr(e),
        }


def collect_linked_source_pages(
    context,
    seed_url: str,
    combined_data: dict,
    limit: int = 12,
    skip_urls: set[str] | None = None,
) -> list[dict]:
    skip_urls = skip_urls or set()
    candidates: list[tuple[str, str]] = []

    for item in combined_data.get("links", []):
        url = normalize_url_for_visit(item.get("url") or "")
        text = item.get("text") or ""
        if should_collect_link(seed_url, url, text):
            candidates.append((url, text))

    for url in combined_data.get("lesson_url_candidates", []) + combined_data.get("lab_url_candidates", []):
        normalized = normalize_url_for_visit(url)
        if should_collect_link(seed_url, normalized):
            candidates.append((normalized, ""))

    # For Oopy/Notion-style and WikiDocs pages, child-page links can be hidden
    # inside JSON/script data rather than normal anchor tags.  Promote all
    # discovered same-origin/content URLs to crawl candidates.
    for url in combined_data.get("all_url_candidates", []):
        normalized = normalize_url_for_visit(url)
        if should_collect_link(seed_url, normalized):
            candidates.append((normalized, "discovered structured URL"))

    snapshots = []
    seen = set(skip_urls)
    for url, text in unique_keep_order(candidates):
        normalized = normalize_url_for_visit(url)
        if normalized in seen or len(snapshots) >= limit:
            continue
        seen.add(normalized)
        kind = "linked_lab" if LAB_PATTERNS.search(f"{normalized} {text}") else "linked_source"
        print(f"Collecting {kind}: {normalized}")
        snapshots.append(open_and_collect(context, normalized, label=f"{kind}_{len(snapshots) + 1}"))
    return snapshots


def build_combined_data(
    primary_snapshot: dict,
    tab_snapshots: list[dict],
    followed_snapshots: list[dict],
    tree_items: list[dict] | None = None,
) -> dict:
    snapshots = [primary_snapshot] + tab_snapshots + followed_snapshots

    all_links = []
    all_urls = []
    visible_parts = []
    frame_parts = []
    all_headings = []
    all_video_items = []
    all_lesson_items = []
    all_lab_items = []

    for snap in snapshots:
        heading = f"\n\n===== PAGE: {snap.get('title') or snap.get('label')} =====\nURL: {snap.get('url')}\n"
        if snap.get("visible_text"):
            visible_parts.append(heading + snap["visible_text"])
        for frame in snap.get("frame_texts", []):
            if frame.get("text"):
                frame_parts.append(
                    f"\n\n===== FRAME from {snap.get('title')} =====\nFrame URL: {frame.get('frame_url')}\n{frame.get('text')}"
                )
        all_links.extend(snap.get("links", []))
        all_headings.extend(snap.get("headings", []))
        all_video_items.extend(snap.get("video_items", []))
        all_lesson_items.extend(snap.get("lesson_items", []))
        all_lab_items.extend(snap.get("lab_items", []))
        all_urls.extend(snap.get("all_url_candidates", []))
        all_urls.extend(snap.get("iframe_urls", []))
        all_urls.extend(snap.get("video_src_urls", []))
        all_urls.extend(snap.get("script_url_candidates", []))

    all_links = unique_keep_order(all_links)
    all_urls = unique_keep_order(all_urls)

    video_candidates = unique_keep_order([u for u in all_urls if VIDEO_PATTERNS.search(u)])
    lesson_candidates = unique_keep_order([u for u in all_urls if LESSON_PATTERNS.search(u)])
    lab_candidates = unique_keep_order(
        [u for u in all_urls if LAB_PATTERNS.search(u) and not is_media_asset_url(u)]
        + [
            item["url"] for item in all_links
            if LAB_PATTERNS.search((item.get("url") or "") + " " + (item.get("text") or ""))
            and not is_media_asset_url(item.get("url") or "")
        ]
        + [
            item["url"] for item in all_lab_items
            if item.get("url") and not is_media_asset_url(item.get("url") or "")
        ]
    )
    tree_lab_items = [
        item for item in unique_keep_order(tree_items or [])
        if TREE_LAB_PATTERNS.search(item.get("text") or "")
    ]

    visible_text = clean_text("\n".join(visible_parts + frame_parts))
    login_or_access_pages = [
        snap for snap in snapshots
        if looks_like_login_or_access_page(snap.get("url", ""), snap.get("title", ""), snap.get("visible_text", ""))
    ]
    usable_text_chars = len(visible_text)
    quality_warnings: list[str] = []
    if login_or_access_pages:
        quality_warnings.append("login_or_access_page_detected")
    if usable_text_chars < 800:
        quality_warnings.append("low_visible_text")
    if len(snapshots) <= 1 and not (all_links or all_video_items or all_lesson_items or all_lab_items or tree_items):
        quality_warnings.append("single_sparse_page")

    return {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "title": primary_snapshot.get("title", "source_pack"),
        "current_url": primary_snapshot.get("url", ""),
        "snapshots": snapshots,
        "visible_text": visible_text,
        "links": all_links,
        "headings": unique_keep_order(all_headings),
        "all_url_candidates": all_urls,
        "video_url_candidates": video_candidates,
        "lesson_url_candidates": lesson_candidates,
        "lab_url_candidates": lab_candidates,
        "video_items": unique_keep_order(all_video_items),
        "lesson_items": unique_keep_order(all_lesson_items),
        "lab_items": unique_keep_order(all_lab_items),
        "tree_items": unique_keep_order(tree_items or []),
        "quality": {
            "ok": not quality_warnings,
            "warnings": quality_warnings,
            "login_or_access_page_count": len(login_or_access_pages),
            "usable_text_chars": usable_text_chars,
        },
        "stats": {
            "page_count": len(snapshots),
            "visible_text_chars": len(visible_text),
            "link_count": len(all_links),
            "video_candidate_count": len(video_candidates),
            "lesson_candidate_count": len(lesson_candidates),
            "lab_candidate_count": len(lab_candidates),
            "tree_lab_item_count": len(tree_lab_items),
            "tree_item_count": len(unique_keep_order(tree_items or [])),
        },
    }


def write_outputs(data: dict, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r"[^a-zA-Z0-9가-힣_-]+", "_", data.get("title") or "source_pack")
    safe_title = safe_title[:70].strip("_") or "source_pack"

    json_path = output_dir / f"{ts}_{safe_title}.json"
    md_path = output_dir / f"{ts}_{safe_title}.md"
    report_path = output_dir / f"{ts}_{safe_title}.report.md"

    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"# Source Pack: {data.get('title', '')}")
    lines.append("")
    lines.append(f"- Captured at: {data.get('captured_at')}")
    lines.append(f"- Page URL: {data.get('current_url')}")
    lines.append("")
    lines.append("## Collection Stats")
    for k, v in data.get("stats", {}).items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("## Headings")
    for item in data.get("headings", []):
        prefix = f"H{item.get('level')}" if item.get("level") else "heading"
        lines.append(f"- {prefix}: {item.get('text')}")

    lines.append("")
    lines.append("## Lab / Exercise URL Candidates")
    for url in data.get("lab_url_candidates", []):
        lines.append(f"- {url}")

    lines.append("")
    lines.append("## Player Navigation Items")
    for item in data.get("tree_items", []):
        marker = "lab/exercise" if TREE_LAB_PATTERNS.search(item.get("text") or "") else "item"
        lines.append(f"- [{marker}] {item.get('text')}")

    lines.append("")
    lines.append("## Video URL Candidates")
    for url in data.get("video_url_candidates", []):
        lines.append(f"- {url}")

    lines.append("")
    lines.append("## Lesson / Course URL Candidates")
    for url in data.get("lesson_url_candidates", []):
        lines.append(f"- {url}")

    lines.append("")
    lines.append("## All Links")
    for item in data.get("links", []):
        text = item.get("text") or "(no text)"
        url = item.get("url")
        kind = item.get("kind")
        lines.append(f"- [{kind}] {text} — {url}")

    lines.append("")
    lines.append("## Visible Page Text")
    lines.append("")
    lines.append(data.get("visible_text", ""))

    md_path.write_text("\n".join(lines), encoding="utf-8")

    report_lines = []
    report_lines.append("# Source Pack Collection Report")
    report_lines.append("")
    report_lines.append(f"- Captured at: {data.get('captured_at')}")
    report_lines.append(f"- Player URL: {data.get('current_url')}")
    report_lines.append(f"- Title: {data.get('title', '')}")
    report_lines.append("")
    report_lines.append("## Stats")
    for key, value in data.get("stats", {}).items():
        report_lines.append(f"- {key}: {value}")
    report_lines.append("")
    report_lines.append("## Files")
    report_lines.append(f"- Markdown: {md_path}")
    report_lines.append(f"- JSON: {json_path}")
    report_lines.append("")
    report_lines.append("## Followed Lab URLs")
    followed = [snap.get("url") for snap in data.get("snapshots", []) if str(snap.get("label", "")).startswith("followed_lab_")]
    for url in followed:
        report_lines.append(f"- {url}")
    if not followed:
        report_lines.append("- None")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return md_path, json_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "url",
        help="Representative source URL, such as an AI Skills Navigator player, article, docs page, lab page, or video URL.",
    )
    parser.add_argument("--out", default="data/source_packs", help="Output directory")
    parser.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "ai-skills-navigator-player", "generic-url"],
        help="Collection profile. auto uses the AI Skills profile for player URLs and generic URL collection otherwise.",
    )
    parser.add_argument("--headless", action="store_true", help="Run headless. Avoid if login/Summary clicks are needed.")
    parser.add_argument(
        "--no-manual-pause",
        action="store_true",
        help="Do not pause for manual login/confirmation. Use only when already authenticated and ready.",
    )
    parser.add_argument("--follow-labs", action="store_true", help="Automatically open lab/exercise links found on pages and append their content.")
    parser.add_argument("--follow-limit", type=int, default=8, help="Maximum lab/exercise links to follow.")
    parser.add_argument(
        "--no-crawl",
        action="store_true",
        help="Do not collect related same-origin article/docs/lesson pages discovered from the seed URL.",
    )
    parser.add_argument("--crawl-limit", type=int, default=12, help="Maximum related source pages to collect.")
    parser.add_argument(
        "--tree-limit",
        type=int,
        default=24,
        help="Maximum AI Skills Navigator player navigation items to click and collect.",
    )
    parser.add_argument(
        "--no-click-tree",
        action="store_true",
        help="Do not click player navigation tree items. By default the collector opens navigation and collects visible cards/labs.",
    )
    parser.add_argument(
        "--user-data-dir",
        default="",
        help="Optional persistent browser profile directory for reusing login sessions.",
    )
    parser.add_argument(
        "--storage-state",
        default="",
        help="Optional Playwright storage_state JSON path to load/save cookies and local storage.",
    )
    parser.add_argument(
        "--auto-login-wait",
        type=int,
        default=0,
        help="In non-interactive mode, wait this many seconds when a real login/access page is detected.",
    )
    args = parser.parse_args()

    output_dir = Path(args.out)
    profile = args.profile
    if profile == "auto":
        profile = "ai-skills-navigator-player" if is_ai_skills_player_url(args.url) else "generic-url"

    if profile == "ai-skills-navigator-player" and not is_ai_skills_player_url(args.url):
        print("\nWarning: this profile expects a final AI Skills Navigator player URL.")
        print("Expected shape: https://aiskillsnavigator.microsoft.com/player?playlistId=...")
        print(f"Received: {args.url}")
        input("Press Enter to continue anyway, or Ctrl-C to stop: ")

    sync_playwright = load_playwright()
    with sync_playwright() as p:
        browser = None
        storage_state_path = Path(args.storage_state) if args.storage_state else None
        if args.user_data_dir:
            try:
                context = p.chromium.launch_persistent_context(
                    str(Path(args.user_data_dir)),
                    headless=args.headless,
                )
            except Exception as exc:
                print(f"Persistent profile unavailable; using a temporary browser profile instead: {exc}")
                fallback_profile = Path(tempfile.mkdtemp(prefix="source-collector-profile-"))
                context = p.chromium.launch_persistent_context(
                    str(fallback_profile),
                    headless=args.headless,
                )
        else:
            browser = p.chromium.launch(headless=args.headless)
            context_kwargs = {}
            if storage_state_path and storage_state_path.exists():
                context_kwargs["storage_state"] = str(storage_state_path)
            context = browser.new_context(**context_kwargs)
        page = context.new_page()
        ai_skills_api_snapshots: list[dict] = []
        if profile == "ai-skills-navigator-player":
            attach_ai_skills_api_capture(page, ai_skills_api_snapshots)

        print("\nOpening page:")
        print(args.url)
        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        wait_for_source_ready(page)

        print(f"\nCollector profile: {profile}")
        print("- Do not log in through automation.")
        if profile == "ai-skills-navigator-player":
            print("- Do not choose playlist cards here.")
            print("- This run starts from the final player URL you provided.")
        maybe_pause_for_login_or_confirmation(
            page,
            args.url,
            manual_pause=not args.no_manual_pause,
            auto_login_wait=args.auto_login_wait,
        )
        wait_for_source_ready(page)

        pages = context.pages
        primary_page = page if page in pages else pages[0]
        expanded_sections = expand_summary_sections(primary_page)
        auto_scroll(primary_page, steps=10, delay_ms=500)
        primary_snapshot = collect_page_snapshot(primary_page, label="primary_before_tree_clicks", expanded_sections=expanded_sections)

        tree_snapshots = []
        tree_items = []
        if profile == "ai-skills-navigator-player" and not args.no_click_tree:
            print("\nOpening player navigation and collecting lesson/lab tree items...")
            tree_snapshots, tree_items = collect_tree_item_snapshots(context, primary_page, limit=args.tree_limit)
            if not tree_snapshots:
                print("\nCollecting AI Skills Navigator Next cards...")
                tree_snapshots.extend(collect_next_button_snapshots(primary_page, limit=args.tree_limit))

        tab_snapshots = []
        for idx, other in enumerate(pages):
            if other == primary_page:
                continue
            try:
                auto_scroll(other, steps=10, delay_ms=500)
                tab_snapshots.append(collect_page_snapshot(other, label=f"open_tab_{idx}", expanded_sections=expand_summary_sections(other)))
            except Exception:
                continue

        tab_snapshots.extend(tree_snapshots)
        tab_snapshots.extend(ai_skills_api_snapshots)

        temp_data = build_combined_data(primary_snapshot, tab_snapshots, [], tree_items=tree_items)
        crawled_snapshots = []
        if not args.no_crawl:
            print("\nCollecting related source pages discovered from the seed URL...")
            crawled_snapshots = collect_linked_source_pages(
                context,
                args.url,
                temp_data,
                limit=args.crawl_limit,
                skip_urls={normalize_url_for_visit(primary_snapshot.get("url") or args.url)},
            )
            if crawled_snapshots:
                tab_snapshots.extend(crawled_snapshots)
                temp_data = build_combined_data(primary_snapshot, tab_snapshots, [], tree_items=tree_items)

        lab_links = temp_data.get("lab_url_candidates", [])

        followed_snapshots = []
        if args.follow_labs and lab_links:
            print(f"\nLab/exercise links found: {len(lab_links)}")
            for idx, lab_url in enumerate(lab_links[: args.follow_limit], start=1):
                print(f"[{idx}/{min(len(lab_links), args.follow_limit)}] Collecting lab: {lab_url}")
                snap = open_and_collect(context, lab_url, label=f"followed_lab_{idx}")
                if snap:
                    followed_snapshots.append(snap)

        data = build_combined_data(primary_snapshot, tab_snapshots, followed_snapshots, tree_items=tree_items)
        md_path, json_path, report_path = write_outputs(data, output_dir)

        print("\n수집 완료")
        print(f"Markdown: {md_path}")
        print(f"JSON: {json_path}")
        print(f"Report: {report_path}")
        print(f"Pages collected: {data['stats']['page_count']}")
        print(f"Visible text chars: {data['stats']['visible_text_chars']}")
        print(f"Links: {data['stats']['link_count']}")
        print(f"Video candidates: {data['stats']['video_candidate_count']}")
        print(f"Lesson candidates: {data['stats']['lesson_candidate_count']}")
        print(f"Lab candidates: {data['stats']['lab_candidate_count']}")

        if storage_state_path and not args.user_data_dir:
            storage_state_path.parent.mkdir(parents=True, exist_ok=True)
            context.storage_state(path=str(storage_state_path))
            print(f"Storage state saved: {storage_state_path}")

        context.close()
        if browser:
            browser.close()


if __name__ == "__main__":
    main()
