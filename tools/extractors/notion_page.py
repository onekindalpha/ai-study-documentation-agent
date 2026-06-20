from __future__ import annotations

import re
from typing import Any

from tools.collector_core.schema import make_graph, make_node


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _render_notion_html(url: str, *, visible: bool = False, timeout: int = 30) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(f"playwright import failed: {exc}") from exc

    with sync_playwright() as p:  # pragma: no cover - exercised in user env
        browser = p.chromium.launch(headless=not visible)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.goto(url, wait_until="networkidle", timeout=max(10, timeout) * 1000)
        page.wait_for_timeout(1500)
        # Expand visible disclosure/toggle buttons when possible.  This is best-effort
        # and deliberately bounded so it never turns into a site-specific bot.
        for selector in [
            "[aria-expanded='false']",
            "[role='button'][aria-expanded='false']",
            ".notion-toggle-block [role='button']",
        ]:
            try:
                loc = page.locator(selector)
                for i in range(min(loc.count(), 40)):
                    try:
                        loc.nth(i).click(timeout=300)
                    except Exception:
                        pass
            except Exception:
                pass
        # Notion lazily loads blocks while scrolling.
        for _ in range(8):
            page.mouse.wheel(0, 1800)
            page.wait_for_timeout(450)
        html = page.content()
        browser.close()
        return html


def _extract_blocks(html: str, url: str) -> tuple[str, list[dict[str, Any]]]:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        raise RuntimeError(f"beautifulsoup import failed: {exc}") from exc

    soup = BeautifulSoup(html, "html.parser")
    for bad in soup(["script", "style", "noscript", "svg"]):
        bad.decompose()
    title = _clean(soup.title.get_text(" ", strip=True) if soup.title else "") or "Notion"
    candidates = []
    # data-block-id is the most stable signal for public Notion blocks.  The class
    # fallbacks cover older exported/rendered variants.
    selectors = [
        "[data-block-id]",
        ".notion-page-content [class*='notion-'][class*='block']",
        "main [class*='notion-'][class*='block']",
    ]
    seen_texts: set[str] = set()
    for selector in selectors:
        for el in soup.select(selector):
            text = _clean(el.get_text(" ", strip=True))
            if len(text) < 25:
                continue
            lower = text.lower()
            if lower in seen_texts:
                continue
            if any(shell in lower for shell in ["notion", "log in", "sign in", "enable javascript"]) and len(text) < 300:
                continue
            seen_texts.add(lower)
            cls = " ".join(el.get("class") or []).lower()
            tag = (el.name or "").lower()
            if tag in {"h1", "h2", "h3"} or "header" in cls:
                kind = "heading"
            elif "code" in cls:
                kind = "code"
            elif "table" in cls:
                kind = "table_row"
            elif "callout" in cls:
                kind = "callout"
            elif "toggle" in cls:
                kind = "toggle"
            else:
                kind = "paragraph"
            candidates.append({"kind": kind, "text": text})
    # Fallback: if block selectors failed but the rendered body is real, split it.
    if not candidates:
        body = soup.find("main") or soup.body or soup
        body_text = body.get_text("\n", strip=True)
        for chunk in [c.strip() for c in re.split(r"\n{2,}", body_text) if len(c.strip()) >= 80]:
            candidates.append({"kind": "paragraph", "text": _clean(chunk)})
    return title, candidates[:300]


def collect(url: str, *, run_dir, trace, plan, max_pages: int = 1, visible: bool = False, max_depth: int = 1, timeout: int = 30, **kwargs) -> dict[str, Any]:
    graph = make_graph(
        input_url=url,
        url_type="notion_page",
        site_hint="notion",
        content_shape=["dynamic_blocks", "text_only"],
        navigation_shape=["single_page", "lazy_loaded_blocks"],
        access_level=plan.access_level,
        evidence_targets=["notion_blocks", "headings", "paragraphs", "code_blocks", "tables", "callouts", "toggles"],
        title="Notion",
    )
    try:
        html = _render_notion_html(url, visible=visible, timeout=timeout)
        title, blocks = _extract_blocks(html, url)
        graph["title"] = title or "Notion"
        trace.event("notion_rendered", title=graph["title"], blocks=len(blocks))
        if not blocks:
            graph["quality"]["missing"].append("notion page adapter captured no usable blocks")
            graph["quality"]["warnings"].append("notion_shell_only")
            return graph
        for idx, block in enumerate(blocks, start=1):
            graph["nodes"].append(
                make_node(
                    node_type="notion_block",
                    title=f"Notion block {idx:03d}",
                    url=url,
                    order=idx,
                    text=block.get("text") or "",
                    evidence=[{"type": block.get("kind") or "paragraph", "text": block.get("text") or ""}],
                    meta={"block_kind": block.get("kind") or "paragraph"},
                )
            )
        graph["quality"]["notion_blocks"] = len(blocks)
    except Exception as exc:
        trace.warning("notion_page_failed", error=str(exc))
        graph["quality"]["missing"].append(f"notion page render/extract failed: {exc}")
        graph["quality"]["warnings"].append("notion_shell_only")
    return graph
