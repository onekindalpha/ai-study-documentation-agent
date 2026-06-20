from __future__ import annotations

import html
import re
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse, urldefrag


@dataclass
class PageData:
    url: str
    title: str
    text: str
    links: list[dict]
    images: list[dict]
    code_blocks: list[str]
    headings: list[str]
    html: str = ""


def normalize_url(base_url: str, href: str) -> str:
    if not href:
        return ""
    href = html.unescape(href.strip())
    full = urljoin(base_url, href)
    full, _ = urldefrag(full)
    return full


def same_domain(url: str, root_url: str) -> bool:
    return urlparse(url).netloc.lower() == urlparse(root_url).netloc.lower()


def fetch_html(url: str, timeout: int = 25, user_agent: str = "Mozilla/5.0 StudyCaptureCopilot/0.1") -> str:
    try:
        import certifi
        import requests

        resp = requests.get(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            timeout=timeout,
            verify=certifi.where(),
        )
        resp.raise_for_status()
        if not resp.encoding:
            resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text
    except Exception as requests_exc:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                charset = resp.headers.get_content_charset() or "utf-8"
                return raw.decode(charset, errors="replace")
        except Exception as urllib_exc:
            raise RuntimeError(f"fetch_html failed: requests={requests_exc}; urllib={urllib_exc}") from urllib_exc

def render_html_with_playwright(url: str, *, visible: bool = False, timeout_ms: int = 30000, screenshot_path: Path | None = None) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise RuntimeError(f"playwright import failed: {exc}") from exc

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not visible)
        page = browser.new_page(viewport={"width": 1440, "height": 1200})
        page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        page.wait_for_timeout(1200)
        if screenshot_path:
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot_path), full_page=True)
        content = page.content()
        browser.close()
        return content


class FallbackHTMLParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[dict] = []
        self.images: list[dict] = []
        self.code_blocks: list[str] = []
        self.headings: list[str] = []
        self._tag_stack: list[str] = []
        self._current_link: dict | None = None
        self._current_code: list[str] | None = None
        self._current_heading: list[str] | None = None
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = {k.lower(): v or "" for k, v in attrs}
        self._tag_stack.append(tag)
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = normalize_url(self.base_url, attrs_d.get("href", ""))
            if href:
                self._current_link = {"url": href, "text": ""}
        if tag == "img":
            src = normalize_url(self.base_url, attrs_d.get("src", ""))
            if src:
                self.images.append({"url": src, "alt": attrs_d.get("alt", "")})
        if tag in {"pre", "code"} and self._current_code is None:
            self._current_code = []
        if tag in {"h1", "h2", "h3", "h4"}:
            self._current_heading = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._current_link:
            self._current_link["text"] = re.sub(r"\s+", " ", self._current_link.get("text", "")).strip()
            self.links.append(self._current_link)
            self._current_link = None
        if tag in {"pre", "code"} and self._current_code is not None:
            code = "".join(self._current_code).strip()
            if len(code) > 10:
                self.code_blocks.append(code)
            self._current_code = None
        if tag in {"h1", "h2", "h3", "h4"} and self._current_heading is not None:
            heading = re.sub(r"\s+", " ", "".join(self._current_heading)).strip()
            if heading:
                self.headings.append(heading)
            self._current_heading = None
        if self._tag_stack:
            self._tag_stack.pop()
        if tag in {"p", "div", "section", "article", "li", "br", "h1", "h2", "h3", "h4"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        if self._current_link is not None:
            self._current_link["text"] += data
        if self._current_code is not None:
            self._current_code.append(data)
        if self._current_heading is not None:
            self._current_heading.append(data)
        if data.strip():
            self.text_parts.append(data)


def parse_html(raw_html: str, url: str) -> PageData:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw_html, "html.parser")
        for bad in soup(["script", "style", "noscript", "svg"]):
            bad.decompose()
        title = (soup.title.get_text(" ", strip=True) if soup.title else "")
        headings = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3", "h4"]) if h.get_text(strip=True)]
        links = []
        for a in soup.find_all("a"):
            href = normalize_url(url, a.get("href") or "")
            if href:
                links.append({"url": href, "text": a.get_text(" ", strip=True)})
        images = []
        for img in soup.find_all("img"):
            src = normalize_url(url, img.get("src") or img.get("data-src") or "")
            if src:
                images.append({"url": src, "alt": img.get("alt") or ""})
        code_blocks = [c.get_text("\n", strip=False).strip() for c in soup.find_all(["pre", "code"]) if len(c.get_text(strip=True)) > 10]
        main = soup.find("main") or soup.find("article") or soup.body or soup
        text = main.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return PageData(url=url, title=title, text=text, links=dedupe_links(links), images=images[:80], code_blocks=code_blocks[:80], headings=headings[:80], html=raw_html)
    except Exception:
        parser = FallbackHTMLParser(url)
        parser.feed(raw_html)
        title = re.sub(r"\s+", " ", " ".join(parser.title_parts)).strip()
        text = re.sub(r"\s+", " ", " ".join(parser.text_parts)).strip()
        return PageData(url=url, title=title, text=text, links=dedupe_links(parser.links), images=parser.images[:80], code_blocks=parser.code_blocks[:80], headings=parser.headings[:80], html=raw_html)


def dedupe_links(links: Iterable[dict]) -> list[dict]:
    seen = set()
    result = []
    for link in links:
        url = link.get("url") or ""
        if not url or url in seen:
            continue
        if url.startswith("mailto:") or url.startswith("tel:") or url.startswith("javascript:"):
            continue
        seen.add(url)
        result.append(link)
    return result


def fetch_page_data(url: str, *, timeout: int = 25, render: bool = False, visible: bool = False, screenshot_path: Path | None = None) -> PageData:
    if render:
        raw = render_html_with_playwright(url, visible=visible, screenshot_path=screenshot_path)
    else:
        try:
            raw = fetch_html(url, timeout=timeout)
        except Exception as exc:
            msg = str(exc)
            if "CERTIFICATE_VERIFY_FAILED" in msg or "SSLCertVerificationError" in msg or "unable to get local issuer certificate" in msg:
                raw = render_html_with_playwright(url, visible=visible, screenshot_path=screenshot_path)
            else:
                raise
    return parse_html(raw, url)
