from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(*parts: str, prefix: str = "node") -> str:
    raw = "|".join(p or "" for p in parts)
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def slugify(value: str, max_len: int = 60) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^a-z0-9가-힣._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-._")
    return (value[:max_len].strip("-._") or "source")


def make_run_id(url: str, url_type: str | None = None) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "") or "url"
    label = slugify(f"{url_type or 'url'}-{host}-{parsed.path}", 72)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{label}"


def empty_quality() -> dict[str, Any]:
    return {
        "pages_collected": 0,
        "text_chars": 0,
        "transcript_segments": 0,
        "images_collected": 0,
        "code_blocks": 0,
        "tables": 0,
        "lab_steps": 0,
        "child_links_collected": 0,
        "usable_units": 0,
        "usable_text_units_count": 0,
        "usable_text_chars": 0,
        "high_value_units": 0,
        "toc_ratio": 0.0,
        "shell_ratio": 0.0,
        "boilerplate_ratio": 0.0,
        "evidence_quality_score": 0.0,
        "missing": [],
        "warnings": [],
    }


def make_graph(
    *,
    input_url: str,
    url_type: str,
    site_hint: str = "",
    content_shape: list[str] | None = None,
    navigation_shape: list[str] | None = None,
    access_level: str = "public",
    evidence_targets: list[str] | None = None,
    title: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "created_at": now_iso(),
        "input_url": input_url,
        "url_type": url_type,
        "site_hint": site_hint,
        "content_shape": content_shape or [],
        "navigation_shape": navigation_shape or [],
        "access_level": access_level,
        "evidence_targets": evidence_targets or [],
        "status": "partial",
        "title": title,
        "summary_hint": "",
        "nodes": [],
        "assets": [],
        "quality": empty_quality(),
    }


def make_node(
    *,
    node_type: str,
    title: str,
    url: str = "",
    order: int = 0,
    text: str = "",
    children: list[dict[str, Any]] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": stable_id(node_type, title, url, str(order), prefix=node_type.replace("_", "-")),
        "type": node_type,
        "title": title or node_type,
        "url": url,
        "order": order,
        "text": text or "",
        "children": children or [],
        "evidence": evidence or [],
        "meta": meta or {},
    }


def update_quality_from_graph(graph: dict[str, Any]) -> None:
    quality = graph.setdefault("quality", empty_quality())
    nodes = graph.get("nodes") or []
    text_chars = 0
    transcript_segments = 0
    code_blocks = 0
    lab_steps = 0
    usable_units = 0
    usable_text_chars = 0
    high_value_units = 0

    def walk(items: list[dict[str, Any]]) -> None:
        nonlocal text_chars, transcript_segments, code_blocks, lab_steps, usable_units, usable_text_chars, high_value_units
        for node in items:
            text_chars += len(node.get("text") or "")
            node_type = node.get("type") or ""
            if node_type == "transcript_segment":
                transcript_segments += 1
            evidence = node.get("evidence") or []
            for item in evidence:
                kind = item.get("type") or item.get("kind") or ""
                if kind == "code":
                    code_blocks += 1
                if kind in {"lab_step", "exercise_step"}:
                    lab_steps += 1
                if kind in {"code", "command", "error", "table_row", "example", "step_guide"}:
                    high_value_units += 1
            meta = node.get("meta") if isinstance(node.get("meta"), dict) else {}
            role = str(meta.get("role") or "").lower()
            unit_type = str(meta.get("unit_type") or node_type or "").lower()
            if role == "usable" or unit_type in {"transcript_segment", "notion_block", "page"}:
                if len(node.get("text") or "") >= 35:
                    usable_units += 1
                    usable_text_chars += len(node.get("text") or "")
                    if unit_type in {"code_block", "command_block", "error_block", "table_row", "example", "step_guide", "transcript_segment"}:
                        high_value_units += 1
            walk(node.get("children") or [])

    walk(nodes)
    quality["pages_collected"] = max(quality.get("pages_collected", 0), len([n for n in nodes if n.get("url")]))
    quality["text_chars"] = max(quality.get("text_chars", 0), text_chars)
    quality["transcript_segments"] = max(quality.get("transcript_segments", 0), transcript_segments)
    quality["code_blocks"] = max(quality.get("code_blocks", 0), code_blocks)
    quality["lab_steps"] = max(quality.get("lab_steps", 0), lab_steps)
    quality["usable_units"] = max(quality.get("usable_units", 0), usable_units)
    quality["usable_text_units_count"] = max(quality.get("usable_text_units_count", 0), usable_units)
    quality["usable_text_chars"] = max(quality.get("usable_text_chars", 0), usable_text_chars)
    quality["high_value_units"] = max(quality.get("high_value_units", 0), high_value_units)
