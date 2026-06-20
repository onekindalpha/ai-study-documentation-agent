from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def is_noise_learning_node(node: dict[str, Any]) -> bool:
    title = str(node.get("title") or "").strip().lower()
    text = str(node.get("text") or "").strip().lower()
    node_type = str(node.get("type") or "").lower()
    if node_type.endswith("_root") and len(text) < 1200:
        return True
    if title in {"home", "devlog", "about", "search", "share"}:
        return True
    if "새로운걸 공부하고 기록하는 것을 좋아합니다" in title:
        return True
    if "backend engineer" in text[:800]:
        return True
    return False


def compact_excerpt(text: str, limit: int = 900) -> str:
    cleaned = " ".join(str(text or "").split())
    for marker in ["Search Share ", "TOP Home Devlog Github"]:
        cleaned = cleaned.replace(marker, " ")
    cleaned = cleaned.replace("새로운걸 공부하고 기록하는 것을 좋아합니다. /", " ")
    cleaned = cleaned.replace("기술 면접 대비 CS전공 핵심요약집 /", " ")
    cleaned = " ".join(cleaned.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def learning_digest_nodes(graph: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    nodes = [
        node for node in graph.get("nodes") or []
        if isinstance(node, dict)
        and len(str(node.get("text") or "")) >= 500
        and not is_noise_learning_node(node)
    ]
    nodes.sort(key=lambda node: len(str(node.get("text") or "")), reverse=True)
    return nodes[:limit]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def node_to_markdown(node: dict[str, Any], depth: int = 2) -> str:
    hashes = "#" * min(depth, 6)
    lines = [f"{hashes} {node.get('title') or node.get('type')}"]
    if node.get("url"):
        lines.append(f"- URL: {node['url']}")
    if node.get("type"):
        lines.append(f"- Type: {node['type']}")
    text = (node.get("text") or "").strip()
    if text:
        lines.append("")
        lines.append(text[:12000])
    evidence = node.get("evidence") or []
    if evidence:
        lines.append("")
        lines.append("Evidence:")
        for item in evidence[:20]:
            lines.append(f"- {item.get('type') or item.get('kind')}: {item.get('text') or item.get('url') or item.get('value') or ''}")
    for child in node.get("children") or []:
        lines.append("")
        lines.append(node_to_markdown(child, depth + 1))
    return "\n".join(lines).strip()


def graph_to_source_pack(graph: dict[str, Any]) -> str:
    q = graph.get("quality") or {}
    curriculum = graph.get("curriculum_overview") if isinstance(graph.get("curriculum_overview"), list) else []
    lines = [
        f"# Source Pack: {graph.get('title') or graph.get('input_url')}",
        "",
        "## Collection Metadata",
        f"- Input URL: {graph.get('input_url')}",
        f"- URL type: {graph.get('url_type')}",
        f"- Site hint: {graph.get('site_hint')}",
        f"- Status: {graph.get('status')}",
        f"- Content shape: {', '.join(graph.get('content_shape') or [])}",
        f"- Navigation shape: {', '.join(graph.get('navigation_shape') or [])}",
        f"- Access level: {graph.get('access_level')}",
        "",
        "## Quality",
        f"- Pages collected: {q.get('pages_collected', 0)}",
        f"- Text chars: {q.get('text_chars', 0)}",
        f"- Transcript segments: {q.get('transcript_segments', 0)}",
        f"- Images collected: {q.get('images_collected', 0)}",
        f"- Code blocks: {q.get('code_blocks', 0)}",
        f"- Lab steps: {q.get('lab_steps', 0)}",
        f"- Usable text units: {q.get('usable_text_units_count', q.get('usable_units', 0))}",
        f"- Usable text chars: {q.get('usable_text_chars', 0)}",
        f"- High value units: {q.get('high_value_units', 0)}",
        f"- Evidence quality score: {q.get('evidence_quality_score', 0)}",
        f"- Missing: {', '.join(q.get('missing') or [])}",
        f"- Warnings: {', '.join(q.get('warnings') or [])}",
        "",
    ]
    if graph.get("summary_hint"):
        lines.extend([
            "## Source Structure Summary",
            str(graph.get("summary_hint") or ""),
            "",
        ])
    if curriculum:
        lines.append("## Curriculum Overview")
        for item in curriculum[:40]:
            lesson = str(item.get("lesson") or "").strip()
            title = str(item.get("title") or "").strip()
            briefing = str(item.get("briefing") or "").strip()
            prefix = f"Lesson {lesson}: " if lesson else ""
            lines.append(f"- {prefix}{title}" + (f" — {briefing}" if briefing else ""))
        lines.append("")
    digest = learning_digest_nodes(graph)
    if digest:
        lines.append("## Learning Flow Digest")
        lines.append("하위 페이지 본문에서 글 작성에 우선 사용할 학습 근거입니다.")
        lines.append("")
        for idx, node in enumerate(digest, start=1):
            title = str(node.get("title") or node.get("type") or f"Section {idx}").strip()
            url = str(node.get("url") or "").strip()
            text = compact_excerpt(str(node.get("text") or ""), limit=1100)
            lines.append(f"### {idx}. {title}")
            if url:
                lines.append(f"- URL: {url}")
            lines.append(f"- Body chars: {len(str(node.get('text') or ''))}")
            lines.append("")
            lines.append(text)
            lines.append("")
    lines.append("## Collected Nodes")
    for node in graph.get("nodes") or []:
        lines.append("")
        lines.append(node_to_markdown(node, 2))
    return "\n".join(lines).strip() + "\n"


def write_outputs(run_dir: Path, graph: dict[str, Any]) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    source_graph_path = run_dir / "source_graph.json"
    report_path = run_dir / "collection_report.json"
    source_pack_path = run_dir / "source_pack.md"

    report = {
        "input_url": graph.get("input_url"),
        "url_type": graph.get("url_type"),
        "site_hint": graph.get("site_hint"),
        "status": graph.get("status"),
        "title": graph.get("title"),
        "quality": graph.get("quality") or {},
        "node_count": len(graph.get("nodes") or []),
        "asset_count": len(graph.get("assets") or []),
        "output_files": {
            "source_graph": str(source_graph_path),
            "collection_report": str(report_path),
            "source_pack": str(source_pack_path),
            "trace": str(run_dir / "trace.jsonl"),
        },
    }

    write_json(source_graph_path, graph)
    write_json(report_path, report)
    source_pack_path.write_text(graph_to_source_pack(graph), encoding="utf-8")
    return report["output_files"]
