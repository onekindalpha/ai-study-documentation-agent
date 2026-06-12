from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
        f"- Missing: {', '.join(q.get('missing') or [])}",
        f"- Warnings: {', '.join(q.get('warnings') or [])}",
        "",
        "## Collected Nodes",
    ]
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
