from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.collector_core.detector import detect_url
from tools.collector_core.quality_gate import apply_quality_gate
from tools.collector_core.schema import make_run_id
from tools.collector_core.source_pack_writer import write_outputs
from tools.collector_core.trace import TraceLogger


def load_extractor(name: str):
    if name == "youtube":
        from tools.extractors import youtube as module
    elif name == "agent_academy":
        from tools.extractors import agent_academy as module
    elif name == "wikidocs":
        from tools.extractors import wikidocs as module
    elif name == "oopy":
        from tools.extractors import oopy as module
    elif name == "protected_course":
        from tools.extractors import protected_course as module
    elif name == "ai_skills":
        from tools.extractors import ai_skills as module
    else:
        from tools.extractors import generic_web as module
    return module


def collect_url(
    url: str,
    *,
    output_root: str | Path = "data/source_runs",
    run_id: str | None = None,
    visible: bool = False,
    max_pages: int = 20,
    max_depth: int = 1,
    timeout: int = 30,
) -> dict[str, Any]:
    plan = detect_url(url)
    run_id = run_id or make_run_id(url, plan.extractor)
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace = TraceLogger(run_dir)
    trace.event("collection_started", input_url=url, run_id=run_id)
    trace.event("url_detected", **plan.to_dict())

    extractor = load_extractor(plan.extractor)
    try:
        graph = extractor.collect(
            url,
            run_dir=run_dir,
            trace=trace,
            plan=plan,
            visible=visible,
            max_pages=max_pages,
            max_depth=max_depth,
            timeout=timeout,
        )
    except TypeError:
        graph = extractor.collect(
            url,
            run_dir=run_dir,
            trace=trace,
            plan=plan,
            visible=visible,
            max_pages=max_pages,
            timeout=timeout,
        )
    except Exception as exc:
        trace.error("extractor_failed", extractor=plan.extractor, error=str(exc))
        from tools.collector_core.schema import make_graph
        graph = make_graph(
            input_url=url,
            url_type=plan.extractor,
            site_hint=plan.site_hint,
            content_shape=plan.content_shape,
            navigation_shape=plan.navigation_shape,
            access_level=plan.access_level,
            evidence_targets=plan.evidence_targets,
            title="Collection failed",
        )
        graph["status"] = "fail"
        graph["quality"]["missing"].append(str(exc))

    graph = apply_quality_gate(graph)
    files = write_outputs(run_dir, graph)
    trace.event("collection_finished", status=graph.get("status"), quality=graph.get("quality"), files=files)
    return {
        "ok": graph.get("status") in {"pass", "partial"},
        "run_id": run_id,
        "run_dir": str(run_dir),
        "plan": plan.to_dict(),
        "graph": graph,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Universal Learning Source Collector v0")
    parser.add_argument("url", help="Learning URL to collect")
    parser.add_argument("--output-root", default="data/source_runs")
    parser.add_argument("--run-id")
    parser.add_argument("--visible", action="store_true", help="Open browser visibly for rendered pages")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--json", action="store_true", help="Print compact JSON result")
    args = parser.parse_args()

    result = collect_url(
        args.url,
        output_root=args.output_root,
        run_id=args.run_id,
        visible=args.visible,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        timeout=args.timeout,
    )
    graph = result["graph"]
    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "graph"} | {"status": graph.get("status"), "quality": graph.get("quality")}, ensure_ascii=False, indent=2))
    else:
        print("\n=== UNIVERSAL COLLECTOR RESULT ===")
        print(f"run_dir: {result['run_dir']}")
        print(f"url_type: {graph.get('url_type')}")
        print(f"status: {graph.get('status')}")
        print(f"title: {graph.get('title')}")
        print("quality:")
        print(json.dumps(graph.get("quality") or {}, ensure_ascii=False, indent=2))
        print("files:")
        for label, path in result["files"].items():
            print(f"- {label}: {path}")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
