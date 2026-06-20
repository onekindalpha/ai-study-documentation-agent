from __future__ import annotations

import argparse
import json
import sys
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that the app URL-only auto collector runs before article generation.")
    parser.add_argument("url", help="Seed URL to collect through the running app")
    parser.add_argument("--app", default="http://127.0.0.1:7860", help="Running app base URL")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    payload = json.dumps(
        {
            "url": args.url,
            "timeout_seconds": args.timeout_seconds,
        }
    ).encode("utf-8")
    request = Request(
        args.app.rstrip("/") + "/api/debug-collect-url",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=args.timeout_seconds + 30) as response:
        data = json.loads(response.read().decode("utf-8"))

    report = data.get("collector_report") or {}
    graph = report.get("source_graph") or {}
    stats = graph.get("stats") or report.get("stats") or {}
    quality = graph.get("quality") or report.get("quality") or {}

    print("=== APP AUTO-COLLECT SMOKE TEST ===")
    print(f"ok: {data.get('ok')}")
    print(f"quality_sufficient: {data.get('quality_sufficient')}")
    print(f"quality_reasons: {data.get('quality_reasons')}")
    print(f"markdown_path: {report.get('markdown_path')}")
    print(f"json_path: {report.get('json_path')}")
    print(f"elapsed_seconds: {report.get('elapsed_seconds')}")
    print(f"source_pack_chars: {data.get('source_pack_chars')}")
    print(f"stats: {json.dumps(stats, ensure_ascii=False)}")
    print(f"quality: {json.dumps(quality, ensure_ascii=False)}")
    print(f"collector_error: {report.get('error')}")
    print("\n=== PREVIEW ===")
    print((data.get("source_pack_preview") or "")[:1200])

    return 0 if data.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
