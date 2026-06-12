from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.universal_learning_collector import collect_url

DEFAULT_URLS = [
    "https://youtu.be/d2X38zE7VsU?si=mWMMJgacxsJ3P_H7",
    "https://microsoft.github.io/agent-academy/",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run universal collector smoke tests.")
    parser.add_argument("urls", nargs="*", default=DEFAULT_URLS)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--max-pages", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    results = []
    for url in args.urls:
        print(f"\n===== COLLECT: {url} =====")
        result = collect_url(url, visible=args.visible, max_pages=args.max_pages, max_depth=args.max_depth, timeout=args.timeout)
        graph = result["graph"]
        row = {
            "url": url,
            "run_dir": result["run_dir"],
            "url_type": graph.get("url_type"),
            "status": graph.get("status"),
            "quality": graph.get("quality"),
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    print("\n===== SUMMARY =====")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    failed = [r for r in results if r["status"] == "fail"]
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
