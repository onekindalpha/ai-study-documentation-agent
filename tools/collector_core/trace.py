from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import now_iso


class TraceLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "trace.jsonl"
        self.step = 0

    def event(self, event: str, **payload: Any) -> None:
        self.step += 1
        row = {
            "step": self.step,
            "event": event,
            "created_at": now_iso(),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{self.step:03d}] {event}: {json.dumps(payload, ensure_ascii=False)[:500]}")

    def warning(self, message: str, **payload: Any) -> None:
        self.event("warning", message=message, **payload)

    def error(self, message: str, **payload: Any) -> None:
        self.event("error", message=message, **payload)
