#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "main.py"
REQ = ROOT / "requirements.txt"

NEW_FUNC = r'''def run_source_pack_collector(url: str, timeout_seconds: int = 300, run_id: str = "") -> tuple[str, dict[str, Any]]:
    """Run URL evidence collection.

    v2 tries open-source extractor stack first for web/Oopy/WikiDocs/YouTube.
    AI Skills Navigator still falls back to the existing specialized Playwright/API collector,
    because that collector knows the player tree/API/lab flow.
    """
    safe_run = re.sub(r"[^a-zA-Z0-9_-]+", "_", run_id or make_generation_run_id())
    output_dir = DATA_DIR / "source_packs" / safe_run
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = DATA_DIR / "browser_profiles" / safe_run

    started = time.perf_counter()
    v2_report: dict[str, Any] = {}

    # 1) New source graph v2 collector.
    # It is preferred for Oopy/Notion/WikiDocs/general web/YouTube and will deliberately
    # return non-zero for AI Skills Navigator so the specialized collector can handle it.
    v2_script = BASE_DIR / "tools" / "collect_source_graph_v2.py"
    if v2_script.exists():
        v2_cmd = [
            os.sys.executable,
            str(v2_script),
            url,
            "--out",
            str(output_dir),
            "--run-id",
            safe_run,
            "--max-pages",
            "24",
            "--max-depth",
            "2",
        ]
        try:
            v2_proc = subprocess.run(
                v2_cmd,
                cwd=str(BASE_DIR),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=min(timeout_seconds, 210),
                check=False,
            )
            v2_report = {
                "collector_v2_returncode": v2_proc.returncode,
                "collector_v2_stdout": (v2_proc.stdout or "")[-6000:],
                "collector_v2_stderr": (v2_proc.stderr or "")[-6000:],
            }
            md_candidates = sorted(output_dir.glob("*source_graph_v2*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
            json_candidates = sorted(output_dir.glob("*source_graph_v2*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if v2_proc.returncode == 0 and md_candidates:
                md_path = md_candidates[0]
                source_text = md_path.read_text(encoding="utf-8", errors="replace")
                report: dict[str, Any] = {
                    "ok": True,
                    "collector": "source_graph_v2",
                    "run_id": safe_run,
                    "seed_url": url,
                    "markdown_path": str(md_path),
                    "json_path": str(json_candidates[0]) if json_candidates else "",
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "stdout": (v2_proc.stdout or "")[-6000:],
                    "stderr": (v2_proc.stderr or "")[-6000:],
                }
                if json_candidates:
                    try:
                        import json as _json
                        data = _json.loads(json_candidates[0].read_text(encoding="utf-8", errors="replace"))
                        report["json"] = data
                        report["stats"] = data.get("stats") or {}
                        report["source_graph"] = source_text[:20000]
                    except Exception as e:
                        report["json_load_error"] = f"{type(e).__name__}: {e}"
                return source_text, report
        except subprocess.TimeoutExpired as exc:
            v2_report = {
                "collector_v2_returncode": "timeout",
                "collector_v2_stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                "collector_v2_stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            }
        except Exception as e:
            v2_report = {"collector_v2_error": f"{type(e).__name__}: {e}"}

    # 2) Existing specialized collector fallback.
    before = {path.resolve() for path in output_dir.glob("*.md")}
    cmd = [
        os.sys.executable,
        str(BASE_DIR / "tools" / "collect_source_pack.py"),
        url,
        "--headless",
        "--out",
        str(output_dir),
        "--no-manual-pause",
        "--follow-labs",
        "--follow-limit",
        "8",
        "--crawl-limit",
        "10",
        "--tree-limit",
        "24",
        "--user-data-dir",
        str(profile_dir),
        "--auto-login-wait",
        "45",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return "", {
            "ok": False,
            "error": f"source pack collector timeout after {timeout_seconds}s",
            "stdout": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "cmd": cmd,
            "output_dir": str(output_dir),
            **v2_report,
        }

    markdown_files = sorted(output_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    new_files = [p for p in markdown_files if p.resolve() not in before]
    selected = new_files[0] if new_files else (markdown_files[0] if markdown_files else None)
    report_path = selected.with_suffix(".report.md") if selected else None
    json_candidates = sorted(output_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    json_path = json_candidates[0] if json_candidates else None

    report: dict[str, Any] = {
        "ok": proc.returncode == 0 and selected is not None,
        "collector": "legacy_specialized",
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "")[-6000:],
        "stderr": (proc.stderr or "")[-6000:],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "cmd": cmd,
        "output_dir": str(output_dir),
        "markdown_path": str(selected) if selected else "",
        "json_path": str(json_path) if json_path else "",
        "report_path": str(report_path) if report_path and report_path.exists() else "",
        **v2_report,
    }
    if proc.returncode != 0:
        report["error"] = f"source pack collector failed with code {proc.returncode}"
    if selected and selected.exists():
        source_text = selected.read_text(encoding="utf-8", errors="replace")
        if report_path and report_path.exists():
            report["report_text"] = report_path.read_text(encoding="utf-8", errors="replace")[-20000:]
        if json_path and json_path.exists():
            try:
                import json as _json
                data = _json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
                report["json"] = data
                if isinstance(data, dict):
                    report["stats"] = data.get("stats") or data.get("quality") or {}
            except Exception as e:
                report["json_load_error"] = f"{type(e).__name__}: {e}"
        return source_text, report
    return "", report
'''


def replace_func(text: str) -> str:
    start = text.find("def run_source_pack_collector(")
    if start < 0:
        raise SystemExit("run_source_pack_collector function not found")
    # The next function is append_video_transcript_evidence in current builds.
    end = text.find("def append_video_transcript_evidence", start)
    if end < 0:
        raise SystemExit("append_video_transcript_evidence anchor not found")
    return text[:start] + NEW_FUNC.rstrip() + "\n\n" + text[end:]


def append_reqs():
    add = [
        "requests>=2.32.0",
        "beautifulsoup4>=4.12.3",
        "trafilatura>=1.12.2",
        "crawl4ai>=0.4.248",
        "yt-dlp>=2025.1.15",
        "youtube-transcript-api>=0.6.2",
    ]
    cur = REQ.read_text(encoding="utf-8") if REQ.exists() else ""
    lines = {ln.strip().split("==")[0].split(">=")[0].lower(): ln.strip() for ln in cur.splitlines() if ln.strip() and not ln.strip().startswith("#")}
    out = cur.rstrip() + "\n" if cur.strip() else ""
    for dep in add:
        key = dep.split(">=")[0].lower()
        if key not in lines:
            out += dep + "\n"
    REQ.write_text(out, encoding="utf-8")


def main():
    text = APP.read_text(encoding="utf-8")
    backup = APP.with_suffix(".py.before_collector_v2")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    APP.write_text(replace_func(text), encoding="utf-8")
    append_reqs()
    print("patched app/main.py with collector v2 hook")
    print("updated requirements.txt")

if __name__ == "__main__":
    main()
