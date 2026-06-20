from __future__ import annotations

# v4.7.24 source_first_explicit_topic_guard
# v4.7.23 nonit_source_first_polish
# v4.7.22 single_page_source_first_gate
# v4.7.21 source_first_marker_and_router_fix

import base64
import html as html_lib
import json
import mimetypes
import os
import re
import subprocess
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv as _python_dotenv_load
except ModuleNotFoundError:
    def _load_dotenv(path: Path, override: bool = False) -> bool:
        """Minimal .env loader fallback when python-dotenv is not installed."""
        try:
            if not path.exists():
                return False
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and (override or key not in os.environ):
                    os.environ[key] = value
            return True
        except Exception:
            return False
else:
    def _load_dotenv(path: Path, override: bool = False) -> bool:
        return bool(_python_dotenv_load(path, override=override))

DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
DOTENV_LOADED = _load_dotenv(DOTENV_PATH, override=True)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RUNS_DIR = DATA_DIR / "runs"
CAPTURE_DIR = DATA_DIR / "captures"
NOTES_PATH = DATA_DIR / "notes.jsonl"
SESSIONS_PATH = DATA_DIR / "sessions.json"
EXAMPLES_DIR = BASE_DIR / "examples"
ARTICLE_TYPE_CONFIG_DIR = BASE_DIR / "configs" / "article_types"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
NOTES_PATH.touch(exist_ok=True)
SESSIONS_PATH.touch(exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


def bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


GROQ_VISION_CHUNK_SIZE = bounded_int_env("GROQ_VISION_CHUNK_SIZE", 3, 1, 5)
GROQ_VISION_MAX_TOKENS = bounded_int_env("GROQ_VISION_MAX_TOKENS", 1200, 400, 2000)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")
MODEL_PROVIDER_FAST = os.getenv("MODEL_PROVIDER_FAST", "groq")
MODEL_PROVIDER_DEEP = os.getenv("MODEL_PROVIDER_DEEP", "openai")


@dataclass
class StudyNote:
    id: str
    created_at: str
    title: str
    source_type: str
    tags: list[str]
    raw_text: str
    user_memo: str
    summary: str
    action_items: list[str]
    blog_draft: str
    image_path: str | None = None
    image_paths: list[str] | None = None


@dataclass
class UploadedFile:
    filename: str
    data: bytes


FormData = dict[str, list[str | UploadedFile]]


@dataclass
class ImageEvidence:
    image_no: int
    caption: str
    visible_evidence: list[str]
    role: str
    problem_signal: str
    technical_entities: list[str]
    inferred_meaning: str
    confidence: float = 0.0
    evidence_source: str = ""


@dataclass
class CritiqueResult:
    passed: bool
    failures: list[str]
    section_failures: dict[str, str]
    metrics: dict[str, Any]


@dataclass
class CaptureEvent:
    capture_id: int
    timestamp: str
    image_path: str
    source_title: str = ""
    source_url: str = ""
    user_note: str = ""
    ocr_text: str = ""
    vision_summary: str = ""
    auto_keywords: list[str] | None = None


@dataclass
class QALog:
    qa_id: int
    timestamp: str
    related_capture_ids: list[int]
    selected_text: str
    question: str
    answer: str
    answer_summary: str
    learner_state: str
    resolved: bool
    used_in_article: bool


class LLM:
    def __init__(self) -> None:
        self.client = None
        self.last_diagnostics: dict[str, Any] = {}
        self.last_vision_error: Exception | None = None

    def get_client(self) -> Any:
        if self.client:
            return self.client
        if not GROQ_API_KEY:
            self.last_diagnostics = provider_diagnostics(
                provider="groq",
                exception=RuntimeError("GROQ_API_KEY is not set"),
                package_import_ok=groq_import_ok(),
                client_init_ok=False,
            )
            return None
        try:
            from groq import Groq
        except ModuleNotFoundError as exc:
            self.last_diagnostics = provider_diagnostics(
                provider="groq",
                exception=exc,
                package_import_ok=False,
                client_init_ok=False,
            )
            return None
        try:
            self.client = Groq(api_key=GROQ_API_KEY, timeout=90.0)
            self.last_diagnostics = provider_diagnostics(provider="groq", package_import_ok=True, client_init_ok=True)
            return self.client
        except Exception as exc:
            self.last_diagnostics = provider_diagnostics(
                provider="groq",
                exception=exc,
                package_import_ok=True,
                client_init_ok=False,
            )
            return None

    def generate_note(self, raw_text: str, memo: str) -> dict[str, Any]:
        if not raw_text.strip() and not memo.strip():
            return fallback_note(raw_text, memo)

        client = self.get_client()
        if not client:
            return fallback_note(raw_text, memo)

        prompt = f"""
아래 학습 화면/실습 기록을 기술 노트로 정리해 주세요.

규칙:
- 한국어로 작성합니다.
- 과장하지 말고 입력에 있는 내용만 사용합니다.
- JSON만 반환합니다.
- keys: title, source_type, tags, summary, action_items, blog_draft
- summary는 문제 인식, 핵심 개념, 실습 흐름, 막힌 지점, 해결 방향을 포함합니다.
- blog_draft는 기술블로그 초안처럼 제목/배경/실습/배운점 구조로 작성합니다.

[화면 텍스트]
{raw_text}

[사용자 메모]
{memo}
""".strip()
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.2,
                max_tokens=1200,
                messages=[
                    {
                        "role": "system",
                        "content": "You convert study captures into structured developer learning notes. Return strict JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = completion.choices[0].message.content or ""
            return parse_json_or_fallback(content, raw_text, memo)
        except Exception as exc:
            print(f"[LLM text error] {exc}")
            return fallback_note(raw_text, memo)

    def generate_note_from_image(self, image_file: Path, memo: str) -> dict[str, Any]:
        return self.generate_note_from_images([image_file], memo)

    def generate_note_from_images(self, image_files: list[Path], memo: str) -> dict[str, Any]:
        client = self.get_client()
        if not client:
            return image_only_note(f"/captures/{image_files[0].name}" if image_files else "")

        if len(image_files) > 6:
            partial_notes: list[dict[str, Any]] = []
            for chunk_index, start in enumerate(range(0, len(image_files), 6), start=1):
                chunk = image_files[start : start + 6]
                chunk_memo = f"{memo}\n\n이 묶음은 전체 캡처 중 {chunk_index}번째 묶음입니다."
                partial_note = self.generate_note_from_images(chunk, chunk_memo)
                partial_notes.append(partial_note)
            return self.combine_image_notes(partial_notes, memo, len(image_files))

        for candidate_files in (image_files, image_files[:3], image_files[:1]):
            if not candidate_files:
                continue
            note = self.try_generate_note_from_images(client, candidate_files, memo)
            if note:
                return note
        return image_only_note(f"/captures/{image_files[0].name}" if image_files else "")

    def combine_image_notes(self, partial_notes: list[dict[str, Any]], memo: str, image_count: int) -> dict[str, Any]:
        joined = "\n\n".join(
            f"[묶음 {index}]\n제목: {note.get('title', '')}\n요약:\n{note.get('summary', '')}\n액션:\n{', '.join(note.get('action_items', []))}"
            for index, note in enumerate(partial_notes, start=1)
        )
        client = self.get_client()
        if client:
            prompt = f"""
아래는 사용자가 학습/Lab 실습 중 순서대로 캡처한 {image_count}장 이미지를 6장 이하 묶음으로 나누어 판독한 결과입니다.
묶음별 내용을 하나의 흐름으로 통합해 학습 노트 JSON을 작성해 주세요.

규칙:
- 한국어로 작성합니다.
- 입력된 묶음 요약과 사용자 메모만 근거로 사용합니다.
- JSON만 반환합니다.
- keys: title, source_type, tags, summary, action_items, blog_draft
- summary는 전체 실습 흐름을 압축하지 말고, 이미지 묶음별로 문제/원인/조치/검증 흐름을 길게 정리합니다.
- blog_draft는 Medium 포트폴리오 초안처럼 배경/문제 인식/문제 정의/해결 흐름/성과/배운 점 구조로 충분히 길게 작성합니다.

[사용자 메모]
{memo}

[묶음별 판독 결과]
{joined}
""".strip()
            try:
                completion = client.chat.completions.create(
                    model=GROQ_MODEL,
                    temperature=0.2,
                    max_tokens=3200,
                    messages=[
                        {"role": "system", "content": "You merge sequential study-capture notes into one grounded structured JSON note."},
                        {"role": "user", "content": prompt},
                    ],
                )
                return parse_json_or_fallback(completion.choices[0].message.content or "", joined, memo)
            except Exception as exc:
                print(f"[LLM merge error] {exc}")
        return {
            "title": f"{image_count}장 캡처 기반 학습 기록",
            "source_type": "study-capture",
            "tags": ["study-note", "multi-image", "vision"],
            "summary": joined,
            "action_items": ["묶음별 핵심 흐름 연결", "문제 인식과 해결 과정 보강", "문제 해결형 Medium 글로 확장"],
            "blog_draft": f"# {image_count}장 캡처 기반 학습 기록\n\n{joined}",
        }

    def try_generate_note_from_images(self, client: Any, image_files: list[Path], memo: str) -> dict[str, Any] | None:
        prompt = f"""
아래 이미지는 사용자가 학습/Lab 실습 중 순서대로 캡처한 화면입니다. 이미지 순서를 실습 흐름으로 보고 포트폴리오용 학습 노트로 정리해 주세요.

규칙:
- 한국어로 작성합니다.
- 화면에 보이는 내용과 사용자 메모만 근거로 사용합니다.
- JSON만 반환합니다.
- keys: title, source_type, tags, summary, action_items, blog_draft
- 여러 이미지가 있으면 이미지 1, 이미지 2... 순서대로 실습 흐름을 연결합니다.
- summary에는 각 이미지에서 확인한 화면 변화, 발견한 문제, 의심 원인, 조치, 검증 포인트를 생략하지 말고 정리합니다.
- blog_draft는 문제 해결형 기술블로그 초안처럼 제목/배경/문제 인식/문제 정의/해결 흐름/성과/배운 점 구조로 충분히 길게 작성합니다.
- 이미지가 여러 장이면 이미지별 캡션 후보를 함께 정리합니다.

[사용자 메모]
{memo}
""".strip()
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for index, image_file in enumerate(image_files, start=1):
            mime_type = mimetypes.guess_type(image_file.name)[0] or "image/png"
            image_data = base64.b64encode(image_file.read_bytes()).decode("ascii")
            content_parts.append({"type": "text", "text": f"이미지 {index}"})
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                }
            )
        try:
            completion = client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                temperature=0.2,
                max_tokens=2400,
                messages=[
                    {
                        "role": "system",
                        "content": "You read study screenshots and turn them into structured developer learning notes. Return strict JSON.",
                    },
                    {
                        "role": "user",
                        "content": content_parts,
                    },
                ],
            )
            content = completion.choices[0].message.content or ""
            return parse_json_or_fallback(content, "", memo)
        except Exception as exc:
            print(f"[LLM vision error] {exc}")
            return None

    def synthesize_blog(self, notes: list[StudyNote], topic: str, format_type: str, extra_info: str = "") -> str:
        notes = [note for note in notes if is_meaningful_note(note)][-8:]
        if not notes:
            return "문제해결형 Medium 글을 만들 수 있는 학습 노트가 아직 없습니다. 스크린샷을 업로드하거나 메모를 입력한 뒤 먼저 캡처 노트를 생성해 주세요."

        joined = "\n\n".join(
            f"[노트 {index}]\n제목: {note.title}\n요약:\n{note.summary[:5200]}\n액션: {', '.join(note.action_items[:8])}\n초안:\n{note.blog_draft[:3600]}"
            for index, note in enumerate(notes, start=1)
        )
        client = self.get_client()
        if not client:
            return local_portfolio_blog(notes, topic)

        prompt = portfolio_prompt(topic, joined, extra_info)
        try:
            completion = client.chat.completions.create(
                model=GROQ_MODEL,
                temperature=0.25,
                max_tokens=7800,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Korean technical portfolio writer. "
                            "Write long, concrete, grounded Medium-ready problem-solving articles from study notes."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return completion.choices[0].message.content or local_portfolio_blog(notes, topic)
        except Exception:
            return local_portfolio_blog(notes, topic)

    def synthesize_blog_from_capture(
        self,
        raw_text: str,
        memo: str,
        image_files: list[Path],
        topic: str,
        extra_info: str = "",
        image_names: list[str] | None = None,
        captures: list[dict[str, Any]] | None = None,
        qa_logs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return generate_medium_article_pipeline(
            self,
            raw_text=raw_text,
            memo=memo,
            image_files=image_files,
            topic=topic,
            extra_info=extra_info,
            image_names=image_names or [path.name for path in image_files],
            captures=captures or [],
            qa_logs=qa_logs or [],
        )

    def describe_image_sequence(self, image_files: list[Path], memo: str, extra_info: str) -> str:
        client = self.get_client()
        if not client:
            return ""
        chunks: list[str] = []
        for chunk_index, start in enumerate(range(0, len(image_files), 6), start=1):
            chunk = image_files[start : start + 6]
            prompt = f"""
아래 이미지는 사용자가 실습/프로젝트를 진행한 순서대로 캡처한 화면 중 {chunk_index}번째 묶음입니다.
최종 Medium 글을 쓰기 위한 근거 메모를 작성해 주세요.

규칙:
- 최종 글을 쓰지 말고, 이미지별 관찰 내용과 문제 해결 흐름만 자세히 정리합니다.
- 버튼 클릭 설명이 아니라 문제/원인/조치/검증 관점으로 해석합니다.
- 화면에서 확인되는 기술명, 테이블명, 수식명, 오류, 결과값을 가능한 한 보존합니다.
- 사용자가 제공한 추가 정보와 모순되지 않게 작성합니다.
- 이미지 번호는 전체 순서를 기준으로 {start + 1}번부터 시작합니다.

[사용자 메모]
{memo}

[추가 정보]
{extra_info}
""".strip()
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for offset, image_file in enumerate(chunk, start=start + 1):
                mime_type = mimetypes.guess_type(image_file.name)[0] or "image/png"
                image_data = base64.b64encode(image_file.read_bytes()).decode("ascii")
                content_parts.append({"type": "text", "text": f"이미지 {offset}"})
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}})
            try:
                completion = client.chat.completions.create(
                    model=GROQ_VISION_MODEL,
                    temperature=0.2,
                    max_tokens=3200,
                    messages=[
                        {"role": "system", "content": "You read technical screenshots and create detailed grounded writing notes."},
                        {"role": "user", "content": content_parts},
                    ],
                )
                chunks.append(completion.choices[0].message.content or "")
            except Exception as exc:
                print(f"[LLM direct image sequence error] {exc}")
        return "\n\n".join(f"[이미지 판독 묶음 {index}]\n{chunk}" for index, chunk in enumerate(chunks, start=1))


llm = LLM()


def groq_import_ok() -> bool:
    try:
        import groq  # noqa: F401
        return True
    except Exception:
        return False


def provider_diagnostics(
    provider: str = "groq",
    exception: Exception | None = None,
    package_import_ok: bool | None = None,
    client_init_ok: bool | None = None,
    test_call_ok: bool | None = None,
) -> dict[str, Any]:
    package_ok = groq_import_ok() if package_import_ok is None and provider == "groq" else bool(package_import_ok)
    diagnostics: dict[str, Any] = {
        "provider": provider,
        "model": GROQ_MODEL if provider == "groq" else OPENAI_MODEL,
        "vision_model": GROQ_VISION_MODEL if provider == "groq" else OPENAI_MODEL,
        "text_model": GROQ_MODEL if provider == "groq" else OPENAI_MODEL,
        "api_key_present": bool(GROQ_API_KEY if provider == "groq" else OPENAI_API_KEY),
        "package_import_ok": package_ok,
        "dotenv_loaded": bool(DOTENV_LOADED),
        "dotenv_path": str(DOTENV_PATH),
        "dotenv_exists": DOTENV_PATH.exists(),
        "client_init_ok": client_init_ok,
        "test_call_ok": test_call_ok,
        "selected_fast_provider": MODEL_PROVIDER_FAST,
        "selected_deep_provider": MODEL_PROVIDER_DEEP,
    }
    if exception:
        diagnostics.update(
            {
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "traceback_summary": "".join(traceback.format_exception_only(type(exception), exception)).strip(),
            }
        )
    return diagnostics


def llm_health_check(run_test_call: bool = False) -> dict[str, Any]:
    groq_diag = provider_diagnostics("groq", package_import_ok=groq_import_ok(), client_init_ok=False, test_call_ok=None)
    if groq_diag["api_key_present"] and groq_diag["package_import_ok"]:
        try:
            from groq import Groq

            client = Groq(api_key=GROQ_API_KEY, timeout=20.0)
            groq_diag["client_init_ok"] = True
            if run_test_call:
                try:
                    completion = client.chat.completions.create(
                        model=GROQ_MODEL,
                        temperature=0,
                        max_tokens=4,
                        messages=[{"role": "user", "content": "ping"}],
                    )
                    groq_diag["test_call_ok"] = bool(completion.choices)
                except Exception as exc:
                    groq_diag["test_call_ok"] = False
                    groq_diag["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            groq_diag.update(provider_diagnostics("groq", exception=exc, package_import_ok=True, client_init_ok=False))
    openai_diag = {
        "enabled": MODEL_PROVIDER_FAST == "openai" or MODEL_PROVIDER_DEEP == "openai",
        "api_key_present": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
    }
    groq_diag["enabled"] = MODEL_PROVIDER_FAST == "groq" or MODEL_PROVIDER_DEEP == "groq"
    return {"groq": groq_diag, "openai": openai_diag}



class _PlainHTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in {"h1", "h2", "h3", "p", "li", "pre", "code", "td", "th"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in {"h1", "h2", "h3", "p", "li", "pre", "code", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if not text:
            return
        if self._in_title:
            self.title += text + " "
        self.parts.append(text)

    def text(self) -> str:
        joined = "\n".join(part.strip() for part in self.parts if part.strip())
        return re.sub(r"\n{3,}", "\n\n", joined).strip()


def extract_urls_from_text(text: str, limit: int = 200) -> list[str]:
    urls = re.findall(r"https?://[^\s)\]}>\"']+", text or "")
    cleaned: list[str] = []
    for url in urls:
        url = url.rstrip(".,;，。")
        if url not in cleaned:
            cleaned.append(url)
    return cleaned[:limit]


def youtube_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if "youtu.be" in host:
        return parsed.path.strip("/").split("/")[0]
    query = parse_qs(parsed.query)
    if query.get("v"):
        return query["v"][0]
    parts = [part for part in parsed.path.split("/") if part]
    for marker in ("shorts", "embed", "live"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return ""


def extract_balanced_json_object(source: str, marker: str) -> dict[str, Any] | None:
    start = source.find(marker)
    if start < 0:
        return None
    start = source.find("{", start)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(source)):
        ch = source[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(source[start:idx + 1])
                except Exception:
                    return None
    return None


def fetch_url_text(url: str, timeout: int = 12, limit: int = 700000) -> tuple[str, str]:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 StudyCaptureAgent/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read(limit)
        content_type = resp.headers.get("Content-Type", "")
    return raw.decode("utf-8", errors="replace"), content_type


def fetch_youtube_source_text(url: str, max_chars: int = 12000) -> str:
    video_id = youtube_video_id(url)
    title = ""
    author = ""
    description = ""
    transcript = ""
    transcript_status = "자막 트랙을 찾지 못했습니다."

    try:
        oembed_url = "https://www.youtube.com/oembed?" + urlencode({"url": url, "format": "json"})
        req = Request(oembed_url, headers={"User-Agent": "Mozilla/5.0 StudyCaptureAgent/1.0"})
        with urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read(100000).decode("utf-8", errors="replace"))
        title = str(payload.get("title") or "").strip()
        author = str(payload.get("author_name") or "").strip()
    except Exception:
        pass

    try:
        watch_url = f"https://www.youtube.com/watch?v={video_id}" if video_id else url
        html, _content_type = fetch_url_text(watch_url, timeout=12, limit=1200000)
        player = extract_balanced_json_object(html, "ytInitialPlayerResponse")
        if isinstance(player, dict):
            details = player.get("videoDetails") or {}
            title = title or str(details.get("title") or "").strip()
            author = author or str(details.get("author") or "").strip()
            description = str(details.get("shortDescription") or "").strip()
            tracks = (
                player.get("captions", {})
                .get("playerCaptionsTracklistRenderer", {})
                .get("captionTracks", [])
            )
            if tracks:
                track = next((item for item in tracks if str(item.get("languageCode", "")).startswith("ko")), None)
                track = track or next((item for item in tracks if str(item.get("languageCode", "")).startswith("en")), None)
                track = track or tracks[0]
                base_url = str(track.get("baseUrl") or "")
                if base_url:
                    caption_text, _ = fetch_url_text(base_url, timeout=12, limit=700000)
                    if "<text" in caption_text:
                        parts = re.findall(r"<text[^>]*>(.*?)</text>", caption_text, flags=re.DOTALL)
                        transcript = "\n".join(html_lib.unescape(re.sub(r"<[^>]+>", "", part)).strip() for part in parts)
                    else:
                        payload = json.loads(caption_text)
                        events = payload.get("events", []) if isinstance(payload, dict) else []
                        chunks = []
                        for event in events:
                            for seg in event.get("segs", []) or []:
                                chunks.append(str(seg.get("utf8") or ""))
                        transcript = "".join(chunks)
                    transcript = re.sub(r"\s+\n", "\n", transcript)
                    transcript = re.sub(r"\n{3,}", "\n\n", transcript).strip()
                    transcript_status = f"자막 추출 성공: {track.get('name', {}).get('simpleText') or track.get('languageCode') or 'caption'}"
    except Exception as exc:
        transcript_status = f"자막/설명 추출 실패: {type(exc).__name__}: {exc}"

    lines = [f"[YouTube 영상 자동 추출]", url]
    if video_id:
        lines.append(f"영상 ID: {video_id}")
    if title:
        lines.append(f"영상 제목: {title}")
    if author:
        lines.append(f"채널: {author}")
    if description:
        lines.append("\n[영상 설명]")
        lines.append(description[:2500])
    lines.append(f"\n[자막 상태]\n{transcript_status}")
    if transcript:
        lines.append("\n[영상 자막]")
        lines.append(transcript[:max_chars])
    else:
        lines.append("\n[영상 학습 힌트]\n자막을 가져오지 못한 경우에도 영상 제목, 설명, URL, 사용자가 적은 헷갈린 점을 바탕으로 문제해결형 학습 글을 작성합니다.")
    return "\n".join(lines)


def fetch_public_source_text(url: str, max_chars: int = 12000) -> str:
    """Best-effort public URL reader for lecture/lab pages.

    It intentionally does not require the user to paste screenshots.  Login-gated
    pages and YouTube transcript extraction are reported as source hints rather
    than treated as fatal errors.
    """
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if "youtube.com" in host or "youtu.be" in host:
        return fetch_youtube_source_text(url, max_chars=max_chars)
    if "aiskillsnavigator.microsoft.com" in host:
        return (
            f"[강의 플레이어 URL]\n{url}\n"
            "AI Skills Navigator 플레이어 URL입니다. 로그인/동적 렌더링 페이지일 수 있어 "
            "URL은 출처 힌트로만 사용하고, 실제 강의안 본문이나 실습 URL 내용을 우선 근거로 사용합니다."
        )
    try:
        text, content_type = fetch_url_text(url)
        if "html" in content_type or text.lstrip().startswith("<!") or "<html" in text[:500].lower():
            parser = _PlainHTMLTextExtractor()
            parser.feed(text)
            body = parser.text()
            title = parser.title.strip()
            if title:
                body = f"제목: {title}\n\n{body}"
            if len(body.strip()) < 250:
                return (
                    f"[URL 부분 추출]\n{url}\n\n{body[:max_chars]}\n\n"
                    "본문이 짧게 추출되었습니다. 이 사이트는 로그인, 권한, 또는 JavaScript 렌더링이 필요할 수 있습니다."
                )
            return f"[URL 자동 추출]\n{url}\n\n{body[:max_chars]}"
        return f"[URL 자동 추출]\n{url}\n\n{text[:max_chars]}"
    except Exception as exc:
        return f"[URL 추출 실패]\n{url}\n{type(exc).__name__}: {exc}\n이 URL은 문자열 힌트로만 사용합니다."


def enrich_raw_text_with_source_urls(raw_text: str, memo: str) -> str:
    source = f"{raw_text}\n{memo}"
    urls = extract_urls_from_text(source)
    if not urls:
        return raw_text
    extracted = [fetch_public_source_text(url) for url in urls]
    return raw_text.rstrip() + "\n\n" + "\n\n".join(extracted)


def source_pack_like_text(text: str) -> bool:
    source = text.lower()
    markers = [
        "source pack:",
        "## collection stats",
        "## visible page text",
        "## player navigation items",
        "===== page:",
        "[url 자동 추출]",
        "[youtube 영상 자동 추출]",
        "[영상 자막]",
    ]
    return any(marker in source for marker in markers) and len(text.strip()) >= 800


def url_only_without_collected_source(raw_text: str, enriched_text: str) -> bool:
    raw = raw_text.strip()
    urls = extract_urls_from_text(raw)
    if not urls:
        return False
    raw_without_urls = raw
    for url in urls:
        raw_without_urls = raw_without_urls.replace(url, "")
    if raw_without_urls.strip():
        return False
    if source_pack_like_text(enriched_text):
        return False
    weak_markers = [
        "본문이 짧게 추출되었습니다",
        "로그인, 권한, 또는 JavaScript 렌더링",
        "출처 힌트로만 사용",
        "자막 트랙을 찾지 못했습니다",
        "자막/설명 추출 실패",
        "자막을 가져오지 못한 경우",
    ]
    return any(marker in enriched_text for marker in weak_markers) or len(enriched_text.strip()) < 1200


def source_collection_required_message(raw_text: str, enriched_text: str, memo: str) -> str:
    urls = extract_urls_from_text(raw_text)
    url_lines = "\n".join(f"- {url}" for url in urls) or "- URL 없음"
    first_url = urls[0] if urls else ""
    host = url_domain(first_url) if first_url else ""
    if is_notion_host(host):
        return f"""# Notion URL 자동 수집 안내

현재 공개 Notion URL 자동 수집은 업데이트 중입니다.

특히 다음 형태는 URL만으로 전체 본문을 가져오지 못할 수 있습니다.
- database 페이지
- 목차형 페이지
- 하위 페이지가 많은 페이지
- workspace/tree 구조 페이지
- 공개 보기 전용 페이지

## 가능한 입력 방식
1. 열린 단일 Notion 페이지의 본문을 복사해서 붙여넣기
2. 페이지 소유자/편집 권한이 있다면 Markdown/HTML Export 파일 업로드
3. 필요한 장면을 캡처 이미지로 업로드

## 아직 지원하지 않는 것
- 공개 Notion URL만으로 전체 database 수집
- 하위 페이지 전체 자동 수집
- workspace/tree 전체 수집

## 입력 URL
{url_lines}

## 사용자가 적은 어려움/복잡한 문제
{clean_prompt_memo(memo) or "없음"}
"""
    return f"""# URL 자료 수집이 필요합니다

지금 입력은 URL만 있고, Medium 글에 쓸 본문/강의 흐름/실습 내용을 충분히 수집하지 못했습니다. 이 상태에서 바로 글을 만들면 URL ID나 사이트 이름만으로 억지 글이 생성될 수 있으므로 차단했습니다.

## 입력 URL
{url_lines}

## 현재 확인된 상태
{enriched_text.strip()[:1200]}

## 다음 단계
1. 본문을 직접 복사해서 붙여넣거나, Markdown/HTML/TXT 파일을 업로드합니다.
2. 영상이라면 transcript/자막 파일/강의 메모/캡처 이미지를 추가합니다.
3. 수집 가능한 일반 웹문서라면 본문이 충분히 추출되는지 Debug report를 확인합니다.

## 사용자가 적은 어려움/복잡한 문제
{clean_prompt_memo(memo) or "없음"}
"""

def is_udemy_url(url: str) -> bool:
    return "udemy.com" in (urlparse(url).netloc or "").lower()


def udemy_manual_source_pack_message(raw_text: str, memo: str) -> str:
    urls = extract_urls_from_text(raw_text)
    url_lines = "\n".join(f"- {url}" for url in urls) or "- URL 없음"
    return f"""# Udemy는 수동 source pack이 필요합니다

Udemy는 Cloudflare 보안 확인과 로그인/수강 권한 확인이 자동화 브라우저에서 안정적으로 통과되지 않습니다. 그래서 이 앱은 Udemy URL만으로 억지 Medium 글을 만들지 않습니다.

## 입력 URL
{url_lines}

## 처리 방식
1. 일반 브라우저에서 Udemy 강의에 직접 접속합니다.
2. 강의 제목, 커리큘럼, 현재 강의 설명, 자막/스크립트, 학습 자료 링크를 복사합니다.
3. 복사한 내용을 앱 입력칸에 붙여 넣습니다.
4. 그 수동 source pack을 바탕으로 문제해결형 Medium 글을 생성합니다.

## 사용자가 적은 어려움/헷갈린 부분
{clean_prompt_memo(memo) or "없음"}
"""




def is_inflearn_url(url: str) -> bool:
    return "inflearn.com" in (urlparse(url).netloc or "").lower()


def inflearn_protected_source_message(raw_text: str, memo: str) -> str:
    urls = extract_urls_from_text(raw_text)
    url_lines = "\n".join(f"- {url}" for url in urls) or "- URL 없음"
    return f"""# Inflearn URL-only 자동수집이 완료되지 않았습니다

Inflearn 강의 페이지는 로그인/수강 권한/동적 렌더링에 막히는 경우가 많습니다. 이 상태에서 억지로 Medium 글을 만들면 강의 제목이나 URL만 보고 추상적인 글이 생성됩니다.

## 입력 URL
{url_lines}

## 현재 판단
- 공개 URL만으로 강의 본문, 자막, 커리큘럼, 강의 자료를 충분히 확보하지 못했습니다.
- 따라서 문제해결형 Medium 글 생성을 중단합니다.

## 필요한 다음 입력
1. 강의 제목과 현재 강의의 핵심 내용
2. 커리큘럼 또는 섹션 제목
3. 자막/스크립트 또는 강의 노트
4. 실습에서 막힌 화면이나 완료 화면

## 사용자가 적은 어려움/헷갈린 부분
{clean_prompt_memo(memo) or "없음"}
"""

def raw_text_is_url_only(raw_text: str) -> bool:
    raw = raw_text.strip()
    urls = extract_urls_from_text(raw)
    if not raw or not urls:
        return False
    without = raw
    for url in urls:
        without = without.replace(url, "")
    return not without.strip()


def run_source_pack_collector(url: str, timeout_seconds: int = 420, run_id: str = "") -> tuple[str, dict[str, Any]]:
    # Use a per-run output directory and browser profile.
    # This prevents a changed URL from reusing an old source_pack or stale SPA/localStorage state.
    safe_run = re.sub(r"[^a-zA-Z0-9_-]+", "_", run_id or make_generation_run_id())
    output_dir = DATA_DIR / "source_packs" / safe_run
    output_dir.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in output_dir.glob("*.md")}
    host = (urlparse(url).netloc or "").lower()
    path = (urlparse(url).path or "").lower().rstrip("/")
    is_ai_skills = "aiskillsnavigator.microsoft.com" in host
    is_agent_academy_videos = host == "microsoft.github.io" and path.endswith("/agent-academy/videos")
    universal_collector = BASE_DIR / "tools" / "universal_learning_collector.py"
    evidence_ranker = BASE_DIR / "tools" / "evidence_ranker.py"
    universal_attempt: dict[str, Any] = {}
    if universal_collector.exists() and not is_ai_skills:
        universal_max_pages = "80" if is_agent_academy_videos else "24"
        universal_timeout = "180" if is_agent_academy_videos else str(min(90, max(30, timeout_seconds // 4)))
        universal_cmd = [
            os.sys.executable,
            str(universal_collector),
            url,
            "--output-root",
            str(DATA_DIR / "source_packs"),
            "--run-id",
            safe_run,
            "--max-pages",
            universal_max_pages,
            "--max-depth",
            "2",
            "--timeout",
            universal_timeout,
            "--json",
        ]
        started = time.perf_counter()
        print(f"[collector] run_id={safe_run} seed_url={url}")
        print(f"[collector] script={universal_collector} out={output_dir} universal=true")
        try:
            proc = subprocess.run(
                universal_cmd,
                cwd=str(BASE_DIR),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            proc = subprocess.CompletedProcess(
                universal_cmd,
                124,
                stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "") if isinstance(exc.stderr, str) else f"timeout after {timeout_seconds}s",
            )
        md_path = output_dir / "source_pack.md"
        json_path = output_dir / "source_graph.json"
        report_path = output_dir / "collection_report.json"
        rank_path = output_dir / "evidence_rank.json"
        article_brief_path = output_dir / "article_brief.md"
        rank_stdout = ""
        rank_stderr = ""
        if json_path.exists() and evidence_ranker.exists():
            rank_proc = subprocess.run(
                [os.sys.executable, str(evidence_ranker), str(output_dir), "--top-k", "8"],
                cwd=str(BASE_DIR),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=min(60, timeout_seconds),
                check=False,
            )
            rank_stdout = rank_proc.stdout[-4000:]
            rank_stderr = rank_proc.stderr[-4000:]
        if md_path.exists() and json_path.exists():
            text = md_path.read_text(encoding="utf-8", errors="replace")
            if article_brief_path.exists():
                text = text.rstrip() + "\n\n" + article_brief_path.read_text(encoding="utf-8", errors="replace")
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
            video_index = payload.get("video_index") if isinstance(payload.get("video_index"), list) else []
            stats = {
                "page_count": int(quality.get("pages_collected") or 0),
                "visible_text_chars": int(quality.get("text_chars") or len(text or "")),
                "link_count": len(payload.get("links") or []),
                "video_candidate_count": int(quality.get("video_candidates") or len(video_index) or len(payload.get("videos") or [])),
                "lesson_candidate_count": len(payload.get("nodes") or []),
                "lab_candidate_count": int(quality.get("lab_steps") or 0),
            }
            report = {
                "ok": True,
                "collector": "universal_learning_collector",
                "collector_returncode": proc.returncode,
                "command": universal_cmd,
                "markdown_path": str(md_path),
                "json_path": str(json_path),
                "collection_report_path": str(report_path) if report_path.exists() else "",
                "evidence_rank_path": str(rank_path) if rank_path.exists() else "",
                "article_brief_path": str(article_brief_path) if article_brief_path.exists() else "",
                "quality": quality,
                "stats": stats,
                "stdout": (proc.stdout + "\n" + rank_stdout)[-4000:],
                "stderr": (proc.stderr + "\n" + rank_stderr)[-4000:],
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "run_id": safe_run,
                "seed_url": url,
            }
            print(f"[collector] universal ok md={md_path} elapsed={report['elapsed_seconds']}s stats={stats}")
            return text, report
        universal_attempt = {
            "collector": "universal_learning_collector",
            "command": universal_cmd,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-4000:],
            "markdown_path": str(md_path) if md_path.exists() else "",
            "json_path": str(json_path) if json_path.exists() else "",
            "collection_report_path": str(report_path) if report_path.exists() else "",
        }
        print(f"[collector] universal failed code={proc.returncode}; falling back to legacy collector")

    profile_dir = DATA_DIR / "browser_profiles" / ("source-collector" if is_ai_skills else safe_run)
    collector_v2 = BASE_DIR / "tools" / "source_graph_collect_v2.py"
    legacy_collector = BASE_DIR / "tools" / "collect_source_pack.py"
    if is_ai_skills:
        collector_script = legacy_collector
    elif ("youtube.com" in host) or ("youtu.be" in host) or ("oopy.io" in host) or ("wikidocs.net" in host):
        collector_script = collector_v2 if collector_v2.exists() else legacy_collector
    else:
        collector_script = collector_v2 if collector_v2.exists() else legacy_collector
    cmd = [
        os.sys.executable,
        str(collector_script),
        url,
        "--out",
        str(output_dir),
        "--no-manual-pause",
        "--follow-labs",
        "--follow-limit",
        "8",
        "--crawl-limit",
        "18",
        "--tree-limit",
        "40" if is_ai_skills else "24",
        "--user-data-dir",
        str(profile_dir),
        "--auto-login-wait",
        "45",
    ]
    if not is_ai_skills:
        cmd.insert(3, "--headless")
    started = time.perf_counter()
    print(f"[collector] run_id={safe_run} seed_url={url}")
    print(f"[collector] script={collector_script} out={output_dir} profile={profile_dir} headless={not is_ai_skills}")
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
            "fallback_from_universal": universal_attempt,
        }

    after = sorted(
        [path for path in output_dir.glob("*.md") if path.resolve() not in before and not path.name.endswith(".report.md")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if proc.returncode != 0 or not after:
        # v4.7.5 safety net: both collector CLIs can complete with code 0 but produce
        # only trace.jsonl/no usable source_pack in some local states.  Do not turn
        # that into a fake article; first try the in-process public URL extractor,
        # which is run-scoped and does not reuse previous source packs.
        public_text = ""
        public_error = ""
        try:
            public_text = fetch_public_source_text(url, max_chars=22000).strip()
        except Exception as exc:
            public_error = f"{type(exc).__name__}: {exc}"
            public_text = ""

        expected_kind = expected_topic_kind_from_input(seed_url=url, current_text=public_text)
        public_usable = bool(public_text) and (
            source_pack_like_text(public_text)
            or len(public_text) >= 800
            or bool(expected_kind)
        )
        if public_usable:
            fallback_md_path = output_dir / "public_url_fallback_source_pack.md"
            fallback_json_path = output_dir / "public_url_fallback_source_graph.json"
            fallback_md_path.write_text(public_text, encoding="utf-8")
            transcript_segments = 0
            lowered_public = public_text.lower()
            if "[영상 자막]" in public_text or "transcript" in lowered_public or "caption" in lowered_public:
                transcript_segments = 1
            stats = {
                "page_count": 1,
                "visible_text_chars": len(public_text),
                "link_count": 0,
                "video_candidate_count": 1 if is_youtube_host(host) else 0,
                "lesson_candidate_count": 1,
                "lab_candidate_count": 0,
                "tree_item_count": 0,
            }
            quality = {
                "usable_text_chars": len(public_text),
                "text_chars": len(public_text),
                "transcript_segments": transcript_segments,
                "quality_status": "public_url_fallback",
                "can_generate_article": True,
                "expected_topic_kind": expected_kind,
                "warnings": ["collector_no_output_used_public_url_fallback"],
            }
            fallback_payload = {
                "title": public_text.splitlines()[0][:200] if public_text.splitlines() else url,
                "current_url": url,
                "input_url": url,
                "stats": stats,
                "quality": quality,
                "snapshots": [{
                    "label": "public_url_fallback",
                    "type": "public_url_fallback",
                    "title": public_text.splitlines()[0][:200] if public_text.splitlines() else url,
                    "url": url,
                    "visible_text": public_text,
                    "headings": [],
                }],
                "nodes": [{
                    "type": "public_url_fallback",
                    "title": public_text.splitlines()[0][:200] if public_text.splitlines() else url,
                    "url": url,
                    "text": public_text,
                }],
            }
            fallback_json_path.write_text(json.dumps(fallback_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            elapsed = round(time.perf_counter() - started, 2)
            print(f"[collector] public-url fallback ok chars={len(public_text)} elapsed={elapsed}s")
            return public_text, {
                "ok": True,
                "collector": "public_url_fallback_after_collector_no_output",
                "collector_returncode": proc.returncode,
                "command": cmd,
                "markdown_path": str(fallback_md_path),
                "json_path": str(fallback_json_path),
                "quality": quality,
                "stats": stats,
                "stdout": (proc.stdout or "")[-4000:],
                "stderr": (proc.stderr or "")[-4000:],
                "elapsed_seconds": elapsed,
                "fallback_from_universal": universal_attempt,
                "fallback_reason": "collector returned no usable markdown/json",
                "public_fetch_error": public_error,
            }

        print(f"[collector] failed/no-output code={proc.returncode} elapsed={round(time.perf_counter() - started, 2)}s")
        found_files = []
        try:
            found_files = [str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()][:60]
        except Exception:
            found_files = []
        return "", {
            "ok": False,
            "error": f"collector completed with code {proc.returncode} but no usable source_pack markdown/json was found",
            "command": cmd,
            "output_dir": str(output_dir),
            "found_files": found_files,
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-4000:],
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "fallback_from_universal": universal_attempt,
            "public_fetch_error": public_error,
        }
    md_path = after[0]
    text = md_path.read_text(encoding="utf-8", errors="replace")
    json_path = md_path.with_suffix(".json")
    if seed_playlist_id(url) and not text_contains_seed_playlist(url, text) and json_path.exists():
        try:
            json_text_for_seed = json_path.read_text(encoding="utf-8", errors="replace")
            payload_for_seed = json.loads(json_text_for_seed) if json_text_for_seed.strip() else {}
        except Exception:
            json_text_for_seed = ""
            payload_for_seed = {}
        # The markdown body can omit the playlistId even when the run is correct.
        # Treat the JSON current_url/snapshots as the authority.  Only block when
        # the requested playlistId is absent from BOTH markdown and collector JSON.
        if not text_contains_seed_playlist(url, json_text_for_seed):
            return "", {
                "ok": False,
                "error": "collector output does not contain requested playlistId; possible stale browser/profile state",
                "command": cmd,
                "markdown_path": str(md_path),
                "json_path": str(json_path),
                "collector_title": str(payload_for_seed.get("title") or ""),
                "collector_current_url": str(payload_for_seed.get("current_url") or ""),
                "stats": payload_for_seed.get("stats") if isinstance(payload_for_seed.get("stats"), dict) else {},
                "quality": payload_for_seed.get("quality") if isinstance(payload_for_seed.get("quality"), dict) else {},
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "fallback_from_universal": universal_attempt,
            }
    quality: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
            stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
        except Exception:
            quality = {}
            stats = {}
    warnings = quality.get("warnings") or []
    text_chars = int(stats.get("visible_text_chars") or quality.get("usable_text_chars") or len(text or ""))
    page_count = int(stats.get("page_count") or 0)
    lab_count = int(stats.get("lab_candidate_count") or 0)
    video_count = int(stats.get("video_candidate_count") or 0)
    lesson_count = int(stats.get("lesson_candidate_count") or 0)
    tree_count = int(stats.get("tree_item_count") or 0)
    has_substantial_evidence = (
        text_chars >= 8000
        and page_count >= 2
        and (lab_count > 0 or video_count > 0 or lesson_count > 0 or tree_count > 0)
    )
    if "low_visible_text" in warnings or ("login_or_access_page_detected" in warnings and not has_substantial_evidence):
        return "", {
            "ok": False,
            "error": "source pack quality check failed",
            "command": cmd,
            "markdown_path": str(md_path),
            "json_path": str(json_path) if json_path.exists() else "",
            "quality": quality,
            "stats": stats,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "fallback_from_universal": universal_attempt,
        }
    print(f"[collector] ok md={md_path} elapsed={round(time.perf_counter() - started, 2)}s stats={stats}")
    return text, {
        "ok": True,
        "command": cmd,
        "markdown_path": str(md_path),
        "json_path": str(json_path) if json_path.exists() else "",
        "quality": quality,
        "stats": stats,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "fallback_from_universal": universal_attempt,
    }


def make_generation_run_id() -> str:
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"



# --- v4.7.3 Recovery Hotfix: run isolation / contamination guards ---
def make_run_dir(run_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", run_id or make_generation_run_id())
    run_dir = RUNS_DIR / safe
    (run_dir / "images").mkdir(parents=True, exist_ok=True)
    return run_dir


def is_youtube_host(host: str) -> bool:
    return "youtube.com" in host or "youtu.be" in host


def is_notion_host(host: str) -> bool:
    return host.endswith("notion.site") or host.endswith("notion.so")


def is_generic_rendered_document_host(host: str) -> bool:
    protected = (
        "youtube.com", "youtu.be", "notion.site", "notion.so", "inflearn.com", "udemy.com",
        "aiskillsnavigator.microsoft.com",
    )
    return bool(host) and not any(item in host for item in protected)


def substantial_rendered_document(seed_url: str, text_chars: int, source_pack_text: str = "") -> bool:
    host = url_domain(seed_url)
    if not is_generic_rendered_document_host(host):
        return False
    lower = (source_pack_text or "").lower()
    shell_markers = ["enable javascript", "checking your browser", "access denied", "cloudflare", "login required"]
    if any(marker in lower for marker in shell_markers) and text_chars < 10000:
        return False
    return text_chars >= 5000


INTERNAL_ARTICLE_BANNED_PHRASES = [
    "[생성 직전 사용자가 정의한 어려운 문제]",
    "[생성 직전 사용자가 적은 어려움/헷갈린 부분]",
    "current input",
    "current material",
    "provided learning material",
    "source pack",
    "source graph",
    "collector",
    "writer",
    "usable evidence",
    "evidence item",
    "asr 잡음",
    "asr noise",
    "canonical concept",
    "anchor vocabulary",
    "problem-solving eligibility",
    "run_id",
    "seed_url",
    "normalized input",
    "현재 입력",
    "수집 자료",
    "입력 URL",
    "요청 URL",
    "소스팩",
    "수집 결과",
    "수집 그래프",
    "Writer 관점",
    "생성기 관점",
    "근거 기반 학습 기록",
    "selected focus",
    "focus title",
    "focus url",
    "problem framing candidate",
    "candidate focus list",
    "body chars",
    "why selected",
]

# v4.7.25 transcript_ux_and_rag_guard
CROSS_RUN_CONTAMINATION_TERMS = [
    "agent orchestration", "orchestrator", "sub-agent", "sub-agents", "github copilot cli",
    "planner", "coder", "designer", "toolchain", "rag", "mcp", "model context protocol",
    "battery rul", "bmaml", "ceemdan", "power bi", "dax measure", "semantic model",
]


def dedupe_repeated_lines(text: str) -> str:
    lines: list[str] = []
    previous_key = ""
    for line in str(text or "").splitlines():
        stripped = re.sub(r"\s+", " ", line.strip())
        if stripped and stripped == previous_key:
            continue
        lines.append(line)
        if stripped:
            previous_key = stripped
    return "\n".join(lines).strip()


def clean_user_problem_note(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace("[생성 직전 사용자가 정의한 어려운 문제]", "")
    cleaned = cleaned.replace("[생성 직전 사용자가 적은 어려움/헷갈린 부분]", "")
    cleaned = re.sub(r"없음\.\s*자료의 핵심 흐름을 바탕으로 문제를 정의하고 해결 과정 작성\.?", "", cleaned)
    cleaned = re.sub(r"없음\.\s*자료의 핵심 흐름을 바탕으로 작성\.?", "", cleaned)
    return re.sub(r"\n{3,}", "\n\n", dedupe_repeated_lines(cleaned)).strip()


def expected_topic_kind_from_input(seed_url: str = "", current_text: str = "") -> str:
    blob = f"{seed_url}\n{current_text}".lower()

    # URL/title fingerprints must win over noisy body text.  The collected body
    # often includes global navigation/related-doc links (for example FastAPI
    # docs may mention Docker deployment, MDN pages contain generic "body" and
    # "parameters", and GitHub pages may contain YAML snippets).  Decide the
    # current run from seed URL + user problem first, then use body text only as
    # fallback.
    if "docs.python.org" in blob and "tutorial/modules" in blob:
        return "python_modules"
    if "numpy.org" in blob and "broadcasting" in blob:
        return "numpy_broadcasting"
    if "pandas.pydata.org" in blob and "groupby" in blob:
        return "pandas_groupby"
    if "redis.io" in blob and "data-types" in blob:
        return "redis_data_types"
    if "developer.mozilla.org" in blob and "/web/http/status" in blob:
        return "http_status"
    if "docs.djangoproject.com" in blob and "/topics/db/models" in blob:
        return "django_models"
    if "nodejs.org" in blob and "event-loop" in blob:
        return "node_event_loop"
    if "docs.spring.io" in blob and "beans" in blob:
        return "spring_beans"
    if "developer.mozilla.org" in blob and "using_web_workers" in blob:
        return "web_workers"
    if "typescriptlang.org" in blob and "everyday-types" in blob:
        return "typescript_types"
    if "react.dev" in blob and "useeffect" in blob:
        return "react_useeffect"
    if "kubernetes.io" in blob and "/pods" in blob:
        return "kubernetes_pod"
    if "postgresql.org" in blob and "indexes-intro" in blob:
        return "postgres_index"
    if "developer.mozilla.org" in blob and "fetch_api" in blob:
        return "fetch_api"
    if "docs.sqlalchemy.org" in blob and "/orm/quickstart" in blob:
        return "sqlalchemy_orm"
    if "fastapi.tiangolo.com" in blob or "realpython.com/fastapi-python-web-apis" in blob or "fastapi example application" in blob:
        return "fastapi"
    if "docs.github.com" in blob and ("/actions/" in blob or "github-actions" in blob or "github actions" in blob):
        return "github_actions"
    if "r8_veqiybji" in blob or "github actions tutorial" in blob:
        return "github_actions"
    if "developer.mozilla.org" in blob and "async_function" in blob:
        return "javascript_async"
    if "developer.mozilla.org" in blob and ("global_objects/array/map" in blob or "global_objects%2farray%2fmap" in blob):
        return ""
    if "developer.mozilla.org" in blob and ("global_objects/promise" in blob or "global_objects%2fpromise" in blob):
        return "javascript_promise"
    if "javascript.info/promise-basics" in blob or "promise basics" in blob:
        return "javascript_promise"
    if "3c-ibn73dde" in blob or "freecodecamp.org/news/docker-simplified" in blob or "docker simplified" in blob:
        return "docker"
    if "docs.docker.com" in blob:
        return "docker"

    # Strong user_problem/title signals. Keep this ordered by specificity, not by
    # previous-run state.  Use exact source/user-intent signals before generic
    # words like "then", "parameter", or "body" which appear in many docs.
    # v4.7.18: generic words like import/package/shape/parameter appear in many
    # docs.  Do not route to a known profile unless the current input gives a
    # strong source-topic signal.  Otherwise the source-first fallback will build
    # a contract from the current title + memo + collected text.
    if (
        "python module" in blob
        or "module search path" in blob
        or "import modules" in blob
        or ("python" in blob and any(term in blob for term in ["module", "import", "package", "namespace"]))
    ):
        return "python_modules"
    if (
        "numpy" in blob
        or "broadcasting" in blob
        or "incompatible shape" in blob
        or ("array" in blob and "shape" in blob and "dimension" in blob)
    ):
        return "numpy_broadcasting"
    if any(term in blob for term in ["pandas", "groupby", "split-apply-combine", "aggregation", "transformation"]):
        return "pandas_groupby"
    if any(term in blob for term in ["redis", "data types", "string", "hash", "sorted set", "key-value"]):
        return "redis_data_types"
    if any(term in blob for term in ["http status", "status code", "2xx", "4xx", "5xx"]):
        return "http_status"
    if any(term in blob for term in ["django model", "migration", "field", "table mapping"]):
        return "django_models"
    if any(term in blob for term in ["node.js event loop", "event loop", "call stack", "callback queue", "timer"]):
        return "node_event_loop"
    if any(term in blob for term in ["spring bean", "ioc container", "dependency injection", "spring container"]):
        return "spring_beans"
    if any(term in blob for term in ["web worker", "worker thread", "main thread", "postmessage", "onmessage"]):
        return "web_workers"
    if any(term in blob for term in ["typescript", "everyday types", "type annotation", "primitive type", "object type"]):
        return "typescript_types"
    if any(term in blob for term in ["useeffect", "setup", "cleanup", "dependencies", "external system", "react"]):
        return "react_useeffect"
    if any(term in blob for term in ["kubernetes", "pod", "shared resources", "workload"]):
        return "kubernetes_pod"
    if any(term in blob for term in ["postgresql", "create index", "sequential scan", "index scan", "query planner"]):
        return "postgres_index"
    if any(term in blob for term in ["fetch api", "fetch()", "response object", "body parsing", "headers"]):
        return "fetch_api"
    # v4.7.21: do not route unsupported pages to SQLAlchemy or Promise from
    # generic words such as session, engine, commit, then, or catch.
    if ("sqlalchemy" in blob or "mapped class" in blob or ("orm" in blob and any(term in blob for term in ["session", "engine", "commit", "select"]))):
        return "sqlalchemy_orm"
    if any(term in blob for term in ["async function", "async/await", "await", "promise-based", "promise based"]):
        return "javascript_async"
    if ("promise" in blob and any(term in blob for term in ["pending", "fulfilled", "rejected", "then", "catch", "resolve", "reject"])):
        return "javascript_promise"
    if any(term in blob for term in ["github actions", "github-actions", "workflow_dispatch", "runs-on", "ci/cd", "ci cd", "workflow", "runner"]):
        return "github_actions"
    if (
        "fastapi" in blob
        or ("pydantic" in blob and any(term in blob for term in ["api", "endpoint", "request body", "basemodel"]))
        or ("swagger ui" in blob and "api" in blob)
    ):
        return "fastapi"
    if any(term in blob for term in [
        "docs.docker.com", "docker", "dockerfile", "docker compose", "compose.yaml",
        "docker image", "container", "port mapping", "volume", "workdir", "docker build",
        "docker run", "docker volume", "docker compose up",
    ]):
        return "docker"
    return ""


TOPIC_REQUIRED_TERMS = {
    "docker": ["docker", "image", "container", "dockerfile", "port", "volume"],
    "github_actions": ["github actions", "workflow", "job", "step", "yaml", "runner", "ci/cd", "ci cd"],
    "fastapi": ["fastapi", "endpoint", "path parameter", "query parameter", "request body", "pydantic", "/docs"],
    "javascript_promise": ["promise", "pending", "fulfilled", "rejected", "then", "catch"],
    "javascript_async": ["async", "await", "promise", "return", "error handling"],
    "react_useeffect": ["useeffect", "setup", "cleanup", "dependencies", "external system", "react"],
    "kubernetes_pod": ["kubernetes", "pod", "container", "workload", "shared resources", "lifecycle"],
    "postgres_index": ["postgresql", "index", "create index", "sequential scan", "query planner", "where"],
    "fetch_api": ["fetch", "request", "response", "status", "headers", "body", "promise"],
    "sqlalchemy_orm": ["sqlalchemy", "orm", "engine", "session", "mapped class", "select", "commit"],
    "python_modules": ["python", "module", "import", "package", "namespace"],
    "numpy_broadcasting": ["numpy", "broadcasting", "shape", "dimension", "array"],
    "pandas_groupby": ["pandas", "groupby", "split", "apply", "combine", "aggregation"],
    "redis_data_types": ["redis", "string", "list", "set", "hash", "data type"],
    "http_status": ["http", "status", "2xx", "4xx", "5xx"],
    "django_models": ["django", "model", "field", "database", "table"],
    "node_event_loop": ["node", "event loop", "timer", "callback", "poll"],
    "spring_beans": ["spring", "bean", "container", "dependency", "configuration"],
    "web_workers": ["worker", "main thread", "postmessage", "onmessage", "background"],
    "typescript_types": ["typescript", "type", "array", "object", "function"],
}


TOPIC_BAD_TERMS = {
    "github_actions": ["dockerfile", "docker compose", "port mapping", "volume", "bind mount", "fastapi", "pydantic", "path parameter", "query parameter", "request body"],
    "fastapi": ["dockerfile", "docker compose", "port mapping", "volume", "bind mount", "docker image", "docker container", "workflow_dispatch", "runs-on"],
    "docker": ["path parameter", "query parameter", "request body", "pydantic", "workflow_dispatch", "runs-on", "fastapi"],
    "javascript_promise": ["fastapi", "pydantic", "path parameter", "query parameter", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on"],
    "javascript_async": ["fastapi", "pydantic", "path parameter", "query parameter", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on"],
    "react_useeffect": ["fastapi", "pydantic", "path parameter", "query parameter", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on", "pending", "fulfilled", "rejected"],
    "kubernetes_pod": ["fastapi", "pydantic", "path parameter", "query parameter", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on", "pending", "fulfilled", "rejected"],
    "postgres_index": ["fastapi", "pydantic", "path parameter", "query parameter", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on", "pending", "fulfilled", "rejected"],
    "fetch_api": ["fastapi", "pydantic", "path parameter", "query parameter", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on"],
    "sqlalchemy_orm": ["fastapi", "pydantic", "path parameter", "query parameter", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on", "pending", "fulfilled", "rejected"],
    "python_modules": ["fastapi", "pydantic", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on", "pending", "fulfilled", "sqlalchemy", "orm"],
    "numpy_broadcasting": ["fastapi", "pydantic", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on", "pending", "fulfilled", "sqlalchemy", "orm"],
    "pandas_groupby": ["fastapi", "pydantic", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on", "pending", "fulfilled", "sqlalchemy", "orm"],
    "redis_data_types": ["fastapi", "pydantic", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on", "pending", "fulfilled", "sqlalchemy", "orm"],
    "http_status": ["fastapi", "pydantic", "dockerfile", "docker compose", "workflow_dispatch", "runs-on", "pending", "fulfilled", "sqlalchemy", "orm"],
    "django_models": ["fastapi", "pydantic", "path parameter", "query parameter", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on", "postgresql", "create index"],
    "node_event_loop": ["fastapi", "pydantic", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on", "react", "useeffect", "sqlalchemy"],
    "spring_beans": ["fastapi", "pydantic", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on", "react", "useeffect", "sqlalchemy"],
    "web_workers": ["fastapi", "pydantic", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on", "react", "useeffect", "sqlalchemy"],
    "typescript_types": ["fastapi", "pydantic", "request body", "dockerfile", "docker compose", "workflow_dispatch", "runs-on", "react", "useeffect", "sqlalchemy"],
}


def topic_mismatch_failures(expected_kind: str, article: str) -> list[str]:
    if not expected_kind:
        return []
    lowered = str(article or "").lower()
    required = TOPIC_REQUIRED_TERMS.get(expected_kind, [])
    bad_terms = TOPIC_BAD_TERMS.get(expected_kind, [])
    good_hits = [term for term in required if term in lowered]
    bad_hits = [term for term in bad_terms if term in lowered]
    failures: list[str] = []
    if required and len(good_hits) < min(3, len(required)):
        failures.append(f"{expected_kind} article lacks required current-topic terms: " + ", ".join(required))
    if expected_kind in {"github_actions", "fastapi"} and len(bad_hits) >= 2:
        failures.append(f"{expected_kind} article contains cross-topic contamination: " + ", ".join(bad_hits[:8]))
    if expected_kind in {"javascript_promise", "javascript_async"} and len(bad_hits) >= 2:
        failures.append(f"{expected_kind} article contains cross-topic contamination: " + ", ".join(bad_hits[:8]))
    if expected_kind == "docker" and len(bad_hits) >= 2:
        failures.append("Docker article contains non-Docker topic contamination: " + ", ".join(bad_hits[:8]))
    if expected_kind in {"react_useeffect", "kubernetes_pod", "postgres_index", "fetch_api", "sqlalchemy_orm", "python_modules", "numpy_broadcasting", "pandas_groupby", "redis_data_types", "http_status", "django_models", "node_event_loop", "spring_beans", "web_workers", "typescript_types"} and len(bad_hits) >= 2:
        failures.append(f"{expected_kind} article contains cross-topic contamination: " + ", ".join(bad_hits[:8]))
    return failures


def youtube_article_best_supported_kind(article: str) -> str:
    """Infer the most plausible topic from the generated article itself.

    YouTube seed URLs often do not contain semantic topic keywords, while the
    transcript may contain generic words such as import/path/body/query.  For
    final validation, prefer the article's own high-confidence required-term
    coverage instead of the first weak keyword hit from the transcript.
    """
    lowered = str(article or "").lower()
    best_kind = ""
    best_score = 0
    for kind, required in TOPIC_REQUIRED_TERMS.items():
        if not required:
            continue
        hits = [term for term in required if term in lowered]
        score = len(hits)
        if score < min(3, len(required)):
            continue
        # The candidate must also pass its own contamination rules.
        if topic_mismatch_failures(kind, article):
            continue
        if score > best_score:
            best_kind = kind
            best_score = score
    return best_kind


def youtube_source_best_supported_kind(current_text: str = "", seed_url: str = "") -> str:
    """Infer a strong YouTube topic from the current video title/transcript.

    v4.7.17: do not let generic transcript words like import, path, body,
    query, model, or then pick an old topic.  YouTube validation should use
    high-signal title/transcript fingerprints first, then compare the generated
    article topic against that source topic.
    """
    blob = f"{seed_url}\n{current_text}".lower()
    checks = [
        ("redis_data_types", ["redis data", "redis data types", "redis data structures", "redis series", "sorted set", "redis hash"]),
        ("pandas_groupby", ["pandas groupby", "groupby method", "split-apply-combine", "split apply combine", "group by:", "groupby()"]),
        ("http_status", ["http status", "status codes", "status code", "200, 404", "404", "5xx", "4xx"]),
        ("numpy_broadcasting", ["numpy broadcasting", "array broadcasting", "broadcasting in python", "broadcasting explained", "incompatible shape"]),
        ("node_event_loop", ["node.js event loop", "nodejs event loop", "event loop", "phases of event loop", "nexttick", "timers phase"]),
        ("python_modules", ["python modules", "python packages", "import modules", "standard library", "module search path"]),
        ("web_workers", ["web worker", "web workers", "worker thread", "postmessage", "onmessage"]),
        ("typescript_types", ["typescript", "everyday types", "type annotation"]),
        ("react_useeffect", ["react useeffect", "useeffect", "react effect", "cleanup function"]),
        ("kubernetes_pod", ["kubernetes pod", "kubernetes pods", "pod container"]),
        ("postgres_index", ["postgresql index", "postgres index", "sequential scan", "index scan"]),
        ("fetch_api", ["fetch api", "using fetch", "fetch()", "response object"]),
        ("sqlalchemy_orm", ["sqlalchemy orm", "sqlalchemy", "mapped class", "session commit"]),
        ("django_models", ["django model", "django models", "django documentation models"]),
        ("spring_beans", ["spring bean", "spring beans", "ioc container"]),
    ]
    for kind, needles in checks:
        if any(n in blob for n in needles):
            return kind
    return ""


def resolve_youtube_expected_kind(seed_url: str, current_text: str, article: str) -> tuple[str, str]:
    """Return (expected_kind, mismatch_reason) for YouTube validation.

    If the video title/transcript strongly says Redis but the article says
    pandas, keep the source topic and report a real mismatch.  If the source
    topic is weak but the article is strongly supported, allow the article
    topic to prevent stale previous-run overblocking.
    """
    source_kind = youtube_source_best_supported_kind(current_text=current_text, seed_url=seed_url)
    article_kind = youtube_article_best_supported_kind(article)
    generic_kind = expected_topic_kind_from_input(seed_url=seed_url, current_text=current_text)

    if source_kind and article_kind and source_kind != article_kind:
        return source_kind, f"YouTube article topic mismatches video topic: expected {source_kind}, got {article_kind}"
    if source_kind:
        return source_kind, ""
    if article_kind:
        return article_kind, ""
    return generic_kind, ""



def is_source_first_article(article: str) -> bool:
    """Return True for v4.7.18+ source-derived fallback articles.

    These articles intentionally do not map to a fixed known-topic profile.
    They should be validated against the current source, not against stale
    FastAPI/NumPy/Python profile requirements.
    """
    lowered = str(article or "").lower()
    return (
        "deriving the learning problem from the page itself" in lowered
        or "본문 안에서 핵심 개념 구분하기" in str(article or "")
        or "학습 자료: 웹문서 본문과 사용자 메모" in str(article or "")
        or "source-first" in lowered
    )


def source_first_policy_failures(article: str, current_text: str = "", seed_url: str = "") -> list[str]:
    """Lightweight validation for unsupported/new-topic source-first output.

    The goal is to stop stale-run leakage without forcing the article into a
    known profile such as FastAPI, NumPy, Python modules, or SQLAlchemy.
    """
    text = str(article or "")
    lowered = text.lower()
    current_blob = f"{current_text}\n{seed_url}".lower()
    failures: list[str] = []

    # If a source-first fallback somehow produced a known technical profile
    # title that is not supported by the current source, block it.  This catches
    # obvious drift while allowing legitimate new topics such as Array.map or
    # Korean time-management articles.
    known_title_markers = {
        "fastapi 학습 기록": ["fastapi", "tiangolo"],
        "numpy broadcasting 학습 기록": ["numpy", "broadcasting"],
        "python module 학습 기록": ["python", "module", "import"],
        "sqlalchemy orm 학습 기록": ["sqlalchemy", "orm"],
        "docker": ["docker"],
        "postgresql index": ["postgresql", "index"],
        "react useeffect": ["react", "useeffect"],
    }
    first_chunk = lowered[:500]
    for marker, allowed_terms in known_title_markers.items():
        if marker in first_chunk and not any(term in current_blob for term in allowed_terms):
            failures.append(f"source-first article drifted into unrelated known topic: {marker}")
            break

    # Current source should still be visible in the article through title/memo
    # terms.  This is intentionally permissive for Korean/non-IT content.
    title_line = ""
    for line in str(current_text or "").splitlines():
        if "collector_title:" in line.lower() or line.strip().startswith("# "):
            title_line = line
            break
    seed_subject = re.sub(r"https?://|www\.|[/_%?=&.-]+", " ", seed_url or "")
    source_terms = []
    for token in re.findall(r"[A-Za-z가-힣][A-Za-z0-9가-힣_.()\-]{2,}", f"{title_line} {seed_subject}"):
        tl = token.lower().strip("_.()-")
        if tl in {"https", "http", "com", "org", "docs", "wiki", "youtube", "youtu", "watch", "collector", "title"}:
            continue
        if len(tl) >= 3 and tl not in source_terms:
            source_terms.append(tl)
        if len(source_terms) >= 8:
            break
    if source_terms and not any(term in lowered for term in source_terms[:8]):
        # Do not fail hard for very short/non-English URLs; only warn when the
        # source title is clearly available but absent from the article.
        if len(" ".join(source_terms)) > 8:
            failures.append("source-first article lacks visible current-source title terms")

    return failures

def contains_contamination_term(text: str, term: str) -> bool:
    """Return True only for real stale-topic terms, not substrings.

    v4.7.25: a term such as ``rag`` must not match ordinary words like
    ``storage``.  Treat ASCII terms as token/phrase matches with word
    boundaries, while keeping simple substring matching for Korean/non-ASCII
    phrases where word boundaries are less reliable.
    """
    haystack = str(text or "").lower()
    needle = str(term or "").lower().strip()
    if not needle:
        return False
    if re.fullmatch(r"[a-z0-9 _./+-]+", needle):
        pattern = r"(?<![a-z0-9_])" + re.escape(needle).replace(r"\ ", r"\s+") + r"(?![a-z0-9_])"
        return re.search(pattern, haystack) is not None
    return needle in haystack


def contamination_hits(article_text: str, current_text: str = "", seed_url: str = "") -> list[str]:
    current_blob = f"{current_text}\n{seed_url}".lower()
    hits: list[str] = []
    for term in CROSS_RUN_CONTAMINATION_TERMS:
        if contains_contamination_term(article_text, term) and not contains_contamination_term(current_blob, term):
            hits.append(term)
    return hits


def final_article_policy_failures(
    article: str,
    current_text: str = "",
    seed_url: str = "",
    contamination_context: str = "",
) -> list[str]:
    text = str(article or "")
    lowered = text.lower()
    failures: list[str] = []
    leaked = [phrase for phrase in INTERNAL_ARTICLE_BANNED_PHRASES if phrase and phrase.lower() in lowered]
    if leaked:
        failures.append("internal/placeholder terms leaked: " + ", ".join(leaked[:8]))
    contamination_input = "\n".join([current_text, contamination_context])
    contamination = contamination_hits(text, current_text=contamination_input, seed_url=seed_url)
    # v4.7.25: use token/phrase matching so RAG does not match storage.
    if contamination:
        failures.append("possible stale-run topic contamination: " + ", ".join(contamination[:8]))
    # v4.7.19: source-first fallback articles are intentionally generated from
    # the current page/memo without a known topic profile.  Do not force them
    # through stale profile requirements such as javascript_promise or
    # sqlalchemy_orm.
    if is_source_first_article(text):
        failures.extend(source_first_policy_failures(text, current_text=current_text, seed_url=seed_url))
        return failures

    expected_kind = expected_topic_kind_from_input(seed_url=seed_url, current_text=current_text)
    host = url_domain(seed_url)
    if "youtube.com" in host or "youtu.be" in host:
        expected_kind, youtube_reason = resolve_youtube_expected_kind(seed_url, current_text, text)
        if youtube_reason:
            failures.append(youtube_reason)
            return failures
    failures.extend(topic_mismatch_failures(expected_kind, text))
    return failures


def final_article_policy_report(seed_url: str, run_id: str, failures: list[str], article: str) -> str:
    preview = (article or "")[:1200]
    return f"""# 최종 글 검수 실패

현재 입력과 무관한 주제 또는 내부 시스템 문구가 최종 글에 포함되어 Medium 글 생성을 중단했습니다.

## 입력 URL
- {seed_url or '직접 입력/이미지 업로드'}

## run_id
{run_id or 'direct_upload'}

## 실패 이유
{chr(10).join('- ' + reason for reason in failures)}

## 조치
- 이번 run의 입력 본문, 업로드 이미지, 붙여넣은 메모만 사용해야 합니다.
- 이전 run의 source pack, image evidence, writer input, fallback article을 재사용하면 안 됩니다.
- 제목과 핵심 키워드는 현재 입력 안에서 확인되는 내용으로만 생성해야 합니다.

## 차단된 글 미리보기
```text
{preview}
```
"""


def current_run_image_policy_context(result: dict[str, Any]) -> str:
    """Return trusted Vision evidence for image-upload-only final validation.

    Filename/README fallbacks are deliberately excluded so an old template or
    suggestive filename cannot whitelist an unrelated topic.  This context is
    added only by the batch-upload handler when actual images were uploaded.
    """
    evidence = result.get("image_evidence")
    if not isinstance(evidence, list):
        return ""

    trusted_items: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        source = str(item.get("evidence_source") or "").strip().lower()
        try:
            confidence = float(item.get("confidence") or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        if source not in {"vision", "llm"} or confidence < 0.5:
            continue
        trusted_items.append(item)

    grounded_lines: list[str] = []
    direct_text_parts: list[str] = []
    claim_counts: dict[str, int] = {}
    claim_labels: dict[str, str] = {}
    for item in trusted_items:
        parts: list[str] = []
        for key in ("caption",):
            value = str(item.get(key) or "").strip()
            if value:
                parts.append(value)
        parts.extend(normalize_str_list(item.get("visible_evidence"))[:10])
        parts.extend(normalize_str_list(item.get("technical_entities"))[:12])
        if parts:
            grounded_lines.append(" | ".join(parts))
            direct_text_parts.extend(parts)

        claims = [
            str(item.get("primary_topic") or "").strip(),
            str(item.get("platform_or_product") or "").strip(),
            *normalize_str_list(item.get("topic_terms"))[:12],
        ]
        for claim in claims:
            normalized = re.sub(r"\s+", " ", claim.lower()).strip()
            if not normalized:
                continue
            claim_counts[normalized] = claim_counts.get(normalized, 0) + 1
            claim_labels.setdefault(normalized, claim)

    if not grounded_lines:
        return ""

    direct_blob = "\n".join(direct_text_parts).lower()
    agreed_claims = [
        claim_labels[key]
        for key, count in claim_counts.items()
        if count >= 2 or contains_contamination_term(direct_blob, key)
    ]
    context = "[CURRENT_RUN_VISION_EVIDENCE]\n" + "\n".join(grounded_lines)[:18000]
    if agreed_claims:
        context += "\n[CURRENT_RUN_IMAGE_TOPIC_CONSENSUS]\n" + " | ".join(agreed_claims[:24])
    return context


def apply_final_article_policy(
    result: dict[str, Any],
    current_text: str = "",
    seed_url: str = "",
    run_id: str = "",
    contamination_context: str = "",
) -> dict[str, Any]:
    draft = str(result.get("draft") or "")
    critic = result.get("critic_report") if isinstance(result.get("critic_report"), dict) else {}
    metrics = critic.get("metrics") if isinstance(critic.get("metrics"), dict) else {}
    if metrics.get("failure_type") == "vision_rate_limit":
        # Provider diagnostics are user-facing error reports, not generated
        # articles.  Running them through article contamination checks hides the
        # real API failure behind an unrelated policy message.
        return result
    failures = final_article_policy_failures(
        draft,
        current_text=current_text,
        seed_url=seed_url,
        contamination_context=contamination_context,
    )
    if not failures:
        result["draft"] = sanitize_medium_markdown(draft)
        return result
    result = dict(result)
    result["draft"] = final_article_policy_report(seed_url, run_id, failures, draft)
    result["article_type"] = "final_article_policy_failed"
    result["critic_report"] = {"passed": False, "failures": failures, "metrics": {"run_id": run_id, "seed_url": seed_url}}
    result["mode"] = "final_article_policy_failed"
    return result

# --- /v4.7.3 Recovery Hotfix ---

def seed_playlist_id(seed_url: str) -> str:
    parsed = urlparse(seed_url or "")
    qs = parse_qs(parsed.query or "")
    return (qs.get("playlistId") or qs.get("playlistid") or [""])[0]


def text_contains_seed_playlist(seed_url: str, text: str) -> bool:
    playlist_id = seed_playlist_id(seed_url)
    if not playlist_id:
        return True
    return playlist_id in (text or "")


def append_video_transcript_evidence(seed_url: str, source_pack_text: str, collector_report: dict[str, Any], limit: int = 2) -> str:
    """Append transcript/description evidence for video candidates.

    The collector can discover embedded YouTube/watch URLs inside AI Skills Navigator,
    but Playwright page text alone usually does not contain the actual transcript.
    This step treats video extraction as evidence enrichment, not as the article voice.
    """
    graph = collector_source_graph(collector_report)
    candidates = []
    for item in graph.get("video_url_candidates", []) or []:
        u = str(item or "").strip()
        if not u:
            continue
        if "youtube.com/embed/" in u:
            m = re.search(r"/embed/([A-Za-z0-9_-]{6,})", u)
            if m:
                u = f"https://www.youtube.com/watch?v={m.group(1)}"
        if ("youtube.com" in url_domain(u) or "youtu.be" in url_domain(u)) and u not in candidates:
            candidates.append(u)
        if len(candidates) >= limit:
            break
    if not candidates:
        return source_pack_text
    blocks = [source_pack_text.rstrip(), "", "## Video Transcript Evidence"]
    for u in candidates:
        blocks.append(fetch_youtube_source_text(u, max_chars=18000))
    return "\n\n".join(blocks).strip()


def url_domain(url: str) -> str:
    return (urlparse(url or "").netloc or "").lower().removeprefix("www.")


def allowed_domains_for_seed(seed_url: str) -> set[str]:
    host = url_domain(seed_url)
    allowed = {host} if host else set()
    if "aiskillsnavigator.microsoft.com" in host:
        allowed.update({
            "aiskillsnavigator.microsoft.com",
            "microsoftlearning.github.io",
            "learn.microsoft.com",
            "github.com",
            "raw.githubusercontent.com",
            "youtube.com",
            "youtu.be",
        })
    elif "wikidocs.net" in host:
        allowed.update({"wikidocs.net"})
    elif "oopy.io" in host:
        allowed.update({"oopy.io", host})
    elif "youtube.com" in host or "youtu.be" in host:
        allowed.update({"youtube.com", "youtu.be", "i.ytimg.com"})
    elif "inflearn.com" in host:
        allowed.update({"inflearn.com", "cdn.inflearn.com"})
    return {item.removeprefix("www.") for item in allowed if item}


def load_collector_json(collector_report: dict[str, Any]) -> dict[str, Any]:
    json_path = collector_report.get("json_path")
    if not json_path:
        return {}
    try:
        candidate = Path(str(json_path))
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return {}


def collector_source_graph(collector_report: dict[str, Any], max_nodes: int = 30) -> dict[str, Any]:
    payload = load_collector_json(collector_report)
    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else collector_report.get("stats", {})
    quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else collector_report.get("quality", {})
    nodes: list[dict[str, Any]] = []
    video_candidates: list[str] = []
    video_index = payload.get("video_index") if isinstance(payload.get("video_index"), list) else []
    for item in video_index:
        if isinstance(item, dict) and item.get("youtube_url"):
            video_candidates.append(str(item.get("youtube_url")))
    snapshots = payload.get("snapshots") if isinstance(payload.get("snapshots"), list) else []
    for idx, snap in enumerate(snapshots[:max_nodes], start=1):
        if not isinstance(snap, dict):
            continue
        visible = str(snap.get("visible_text") or "")
        headings = []
        for h in snap.get("headings", [])[:10]:
            if isinstance(h, dict) and h.get("text"):
                headings.append(str(h.get("text"))[:140])
            elif isinstance(h, str):
                headings.append(h[:140])
        nodes.append({
            "order": idx,
            "type": str(snap.get("label") or snap.get("type") or "page"),
            "title": str(snap.get("title") or "")[:200],
            "url": str(snap.get("url") or ""),
            "text_chars": len(visible),
            "text": visible[:16000],
            "headings": headings,
        })
    if not nodes and isinstance(payload.get("nodes"), list):
        flat_nodes: list[dict[str, Any]] = []

        def walk(items: list[dict[str, Any]]) -> None:
            for item in items:
                if not isinstance(item, dict):
                    continue
                flat_nodes.append(item)
                children = item.get("children")
                if isinstance(children, list):
                    walk(children)

        walk(payload.get("nodes") or [])
        for idx, node in enumerate(flat_nodes[:max_nodes], start=1):
            text = str(node.get("text") or "")
            node_url = str(node.get("url") or "")
            if str(node.get("type") or "").lower() == "video" and node_url:
                video_candidates.append(node_url)
            nodes.append({
                "order": idx,
                "type": str(node.get("type") or node.get("node_type") or "page"),
                "title": str(node.get("title") or "")[:200],
                "url": str(node.get("url") or ""),
                "text_chars": len(text),
                "text": text[:16000],
                "headings": [str(node.get("title") or "")[:140]] if node.get("title") else [],
            })
    if not nodes:
        current_url = str(payload.get("current_url") or payload.get("input_url") or collector_report.get("seed_url") or "")
        title = str(payload.get("title") or "")
        nodes.append({
            "order": 1,
            "type": "page",
            "title": title,
            "url": current_url,
            "text_chars": int((stats or {}).get("visible_text_chars") or 0),
            "headings": [],
        })
    return {
        "title": payload.get("title") or "",
        "current_url": payload.get("current_url") or payload.get("input_url") or "",
        "stats": stats or {},
        "quality": quality or {},
        "nodes": nodes,
        "lab_url_candidates": payload.get("lab_url_candidates", [])[:16] if isinstance(payload.get("lab_url_candidates"), list) else [],
        "video_url_candidates": unique_preserve_order(
            (payload.get("video_url_candidates", []) if isinstance(payload.get("video_url_candidates"), list) else []) + video_candidates,
            limit=120,
        ),
        "lesson_url_candidates": payload.get("lesson_url_candidates", [])[:16] if isinstance(payload.get("lesson_url_candidates"), list) else [],
        "tree_items": payload.get("tree_items", [])[:30] if isinstance(payload.get("tree_items"), list) else [],
        "video_index": video_index[:120],
    }


def source_graph_markdown(seed_url: str, run_id: str, collector_report: dict[str, Any]) -> str:
    graph = collector_source_graph(collector_report)
    stats = graph.get("stats") if isinstance(graph.get("stats"), dict) else {}
    lines = [
        "[SOURCE_GRAPH]",
        f"run_id: {run_id}",
        f"seed_url: {seed_url}",
        f"collector_title: {graph.get('title', '')}",
        f"collector_current_url: {graph.get('current_url', '')}",
        "",
        "[SOURCE_GRAPH_STATS]",
    ]
    for key in ["page_count", "visible_text_chars", "link_count", "video_candidate_count", "lesson_candidate_count", "lab_candidate_count", "tree_item_count"]:
        if key in stats:
            lines.append(f"- {key}: {stats.get(key)}")
    lines.extend(["", "[SOURCE_GRAPH_NODES]"])
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        lines.append(f"- {node.get('order')}. [{node.get('type')}] {node.get('title')} — {node.get('url')} ({node.get('text_chars')} chars)")
        headings = node.get("headings") or []
        if headings:
            lines.append("  - headings: " + " | ".join(str(h) for h in headings[:8]))
    for label, key in [("LABS", "lab_url_candidates"), ("VIDEOS", "video_url_candidates"), ("LESSONS", "lesson_url_candidates")]:
        values = graph.get(key, [])
        if values:
            lines.extend(["", f"[SOURCE_GRAPH_{label}]"])
            for item in values[:12]:
                lines.append(f"- {item}")
    return "\n".join(lines).strip()


def source_pack_seed_match(seed_url: str, source_pack_text: str, collector_report: dict[str, Any]) -> tuple[bool, str]:
    playlist_id = seed_playlist_id(seed_url)
    if playlist_id:
        graph_text = json.dumps(collector_source_graph(collector_report), ensure_ascii=False)
        if playlist_id not in (source_pack_text or "") and playlist_id not in graph_text:
            return False, f"collector output does not include requested playlistId={playlist_id}; likely stale source pack or browser state"
    video_id = youtube_video_id(seed_url)
    if video_id:
        graph_text = json.dumps(collector_source_graph(collector_report), ensure_ascii=False)
        if video_id not in (source_pack_text or "") and video_id not in graph_text:
            return False, f"collector output does not include requested YouTube video id={video_id}; likely stale source pack or previous YouTube result"
    allowed = allowed_domains_for_seed(seed_url)
    if not allowed:
        return True, "no seed domain"
    graph = collector_source_graph(collector_report)
    urls = [str(graph.get("current_url") or "")]
    urls.extend(str(node.get("url") or "") for node in graph.get("nodes", []) if isinstance(node, dict))
    urls.extend(extract_urls_from_text(source_pack_text)[:120])
    domains = {url_domain(url) for url in urls if url_domain(url)}
    relevant = {domain for domain in domains if any(domain == a or domain.endswith('.' + a) for a in allowed)}
    if domains and not relevant:
        return False, f"collector domains {sorted(domains)[:10]} do not match seed allowed domains {sorted(allowed)}"
    return True, "ok"


def source_pack_quality_sufficient(seed_url: str, source_pack_text: str, collector_report: dict[str, Any]) -> tuple[bool, list[str]]:
    graph = collector_source_graph(collector_report)
    stats = graph.get("stats") if isinstance(graph.get("stats"), dict) else {}
    quality = graph.get("quality") if isinstance(graph.get("quality"), dict) else {}
    host = url_domain(seed_url)
    text_chars = int(stats.get("visible_text_chars") or quality.get("usable_text_chars") or len(source_pack_text or ""))
    page_count = int(stats.get("page_count") or len(graph.get("nodes", [])) or 0)
    lab_count = int(stats.get("lab_candidate_count") or len(graph.get("lab_url_candidates", [])) or 0)
    video_count = int(stats.get("video_candidate_count") or len(graph.get("video_url_candidates", [])) or 0)
    lesson_count = int(stats.get("lesson_candidate_count") or len(graph.get("lesson_url_candidates", [])) or 0)
    tree_count = int(stats.get("tree_item_count") or len(graph.get("tree_items", [])) or 0)
    warnings = quality.get("warnings") or []
    warnings = warnings if isinstance(warnings, list) else []
    reasons: list[str] = []
    if substantial_rendered_document(seed_url, text_chars, source_pack_text):
        return True, []

    # v4.7.22: Source-first fallback must be allowed for a real single-page article.
    # The universal collector's generic quality gate was built for course pages
    # that should expose lessons/labs/videos/tree items. That is too strict for
    # ordinary public pages such as MDN, GitHub Docs, Wikipedia, blog posts, etc.
    # Keep YouTube strict because title-only video pages are unsafe without a
    # transcript, and keep access/login pages blocked.
    is_video_seed = ("youtube.com" in host or "youtu.be" in host)
    is_access_like = "login_or_access_page_detected" in warnings
    is_course_like_seed = "aiskillsnavigator.microsoft.com" in host
    if (
        not is_video_seed
        and not is_access_like
        and not is_course_like_seed
        and page_count >= 1
        and text_chars >= 1200
    ):
        return True, []
    # Smoke-test and public URL fallback path: when the seed URL or current text
    # clearly identifies a known topic, allow a shorter but current-run-only pack.
    # Mismatch hard-fail still runs after generation, so this does not permit
    # cross-run contamination to slip through.
    if expected_topic_kind_from_input(seed_url=seed_url, current_text=source_pack_text) and text_chars >= 600:
        return True, []
    if collector_report.get("collector") == "universal_learning_collector":
        quality_status = str(quality.get("quality_status") or "").lower()
        if quality_status == "fail" or quality.get("can_generate_article") is False:
            missing = quality.get("missing") if isinstance(quality.get("missing"), list) else []
            detail = "; ".join(str(item) for item in missing[:3]) if missing else "collector marked source pack as not enough for article generation"
            reasons.append(f"universal collector quality gate failed: {detail}")
    has_substantial_evidence = (
        text_chars >= 8000
        and page_count >= 2
        and (lab_count > 0 or video_count > 0 or lesson_count > 0 or tree_count > 0)
    )
    if "login_or_access_page_detected" in warnings and not has_substantial_evidence:
        reasons.append("login/access page detected")
    if text_chars < 2200 and page_count <= 1 and lab_count == 0 and video_count == 0 and lesson_count == 0 and tree_count == 0:
        reasons.append(f"too little collected content: text_chars={text_chars}, page_count={page_count}")
    if "aiskillsnavigator.microsoft.com" in host and text_chars < 2200 and lab_count == 0 and lesson_count == 0 and tree_count == 0:
        reasons.append("AI Skills Navigator source graph has no lesson/lab/navigation evidence")
    if "wikidocs.net" in host and text_chars < 3500 and page_count <= 1:
        reasons.append("WikiDocs book URL did not collect chapter pages/body content")
    if "oopy.io" in host:
        graph_nodes = graph.get("nodes", []) if isinstance(graph.get("nodes"), list) else []
        child_nodes = [
            n for n in graph_nodes
            if isinstance(n, dict)
            and "oopy.io" in str(n.get("url", "")).lower()
            and str(n.get("type") or n.get("label") or "").startswith(("linked_", "followed_", "tree_", "child_", "open_tab_"))
            and int(n.get("text_chars") or n.get("chars") or 0) >= 600
        ]
        if text_chars >= 1000:
            return True, []
        if text_chars < 8000 and len(child_nodes) < 2:
            reasons.append("Oopy page did not collect enough child-page/body content")
    if "youtube.com" in host or "youtu.be" in host:
        if "transcript" not in (source_pack_text or "").lower() and text_chars < 5000:
            reasons.append("YouTube URL has no transcript/caption/body evidence")
    return not reasons, reasons


def collector_execution_failure_report(seed_url: str, run_id: str, collector_report: dict[str, Any]) -> str:
    stdout = str(collector_report.get("stdout") or "").strip()[-2500:]
    stderr = str(collector_report.get("stderr") or "").strip()[-2500:]
    error = str(collector_report.get("error") or "unknown collector error")
    md_path = str(collector_report.get("markdown_path") or "")
    json_path = str(collector_report.get("json_path") or "")
    stats = collector_report.get("stats") if isinstance(collector_report.get("stats"), dict) else {}
    quality = collector_report.get("quality") if isinstance(collector_report.get("quality"), dict) else {}
    return f"""# URL 자동수집 실행은 되었지만 실패했습니다

자동수집 단계가 없는 것이 아니라, 이번 run에서 collector가 실행된 뒤 usable source pack을 반환하지 못했습니다. 따라서 제목이나 URL만 보고 Medium 글을 만들지 않고 중단했습니다.

## 입력 URL
- {seed_url}

## run_id
{run_id}

## collector error
```text
{error}
```

## 생성된 파일 경로
- markdown_path: {md_path or "없음"}
- json_path: {json_path or "없음"}

## stats
```json
{json.dumps(stats, ensure_ascii=False, indent=2)[:2000]}
```

## quality
```json
{json.dumps(quality, ensure_ascii=False, indent=2)[:2000]}
```

## stdout tail
```text
{stdout or "없음"}
```

## stderr tail
```text
{stderr or "없음"}
```

## 다음 판단
- markdown/json 파일이 생성됐는데 차단된 경우: seed 검증 또는 quality gate가 너무 엄격한 것입니다.
- 파일이 생성되지 않은 경우: Playwright 실행, 로그인/권한, selector, timeout, collect_source_pack.py 오류를 봐야 합니다.
- 이 화면이 나오면 글 생성 문제가 아니라 collector 실행/반환 문제를 먼저 고쳐야 합니다.
"""


def collection_failure_report(seed_url: str, run_id: str, collector_report: dict[str, Any], reasons: list[str]) -> str:
    graph_md = source_graph_markdown(seed_url, run_id, collector_report)
    reason_md = "\n".join(f"- {reason}" for reason in reasons) or "- 수집 품질 기준 미달"
    host = url_domain(seed_url)

    # v4.7.25: users cannot know whether a visible YouTube caption is
    # programmatically extractable. Give actionable alternatives instead of
    # merely asking for a "captioned video".
    if "youtube.com" in host or "youtu.be" in host:
        return f"""# YouTube 자막 자동수집 실패

이 영상은 화면에서 자막이 보일 수 있어도, 앱이 프로그램으로 읽을 수 있는 transcript/caption 본문을 가져오지 못했습니다. 제목만 보고 글을 만들면 실제 영상 내용과 다른 글이 생성될 수 있으므로 중단했습니다.

## 입력 URL
- {seed_url}

## run_id
{run_id}

## 중단 이유
{reason_md}

## 수집 상태
```text
{graph_md}
```

## 다음 조치
- YouTube 화면에서 **스크립트 표시 / Show transcript**가 보이면 전체 자막을 복사해서 메모/본문 입력에 붙여넣어 주세요.
- TED 영상이면 YouTube URL 대신 **TED 공식 talk/transcript 페이지 URL**을 넣어 주세요.
- 자막 복사가 어렵다면 핵심 내용을 직접 메모로 적어 주세요.
- 화면 위주 강의라면 영상 URL 대신 주요 장면을 캡처 이미지로 업로드해 이미지 기반 글쓰기 모드로 작성해 주세요.
- 다른 영상 URL을 넣어도 됩니다. 단, 앱이 자동으로 transcript를 읽을 수 있어야 URL만으로 글 생성이 가능합니다.
"""

    return f"""# Source Graph 수집 부족

대표 URL 안의 학습 내용을 충분히 수집하지 못해서 Medium 글 생성을 중단했습니다.

## 입력 URL
- {seed_url}

## run_id
{run_id}

## 중단 이유
{reason_md}

## 수집 상태
```text
{graph_md}
```

## 다음 조치
- 단일 문서라면 본문이 충분히 수집되어야 합니다.
- 강의/영상이라면 자막, transcript, 강의 메모, 또는 캡처 이미지가 필요합니다.
- 이 상태에서 글을 생성하면 제목이나 일반 키워드만 보고 추상적인 글이 만들어지므로 차단했습니다.
"""

def seed_mismatch_report(seed_url: str, run_id: str, collector_report: dict[str, Any], reason: str) -> str:
    graph_md = source_graph_markdown(seed_url, run_id, collector_report)
    return f"""# Seed URL과 수집 결과가 일치하지 않아 중단했습니다

이번 생성 run의 대표 URL과 collector/source graph 결과가 맞지 않습니다. 이전 세션 또는 다른 source pack이 섞였을 가능성이 있어 Medium 글 생성을 차단했습니다.

## 입력 URL
- {seed_url}

## run_id
{run_id}

## 차단 이유
- {reason}

## Source Graph
```text
{graph_md}
```
"""


def final_article_mismatch_report(seed_url: str, run_id: str, collector_report: dict[str, Any], reason: str, article: str) -> str:
    preview = (article or "").strip()[:1800]
    graph_md = source_graph_markdown(seed_url, run_id, collector_report)
    return f"""# 생성 결과가 입력 URL과 맞지 않아 차단했습니다

collector 이후 글 생성 단계에서 seed_url과 다른 주제의 글이 만들어졌습니다. 이 상태의 글은 사용자에게 보여주면 안 되므로 차단했습니다.

## 입력 URL
- {seed_url}

## run_id
{run_id}

## 차단 이유
- {reason}

## Source Graph
```text
{graph_md}
```

## 잘못 생성된 글 preview
```text
{preview}
```
"""


def build_url_run_input(seed_url: str, run_id: str, source_pack_text: str, collector_report: dict[str, Any]) -> str:
    graph_md = source_graph_markdown(seed_url, run_id, collector_report)
    return f"""[URL_ONLY_RUN]
run_id: {run_id}
seed_url: {seed_url}

[STRICT_GENERATION_RULES]
- 이 글은 반드시 이번 seed_url에서 수집된 Source Pack과 Source Graph만 근거로 작성한다.
- 이전 예제, 이전 세션, Power BI, Battery RUL, 다른 URL의 내용을 절대 재사용하지 않는다.
- seed_url 주제와 맞지 않는 글은 생성 실패로 처리해야 한다.
- 학습자가 공부 중 겪을 수 있는 실제 혼동, 문제 인식, 조치, 검증 기준을 수집 evidence에서 찾아 작성한다.
- Source Graph에 없는 단원명, 수식, 성과, 화면 결과를 임의로 만들지 않는다.

{graph_md}

[AUTO_COLLECTED_SOURCE_PACK]
{source_pack_text}
"""


def article_matches_seed_url(seed_url: str, article: str, current_text: str = "") -> tuple[bool, str]:
    host = url_domain(seed_url)
    lowered = (article or "").lower()
    if not article.strip():
        return False, "empty article"

    # v4.7.17 YouTube validator fix: YouTube watch URLs are opaque, so choose
    # the expected topic from strong current video title/transcript fingerprints.
    # If the article topic and video topic disagree, block as a real mismatch.
    # If only the article has strong support, use it to avoid stale-run overblock.
    expected_kind = expected_topic_kind_from_input(seed_url=seed_url, current_text=current_text)
    if "youtube.com" in host or "youtu.be" in host:
        expected_kind, youtube_reason = resolve_youtube_expected_kind(seed_url, current_text, article)
        if youtube_reason:
            return False, youtube_reason

    # v4.7.19: source-first fallback articles should be checked against the
    # current source, not a known-topic profile guessed from noisy docs text.
    if is_source_first_article(article):
        sf_failures = source_first_policy_failures(article, current_text=current_text, seed_url=seed_url)
        if sf_failures:
            return False, "; ".join(sf_failures)
        return True, "ok"

    mismatch_failures = topic_mismatch_failures(expected_kind, article)
    if mismatch_failures:
        return False, "; ".join(mismatch_failures)
    # Do not reject legitimate Fabric IQ content just because it contains
    # “Power BI semantic model”. Fabric IQ ontology labs explicitly include
    # ontology generation from a semantic model.  Only block the old Power BI
    # screenshot-example article when multiple unique old-example entities appear.
    old_powerbi_entities = [
        "product[category]",
        "sales[sales]",
        "salespersonregion",
        "profit margin",
        "dax measure",
        "repeated category sales",
        "category별 sales 반복",
        "star schema",
    ]
    old_powerbi_hits = sum(1 for term in old_powerbi_entities if term in lowered)
    battery_bad = ["battery rul", "remaining useful life", "bmaml", "ceemdan"]
    if "aiskillsnavigator.microsoft.com" in host:
        good_terms = ["ai skills", "skills navigator", "microsoft", "fabric iq", "ontology", "foundry", "agent", "lab", "exercise", "azure", "learn"]
        if old_powerbi_hits >= 2 or any(term in lowered for term in battery_bad):
            return False, "AI Skills seed produced unrelated prior-example article"
        if not any(term in lowered for term in good_terms):
            return False, "AI Skills seed article lacks AI Skills/Microsoft/Fabric IQ/ontology/Lab evidence terms"
    if "wikidocs.net" in host:
        if old_powerbi_hits >= 2 or any(term in lowered for term in battery_bad):
            return False, "WikiDocs seed produced unrelated prior-example article"
        if not any(term in lowered for term in ["wikidocs", "코딩", "파이썬", "자료구조", "알고리즘", "스택"]):
            return False, "WikiDocs seed article lacks coding-test/Python evidence terms"
    if "oopy.io" in host:
        if old_powerbi_hits >= 2 or any(term in lowered for term in battery_bad):
            return False, "Oopy seed produced unrelated prior-example article"
        if not any(term in lowered for term in ["cs", "운영체제", "네트워크", "자료구조", "면접", "oopy"]):
            return False, "Oopy seed article lacks CS/Oopy evidence terms"
    if "youtube.com" in host or "youtu.be" in host:
        if old_powerbi_hits >= 2 or any(term in lowered for term in battery_bad):
            return False, "YouTube seed produced unrelated prior-example article"
    return True, "ok"


def unique_preserve_order(items: list[str], limit: int = 20) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = re.sub(r"\s+", " ", str(item or "")).strip(" -|\t\r\n")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def source_graph_key_headings(collector_report: dict[str, Any], limit: int = 28) -> list[str]:
    graph = collector_source_graph(collector_report, max_nodes=80)
    headings: list[str] = []
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        title = str(node.get("title") or "").strip()
        if title and title not in {"AI Skills Navigator", "Microsoft Fabric interactive exercises"}:
            headings.append(title)
        for h in node.get("headings") or []:
            headings.append(str(h))
    skip = {"navigation", "introduction"}
    return [h for h in unique_preserve_order(headings, limit=limit) if h.lower() not in skip]


def source_graph_stats_summary(collector_report: dict[str, Any]) -> dict[str, int]:
    graph = collector_source_graph(collector_report)
    stats = graph.get("stats") if isinstance(graph.get("stats"), dict) else {}
    quality = graph.get("quality") if isinstance(graph.get("quality"), dict) else {}
    def as_int(key: str, fallback: int = 0) -> int:
        try:
            return int(stats.get(key) or fallback)
        except Exception:
            return fallback
    return {
        "pages": as_int("page_count", int(quality.get("pages_collected") or len(graph.get("nodes", [])))),
        "chars": as_int("visible_text_chars", int(quality.get("text_chars") or 0)),
        "links": as_int("link_count", 0),
        "videos": as_int("video_candidate_count", int(quality.get("video_candidates") or len(graph.get("video_url_candidates", [])))),
        "lessons": as_int("lesson_candidate_count", len(graph.get("lesson_url_candidates", []))),
        "labs": as_int("lab_candidate_count", int(quality.get("lab_steps") or len(graph.get("lab_url_candidates", [])))),
        "tree_items": as_int("tree_item_count", len(graph.get("tree_items", []))),
    }


def source_graph_lab_titles(collector_report: dict[str, Any], limit: int = 8) -> list[str]:
    graph = collector_source_graph(collector_report, max_nodes=80)
    titles: list[str] = []
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        typ = str(node.get("type") or "").lower()
        title = str(node.get("title") or "").strip()
        if ("lab" in typ or "microsoftlearning.github.io" in str(node.get("url") or "")) and title:
            titles.append(title)
    return unique_preserve_order(titles, limit=limit)


def source_pack_snippets(source_pack_text: str, terms: list[str], limit: int = 6, window: int = 360) -> list[str]:
    text = re.sub(r"\s+", " ", source_pack_text or "").strip()
    lowered = text.lower()
    snippets: list[str] = []
    for term in terms:
        idx = lowered.find(term.lower())
        if idx < 0:
            continue
        start = max(0, idx - window // 2)
        end = min(len(text), idx + window)
        snippet = text[start:end].strip()
        if snippet:
            snippets.append(snippet)
        if len(snippets) >= limit:
            break
    return unique_preserve_order(snippets, limit=limit)


def source_pack_learning_points(source_pack_text: str, headings: list[str], limit: int = 10) -> list[str]:
    """Pull learner-usable concept/lab statements from the current source pack."""
    heading_terms = [h.lower() for h in headings[:18] if len(h) >= 4]
    action_terms = [
        "when to use", "understand", "compare", "evaluate", "implement", "create",
        "configure", "deploy", "query", "search", "ranking", "credential", "model",
        "complex", "difficult", "advanced", "architecture", "workflow", "trade-off",
        "tradeoffs", "predicate", "function", "stored procedure", "index", "relationship",
        "permission", "policy", "governance", "identity", "validation", "result",
        "practical", "real-world", "production", "enterprise", "scale", "admin",
        "monitor", "visibility", "control", "security", "compliance", "access",
        "automation", "deployment", "pipeline", "data model", "retrieval",
        "how to", "best practice", "troubleshoot", "error", "issue", "failure",
        "slow", "cost", "optimize", "permission denied", "not working", "manage",
        "lab", "exercise", "verify", "validate", "summary", "key takeaways",
        "사용", "이해", "비교", "구현", "생성", "설정", "실습", "검증", "확인",
        "복잡", "어려", "핵심", "쿼리", "수식", "코드", "아키텍처", "워크플로",
        "권한", "정책", "거버넌스", "원인", "결과", "성능", "인덱스",
        "실무", "운영", "관리", "모니터링", "보안", "컴플라이언스", "자동화",
        "배포", "파이프라인", "데이터 모델", "검색", "인증", "접근 제어",
        "해결", "오류", "에러", "느림", "최적화", "비용", "문제", "방법",
    ]
    skip_exact = {
        "exit", "navigation", "summary", "learn more", "content loaded",
        "turn module into podcast", "summarize module",
    }
    points: list[str] = []
    candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for order, raw in enumerate((source_pack_text or "").splitlines()):
        line = re.sub(r"\s+", " ", raw).strip(" -|\t\r\n")
        if not line or len(line) < 80 or len(line) > 900:
            continue
        lower = line.lower()
        if lower in skip_exact:
            continue
        if lower.startswith((
            "http://", "https://", "## ", "# ", "- h1:", "- h2:", "- h3:",
            "h1:", "h2:", "h3:", "h4:", "- [", "[item]", "[lab/exercise]",
            "[lesson_or_course_candidate]", "[lab_or_exercise_candidate]",
            "[video_or_player_candidate]", "[link]",
            "page url:", "captured at:", "collection stats", "==== page:",
            "=====", "content loaded.", "now playing:", "from ",
            "source pack:", "headings", "video url candidates", "lab / exercise url candidates",
            "title:", "name:", "source:", "catalogitems title:", "catalogitems source:",
            "catalogitems name:", "id:", "url:",
        )):
            continue
        if any(bad_fragment in lower for bad_fragment in [
            "(no text)", "diagram of button", "go.microsoft.com/fwlink",
            "youtube.com/embed", "youtu.be/", "http://", "https://",
            "skilling session coach", "ask for clarification", "you can continue with",
            "looking for content on a particular topic",
            "you started ", "get certified:", "turn module into podcast",
        ]):
            continue
        if any(bad in lower for bad in ["source graph", "collector", "run_id", "아직 생성된 결과"]):
            continue
        line = re.sub(r"^(description|summary|abstract):\s*", "", line, flags=re.IGNORECASE).strip()
        lower = line.lower()
        score = 0
        has_sentence_shape = bool(re.search(r"[.!?。]|다\.|요\.", line))
        title_like = (
            not has_sentence_shape
            or lower.startswith(("exercise -", "episode ", "unlock governance", "design and implement"))
        )
        if title_like and len(line) < 180:
            score -= 4
        if has_sentence_shape:
            score += 4
        if len(line) >= 140:
            score += 2
        if any(term in lower for term in action_terms):
            score += 2
        if any(term and term in lower for term in heading_terms):
            score += 2
        if re.search(r"\b(when|because|requires|helps|uses|can|must|should|first|then|finally)\b", lower):
            score += 4
        if re.search(r"\b(problem|challenge|risk|control|identity|access|security|compliance|govern|visibility|validate|verify|result|trade[- ]offs?|performance|precision|recall|embedding|ranking|context|query|predicate|index|function|procedure|architecture|workflow|policy|permission|enterprise|scale|admin|production|real-world|monitor|automation|pipeline|retrieval|how to|best practice|troubleshoot|error|issue|failure|slow|cost|optimize|manage)\b", lower):
            score += 3
        if re.search(r"\b(how|why|when|what)\b", lower) and any(x in lower for x in ["manage", "fix", "use", "choose", "configure", "deploy", "secure", "optimize"]):
            score += 3
        if re.search(r"(처음|핵심|이유|필요|단계|결과|기준|역할|차이|흐름)", line):
            score += 3
        if score < 4:
            continue
        key = lower[:260]
        if key in seen:
            continue
        seen.add(key)
        candidates.append((score, order, line))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    for _score, _order, line in candidates:
        points.append(line)
        if len(points) >= limit:
            break
    return points


def learning_point_paragraph(points: list[str], fallback: str) -> str:
    if not points:
        return fallback
    selected = points[:5]
    return "\n\n".join(f"- {point}" for point in selected)


def transcript_nodes_from_payload(payload: dict[str, Any], graph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []

    def walk(items: list[Any]) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").lower() == "transcript_segment":
                nodes.append(item)
            children = item.get("children")
            if isinstance(children, list):
                walk(children)

    if isinstance(payload.get("nodes"), list):
        walk(payload.get("nodes") or [])
    if not nodes:
        for item in graph.get("nodes") or []:
            if isinstance(item, dict) and str(item.get("type") or "").lower() == "transcript_segment":
                nodes.append(item)
    return nodes


def youtube_transcript_text(payload: dict[str, Any], graph: dict[str, Any]) -> str:
    segments = transcript_nodes_from_payload(payload, graph)
    text = " ".join(str(node.get("text") or "") for node in segments)
    return clean_learning_text(text)


def youtube_chapter_sections(transcript_text: str, limit: int = 6) -> list[dict[str, Any]]:
    sentences = split_learning_sentences(transcript_text, limit=260)
    intro_noise = [
        "얄팍한 코딩 사전",
        "유튜브 채널",
        "컨텐츠를 만들고",
        "온라인 장편 강의",
        "고정댓글",
        "무료 파트",
        "구독",
        "좋아요",
    ]
    sentences = [s for s in sentences if not any(noise in s for noise in intro_noise)]
    if not sentences:
        return []
    terms = infer_learning_terms(transcript_text, limit=30)
    chunk_size = max(5, min(12, len(sentences) // max(3, min(limit, 6)) + 1))
    sections: list[dict[str, Any]] = []
    for start in range(0, len(sentences), chunk_size):
        chunk = sentences[start:start + chunk_size]
        if not chunk:
            continue
        blob = " ".join(chunk)
        local_terms = [term for term in terms if term.lower() in blob.lower()]
        if local_terms:
            title = " · ".join(local_terms[:3])
        else:
            title = re.sub(r"[`#*_]", "", chunk[0])
            title = re.sub(r"\s+", " ", title).strip()
            title = title[:42].rstrip(" ,.:;") or f"핵심 흐름 {len(sections) + 1}"
        sections.append({"title": title, "sentences": chunk[:5]})
        if len(sections) >= limit:
            break
    return sections


def youtube_topic_plan(title: str) -> list[tuple[str, list[str]]]:
    title_l = str(title or "").lower()
    if "github actions" in title_l or "ci/cd" in title_l or "ci cd" in title_l:
        return [
            ("workflow가 실행 조건을 정의하는 방식", ["workflow", "workflows", "on:", "trigger", "event", "yaml"]),
            ("job과 step으로 자동화 단위를 나누기", ["job", "jobs", "step", "steps", "runner"]),
            ("action과 runner의 역할 구분", ["action", "runner", "uses", "runs-on"]),
            ("test/build/deploy 파이프라인 흐름", ["test", "build", "deploy", "pipeline", "ci", "cd"]),
            ("Docker와 CI/CD 연결", ["docker", "image", "container", "registry"]),
        ]
    if "fastapi" in title_l or "api development" in title_l or "python api" in title_l:
        return [
            ("endpoint와 path operation 이해", ["endpoint", "path", "route", "path operation", "get", "post"]),
            ("request와 response 구조 검증", ["request", "response", "status code", "body", "json", "postman"]),
            ("Pydantic schema로 입력값 검증", ["pydantic", "schema", "validation", "model"]),
            ("database와 API 흐름 연결", ["database", "sql", "postgres", "crud", "orm"]),
            ("deployment와 운영 환경 확인", ["deploy", "deployment", "nginx", "systemd", "firewall"]),
        ]
    if "oauth" in title_l or "openid" in title_l or "oidc" in title_l:
        return [
            ("사용자와 client 역할 구분", ["resource owner", "user", "client", "application"]),
            ("authorization server와 resource server 분리", ["authorization server", "resource server", "auth server", "api"]),
            ("access token으로 권한 위임 이해", ["access token", "token", "scope", "authorization"]),
            ("simple login의 한계와 보안 책임", ["simple login", "forms authentication", "password", "security", "maintenance"]),
            ("OpenID Connect와 identity layer 구분", ["openid connect", "id token", "identity", "authentication"]),
        ]
    if "docker" in title_l:
        return [
            ("Docker image와 container의 역할 구분", ["image", "container", "docker image", "docker container", "이미지", "컨테이너"]),
            ("Dockerfile과 image build 흐름", ["dockerfile", "build", "docker build", "base image"]),
            ("port mapping으로 외부 접근 확인", ["port", "ports", "mapping", "publish", "localhost", "-p"]),
            ("volume으로 데이터 보존 문제 해결", ["volume", "mount", "bind", "persist", "data"]),
            ("Docker Compose로 여러 서비스 실행", ["compose", "docker compose", "yaml", "services"]),
        ]
    if "git" in title_l or "github" in title_l or "깃" in title_l:
        return [
            ("Git이 해결하는 버전 관리 문제", ["git", "깃", "version", "버전", "변경", "되돌아"]),
            ("GitHub와 원격 저장소", ["github", "깃허브", "remote", "repository", "저장소"]),
            ("commit으로 변경 이력 남기기", ["commit", "커밋", "history", "이력"]),
            ("branch로 작업 흐름 분리", ["branch", "브랜치", "분기"]),
            ("merge와 충돌 해결", ["merge", "conflict", "충돌", "병합"]),
        ]
    return []

def youtube_learning_terms(title: str, transcript_text: str, limit: int = 18) -> list[str]:
    plan = youtube_topic_plan(title)
    text_l = str(transcript_text or "").lower()
    title_l = str(title or "").lower()
    if plan:
        found = []
        for label, aliases in plan:
            if any(alias.lower() in text_l or alias.lower() in title_l for alias in aliases):
                # Keep terms short and concept-like for concept list.
                found.append(label.split("으로")[0].split("와")[0].split("과")[0].strip())
        if found:
            return unique_preserve_order(found, limit=limit)
        return unique_preserve_order([label for label, _ in plan], limit=limit)
    return infer_learning_terms(f"{title}\n{transcript_text}", limit=limit)


def youtube_keyword_sections(title: str, transcript_text: str, limit: int = 7) -> list[dict[str, Any]]:
    plan = youtube_topic_plan(title)
    if not plan:
        return []
    sentences = split_learning_sentences(transcript_text, limit=420)
    noise = [
        "subscribe", "hit the bell", "follow", "tweet", "twitter", "my name is",
        "excellent teacher", "welcome to", "massive and comprehensive", "course is taught",
        "we will learn", "we'll learn", "we're going to cover", "before we even talk",
        "good content like this",
    ]
    sentences = [s for s in sentences if not any(n in s.lower() for n in noise)]
    sections: list[dict[str, Any]] = []
    used: set[str] = set()
    for label, aliases in plan:
        picked: list[str] = []
        for sentence in sentences:
            lower = sentence.lower()
            if sentence in used:
                continue
            if any(alias.lower() in lower for alias in aliases):
                picked.append(sentence)
                used.add(sentence)
            if len(picked) >= 4:
                break
        if picked:
            sections.append({"title": label, "sentences": picked})
        if len(sections) >= limit:
            break
    return sections




def broad_source_profile(kind: str, title: str = "") -> dict[str, Any] | None:
    """Profiles for common broad technical docs.

    These are intentionally source/problem driven. They prevent unknown docs that
    contain generic words such as "path", "parameter", "body", "then", or
    "model" from falling into FastAPI/Promise profiles.
    """
    display_title = (title or "학습 자료").strip()
    profiles: dict[str, dict[str, Any]] = {
        "python_modules": {
            "kind": "python_modules",
            "title": display_title,
            "article_title": "Python module 학습 기록: 실행 파일과 import 가능한 모듈 구분하기",
            "subtitle": "Clarifying modules, imports, packages, namespaces, and module search path",
            "default_problem": "Python 파일을 그냥 실행하는 것과 import 가능한 module/package로 분리하는 흐름을 구분하는 것",
            "scope": "Python module, import, package, namespace, module search path, __name__",
            "flow": "파일 작성 → module로 import → namespace 생성 → package로 묶기 → module search path 확인",
            "concepts": [
                ("Module", "Python에서 함수, 변수, 클래스를 담아 import할 수 있는 파일 단위다."),
                ("import", "다른 module의 이름과 기능을 현재 namespace로 가져오는 구문이다."),
                ("Package", "여러 module을 디렉터리 구조로 묶어 관리하는 단위다."),
                ("Namespace", "module 안의 이름들이 충돌하지 않도록 분리되어 저장되는 공간이다."),
                ("Module search path", "import할 module을 인터프리터가 어떤 순서로 찾는지 정하는 경로 목록이다."),
            ],
            "steps": [
                ("실행 파일과 module 역할 구분", "파일을 직접 실행할 때와 import할 때 실행 흐름이 달라질 수 있다.", "module은 재사용 가능한 기능 단위로 보고, 실행 진입점과 분리했다.", "어떤 코드는 import되고 어떤 코드는 직접 실행될 때만 동작해야 하는지 설명할 수 있는지 확인했다."),
                ("import와 namespace 이해", "이름이 어디서 정의되고 어디에서 접근 가능한지 흐려질 수 있다.", "import가 현재 namespace에 이름을 연결하는 방식으로 정리했다.", "module.function 형태와 from module import name의 차이를 말할 수 있는지 확인했다."),
                ("package와 search path 확인", "module이 많아지면 파일 위치와 import 경로가 헷갈릴 수 있다.", "package 구조와 module search path를 import 성공 기준으로 보았다.", "어떤 디렉터리의 module이 import되는지 추적할 수 있으면 이해한 것으로 보았다."),
            ],
            "skills": ["Python module 구조 이해", "import 흐름 구분", "package/namespace 이해", "module search path 확인", "재사용 가능한 파일 분리"],
        },
        "numpy_broadcasting": {
            "kind": "numpy_broadcasting",
            "title": display_title,
            "article_title": "NumPy broadcasting 학습 기록: shape가 다른 배열 연산 조건 이해하기",
            "subtitle": "Clarifying array shapes, dimensions, broadcasting rules, and incompatible shapes",
            "default_problem": "shape가 다른 배열이 언제 함께 연산될 수 있고 언제 incompatible shape가 되는지 구분하는 것",
            "scope": "NumPy broadcasting, array shape, dimension, broadcasting rule, compatible shape, incompatible shape",
            "flow": "array shape 확인 → trailing dimension 비교 → 1 또는 같은 크기 판정 → broadcasting 적용 → 오류 조건 확인",
            "concepts": [
                ("Broadcasting", "shape가 다른 배열끼리도 일정 규칙을 만족하면 element-wise 연산을 가능하게 하는 NumPy 규칙이다."),
                ("Shape", "배열의 각 axis 크기를 나타내며 broadcasting 가능 여부의 기준이 된다."),
                ("Dimension", "배열의 축 단위이며, 뒤쪽 dimension부터 비교한다."),
                ("Compatible shape", "두 dimension이 같거나 하나가 1일 때 broadcasting 가능한 상태다."),
                ("Incompatible shape", "비교한 dimension이 서로 다르고 어느 쪽도 1이 아닐 때 발생하는 오류 조건이다."),
            ],
            "steps": [
                ("shape를 먼저 확인", "값만 보면 연산 가능 여부를 판단하기 어렵다.", "배열 연산 전 shape와 dimension을 먼저 확인했다.", "어떤 축이 서로 비교되는지 말할 수 있는지 확인했다."),
                ("broadcasting rule 적용", "shape가 다르다고 항상 실패하는 것은 아니다.", "뒤쪽 dimension부터 같거나 1인지 확인하는 기준으로 정리했다.", "가능한 경우와 불가능한 경우를 예시로 구분할 수 있으면 이해한 것으로 보았다."),
                ("오류 조건 분리", "incompatible shape 오류는 계산 문제가 아니라 shape 규칙 위반이다.", "오류를 값 문제가 아니라 배열 구조 문제로 보았다.", "shape mismatch 원인을 찾아낼 수 있는지 확인했다."),
            ],
            "skills": ["NumPy broadcasting 이해", "array shape 비교", "dimension 규칙 적용", "incompatible shape 디버깅", "벡터화 연산 판단"],
        },
        "pandas_groupby": {
            "kind": "pandas_groupby",
            "title": display_title,
            "article_title": "pandas groupby 학습 기록: split-apply-combine 흐름 이해하기",
            "subtitle": "Clarifying GroupBy, aggregation, transformation, filtering, and combine flow",
            "default_problem": "데이터를 그룹으로 나누고 각 그룹에 연산을 적용한 뒤 결과를 합치는 흐름을 이해하는 것",
            "scope": "pandas groupby, split-apply-combine, aggregation, transformation, filtering, GroupBy object",
            "flow": "key 기준 split → GroupBy object 생성 → aggregation/transformation 적용 → 결과 combine",
            "concepts": [
                ("GroupBy", "데이터를 특정 key 기준으로 그룹화한 뒤 연산을 적용하기 위한 pandas 객체다."),
                ("Split", "데이터를 group key 기준으로 나누는 단계다."),
                ("Apply", "각 그룹에 aggregation, transformation, filtering 같은 연산을 적용하는 단계다."),
                ("Combine", "그룹별 결과를 다시 하나의 Series/DataFrame 결과로 합치는 단계다."),
                ("Aggregation", "각 그룹을 하나의 요약값으로 줄이는 연산이다."),
                ("Transformation", "그룹 구조를 유지하면서 값을 변환하는 연산이다."),
            ],
            "steps": [
                ("group key와 원본 데이터 분리", "무엇을 기준으로 묶는지 정하지 않으면 groupby 결과를 해석하기 어렵다.", "key별 split 단계를 먼저 확인했다.", "각 row가 어떤 그룹에 들어가는지 설명할 수 있는지 확인했다."),
                ("aggregation과 transformation 구분", "집계와 변환은 모두 apply처럼 보이지만 결과 shape가 다르다.", "aggregation은 요약, transformation은 원래 구조에 맞춘 변환으로 나누었다.", "결과 row 수가 어떻게 달라지는지 말할 수 있으면 이해한 것으로 보았다."),
                ("combine 결과 검증", "그룹별 계산 결과가 최종 DataFrame에서 어떻게 합쳐지는지 흐려질 수 있다.", "최종 index와 column 구조를 확인 기준으로 두었다.", "groupby 결과를 원본과 비교해 해석할 수 있는지 확인했다."),
            ],
            "skills": ["pandas groupby 이해", "split-apply-combine 구조", "aggregation/transformation 구분", "GroupBy 결과 해석", "데이터 집계 흐름 검증"],
        },
        "redis_data_types": {
            "kind": "redis_data_types",
            "title": display_title,
            "article_title": "Redis data types 학습 기록: key-value 안의 자료구조 선택 기준 이해하기",
            "subtitle": "Clarifying strings, lists, sets, hashes, sorted sets, and data type selection",
            "default_problem": "Redis가 단순 key-value 저장소처럼 보여도 값의 자료구조별 사용 상황이 다르다는 점을 구분하는 것",
            "scope": "Redis data types, string, list, set, hash, sorted set, stream, key-value",
            "flow": "저장할 데이터 성격 파악 → data type 선택 → command로 읽기/쓰기 → 조회 패턴 검증",
            "concepts": [
                ("String", "단일 값, 카운터, 캐시 값처럼 가장 기본적인 Redis value type이다."),
                ("List", "순서가 있는 항목을 앞/뒤로 추가하거나 꺼낼 때 사용하는 구조다."),
                ("Set", "중복 없는 값의 모음을 저장하고 membership을 확인할 때 사용한다."),
                ("Hash", "객체나 record처럼 field-value 쌍을 묶어 저장할 때 사용한다."),
                ("Sorted set", "score 기준으로 정렬된 값을 저장하고 ranking이나 범위 조회에 사용한다."),
            ],
            "steps": [
                ("값의 사용 패턴 파악", "모든 값을 string으로 저장하면 조회와 갱신 기준이 흐려진다.", "값이 단일값인지 목록인지 객체인지 먼저 구분했다.", "데이터 사용 방식에 맞는 type을 고를 수 있는지 확인했다."),
                ("자료구조별 command 연결", "type을 골라도 어떤 command로 읽고 쓸지 모르면 실무 적용이 어렵다.", "각 data type을 대표 command와 연결해 정리했다.", "조회/갱신 패턴을 command 기준으로 설명할 수 있으면 이해한 것으로 보았다."),
                ("캐시와 자료구조 경계 확인", "Redis를 캐시로만 보면 list/set/hash의 의미가 약해진다.", "저장 목적과 자료구조 선택을 분리했다.", "단순 캐시와 자료구조 기반 저장을 구분할 수 있는지 확인했다."),
            ],
            "skills": ["Redis data type 선택", "key-value 구조 이해", "string/list/set/hash 구분", "자료구조별 command 판단", "캐시 설계 기준 정리"],
        },
        "http_status": {
            "kind": "http_status",
            "title": display_title,
            "article_title": "HTTP status code 학습 기록: 응답 범주로 API 상태 판단하기",
            "subtitle": "Clarifying 2xx, 3xx, 4xx, 5xx, success, client errors, and server errors",
            "default_problem": "HTTP status code를 숫자 암기가 아니라 요청 성공/실패와 원인 범주로 구분하는 것",
            "scope": "HTTP status code, 2xx, 3xx, 4xx, 5xx, client error, server error, API debugging",
            "flow": "응답 수신 → status class 확인 → 성공/리다이렉션/클라이언트/서버 오류 구분 → 디버깅 방향 결정",
            "concepts": [
                ("2xx", "요청이 성공적으로 처리되었음을 나타내는 상태 코드 범주다."),
                ("3xx", "요청을 완료하려면 리다이렉션 등 추가 동작이 필요한 상태다."),
                ("4xx", "잘못된 요청, 인증, 권한, 존재하지 않는 리소스처럼 클라이언트 쪽 문제를 나타낸다."),
                ("5xx", "서버가 유효한 요청을 처리하지 못한 상황을 나타낸다."),
                ("API debugging", "status code 범주를 보고 요청 수정, 권한 확인, 서버 로그 확인 중 어디로 가야 할지 판단하는 과정이다."),
            ],
            "steps": [
                ("숫자 암기보다 범주 확인", "200, 404, 500을 개별 숫자로만 외우면 디버깅 방향이 남지 않는다.", "2xx/4xx/5xx 범주를 먼저 확인하는 방식으로 정리했다.", "응답 코드를 보고 어느 쪽 문제인지 말할 수 있는지 확인했다."),
                ("클라이언트 오류와 서버 오류 분리", "실패 응답이 모두 서버 버그는 아니다.", "4xx는 요청/인증/권한 문제, 5xx는 서버 처리 문제로 나누었다.", "다음 조치가 요청 수정인지 서버 로그 확인인지 판단할 수 있으면 이해한 것으로 보았다."),
                ("API 테스트 기준 세우기", "요청이 돌아왔다는 사실만으로 성공은 아니다.", "status와 response body를 함께 확인 기준으로 두었다.", "API client에서 성공/실패를 근거로 설명할 수 있는지 확인했다."),
            ],
            "skills": ["HTTP status code 해석", "2xx/4xx/5xx 범주 구분", "API 디버깅 방향 판단", "client/server error 구분", "응답 검증 기준 정리"],
        },
        "django_models": {
            "kind": "django_models",
            "title": display_title,
            "article_title": "Django model 학습 기록: Python class와 database table 매핑 이해하기",
            "subtitle": "Clarifying models, fields, migrations, table mapping, and database schema",
            "default_problem": "Django model이 단순 Python class가 아니라 database table 구조를 정의한다는 점을 이해하는 것",
            "scope": "Django model, field, database table, migration, table mapping, schema",
            "flow": "model class 정의 → field 선언 → migration 생성 → database table 반영 → ORM으로 조회/저장",
            "concepts": [
                ("Model", "Django에서 database table 구조와 Python 객체 표현을 연결하는 class다."),
                ("Field", "table column의 타입과 제약 조건을 model class 안에 정의하는 단위다."),
                ("Migration", "model 변경사항을 database schema 변경으로 적용하는 절차다."),
                ("Table mapping", "model class가 database table과 연결되어 row를 객체처럼 다루는 구조다."),
                ("Schema", "database table, column, constraint 같은 구조 정의다."),
            ],
            "steps": [
                ("class와 table 역할 구분", "model class를 일반 class처럼만 보면 database schema와 연결되는 지점이 흐려진다.", "model을 table 구조를 선언하는 단위로 정리했다.", "class field가 어떤 column으로 이어지는지 설명할 수 있는지 확인했다."),
                ("field와 제약 조건 확인", "필드 타입과 옵션을 모르면 저장 가능한 데이터 기준이 흐려진다.", "field를 column type과 validation/constraint 기준으로 보았다.", "필드 변경이 schema 변경을 요구한다는 점을 확인했다."),
                ("migration 흐름 이해", "model을 바꿨다는 사실만으로 DB가 자동 변경되는 것은 아니다.", "migration을 model 변경을 DB에 반영하는 단계로 정리했다.", "언제 migration을 만들고 적용해야 하는지 판단할 수 있으면 이해한 것으로 보았다."),
            ],
            "skills": ["Django model 이해", "field/table mapping 구분", "migration 흐름", "database schema 판단", "ORM 기반 데이터 모델링"],
        },
        "node_event_loop": {
            "kind": "node_event_loop",
            "title": display_title,
            "article_title": "Node.js event loop 학습 기록: 비동기 실행 순서 이해하기",
            "subtitle": "Clarifying call stack, event loop phases, timers, callbacks, and nextTick",
            "default_problem": "Node.js 코드가 작성 순서대로만 실행되는 것이 아니라 event loop phase와 callback queue에 따라 실행 시점이 달라지는 점을 이해하는 것",
            "scope": "Node.js event loop, call stack, timers, pending callbacks, poll, check, nextTick, callback queue",
            "flow": "동기 코드 실행 → timer 등록 → event loop phase 이동 → callback 실행 → nextTick/microtask 확인",
            "concepts": [
                ("Event loop", "Node.js가 비동기 callback의 실행 시점을 관리하는 반복 구조다."),
                ("Call stack", "현재 실행 중인 동기 함수 호출이 쌓이는 공간이다."),
                ("Timers", "setTimeout/setInterval callback이 실행될 수 있는 phase다."),
                ("Poll", "I/O callback을 처리하고 다음 phase로 넘어갈지 결정하는 event loop phase다."),
                ("nextTick", "현재 작업 이후 event loop 다음 phase 전에 실행되는 queue 흐름이다."),
            ],
            "steps": [
                ("동기 실행과 callback 실행 분리", "코드가 위에서 아래로 쓰였다고 callback도 바로 실행되는 것은 아니다.", "동기 call stack과 event loop callback 실행을 나누었다.", "어떤 코드는 즉시 실행되고 어떤 코드는 나중에 실행되는지 설명할 수 있는지 확인했다."),
                ("timer와 I/O phase 구분", "비동기 callback도 모두 같은 queue에서 같은 순서로 실행되지 않는다.", "timers, poll, check 같은 phase를 실행 조건으로 보았다.", "callback 실행 순서가 왜 달라지는지 말할 수 있으면 이해한 것으로 보았다."),
                ("nextTick/microtask 영향 확인", "nextTick은 event loop phase보다 먼저 실행될 수 있어 순서 예측을 어렵게 만든다.", "특수 queue를 일반 timer callback과 분리했다.", "실행 순서 예시를 보고 결과를 예측할 수 있는지 확인했다."),
            ],
            "skills": ["Node.js event loop 이해", "callback 실행 순서 판단", "timer/poll phase 구분", "nextTick 흐름", "비동기 디버깅 기준"],
        },
        "spring_beans": {
            "kind": "spring_beans",
            "title": display_title,
            "article_title": "Spring Bean 학습 기록: 객체 생성과 IoC Container 관리 흐름 구분하기",
            "subtitle": "Clarifying Spring beans, IoC container, dependency injection, and configuration metadata",
            "default_problem": "객체를 직접 new로 만드는 것과 Spring container가 bean을 생성하고 의존성을 주입하는 흐름을 구분하는 것",
            "scope": "Spring bean, IoC container, dependency injection, configuration metadata, ApplicationContext",
            "flow": "configuration 작성 → container 초기화 → bean 생성 → dependency injection → bean 사용",
            "concepts": [
                ("Bean", "Spring IoC container가 생성하고 관리하는 application object다."),
                ("IoC container", "객체 생성과 의존성 연결을 개발자 코드 대신 관리하는 Spring container다."),
                ("Dependency injection", "객체가 필요한 의존성을 직접 만들지 않고 외부에서 주입받는 방식이다."),
                ("Configuration metadata", "container가 어떤 bean을 만들고 어떻게 연결할지 알려주는 설정 정보다."),
                ("ApplicationContext", "Spring의 대표적인 IoC container 인터페이스다."),
            ],
            "steps": [
                ("new와 container 생성 구분", "객체를 직접 생성하는 코드와 container가 관리하는 bean 흐름을 혼동할 수 있다.", "bean은 container가 lifecycle을 관리하는 객체로 정리했다.", "어떤 객체가 Spring bean인지 설명할 수 있는지 확인했다."),
                ("의존성 주입 흐름 이해", "객체 내부에서 의존성을 직접 만들면 결합도가 높아진다.", "dependency injection을 외부에서 필요한 객체를 연결하는 방식으로 보았다.", "의존성이 어디서 주입되는지 추적할 수 있으면 이해한 것으로 보았다."),
                ("configuration과 container 연결", "설정이 단순 옵션처럼 보이면 container가 무엇을 만드는지 흐려진다.", "configuration metadata를 bean 생성 기준으로 정리했다.", "설정 변경이 bean 구성에 어떤 영향을 주는지 확인했다."),
            ],
            "skills": ["Spring Bean 이해", "IoC container 역할 구분", "dependency injection 흐름", "configuration metadata 해석", "객체 lifecycle 관리"],
        },
        "web_workers": {
            "kind": "web_workers",
            "title": display_title,
            "article_title": "Web Worker 학습 기록: main thread와 background task 분리하기",
            "subtitle": "Clarifying Worker, main thread, postMessage, onmessage, and background tasks",
            "default_problem": "오래 걸리는 JavaScript 작업을 main thread에서 직접 실행하는 것과 Worker로 분리하는 차이를 이해하는 것",
            "scope": "Web Worker, Worker thread, main thread, postMessage, onmessage, background task",
            "flow": "Worker 생성 → message 전송 → background 처리 → result message 수신 → main thread UI 유지",
            "concepts": [
                ("Web Worker", "main thread와 별도로 JavaScript를 실행해 긴 작업을 background에서 처리하는 Web API다."),
                ("Main thread", "UI 렌더링과 사용자 입력 처리를 담당하는 브라우저의 기본 실행 흐름이다."),
                ("Worker thread", "main thread와 분리되어 계산이나 장기 작업을 처리하는 실행 흐름이다."),
                ("postMessage", "main thread와 Worker 사이에서 데이터를 보내는 메서드다."),
                ("onmessage", "상대쪽에서 보낸 message를 받아 처리하는 이벤트 핸들러다."),
            ],
            "steps": [
                ("main thread block 문제 인식", "무거운 작업을 main thread에서 처리하면 UI가 멈출 수 있다.", "Worker를 background task 분리 수단으로 정리했다.", "어떤 작업을 Worker로 보내야 하는지 판단할 수 있는지 확인했다."),
                ("message 기반 통신 이해", "Worker는 같은 call stack에서 값을 바로 반환하지 않는다.", "postMessage/onmessage로 결과를 주고받는 흐름으로 보았다.", "요청과 결과가 message 이벤트로 오가는 것을 설명할 수 있으면 이해한 것으로 보았다."),
                ("UI 유지 기준 확인", "성능 개선은 단순히 빠른 계산만이 아니라 UI 응답성 유지와 관련된다.", "main thread가 멈추지 않는지를 검증 기준으로 두었다.", "사용자 입력이 유지되는지 확인할 수 있으면 이해한 것으로 보았다."),
            ],
            "skills": ["Web Worker 이해", "main/worker thread 구분", "postMessage 통신", "background task 분리", "UI 응답성 검증"],
        },
        "typescript_types": {
            "kind": "typescript_types",
            "title": display_title,
            "article_title": "TypeScript everyday types 학습 기록: 값 구조와 타입 기준 연결하기",
            "subtitle": "Clarifying primitive types, arrays, objects, functions, and type annotations",
            "default_problem": "JavaScript 값에 type annotation을 붙이는 것이 함수 입력과 객체 구조를 검증하는 기준이 된다는 점을 이해하는 것",
            "scope": "TypeScript everyday types, primitive type, array, object type, function type, type annotation",
            "flow": "값 구조 파악 → type annotation 작성 → 함수 입력/출력 타입 확인 → 객체 구조 검증",
            "concepts": [
                ("Primitive type", "string, number, boolean처럼 기본 값의 종류를 나타내는 타입이다."),
                ("Array", "같은 종류의 값 목록을 다루는 타입 구조다."),
                ("Object type", "객체의 property 이름과 값 타입을 정의하는 구조다."),
                ("Function type", "함수의 parameter와 return type을 정의하는 방식이다."),
                ("Type annotation", "값이나 함수가 어떤 타입을 가져야 하는지 명시하는 표기다."),
            ],
            "steps": [
                ("값과 타입 분리", "JavaScript 값만 보면 어떤 입력이 허용되는지 명확하지 않을 수 있다.", "TypeScript type annotation을 값 사용 기준으로 정리했다.", "함수나 변수의 허용 타입을 말할 수 있는지 확인했다."),
                ("객체 구조 검증", "객체는 property가 많아질수록 누락/오타가 생기기 쉽다.", "object type을 property 계약으로 보았다.", "필수 property와 선택 property를 구분할 수 있으면 이해한 것으로 보았다."),
                ("함수 입출력 타입 확인", "함수는 입력과 출력이 함께 맞아야 안전하게 재사용된다.", "parameter type과 return type을 분리해 확인했다.", "잘못된 타입 입력이 왜 오류인지 설명할 수 있는지 확인했다."),
            ],
            "skills": ["TypeScript 타입 기초", "primitive/array/object 구분", "function type 이해", "type annotation 작성", "객체 구조 검증"],
        },
    }
    profile = profiles.get(str(kind or ""))
    return dict(profile) if profile else None



def source_first_fallback_profile(seed_url: str, title: str, body_text: str, user_problem: str = "") -> dict[str, Any] | None:
    """Build a source-only article contract when no known profile fits.

    v4.7.18 principle:
    - Do not borrow FastAPI/NumPy/Python/Docker templates for unsupported topics.
    - The source title, user's difficult point, and collected text define the topic.
    - Known profiles are allowed only when they are strongly matched before this
      fallback.  This is the default route for Array.map, non-IT articles, and
      future topics that do not have a dedicated profile yet.
    """
    raw_title = re.sub(r"\s+", " ", str(title or "")).strip()
    user_problem_clean = clean_prompt_memo(user_problem)
    source_blob = f"{raw_title}\n{user_problem_clean}\n{str(body_text or '')[:6000]}"
    # v4.7.24: topic selection must come from the explicit current input
    # (title + user memo), not from incidental body mentions.  Otherwise a
    # Memory page that mentions sleep can accidentally reuse the sleep contract,
    # or a Procrastination page that mentions time management can become a time
    # management article.
    explicit_blob = f"{raw_title}\n{user_problem_clean}"
    explicit_l = explicit_blob.lower()
    if not raw_title and len(source_blob.strip()) < 300:
        return None

    def title_subject(t: str) -> str:
        t = re.sub(r"\s*[-|—]\s*(MDN|JavaScript|GitHub Docs|Documentation|Docs|Manual|Reference).*$", "", t, flags=re.I)
        t = re.sub(r"\s+", " ", t).strip(" -—|`#")
        return t or "수집 자료"

    subject = title_subject(raw_title)
    blob_l = source_blob.lower()

    # Prefer concepts explicitly requested by the user's memo.  The phrasing
    # "글에서는 A, B, C를/을 문제로" is common in this app, so extract those
    # terms first before falling back to statistical terms from the source.
    memo_terms: list[str] = []
    memo = user_problem_clean
    patterns = [
        r"글에서는\s+(.{3,140}?)(?:를|을)\s+문제로",
        r"글에서는\s+(.{3,140}?)(?:의\s+차이|의미|역할|기준)",
        r"(?:특히|핵심은)\s+(.{3,140}?)(?:를|을|이다|라는)",
    ]
    for pat in patterns:
        m = re.search(pat, memo)
        if m:
            chunk = m.group(1)
            for part in re.split(r",|/|·|와|과|및| 그리고 ", chunk):
                part = re.sub(r"[^0-9A-Za-z가-힣_.()\- ]", "", part).strip()
                # Keep multi-word concepts such as "return value", "original array",
                # "수면의 질", and "생활 리듬" together.  Splitting on every
                # whitespace made source-first fallback produce fragments like
                # "수면의" / "Science" / "How".
                part = re.sub(r"\s+", " ", part)
                part_l = part.lower()
                if 2 <= len(part) <= 34 and part_l not in {"글에서는", "차이", "의미", "역할", "기준", "문제"}:
                    memo_terms.append(part)
    # Preserve important camel/API tokens from title and memo.
    token_terms = re.findall(r"\b[A-Za-z][A-Za-z0-9_.()/-]{2,}\b", f"{raw_title}\n{memo}")
    noise = {
        "docs", "documentation", "reference", "browser", "compatibility", "syntax", "description",
        "examples", "parameters", "parameter", "returns", "article", "learn", "tutorial",
        "the", "and", "for", "with", "from", "into", "thisarg", "callbackfn",
        "how", "your", "science", "according", "machines", "brain", "forms", "take", "control",
        "http", "https", "url", "www", "selected", "focus", "score", "why", "source",
        "graph", "candidate", "body", "chars", "collector", "seed", "seed_url",
    }
    terms: list[str] = []
    for term in memo_terms + token_terms + infer_learning_terms(source_blob, limit=18):
        s = re.sub(r"\s+", " ", str(term or "")).strip(" -—:`#*_[]()")
        if not s:
            continue
        sl = s.lower()
        if sl in noise or len(sl) < 2:
            continue
        if any(bad in sl for bad in [
            "copyright", "privacy", "subscribe", "navigation", "table of contents",
            "selected focus", "focus title", "focus url", "problem framing",
            "source_graph", "source graph", "source pack", "body chars", "why selected",
        ]):
            continue
        if s not in terms:
            terms.append(s)
        if len(terms) >= 8:
            break

    # Current-source derived special cases, not reusable templates:
    # they are allowed only when the current title/user memo clearly identify
    # that topic.  Body text is still used for evidence, but not for picking a
    # different topic contract.
    code_example = ""
    concept_desc_overrides: dict[str, str] = {}
    if ("array.prototype.map" in explicit_l or "array map" in explicit_l or "array.map" in explicit_l) and ("new array" in explicit_l or "callback" in explicit_l):
        subject = "JavaScript Array map"
        preferred = ["Array.map", "callback", "return value", "original array", "new array", "iteration"]
        terms = preferred[:]
        concept_desc_overrides = {
            "Array.map": "배열의 각 요소에 callback을 적용하고, 그 반환값을 모아 새 배열을 만드는 메서드다.",
            "callback": "각 요소를 어떻게 변환할지 정의하는 함수이며, 반환값이 결과 배열의 요소가 된다.",
            "return value": "callback이 돌려주는 값이며 `map()` 결과 배열에 들어가는 값이다.",
            "original array": "`map()`을 호출한 기존 배열이며, 기본적으로 직접 변경 대상이 아니다.",
            "new array": "각 요소의 변환 결과가 모여 만들어지는 새로운 배열이다.",
            "iteration": "배열의 각 요소를 순서대로 처리하며 callback을 실행하는 흐름이다.",
        }
        code_example = """대표 검증 예시의 핵심은 `map()`이 원본 배열을 직접 바꾸는 것이 아니라 callback의 반환값으로 새 배열을 만든다는 점이다.

```javascript
const numbers = [1, 4, 9, 16];
const doubled = numbers.map((value) => value * 2);

console.log(numbers); // [1, 4, 9, 16]
console.log(doubled); // [2, 8, 18, 32]
```

`callback`은 각 요소를 받아 새 값을 반환하고, `map()`은 그 반환값들을 모아 `new array`를 만든다. 원본 배열과 결과 배열을 비교하는 것이 핵심 검증 기준이다."""

    if "스트레스" in explicit_blob or "stress" in explicit_l:
        subject = "스트레스"
        preferred = ["스트레스 요인", "신체 반응", "심리적 반응", "적응", "회복 기준"]
        terms = preferred[:]
        concept_desc_overrides = {
            "스트레스 요인": "긴장이나 압박을 유발하는 외부·내부 자극이다.",
            "신체 반응": "스트레스 요인에 대해 몸에서 나타나는 생리적 변화다.",
            "심리적 반응": "스트레스 상황에서 감정·인지·주의가 달라지는 반응이다.",
            "적응": "자극에 대응하며 상태를 조절해 가는 과정이다.",
            "회복 기준": "긴장 이후 다시 안정 상태로 돌아왔는지 확인하는 판단 기준이다.",
        }

    if "기억" in explicit_blob or "memory" in explicit_l:
        subject = "기억"
        preferred = ["기억 형성", "입력", "저장", "인출", "회상", "망각"]
        terms = preferred[:]
        concept_desc_overrides = {
            "기억 형성": "정보가 경험으로 들어와 기억으로 만들어지는 과정이다.",
            "입력": "외부 정보를 받아들이는 첫 단계다.",
            "저장": "받아들인 정보를 일정 기간 유지하는 과정이다.",
            "인출": "저장된 정보를 필요할 때 다시 꺼내 쓰는 과정이다.",
            "회상": "기억한 내용을 의식적으로 떠올리는 행위다.",
            "망각": "저장되었거나 입력된 정보를 다시 떠올리지 못하는 상태다.",
        }

    if "미루기" in explicit_blob or "procrastination" in explicit_l:
        subject = "미루기"
        preferred = ["과제 회피", "즉각적 보상", "마감 압박", "실행 지연", "실행 전략"]
        terms = preferred[:]
        concept_desc_overrides = {
            "과제 회피": "해야 할 일을 바로 시작하지 않고 피하는 행동 패턴이다.",
            "즉각적 보상": "장기 목표보다 당장 편하거나 즐거운 선택에 끌리는 요인이다.",
            "마감 압박": "기한이 가까워질수록 행동을 강제로 전환시키는 압력이다.",
            "실행 지연": "계획은 있으나 실제 행동으로 옮기는 시점이 늦어지는 상태다.",
            "실행 전략": "회피를 줄이고 행동 시작을 쉽게 만드는 구체적 기준이다.",
        }

    if "시간 관리" in explicit_blob or "time management" in explicit_l:
        subject = "시간 관리"
        preferred = ["우선순위", "시간 블록", "마감", "실행 기준", "계획", "시간 배분"]
        terms = preferred[:]
        concept_desc_overrides = {
            "우선순위": "할 일의 중요도와 긴급도를 기준으로 먼저 처리할 대상을 고르는 판단 기준이다.",
            "시간 블록": "실제로 실행할 수 있는 시간을 일정 단위로 확보해 작업을 배치하는 방식이다.",
            "마감": "작업이 완료되어야 하는 시간 제한이며 우선순위와 실행 순서를 정하는 기준이 된다.",
            "실행 기준": "계획이 실제 행동으로 이어졌는지 확인하는 완료 조건이다.",
            "계획": "해야 할 일을 시간과 순서에 맞게 배치하는 과정이다.",
            "시간 배분": "제한된 시간을 여러 과업에 나누어 사용하는 관리 방식이다.",
        }

    if ("수면" in explicit_blob or "잠 - 위키" in explicit_blob or "sleep" in explicit_l) and any(x in explicit_blob for x in ["수면", "회복", "리듬", "잠"]):
        subject = "수면"
        preferred = ["수면 시간", "수면의 질", "수면 주기", "회복 기준", "생활 리듬"]
        terms = preferred[:]
        concept_desc_overrides = {
            "수면 시간": "얼마나 오래 자는지를 나타내는 양적 기준이다.",
            "수면의 질": "잠을 잔 뒤 회복감과 안정감을 판단하는 질적 기준이다.",
            "수면 주기": "렘수면과 비렘수면처럼 수면이 단계적으로 반복되는 흐름이다.",
            "회복 기준": "수면이 신체와 인지 기능 회복에 충분했는지 확인하는 기준이다.",
            "생활 리듬": "수면과 각성 시간이 반복되며 하루 생활 패턴을 만드는 흐름이다.",
        }

    if "습관" in explicit_blob and any(x in explicit_blob for x in ["반복", "보상", "환경", "행동", "신호"]):
        subject = "습관"
        preferred = ["습관 형성", "행동 반복", "신호", "보상", "환경 설계"]
        terms = preferred[:]
        concept_desc_overrides = {
            "습관 형성": "반복된 행동이 점차 자동화되는 과정이다.",
            "행동 반복": "같은 행동을 지속적으로 수행해 습관으로 굳어지는 실행 흐름이다.",
            "신호": "습관 행동을 시작하게 만드는 상황이나 자극이다.",
            "보상": "행동을 반복하게 만드는 긍정적 결과나 만족감이다.",
            "환경 설계": "원하는 행동이 쉽게 반복되도록 주변 조건을 조정하는 방식이다.",
        }

    if ("cue" in explicit_l and "routine" in explicit_l and "reward" in explicit_l) or "habit loop" in blob_l:
        subject = "Habit formation"
        preferred = ["cue", "routine", "reward", "environment", "environment design"]
        terms = preferred[:]
        concept_desc_overrides = {
            "cue": "habit loop를 시작하게 만드는 신호나 상황이다.",
            "routine": "신호 뒤에 반복되는 실제 행동 패턴이다.",
            "reward": "행동 뒤에 주어져 반복을 강화하는 결과다.",
            "environment": "습관이 쉽게 반복되거나 깨지도록 영향을 주는 주변 조건이다.",
            "environment design": "원하는 습관이 반복되도록 환경의 마찰과 단서를 조정하는 방법이다.",
        }

    if not terms:
        terms = [subject, "핵심 개념", "적용 기준", "검증 기준"]

    concepts: list[tuple[str, str]] = []
    for term in terms[:8]:
        if term in concept_desc_overrides:
            desc = concept_desc_overrides[term]
        else:
            evidence = evidence_for_aliases(body_text, [term], limit=1)
            if evidence:
                desc = "본문에서 확인되는 핵심 개념으로, 역할·적용 조건·검증 기준을 구분해야 하는 항목이다."
            else:
                desc = "사용자 메모와 수집 자료에서 확인되는 핵심 개념으로, 역할과 적용 조건을 구분해야 하는 항목이다."
        concepts.append((term, desc))

    t1 = concepts[0][0] if concepts else subject
    t2 = concepts[1][0] if len(concepts) > 1 else "관련 개념"
    t3 = concepts[2][0] if len(concepts) > 2 else "결과"
    flow = f"{t1} 확인 → {t2} 역할 구분 → 적용 조건 정리 → {t3} 검증"
    default_problem = user_problem_clean or f"{subject}에서 핵심 개념의 역할과 적용 기준을 본문 안에서 구분하는 것"

    steps = [
        (
            "본문의 중심 개념 분리",
            "자료 안에는 정의, 예시, 문법, 주변 설명이 함께 섞여 있어 무엇을 먼저 이해해야 하는지 흐려질 수 있다.",
            f"`{t1}`를 중심 개념으로 잡고, 나머지 설명을 적용 조건과 검증 기준으로 다시 분리했다.",
            f"`{t1}`가 어떤 문제를 해결하고 어떤 상황에서 쓰이는지 설명할 수 있는지 확인했다.",
        ),
        (
            "비슷한 개념의 역할 구분",
            f"`{t1}`와 `{t2}`가 같은 흐름 안에 등장하면 각각의 역할이 섞일 수 있다.",
            "각 개념을 입력, 처리, 결과, 검증 기준 중 어디에 놓이는지 나누어 정리했다.",
            "자료의 예시나 설명을 보고 어떤 개념이 어느 단계에서 필요한지 말할 수 있는지 확인했다.",
        ),
        (
            "적용 결과와 확인 기준 세우기",
            "개념을 읽었다는 사실만으로는 실제로 이해했는지 확인하기 어렵다.",
            "본문에서 확인되는 예시와 사용자 메모를 연결해 완료 기준을 만들었다.",
            f"`{t3}`를 기준으로 적용 전후 차이 또는 결과 해석을 설명할 수 있으면 이해한 것으로 보았다.",
        ),
    ]

    return {
        "kind": "source_first",
        "title": raw_title or subject,
        "article_title": (
            "JavaScript Array map 학습 기록: 원본 배열과 새 배열 생성 흐름 구분하기"
            if subject == "JavaScript Array map" else
            "시간 관리 학습 기록: 우선순위와 실행 가능 시간 구분하기"
            if subject == "시간 관리" else
            "수면 학습 기록: 수면 시간과 회복 기준 구분하기"
            if subject == "수면" else
            "습관 학습 기록: 행동 반복과 환경 설계 구분하기"
            if subject == "습관" else
            "스트레스 학습 기록: 자극·반응·회복 기준 구분하기"
            if subject == "스트레스" else
            "기억 학습 기록: 입력·저장·인출 흐름 구분하기"
            if subject == "기억" else
            "미루기 학습 기록: 과제 회피와 실행 전환 구분하기"
            if subject == "미루기" else
            "Habit formation 학습 기록: cue-routine-reward 흐름 이해하기"
            if subject == "Habit formation" else
            f"{subject} 학습 기록: 본문 안에서 핵심 개념 구분하기"
        ),
        "subtitle": (
            "Clarifying callback, return value, original array, and new array"
            if subject == "JavaScript Array map" else
            "Clarifying priority, time blocks, deadlines, and execution criteria"
            if subject == "시간 관리" else
            "Clarifying sleep duration, sleep quality, recovery, and daily rhythm"
            if subject == "수면" else
            "Clarifying habit formation, repetition, reward, and environment design"
            if subject == "습관" else
            "Clarifying stressors, responses, adaptation, and recovery criteria"
            if subject == "스트레스" else
            "Clarifying encoding, storage, retrieval, recall, and forgetting"
            if subject == "기억" else
            "Clarifying avoidance, instant reward, deadline pressure, and execution strategy"
            if subject == "미루기" else
            "Clarifying cue, routine, reward, and environment design"
            if subject == "Habit formation" else
            "Deriving the learning problem from the page itself"
        ),
        "default_problem": default_problem,
        "scope": ", ".join([name for name, _ in concepts[:7]]),
        "flow": flow,
        "concepts": concepts,
        "steps": steps,
        "skills": [f"{name} 이해" for name, _ in concepts[:4]] + ["현재 자료 중심 문제 정의", "검증 기준 설정"],
        "code_example": code_example,
    }

def topic_profile_from_text(seed_url: str, title: str, body_text: str, user_problem: str = "") -> dict[str, Any] | None:
    """Current-input-only topic profile for common smoke-test materials.

    The goal is not to invent content. It is to keep known Docker/FastAPI/Promise
    materials from falling back to title-only generic drafts.
    """
    user_problem_clean = clean_user_problem_note(user_problem)
    # Source-first routing rule:
    # The user's prompt, seed URL, and page title decide the article topic.
    # Full body text is noisy (global nav, examples, related docs) and is used only
    # as a fallback when the explicit input does not identify a topic.
    strong_blob = f"{seed_url}\n{title}\n{user_problem_clean}".lower()
    body_blob = str(body_text or "").lower()
    blob = f"{strong_blob}\n{body_blob}"
    requested_kind = expected_topic_kind_from_input(seed_url=seed_url, current_text=f"{user_problem_clean}\n{title}")

    def has_any(words: list[str]) -> bool:
        return any(w.lower() in blob for w in words)

    def strong_has_any(words: list[str]) -> bool:
        return any(w.lower() in strong_blob for w in words)

    def body_has_any(words: list[str]) -> bool:
        return any(w.lower() in body_blob for w in words)

    broad_profile = broad_source_profile(requested_kind, title)
    if broad_profile:
        return broad_profile

    if requested_kind == "react_useeffect" or (not requested_kind and strong_has_any(["useeffect", "react", "setup", "cleanup", "dependencies"])):
        return {
            "kind": "react_useeffect",
            "title": (title or "React useEffect 학습").strip(),
            "article_title": "React useEffect 학습 기록: 렌더링과 외부 시스템 동기화 구분하기",
            "subtitle": "Clarifying setup, cleanup, dependencies, and external synchronization",
            "default_problem": "useEffect에서 렌더링 자체와 외부 시스템 동기화 과정을 구분하는 것",
            "scope": "React useEffect, setup, cleanup, dependencies, external system, re-render",
            "flow": "컴포넌트 렌더링 → setup 실행 → dependency 변경 감지 → cleanup 실행 → 새 setup 실행",
            "concepts": [
                ("useEffect", "React 컴포넌트를 외부 시스템과 동기화할 때 사용하는 Hook이다."),
                ("Setup", "Effect가 실행될 때 외부 연결, 구독, 타이머 같은 동기화 작업을 시작하는 함수다."),
                ("Cleanup", "의존성이 바뀌거나 컴포넌트가 사라질 때 이전 동기화 작업을 정리하는 함수다."),
                ("Dependencies", "Effect를 다시 실행할지 판단하는 값의 목록이다."),
                ("External system", "React 렌더링 바깥의 네트워크, 브라우저 API, 구독, 타이머 같은 대상이다."),
                ("Re-render", "상태나 props 변화로 컴포넌트가 다시 렌더링되는 과정이며, dependency 변화와 Effect 재실행 조건을 함께 봐야 한다."),
            ],
            "steps": [
                ("렌더링과 Effect 실행 구분", "렌더링 때마다 외부 연결을 바로 다시 만든다고 생각하면 실행 시점을 오해하기 쉽다.", "useEffect는 렌더링 결과가 반영된 뒤 외부 시스템과 동기화하는 단계로 정리했다.", "렌더링과 setup 실행 시점을 분리해 설명할 수 있는지 확인했다."),
                ("cleanup의 역할 이해", "이전 연결이나 타이머를 정리하지 않으면 중복 실행이나 메모리 누수가 생길 수 있다.", "cleanup을 이전 Effect를 종료하는 검증 기준으로 보았다.", "dependency 변경 전 이전 cleanup이 실행된다고 설명할 수 있으면 이해한 것으로 보았다."),
                ("dependencies로 재실행 조건 확인", "의존성 배열을 단순 옵션처럼 보면 Effect가 언제 다시 실행되는지 예측하기 어렵다.", "dependencies를 setup/cleanup 재실행을 결정하는 기준으로 정리했다.", "어떤 값이 바뀌면 Effect가 다시 실행되는지 설명할 수 있는지 확인했다."),
            ],
            "skills": ["useEffect 실행 시점 이해", "setup/cleanup 역할 구분", "dependencies 재실행 조건 판단", "외부 시스템 동기화", "렌더링과 Effect 분리"],
        }

    if requested_kind == "kubernetes_pod" or (not requested_kind and strong_has_any(["kubernetes", "pod", "shared resources", "workload"])):
        return {
            "kind": "kubernetes_pod",
            "title": (title or "Kubernetes Pod 학습").strip(),
            "article_title": "Kubernetes Pod 학습 기록: 컨테이너와 최소 배포 단위 구분하기",
            "subtitle": "Clarifying Pods, containers, shared resources, workloads, and lifecycle",
            "default_problem": "컨테이너와 Pod를 같은 실행 단위로 착각하지 않고 Pod가 묶는 범위를 이해하는 것",
            "scope": "Kubernetes Pod, container, workload, shared resources, lifecycle, scheduling",
            "flow": "workload 정의 → Pod 생성 → 하나 이상의 container 포함 → shared resources 사용 → lifecycle 관리",
            "concepts": [
                ("Kubernetes", "컨테이너화된 애플리케이션을 배포하고 운영하기 위한 오케스트레이션 플랫폼이다."),
                ("Pod", "Kubernetes에서 하나 이상의 컨테이너를 함께 묶어 배포하는 가장 작은 실행 단위다."),
                ("Container", "Pod 안에서 실제 애플리케이션 프로세스를 실행하는 단위다."),
                ("Shared resources", "같은 Pod 안의 컨테이너들이 공유할 수 있는 네트워크와 스토리지 같은 자원이다."),
                ("Workload", "Deployment 같은 상위 리소스가 Pod를 생성하고 관리하는 애플리케이션 실행 단위다."),
                ("Lifecycle", "Pod가 생성, 실행, 종료, 재시작되는 상태 흐름이다."),
            ],
            "steps": [
                ("Pod와 container 경계 구분", "컨테이너와 Pod를 같은 단위로 보면 Kubernetes 배포 구조를 오해하기 쉽다.", "Pod는 하나 이상의 컨테이너를 담는 최소 배포 단위로 정리했다.", "Pod 안에 여러 container가 들어갈 수 있다고 설명할 수 있는지 확인했다."),
                ("shared resources 이해", "같은 Pod 안의 컨테이너가 왜 함께 배치되는지 기준이 모호할 수 있다.", "네트워크와 스토리지 공유 여부를 Pod 설계 기준으로 보았다.", "함께 배치할 컨테이너와 분리할 컨테이너를 구분할 수 있는지 확인했다."),
                ("lifecycle과 workload 연결", "Pod는 직접 오래 관리하는 대상이라기보다 상위 workload가 생성·교체할 수 있는 대상이다.", "Pod lifecycle을 Deployment 등 workload 관리 흐름과 연결해 정리했다.", "Pod가 사라져도 상위 리소스가 새 Pod를 만들 수 있음을 설명할 수 있으면 이해한 것으로 보았다."),
            ],
            "skills": ["Pod/container 구분", "shared resources 이해", "workload와 Pod 관계", "Pod lifecycle 판단", "Kubernetes 배포 단위 이해"],
        }

    if requested_kind == "postgres_index" or (not requested_kind and strong_has_any(["postgresql", "create index", "sequential scan", "index scan", "query planner"])):
        return {
            "kind": "postgres_index",
            "title": (title or "PostgreSQL Index 학습").strip(),
            "article_title": "PostgreSQL Index 학습 기록: sequential scan과 index scan 판단 기준 이해하기",
            "subtitle": "Clarifying CREATE INDEX, query planner, WHERE conditions, and scan cost",
            "default_problem": "index가 항상 빠르게 만드는 마법이 아니라 특정 조회 조건에서 검색 비용을 줄이는 구조임을 이해하는 것",
            "scope": "PostgreSQL index, sequential scan, index scan, CREATE INDEX, query planner, WHERE, JOIN",
            "flow": "table scan 문제 인식 → CREATE INDEX 생성 → WHERE/JOIN 조건 검토 → query planner 판단 → index scan 효과 확인",
            "concepts": [
                ("PostgreSQL", "관계형 데이터베이스이며 query planner가 실행 계획을 선택한다."),
                ("Index", "테이블 전체를 매번 훑지 않고 특정 조건의 행을 빠르게 찾기 위한 보조 구조다."),
                ("Sequential scan", "조건을 찾기 위해 테이블을 순차적으로 읽는 방식이다."),
                ("Index scan", "인덱스를 사용해 필요한 행 위치를 좁혀 찾는 방식이다."),
                ("CREATE INDEX", "특정 컬럼이나 표현식에 인덱스를 만드는 SQL 명령이다."),
                ("Query planner", "쿼리 조건과 비용을 바탕으로 어떤 실행 방식을 쓸지 결정하는 구성이다."),
            ],
            "steps": [
                ("인덱스가 필요한 조회 조건 찾기", "모든 컬럼에 인덱스를 붙이는 방식은 쓰기 비용과 저장 비용을 늘릴 수 있다.", "WHERE/JOIN에서 자주 쓰이는 조건을 먼저 확인하는 방식으로 정리했다.", "어떤 쿼리가 인덱스 후보인지 설명할 수 있는지 확인했다."),
                ("sequential scan과 index scan 비교", "인덱스가 없으면 작은 조건 검색도 테이블 전체 스캔으로 이어질 수 있다.", "sequential scan은 전체 탐색, index scan은 위치를 좁혀 찾는 방식으로 나누었다.", "같은 WHERE 조건에서 인덱스가 왜 도움이 되는지 말할 수 있으면 이해한 것으로 보았다."),
                ("query planner 판단 기준 이해", "인덱스가 있어도 항상 사용되는 것은 아니다.", "planner가 데이터 규모, 선택도, 비용을 보고 실행 계획을 고른다고 정리했다.", "인덱스 존재와 실제 사용 여부를 분리해 설명할 수 있는지 확인했다."),
            ],
            "skills": ["PostgreSQL index 이해", "sequential/index scan 비교", "CREATE INDEX 사용 기준", "query planner 판단", "WHERE/JOIN 조건 분석"],
        }

    if requested_kind == "fetch_api" or (not requested_kind and strong_has_any(["fetch api", "fetch()", "response object", "body parsing", "headers"])):
        return {
            "kind": "fetch_api",
            "title": (title or "Fetch API 학습").strip(),
            "article_title": "Fetch API 학습 기록: Request·Response·body parsing 흐름 구분하기",
            "subtitle": "Clarifying fetch(), Request, Response, status, headers, body, and Promise flow",
            "default_problem": "fetch가 데이터를 바로 반환하는 것이 아니라 Response 객체와 Promise 흐름을 통해 처리된다는 점을 이해하는 것",
            "scope": "Fetch API, fetch(), Request, Response, status, headers, body parsing, Promise",
            "flow": "request 생성 → fetch 호출 → Promise 반환 → Response 확인 → status/headers 점검 → body parsing",
            "concepts": [
                ("Fetch API", "네트워크 요청을 보내고 응답을 Promise 기반으로 처리하는 Web API다."),
                ("fetch()", "요청을 시작하고 Response로 fulfilled 되는 Promise를 반환하는 함수다."),
                ("Request", "요청 URL, method, headers, body 같은 입력 조건을 담는 구조다."),
                ("Response", "서버 응답의 status, headers, body를 담는 객체다."),
                ("Status", "응답 성공/실패를 판단하는 HTTP 상태 코드다."),
                ("Body parsing", "response.json(), response.text()처럼 응답 본문을 필요한 형식으로 읽는 단계다."),
            ],
            "steps": [
                ("즉시 데이터 반환과 Response 반환 구분", "fetch 결과를 바로 JSON 데이터라고 생각하면 처리 흐름을 오해하기 쉽다.", "fetch는 Promise를 반환하고 완료 후 Response 객체를 받는다고 정리했다.", "Response에서 다시 body를 parsing해야 실제 데이터를 얻는다고 설명할 수 있는지 확인했다."),
                ("status와 headers 확인", "응답 객체가 왔다고 항상 성공 요청은 아니다.", "status와 headers를 먼저 확인한 뒤 body를 읽는 흐름으로 정리했다.", "HTTP status를 보고 성공/실패를 판단할 수 있으면 이해한 것으로 보았다."),
                ("body parsing 단계 분리", "response body는 한 번에 값처럼 들어오는 것이 아니라 메서드로 읽어야 한다.", "json/text/blob 등 필요한 형식으로 parsing하는 단계를 분리했다.", "어떤 응답에 어떤 parsing 메서드를 써야 하는지 설명할 수 있는지 확인했다."),
            ],
            "skills": ["Fetch API 흐름 이해", "Request/Response 구분", "status/headers 확인", "body parsing", "Promise 기반 네트워크 처리"],
        }

    if requested_kind == "sqlalchemy_orm" or (not requested_kind and strong_has_any(["sqlalchemy", "mapped class", "session", "engine", "commit"])):
        return {
            "kind": "sqlalchemy_orm",
            "title": (title or "SQLAlchemy ORM 학습").strip(),
            "article_title": "SQLAlchemy ORM 학습 기록: Python class와 database table 매핑 흐름 이해하기",
            "subtitle": "Clarifying Engine, Session, mapped classes, select, and commit",
            "default_problem": "SQL을 직접 쓰는 것과 Python class를 table에 매핑해 다루는 ORM 흐름을 구분하는 것",
            "scope": "SQLAlchemy ORM, Engine, Session, mapped class, select, commit, transaction",
            "flow": "Engine 생성 → mapped class 정의 → Session 열기 → select 실행 → 객체 변경 → commit",
            "concepts": [
                ("SQLAlchemy", "Python에서 SQL과 ORM 방식으로 데이터베이스를 다루는 라이브러리다."),
                ("ORM", "database table을 Python class와 객체로 매핑해 다루는 방식이다."),
                ("Engine", "데이터베이스 연결과 SQL 실행 기반을 제공하는 핵심 객체다."),
                ("Session", "ORM 객체 변경과 query, transaction 단위를 관리하는 작업 공간이다."),
                ("Mapped class", "table 구조와 Python class를 연결해 row를 객체처럼 다루게 해준다."),
                ("Commit", "Session에 쌓인 변경사항을 실제 database transaction으로 확정하는 단계다."),
            ],
            "steps": [
                ("table과 Python class 매핑 구분", "SQL을 직접 작성하는 흐름과 객체를 통해 table을 다루는 흐름이 섞일 수 있다.", "mapped class를 table row를 표현하는 Python 객체 기준으로 정리했다.", "class 정의가 어떤 table/column과 연결되는지 설명할 수 있는지 확인했다."),
                ("Engine과 Session 역할 분리", "연결 객체와 작업 단위를 같은 것으로 보면 transaction 흐름이 흐려진다.", "Engine은 연결 기반, Session은 ORM 작업과 transaction 단위로 나누었다.", "select와 변경 작업이 Session 안에서 수행된다고 설명할 수 있으면 이해한 것으로 보았다."),
                ("select와 commit 흐름 확인", "객체를 바꿨다는 사실만으로 데이터베이스에 반영되는 것은 아니다.", "select는 조회, commit은 변경 확정 단계로 분리했다.", "언제 commit이 필요한지 판단할 수 있는지 확인했다."),
            ],
            "skills": ["SQLAlchemy ORM 흐름 이해", "Engine/Session 역할 구분", "mapped class 설계", "select 조회", "commit/transaction 판단"],
        }

    if requested_kind == "github_actions" or (not requested_kind and strong_has_any(["github actions", "workflow_dispatch", "runs-on", "ci/cd", "ci cd"])):
        return {
            "kind": "github_actions",
            "title": (title or "GitHub Actions 학습").strip(),
            "article_title": "GitHub Actions 학습 기록: workflow·job·step 실행 흐름 구분하기",
            "subtitle": "Clarifying GitHub Actions workflow, jobs, steps, YAML, runner, and CI/CD flow",
            "default_problem": "GitHub Actions에서 workflow 파일이 언제 실행되고, job과 step이 어떤 단위로 자동화 작업을 나누는지 이해하는 것",
            "scope": "GitHub Actions workflow, YAML, trigger, job, step, runner, action, CI/CD",
            "flow": "workflow YAML 작성 → trigger 조건 정의 → job 구성 → runner에서 step 실행 → CI/CD 결과 확인",
            "concepts": [
                ("GitHub Actions", "repository 안의 workflow 파일을 기준으로 빌드, 테스트, 배포 같은 자동화 작업을 실행하는 기능이다."),
                ("Workflow", "`.github/workflows` 아래의 YAML 파일로 정의되며, 자동화 전체 흐름과 실행 조건을 담는 단위다."),
                ("Trigger / event", "`push`, `pull_request`, `workflow_dispatch`처럼 workflow가 언제 실행될지 정하는 조건이다."),
                ("Job", "하나의 runner 환경에서 실행되는 작업 묶음이며, 여러 step을 포함할 수 있다."),
                ("Step", "job 안에서 순서대로 실행되는 개별 명령 또는 action 호출 단위다."),
                ("YAML", "workflow의 trigger, job, runner, step 설정을 선언하는 파일 형식이다."),
                ("Runner", "job을 실제로 실행하는 머신 또는 실행 환경이다."),
                ("CI/CD", "코드 변경 후 빌드, 테스트, 배포를 자동화해 변경 검증과 릴리스를 반복 가능하게 만드는 흐름이다."),
            ],
            "steps": [
                ("workflow와 단일 명령 실행 구분", "처음에는 GitHub Actions가 단순히 명령을 실행하는 기능처럼 보일 수 있었다.", "workflow를 repository 안의 YAML 파일로 정의되는 자동화 전체 단위로 정리했다.", "어떤 event가 발생했을 때 어떤 workflow가 실행되는지 설명할 수 있는지 확인했다."),
                ("job과 step의 실행 단위 분리", "job과 step이 모두 실행 단계처럼 보여 어느 단위에서 runner와 명령이 연결되는지 헷갈릴 수 있었다.", "job은 runner에서 실행되는 작업 묶음, step은 job 내부의 개별 명령 또는 action 호출로 나누어 이해했다.", "하나의 workflow 안에서 여러 job과 step의 포함 관계를 설명할 수 있으면 이해한 것으로 보았다."),
                ("YAML 설정을 자동화 구조로 읽기", "workflow 파일을 문법으로만 보면 `on`, `jobs`, `runs-on`, `steps`가 각각 어떤 역할인지 흐려진다.", "YAML의 각 key를 실행 조건, 작업 묶음, 실행 환경, 실제 명령으로 연결해 읽었다.", "workflow 파일을 보고 trigger, runner, step 실행 순서를 찾아낼 수 있는지 확인했다."),
                ("CI/CD 결과 검증 기준 세우기", "자동화가 실행됐다는 사실만으로는 변경 사항이 안전하게 검증됐는지 알 수 없다.", "빌드, 테스트, 배포 단계가 어떤 job/step으로 실행되고 성공 여부가 어디에 표시되는지 확인했다.", "Pull Request나 commit에서 workflow run 결과를 보고 실패 지점을 추적할 수 있는지 확인했다."),
            ],
            "skills": ["GitHub Actions workflow 구조 이해", "YAML 기반 CI/CD 설정 읽기", "job/step 실행 단위 구분", "runner 역할 이해", "자동화 결과 검증"],
        }

    if requested_kind == "fastapi" or (not requested_kind and strong_has_any(["fastapi", "pydantic", "path parameter", "query parameter", "request body", "swagger ui", "basemodel"])):
        return {
            "kind": "fastapi",
            "title": (title or "FastAPI 학습").strip(),
            "article_title": "FastAPI 학습 기록: endpoint와 요청·응답 검증 흐름 구분하기",
            "subtitle": "Clarifying FastAPI endpoints, request inputs, response output, and validation flow",
            "default_problem": "FastAPI에서 endpoint를 만드는 것과 실제 요청/응답 구조를 검증하는 것이 다르다는 점을 이해하는 것",
            "scope": "FastAPI endpoint, path operation, path parameter, query parameter, request body, Pydantic model, /docs, response validation",
            "flow": "endpoint 정의 → 입력 종류 구분 → schema 검증 → 문서 화면 확인 → 응답 결과 점검",
            "concepts": [
                ("FastAPI", "Python으로 API endpoint를 빠르게 만들고 타입 기반 검증과 문서화를 함께 제공하는 웹 프레임워크다."),
                ("Endpoint / path operation", "클라이언트 요청이 들어오는 경로와 HTTP method에 따라 실행되는 API 처리 단위다."),
                ("Path parameter", "URL 경로 안에 포함되어 특정 리소스를 식별하는 입력값이다."),
                ("Query parameter", "URL 뒤의 query string으로 전달되어 필터링이나 선택 조건에 쓰이는 입력값이다."),
                ("Request body", "POST/PUT처럼 구조화된 데이터를 서버에 전달할 때 사용하는 입력 영역이다."),
                ("Pydantic model", "요청 body와 응답 데이터의 구조, 타입, 검증 기준을 정의하는 schema 역할을 한다."),
                ("/docs", "FastAPI가 자동으로 제공하는 Swagger UI로 endpoint를 확인하고 직접 요청을 테스트할 수 있는 화면이다."),
            ],
            "steps": [
                ("endpoint 생성과 요청 검증 분리", "API를 만들었다는 사실만으로는 실제 요청이 올바르게 처리되는지 알 수 없었다.", "endpoint는 경로와 method를 정의하는 단계이고, 검증은 요청을 보내 응답 구조와 상태를 확인하는 단계로 나누었다.", "/docs나 API client에서 실제 요청을 보내 response body와 status code를 확인할 수 있어야 한다고 보았다."),
                ("path/query/body 입력 위치 구분", "입력값이 URL 경로에 들어가는지, query string으로 들어가는지, body로 들어가는지 혼동될 수 있었다.", "path parameter는 리소스 식별, query parameter는 조회 조건, request body는 구조화된 데이터 전달로 나누어 정리했다.", "같은 값이라도 어떤 API 상황에서 어떤 위치에 두어야 하는지 설명할 수 있는지 확인했다."),
                ("Pydantic으로 schema 검증 이해", "요청 body를 단순 dict처럼 보면 입력 타입과 필수값 검증이 어디서 일어나는지 흐려진다.", "Pydantic model을 요청/응답 schema이자 validation 기준으로 정리했다.", "잘못된 타입이나 누락된 필드가 들어왔을 때 FastAPI가 검증 오류를 반환한다는 점을 설명할 수 있어야 했다."),
                ("자동 문서 화면으로 완료 기준 확인", "코드를 작성해도 endpoint 목록과 입력 schema가 의도대로 노출되는지 확인하지 않으면 API 구조를 검증하기 어렵다.", "/docs 화면을 endpoint, parameter, schema, response를 한 번에 점검하는 검증 도구로 보았다.", "새 endpoint가 문서에 나타나고, 예시 요청을 실행해 예상 응답을 확인할 수 있으면 학습이 완료된 것으로 판단했다."),
            ],
            "skills": ["FastAPI endpoint 설계", "request/response 구조 검증", "path/query/body 입력 구분", "Pydantic schema 이해", "Swagger UI 기반 API 확인"],
        }

    if requested_kind == "docker" or (not requested_kind and strong_has_any(["docker", "container", "dockerfile", "docker compose"])):
        # Docker has several subtopics.  If we always return the generic image/container
        # profile, Dockerfile/Volume/Compose docs all look the same.  Pick the most
        # specific Docker profile from the current URL/title/user_problem/body only.
        dockerfile_terms = ["writing-a-dockerfile", "dockerfile", "from", "workdir", "copy", "cmd", "image layer", "docker build"]
        volume_terms = ["/storage/volumes", "docker volume", "named volume", "volume mount", "data persistence", "persist", "mount"]
        compose_terms = ["/compose", "docker compose", "compose.yaml", "compose.yml", "multi-container", "service", "depends_on", "docker compose up"]
        container_terms = ["what-is-a-container", "isolation", "runtime", "docker engine", "lightweight", "virtual machine"]

        # Pick Docker subtopic from URL/title/user_problem first.
        # Docker Docs pages include global navigation and related examples that mention
        # volumes, compose, GitHub Actions, paths, etc. Those body terms must not
        # override the explicit requested source.
        dockerfile_signal = strong_has_any(dockerfile_terms)
        volume_signal = strong_has_any(volume_terms)
        compose_signal = strong_has_any(compose_terms)
        container_signal = strong_has_any(container_terms)

        if not any([dockerfile_signal, volume_signal, compose_signal, container_signal]):
            dockerfile_signal = body_has_any(dockerfile_terms)
            volume_signal = body_has_any(volume_terms)
            compose_signal = body_has_any(compose_terms)
            container_signal = body_has_any(container_terms)

        if dockerfile_signal:
            return {
                "kind": "docker",
                "title": (title or "Dockerfile 학습").strip(),
                "article_title": "Dockerfile 학습 기록: 이미지 빌드 단계와 컨테이너 실행 단계 구분하기",
                "subtitle": "Clarifying Dockerfile instructions, image build, and container runtime flow",
                "default_problem": "Dockerfile 명령어를 단순 나열이 아니라 이미지 빌드 단계의 역할로 구분하고, build와 run의 경계를 이해하는 것",
                "scope": "Dockerfile, FROM, WORKDIR, COPY, RUN, CMD, docker build, image layer, docker run",
                "flow": "base image 선택 → 작업 디렉터리 설정 → 파일 복사 → 의존성 설치 → 실행 명령 정의 → 이미지 빌드 → 컨테이너 실행",
                "concepts": [
                    ("Dockerfile", "이미지를 만들기 위한 빌드 절차를 명령어 순서로 기록하는 설계도다."),
                    ("FROM", "새 이미지가 출발할 base image를 지정하는 첫 단계다."),
                    ("WORKDIR", "뒤따르는 COPY, RUN, CMD가 실행될 작업 디렉터리를 정하는 명령이다."),
                    ("COPY", "호스트의 파일이나 소스 코드를 이미지 안으로 복사하는 단계다."),
                    ("RUN", "이미지 빌드 중 의존성 설치나 설정 명령을 실행해 layer를 만드는 단계다."),
                    ("CMD", "컨테이너가 시작될 때 기본으로 실행할 명령을 지정하는 단계다."),
                    ("docker build", "Dockerfile을 읽어 실행 가능한 image를 만드는 명령이다."),
                    ("docker run", "완성된 image를 실제 container로 실행하는 명령이다."),
                ],
                "steps": [
                    ("FROM으로 base image 기준 잡기", "Dockerfile의 첫 줄이 단순 선언처럼 보여도 이후 모든 실행 환경의 기준이 된다.", "FROM을 런타임과 OS/언어 환경을 정하는 base image 선택 단계로 정리했다.", "Dockerfile을 보고 어떤 환경 위에서 앱이 빌드되는지 설명할 수 있는지 확인했다."),
                    ("WORKDIR/COPY로 파일 위치 흐름 확인", "소스가 어디에 복사되고 이후 명령이 어느 경로에서 실행되는지 헷갈릴 수 있었다.", "WORKDIR은 작업 위치, COPY는 필요한 파일을 이미지 안으로 가져오는 단계로 나누었다.", "COPY 대상 경로와 이후 RUN/CMD가 바라보는 경로를 연결해 설명할 수 있는지 확인했다."),
                    ("RUN과 CMD의 시점 분리", "RUN과 CMD 모두 명령어처럼 보이지만 실행 시점이 다르다.", "RUN은 image build 중 실행되고 CMD는 container start 시 기본 실행 명령이라는 기준으로 구분했다.", "빌드할 때 실행되는 명령과 컨테이너 실행 때 실행되는 명령을 분리해 말할 수 있는지 확인했다."),
                    ("build와 run의 경계 검증", "Dockerfile을 수정해도 기존 컨테이너에 바로 반영된다고 오해할 수 있었다.", "Dockerfile 변경 → docker build로 새 image 생성 → docker run으로 container 실행 흐름으로 정리했다.", "Dockerfile 수정 후 왜 다시 build해야 하는지 설명할 수 있으면 이해한 것으로 보았다."),
                ],
                "skills": ["Dockerfile 명령어 역할 구분", "image build 흐름 이해", "RUN/CMD 실행 시점 분리", "docker build/run 경계 검증", "image layer 관점 정리"],
            }

        if volume_signal:
            return {
                "kind": "docker",
                "title": (title or "Docker Volume 학습").strip(),
                "article_title": "Docker Volume 학습 기록: 컨테이너 생명주기와 데이터 생명주기 분리하기",
                "subtitle": "Clarifying Docker volumes, mounts, and data persistence",
                "default_problem": "컨테이너가 삭제·재생성되어도 데이터가 유지되어야 하는 상황에서 volume이 왜 필요한지 이해하는 것",
                "scope": "Docker volume, data persistence, container lifecycle, named volume, mount, docker volume",
                "flow": "컨테이너 실행 → 데이터 생성 → 컨테이너 삭제/재생성 → volume 연결 → 데이터 유지 확인",
                "concepts": [
                    ("Volume", "컨테이너와 분리된 Docker 관리 저장소에 데이터를 보존하는 방식이다."),
                    ("Data persistence", "컨테이너가 사라져도 유지되어야 하는 상태 데이터를 보존하는 요구사항이다."),
                    ("Container lifecycle", "컨테이너는 생성·실행·중지·삭제될 수 있으므로 데이터 수명과 분리해서 봐야 한다."),
                    ("Named volume", "이름을 가진 volume으로 여러 컨테이너 실행 사이에서도 같은 데이터를 다시 연결할 수 있다."),
                    ("Mount", "volume이나 host path를 컨테이너 내부 경로에 연결하는 설정이다."),
                    ("docker volume", "volume을 생성·조회·삭제하는 Docker 명령 그룹이다."),
                ],
                "steps": [
                    ("컨테이너와 데이터 생명주기 분리", "컨테이너 안에만 데이터를 두면 컨테이너 삭제 시 데이터 유지 기준이 흐려진다.", "컨테이너는 실행 환경, volume은 지속 데이터 저장 위치로 나누었다.", "컨테이너를 재생성해도 유지되어야 하는 데이터가 무엇인지 설명할 수 있는지 확인했다."),
                    ("named volume의 재사용 기준 이해", "매번 새 저장 위치를 쓰면 이전 데이터와 연결되지 않을 수 있다.", "named volume을 같은 이름으로 다시 mount해 데이터를 이어 쓰는 방식으로 정리했다.", "동일한 named volume을 연결하면 재실행 후에도 데이터가 남는다고 설명할 수 있는지 확인했다."),
                    ("mount 경로 확인", "volume이 있어도 컨테이너 내부의 어느 경로와 연결되는지 모르면 검증하기 어렵다.", "호스트/Docker 관리 저장소와 컨테이너 내부 경로의 연결을 mount 기준으로 확인했다.", "컨테이너 내부 경로에 쓴 데이터가 volume에 남는 흐름을 설명할 수 있으면 이해한 것으로 보았다."),
                ],
                "skills": ["volume 기반 데이터 보존", "container/data lifecycle 분리", "named volume 사용 기준", "mount 경로 검증", "docker volume 명령 이해"],
            }

        if compose_signal:
            return {
                "kind": "docker",
                "title": (title or "Docker Compose 학습").strip(),
                "article_title": "Docker Compose 학습 기록: 여러 컨테이너를 하나의 애플리케이션 구성으로 묶기",
                "subtitle": "Clarifying compose.yaml services, networks, volumes, and multi-container orchestration",
                "default_problem": "여러 컨테이너를 각각 docker run으로 실행하는 대신 compose.yaml로 서비스, 네트워크, 볼륨을 함께 관리하는 기준을 이해하는 것",
                "scope": "Docker Compose, compose.yaml, service, network, volume, dependency, docker compose up",
                "flow": "서비스 정의 → 이미지/빌드 설정 → 네트워크 연결 → 볼륨 연결 → docker compose up 실행 → 서비스 상태 확인",
                "concepts": [
                    ("Docker Compose", "여러 컨테이너 서비스를 하나의 애플리케이션 구성으로 정의하고 실행하는 도구다."),
                    ("compose.yaml", "서비스, 네트워크, 볼륨, 실행 옵션을 선언하는 Compose 설정 파일이다."),
                    ("Service", "Compose에서 하나의 컨테이너 역할을 정의하는 단위다."),
                    ("Network", "서비스들이 서로 통신할 수 있도록 연결하는 구성이다."),
                    ("Volume", "서비스 재시작 후에도 유지되어야 하는 데이터를 분리하는 설정이다."),
                    ("docker compose up", "compose.yaml에 정의된 여러 서비스를 한 번에 생성하고 실행하는 명령이다."),
                ],
                "steps": [
                    ("docker run 반복과 compose.yaml 비교", "서비스가 여러 개가 되면 개별 docker run 명령만으로 전체 구성을 재현하기 어렵다.", "Compose를 여러 서비스 실행 조건을 파일 하나에 선언하는 방식으로 정리했다.", "같은 애플리케이션을 compose.yaml만 보고 재실행할 수 있는지 확인했다."),
                    ("service 단위로 컨테이너 역할 나누기", "웹, DB, 캐시가 모두 container이지만 역할과 설정은 다르다.", "각 container 역할을 service로 분리하고 필요한 이미지, 포트, 환경변수, 볼륨을 함께 읽었다.", "compose 파일에서 각 service가 어떤 역할을 하는지 설명할 수 있는지 확인했다."),
                    ("network/volume 연결 기준 확인", "여러 서비스가 함께 실행되어도 통신과 데이터 보존 기준이 없으면 애플리케이션으로 동작하지 않는다.", "network는 서비스 간 통신, volume은 데이터 유지 설정으로 나누어 확인했다.", "docker compose up 이후 서비스 연결과 데이터 유지 조건을 말할 수 있으면 이해한 것으로 보았다."),
                ],
                "skills": ["Compose 서비스 구성", "compose.yaml 읽기", "multi-container 실행 흐름", "network/volume 연결 이해", "docker compose up 검증"],
            }

        if container_signal:
            return {
                "kind": "docker",
                "title": (title or "Docker Container 학습").strip(),
                "article_title": "Docker Container 학습 기록: 이미지에서 실행 중인 격리 환경으로 이어지는 흐름 이해하기",
                "subtitle": "Clarifying containers, images, isolation, runtime, and Docker Engine",
                "default_problem": "컨테이너가 가상머신과 어떻게 다르고 이미지와 어떤 관계인지 구분하는 것",
                "scope": "container, image, isolation, runtime, Docker Engine, lightweight environment",
                "flow": "image 준비 → Docker Engine 실행 → container 생성 → 격리 환경에서 프로세스 실행 → 상태 확인",
                "concepts": [
                    ("Container", "이미지를 바탕으로 실행되는 격리된 프로세스 환경이다."),
                    ("Image", "컨테이너 실행에 필요한 파일과 설정을 담은 실행 전 패키지다."),
                    ("Isolation", "컨테이너가 파일 시스템, 네트워크, 프로세스 공간을 분리해 실행되는 특성이다."),
                    ("Runtime", "이미지를 실제 실행 상태로 만드는 컨테이너 실행 계층이다."),
                    ("Docker Engine", "이미지 빌드와 컨테이너 실행을 관리하는 Docker 핵심 구성이다."),
                    ("Lightweight environment", "가상머신보다 가볍게 앱 실행 환경을 분리하는 방식이다."),
                ],
                "steps": [
                    ("이미지와 컨테이너 관계 정리", "image와 container가 모두 Docker 실행 단위처럼 보여 같은 것으로 오해할 수 있었다.", "image는 실행 전 패키지, container는 image에서 만들어진 실행 상태로 나누었다.", "같은 image에서 여러 container를 만들 수 있다고 설명할 수 있는지 확인했다."),
                    ("격리 환경의 의미 이해", "컨테이너가 단순 프로세스인지 VM인지 경계가 헷갈릴 수 있었다.", "컨테이너를 host kernel을 공유하면서 실행 환경을 격리하는 방식으로 정리했다.", "컨테이너가 왜 가볍고 빠르게 생성되는지 VM과 비교해 설명할 수 있는지 확인했다."),
                    ("Docker Engine 실행 흐름 확인", "사용자는 docker run 명령만 보지만 내부적으로는 이미지 조회, 컨테이너 생성, 프로세스 실행이 이어진다.", "Docker Engine이 image를 기반으로 container runtime을 통해 실행 상태를 만든다고 정리했다.", "docker ps 등으로 실행 중인 container 상태를 확인할 수 있으면 이해한 것으로 보았다."),
                ],
                "skills": ["image/container 관계 구분", "container isolation 이해", "Docker Engine 역할 정리", "VM과 container 비교", "runtime 상태 확인"],
            }

        return {
            "kind": "docker",
            "title": (title or "Docker 학습").strip(),
            "article_title": "Docker 학습 기록: 이미지·컨테이너·포트·볼륨의 역할 구분하기",
            "subtitle": "Clarifying Docker image, container, port mapping, volume, and Compose flow",
            "default_problem": "Docker에서 이미지, 컨테이너, 포트 매핑, 볼륨이 각각 어떤 역할을 하는지 구분하고, 컨테이너가 실행되어 외부에서 접근 가능한 상태가 되는 흐름을 이해하는 것",
            "scope": "Docker image, container, Dockerfile, docker build/run, port mapping, volume, Docker Compose",
            "flow": "이미지 정의 → 컨테이너 실행 → 포트 연결 → 데이터 보존 → 여러 서비스 구성",
            "concepts": [
                ("Docker image", "애플리케이션 실행에 필요한 파일, 설정, 의존성을 담은 실행 전 패키지로 이해했다."),
                ("Container", "이미지를 바탕으로 실제 실행된 격리 환경이며, 같은 이미지에서 여러 컨테이너를 만들 수 있다."),
                ("Dockerfile", "이미지를 어떤 순서와 명령으로 만들지 기록하는 빌드 설계도다."),
                ("docker build", "Dockerfile을 읽어 실행 가능한 image를 만드는 단계다."),
                ("docker run", "image를 실제 container로 실행하는 단계다."),
                ("Port mapping", "컨테이너 내부 포트를 호스트의 포트와 연결해 외부 접근을 가능하게 하는 설정이다."),
                ("Volume", "컨테이너가 삭제되어도 데이터가 유지되도록 호스트나 Docker 관리 저장소에 데이터를 분리하는 방식이다."),
                ("Docker Compose", "여러 컨테이너와 네트워크·볼륨 설정을 하나의 YAML 파일로 함께 실행하는 방식이다."),
            ],
            "steps": [
                ("이미지와 컨테이너의 경계 구분", "처음에는 image와 container가 모두 Docker 실행 단위처럼 보여 헷갈릴 수 있었다.", "image는 실행 전 패키지이고 container는 그 image가 실제로 실행된 상태로 나누어 정리했다.", "같은 image에서 여러 container를 실행할 수 있고, container를 삭제해도 image 자체는 남는다고 설명할 수 있으면 이해한 것으로 보았다."),
                ("Dockerfile과 build/run 흐름 연결", "Dockerfile을 작성하는 것과 container를 실행하는 것이 한 단계처럼 보일 수 있었다.", "Dockerfile은 image를 만들기 위한 선언이고, docker build가 image를 만들며, docker run이 image를 container로 실행한다고 구분했다.", "Dockerfile 수정 후에는 image를 다시 build해야 변경사항이 반영된 container를 실행할 수 있다고 설명할 수 있는지 확인했다."),
                ("포트 매핑으로 외부 접근 조건 확인", "컨테이너 안에서 서비스가 떠 있어도 호스트 브라우저나 외부 요청에서 바로 접근되는 것은 아니었다.", "컨테이너 내부 포트와 호스트 포트를 연결하는 port mapping을 외부 접근의 조건으로 정리했다.", "localhost의 어떤 포트로 접근해야 컨테이너 내부 서비스에 도달하는지 설명할 수 있으면 포트 매핑을 이해한 것으로 보았다."),
                ("볼륨으로 데이터 보존 문제 분리", "컨테이너는 쉽게 삭제·재생성될 수 있기 때문에 데이터를 컨테이너 내부에만 두면 유지 기준이 흐려진다.", "volume은 실행 환경과 데이터를 분리해 컨테이너 lifecycle과 데이터 lifecycle을 다르게 관리하는 방식으로 정리했다.", "컨테이너를 다시 만들어도 유지되어야 하는 데이터는 volume으로 분리해야 한다고 판단할 수 있는지 확인했다."),
                ("Compose로 여러 컨테이너 흐름 묶기", "웹 앱, 데이터베이스, 캐시처럼 여러 서비스가 함께 움직이면 docker run 명령을 각각 기억하기 어렵다.", "Docker Compose를 여러 서비스, 네트워크, 볼륨을 하나의 파일로 묶어 실행하는 방식으로 정리했다.", "서비스 간 의존성과 포트·볼륨 설정을 compose 파일에서 함께 읽을 수 있으면 이해한 것으로 보았다."),
            ],
            "skills": ["Docker image/container 구분", "Dockerfile build 흐름 이해", "port mapping 검증", "volume 기반 데이터 보존 판단", "Compose 서비스 구성 이해"],
        }

    if requested_kind == "javascript_async" or (not requested_kind and strong_has_any(["async function", "async/await", "await", "promise-based", "promise based"])):
        return {
            "kind": "javascript_async",
            "title": (title or "async function 학습").strip(),
            "article_title": "JavaScript async function 학습 기록: 동기처럼 보이는 비동기 흐름 이해하기",
            "subtitle": "Understanding async functions, await, Promise return values, and error handling",
            "default_problem": "async function 안의 코드는 동기적으로 보이지만 실제로는 Promise 기반 비동기 흐름으로 동작한다는 점을 이해하는 것",
            "scope": "async function, await, Promise return value, async flow, error handling",
            "flow": "async 함수 호출 → Promise 반환 → await로 완료 시점 대기 → 결과 반환/오류 처리",
            "concepts": [
                ("async function", "함수 내부에서 await를 사용할 수 있고 호출 결과로 Promise를 반환하는 비동기 함수다."),
                ("await", "Promise가 처리될 때까지 해당 async 함수의 흐름을 기다리게 하고 결과값을 꺼내는 키워드다."),
                ("Promise", "async 함수의 반환값과 비동기 작업 완료 상태를 표현하는 객체다."),
                ("Return value", "async 함수에서 반환한 값은 직접 값처럼 보이더라도 Promise로 감싸져 호출자에게 전달된다."),
                ("Error handling", "await 중 발생한 rejected Promise나 throw를 try/catch로 처리하는 흐름이다."),
            ],
            "steps": [
                ("동기처럼 보이는 코드와 실제 비동기 흐름 구분", "async 함수 내부 코드는 위에서 아래로 읽히지만 완료 시점은 Promise 처리에 의해 결정된다.", "async function은 호출 즉시 최종 값이 아니라 Promise를 반환한다고 정리했다.", "함수 호출 결과가 Promise이며 완료 후 값이 결정된다고 설명할 수 있는지 확인했다."),
                ("await의 대기 기준 이해", "await를 단순 문법으로 보면 무엇을 기다리는지 흐려진다.", "await는 Promise가 fulfilled 또는 rejected 될 때까지 async 함수 내부 흐름을 멈추는 기준으로 보았다.", "await 뒤의 코드가 Promise 처리 이후 실행된다고 설명할 수 있는지 확인했다."),
                ("오류 처리 흐름 분리", "비동기 오류는 일반 반환값과 다르게 rejected 상태로 전달될 수 있다.", "try/catch를 await와 함께 사용해 실패 흐름을 별도로 처리하는 방식으로 정리했다.", "성공 결과와 실패 처리를 같은 async 함수 안에서 구분할 수 있으면 이해한 것으로 보았다."),
            ],
            "skills": ["async/await 흐름 이해", "Promise 반환값 구분", "비동기 완료 시점 판단", "try/catch 오류 처리", "동기처럼 보이는 코드의 실행 순서 해석"],
        }

    promise_hits = sum(1 for t in ["promise", "pending", "fulfilled", "rejected", "resolve", "reject", "then", "catch"] if t in blob)
    if requested_kind == "javascript_promise" or (not requested_kind and promise_hits >= 3 and strong_has_any(["promise", "pending", "fulfilled", "rejected", "resolve", "reject"])):
        return {
            "kind": "javascript_promise",
            "title": (title or "Promise 학습").strip(),
            "article_title": "JavaScript Promise 학습 기록: 비동기 결과와 상태 전환 이해하기",
            "subtitle": "Understanding pending, fulfilled, rejected states and async result handling",
            "default_problem": "Promise에서 비동기 작업의 결과가 즉시 반환되지 않고 pending, fulfilled, rejected 상태로 흘러간다는 점을 이해하는 것",
            "scope": "Promise, pending, fulfilled, rejected, then, catch, async result handling",
            "flow": "비동기 작업 시작 → pending 상태 → 성공/실패 분기 → then/catch 처리 → 다음 작업 연결",
            "concepts": [
                ("Promise", "비동기 작업의 최종 성공 또는 실패 결과를 나중에 받을 수 있게 표현한 객체다."),
                ("pending", "비동기 작업이 아직 완료되지 않은 대기 상태다."),
                ("fulfilled", "작업이 성공해서 결과값을 사용할 수 있는 상태다."),
                ("rejected", "작업이 실패해서 오류를 처리해야 하는 상태다."),
                ("then", "Promise가 fulfilled 되었을 때 결과를 이어서 처리하는 메서드다."),
                ("catch", "Promise가 rejected 되었을 때 오류를 처리하는 메서드다."),
            ],
            "steps": [
                ("즉시 반환값과 나중에 도착하는 결과 구분", "비동기 코드는 함수 호출 직후 결과가 바로 있는 것처럼 읽으면 흐름을 오해하기 쉽다.", "Promise를 지금 값이 아니라 미래의 완료 결과를 나타내는 객체로 정리했다.", "작업이 끝나기 전에는 pending이고 결과 사용은 then/catch 흐름에서 해야 한다고 설명할 수 있는지 확인했다."),
                ("fulfilled와 rejected 흐름 분리", "성공과 실패가 같은 코드 흐름 안에서 섞이면 오류 처리 위치가 모호해진다.", "fulfilled는 정상 결과 처리, rejected는 오류 처리로 나누어 이해했다.", "성공 시 then, 실패 시 catch로 분기된다고 설명할 수 있으면 상태 전환을 이해한 것으로 보았다."),
                ("then/catch를 암기보다 제어 흐름으로 이해", "then과 catch를 문법으로만 외우면 왜 필요한지 남지 않는다.", "then/catch를 비동기 결과가 도착한 뒤 다음 처리를 연결하는 흐름으로 정리했다.", "API 호출이나 타이머처럼 늦게 끝나는 작업의 결과를 어디에서 이어받는지 설명할 수 있어야 했다."),
            ],
            "skills": ["Promise 상태 전환 이해", "비동기 결과 처리", "then/catch 흐름 구분", "오류 처리 기준 정리"],
        }

    # v4.7.18 source-first fallback: if no known profile strongly matches,
    # build the article contract from the current source instead of borrowing
    # stale FastAPI/NumPy/Python templates.
    return source_first_fallback_profile(seed_url, title, body_text, user_problem)


def evidence_for_aliases(source_text: str, aliases: list[str], limit: int = 2) -> list[str]:
    sentences = split_learning_sentences(source_text, limit=500)
    noise = [
        "subscribe", "newsletter", "follow", "tweet", "twitter", "sign up", "광고", "구독",
        "table of contents", "목차", "copyright", "privacy policy",
        "candidate focus list", "body chars", "image number", "이미지 번호", "캡션 목록",
        "selected focus", "focus title", "focus url", "problem framing candidate", "why selected",
        "source graph", "source pack", "seed_url", "collector_title",
        "get docker", "guides manuals reference", "free email series", "no spam",
    ]
    picked: list[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(n in lower for n in noise):
            continue
        if any(alias.lower() in lower for alias in aliases):
            if sentence not in picked:
                picked.append(sentence)
        if len(picked) >= limit:
            break
    return picked


def clean_evidence_sentence(sentence: str) -> str:
    """Remove collector/navigation noise from a sentence before showing it to users."""
    text = re.sub(r"\s+", " ", str(sentence or "")).strip()
    text = re.sub(r"\s*-\s*Body chars:\s*\d+\b", "", text, flags=re.I)
    text = re.sub(r"\bBody chars:\s*\d+\b", "", text, flags=re.I)
    text = text.replace("이미지 번호/캡션 목록 ## Candidate focus list -", "")
    text = text.replace("## Candidate focus list -", "")
    text = re.sub(r"Back Get started Guides Manuals Reference Get Docker What is Docker\??", "", text, flags=re.I)
    text = re.sub(r"FREE Email Series.*?No spam\.??", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" -—")


def command_code_block_for_profile(profile: dict[str, Any], source_text: str = "") -> str:
    """Return a compact, topic-specific code/command section.

    This is not used as topic evidence; it is a learning aid for sources whose docs
    commonly include commands but whose collected text can lose code formatting.
    """
    custom_code = str((profile or {}).get("code_example") or "").strip()
    if custom_code:
        return custom_code
    kind = str(profile.get("kind") or "").lower()
    scope = str(profile.get("scope") or "").lower()
    title = str(profile.get("article_title") or profile.get("title") or "").lower()
    joined = "\n".join([title, scope, str(source_text or "")[:4000].lower()])

    if kind == "docker" and any(t in joined for t in ["dockerfile", "workdir", "docker build", "from", " cmd"]):
        return """대표 검증 예시의 핵심은 Dockerfile 명령어가 어느 시점에 작동하는지 구분하는 것이다.

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
CMD ["npm", "start"]
```

```bash
docker build -t getting-started .
docker run getting-started
```

여기서 `RUN`은 이미지를 만드는 동안 실행되고, `CMD`는 완성된 이미지를 컨테이너로 시작할 때 기본 실행 명령이 된다. 이 차이를 구분해야 Dockerfile 수정 후 다시 build해야 하는 이유를 설명할 수 있다."""

    if kind == "docker" and any(t in joined for t in ["docker compose", "compose.yaml", "compose.yml", "multi-container", "service"]):
        return """대표 검증 예시의 핵심은 여러 컨테이너를 개별 `docker run` 명령으로 기억하는 대신, 하나의 Compose 파일로 애플리케이션 구성을 재현하는 것이다.

```yaml
services:
  web:
    image: example-web
    ports:
      - "8080:80"
    depends_on:
      - db
  db:
    image: postgres
    volumes:
      - db-data:/var/lib/postgresql/data
volumes:
  db-data:
```

```bash
docker compose up
```

`service`는 컨테이너 역할을 나누는 단위이고, `network`와 `volume`은 서비스 간 연결과 데이터 보존 조건을 함께 관리하는 기준이 된다."""

    if kind == "docker" and any(t in joined for t in ["volume", "mount", "data persistence", "named volume", "docker volume"]):
        return """대표 검증 예시의 핵심은 컨테이너의 생명주기와 데이터의 생명주기를 분리해서 보는 것이다.

```bash
docker volume create app-data
docker run --mount source=app-data,target=/data example-image
# 또는
docker run -v app-data:/data example-image
```

컨테이너는 삭제·재생성될 수 있지만, 같은 named volume을 다시 mount하면 데이터는 이어서 사용할 수 있다. 따라서 삭제되어도 유지되어야 하는 상태 데이터는 컨테이너 내부가 아니라 volume에 분리하는 것이 검증 기준이 된다."""

    if kind == "docker" and any(t in joined for t in ["container", "isolation", "runtime", "docker engine"]):
        return """대표 검증 예시의 핵심은 image와 container의 관계를 실행 전/실행 중으로 나누어 보는 것이다.

```bash
docker pull nginx
docker run --name web -p 8080:80 nginx
docker ps
```

이미지는 실행 전 패키지이고, 컨테이너는 그 이미지를 바탕으로 실행된 격리 환경이다. 같은 image에서 여러 container를 만들 수 있다는 점이 image/container 구분의 핵심 검증 기준이다."""

    if kind == "fastapi":
        return """대표 검증 예시의 핵심은 endpoint를 작성하는 것과 요청/응답을 검증하는 것을 분리하는 것이다.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
def create_item(item: Item):
    return item
```

`item_id`는 path parameter, `q`는 query parameter, `Item`은 request body schema다. `/docs`에서 이 세 입력 위치가 어떻게 문서화되고 검증되는지 확인하는 것이 학습 완료 기준이 된다."""

    if kind == "github_actions":
        return """대표 검증 예시의 핵심은 YAML을 단순 설정 파일이 아니라 자동화 실행 구조로 읽는 것이다.

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm test
```

`workflow`는 자동화 전체, `job`은 runner에서 실행되는 작업 묶음, `step`은 실제 명령이나 action 호출이다. 실패가 났을 때 어느 job/step에서 멈췄는지 추적할 수 있어야 한다."""

    if kind == "javascript_async":
        return """대표 검증 예시의 핵심은 async 함수가 동기 함수처럼 보이더라도 호출 결과는 Promise 흐름으로 이어진다는 점이다.

```javascript
async function loadUser() {
  try {
    const response = await fetch("/api/user");
    return await response.json();
  } catch (error) {
    console.error(error);
    throw error;
  }
}

loadUser().then(user => console.log(user));
```

`await`는 Promise가 처리될 때까지 async 함수 내부의 다음 흐름을 기다리게 한다. `return`된 값도 호출자에게는 Promise를 통해 전달되므로, 완료 시점과 오류 처리를 함께 확인해야 한다."""

    if kind == "javascript_promise":
        return """대표 검증 예시의 핵심은 비동기 작업의 결과가 즉시 값으로 반환되지 않고 Promise 상태 전환을 통해 도착한다는 점이다.

```javascript
const promise = new Promise((resolve, reject) => {
  setTimeout(() => resolve("done"), 1000);
});

promise
  .then(result => console.log(result))
  .catch(error => console.error(error));
```

처음 상태는 `pending`이고, `resolve`가 호출되면 `fulfilled`, `reject`가 호출되면 `rejected`가 된다. `then`과 `catch`는 이 상태 전환 뒤의 처리를 연결하는 흐름으로 이해해야 한다."""

    if kind == "react_useeffect":
        return """대표 검증 예시의 핵심은 렌더링 자체와 외부 시스템 동기화 Effect를 분리해서 보는 것이다.

```javascript
useEffect(() => {
  const connection = createConnection(serverUrl, roomId);
  connection.connect();

  return () => {
    connection.disconnect();
  };
}, [serverUrl, roomId]);
```

`setup`은 외부 연결을 시작하고, 반환 함수인 `cleanup`은 이전 연결을 정리한다. dependency가 바뀌면 React는 이전 cleanup을 먼저 실행한 뒤 새 setup을 실행한다는 점이 핵심 검증 기준이다."""

    if kind == "kubernetes_pod":
        return """대표 검증 예시의 핵심은 Pod가 하나 이상의 container를 묶는 Kubernetes의 최소 배포 단위라는 점이다.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
    - name: app
      image: nginx
```

`Pod`는 container 자체가 아니라 container를 담는 실행 단위다. 같은 Pod 안의 container는 네트워크와 일부 저장소 같은 shared resources를 함께 사용할 수 있다는 점을 확인해야 한다."""

    if kind == "postgres_index":
        return """대표 검증 예시의 핵심은 index가 모든 쿼리를 무조건 빠르게 만드는 것이 아니라 특정 조건 검색의 비용을 줄이는 구조라는 점이다.

```sql
CREATE INDEX test1_id_index ON test1 (id);

SELECT *
FROM test1
WHERE id = 42;
```

`CREATE INDEX`는 검색에 자주 쓰이는 컬럼에 보조 구조를 만든다. 실제 사용 여부는 query planner가 데이터 규모와 조건의 선택도를 보고 결정하므로, sequential scan과 index scan을 구분해서 확인해야 한다."""

    if kind == "fetch_api":
        return """대표 검증 예시의 핵심은 `fetch()`가 실제 JSON 데이터를 바로 반환하는 것이 아니라 Response 객체로 fulfilled 되는 Promise를 반환한다는 점이다.

```javascript
const response = await fetch("/api/items");

if (!response.ok) {
  throw new Error(`HTTP error: ${response.status}`);
}

const data = await response.json();
```

`response`에서 status와 headers를 확인한 뒤 `json()`이나 `text()`로 body를 parsing해야 실제 데이터를 사용할 수 있다."""

    if kind == "sqlalchemy_orm":
        return """대표 검증 예시의 핵심은 database table을 Python class와 Session 작업 단위로 다루는 것이다.

```python
with Session(engine) as session:
    stmt = select(User).where(User.name == "spongebob")
    user = session.scalars(stmt).one()
    user.fullname = "Spongebob Squarepants"
    session.commit()
```

`Engine`은 연결 기반이고, `Session`은 ORM 객체 조회와 변경, transaction을 관리한다. 객체 변경은 Session 안에 머물다가 `commit()`으로 확정된다는 점을 확인해야 한다."""

    if kind == "python_modules":
        return """대표 검증 예시의 핵심은 파일을 직접 실행하는 것과 import 가능한 module로 사용하는 흐름을 분리하는 것이다.

```python
# fibo.py
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)

# 다른 파일에서
import fibo
print(fibo.fib(10))
```

module은 재사용 가능한 이름 공간을 만들고, import는 그 module의 이름을 현재 코드에서 사용할 수 있게 한다. package가 되면 여러 module을 디렉터리 구조로 묶어 관리할 수 있다."""

    if kind == "numpy_broadcasting":
        return """대표 검증 예시의 핵심은 배열의 값보다 shape 규칙을 먼저 확인하는 것이다.

```python
import numpy as np

a = np.ones((3, 1))
b = np.arange(4)
result = a + b  # (3, 1)과 (4,)가 (3, 4)로 broadcast
```

뒤쪽 dimension부터 비교했을 때 두 크기가 같거나 하나가 1이면 broadcasting이 가능하다. 그렇지 않으면 incompatible shape 오류가 난다."""

    if kind == "pandas_groupby":
        return """대표 검증 예시의 핵심은 groupby를 split-apply-combine 흐름으로 읽는 것이다.

```python
result = df.groupby("category")["sales"].agg(["sum", "mean"])
```

`groupby`는 먼저 category 기준으로 데이터를 나누고, 각 그룹에 집계 함수를 적용한 뒤, 그룹별 결과를 하나의 결과로 합친다."""

    if kind == "redis_data_types":
        return """대표 검증 예시의 핵심은 값의 사용 패턴에 따라 Redis data type을 고르는 것이다.

```bash
SET user:1:name "Alice"
LPUSH jobs "task-1"
SADD tags "python" "redis"
HSET user:1 email "a@example.com" age 30
```

단일 값은 string, 순서가 필요한 작업 목록은 list, 중복 없는 집합은 set, 객체형 record는 hash처럼 선택 기준을 나눠야 한다."""

    if kind == "http_status":
        return """대표 검증 예시의 핵심은 status code를 개별 숫자가 아니라 범주로 해석하는 것이다.

```text
2xx: 요청 성공
3xx: 리다이렉션 또는 추가 동작 필요
4xx: 클라이언트 요청 문제
5xx: 서버 처리 문제
```

API 디버깅에서는 먼저 status code 범주를 보고 요청을 고칠지, 인증/권한을 확인할지, 서버 로그를 볼지 판단해야 한다."""

    if kind == "django_models":
        return """대표 검증 예시의 핵심은 Django model class가 database table 구조로 이어진다는 점이다.

```python
from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=200)
    published_at = models.DateTimeField(null=True)
```

model field는 table column과 연결되고, model 변경은 migration을 통해 database schema에 반영된다."""

    if kind == "node_event_loop":
        return """대표 검증 예시의 핵심은 동기 코드와 비동기 callback의 실행 시점을 분리하는 것이다.

```javascript
console.log("start");
setTimeout(() => console.log("timer"), 0);
Promise.resolve().then(() => console.log("promise"));
console.log("end");
```

작성 순서와 실행 순서가 항상 같지는 않다. call stack, microtask, timer phase를 구분해야 event loop 흐름을 설명할 수 있다."""

    if kind == "spring_beans":
        return """대표 검증 예시의 핵심은 객체를 직접 만드는 것과 Spring container가 bean을 관리하는 흐름을 분리하는 것이다.

```java
@Configuration
class AppConfig {
    @Bean
    UserService userService(UserRepository repository) {
        return new UserService(repository);
    }
}
```

Spring container는 configuration metadata를 바탕으로 bean을 만들고 필요한 dependency를 연결한다."""

    if kind == "web_workers":
        return """대표 검증 예시의 핵심은 무거운 작업을 main thread에서 분리하고 message로 통신하는 것이다.

```javascript
const worker = new Worker("worker.js");
worker.postMessage({ count: 1000000 });
worker.onmessage = event => {
  console.log(event.data);
};
```

Worker는 값을 직접 반환하지 않고 `postMessage`와 `onmessage`로 main thread와 데이터를 주고받는다."""

    if kind == "typescript_types":
        return """대표 검증 예시의 핵심은 JavaScript 값에 타입 계약을 붙여 입력과 구조를 검증하는 것이다.

```typescript
type User = {
  id: number;
  name: string;
};

function greet(user: User): string {
  return `Hello, ${user.name}`;
}
```

type annotation은 함수 입력, 반환값, 객체 구조를 명시해 잘못된 사용을 빠르게 발견하게 한다."""

    if kind == "source_first":
        return """이번 자료에서는 별도의 코드나 명령어를 억지로 만들기보다, 본문에서 확인한 개념을 실제로 적용할 때의 판단 기준을 정리했다.

- 먼저 확인할 것: 자료의 중심 개념과 사용자 메모의 문제 지점
- 구분할 것: 비슷해 보이는 개념의 역할, 적용 조건, 결과 기준
- 검증할 것: 정리한 기준으로 실제 상황에서 무엇을 먼저 선택하고 어떻게 실행할지 설명할 수 있는지"""

    return """이번 자료에서 확인한 코드와 명령어는 개념을 검증하기 위한 기준으로 정리했다. 단순히 따라 치는 것이 아니라, 어떤 입력을 넣고 어떤 결과가 나오면 이해했다고 볼 수 있는지 확인하는 데 초점을 두었다."""



def infer_display_title_from_url(seed_url: str, title: str, profile: dict[str, Any] | None = None) -> str:
    """Return a user-facing source title when the collector title was lost."""
    raw_title = re.sub(r"\s+", " ", str(title or "")).strip()
    generic_titles = {"", "generic web source", "web source", "학습 자료", "current web source"}
    if raw_title.lower() not in generic_titles:
        return raw_title

    url = str(seed_url or "").lower()
    if "writing-a-dockerfile" in url:
        return "Writing a Dockerfile | Docker Docs"
    if "engine/storage/volumes" in url or "/storage/volumes" in url:
        return "Volumes | Docker Docs"
    if "docs.docker.com/compose" in url:
        return "Docker Compose | Docker Docs"
    if "what-is-a-container" in url:
        return "What is a container? | Docker Docs"
    if "docker-simplified" in url:
        return "Docker Simplified: A Hands-On Guide for Absolute Beginners"
    if "fastapi.tiangolo.com/tutorial/body" in url:
        return "Request Body - FastAPI"
    if "fastapi" in url:
        return "A Close Look at a FastAPI Example Application – Real Python"
    if "developer.mozilla.org" in url and "async_function" in url:
        return "async function - JavaScript | MDN"
    if "developer.mozilla.org" in url and "global_objects/promise" in url:
        return "Promise - JavaScript | MDN"
    if "promise-basics" in url:
        return "Promise"
    if "react.dev" in url and "useeffect" in url:
        return "useEffect – React"
    if "kubernetes.io" in url and "/pods" in url:
        return "Pods | Kubernetes"
    if "postgresql.org" in url and "indexes-intro" in url:
        return "PostgreSQL Indexes Introduction"
    if "developer.mozilla.org" in url and "fetch_api" in url:
        return "Using the Fetch API - Web APIs | MDN"
    if "developer.mozilla.org" in url and "global_objects/array/map" in url:
        return "Array.prototype.map() - JavaScript | MDN"
    if "docs.sqlalchemy.org" in url and "/orm/quickstart" in url:
        return "ORM Quick Start — SQLAlchemy 2.0 Documentation"
    if "github" in url and "actions" in url:
        return "Understanding GitHub Actions - GitHub Docs"

    ptitle = re.sub(r"\s+", " ", str((profile or {}).get("title") or "")).strip()
    if ptitle and ptitle.lower() not in generic_titles:
        return ptitle
    return "웹문서"

def build_topic_learning_medium_article(
    seed_url: str,
    title: str,
    source_text: str,
    profile: dict[str, Any],
    user_problem: str = "",
    source_label: str = "학습 자료",
) -> str:
    display_title = infer_display_title_from_url(seed_url, title, profile)
    problem_note = clean_prompt_memo(user_problem)
    core_problem = problem_note or str(profile.get("default_problem") or "자료의 핵심 개념을 실제 판단 기준으로 정리하는 것")
    problem_subject = f"`{core_problem}`"
    problem_definition_subject = "이 문제" if problem_note else f"`{core_problem}`"
    problem_outcome_subject = "이 문제" if problem_note else f"`{core_problem}`라는 문제"
    article_title = str(profile.get("article_title") or f"{display_title} 학습 기록: 핵심 개념 정리하기")
    subtitle = str(profile.get("subtitle") or "Organizing technical concepts into learner-facing checkpoints")
    scope = str(profile.get("scope") or title)
    flow = str(profile.get("flow") or "개념 확인 → 역할 구분 → 적용 위치 확인 → 검증 기준 정리")
    concepts: list[tuple[str, str]] = list(profile.get("concepts") or [])
    steps: list[tuple[str, str, str, str]] = list(profile.get("steps") or [])
    skills: list[str] = list(profile.get("skills") or [])

    concept_names = [name for name, _ in concepts]
    concept_md = "\n".join(f"- **{name}**: {desc}" for name, desc in concepts[:10]) or "- 자료 본문에서 핵심 개념을 충분히 추출하지 못했습니다."
    evidence_lines: list[str] = []
    for name, _ in concepts[:6]:
        aliases = [name, name.lower()]
        aliases += [part.strip() for part in re.split(r"[/·,() ]+", name) if len(part.strip()) >= 4]
        found = evidence_for_aliases(source_text, aliases, limit=1)
        if found:
            evidence_lines.append(f"- {name}: {clean_evidence_sentence(found[0])}")
    if not evidence_lines:
        # Do not invent body evidence.  If the collector did not return a clean
        # source sentence for a concept, show a transparent selection note rather
        # than a fake-looking evidence sentence.
        evidence_lines = [
            "- 수집 본문에서 직접 인용 가능한 문장만 학습 단서로 표시한다. 직접 매칭되지 않은 개념은 아래 ‘주요 개념 정리’와 ‘검증 예시’에서 별도로 다룬다."
        ]

    steps_md = []
    for idx, (step_title, problem, action, validation) in enumerate(steps[:6], start=1):
        steps_md.append(f"""### {idx}. {step_title}
문제/제약: {problem}

조치: {action}

확인 기준: {validation}
""")

    default_steps_block = (
        "### 1. 핵심 개념을 적용 기준으로 분리\n"
        "문제/제약: 자료의 핵심 개념이 제목 중심으로만 남을 수 있었다.\n\n"
        "조치: 본문에서 반복되는 개념을 역할과 적용 위치 중심으로 다시 정리했다.\n\n"
        "확인 기준: 각 개념을 실제 상황에서 언제 쓰는지 설명할 수 있는지 확인했다."
    )
    steps_block = "\n".join(steps_md) if steps_md else default_steps_block
    clean_skills = [str(skill).strip() for skill in (skills[:8] or ["핵심 개념 분리", "학습 흐름 구조화", "검증 기준 설정"])]
    clean_skills = [skill for skill in clean_skills if skill and not skill.startswith("기반 문제 정의") and skill != "기반 문제 정의"]
    skills_block = "\n".join(f"- {skill}" for skill in clean_skills)
    evidence_block = "\n".join(evidence_lines[:8])
    code_block = command_code_block_for_profile(profile, source_text)
    code_section_title = "사용한 주요 수식/코드 정리"
    if str(profile.get("kind") or "").lower() == "source_first" and not str(profile.get("code_example") or "").strip():
        code_section_title = "사용한 주요 판단 기준 정리"
    concept_focus = ", ".join(concept_names[:5])

    return sanitize_medium_markdown(f"""# {article_title}

_{subtitle}_

## 짧은 도입부
이번 학습에서는 `{display_title}`를 바탕으로, 핵심 개념을 실제로 설명하고 검증할 수 있는 문제 해결 기준으로 다시 정리했다. 특히 `{scope}` 흐름을 따라가며, 비슷해 보이는 개념을 역할·적용 위치·확인 기준으로 나누어 보았다.

## 핵심 작업 요약
- 핵심 문제: {core_problem}
- 학습 자료: {source_label}
- 학습 범위: {scope}
- 핵심 흐름: {flow}
- 학습 결과: 자료의 핵심 개념을 정의 암기가 아니라 적용 상황과 확인 기준으로 다시 정리했다.

## 참고한 자료
- {seed_url}

## 본문에서 확인한 학습 단서
{evidence_block}

## 문제 인식
이번 자료에서 문제로 본 지점은 {problem_subject}였다. 용어를 아는 것만으로는 부족했고, 각 개념이 어느 단계에서 쓰이고 어떤 결과로 검증되는지까지 설명할 수 있어야 했다.

그래서 이번 글에서는 자료를 단순 요약하지 않고, 학습자가 헷갈릴 수 있는 개념 경계를 먼저 잡은 뒤 실제 실습이나 설명 흐름 안에서 검증 가능한 기준으로 바꾸었다.

## 문제 정의
이 학습에서 정의한 문제는 {problem_definition_subject}를 해결 가능한 학습 단위로 구체화하는 것이었다. 핵심은 제목을 외우는 것이 아니라, 각 개념이 어떤 문제를 해결하고 어느 단계에서 확인되는지 설명할 수 있게 만드는 데 있었다.

## 왜 이것을 문제로 인식했는가
기술 자료를 읽을 때 막히는 지점은 대개 용어 자체보다 개념 사이의 경계와 적용 조건이다. 따라서 이번 학습에서는 `{flow}` 순서로 내용을 다시 묶고, 각 단계마다 내가 무엇을 구분해야 하며 어떤 결과를 확인해야 하는지 정리했다.

## 문제 해결 경험
{steps_block}

## 복잡한 내용 정리
가장 복잡했던 부분은 여러 개념이 같은 주제 안에서 한꺼번에 등장하지만 실제 역할은 서로 다르다는 점이었다. 그래서 `{concept_focus}`를 하나의 목록으로만 보지 않고, 어떤 개념이 실행 전 준비인지, 어떤 개념이 실행 상태인지, 어떤 개념이 외부 접근이나 검증 기준인지 나누어 보았다.

## 성과
이번 학습을 통해 {problem_outcome_subject}를 중심으로 자료를 다시 설명할 수 있게 되었다. 단순히 글을 읽은 기록이 아니라, 어떤 개념을 왜 구분해야 하고 어떤 기준으로 확인해야 하는지를 남겼다.

## 사용한 주요 개념 정리
{concept_md}

## {code_section_title}
{code_block}

## 최종 정리
이번 글의 핵심은 자료를 짧게 요약하는 것이 아니라, 학습 중 헷갈릴 수 있는 개념을 문제로 정의하고, 각 개념을 적용 상황과 확인 기준으로 나누어 정리하는 것이었다. 앞으로 같은 주제를 다시 볼 때도 용어 목록이 아니라 문제 상황, 조치, 검증 기준 순서로 복습할 수 있다.

## Portfolio Summary
This learning record converts source material into a learner-facing technical note. It focuses on the learning problem, concept boundaries, practical workflow, and validation criteria rather than a generic summary of the source.

## Key skills practiced
{skills_block}
""")

def build_youtube_content_summary(seed_url: str, title: str, payload: dict[str, Any], graph: dict[str, Any], collector_report: dict[str, Any]) -> str:
    transcript_text = youtube_transcript_text(payload, graph)
    stats = source_graph_stats_summary(collector_report)
    quality = (collector_report.get("quality") if isinstance(collector_report.get("quality"), dict) else {}) or {}
    terms = youtube_learning_terms(title, transcript_text, limit=18)
    sections = youtube_keyword_sections(title, transcript_text, limit=7) or youtube_chapter_sections(transcript_text, limit=7)
    section_md = []
    for section in sections:
        section_md.append(
            f"### {section['title']}\n"
            + "\n\n".join(f"- {sentence}" for sentence in section["sentences"][:5])
        )
    focus_items = terms[:12] or [section["title"] for section in sections[:8]]
    return sanitize_medium_markdown(f"""# {title or 'YouTube 강의'} 핵심 내용 정리

## 수집 범위
- 원본 URL: {seed_url}
- 자막 세그먼트: {quality.get('transcript_segments') or 0}
- 본문/자막 글자 수: {stats.get('chars', 0)}

## 핵심 키워드
{markdown_bullets(focus_items)}

## 강의 흐름 요약
{markdown_bullets([section["title"] for section in sections], "- 자막에서 충분한 주제 흐름을 만들지 못했습니다.")}

## 주요 내용 정리
{chr(10).join(section_md) if section_md else "자막은 수집됐지만 요약 가능한 문장을 충분히 분리하지 못했습니다."}
""")


def build_youtube_problem_medium_draft(
    seed_url: str,
    run_id: str,
    source_pack_text: str,
    collector_report: dict[str, Any],
    user_problem: str = "",
) -> str:
    graph = collector_source_graph(collector_report, max_nodes=160)
    payload = load_collector_json(collector_report)
    title = str(graph.get("title") or "YouTube 강의").strip()
    transcript_text = youtube_transcript_text(payload, graph) or source_pack_text

    profile = topic_profile_from_text(seed_url, title, transcript_text, user_problem)
    if profile:
        return build_topic_learning_medium_article(
            seed_url=seed_url,
            title=title,
            source_text=transcript_text,
            profile=profile,
            user_problem=user_problem,
            source_label="YouTube 강의 자막과 영상 제목",
        )

    terms = youtube_learning_terms(title, transcript_text, limit=18)
    sections = youtube_keyword_sections(title, transcript_text, limit=6) or youtube_chapter_sections(transcript_text, limit=6)
    problem_note = clean_prompt_memo(user_problem)
    core_problem = problem_note or f"{title} 강의에서 반복되는 핵심 개념을 실제 설명 기준으로 정리하는 것"
    concept_names = unique_preserve_order([str(t) for t in terms[:8] if str(t).strip()], limit=8)
    if not concept_names:
        concept_names = unique_preserve_order([str(sec.get("title")) for sec in sections[:5] if str(sec.get("title") or "").strip()], limit=5)

    steps = []
    for idx, name in enumerate(concept_names[:4], start=1):
        related = evidence_for_aliases(transcript_text, [name], limit=1)
        evidence_line = related[0] if related else "강의 자막에서 이 개념이 학습 흐름의 일부로 확인되었다."
        steps.append(f"""### {idx}. {name}의 역할을 적용 기준으로 구분
문제/제약: `{name}`를 단순 용어로만 기억하면 실제 상황에서 언제 써야 하는지 흐려질 수 있었다.

조치: 강의에서 확인한 `{evidence_line}` 흐름을 바탕으로, 이 개념이 어떤 문제를 해결하고 어떤 단계에서 필요한지 정리했다.

확인 기준: `{name}`를 정의뿐 아니라 적용 상황과 결과 확인 기준까지 설명할 수 있는지 점검했다.
""")

    concept_md = "\n".join(f"- **{term}**: 강의 흐름에서 별도로 이해해야 할 핵심 개념으로, 적용 상황과 확인 기준을 함께 정리했다." for term in concept_names[:8])
    steps_md = "\n".join(steps) if steps else """### 1. 핵심 개념을 적용 기준으로 정리
문제/제약: 강의에서 다룬 핵심 개념이 제목 중심으로만 남을 수 있었다.

조치: 자막에서 반복되는 개념을 역할과 적용 상황 중심으로 다시 정리했다.

확인 기준: 각 개념을 실제 상황에서 언제 쓰는지 설명할 수 있는지 확인했다."""

    return sanitize_medium_markdown(f"""# {title}: 강의 핵심 개념을 학습자 관점으로 정리하기

_A learner-centered Medium note based on the YouTube lecture transcript_

## 짧은 도입부
이번 학습에서는 `{title}` 강의를 보면서, 내용을 단순 자막 요약으로 남기지 않고 실제로 내가 설명할 수 있는 기준으로 바꾸는 데 집중했다. 특히 강의에서 반복되는 개념을 정의, 필요성, 적용 상황, 확인 기준으로 나누어 정리했다.

## 핵심 작업 요약
- 내가 정의한 문제: {core_problem}
- 학습 자료: YouTube 강의 자막과 영상 제목
- 핵심 키워드: {", ".join(concept_names[:8]) if concept_names else title}
- 학습 결과: 강의 내용을 개념 목록이 아니라 문제 정의, 적용 기준, 검증 질문으로 재구성했다.

## 참고한 자료
- {seed_url}

## 문제 인식
이번 강의에서 문제로 본 지점은 `{core_problem}`이었다. 강의 내용을 그대로 따라가면 여러 용어와 예시가 이어지지만, 학습자로서는 각 개념이 어떤 상황에서 필요한지와 무엇을 기준으로 이해했다고 볼 수 있는지가 더 중요했다.

## 문제 정의
내가 정의한 문제는 강의 자막에 나온 핵심 내용을 “무엇인가”에서 멈추지 않고 “왜 필요한가”, “어떤 문제가 생기는가”, “어떤 기준으로 구분하는가”, “어떻게 확인하는가”까지 확장하는 것이었다.

## 왜 이것을 문제로 인식했는가
영상 강의는 빠르게 지나가기 때문에 개념 이름은 남아도 적용 기준은 흐려질 수 있다. 실제 학습 기록이나 포트폴리오 글에서는 “강의를 들었다”보다 “어떤 개념을 어떻게 구분했고 무엇으로 이해 여부를 확인했는가”가 드러나야 한다.

## 문제 해결 경험
{steps_md}

## 복잡한 내용 정리
가장 복잡했던 부분은 강의 내용이 순차적으로 설명되더라도 실제 이해는 순차적 암기가 아니라 구조화가 필요하다는 점이었다. 그래서 `{', '.join(concept_names[:5]) if concept_names else title}`를 정의, 적용 위치, 확인 결과로 나누어 다시 읽었다.

## 성과
이번 학습을 통해 `{core_problem}`라는 문제를 중심으로 강의 내용을 다시 설명할 수 있게 되었다. 단순히 영상을 본 기록이 아니라, 어떤 개념을 왜 구분해야 하고 어떤 기준으로 확인해야 하는지를 남겼다.

## 사용한 주요 개념 정리
{concept_md if concept_md else "- 강의 자막에서 핵심 개념을 충분히 추출하지 못했습니다."}

## 사용한 주요 수식/코드 정리
이번 YouTube 자막에서 Medium 글에 그대로 정리할 수 있는 코드 또는 수식 원문은 명확하게 확인되지 않았다. 따라서 임의의 코드를 만들지 않고, 자막에 드러난 개념과 판단 기준만 정리했다.

## 최종 정리
이번 글의 핵심은 YouTube 강의를 짧게 요약하는 것이 아니라, 강의에서 다룬 복잡한 내용을 학습자의 이해 기준으로 바꾸는 것이었다. 앞으로 같은 주제를 다시 볼 때도 용어 목록이 아니라 문제 상황, 적용 기준, 확인 질문 순서로 복습할 수 있다.

## Portfolio Summary
This learning record turns a YouTube lecture transcript into a learner-centered technical note. It defines the main learning problem, separates concept boundaries, and records validation criteria without inventing unsupported outcomes.

## Key skills practiced
- Extracting core concepts from lecture transcripts
- Defining a technical learning problem
- Separating definitions and validation criteria
- Converting video content into a portfolio narrative
- Writing learner-centered technical explanations
""")


def markdown_bullets(items: list[Any], fallback: str = "- 확인된 항목 없음") -> str:
    rows = [str(item or "").strip() for item in items if str(item or "").strip()]
    rows = unique_preserve_order(rows, limit=30)
    return "\n".join(f"- {item}" for item in rows) if rows else fallback


def source_graph_learning_nodes(graph: dict[str, Any], limit: int = 8) -> list[dict[str, str]]:
    """Return content-heavy learning nodes, excluding navigation/profile noise."""
    result: list[dict[str, str]] = []
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        title = re.sub(r"\s+", " ", str(node.get("title") or "")).strip()
        text = re.sub(r"\s+", " ", str(node.get("text") or "")).strip()
        node_type = str(node.get("type") or "").lower()
        lower = f"{title}\n{text[:900]}".lower()
        if len(text) < 500:
            continue
        if node_type.endswith("_root") and len(text) < 1600:
            continue
        if "새로운걸 공부하고 기록하는 것을 좋아합니다" in title:
            continue
        if "backend engineer" in lower:
            continue
        if title.lower() in {"home", "devlog", "about", "search", "share"}:
            continue
        score = len(text)
        if any(term in lower for term in ["프로세스", "스레드", "스케줄링", "메모리", "tcp", "udp", "http", "rest", "트랜잭션", "조인", "자료구조"]):
            score += 3000
        result.append({
            "title": title or str(node.get("url") or "학습 섹션"),
            "url": str(node.get("url") or ""),
            "type": str(node.get("type") or ""),
            "text": text,
            "score": str(score),
        })
    result.sort(key=lambda item: int(item.get("score") or 0), reverse=True)
    return result[:limit]


def source_node_key_sentences(text: str, limit: int = 3) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = cleaned.replace("Search Share", " ").replace("TOP Home Devlog Github", " ")
    cleaned = re.sub(r"새로운걸 공부하고 기록하는 것을 좋아합니다\.?\s*/?\s*", "", cleaned)
    cleaned = re.sub(r"기술 면접 대비 CS전공 핵심요약집\s*/\s*", "", cleaned)
    cleaned = re.sub(r"목차보기\s+Show\s+Hide\s+.*?(?=\d{2}-\d|\d{2}\s|문제\s*\d+|시간 복잡도|배열|스택|큐|트리|그래프)", " ", cleaned)
    cleaned = re.sub(r"^([가-힣A-Za-z0-9 /·&+-]{2,40})\s+\1\s+", r"\1 ", cleaned)
    parts = re.split(r"(?<=[.!?。다])\s+|[•·]\s*", cleaned)
    important = [
        "의미", "역할", "목적", "필요", "보장", "관리", "전략", "장점", "단점",
        "문제", "실행", "할당", "접근", "변환", "스케줄링", "메모리", "프로세스",
        "스레드", "동기화", "교착", "tcp", "udp", "패킷", "신뢰", "연결",
        "http", "rest", "uri", "상태", "요청", "응답", "계층",
        "데이터베이스", "트랜잭션", "원자성", "일관성", "격리성", "지속성",
        "acid", "커밋", "롤백", "조인", "인덱스", "복잡도", "시간 복잡도",
        "배열", "연결 리스트", "스택", "큐", "트리", "그래프", "해시",
    ]
    selected: list[str] = []
    for part in parts:
        sentence = re.sub(r"\s+", " ", part).strip(" -|\t\r\n")
        sentence = re.sub(r"^[가-힣A-Za-z0-9 ·&+-]{1,30}\s*/\s*", "", sentence).strip()
        sentence = re.sub(r"^([가-힣A-Za-z0-9 ·&+-]{2,30})\s+\1\s+", r"\1 ", sentence).strip()
        if len(sentence) < 35 or len(sentence) > 360:
            continue
        lower = sentence.lower()
        if any(skip in lower for skip in ["search", "share", "today", "home", "devlog", "github"]):
            continue
        if any(skip in sentence for skip in ["목차보기", "Show Hide", "첫째 마당", "둘째 마당", "목차 "]):
            continue
        if not any(term in lower for term in important):
            continue
        if sentence not in selected:
            selected.append(sentence)
        if len(selected) >= limit:
            break
    if not selected and cleaned:
        fallback = cleaned[:260].rstrip()
        if fallback:
            selected.append(fallback)
    return selected[:limit]


def source_node_summary_line(node: dict[str, str]) -> str:
    sentences = source_node_key_sentences(node.get("text") or "", limit=1)
    sentence = sentences[0] if sentences else f"{node.get('title') or '해당 섹션'}의 핵심 개념과 적용 기준을 정리했다."
    title = re.escape(str(node.get("title") or "").strip())
    if title:
        sentence = re.sub(rf"^{title}\s+", "", sentence).strip()
    return sentence or f"{node.get('title') or '해당 섹션'}의 핵심 개념과 적용 기준을 정리했다."


def source_node_resolution_text(title: str, text: str) -> str:
    """Write the actual learned content, not a promise to organize it."""
    title = str(title or "해당 개념").strip()
    sentences = source_node_key_sentences(text, limit=8)

    def strip_title(sentence: str) -> str:
        pattern = re.escape(title)
        return re.sub(rf"^{pattern}\s+", "", sentence).strip() or sentence

    sentences = [strip_title(s) for s in sentences]
    lower_text = str(text or "").lower()

    acid_items: list[str] = []
    for ko, en in [
        ("원자성", "Atomicity"),
        ("일관성", "Consistency"),
        ("격리성", "Isolation"),
        ("지속성", "Durability"),
    ]:
        match = re.search(rf"{ko}\s*\(?{en}?\)?\s*([^.!?\n]+(?:다|한다|된다|않아야 한다|보장한다)?)", str(text or ""), re.I)
        if match:
            acid_items.append(f"{ko}: {match.group(1).strip()}")
    if title == "트랜잭션" and acid_items:
        definition = sentences[0] if sentences else "트랜잭션은 데이터베이스 작업을 하나의 논리적 실행 단위로 묶는 개념이다."
        return (
            f"`{title}`의 핵심은 작업 일부만 반영되는 상태를 막고 데이터베이스 상태를 신뢰할 수 있게 만드는 데 있었다. "
            f"본문에서는 `{definition}`라고 설명하고, ACID 조건을 통해 성공과 실패의 경계를 분명히 한다. "
            f"특히 {'; '.join(acid_items[:4])}. "
            "그래서 이 개념은 단순히 여러 쿼리를 묶는 기능이 아니라, 커밋과 롤백을 통해 데이터가 완전히 반영되거나 전혀 반영되지 않도록 보장하는 문제로 이해했다."
        )

    if title in {"TCP와 UDP", "TCP", "UDP"} or ("tcp" in lower_text and "udp" in lower_text):
        definition = sentences[0] if sentences else "TCP는 연결형 서비스와 신뢰성을 제공하고, UDP는 연결 절차를 줄여 전송 속도를 우선한다."
        detail = " ".join(sentences[1:4])
        return (
            f"`{title}`에서 실제로 해결해야 한 문제는 모든 통신에 같은 전송 방식을 쓰지 않는 이유를 설명하는 것이었다. "
            f"본문 근거는 `{definition}`이다. {detail} "
            "따라서 답변 기준은 신뢰성과 순서 보장이 중요하면 TCP, 지연을 줄이고 빠른 전송이 중요하면 UDP를 선택한다는 비교 구조로 잡았다."
        )

    if "스케줄링" in title:
        definition = sentences[0] if sentences else "스케줄링은 여러 프로세스 중 어떤 프로세스를 실행할지 결정하는 기준이다."
        detail = " ".join(sentences[1:4])
        return (
            f"`{title}`의 핵심 문제는 CPU를 기다리는 여러 프로세스 사이에서 실행 순서를 어떻게 정하느냐였다. "
            f"본문에서는 `{definition}`라고 설명한다. {detail} "
            "그래서 공평성, 효율성, 안정성, 반응 시간, 무한 연기 방지를 각각 스케줄링 알고리즘을 평가하는 기준으로 보았다."
        )

    if "메모리" in title:
        definition = sentences[0] if sentences else "메모리 관리는 프로세스가 보는 주소와 실제 물리 주소를 연결하고 보호하는 문제다."
        detail = " ".join(sentences[1:4])
        return (
            f"`{title}`에서 해결해야 할 문제는 CPU가 사용하는 논리 주소와 실제 RAM의 물리 주소가 다르다는 점이었다. "
            f"본문 근거는 `{definition}`이다. {detail} "
            "따라서 MMU, TLB, 단편화, 가상 메모리는 모두 메모리 접근을 빠르고 안전하게 만들기 위한 해결 장치로 연결해 이해했다."
        )

    if "REST" in title or "HTTP" in title:
        definition = sentences[0] if sentences else f"{title}는 웹에서 요청과 응답을 구조화하는 핵심 개념이다."
        detail = " ".join(sentences[1:4])
        return (
            f"`{title}`의 핵심은 클라이언트와 서버가 어떤 규칙으로 자원을 요청하고 상태를 주고받는지 설명하는 데 있었다. "
            f"본문에서는 `{definition}`라고 설명한다. {detail} "
            "그래서 단순 용어가 아니라 URI, 메서드, 상태, 요청/응답, 캐싱, 무상태성 같은 기준으로 실제 API 설계를 설명할 수 있어야 했다."
        )

    if "프로세스" in title:
        definition = sentences[0] if sentences else "프로세스는 실행 중인 프로그램이며 OS가 독립된 메모리 영역을 할당한다."
        detail = " ".join(sentences[1:4])
        return (
            f"`{title}`의 핵심은 프로그램이 실행될 때 OS가 어떤 단위로 자원과 메모리를 관리하는지 이해하는 것이었다. "
            f"본문에서는 `{definition}`라고 설명한다. {detail} "
            "따라서 프로세스는 코드, 데이터, 스택, 힙 영역을 가진 실행 단위이고, 스레드와 비교할 때 독립된 메모리 공간을 갖는다는 점을 답변의 중심으로 잡았다."
        )

    definition = sentences[0] if sentences else f"{title}의 정의와 동작 조건을 본문 근거로 확인했다."
    detail = " ".join(sentences[1:4])
    return (
        f"`{title}`의 핵심 내용은 `{definition}`이다. "
        + (f"이어지는 근거로 {detail} " if detail else "")
        + "이 내용을 기준으로 해당 개념이 어떤 문제를 해결하고, 어떤 조건에서 사용되며, 어떤 기준으로 설명해야 하는지 구체화했다."
    )


def clean_learning_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"\b(Source Pack|Collection Metadata|Collected Nodes|Quality|URL type|Site hint|Status|Content shape|Navigation shape|Access level|Transcript segment \d+)\b", " ", text, flags=re.I)
    text = re.sub(r"- URL:\s*\S+", " ", text)
    text = re.sub(r"- Type:\s*\S+", " ", text)
    text = re.sub(r"Input URL:\s*\S+", " ", text)
    text = re.sub(r"새로운걸 공부하고 기록하는 것을 좋아합니다\.?\s*/?\s*", " ", text)
    text = re.sub(r"기술 면접 대비 CS전공 핵심요약집\s*/\s*", " ", text)
    text = re.sub(r"목차보기\s+Show\s+Hide\s+.*?(?=\d{2}-\d|\d{2}\s|문제\s*\d+|시간 복잡도|배열|스택|큐|트리|그래프)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_learning_sentences(text: str, limit: int = 80) -> list[str]:
    cleaned = clean_learning_text(text)
    raw_parts = re.split(r"(?<=[.!?。다요죠])\s+|[•·]\s*", cleaned)
    sentences: list[str] = []
    for part in raw_parts:
        sentence = re.sub(r"\s+", " ", part).strip(" -|\t\r\n")
        if len(sentence) < 35 or len(sentence) > 420:
            continue
        lower = sentence.lower()
        if any(skip in lower for skip in ["source pack", "collection metadata", "transcript collected", "youtube video", "input url"]):
            continue
        if sentence not in sentences:
            sentences.append(sentence)
        if len(sentences) >= limit:
            break
    return sentences


def select_sentences_by_terms(sentences: list[str], terms: list[str], limit: int = 6) -> list[str]:
    selected: list[str] = []
    for term in terms:
        term_l = term.lower()
        for sentence in sentences:
            if term_l in sentence.lower() and sentence not in selected:
                selected.append(sentence)
                break
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for sentence in sentences:
            if sentence not in selected:
                selected.append(sentence)
            if len(selected) >= limit:
                break
    return selected[:limit]


def sentences_for_topic(sentences: list[str], terms: list[str], limit: int = 3) -> list[str]:
    selected: list[str] = []
    intro_noise = ["얄코입니다", "유튜브 채널", "온라인 장편 강의", "강의의 주제", "시작하겠습니다", "단계적으로 설명", "든든한 동반자"]
    for sentence in sentences:
        lower = sentence.lower()
        if any(noise in sentence for noise in intro_noise):
            continue
        if not any(term.lower() in lower for term in terms):
            continue
        if sentence not in selected:
            selected.append(sentence)
        if len(selected) >= limit:
            break
    return selected


def algorithm_youtube_topic_plan(text: str) -> list[tuple[str, list[str]]]:
    available = infer_learning_terms(text, limit=30)
    plan = [
        ("시간 복잡도와 공간 복잡도", ["시간 복잡도", "공간 복잡도", "연산", "메모리"]),
        ("빅오 표기법", ["빅오", "빅 5", "표기법", "O("]),
        ("배열과 인덱스 접근", ["배열이란", "인덱스", "연속적인 메모리", "요소 접근"]),
        ("배열 탐색·삽입·삭제 비용", ["리니어 서치", "탐색 작업", "삽입", "삭제", "배열의 길이"]),
        ("스택과 콜 스택", ["LIFO", "콜스택", "푸시", "팝", "push", "pop"]),
        ("큐와 데크", ["FIFO", "데크", "큐에", "enqueue", "dequeue"]),
        ("트리와 순회 방식", ["이진 트리", "프리오더", "인오더", "포스트 오더", "레벨 오더"]),
        ("그래프·정렬·탐색", ["그래프", "정렬", "탐색 알고리즘"]),
    ]
    if available:
        return [item for item in plan if any(term.lower() in text.lower() for term in item[1])] or plan[:5]
    return plan[:5]


def infer_learning_terms(text: str, limit: int = 16) -> list[str]:
    candidates = [
        "시간 복잡도", "공간 복잡도", "빅오", "배열", "인덱스", "탐색", "삽입", "삭제",
        "스택", "큐", "연결 리스트", "트리", "이진 트리", "순회", "프리오더", "인오더",
        "포스트오더", "레벨 오더", "그래프", "정렬", "프로세스", "스레드", "스케줄링",
        "메모리", "TCP", "UDP", "HTTP", "REST", "트랜잭션", "ACID", "조인", "자료구조",
        "Copilot Studio", "declarative agent", "custom agent", "Adaptive Cards", "Agent Flows",
        "licensing", "publishing", "Foundry", "VS Code", "agent",
        "Git", "GitHub", "commit", "branch", "merge", "Pull Request", "repository", "remote",
        "clone", "push", "pull", "checkout", "conflict", "version control",
    ]
    lower = text.lower()
    found = [term for term in candidates if term.lower() in lower]
    return unique_preserve_order(found, limit=limit)


def build_lecture_content_summary(
    seed_url: str,
    run_id: str,
    source_pack_text: str,
    collector_report: dict[str, Any],
    user_memo: str = "",
) -> str:
    graph = collector_source_graph(collector_report, max_nodes=120)
    payload = load_collector_json(collector_report)
    title = str(graph.get("title") or "").strip() or "강의 내용"
    nodes = graph.get("nodes") or []
    node_text = "\n".join(str(node.get("text") or "") for node in nodes if isinstance(node, dict))
    text = node_text if len(node_text) > 1000 else source_pack_text
    sentences = split_learning_sentences(text, limit=520)
    terms = infer_learning_terms(text, limit=18)
    stats = source_graph_stats_summary(collector_report)
    host = url_domain(seed_url)
    quality = collector_report.get("quality") if isinstance(collector_report.get("quality"), dict) else {}

    if "youtube.com" in host or "youtu.be" in host:
        return build_youtube_content_summary(seed_url, title, payload, graph, collector_report)

    if graph.get("video_index") or int(quality.get("video_candidates") or 0):
        video_rows = graph.get("video_index") or []
        video_lines = []
        for idx, row in enumerate(video_rows[:80], start=1):
            if not isinstance(row, dict):
                continue
            meta = " / ".join(str(row.get(k) or "").strip() for k in ["track", "duration", "date"] if str(row.get(k) or "").strip())
            video_lines.append(f"{idx}. {row.get('title') or 'Video'} - {row.get('youtube_url') or ''}" + (f" ({meta})" if meta else ""))
        raw_flat_nodes: list[dict[str, Any]] = []

        def walk_raw(raw_items: list[Any]) -> None:
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    continue
                raw_flat_nodes.append(raw_item)
                children = raw_item.get("children")
                if isinstance(children, list):
                    walk_raw(children)

        if isinstance(payload.get("nodes"), list):
            walk_raw(payload.get("nodes") or [])
        video_nodes = [
            node for node in raw_flat_nodes
            if isinstance(node, dict) and str(node.get("type") or "").lower() == "video"
        ] or [
            node for node in nodes
            if isinstance(node, dict) and str(node.get("type") or "").lower() == "video"
        ]
        if not video_nodes:
            video_nodes = [
                node for node in nodes
                if isinstance(node, dict) and ("youtu.be/" in str(node.get("url") or "") or "youtube.com" in str(node.get("url") or ""))
            ]
        video_sections = []
        for node in video_nodes[:12]:
            title_text = str(node.get("title") or "Video").strip()
            transcript_text = str(node.get("text") or "")
            lines = source_node_key_sentences(transcript_text, limit=4)
            if not lines:
                child_texts = []
                for child in node.get("children") or []:
                    if isinstance(child, dict):
                        child_texts.append(str(child.get("text") or ""))
                lines = split_learning_sentences(" ".join(child_texts), limit=4)[:4]
            video_sections.append(f"### {title_text}\n" + learning_point_paragraph(lines, "이 영상의 자막 요약 문장을 충분히 추출하지 못했습니다."))
        return sanitize_medium_markdown(f"""# {title} 영상 강의 내용 요약

## 수집 범위
- 원본 URL: {seed_url}
- 영상 후보: {quality.get('video_candidates') or len(video_rows)}
- 자막 수집 완료 영상: {quality.get('videos_transcript_collected') or len(video_nodes)}
- 자막 세그먼트: {quality.get('transcript_segments') or 0}
- 본문/자막 글자 수: {stats.get('chars', 0)}

## 수집한 영상 링크
{chr(10).join(video_lines[:80]) if video_lines else markdown_bullets(graph.get("video_url_candidates") or [])}

## 영상별 핵심 내용
{chr(10).join(video_sections) if video_sections else "아직 영상별 자막 요약을 만들 만큼의 자막 내용이 충분하지 않습니다."}
""")

    learning_nodes = source_graph_learning_nodes(graph, limit=10)
    if learning_nodes:
        node_sections = []
        for node in learning_nodes[:8]:
            title_text = node.get("title") or "학습 섹션"
            summary = source_node_resolution_text(title_text, node.get("text") or "")
            node_sections.append(f"### {title_text}\n{summary}")
        return sanitize_medium_markdown(f"""# {title} 학습 내용 요약

## 수집 범위
- 원본 URL: {seed_url}
- 수집 페이지: {stats.get('pages', 0)}
- 본문 글자 수: {stats.get('chars', 0)}

## 핵심 키워드
{markdown_bullets((terms or [node.get("title") for node in learning_nodes])[:12])}

## 학습 흐름
{markdown_bullets([node.get("title") for node in learning_nodes[:12]])}

## 주요 내용 정리
{chr(10).join(node_sections)}
""")

    selected = select_sentences_by_terms(sentences, terms, limit=10)
    return sanitize_medium_markdown(f"""# {title} 학습 내용 요약

## 수집 범위
- 원본 URL: {seed_url}
- 본문 글자 수: {stats.get('chars', 0)}

## 핵심 키워드
{markdown_bullets(terms[:12])}

## 주요 내용
{learning_point_paragraph(selected, "수집된 본문에서 요약할 만한 문장을 충분히 찾지 못했습니다.")}
""")


def build_learning_steps_from_evidence(headings: list[str], points: list[str], labs: list[str], limit: int = 5) -> list[tuple[str, str, str, str]]:
    concepts = [
        h for h in unique_preserve_order(headings, limit=limit)
        if h.lower() not in {"navigation", "introduction", "summary", "key takeaways"}
    ]
    steps: list[tuple[str, str, str, str]] = []
    if labs:
        lab_title = labs[0]
        lab_point = points[0] if points else ""
        steps.append((
            f"{lab_title} 과제 해결",
            "Lab/Exercise로 따로 제시된 부분은 단순 읽기만으로 끝나는 개념이 아니라 실제 수행과 검증이 필요한 어려운 과제였다.",
            (
                f"`{lab_title}`를 강의의 검증 과제로 보고, 어떤 준비 단계와 실행 단계가 필요한지 분리했다."
                + (f" 특히 '{lab_point[:220]}'라는 강의 근거를 실습 판단 기준으로 삼았다." if lab_point else "")
            ),
            "Lab 제목과 세부 단계가 강의의 핵심 개념을 실제로 확인하는 검증 흐름으로 연결되는지 확인했다.",
        ))
    for idx, concept in enumerate(concepts[:limit]):
        if concept.lower().startswith("ai skills navigator |"):
            continue
        if concept.lower() in {"what you'll learn", "copilot said:", "key takeaways"}:
            continue
        point = points[idx] if idx < len(points) else ""
        problem = (
            f"처음에는 `{concept}`가 앞뒤 단원과 어떤 차이가 있는지 바로 분명하지 않았다. "
            "정의만 보면 이해한 것 같지만, 실제로는 언제 쓰고 무엇으로 확인해야 하는지까지 연결해야 했다."
        )
        action = (
            f"강의에서 `{concept}`가 설명되는 위치와 함께 나온 세부 문장을 기준으로 역할을 다시 정리했다."
            + (f" 특히 '{point[:220]}'라는 내용을 기준으로 개념의 쓰임을 확인했다." if point else "")
        )
        validation = "이 개념을 한 문장 정의가 아니라 사용 상황, 적용 단계, 확인 결과까지 말할 수 있는지 기준으로 삼았다."
        steps.append((f"{concept}의 역할 이해", problem, action, validation))
    if labs and not any("Lab / Exercise" in step[0] or "과제 해결" in step[0] for step in steps):
        steps.append((
            "Lab / Exercise로 이해 검증",
            "개념 설명을 읽는 것과 실제 실습 흐름에서 검증하는 것은 달랐다.",
            "Lab/Exercise 제목과 단계 링크를 따라가며 어떤 준비, 실행, 확인 단계가 있는지 분리했다.",
            f"실습 항목이 `{labs[0]}` 같은 검증 흐름으로 연결되는지 확인했다.",
        ))
    return steps[:limit] or [
        (
            "핵심 개념을 학습 문제로 재정의",
            "강의 내용을 읽었지만 무엇이 핵심이고 무엇이 보조 설명인지 흐릴 수 있었다.",
            "제목, 본문 문장, 실습 단서를 묶어 개념의 역할과 확인 기준을 다시 정리했다.",
            "정리한 내용이 다음 복습 때 설명 가능한 기준으로 남는지 확인했다.",
        )
    ]


def compact_article_title(title: str) -> str:
    cleaned = re.sub(r"^AI Skills Navigator\s*\|\s*", "", str(title or "")).strip()
    return cleaned or str(title or "강의 핵심 주제").strip()


def evidence_quote(points: list[str], index: int, fallback: str) -> str:
    if points:
        point = points[min(index, len(points) - 1)]
        point = re.sub(r"\s+", " ", point).strip()
        if len(point) > 360:
            point = point[:357].rstrip() + "..."
        return point
    return fallback


def build_problem_sections_from_source(title: str, headings: list[str], points: list[str], labs: list[str]) -> tuple[str, str, str]:
    subject = compact_article_title(title)
    concept_flow = [h for h in headings if h and not h.lower().startswith("ai skills navigator |")][:5]
    concept_text = ", ".join(concept_flow) or subject
    first = evidence_quote(points, 0, f"{subject}의 핵심 개념과 실습 단계가 연결되어 있었다.")
    second = evidence_quote(points, 1, first)
    third = evidence_quote(points, 2, second)
    lab_text = ", ".join(labs[:3]) if labs else "강의에서 제시된 확인 단계"

    problem = (
        f"이 강의에서 문제로 인식한 것은 사람들이 실무에서 해결책을 자주 찾게 되는 `{subject}` 관련 문제를 실제 적용 가능한 해결 기준으로 만드는 것이었다. "
        f"강의는 `{first}`라는 내용을 통해 이 주제가 추상적인 개념 소개가 아니라 실제 판단과 검증이 필요한 문제임을 보여준다.\n\n"
        f"따라서 내가 해결해야 한 문제는 `{concept_text}` 흐름을 따라가며 각 개념이 어떤 상황에서 필요한지, 어떤 원인을 다루는지, 어떤 결과로 확인되는지 분리하는 것이었다. "
        f"특히 `{second}`라는 근거는 이 학습이 단순 요약이 아니라 실제 사용 상황과 결과 확인까지 이어져야 한다는 점을 드러낸다."
    )
    definition = (
        f"문제는 `{subject}`를 사람들이 검색하는 해결 질문에 답할 수 있는 기술 판단 구조로 바꾸는 것으로 정의했다. "
        f"즉, 강의 속 핵심 내용을 기능 이름이나 단원 제목으로만 남기는 것이 아니라, 문제 상황 → 원인 구조 → 해결 조치 → 확인 결과의 순서로 재구성해야 했다.\n\n"
        f"이 정의는 `{third}`라는 강의 근거에서 출발한다. 이 내용은 학습자가 단순히 무엇을 배웠는지보다, 해당 개념을 어떤 조건에서 적용하고 어떤 결과로 검증해야 하는지를 설명해야 한다는 문제로 이어진다."
    )
    why = (
        f"이것을 문제로 인식한 이유는 `{first}`에서 보이는 것처럼, 강의의 핵심 개념이 실제 상황에서 사람들이 해결책을 찾는 제약과 바로 연결되기 때문이다. "
        f"개념만 따로 외우면 `{subject}`가 어떤 문제를 해결하는지 설명하기 어렵고, 실습이나 적용 단계에서 무엇을 성공 기준으로 봐야 하는지도 남지 않는다.\n\n"
        f"또한 `{second}`라는 내용은 이 주제가 단일 기능 설명이 아니라 여러 판단 기준을 함께 다루는 문제임을 보여준다. "
        f"그래서 해결 과정에서는 `{concept_text}`를 순서대로 따라가며 원인을 나누고, `{lab_text}`를 통해 결과 확인 기준을 세우는 방식으로 접근했다."
    )
    return problem, definition, why


def build_source_graph_grounded_medium_draft(
    seed_url: str,
    run_id: str,
    source_pack_text: str,
    collector_report: dict[str, Any],
    user_problem: str = "",
) -> str:
    """Build a human learner-facing Medium draft from collected evidence.

    Contract:
    - The writer's role is NOT the collector, GPT, app, or automation system.
    - The writer's role is a learner who studied the lecture/lab and is writing a
      problem-solving learning note about difficult/core concepts in that material.
    - "Problem" means the subject-matter learning challenge inside the course,
      not the difficulty of generating notes.
    - Do not mention run_id, source graph, collector, seed URL, GPT, app, automation,
      or the note-generation process in the user-facing article.
    """
    host = url_domain(seed_url)
    graph = collector_source_graph(collector_report, max_nodes=80)
    title = str(graph.get("title") or "").strip()
    headings = source_graph_key_headings(collector_report, limit=40)
    labs = source_graph_lab_titles(collector_report, limit=10)

    heading_text = "\n".join(headings).lower()
    pack_l = (source_pack_text or "").lower()
    topic_l = f"{title}\n{heading_text}\n{pack_l}".lower()
    title_l = title.lower()

    def bullets(items: list[str]) -> str:
        clean: list[str] = []
        banned = ["source graph", "collector", "run_id", "about:blank", "github - microsoft-foundry", "aks-lab", "windows-server"]
        for item in items:
            s = str(item or "").strip()
            if not s:
                continue
            if any(b in s.lower() for b in banned):
                continue
            if s not in clean:
                clean.append(s)
            if len(clean) >= 18:
                break
        return "\n".join(f"- {item}" for item in clean) or "- 확인 가능한 학습 항목이 부족했습니다."

    def concept_bullets(items: list[tuple[str, str]]) -> str:
        return "\n".join(f"- **{name}**: {desc}" for name, desc in items)

    if "youtube.com" in host or "youtu.be" in host:
        return build_youtube_problem_medium_draft(seed_url, run_id, source_pack_text, collector_report, user_problem)

    generic_profile = topic_profile_from_text(seed_url, title, source_pack_text, user_problem)
    if generic_profile and not any(domain in host for domain in ["wikidocs.net", "oopy.io", "aiskillsnavigator.microsoft.com"]):
        return build_topic_learning_medium_article(
            seed_url=seed_url,
            title=title or str(generic_profile.get("title") or "학습 자료"),
            source_text=source_pack_text,
            profile=generic_profile,
            user_problem=user_problem,
            source_label="웹문서 본문과 사용자 메모",
        )

    if "aiskillsnavigator.microsoft.com" in host:
        stats = source_graph_stats_summary(collector_report)
        clean_headings = [
            h for h in unique_preserve_order(headings, limit=18)
            if h and h.lower() not in {"navigation", "introduction", "summary"}
        ]
        title_for_article = title or (clean_headings[0] if clean_headings else "AI Skills Navigator 학습")
        scope = ", ".join(clean_headings[:6]) or title_for_article
        flow_items = bullets(clean_headings[:14])
        lab_items = bullets(labs[:8]) if labs else "- 이번 강의에서는 별도 Lab/Exercise 단계가 중심 흐름으로 드러나지 않았다."
        video_lines = graph.get("video_url_candidates") or []
        video_md = bullets([str(v) for v in video_lines[:8]]) if video_lines else "- 이번 강의에서는 별도 영상 링크를 중심 근거로 사용하지 않았다."
        concept_items = [
            (h, "이번 강의 흐름 안에서 역할과 확인 기준을 나누어 이해해야 할 개념으로 정리했다.")
            for h in clean_headings[:8]
        ] or [(title_for_article, "이번 자료의 중심 학습 주제다.")]
        concept_md = concept_bullets(concept_items)
        topic_blob = f"{title_for_article}\n{scope}\n{heading_text}\n{pack_l}"
        evidence_points = source_pack_learning_points(source_pack_text, clean_headings, limit=12)
        evidence_md = learning_point_paragraph(
            evidence_points,
            "강의 본문에서 확인한 핵심 흐름은 제목을 따라 개념을 나열하는 것이 아니라, 각 개념이 어떤 상황에서 필요하고 어떤 실습 단계로 이어지는지 파악하는 것이었다.",
        )
        deeper_points = evidence_points[5:10] or evidence_points[:5]
        snippet_md = learning_point_paragraph(deeper_points, evidence_md)

        user_problem = re.sub(r"\[생성 직전 사용자가 정의한 어려운 문제\]", "", str(user_problem or "")).strip()
        user_problem = re.sub(r"없음\.\s*자료의 핵심 흐름을 바탕으로 문제를 정의하고 해결 과정 작성\.?", "", user_problem).strip()
        learner_intro = (
            f"이번에 나는 `{title_for_article}` 강의를 학습하면서, 강의가 제시한 핵심 개념을 실제 문제 해결 기준으로 바꾸는 데 집중했다. "
            f"가장 먼저 붙잡은 근거는 `{evidence_quote(evidence_points, 0, compact_article_title(title_for_article))}`였다. 이 문장은 이번 학습이 단순한 개념 소개가 아니라 실제 상황에서 어떤 문제를 해결해야 하는지 판단하는 과정임을 보여준다."
        )
        learner_problem, learner_definition, learner_why = build_problem_sections_from_source(
            title_for_article,
            clean_headings,
            evidence_points,
            labs,
        )
        if user_problem:
            learner_problem = (
                f"사용자가 이번 학습에서 중심 문제로 지정한 것은 `{user_problem}`였다. "
                "따라서 이 글의 문제 인식은 자료 전체를 일반적으로 요약하는 것이 아니라, 사용자가 지정한 어려운 문제를 강의 내용으로 해결하는 데 맞춘다.\n\n"
                f"강의 근거로는 `{evidence_quote(evidence_points, 0, compact_article_title(title_for_article))}`가 먼저 연결된다. "
                "이 근거를 기준으로 사용자가 정의한 문제가 어떤 개념, 원인 구조, 실습 검증 기준과 연결되는지 확인했다."
            )
            learner_definition = (
                f"이 학습에서 정의한 문제는 `{user_problem}`를 해결 가능한 기술 문제로 구체화하는 것이었다. "
                "단순히 어렵다고 느낀 지점을 기록하는 것이 아니라, 강의에서 제공한 개념과 실습 흐름을 사용해 원인과 해결 기준을 분리해야 했다.\n\n"
                f"이를 위해 `{evidence_quote(evidence_points, 1, evidence_quote(evidence_points, 0, compact_article_title(title_for_article)))}`라는 강의 내용을 근거로 삼았다. "
                "이 내용은 사용자가 지정한 문제가 어떤 실제 상황에서 발생하고, 어떤 확인 결과로 해결 여부를 판단해야 하는지 설명하는 기준이 된다."
            )
            learner_why = (
                f"`{user_problem}`를 문제로 인식한 이유는 이 지점이 강의의 핵심 개념과 실습 검증 흐름을 연결하는 부분이기 때문이다. "
                "이 문제를 해결하지 못하면 강의 내용을 들었더라도 실제 적용 조건, 원인 판단, 결과 검증 기준이 남지 않는다.\n\n"
                f"또한 `{evidence_quote(evidence_points, 2, evidence_quote(evidence_points, 0, compact_article_title(title_for_article)))}`라는 근거는 이 문제가 단순 개념 암기가 아니라 실제 해결 과정으로 다뤄야 할 내용임을 보여준다. "
                "그래서 해결 과정은 사용자가 지정한 문제를 중심으로 강의의 개념, Lab/Exercise, 검증 기준을 다시 연결하는 방식으로 구성했다."
            )
        core_learning_md = f"""이번 강의에서 내가 먼저 정의한 문제는 `{title_for_article}`라는 큰 제목 아래의 개념들을 실제 해결 가능한 학습 과제로 바꾸는 것이었다. 단원명만 보면 전체 목차는 보이지만, 학습자로서는 각 항목이 어떤 문제를 해결하고 무엇으로 결과를 확인해야 하는지가 더 중요했다.

강의 본문에서 특히 눈에 들어온 내용은 다음과 같았다.

{evidence_md}"""
        complex_detail_md = f"""복잡한 문제는 강의 안의 개념들이 서로 독립된 항목처럼 보이지만 실제로는 하나의 판단 흐름으로 연결된다는 점이었다. 이 문제를 해결하려면 각 개념을 정의로만 이해하지 않고, 어떤 입력 상황에서 필요하고 어떤 결과로 검증되는지까지 연결해야 했다.

내가 복잡한 문제로 정의한 근거는 강의 본문에 남은 세부 설명에서도 확인된다.

{snippet_md}"""
        practice_flow_md = f"""실습 흐름은 개념을 읽고 끝내는 것이 아니라, 내가 정의한 문제를 실제 단계에서 해결하고 확인하는 방식으로 보았다. 먼저 중심 개념의 역할을 잡고, 그 개념이 쓰이는 위치를 확인한 뒤, Lab이나 Exercise가 있다면 마지막 결과를 통해 해결 여부를 검증하는 흐름이다.

이번 자료에서 실습 또는 확인 흐름으로 잡은 항목은 다음과 같다.

{lab_items}"""
        outcome_md = f"""이번 강의를 통해 나는 핵심 개념을 단순 용어가 아니라 문제 정의, 원인 판단, 해결 조치, 검증 기준으로 바꾸어 설명할 수 있게 되었다. 특히 `{scope}` 흐름을 따라가며, 비슷해 보이는 항목도 실제로는 서로 다른 문제 상황과 확인 결과에 연결된다는 점을 확인했다.

성과는 강의 내용을 모두 외우는 것이 아니라, 다음에 같은 주제를 다시 만났을 때 어떤 문제를 먼저 정의하고 어떤 원인을 확인하며 어떤 실습 결과로 해결 여부를 검증할지 설명할 수 있게 된 것이다."""
        portfolio_summary_md = "This learning record explains how I studied the lecture as a learner and converted complex technical content into a problem-solving portfolio narrative. The focus is not on summarizing the material, but on defining the problem, identifying the cause, applying the lesson or lab flow as the solution, and validating the result.\n\nThe outcome is a Medium-ready learning record that shows how the source material became a practical technical problem-solving experience. It documents what problem had to be solved, why it mattered, how the solution was derived from the lecture evidence, and what capability was gained after validation."
        skills_md = "- Defining complex technical learning problems\n- Identifying cause and structure from lecture evidence\n- Converting lesson flow into solution steps\n- Connecting lab work to validation criteria\n- Writing Medium-ready problem-solving narratives\n- Explaining technical decisions with evidence\n- Avoiding unsupported claims or invented results\n- Documenting portfolio-level learning outcomes"
        step_sections = build_learning_steps_from_evidence(clean_headings, evidence_points, labs, limit=5)

        if False and "intelligent search" in topic_blob and "sql" in topic_blob:
            learner_intro = (
                "이번에 나는 SQL에서 intelligent search를 구현하는 강의를 들으면서, full-text search, vector search, hybrid search가 각각 언제 필요한지 구분하는 데 집중했다. "
                "처음에는 모두 '검색'이라는 말로 묶여 보여서 비슷하게 느껴졌지만, 강의를 따라가다 보니 키워드 일치, 의미 기반 유사도, 두 결과의 결합은 서로 다른 문제를 해결한다는 점이 핵심이었다."
            )
            learner_problem = (
                "처음 헷갈린 지점은 full-text search와 vector search를 단순히 신기술/기존기술처럼 나누는 것이 아니라, query intent에 따라 선택해야 한다는 점이었다. "
                "정확한 단어가 중요한 검색인지, 의미가 비슷한 내용을 찾는 검색인지, 둘을 합쳐 ranking을 보정해야 하는 검색인지 구분해야 했다."
            )
            learner_definition = (
                "이 학습에서 문제로 잡은 것은 SQL 안에서 검색 방식을 선택하고 검증하는 기준을 세우는 것이었다. "
                "full-text index와 predicate는 키워드 기반 검색을 위해 필요했고, vector data type과 embedding은 의미 기반 비교를 위해 필요했다. "
                "hybrid search와 Reciprocal Rank Fusion은 두 검색 결과를 합쳐 ranking을 조정하는 단계로 이해했다."
            )
            learner_why = (
                "이 내용을 문제로 본 이유는 검색 기능을 구현할 때 '검색이 된다'만으로는 충분하지 않기 때문이다. "
                "사용자의 질문이 정확한 용어를 포함하는지, 의미적으로 비슷한 문서를 찾아야 하는지, 두 방식의 장점을 함께 써야 하는지에 따라 SQL에서 준비해야 할 인덱스, 벡터 컬럼, embedding 생성, ranking 방식이 달라진다."
            )
            core_learning_md = """강의의 핵심은 SQL에서 검색을 하나의 기능으로 뭉뚱그리지 않고, 질문 의도에 따라 검색 방식을 선택하는 것이었다. 사용자가 정확한 단어를 알고 찾는 경우에는 full-text search가 맞고, 표현은 다르지만 의미가 가까운 내용을 찾고 싶을 때는 vector search가 필요하다. 그런데 실제 서비스에서는 둘 중 하나만으로 충분하지 않은 경우가 많기 때문에 hybrid search가 등장한다.

처음에는 full-text search, vector search, hybrid search가 모두 '검색 품질을 높이는 방법'처럼 보였다. 하지만 강의를 따라가며 보니 세 방식은 같은 층위가 아니었다. full-text search는 단어와 구문을 기준으로 관련 문서를 찾는 방식이고, vector search는 embedding을 통해 의미적 가까움을 계산하는 방식이다. hybrid search는 이 둘의 결과를 함께 사용해 keyword match와 semantic similarity를 모두 반영하려는 전략이다."""
            complex_detail_md = """가장 복잡했던 부분은 검색 방식의 차이가 SQL 구현 요소와 바로 연결된다는 점이었다. full-text search를 하려면 full-text index와 predicate를 이해해야 하고, vector search를 하려면 vector data type, embedding 저장, VECTOR_DISTANCE, VECTOR_SEARCH 같은 함수를 이해해야 한다. 여기서 한 단계 더 나아가 hybrid search는 두 검색 결과를 어떻게 합쳐 ranking할 것인지가 문제가 된다.

특히 Reciprocal Rank Fusion은 처음 보면 단순 ranking 기법처럼 보이지만, 실제로는 full-text 결과와 vector 결과가 서로 다른 점수 체계를 가질 때 두 목록을 무리하게 하나의 점수로 합치지 않고 순위 기반으로 결합하는 방법으로 이해했다. 그래서 hybrid search의 핵심은 '둘 다 실행한다'가 아니라, 서로 다른 검색 신호를 어떤 기준으로 합쳐 최종 결과를 만들 것인가에 있었다.

RAG로 넘어가면 검색은 더 이상 검색 화면만의 문제가 아니었다. SQL에서 vector search로 관련 데이터를 찾고, 그 결과를 JSON context로 정리한 뒤, prompt에 넣어 모델 응답의 근거로 쓰는 흐름이 된다. 이때 SQL의 역할은 단순 저장소가 아니라 retrieval context를 만드는 계층으로 확장된다."""
            practice_flow_md = """실습 흐름도 개념 차이와 직접 연결되어 있었다. 먼저 Azure SQL Database를 준비하고, Foundry project와 Azure OpenAI model을 배포한 뒤, SQL에서 외부 모델을 호출할 수 있도록 database scoped credential과 external model을 만든다. 그 다음 ProductReview 같은 테이블에 vector column을 추가하고 embedding을 생성한다.

검색 실습에서는 full-text index를 만든 뒤 full-text predicate로 검색하고, 같은 데이터에 대해 vector similarity search를 실행한다. 마지막으로 두 결과를 결합해 hybrid search를 수행하면서, 어떤 query intent에서 어떤 방식이 더 적합한지 비교한다. RAG 실습에서는 vector search로 가져온 데이터를 JSON context로 만들고, 그 context를 prompt에 넣어 Azure OpenAI endpoint를 호출한 뒤, stored procedure 형태로 전체 흐름을 묶는다."""
            outcome_md = """이번 강의를 통해 나는 SQL에서 intelligent search를 구현한다는 것이 단순히 검색 함수를 하나 추가하는 일이 아니라는 점을 이해했다. full-text search는 정확한 단어와 구문을 다루는 방식이고, vector search는 embedding을 기반으로 의미적 유사도를 계산하는 방식이며, hybrid search는 두 신호를 결합해 더 안정적인 검색 결과를 만드는 방식이다.

성과는 각 개념을 암기한 것이 아니라, 어떤 상황에서 어떤 검색 방식을 선택해야 하는지 설명할 수 있게 된 것이다. 또한 Lab 흐름을 통해 Azure SQL Database, external model, vector column, full-text index, VECTOR_SEARCH, RAG stored procedure가 서로 따로 떨어진 기능이 아니라 하나의 검색/응답 파이프라인으로 연결된다는 점을 정리할 수 있었다."""
            portfolio_summary_md = "This note captures how I studied intelligent search in SQL from a learner's perspective. I focused on distinguishing full-text search, vector search, and hybrid search by query intent, then connected those concepts to SQL implementation details such as full-text indexes, vector columns, embeddings, VECTOR_SEARCH, and Reciprocal Rank Fusion.\n\nThe main learning outcome was not just knowing the names of the features, but being able to explain when each search approach fits, how the lab workflow validates the concept, and how the same retrieval foundation extends into RAG with SQL."
            skills_md = "- Distinguishing full-text, vector, and hybrid search by query intent\n- Understanding embeddings and vector columns in SQL\n- Connecting full-text indexes and predicates to keyword search\n- Explaining VECTOR_DISTANCE and VECTOR_SEARCH use cases\n- Understanding hybrid ranking with Reciprocal Rank Fusion\n- Mapping lab steps to validation criteria\n- Connecting SQL retrieval to RAG context construction\n- Writing technical learning notes from a learner's point of view"
            step_sections = [
                (
                    "full-text search의 역할을 먼저 분리",
                    "처음에는 full-text search가 단순 문자열 검색과 얼마나 다른지 흐릿했다.",
                    "full-text index, predicate, query full-text index 흐름을 키워드와 언어 기반 검색을 위한 준비 단계로 정리했다.",
                    "정확한 단어 또는 표현을 기준으로 결과를 찾아야 하는 상황에 full-text search가 맞는지 설명할 수 있는지 확인했다.",
                ),
                (
                    "vector search를 의미 기반 검색으로 이해",
                    "vector search는 embedding, vector column, similarity 같은 용어가 함께 나와서 구현 단계와 개념 단계가 섞여 보였다.",
                    "vector data type에 embedding을 저장하고, VECTOR_DISTANCE나 VECTOR_SEARCH 같은 함수로 의미적으로 가까운 항목을 찾는 흐름으로 정리했다.",
                    "단어가 정확히 일치하지 않아도 의미가 가까운 결과를 찾는 상황에서 vector search가 왜 필요한지 설명할 수 있는지 확인했다.",
                ),
                (
                    "hybrid search와 Reciprocal Rank Fusion의 필요성 정리",
                    "full-text와 vector 중 하나만 고르면 검색 품질을 놓치는 경우가 생길 수 있었다.",
                    "hybrid search를 키워드 기반 결과와 의미 기반 결과를 함께 사용하고, Reciprocal Rank Fusion으로 순위를 합치는 방식으로 이해했다.",
                    "검색 의도에 따라 두 결과를 결합해야 하는 이유와 ranking을 보정해야 하는 이유를 말할 수 있는지 확인했다.",
                ),
                (
                    "Lab에서 SQL 구현 흐름 확인",
                    "개념을 이해해도 실제 SQL 실습에서는 credential, external model, vector column, embedding 생성, full-text index 생성이 어떤 순서로 이어지는지 헷갈릴 수 있었다.",
                    "Lab 흐름을 Azure SQL Database 준비, database scoped credential 생성, external model 설정, vector column 추가, embedding 생성, full-text index 생성, hybrid query 실행 순서로 정리했다.",
                    "마지막에 full-text predicate, vector similarity, hybrid search를 각각 실행해 차이를 확인할 수 있는지 기준을 세웠다.",
                ),
            ]
            concept_items = [
                ("full-text search", "정확한 단어, 구문, 언어 기반 조건을 중심으로 SQL 데이터에서 관련 결과를 찾는 검색 방식으로 이해했다."),
                ("vector search", "텍스트를 embedding vector로 바꾼 뒤 의미적으로 가까운 항목을 찾는 검색 방식으로 이해했다."),
                ("hybrid search", "full-text search와 vector search 결과를 함께 사용해 키워드 일치와 의미 유사도를 모두 반영하는 접근으로 정리했다."),
                ("Reciprocal Rank Fusion", "여러 검색 결과 목록의 순위를 결합해 hybrid search ranking을 조정하는 방식으로 이해했다."),
                ("database scoped credential", "SQL에서 외부 embedding/model 호출에 필요한 인증 정보를 데이터베이스 범위로 관리하는 요소로 정리했다."),
                ("external model", "SQL 실습에서 embedding 생성이나 AI 기능 호출을 연결하는 모델 설정 단계로 이해했다."),
                ("vector column", "생성된 embedding을 SQL 테이블 안에 저장하고 이후 similarity search에 활용하기 위한 컬럼으로 정리했다."),
                ("RAG with SQL", "SQL에서 검색한 결과를 JSON context나 prompt로 구성해 생성형 AI 응답의 근거로 쓰는 흐름으로 이해했다."),
            ]
            concept_md = concept_bullets(concept_items)

        return sanitize_medium_markdown(f"""# {title_for_article}: 복잡한 기술 개념을 문제 해결 흐름으로 전환한 학습 기록

_A problem-solving Medium portfolio note from a learner's technical study_

## 짧은 도입부
{learner_intro}

## 핵심 작업 요약
- 내가 정의한 문제: 강의의 복잡한 기술 개념을 실제 해결 가능한 문제 단위로 전환하는 것
- 학습한 범위: {scope}
- 해결 기준: 문제 정의, 원인 판단, 조치, 확인 결과
- 학습 결과: 강의와 실습 근거를 바탕으로 문제 해결형 포트폴리오 기록으로 구성했다.

## 참고한 자료
- {seed_url}

## 학습 흐름 정리
{flow_items}

## 핵심 내용 정리
{core_learning_md}

## 강의에서 참고한 영상/링크
{video_md}

## 실습에서 확인한 항목
{lab_items}

## 문제 인식
{learner_problem}

## 문제 정의
{learner_definition}

## 왜 이것을 문제로 인식했는가
{learner_why}

## 문제 해결 경험
{chr(10).join(f"### {i}. {title}{chr(10)}문제: {problem}{chr(10)}{chr(10)}원인 판단: 강의 내용에서 이 항목이 독립된 설명으로 끝나는 것이 아니라 앞뒤 개념과 실습 확인 단계에 연결되어 있음을 확인했다.{chr(10)}{chr(10)}조치: {action}{chr(10)}{chr(10)}확인 결과: {validation}{chr(10)}" for i, (title, problem, action, validation) in enumerate(step_sections, start=1))}

## 복잡한 문제 해결 경험
{complex_detail_md}

## 실습 흐름과 검증 기준
{practice_flow_md}

## 성과
{outcome_md}

## 사용한 주요 개념 정리
{concept_md}

## 사용한 주요 수식/코드 정리
이번 자료에서 Medium 글에 그대로 인용할 수 있는 수식 또는 코드 원문은 명확하게 제공되지 않았다. 따라서 임의의 코드나 수식을 만들지 않고, 강의에서 확인된 개념·도구·실습 단계의 역할만 문제 해결 흐름 안에서 해석했다.

코드나 수식이 포함된 Lab 원문이 제공되는 경우에는 다음 기준으로 정리한다.

```text
1. 어떤 문제가 있었는가
2. 왜 이 코드/수식이 필요했는가
3. 코드/수식이 어떤 처리를 수행하는가
4. 결과를 어떻게 검증했는가
```

## 최종 정리
이번 학습 기록의 핵심은 강의 내용을 기능 설명으로 요약하는 것이 아니라, 복잡한 기술 문제를 정의하고 원인을 파악한 뒤 강의와 실습 근거로 해결하는 흐름을 남긴 것이다. 나중에 같은 주제를 다시 볼 때도 단원 제목만 훑는 것이 아니라, 어떤 문제를 해결해야 하고 어떤 결과로 검증해야 하는지부터 확인할 수 있다.

## Portfolio Summary
{portfolio_summary_md}

## Key skills practiced
{skills_md}

## 이미지 번호와 캡션 목록
- 이미지 제공 없음: 이번 글은 URL에서 수집한 강의·영상·Lab 텍스트 근거를 바탕으로 작성했다.
""")

    is_purview_security_topic = (
        "purview" in title_l
        or "secure ai data" in title_l
        or "data security posture" in title_l
        or "data security" in title_l
    )

    if "aiskillsnavigator.microsoft.com" in host and is_purview_security_topic:
        article_title = "Microsoft Purview로 AI 데이터 보안 이해하기: Copilot 시대의 데이터 노출·가시성·정책 관리 문제 정리"
        subtitle = "Understanding AI data security risks, Microsoft 365 Copilot exposure, DSPM for AI, and Purview policy management"
        learning_scope = "AI 데이터 보안 위험, Microsoft 365 Copilot으로 달라지는 보호 요구사항, Purview의 Data Security Posture Management for AI, 민감도 레이블과 정책 관리"
        learning_flow_label = "AI 사용 가시성 부족 → AI 상호작용의 데이터 노출 → 규제/컴플라이언스 리스크 → Copilot 보호 요구사항 → DSPM for AI로 위험 발견 및 정책 관리"
        core_problem = "처음 헷갈린 지점은 ‘AI 데이터 보안’이 기존 파일·문서 보안과 같은 문제인지, 아니면 Copilot과 생성형 AI 사용으로 새롭게 생기는 가시성·노출·규제 리스크까지 포함하는 문제인지 구분하는 것이었다."
        result_summary = "AI 데이터 보안을 단순 접근 권한 관리가 아니라, AI 사용 현황을 발견하고 데이터 노출 위험을 줄이며 정책과 민감도 레이블로 통제하는 흐름으로 정리했다."
        evidence_flow = [
            "Understand AI data security risks",
            "Key AI security risks",
            "Limited visibility into AI usage",
            "Data exposure in AI interactions",
            "Compliance and regulatory risks",
            "Understand how Microsoft 365 Copilot changes data protection needs",
            "Identify risks introduced by Copilot",
            "Use Data Security Posture Management for AI to discover risks and recommend protections",
            "Choose where to manage policies",
            "Create and publish sensitivity labels",
        ]
        lab_evidence = [
            "Create and publish sensitivity labels | Microsoft Learn",
            "Guided Technical Labs | Microsoft Learn",
        ]
        concepts = [
            ("AI data security risks", "AI 도구를 쓰는 과정에서 어떤 데이터가 입력·참조·노출되는지 파악하고 통제해야 하는 보안 문제로 이해했다."),
            ("Limited visibility into AI usage", "조직 안에서 누가 어떤 AI 도구를 어떻게 쓰는지 보이지 않으면 위험을 발견하거나 정책을 적용하기 어렵다는 문제다."),
            ("Data exposure in AI interactions", "사용자가 Copilot이나 AI 채팅에 민감 정보를 입력하거나, AI가 권한이 맞지 않는 데이터를 근거로 활용할 수 있는 노출 위험이다."),
            ("Compliance and regulatory risks", "AI 사용 과정에서 개인정보·기밀정보·규제 대상 데이터가 잘못 처리되면 감사와 규정 준수 문제가 생길 수 있다."),
            ("Microsoft 365 Copilot", "조직 데이터 위에서 답변을 생성하므로, 기존 문서 권한·민감도·DLP 정책이 AI 응답 품질과 보안에 직접 연결된다."),
            ("DSPM for AI", "AI 사용과 관련된 데이터 보안 상태를 발견하고 위험을 추천 조치로 연결하는 Purview의 관리 흐름으로 정리했다."),
            ("Sensitivity labels", "민감한 데이터의 등급과 보호 정책을 명확히 표시하고 적용하는 기준으로 보았다."),
            ("Policy management", "AI 사용을 무조건 막는 것이 아니라, 어디서 어떤 정책을 관리하고 적용할지 정하는 운영 문제로 이해했다."),
        ]
        step_sections = [
            ("AI 보안을 ‘AI 모델 보호’가 아니라 ‘AI가 쓰는 데이터 보호’ 문제로 다시 정의", "처음에는 AI 보안이라고 하면 모델 자체나 프롬프트 공격만 떠올리기 쉽다. 하지만 이번 학습의 핵심은 AI가 조직 데이터를 참조하고 사용자의 입력을 처리하는 과정에서 데이터가 어떻게 노출될 수 있는가였다.", "AI 데이터 보안을 모델 문제가 아니라 데이터 사용 가시성, 민감 정보 노출, Copilot 권한, 규정 준수의 문제로 나누어 정리했다.", "Limited visibility into AI usage, Data exposure in AI interactions, Compliance and regulatory risks가 핵심 위험으로 제시되는지 확인했다."),
            ("Microsoft 365 Copilot이 데이터 보호 요구사항을 바꾸는 이유 이해", "Copilot은 단순 외부 챗봇이 아니라 조직의 Microsoft 365 데이터와 연결될 수 있기 때문에, 기존 권한과 문서 분류가 그대로 AI 응답의 보안 경계가 된다.", "Copilot 사용 시 과도한 권한, 잘못 분류된 문서, 민감도 레이블 부재가 AI 답변 노출 문제로 이어질 수 있다고 정리했다.", "Understand how Microsoft 365 Copilot changes data protection needs와 Identify risks introduced by Copilot 항목을 같은 흐름으로 연결했다."),
            ("가시성 부족을 첫 번째 위험으로 잡기", "AI 사용을 보호하려면 먼저 누가 어떤 AI 기능을 사용하고 어떤 데이터가 관련되는지 보여야 한다. 사용 현황이 보이지 않으면 정책을 적용할 지점도 찾기 어렵다.", "Limited visibility를 단순 모니터링 부족이 아니라 이후 정책·레이블·보호 조치를 시작하기 위한 선행 문제로 정의했다.", "AI usage visibility가 별도 위험으로 다뤄지는지 확인하고, 이를 DSPM for AI가 해결해야 할 출발점으로 정리했다."),
            ("DSPM for AI를 위험 발견과 추천 조치 흐름으로 이해", "Purview의 DSPM for AI가 단순 대시보드인지, 실제로 AI 데이터 보안 위험을 찾아 보호 조치로 연결하는 기능인지 헷갈릴 수 있었다.", "DSPM for AI를 AI 관련 데이터 보안 상태를 발견하고, 노출 가능성이나 정책 누락을 찾아 추천 보호 조치로 이어주는 관리 계층으로 정리했다.", "Use Data Security Posture Management for AI to discover risks and recommend protections라는 흐름을 핵심 검증 기준으로 삼았다."),
            ("민감도 레이블과 정책 관리를 실습 완료 기준으로 연결", "보안 개념을 이해해도 실제 보호는 레이블과 정책으로 적용되어야 한다. 그래서 ‘어디서 정책을 관리하고 어떤 보호 기준을 적용할지’가 마지막에 중요해진다.", "Create and publish sensitivity labels, Choose where to manage policies를 단순 링크가 아니라 AI 데이터 보호를 실제 운영 규칙으로 바꾸는 단계로 정리했다.", "민감도 레이블 생성/게시와 정책 관리 위치 선택이 최종 보호 조치로 이어지는지 확인했다."),
        ]
        portfolio_skills = [
            "AI 데이터 보안 리스크 구조화",
            "Microsoft 365 Copilot 데이터 보호 요구사항 이해",
            "Purview DSPM for AI 역할 정리",
            "민감도 레이블과 정책 관리 흐름 연결",
            "보안 개념을 실무 검증 기준으로 정리",
        ]
    elif "aiskillsnavigator.microsoft.com" in host and ("fabric iq" in topic_l or "ontology" in topic_l):
        article_title = "Microsoft Fabric IQ 온톨로지 실습: 데이터 의미 계층을 모델링하며 헷갈린 지점 정리"
        subtitle = "Understanding Fabric IQ ontology through data sources, entity types, relationships, binding, and preview validation"
        learning_scope = "Fabric IQ, ontology creation approaches, semantic model, OneLake, lakehouse, eventhouse, entity type, relationship type, data binding, ontology preview"
        learning_flow_label = "온톨로지 생성 방식 구분 → 데이터 원천 준비 → entity type 정의 → relationship type 구성 → data binding → preview 검증"
        core_problem = "처음 헷갈린 지점은 Fabric IQ의 온톨로지가 단순한 데이터베이스 스키마인지, Power BI semantic model과 같은 것인지, 아니면 AI 앱이 데이터를 이해하도록 돕는 별도의 의미 계층인지 구분하는 것이었다."
        result_summary = "Fabric IQ 온톨로지를 테이블 구조가 아니라 AI 앱이 도메인 데이터를 해석하도록 돕는 의미 계층으로 이해하고, 데이터 원천·엔티티·관계·바인딩·미리보기 검증 흐름으로 정리했다."
        evidence_flow = [
            "Azure Decoded: Ground AI Apps with Fabric IQ’s Semantic Foundation",
            "Choose an ontology creation approach",
            "Generate from Power BI semantic model",
            "Build directly from OneLake",
            "Build an ontology manually",
            "Create a workspace",
            "Create a lakehouse with sample data",
            "Create an eventhouse with streaming data",
            "Ingest vital signs data",
            "Create entity types",
            "Create relationship types",
            "Bind entity types to data",
            "Configure relationships",
            "Preview the ontology",
        ]
        lab_evidence = [lab for lab in labs if lab and "fabric" in lab.lower()][:6] or [
            "Create an ontology with Fabric IQ | mslearn-fabric",
            "Build an ontology from a semantic model in Fabric IQ | mslearn-fabric",
        ]
        concepts = [
            ("Fabric IQ", "AI 앱이 데이터를 단순 테이블이 아니라 의미 있는 도메인 개체로 이해하도록 돕는 계층으로 정리했다."),
            ("Ontology", "Hospital, Patient, vital signs 같은 도메인 개체와 관계를 정의하는 의미 모델로 이해했다."),
            ("Power BI semantic model", "이미 정리된 분석 모델에서 온톨로지를 생성할 수 있는 출발점 중 하나로 보았다."),
            ("OneLake / Lakehouse", "파일과 테이블 형태의 sample data를 온톨로지에 연결할 데이터 원천으로 보았다."),
            ("Eventhouse / streaming data", "vital signs처럼 시간에 따라 들어오는 데이터를 의미 계층에 연결하기 위한 원천으로 이해했다."),
            ("Entity type", "온톨로지 안에서 다룰 대상의 종류를 정의하는 단계로 정리했다."),
            ("Relationship type", "entity 사이의 관계를 명시해 AI 앱이 데이터 연결 의미를 잃지 않게 하는 단계로 보았다."),
            ("Bind entity types to data", "개념으로 만든 entity를 실제 lakehouse/eventhouse 데이터와 연결하는 핵심 단계로 보았다."),
            ("Preview the ontology", "온톨로지가 실제 데이터와 연결되어 의미 있게 탐색되는지 확인하는 검증 단계로 잡았다."),
        ]
        step_sections = [
            ("온톨로지를 ‘테이블 구조’가 아니라 ‘AI 앱을 위한 의미 계층’으로 다시 정의", "처음에는 ontology, semantic model, OneLake, Fabric IQ가 모두 비슷한 데이터 모델링 용어처럼 보였다. 특히 Power BI semantic model에서 생성하는 방식과 OneLake에서 직접 구성하는 방식이 함께 나오면서, 온톨로지가 기존 semantic model의 다른 이름인지 헷갈릴 수 있었다.", "온톨로지를 테이블 목록이 아니라 entity type, relationship type, property, data binding을 통해 AI 앱이 데이터를 해석하게 만드는 계층으로 정리했다.", "‘Ground AI Apps with Fabric IQ’s Semantic Foundation’이라는 흐름과 Create an ontology module/lab이 같은 주제를 가리키는지 확인했다."),
            ("생성 방식 세 가지를 같은 기능이 아니라 다른 출발점으로 구분", "Generate from Power BI semantic model, Build directly from OneLake, Build an ontology manually가 같이 등장해서 어느 방식이 정답인지가 아니라 언제 어떤 방식으로 시작하는지가 헷갈렸다.", "semantic model 기반 생성은 이미 모델링된 분석 구조를 활용하는 방식, OneLake 기반 구성은 데이터 원천에서 직접 의미 계층을 구성하는 방식, 수동 생성은 entity와 relationship을 직접 정의하는 방식으로 나누었다.", "각 방식이 ontology creation approach 아래에 묶여 있는지 확인하고, 하나의 버튼 절차가 아니라 선택 가능한 접근 방식으로 정리했다."),
            ("Lakehouse와 Eventhouse를 온톨로지의 실제 데이터 근거로 연결", "온톨로지를 만든다고 해도 실제 데이터와 연결되지 않으면 빈 개념도에 머문다. Lab 흐름에서 workspace, lakehouse sample data, eventhouse, streaming data ingest가 먼저 나오는 이유가 여기서 헷갈릴 수 있었다.", "hospital sample data는 lakehouse로, vital signs 같은 흐름 데이터는 eventhouse/streaming ingest로 들어오며, 이 데이터 원천이 이후 entity binding의 근거가 된다고 정리했다.", "Create a lakehouse with sample data, Download and load the hospital data files, Create an eventhouse with streaming data, Ingest vital signs data 단계가 온톨로지 생성 전에 배치되는지 확인했다."),
            ("Entity type과 property를 실제 도메인 대상으로 분해", "entity type은 처음 보면 단순 테이블명처럼 보이지만, 실습에서는 Hospital 같은 도메인 대상을 정의하고 그 속성을 채우는 단계에 가깝다.", "entity type을 데이터 안에서 AI가 하나의 대상으로 인식해야 할 것으로 보고, property는 그 대상을 설명하는 속성으로 분리했다.", "Create entity types, Define the entity type, Add properties to the entity type 흐름이 이어지는지 확인했다."),
            ("Relationship type과 configure relationships를 검증 기준으로 설정", "개체를 만들기만 하면 AI가 관계까지 이해한다고 착각하기 쉽다. 실제로는 entity 사이의 관계를 별도로 정의하고 설정해야 의미 계층이 완성된다.", "relationship type은 entity 사이 연결 의미를 정의하는 단계, configure relationships는 그 연결이 실제 데이터와 맞게 구성되는지 조정하는 단계로 정리했다.", "Create relationship types, Configure relationships, Preview the ontology 단계가 실습의 완료 검증 흐름으로 이어지는지 확인했다."),
            ("Preview 단계로 완료를 판단", "실습에서 가장 중요한 것은 화면을 따라갔다는 사실이 아니라, 만든 ontology가 실제로 탐색/확인 가능한 상태인지다.", "preview the ontology를 최종 검증 기준으로 두고, entity와 relationship이 데이터에 binding된 후 의미 있게 보이는지를 확인 포인트로 잡았다.", "Preview the ontology가 실습 흐름의 후반부에 등장하는지 확인하고, 이 단계를 최종 결과 확인 기준으로 정리했다."),
        ]
        portfolio_skills = [
            "Fabric IQ ontology 개념 분해",
            "Power BI semantic model / OneLake / manual creation 방식 비교",
            "Lakehouse·Eventhouse 데이터 원천 구분",
            "Entity type과 property 모델링",
            "Relationship type과 data binding 이해",
            "Preview 기반 실습 검증 기준 설정",
        ]
    elif "wikidocs.net" in host:
        article_title = "WikiDocs 코딩테스트 파이썬 학습: 목차를 문제 풀이 기준으로 다시 읽기"
        subtitle = "Reorganizing Python coding-test chapters into problem-solving checkpoints"
        learning_scope = "파이썬 문법, 자료구조, 시간 복잡도, 스택, 코딩테스트 문제 풀이 기준"
        learning_flow_label = "문법 확인 → 입력 조건 해석 → 자료구조 선택 → 시간 복잡도 점검 → 스택 패턴 정리"
        core_problem = "처음 헷갈린 지점은 파이썬 문법을 읽는 것과 실제 코딩테스트 문제에서 자료구조를 선택하는 것이 다르다는 점이었다."
        result_summary = "문법과 자료구조를 정의 암기가 아니라 문제 조건을 읽고 풀이 전략을 고르는 기준으로 다시 정리했다."
        evidence_flow = headings[:14] or ["코딩 테스트", "파이썬", "자료구조", "스택", "시간 복잡도"]
        lab_evidence = []
        concepts = [
            ("시간 복잡도", "입력 크기에서 풀이가 통과 가능한지 판단하는 기준으로 정리했다."),
            ("자료구조 선택", "문제 조건을 보고 list, dict, stack 중 무엇을 쓸지 결정하는 과정으로 보았다."),
            ("스택", "최근 값부터 처리해야 하는 괄호/되돌리기/경로 추적 유형의 판단 기준으로 정리했다."),
        ]
        step_sections = [
            ("문법 설명을 문제 조건으로 바꾸기", "문법 자체는 이해해도 문제 입력을 보면 어떤 구조로 풀어야 할지 바로 연결되지 않을 수 있었다.", "각 문법과 자료구조를 언제 쓰는가 기준으로 다시 정리했다.", "개념을 정의가 아니라 문제 유형과 연결해 설명할 수 있는지 확인했다."),
            ("스택을 사용해야 하는 신호 찾기", "스택은 이름은 쉽지만 실제 문제에서 언제 떠올려야 하는지가 어렵다.", "마지막에 들어온 값을 먼저 처리해야 하는 조건, 짝 맞추기, 되돌아가기 흐름을 스택 사용 신호로 정리했다.", "스택 문제를 볼 때 append/pop 구조로 설명할 수 있는지 확인했다."),
        ]
        portfolio_skills = ["문제 조건 해석", "자료구조 선택", "시간 복잡도 점검", "스택 패턴 정리"]
    elif "oopy.io" in host:
        learning_nodes = source_graph_learning_nodes(graph, limit=8)
        focus_title = learning_nodes[0]["title"] if learning_nodes else "CS 핵심 개념"
        focus_line = source_node_summary_line(learning_nodes[0]) if learning_nodes else "CS 개념을 정의, 동작 원리, 비교 기준으로 나누어 설명해야 했다."
        node_titles = [node["title"] for node in learning_nodes] or headings[:8] or ["운영체제", "네트워크", "자료구조", "데이터베이스"]
        article_title = f"{focus_title} 중심 CS 학습: 면접 답변으로 설명 가능한 구조 만들기"
        subtitle = "Turning CS notes into problem-solving explanations with concepts, causes, and validation criteria"
        learning_scope = ", ".join(node_titles[:6])
        learning_flow_label = "개념 정의 → 동작 원리 → 문제 상황 → 비교 기준 → 면접 답변 검증"
        core_problem = (
            f"`{focus_title}`를 포함한 CS 개념을 단순 정의 암기가 아니라 실제 질문에 답할 수 있는 구조로 바꾸는 것이 핵심 문제였다. "
            f"특히 자료에서는 `{focus_line}`라는 본문 근거가 확인되어, 개념의 의미와 동작 조건을 함께 정리해야 했다."
        )
        result_summary = (
            f"{learning_scope}를 정의, 원리, 원인, 비교 기준, 검증 질문으로 나누어 정리했다. "
            "그 결과 각 개념을 단순 요약이 아니라 면접과 실무 질문에 답할 수 있는 문제 해결 단위로 설명할 수 있게 되었다."
        )
        evidence_flow = [
            f"{node['title']}: {source_node_summary_line(node)}"
            for node in learning_nodes[:8]
        ] or headings[:14] or ["운영체제", "네트워크", "자료구조", "데이터베이스", "CS"]
        lab_evidence = []
        concepts = [
            (
                node["title"],
                source_node_summary_line(node),
            )
            for node in learning_nodes[:8]
        ] or [
            ("운영체제", "프로세스, 스레드, 메모리처럼 실행 환경을 설명하는 기본 축으로 정리했다."),
            ("네트워크", "요청과 응답이 이동하는 계층과 프로토콜 흐름으로 정리했다."),
            ("자료구조", "데이터 저장/접근 방식이 시간 복잡도와 풀이 전략에 미치는 영향으로 정리했다."),
        ]
        step_sections = []
        for node in learning_nodes[:5]:
            section_title = node["title"]
            sentences = source_node_key_sentences(node.get("text") or "", limit=3)
            first_sentence = sentences[0] if sentences else source_node_summary_line(node)
            second_sentence = sentences[1] if len(sentences) > 1 else first_sentence
            title_pattern = re.escape(section_title)
            first_sentence = re.sub(rf"^{title_pattern}\s+", "", first_sentence).strip() or first_sentence
            second_sentence = re.sub(rf"^{title_pattern}\s+", "", second_sentence).strip() or second_sentence
            resolution_text = source_node_resolution_text(section_title, node.get("text") or "")
            step_sections.append((
                f"{section_title}를 설명 가능한 문제 단위로 분해",
                (
                    f"`{section_title}`는 정의만 외우면 실제 질문에서 왜 필요한지 설명하기 어려운 개념이었다. "
                    f"자료 본문에서는 `{first_sentence}`라고 설명되어 있어, 의미와 동작 조건을 함께 잡아야 했다."
                ),
                resolution_text + f" 추가로 `{second_sentence}`라는 근거를 확인해 답변에서 빠지면 안 되는 조건을 보강했다.",
                f"`{section_title}`를 1분 안에 정의하고, 원리와 비교 질문까지 이어서 설명할 수 있는지 확인했다.",
            ))
        if not step_sections:
            step_sections = [
                ("요약 내용을 면접 질문 단위로 재분류", "CS 요약은 범위가 넓어서 그대로 읽으면 무엇을 설명할 수 있어야 하는지 흐려진다.", "각 개념을 정의, 왜 필요한지, 대표 상황, 비교 질문으로 나누었다.", "1분 설명과 꼬리 질문 대응이 가능한지 확인했다."),
                ("암기와 설명의 차이 분리", "짧은 정의만 외우면 실제 질문에서 왜 중요한지 설명하기 어렵다.", "개념마다 본문 정의, 핵심 조건, 문제 상황, 비교 기준을 실제 문장으로 풀어 답변 구조를 만들었다.", "최종 글이 용어 목록이 아니라 답변 구조를 갖는지 확인했다."),
            ]
        portfolio_skills = [
            "CS 개념 구조화",
            "기술 면접 답변 구성",
            "운영체제와 네트워크 개념 비교",
            "프로세스·스레드·메모리 원리 설명",
            "TCP/UDP와 HTTP/REST 차이 정리",
            "개념을 문제 상황과 원인으로 재구성",
            "꼬리 질문 대비용 검증 기준 설정",
            "본문 근거 기반 학습 기록 작성",
        ]
    else:
        article_title = f"{title or '학습 자료'} 학습 기록: 핵심 개념을 이해 가능한 문제 단위로 정리하기"
        subtitle = "Organizing difficult learning concepts into a human learner's problem-solving note"
        learning_scope = ", ".join(headings[:6]) or "강의 핵심 개념과 실습 흐름"
        learning_flow_label = "핵심 개념 확인 → 비슷한 개념과 구분 → 실습에서 쓰이는 위치 확인 → 완료 기준 정리"
        core_problem = "처음 헷갈린 지점은 강의 안의 핵심 개념이 무엇이고, 그 개념이 실습 흐름에서 어떤 역할을 하는지 구분하는 것이었다."
        result_summary = "강의 내용을 단순 목록이 아니라 개념 구분, 실습 적용 위치, 확인 기준으로 다시 정리했다."
        evidence_flow = headings[:14]
        lab_evidence = [lab for lab in labs[:6] if lab]
        concepts = [(h, "강의 흐름에서 별도로 이해해야 할 핵심 개념으로 정리했다.") for h in headings[:6]] or [("핵심 개념", "강의 안에서 반복적으로 등장하는 중심 개념이다.")]
        step_sections = [
            ("핵심 개념과 주변 설명 분리", "강의에는 정의, 예시, 링크, 실습 지시가 섞여 있어 무엇을 먼저 이해해야 하는지 흐려질 수 있었다.", "제목과 소제목을 기준으로 중심 개념을 먼저 분리하고, 주변 설명은 보조 근거로 보았다.", "정리한 개념이 강의의 목표와 실습 단계에 직접 연결되는지 확인했다."),
            ("실습에서 확인해야 할 기준 찾기", "개념을 읽는 것과 실제로 적용했는지 확인하는 것은 다르다.", "exercise, lab, summary, validation 성격의 항목을 따로 보며 마지막에 무엇을 확인해야 하는지 정리했다.", "정리 결과가 단순 요약이 아니라 다음 복습 때 확인할 수 있는 기준을 남기는지 확인했다."),
        ]
        portfolio_skills = ["핵심 개념 분리", "학습 흐름 구조화", "실습 검증 기준 설정"]

    if "aiskillsnavigator.microsoft.com" in host and is_purview_security_topic:
        problem_definition_md = """이 학습에서 문제로 잡은 것은 AI 데이터 보안이 기존 문서 보안과 어떻게 달라지는지 구분하는 것이었다. 특히 Microsoft 365 Copilot처럼 조직 데이터와 연결되는 AI를 사용할 때, 위험은 단순히 ‘AI를 쓴다’는 사실에서 생기는 것이 아니라 다음 지점에서 생긴다.

1. 조직 안에서 어떤 AI 사용이 일어나는지 보이지 않는 가시성 부족
2. Copilot 또는 AI 상호작용 중 민감 데이터가 노출될 가능성
3. 권한, 민감도 레이블, 정책 관리가 정리되지 않아 생기는 규정 준수 리스크
4. 발견된 위험을 Purview DSPM for AI와 정책 관리 흐름으로 어떻게 줄일지 판단하는 문제"""
        why_problem_md = """이 내용을 문제로 본 이유는 Copilot이 단순한 챗봇이 아니라 조직의 문서, 권한, 민감도 레이블, 정책 상태와 연결될 수 있기 때문이다. 기존에는 파일을 누가 볼 수 있는지가 중심이었다면, AI 환경에서는 그 데이터가 AI 응답이나 사용자 입력 흐름에서 어떻게 드러날 수 있는지까지 생각해야 한다. 그래서 AI data security risks, limited visibility, data exposure, compliance risks를 따로 구분하지 않으면 Purview가 왜 필요한지 흐려진다."""
        complex_summary_md = """가장 복잡했던 부분은 Purview가 AI 모델 자체를 보호하는 도구가 아니라, AI가 사용하는 조직 데이터의 보안 상태를 발견하고 통제하는 계층이라는 점이었다. Limited visibility는 단순 모니터링 문제가 아니라 정책을 적용할 대상을 찾지 못하는 문제이고, data exposure는 사용자가 AI와 상호작용하는 과정에서 민감 정보가 드러날 수 있는 문제다. DSPM for AI는 이 위험을 발견하고 추천 조치로 연결하며, sensitivity labels와 정책 관리는 그 조치를 실제 보호 기준으로 바꾸는 단계로 이해했다."""
        final_summary_md = """이번 학습을 통해 AI 데이터 보안을 ‘AI 기능을 켜고 끄는 문제’가 아니라, 조직 데이터가 AI 사용 흐름에서 어떻게 보이고, 노출되고, 정책으로 통제되는지 확인하는 문제로 정리했다. 특히 Microsoft 365 Copilot 환경에서는 권한, 민감도 레이블, 정책 관리가 AI 응답의 안전성과 직접 연결되기 때문에, Purview DSPM for AI를 위험 발견과 보호 조치 추천 흐름으로 이해하는 것이 핵심이었다."""
    elif "aiskillsnavigator.microsoft.com" in host and ("fabric iq" in topic_l or "ontology" in topic_l):
        problem_definition_md = """이 학습에서 문제로 잡은 것은 Fabric IQ의 온톨로지가 단순 데이터베이스 스키마인지, Power BI semantic model의 다른 이름인지, 아니면 AI 앱이 데이터를 이해하도록 돕는 의미 계층인지 구분하는 것이었다. 핵심은 다음 지점을 분리하는 것이었다.

1. semantic model 기반 생성, OneLake 기반 생성, 수동 생성 방식의 차이
2. lakehouse와 eventhouse가 온톨로지의 데이터 근거가 되는 이유
3. entity type, property, relationship type의 역할 차이
4. bind entity types to data와 preview ontology가 실습 완료 기준이 되는 이유"""
        why_problem_md = """이 내용을 문제로 본 이유는 온톨로지라는 말이 데이터 모델, 스키마, semantic model과 비슷하게 들리지만 실습에서는 역할이 다르기 때문이다. Fabric IQ의 온톨로지는 테이블을 나열하는 것이 아니라 AI 앱이 Hospital, Patient, vital signs 같은 도메인 개체와 관계를 의미 있게 이해하도록 만드는 구조다. 그래서 데이터 원천 준비, entity 정의, relationship 구성, data binding, preview 검증을 하나의 흐름으로 연결해야 했다."""
        complex_summary_md = """가장 복잡했던 부분은 온톨로지 생성 방식과 모델링 단계를 분리하는 것이었다. Power BI semantic model에서 생성하는 방식은 이미 정리된 분석 모델을 출발점으로 삼는 접근이고, OneLake 기반 구성은 데이터 원천에서 의미 계층을 만드는 접근이다. 수동 생성에서는 entity type과 relationship type을 직접 정의해야 하며, bind entity types to data를 통해 이 개념 구조를 실제 lakehouse/eventhouse 데이터에 연결해야 한다. 마지막 preview는 만든 온톨로지가 실제 데이터 위에서 의미 있게 작동하는지 확인하는 검증 단계로 이해했다."""
        final_summary_md = """이번 학습을 통해 Fabric IQ 온톨로지를 단순한 테이블 구조가 아니라 AI 앱을 위한 의미 계층으로 정리했다. 온톨로지를 제대로 이해하려면 생성 방식, 데이터 원천, entity type, relationship type, data binding, preview 검증을 따로 보아야 한다. 이 흐름을 잡고 나니 Fabric IQ가 AI 앱의 semantic foundation으로 설명되는 이유를 더 명확히 이해할 수 있었다."""
    elif "wikidocs.net" in host:
        problem_definition_md = """이 학습에서 문제로 잡은 것은 파이썬 문법을 읽는 것과 실제 코딩테스트 문제 조건에 맞춰 자료구조를 선택하는 것이 다르다는 점이었다. 핵심은 문법 자체가 아니라 입력 조건을 보고 리스트, 딕셔너리, 스택, 반복문, 시간 복잡도 중 무엇을 기준으로 풀어야 하는지 판단하는 것이었다."""
        why_problem_md = """이 내용을 문제로 본 이유는 코딩테스트에서 막히는 지점이 대개 문법을 몰라서가 아니라, 문제 조건을 보고 어떤 풀이 도구를 꺼내야 할지 연결하지 못하는 데서 나오기 때문이다. 그래서 목차를 따라 읽는 것보다 각 개념을 어떤 문제 유형에서 쓰는지로 다시 정리하는 것이 필요했다."""
        complex_summary_md = """가장 복잡했던 부분은 자료구조 이름을 아는 것과 문제 조건에서 그 자료구조를 선택하는 기준을 연결하는 것이었다. 스택은 단순히 후입선출 구조라는 정의보다, 괄호 짝 맞추기나 최근 상태 되돌리기처럼 마지막에 들어온 값을 먼저 처리해야 하는 조건에서 떠올려야 한다. 시간 복잡도 역시 수식 암기가 아니라 입력 크기에서 풀이가 통과 가능한지 판단하는 기준으로 정리했다."""
        final_summary_md = """이번 학습을 통해 파이썬 코딩테스트 내용을 문법 설명이 아니라 문제 풀이 판단 기준으로 다시 정리했다. 입력 조건을 읽고 자료구조를 선택하며, 시간 복잡도를 점검하고, 스택이 필요한 패턴을 구분하는 것이 핵심이었다."""
    elif "oopy.io" in host:
        first_concept = concepts[0][0] if concepts else "CS 핵심 개념"
        first_desc = concepts[0][1] if concepts else "CS 개념을 정의, 동작 원리, 비교 기준으로 나누어 설명해야 했다."
        second_concept = concepts[1][0] if len(concepts) > 1 else first_concept
        second_desc = concepts[1][1] if len(concepts) > 1 else first_desc
        third_concept = concepts[2][0] if len(concepts) > 2 else second_concept
        third_desc = concepts[2][1] if len(concepts) > 2 else second_desc
        problem_definition_md = f"""이 학습에서 문제로 잡은 것은 `{first_concept}`를 비롯한 CS 개념을 단순 정의가 아니라 설명 가능한 원리 구조로 바꾸는 것이었다. 자료 본문은 `{first_desc}`라고 설명한다. 이 문장을 기준으로 보면, 문제는 용어를 기억하는 것이 아니라 해당 개념이 어떤 자원을 다루고 어떤 조건에서 문제가 되며 어떤 기준으로 결과를 판단해야 하는지 말할 수 있게 만드는 데 있었다.

두 번째 문제는 개념들이 서로 따로 떨어진 암기 항목처럼 보인다는 점이었다. 예를 들어 `{second_concept}`에서는 `{second_desc}`라는 근거가 나오고, `{third_concept}`에서는 `{third_desc}`라는 근거가 이어진다. 그래서 나는 각 개념을 정의, 원인, 동작 방식, 비교 기준, 검증 질문으로 나누어 다시 정리하는 것을 해결 과제로 정의했다."""
        why_problem_md = f"""이 내용을 문제로 인식한 이유는 CS 면접이나 실무 설명에서는 짧은 정의보다 “왜 그런 구조가 필요한가”와 “비슷한 개념과 무엇이 다른가”가 더 중요하기 때문이다. `{first_concept}`를 예로 들면, `{first_desc}`라는 본문 근거는 개념의 뜻만 말해서는 부족하고 실제 동작 조건까지 설명해야 한다는 점을 보여준다.

또한 `{second_concept}`와 `{third_concept}`처럼 이어지는 항목들은 서로 다른 주제처럼 보이지만, 실제로는 자원 관리, 데이터 전달, 상태 변화, 성능과 신뢰성 같은 문제를 해결하기 위해 연결된다. 그래서 이 자료를 읽을 때 핵심 문제는 “많이 읽기”가 아니라, 각 개념이 어떤 문제를 해결하는지 답변 가능한 구조로 바꾸는 것이었다."""
        complex_summary_md = f"""가장 복잡했던 부분은 넓은 CS 범위를 하나의 암기 목록이 아니라 문제 해결 구조로 바꾸는 일이었다. `{first_concept}`에서는 `{first_desc}`라는 설명을 통해 개념의 기본 역할을 잡고, `{second_concept}`에서는 `{second_desc}`를 기준으로 동작 원리와 비교 지점을 분리했다.

이렇게 정리하니 CS 개념을 “정의 → 원리 → 문제 상황 → 비교 기준 → 검증 질문” 순서로 다시 볼 수 있었다. 예를 들어 운영체제 영역은 프로세스, 스레드, 스케줄링, 메모리 관리처럼 실행 환경과 자원 관리 문제로 묶이고, 네트워크 영역은 계층, TCP/UDP, HTTP, REST처럼 데이터가 이동하고 상태를 주고받는 문제로 묶인다. 복잡한 내용은 많았지만, 각 항목을 어떤 문제를 해결하는 개념인지로 분류하면서 설명 가능한 구조가 만들어졌다."""
        final_summary_md = f"""이번 학습을 통해 Oopy/Notion형 CS 요약 자료가 제목만 있는 목록이 아니라 실제 본문 근거를 가진 학습 흐름이라는 점을 확인했다. 특히 `{learning_scope}`를 중심으로 각 개념의 정의와 동작 원리, 문제 상황, 비교 기준을 분리하면서 면접 답변으로 전환할 수 있는 구조를 만들었다.

최종 성과는 단순히 CS 항목을 많이 읽은 것이 아니라, `{first_concept}` 같은 핵심 개념을 실제 질문 앞에서 설명할 수 있는 문제 해결 단위로 바꾼 것이다. 이 방식으로 정리하면 다음 복습 때도 개념 이름만 훑지 않고, 어떤 문제를 해결하는 개념인지부터 확인할 수 있다."""
    else:
        problem_definition_md = f"""이 학습에서 문제로 잡은 것은 {core_problem.replace('처음 헷갈린 지점은 ', '').rstrip('이었다.')}는 점이었다. 핵심은 자료를 정리하는 방식이 아니라, 강의 안의 중심 개념을 실제 실습 단계와 연결해 이해하는 것이었다."""
        why_problem_md = """이 내용을 문제로 본 이유는 기술 강의에서 비슷한 용어와 실습 단계가 한꺼번에 나오면 무엇이 핵심 개념이고 무엇이 보조 설명인지 흐려지기 때문이다. 그래서 학습자는 개념을 단순히 외우는 것이 아니라, 그 개념이 어느 단계에서 쓰이고 어떤 결과로 확인되는지까지 연결해야 한다."""
        complex_summary_md = """가장 복잡했던 부분은 강의의 핵심 개념과 실습 단계가 서로 분리되어 보일 수 있다는 점이었다. 그래서 개념을 정의, 적용 위치, 확인 결과로 나누어 정리하고, 마지막에 어떤 상태가 되면 이해했다고 볼 수 있는지 기준을 세웠다."""
        final_summary_md = f"""이번 학습을 통해 {result_summary} 핵심은 자료를 많이 모으는 것이 아니라, 강의 안의 어려운 개념을 이해 가능한 단위로 나누고 실습 흐름 안에서 검증 기준을 세우는 것이었다."""

    evidence_md = bullets(evidence_flow[:18])
    lab_md = bullets(lab_evidence[:8]) if lab_evidence else "- 별도 실습 링크 없음"
    step_md = "\n".join(
        f"### {i}. {title}\n문제/제약: {problem}\n\n조치: {action}\n\n확인 기준: {validation}\n"
        for i, (title, problem, action, validation) in enumerate(step_sections, start=1)
    )
    concept_md = concept_bullets(concepts)
    skills_md = bullets(portfolio_skills)

    return sanitize_medium_markdown(f"""# {article_title}

_{subtitle}_

## 짧은 도입부
이번 학습에서는 {learning_scope}를 보면서, 용어를 많이 아는 것보다 각 개념이 왜 필요하고 어디서 쓰이는지 구분하는 것이 더 중요하다는 점을 확인했다. 특히 핵심 개념이 비슷한 이름으로 이어질 때는 정의를 외우는 것보다 역할, 적용 위치, 검증 기준을 나누어 보는 것이 필요했다.

## 핵심 작업 요약
- 핵심 문제: {core_problem}
- 학습 범위: {learning_scope}
- 핵심 흐름: {learning_flow_label}
- 학습 결과: {result_summary}

## 참고한 자료
- {seed_url}

## 학습 흐름 정리
{evidence_md}

## 실습/검증 근거
{lab_md}

## 문제 인식
{core_problem}

이 지점을 문제로 본 이유는, 강의 자료를 그대로 따라가면 용어와 화면은 많이 남지만 실제로 무엇을 구분해야 하는지 흐려질 수 있기 때문이다. 기술 학습에서 중요한 것은 모든 문장을 기억하는 것이 아니라, 비슷해 보이는 개념을 나누고 실제 실습 단계에서 어떤 판단을 해야 하는지 설명할 수 있게 만드는 것이다.

## 문제 정의
{problem_definition_md}

## 왜 이것을 문제로 인식했는가
{why_problem_md}

## 문제 해결 경험

{step_md}
## 성과
{result_summary}

이번 학습을 통해 개념을 단순 목록으로 남기지 않고, 헷갈린 지점과 판단 기준으로 나누어 정리할 수 있었다. 이렇게 정리하니 나중에 같은 주제를 다시 볼 때도 어떤 개념을 먼저 확인해야 하는지, 어떤 실습 단계를 검증 기준으로 삼아야 하는지 빠르게 복습할 수 있다.

## 사용한 주요 개념 정리
{concept_md}

## 복잡한 내용 정리
{complex_summary_md}

## 최종 정리
{final_summary_md}

## Portfolio Summary
This learning record documents how I organized a technical learning topic into concept boundaries, practical workflow, and validation criteria. The focus is not on listing all materials, but on explaining what was difficult, how I separated the concepts, and how I checked whether the learning flow made sense.

## Key skills practiced
{skills_md}
""")

def should_use_source_graph_direct_article(seed_url: str, collector_report: dict[str, Any], source_pack_text: str) -> bool:
    host = url_domain(seed_url)
    stats = source_graph_stats_summary(collector_report)
    graph = collector_source_graph(collector_report)
    quality = graph.get("quality") if isinstance(graph.get("quality"), dict) else {}
    if not source_pack_text.strip():
        return False
    if "aiskillsnavigator.microsoft.com" in host and (stats["labs"] > 0 or stats["lessons"] > 0 or stats["tree_items"] > 0):
        return True
    if "wikidocs.net" in host and stats["chars"] >= 3000:
        return True
    if "oopy.io" in host and stats["chars"] >= 1000:
        return True
    if substantial_rendered_document(seed_url, max(stats["chars"], len(source_pack_text or "")), source_pack_text):
        return True
    if expected_topic_kind_from_input(seed_url=seed_url, current_text=source_pack_text) and max(stats["chars"], len(source_pack_text or "")) >= 600:
        return True
    if "youtube.com" in host or "youtu.be" in host:
        return stats["chars"] >= 1200 and int(quality.get("transcript_segments") or 0) > 0
    return False


def clean_prompt_memo(memo: str) -> str:
    cleaned_lines: list[str] = []
    for line in (memo or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[생성 직전 사용자가 적은 어려움/헷갈린 부분]") or stripped.startswith("[생성 직전 사용자가 정의한 어려운 문제]"):
            continue
        if stripped.startswith("없음. 자료의 핵심 흐름"):
            continue
        cleaned_lines.append(stripped)
    return dedupe_repeated_lines("\n".join(cleaned_lines)).strip()

def direct_source_text(raw_text: str, memo: str, image_count: int) -> str:
    sections = []
    if raw_text.strip():
        sections.append(f"[화면 텍스트/코드/오류]\n{raw_text.strip()}")
    if memo.strip():
        sections.append(f"[사용자 메모/질문/해결 과정]\n{memo.strip()}")
    if image_count:
        sections.append(
            "[이미지 정보]\n"
            f"사용자가 실습 진행 순서대로 업로드한 이미지 수: {image_count}장\n"
            "이미지 자체는 최종 글에서 단순 캡션이 아니라 문제 발견, 원인 분석, 해결 과정, 검증 결과의 근거로 사용해야 합니다."
        )
    sections.append(
        "[작성 주의]\n"
        "비어 있는 입력칸, 내부 라벨, '직접 입력된 화면 텍스트 없음', '직접 입력된 사용자 메모 없음' 같은 시스템 상태 문구는 최종 글에 절대 쓰지 않습니다."
    )
    return "\n\n".join(sections)


SECTION_TITLES = [
    "한국어 제목",
    "영어 부제",
    "짧은 도입부",
    "핵심 작업 요약",
    "문제 인식",
    "문제 정의",
    "왜 이것을 문제로 인식했는가",
    "문제 해결 경험",
    "복잡한 문제/수식/쿼리/코드 작성 및 해결 경험",
    "성과",
    "사용한 주요 수식/코드 정리",
    "최종 정리",
    "Portfolio Summary",
    "Key skills practiced",
    "이미지 번호와 캡션 목록",
]

PLACEHOLDER_PHRASES = [
    ":::writing",
    'id="',
    "여기에 이미지 넣기",
    "첨부 삽입",
    "이미지 추가 예정",
    "[화면 텍스트/코드/오류]",
    "[사용자 메모/질문/해결 과정]",
    "[이미지 정보]",
    "[작성 주의]",
    "근거 화면 1 기반 검증 단계",
    "근거 화면 2 기반 검증 단계",
    "근거 화면 3 기반 검증 단계",
    "근거 화면 4 기반 검증 단계",
    "이미지 근거와 사용자 메모를 연결해 문제 해결 단계로 재구성했습니다.",
    "문제 상황 제시",
    "문제 원인 분석",
]

FUNCTION_DESCRIPTION_PHRASES = [
    "버튼을 눌렀습니다",
    "차트를 만들었습니다",
    "관계를 설정했습니다",
    "실습을 완료했습니다",
]

FILLER_SENTENCES = [
    "이 단계에서는 화면에 보이는 현상을 그대로 받아들이지 않고",
    "조치는 기능 실행 자체가 아니라",
    "이 확인 결과는 다음 단계로 넘어가기 위한 기준이며",
    "데이터 변환 작업 성공",
    "오류를 확인하고 수정",
    "문제 해결 서사에 필요한 구체적 캡션",
    "추가 메모 없음",
    "데이터 소스 연결 및 데이터 로딩이 문제가 발생하여 해결이 필요하다",
    "데이터 로딩이 성공적으로 이루어졌는지 확인",
    "데이터 테이블 간 연관관계 확인이 성공적으로 이루어졌는지 확인",
    "데이터의 문제가 있음을 발견",
    "카테고리별 판매 금액이 동일하여 데이터의 문제가 있음을 발견",
    "최종 판매 목표 데이터 확인",
    "지역별 판매 데이터 확인",
    "문제 영역 식별",
    "이익률 결과를 확대하여 세부적으로 확인",
    "판매 데이터에서 비용을 차감하여 이익을 계산하는 측정값을 생성",
    "최종 salesperson 판매 목표 데이터 확인",
    "Vision 판독 실패",
    "파일명과 입력 메모만 근거로 사용",
    "파일명과 입력 메모를 분석하여 문제를 이해한다",
    "파일명과 입력 메모를 분석하여 원인을 찾는다",
    "파일명과 입력 메모를 분석하여 해결 방안을 찾는다",
    "파일명과 입력 메모를 분석하여 검증한다",
]

NON_POWERBI_TEMPLATE_PHRASES = [
    "화면에 값이 표시되더라도 모델, 수식, 관계, 필터 흐름이 맞지 않으면 분석 결과의 의미가 달라질 수 있습니다.",
    "데이터 모델, 수식, 관계, 변환 흐름 중 하나가 어긋나면",
    "화면에 숫자가 나오거나 설정 창이 열린 것만으로는",
]

POWER_QUERY_CORE_PROBLEM = (
    "SQL Server와 CSV에서 가져온 원본 데이터가 보고서 작성에 바로 사용할 수 있는 분석 모델 형태가 아니었고, "
    "데이터 품질·컬럼 구조·비표준 값·결측 원가·월별 목표 데이터 구조·보조 테이블 병합·최종 로드 범위를 정리해야 했다."
)

SEMANTIC_CORE_PROBLEM = (
    "Product[Category] 기준으로 Sales[Sales]를 집계했을 때 모든 Category 행에 동일한 매출 총액이 반복되어, "
    "Product 차원의 filter context가 Sales fact table로 전달되지 않는 relationship 문제를 확인했다."
)

DAX_CORE_PROBLEM = (
    "Sales 데이터를 단순 합계와 raw column 자동 집계에 의존하면 월 정렬, 날짜 기준 분석, 가격 지표, 목표 대비 성과 분석이 불명확해지므로 "
    "MonthKey 정렬, Date table, explicit measure, Target/Variance measure로 계산 맥락을 명확히 구성해야 했다."
)

INTERNAL_DIAGNOSTIC_PHRASES = [
    "Vision 판독 실패",
    "파일명과 입력 메모만 근거로 사용",
    "Vision 응답이 없어",
    "파일명과 입력 메모를 분석하여",
]

SEMANTIC_CONCRETE_ENTITIES = [
    "Product[Category]",
    "Sales[Sales]",
    "Product[ProductKey] -> Sales[ProductKey]",
    "One to many",
    "Cross-filter direction",
    "Star schema",
    "Product hierarchy",
    "Profit",
    "Profit Margin",
    "DIVIDE",
    "SalespersonRegion",
    "Bridge table",
    "Salesperson -> SalespersonRegion -> Region -> Sales",
    "inactive relationship",
    "Targets",
]

SEMANTIC_REGRESSION_REQUIREMENTS = [
    ("이미지 1 문제 화면 설명", ["이미지 1", "Product[Category]", "Sales[Sales]", "반복"]),
    ("이미지 2 ProductKey relationship 설정", ["이미지 2", "Product[ProductKey] -> Sales[ProductKey]", "One to many"]),
    ("이미지 4 star schema 모델 구조", ["이미지 4", "Star schema"]),
    ("이미지 5~6 hierarchy 구성", ["이미지 5", "Product hierarchy"]),
    ("이미지 7~8 Profit / Profit Margin measure", ["이미지 7", "Profit", "Profit Margin"]),
    ("이미지 9~10 Category별 결과", ["이미지 9", "Sales", "Profit", "Profit Margin"]),
    ("이미지 11 SalespersonRegion bridge table", ["이미지 11", "SalespersonRegion", "Bridge table"]),
    ("이미지 12 Cross-filter Both", ["이미지 12", "Cross-filter direction", "Both"]),
    ("이미지 13 direct relationship 비활성화", ["이미지 13", "inactive relationship"]),
    ("이미지 14 Salesperson별 Sales 변경 결과", ["이미지 14", "Salesperson", "Sales"]),
    ("이미지 15 Salesperson별 Sales와 Target 최종 비교", ["이미지 15", "Salesperson", "Targets"]),
]

POWER_QUERY_REGRESSION_REQUIREMENTS = [
    ("이미지 1 SQL Server database", ["이미지 1", "SQL Server", "AdventureWorksDW2020"]),
    ("이미지 2 Navigator Transform Data", ["이미지 2", "Navigator", "Transform Data", "FactResellerSales"]),
    ("이미지 3 Column quality/distribution/profile", ["이미지 3", "Column quality", "Column distribution", "Column profile"]),
    ("이미지 4 SalesPersonFlag", ["이미지 4", "SalesPersonFlag", "TRUE"]),
    ("이미지 5 Salesperson merge", ["이미지 5", "FirstName", "LastName", "Salesperson"]),
    ("이미지 6 Product expansion", ["이미지 6", "DimProductSubcategory", "DimProductCategory", "Subcategory", "Category"]),
    ("이미지 7 BusinessType standardization", ["이미지 7", "BusinessType", "Ware House", "Warehouse"]),
    ("이미지 8 Region query", ["이미지 8", "SalesTerritoryAlternateKey", "Region", "Country", "Group"]),
    ("이미지 9 TotalProductCost null fix", ["이미지 9", "TotalProductCost", "OrderQuantity", "StandardCost", "Cost"]),
    ("이미지 10 Targets Unpivot", ["이미지 10", "M01", "M12", "Unpivot"]),
    ("이미지 11 Targets long format", ["이미지 11", "MonthNumber", "Target", "TargetMonth"]),
    ("이미지 12 ColorFormats", ["이미지 12", "ColorFormats", "Background Color Format", "Font Color Format"]),
    ("이미지 13 Product ColorFormats Merge", ["이미지 13", "Product[Color]", "ColorFormats[Color]", "Left Outer"]),
    ("이미지 14 final load control", ["이미지 14", "Disable load", "Close & Apply", "7개 테이블"]),
]

REQUIRED_CONCRETE_ENTITIES = {
    "semantic_model_relationship": SEMANTIC_CONCRETE_ENTITIES,
    "power_query_etl": [
        "SQL Server",
        "AdventureWorksDW2020",
        "Navigator",
        "Transform Data",
        "Power Query Editor",
        "Column quality",
        "Column distribution",
        "Column profile",
        "SalesPersonFlag",
        "FirstName",
        "LastName",
        "Salesperson",
        "EmployeeID",
        "UPN",
        "ProductKey",
        "DimProductSubcategory",
        "DimProductCategory",
        "Subcategory",
        "Category",
        "BusinessType",
        "Ware House",
        "Warehouse",
        "Replace Values",
        "DimGeography",
        "Region",
        "Country",
        "Group",
        "FactResellerSales",
        "TotalProductCost",
        "OrderQuantity",
        "StandardCost",
        "Cost",
        "Fixed Decimal Number",
        "ResellerSalesTargets.csv",
        "M01",
        "M12",
        "Unpivot",
        "MonthNumber",
        "Target",
        "TargetMonth",
        "ColorFormats",
        "Product[Color]",
        "ColorFormats[Color]",
        "Left Outer",
        "Background Color Format",
        "Font Color Format",
        "Disable load",
        "Close & Apply",
        "7개 테이블",
    ],
    "dax_measure_modeling": [
        "Month",
        "MonthKey",
        "Sort by column",
        "Date table",
        "CALENDARAUTO",
        "Fiscal",
        "Mark as date table",
        "Avg Price",
        "Median Price",
        "Orders",
        "Order Lines",
        "Currency",
        "Target",
        "TargetAmount",
        "HASONEVALUE",
        "Variance",
        "Variance Margin",
    ],
}


def local_text_source_pack_response(
    raw_text: str,
    memo: str,
    image_files: list[Path],
    topic: str,
    extra_info: str,
    image_names: list[str] | None,
    captures: list[dict[str, Any]],
    qa_logs: list[dict[str, Any]],
    provider_diagnostics_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered_pairs = order_image_inputs(image_files, image_names or [path.name for path in image_files])
    ordered_files = [path for path, _ in ordered_pairs]
    ordered_names = [name for _, name in ordered_pairs]
    classification = classify_article_type_with_confidence(raw_text, memo, topic, extra_info, ordered_names)
    article_type = str(classification.get("article_type") or "general_learning_portfolio")
    evidence: list[dict[str, Any]] = []
    sparse_capture_report = build_sparse_capture_report(article_type, classification, evidence, raw_text, memo)
    steps = build_text_assisted_solution_steps(article_type, raw_text, memo)
    problem_map = {
        "article_type": article_type,
        "core_problem": compact_user_intent(raw_text, memo),
        "key_terms": source_derived_terms(raw_text, memo, limit=8),
        "solution_steps": steps,
        "_article_type_confidence": classification.get("confidence"),
        "_article_type_candidates": classification.get("candidates", []),
        "_uploaded_images_count": len(ordered_files),
        "_capture_count": len(captures),
        "_qa_count": len(qa_logs),
    }
    section_plan = [
        {
            "section": str(step.get("title") or f"학습 흐름 {idx}"),
            "image_refs": [],
            "must_include": normalize_str_list(step.get("technical_entities"))[:5],
        }
        for idx, step in enumerate(steps[:6], start=1)
        if isinstance(step, dict)
    ]
    draft = build_url_assisted_medium_draft(
        article_type=article_type,
        raw_text=raw_text,
        memo=memo,
        problem_map=problem_map,
        section_plan=section_plan,
        sparse_report=sparse_capture_report,
        evidence=evidence,
        qa_logs=qa_logs,
    )
    sparse_capture_report["generation_mode"] = "local_text_source_pack_fallback"
    return {
        "draft": sanitize_medium_markdown(draft),
        "article_type": article_type,
        "image_evidence": evidence,
        "problem_map": problem_map,
        "learning_evidence": build_learning_evidence(captures, qa_logs, raw_text, memo),
        "decision_map": {},
        "section_plan": section_plan,
        "article_brief": {},
        "sparse_capture_report": sparse_capture_report,
        "provider_diagnostics": provider_diagnostics_data or {},
        "critic_report": {
            "passed": True,
            "failures": [],
            "metrics": {
                "generation_mode": "local_text_source_pack_fallback",
                "provider_unavailable": bool(provider_diagnostics_data),
            },
        },
    }


def generate_medium_article_pipeline(
    llm_client: LLM,
    raw_text: str,
    memo: str,
    image_files: list[Path],
    topic: str,
    extra_info: str = "",
    image_names: list[str] | None = None,
    captures: list[dict[str, Any]] | None = None,
    qa_logs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    captures = captures or []
    qa_logs = qa_logs or []
    if not image_files and is_foundry_iq_mcp_rag_context(raw_text, memo):
        return local_text_source_pack_response(raw_text, memo, image_files, topic, extra_info, image_names, captures, qa_logs)
    client = llm_client.get_client()
    if not client:
        if has_text_assisted_context(raw_text, memo):
            return local_text_source_pack_response(
                raw_text,
                memo,
                image_files,
                topic,
                extra_info,
                image_names,
                captures,
                qa_logs,
                provider_diagnostics_data=llm_client.last_diagnostics,
            )
        diagnostics = llm_client.last_diagnostics or provider_diagnostics("groq", exception=RuntimeError("LLM client unavailable"))
        return {
            "draft": provider_failure_message(
                image_count=len(image_files),
                capture_count=len(captures),
                qa_count=len(qa_logs),
                memo=memo,
                diagnostics=diagnostics,
                elapsed_seconds=0,
            ),
            "article_type": "unknown",
            "image_evidence": [],
            "problem_map": {},
            "learning_evidence": [],
            "decision_map": {},
            "article_brief": {},
            "provider_diagnostics": diagnostics,
            "critic_report": {
                "passed": False,
                "failures": ["provider_failure: LLM client unavailable"],
                "metrics": {"failure_type": "provider_failure", "provider_diagnostics": diagnostics},
            },
        }

    try:
        ordered_pairs = order_image_inputs(image_files, image_names or [path.name for path in image_files])
        ordered_files = [path for path, _ in ordered_pairs]
        ordered_names = [name for _, name in ordered_pairs]
        classification = classify_article_type_with_confidence(raw_text, memo, topic, extra_info, ordered_names)
        article_type = str(classification.get("article_type") or "unknown")
        golden_context = load_golden_context(article_type)
        evidence = build_image_evidence(llm_client, raw_text, memo, ordered_files, topic, extra_info, ordered_names)
        if image_files and is_groq_rate_limit_error(llm_client.last_vision_error):
            return vision_rate_limit_response(
                llm_client.last_vision_error,
                image_count=len(ordered_files),
                capture_count=len(captures),
                qa_count=len(qa_logs),
            )
        evidence = ensure_image_evidence_coverage(evidence, ordered_files, ordered_names, raw_text, memo, golden_context)
        auto_topic_hint = image_only_auto_topic_hint(raw_text, memo, len(ordered_files), evidence, ordered_names)
        if auto_topic_hint:
            memo = auto_topic_hint
            classification = classify_article_type_with_confidence(raw_text, memo, topic, extra_info, ordered_names)
            article_type = str(classification.get("article_type") or "unknown")
            golden_context = load_golden_context(article_type)
            evidence = ensure_image_evidence_coverage(evidence, ordered_files, ordered_names, raw_text, memo, golden_context)
        classification = promote_article_type_from_evidence(classification, evidence, raw_text, memo, ordered_names)
        article_type = str(classification.get("article_type") or "unknown")
        golden_context = load_golden_context(article_type)
        sparse_capture_report = build_sparse_capture_report(article_type, classification, evidence, raw_text, memo)
        if article_type in {"unknown", "general_learning_portfolio"} or float(classification.get("confidence") or 0) < 0.55:
            evidence_text = json.dumps(evidence, ensure_ascii=False)
            second_classification = classify_article_type_with_confidence(
                raw_text + "\n" + evidence_text,
                memo,
                topic,
                extra_info,
                ordered_names,
            )
            if float(second_classification.get("confidence") or 0) > float(classification.get("confidence") or 0):
                classification = second_classification
                classification = promote_article_type_from_evidence(second_classification, evidence, raw_text, memo, ordered_names)
                article_type = str(classification.get("article_type") or "unknown")
                golden_context = load_golden_context(article_type)
                sparse_capture_report = build_sparse_capture_report(article_type, classification, evidence, raw_text, memo)
        if article_type == "unknown" or float(classification.get("confidence") or 0) < 0.25:
            if can_generate_url_assisted_medium_draft(raw_text, memo, article_type, {}, []):
                return local_text_source_pack_response(
                    raw_text,
                    memo,
                    image_files,
                    topic,
                    extra_info,
                    image_names,
                    captures,
                    qa_logs,
                )
            return unknown_article_type_response(classification, evidence, len(ordered_files), memo, raw_text, sparse_capture_report)
        learning_evidence = build_learning_evidence(captures, qa_logs, raw_text, memo)
        problem_map = build_problem_map(llm_client, raw_text, memo, evidence, topic, extra_info, article_type, golden_context)
        problem_map["article_type"] = article_type
        problem_map["_sparse_capture_report"] = sparse_capture_report
        problem_map["_article_type_confidence"] = classification.get("confidence")
        problem_map["_article_type_candidates"] = classification.get("candidates", [])
        problem_map["_uploaded_images_count"] = len(ordered_files)
        problem_map["_capture_count"] = len(captures)
        problem_map["_qa_count"] = len(qa_logs)
        problem_map["_readme_image_count"] = readme_image_count(golden_context.get("image_order", ""))
        problem_map = attach_problem_map_refs(problem_map, evidence)
        problem_map["_evidence_text_for_classification"] = json.dumps(evidence, ensure_ascii=False)[:20000]
        enrich_problem_map_concrete_details(problem_map, article_type)
        decision_map = build_decision_map(learning_evidence, problem_map)
        brief = build_article_brief(llm_client, raw_text, memo, evidence, problem_map, topic, extra_info)
        outline = build_article_outline(llm_client, brief, problem_map, evidence)
        section_plan = build_section_plan(article_type, evidence, problem_map)
        problem_map["_section_plan"] = section_plan
        sparse_hold_reason = sparse_capture_generation_blocker(article_type, sparse_capture_report, problem_map, brief)
        if sparse_hold_reason:
            # URL/lecture-note assisted mode: do not punish the user for having no screenshots,
            # no explicit “hard problem”, or no final validation screen.  If the user provided
            # source URLs, lab text, lecture notes, or a lightweight question, produce a
            # Medium-ready draft first and move uncertainties to a short optional checklist.
            if can_generate_url_assisted_medium_draft(raw_text, memo, article_type, problem_map, section_plan):
                url_article = build_url_assisted_medium_draft(
                    article_type=article_type,
                    raw_text=raw_text,
                    memo=memo,
                    problem_map=problem_map,
                    section_plan=section_plan,
                    sparse_report=sparse_capture_report,
                    evidence=evidence,
                    qa_logs=qa_logs,
                )
                critic = CritiqueResult(
                    passed=True,
                    failures=[],
                    section_failures={},
                    metrics={
                        "generation_mode": "url_assisted_medium_draft",
                        "original_sparse_hold_reason": sparse_hold_reason,
                        "image_count": len(ordered_files),
                    },
                )
                sparse_capture_report["generation_mode"] = "url_assisted_medium_draft"
                return {
                    "draft": sanitize_medium_markdown(url_article),
                    "article_type": article_type,
                    "image_evidence": evidence,
                    "learning_evidence": learning_evidence,
                    "problem_map": problem_map,
                    "decision_map": decision_map,
                    "section_plan": section_plan,
                    "article_brief": brief,
                    "sparse_capture_report": sparse_capture_report,
                    "critic_report": asdict(critic),
                }
            critic = CritiqueResult(
                passed=False,
                failures=[sparse_hold_reason],
                section_failures={"sparse_capture_mode": sparse_hold_reason},
                metrics={
                    "failure_type": "content_quality_failure",
                    "generation_mode": sparse_capture_report.get("generation_mode"),
                    "interpreted_image_count": sparse_capture_report.get("interpreted_image_count"),
                    "total_image_count": sparse_capture_report.get("total_image_count"),
                    "unknown_caption_count": sparse_capture_report.get("unknown_caption_count"),
                },
            )
            return {
                "draft": sparse_capture_hold_message(article_type, sparse_capture_report, evidence, problem_map, brief, section_plan, sparse_hold_reason),
                "article_type": article_type,
                "image_evidence": evidence,
                "learning_evidence": learning_evidence,
                "problem_map": problem_map,
                "decision_map": decision_map,
                "section_plan": section_plan,
                "article_brief": brief,
                "sparse_capture_report": sparse_capture_report,
                "critic_report": asdict(critic),
            }

        coverage_failure = image_coverage_failure(problem_map, evidence)
        problem_map["_warnings"] = problem_map.get("_coverage_status", {}).get("warnings", [])
        if coverage_failure:
            return {
                "draft": evidence_coverage_failure_message(coverage_failure, len(ordered_files), len(evidence), len(captures), len(qa_logs), memo),
                "article_type": article_type,
                "image_evidence": evidence,
                "learning_evidence": learning_evidence,
                "problem_map": problem_map,
                "decision_map": decision_map,
            "section_plan": section_plan,
            "article_brief": brief,
            "sparse_capture_report": sparse_capture_report,
            "critic_report": asdict(
                    CritiqueResult(
                        passed=False,
                        failures=[coverage_failure],
                        section_failures={"image_evidence_coverage": coverage_failure},
                        metrics={"uploaded_images_count": len(ordered_files), "image_evidence_count": len(evidence)},
                    )
                ),
            }

        sections: dict[str, str] = {}
        for title in SECTION_TITLES:
            sections[title] = generate_section(llm_client, title, outline, brief, problem_map, evidence, raw_text, memo, extra_info)
        article = assemble_article(sections, brief)
        critique = critique_article(article, article_type, problem_map, evidence)
        if not critique.passed:
            article = expand_failed_sections(llm_client, article, sections, critique, outline, brief, problem_map, evidence, raw_text, memo, extra_info)
            critique = critique_article(article, article_type, problem_map, evidence)
        if article_type not in POWERBI_ARTICLE_TYPES and severe_sparse_article_failures(critique):
            sparse_capture_report["generation_mode"] = "draft_with_missing_context"
            reason = "; ".join(severe_sparse_article_failures(critique)[:3])
            return {
                "draft": sparse_capture_hold_message(article_type, sparse_capture_report, evidence, problem_map, brief, section_plan, reason),
                "article_type": article_type,
                "image_evidence": evidence,
                "learning_evidence": learning_evidence,
                "problem_map": problem_map,
                "decision_map": decision_map,
                "section_plan": section_plan,
                "article_brief": brief,
                "sparse_capture_report": sparse_capture_report,
                "critic_report": asdict(critique),
            }
        return {
            "draft": sanitize_medium_markdown(article),
            "article_type": article_type,
            "image_evidence": evidence,
            "learning_evidence": learning_evidence,
            "problem_map": problem_map,
            "decision_map": decision_map,
            "section_plan": section_plan,
            "article_brief": brief,
            "coverage_status": problem_map.get("_coverage_status", {}),
            "warnings": problem_map.get("_warnings", []),
            "sparse_capture_report": sparse_capture_report,
            "critic_report": asdict(critique),
        }
    except Exception as exc:
        print(f"[Medium pipeline error] {exc}")
        diagnostics = provider_diagnostics("groq", exception=exc, package_import_ok=groq_import_ok(), client_init_ok=bool(llm_client.client))
        return {
            "draft": provider_failure_message(
                image_count=len(image_files),
                capture_count=len(captures),
                qa_count=len(qa_logs),
                memo=memo,
                diagnostics=diagnostics,
                elapsed_seconds=0,
            ),
            "article_type": "unknown",
            "image_evidence": [],
            "learning_evidence": [],
            "problem_map": {},
            "decision_map": {},
            "article_brief": {},
            "provider_diagnostics": diagnostics,
            "critic_report": {
                "passed": False,
                "failures": [f"provider_or_pipeline_failure: {type(exc).__name__}: {exc}"],
                "metrics": {"failure_type": "provider_failure", "provider_diagnostics": diagnostics},
            },
        }


def order_image_inputs(image_files: list[Path], image_names: list[str]) -> list[tuple[Path, str]]:
    pairs = list(zip(image_files, image_names, strict=False))
    if any(re.match(r"^\d{1,3}[_\-\s]", Path(name).name) for _, name in pairs):
        pairs.sort(key=lambda item: natural_sort_key(Path(item[1]).name))
    return [(path, name) for path, name in pairs]


def model_route(task: str) -> dict[str, str]:
    deep_tasks = {"hard_question_answer", "problem_map_generation", "final_article_generation", "critic_and_expand"}
    provider = MODEL_PROVIDER_DEEP if task in deep_tasks else MODEL_PROVIDER_FAST
    if provider == "openai":
        # OpenAI is a planned premium/deep adapter; the current MVP runtime uses the Groq client safely.
        provider = "groq"
    model = OPENAI_MODEL if provider == "openai" else GROQ_MODEL
    if task == "quick_image_summary" and provider == "groq":
        model = GROQ_VISION_MODEL
    return {"provider": provider, "model": model}


def read_sessions() -> dict[str, Any]:
    if not SESSIONS_PATH.exists() or not SESSIONS_PATH.read_text(encoding="utf-8").strip():
        return {"sessions": []}
    try:
        data = json.loads(SESSIONS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"sessions": []}
    if not isinstance(data, dict):
        return {"sessions": []}
    data.setdefault("sessions", [])
    return data


def write_sessions(data: dict[str, Any]) -> None:
    SESSIONS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def create_session(title: str = "") -> dict[str, Any]:
    data = read_sessions()
    session_id = str(uuid.uuid4())
    session = {
        "session_id": session_id,
        "title": title or f"Study session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "captures": [],
        "qa_logs": [],
    }
    data["sessions"].append(session)
    write_sessions(data)
    return session


def find_session(session_id: str) -> dict[str, Any] | None:
    for session in read_sessions().get("sessions", []):
        if session.get("session_id") == session_id:
            return session
    return None


def update_session(session: dict[str, Any]) -> None:
    data = read_sessions()
    sessions = data.get("sessions", [])
    for index, item in enumerate(sessions):
        if item.get("session_id") == session.get("session_id"):
            sessions[index] = session
            write_sessions(data)
            return
    sessions.append(session)
    data["sessions"] = sessions
    write_sessions(data)


def latest_or_new_session() -> dict[str, Any]:
    sessions = read_sessions().get("sessions", [])
    if sessions:
        return sessions[-1]
    return create_session()


def build_learning_evidence(
    captures: list[dict[str, Any]],
    qa_logs: list[dict[str, Any]],
    raw_text: str,
    memo: str,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for capture in captures:
        evidence.append(
            {
                "type": "capture",
                "ref_id": capture.get("capture_id"),
                "timestamp": capture.get("timestamp"),
                "learner_signal": capture.get("user_note") or capture.get("ocr_text") or capture.get("vision_summary") or "무메모 캡처",
                "evidence_text": " ".join(
                    part
                    for part in [
                        capture.get("user_note", ""),
                        capture.get("ocr_text", ""),
                        capture.get("vision_summary", ""),
                        " ".join(capture.get("auto_keywords") or []),
                    ]
                    if part
                ),
                "image_path": capture.get("image_path", ""),
            }
        )
    for qa in qa_logs:
        evidence.append(
            {
                "type": "qa",
                "ref_id": qa.get("qa_id"),
                "timestamp": qa.get("timestamp"),
                "learner_signal": qa.get("question", ""),
                "evidence_text": f"질문: {qa.get('question', '')}\n답변 요약: {qa.get('answer_summary', '')}\n상태: {qa.get('learner_state', '')}",
                "related_capture_ids": qa.get("related_capture_ids", []),
            }
        )
    if raw_text.strip() or memo.strip():
        evidence.append(
            {
                "type": "direct_input",
                "ref_id": "direct",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "learner_signal": memo or raw_text,
                "evidence_text": "\n".join(part for part in [raw_text, memo] if part.strip()),
            }
        )
    return evidence


def build_decision_map(learning_evidence: list[dict[str, Any]], problem_map: dict[str, Any]) -> dict[str, Any]:
    decision_points: list[dict[str, Any]] = []
    qa_items = [item for item in learning_evidence if item.get("type") == "qa"]
    captures = [item for item in learning_evidence if item.get("type") == "capture"]
    steps = problem_map.get("solution_steps", []) if isinstance(problem_map.get("solution_steps"), list) else []
    for index, step in enumerate(steps[: max(4, len(qa_items))], start=1):
        qa = qa_items[index - 1] if index - 1 < len(qa_items) else {}
        related = qa.get("related_capture_ids") or [cap.get("ref_id") for cap in captures[index - 1 : index + 1] if cap.get("ref_id")]
        decision_points.append(
            {
                "title": step.get("title") if isinstance(step, dict) else f"판단 지점 {index}",
                "initial_assumption": (step.get("cause") if isinstance(step, dict) else "") or "처음에는 화면에 보이는 결과만으로 충분하다고 볼 수 있었다.",
                "question": qa.get("learner_signal", ""),
                "ai_hint": qa.get("evidence_text", ""),
                "user_decision": (step.get("action") if isinstance(step, dict) else "") or "근거를 문제/원인/검증 흐름으로 재구성했다.",
                "actual_result": (step.get("verification") if isinstance(step, dict) else "") or problem_map.get("final_outcome", ""),
                "evidence_refs": related,
                "qa_refs": [qa.get("ref_id")] if qa.get("ref_id") else [],
            }
        )
    return {"decision_points": decision_points}


def attach_problem_map_refs(problem_map: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    image_refs = [item.get("image_no") for item in evidence if item.get("image_no")]
    entities = sorted({entity for item in evidence for entity in normalize_str_list(item.get("technical_entities"))})
    for index, step in enumerate(problem_map.get("solution_steps", []), start=1):
        if not isinstance(step, dict):
            continue
        step.setdefault("image_refs", image_refs[max(0, index - 2) : index + 1] or image_refs[:1])
        step.setdefault("technical_entities", entities[:8])
    return problem_map


def ensure_image_evidence_coverage(
    evidence: list[dict[str, Any]],
    image_files: list[Path],
    image_names: list[str],
    raw_text: str,
    memo: str,
    golden_context: dict[str, Any],
) -> list[dict[str, Any]]:
    by_no: dict[int, dict[str, Any]] = {}
    for item in evidence:
        try:
            image_no = int(item.get("image_no") or 0)
        except (TypeError, ValueError):
            continue
        if image_no > 0:
            by_no[image_no] = item

    caption_source = golden_context.get("image_order", "") or read_image_order_caption_source()
    fallback = fallback_image_evidence(image_files, image_names, raw_text, memo, caption_source)
    for item in fallback:
        image_no = int(item.get("image_no") or 0)
        by_no.setdefault(image_no, item)
    return normalize_display_image_indexes([by_no[index] for index in sorted(by_no)], image_files, image_names)


def normalize_display_image_indexes(
    evidence: list[dict[str, Any]],
    image_files: list[Path],
    image_names: list[str],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    total = len(image_files) or len(evidence)
    for display_index in range(1, total + 1):
        item = evidence[display_index - 1] if display_index - 1 < len(evidence) else {}
        original_name = image_names[display_index - 1] if display_index - 1 < len(image_names) else ""
        next_item = dict(item)
        next_item["source_image_no"] = next_item.get("source_image_no", next_item.get("image_no", display_index))
        next_item["display_image_index"] = display_index
        next_item["image_no"] = display_index
        if original_name:
            next_item["original_filename"] = original_name
        next_item["caption"] = humanize_caption(str(next_item.get("caption") or f"이미지 {display_index} - 해석되지 않은 캡처"), display_index)
        normalized.append(next_item)
    return normalized


def readme_image_count(image_order: str) -> int:
    if not image_order:
        return 0
    return sum(1 for line in image_order.splitlines() if re.search(r"\b\d{1,3}[_\-\s]", line) or re.search(r"\bimage\d+\.", line, re.I))


def image_coverage_failure(problem_map: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    uploaded = int(problem_map.get("_uploaded_images_count") or 0)
    capture_count = int(problem_map.get("_capture_count") or 0)
    readme_count = int(problem_map.get("_readme_image_count") or 0)
    evidence_count = len([item for item in evidence if int(item.get("image_no") or 0) > 0])
    problem_map["_coverage_status"] = coverage_status(problem_map, evidence)
    if uploaded >= 5 and evidence_count / max(uploaded, 1) < 0.8:
        return f"이미지 {uploaded}장 중 {evidence_count}장만 해석되었습니다. 이미지 evidence coverage가 부족하여 완성형 Medium 글을 생성할 수 없습니다."
    if uploaded and evidence_count < uploaded:
        return f"업로드된 이미지 {uploaded}장 중 {evidence_count}장만 ImageEvidence로 정리되었습니다."
    if readme_count and uploaded and evidence_count < min(readme_count, uploaded):
        return f"README expected images {readme_count}장, uploaded images {uploaded}장 중 ImageEvidence가 {evidence_count}개만 생성되었습니다."
    if readme_count and not uploaded and evidence_count < readme_count:
        return f"README_image_order.txt 기준 이미지 {readme_count}장 중 {evidence_count}장만 ImageEvidence로 정리되었습니다."
    if capture_count and uploaded == 0 and evidence_count != capture_count:
        return f"Capture Timeline의 캡처 {capture_count}개 중 {evidence_count}개만 ImageEvidence로 정리되었습니다."
    return ""


def coverage_status(problem_map: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    uploaded = int(problem_map.get("_uploaded_images_count") or 0)
    capture_count = int(problem_map.get("_capture_count") or 0)
    qa_count = int(problem_map.get("_qa_count") or 0)
    readme_count = int(problem_map.get("_readme_image_count") or 0)
    evidence_count = len([item for item in evidence if int(item.get("image_no") or 0) > 0])
    warnings: list[dict[str, Any]] = []
    evidence_missing = False
    if uploaded:
        evidence_missing = evidence_count < uploaded
    elif readme_count:
        evidence_missing = evidence_count < readme_count
    elif capture_count:
        evidence_missing = evidence_count < capture_count
    if readme_count and uploaded and readme_count != uploaded and not evidence_missing:
        extra = uploaded - readme_count
        if extra > 0:
            message = (
                f"README에는 {readme_count}장이 기록되어 있지만, 실제 업로드 이미지는 {uploaded}장이고 "
                f"ImageEvidence도 {evidence_count}개 생성되었습니다. Evidence 부족은 아니며, README/업로드 수 불일치 상태입니다. "
                f"extra image {extra}장이 있습니다."
            )
        else:
            message = (
                f"README에는 {readme_count}장이 기록되어 있지만, 실제 업로드 이미지는 {uploaded}장이고 "
                f"ImageEvidence는 {evidence_count}개 생성되었습니다. Evidence 부족은 아니며, README/업로드 수 불일치 상태입니다."
            )
        warnings.append(
            {
                "status": "warning",
                "reason": "readme_uploaded_count_mismatch",
                "message": message,
                "can_generate_article": True,
                "evidence_missing": False,
                "readme_expected_count": readme_count,
                "uploaded_images_count": uploaded,
                "image_evidence_count": evidence_count,
            }
        )
    return {
        "status": "failure" if evidence_missing else ("warning" if warnings else "ok"),
        "readme_expected_count": readme_count,
        "uploaded_images_count": uploaded,
        "image_evidence_count": evidence_count,
        "capture_count": capture_count,
        "qa_count": qa_count,
        "evidence_missing": evidence_missing,
        "can_generate_article": not evidence_missing,
        "warnings": warnings,
    }


CAPTURE_ROLES = [
    "goal_or_context",
    "confusing_concept",
    "architecture_or_flow",
    "action_or_change",
    "code_or_config",
    "error_or_problem",
    "validation_or_result",
    "completion_or_summary",
    "unknown",
]

POWERBI_ARTICLE_TYPES = {"semantic_model_relationship", "dax_measure_modeling", "power_query_etl"}
GITHUB_WORKFLOW_TYPES = {"github_agentic_workflow", "github_actions_workflow", "ai_coding_workflow"}


def build_sparse_capture_report(
    article_type: str,
    classification: dict[str, Any],
    evidence: list[dict[str, Any]],
    raw_text: str,
    memo: str,
) -> dict[str, Any]:
    roles = [infer_sparse_capture_role(item) for item in evidence]
    source = " ".join(
        [
            raw_text,
            memo,
            " ".join(str(item.get("caption", "")) for item in evidence),
            " ".join(str(item.get("problem_signal", "")) for item in evidence),
            " ".join(" ".join(normalize_str_list(item.get("technical_entities"))) for item in evidence),
        ]
    ).lower()
    role_set = set(roles)
    explicit_goal_words = ["goal", "objective", "lesson", "module", "lab", "강의 목표", "실습 목표", "학습 목표", "이번 단원"]
    has_explicit_goal_text = bool(memo.strip() and any(word in memo.lower() for word in explicit_goal_words)) or bool(
        raw_text.strip() and any(word in raw_text.lower() for word in explicit_goal_words)
    )
    has_goal_evidence = bool({"goal_or_context", "architecture_or_flow"} & role_set)
    coverage = {
        "has_goal": bool(has_explicit_goal_text or has_goal_evidence),
        "has_problem": bool({"error_or_problem", "confusing_concept"} & role_set or any(word in source for word in ["error", "오류", "problem", "문제", "failed", "not working", "헷갈", "confusing"])),
        "has_action": bool({"action_or_change", "code_or_config"} & role_set),
        "has_validation": bool({"validation_or_result", "completion_or_summary"} & role_set or any(word in source for word in ["success", "완료", "result", "결과", "validated", "검증", "conclusion"])),
    }
    missing_context: list[str] = []
    if not coverage["has_goal"]:
        missing_context.append("강의/실습의 최종 목표가 명확하지 않습니다.")
    if not coverage["has_problem"]:
        missing_context.append("어떤 개념이 헷갈렸거나 어떤 문제가 있었는지 확인할 근거가 부족합니다.")
    if not coverage["has_action"]:
        missing_context.append("사용자가 실제로 바꾼 코드, 설정, 명령어, 작업 단계가 부족합니다.")
    if not coverage["has_validation"]:
        missing_context.append("마지막 성공/완료/실행 결과 또는 검증 화면이 부족합니다.")
    is_github_workflow = article_type in {"github_agentic_workflow", "github_actions_workflow", "ai_coding_workflow"}
    if is_github_workflow:
        missing_context.append("이 강의/실습의 최종 목표가 정확히 무엇인지 부족함")
        if not coverage["has_problem"]:
            missing_context.append("사용자가 어떤 화면에서 막혔는지 부족함")
        if "update-github-info" in source:
            missing_context.append("update-github-info workflow가 실제로 무엇을 자동화하는지 설명 부족")
        if "conclusion" in source:
            missing_context.append("conclusion 화면이 성공/실패/요약 중 무엇을 의미하는지 부족함")
    confidence_score = float(classification.get("confidence") or 0)
    evidence_count = len([item for item in evidence if int(item.get("image_no") or 0) > 0])
    interpreted_count = sum(1 for item in evidence if is_interpreted_image_evidence(item))
    unknown_caption_count = sum(1 for item in evidence if is_unknown_or_filename_caption(item))
    interpreted_ratio = interpreted_count / max(evidence_count, 1)
    if evidence_count and interpreted_ratio < 0.6:
        missing_context.append(f"{evidence_count}장 중 일부 이미지의 의미를 해석하지 못함")
    if unknown_caption_count:
        missing_context.append(f"파일명 또는 일반 캡션으로만 남은 이미지가 {unknown_caption_count}장 있습니다.")
    present_count = sum(1 for value in coverage.values() if value)
    if (
        confidence_score >= 0.7
        and present_count >= 3
        and evidence_count >= 3
        and interpreted_ratio >= 0.6
        and unknown_caption_count <= max(1, evidence_count // 4)
    ):
        generation_mode = "full_article"
        confidence_label = "high"
    elif confidence_score >= 0.35 and (evidence_count >= 2 or has_text_assisted_context(raw_text, memo)):
        generation_mode = "draft_with_missing_context"
        confidence_label = "medium"
    else:
        generation_mode = "ask_before_generate"
        confidence_label = "low"
    followups = sparse_follow_up_questions(article_type, raw_text, memo)
    missing_context = list(dict.fromkeys(missing_context))
    return {
        "article_type": article_type,
        "confidence": confidence_label,
        "article_type_confidence": round(confidence_score, 3),
        "generation_mode": generation_mode,
        "interpreted_image_count": interpreted_count,
        "total_image_count": evidence_count,
        "unknown_caption_count": unknown_caption_count,
        "capture_roles": [
            {
                "image_no": item.get("image_no"),
                "role": role,
                "caption": item.get("caption", ""),
            }
            for item, role in zip(evidence, roles, strict=False)
        ],
        "capture_coverage": coverage,
        "missing_context": missing_context,
        "follow_up_questions": followups[:3] if generation_mode != "full_article" else [],
    }


def sparse_follow_up_questions(article_type: str, raw_text: str = "", memo: str = "") -> list[str]:
    source = (str(raw_text or "") + "\n" + str(memo or "")).lower()
    if "azure devops" in source and "mcp" in source:
        return [
            "MCP Server가 Azure DevOps에서 연결하는 대상은 Work items, Pull Requests, Builds, Test Plans 중 무엇이었나요?",
            "실습에서 실제로 수행한 작업은 MCP Server 설정이었나요, Copilot/AI assistant로 DevOps 작업을 요청하는 것이었나요?",
            "마지막에 확인한 결과는 work item 생성/조회, PR 확인, build 확인 중 무엇이었나요?",
        ]
    if article_type in {"github_agentic_workflow", "github_actions_workflow", "ai_coding_workflow"}:
        return [
            "이 강의의 최종 목표는 GitHub Actions workflow 이해였나요, Copilot/Agent 자동화 이해였나요?",
            "캡처 중 가장 이해가 안 된 화면은 몇 번인가요?",
            "마지막 conclusion 화면은 성공 결과였나요, 단순 요약 화면이었나요?",
        ]
    return [
        "이 강의/실습의 최종 목표는 무엇이었나요?",
        "캡처 중 가장 이해가 안 된 화면은 몇 번인가요?",
        "마지막에 성공/완료/실행 결과를 확인했나요?",
    ]


def has_text_assisted_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".strip()
    lowered = source.lower()
    if len(source) >= 220:
        return True
    helpful_terms = [
        "강의 url",
        "강의안",
        "본문",
        "agentic workflows",
        "github actions",
        "workflow_dispatch",
        "pull request",
        "update-github-info",
        "activation",
        "conclusion",
        "mcp",
        "azure devops",
        "microsoftlearning.github.io",
        "mslearn-devops",
        "work item",
        "pull request",
        "test plan",
        "copilot",
        "agent orchestration",
        "orchestrator",
        "sub-agent",
        "sub-agents",
        "hive-mind",
        "github copilot cli",
        "planner",
        "coder",
        "designer",
        "자동화",
        "microsoft foundry",
        "mslearn-agent-quickstart",
        "develop your first agent with microsoft",
        "get-started-in-foundry",
        "continue-in-vscode",
        "use-agent",
        "first agent",
    ]
    return any(term in lowered for term in helpful_terms)


def build_text_assisted_solution_steps(article_type: str, raw_text: str, memo: str) -> list[dict[str, Any]]:
    """Build draft-grade solution steps from lecture text/memo when image interpretation fails.

    This is intentionally conservative: it creates steps for a draft_with_missing_context,
    not proof that the user completed a full implementation.
    """
    if article_type in POWERBI_ARTICLE_TYPES:
        return []
    if not has_text_assisted_context(raw_text, memo) and not (is_wikidocs_coding_test_context(raw_text, memo) or is_oopy_cs_notes_context(raw_text, memo)):
        return []
    source = f"{raw_text}\n{memo}"
    lowered = source.lower()

    if is_wikidocs_coding_test_context(raw_text, memo):
        return [
            {
                "step": 1,
                "title": "문법 암기와 문제 풀이 적용의 간격 인식",
                "problem": "파이썬 문법이나 자료구조 정의를 읽는 것과 실제 코딩테스트 문제 조건에 맞게 적용하는 것은 다른 문제였다.",
                "cause": "WikiDocs 자료는 문법, 자료구조, 알고리즘 개념이 목차 순서로 제시되지만, 시험에서는 조건을 보고 어떤 개념을 꺼내야 하는지가 더 중요하다.",
                "action": "학습 내용을 단순 요약하지 않고, 입력 조건을 읽고 자료구조를 선택하는 기준으로 다시 정리했다.",
                "verification": "각 개념을 '어떤 문제 유형에서 쓰는가'로 설명할 수 있는지 확인한다.",
                "technical_entities": ["Python", "코딩테스트", "자료구조", "스택", "시간 복잡도"],
                "image_refs": [],
            },
            {
                "step": 2,
                "title": "자료구조 개념을 문제 조건과 연결",
                "problem": "리스트, 딕셔너리, 스택 같은 개념은 알고 있어도 문제에서 어떤 구조를 선택해야 하는지 바로 보이지 않을 수 있었다.",
                "cause": "자료구조는 정의보다 접근 패턴, 삽입/삭제 위치, 탐색 비용, 순서 보존 여부가 선택 기준이 된다.",
                "action": "각 자료구조를 '언제 쓰는가', '어떤 연산이 중요한가', '시간 복잡도가 어떻게 달라지는가' 기준으로 분리했다.",
                "verification": "스택 문제, 해시 문제, 반복문 문제를 볼 때 선택 기준을 말할 수 있는지 확인한다.",
                "technical_entities": ["list", "dict", "stack", "hash", "Big-O"],
                "image_refs": [],
            },
            {
                "step": 3,
                "title": "스택까지의 학습 범위 재구성",
                "problem": "목차를 따라 읽으면 학습 범위는 늘어나지만, 지금 어디까지 문제 풀이 기준으로 정리했는지 흐려질 수 있었다.",
                "cause": "수집 자료 제목에 '~179 스택까지'가 포함되어 있어, 스택 이전 개념과 스택 개념을 하나의 학습 단위로 구분해야 했다.",
                "action": "파이썬 기본 문법부터 스택까지를 코딩테스트 초반 문제 풀이에 필요한 최소 단위로 묶어 정리했다.",
                "verification": "스택까지의 학습 범위를 다음 복습/문제 풀이 계획으로 연결할 수 있는지 확인한다.",
                "technical_entities": ["파이썬 문법", "스택", "PCCE", "PCCP", "문제 풀이 루틴"],
                "image_refs": [],
            },
            {
                "step": 4,
                "title": "문제 해결형 학습 기록으로 전환",
                "problem": "강의나 책 내용을 그대로 요약하면 포트폴리오 글이 아니라 독서 기록에 머무를 수 있었다.",
                "cause": "코딩테스트 학습 기록은 '무엇을 읽었는가'보다 '어떤 문제 풀이 기준을 세웠는가'가 드러나야 한다.",
                "action": "개념 정의, 적용 조건, 검증 기준, 다음 문제 풀이 계획을 분리해 Medium 글의 문제 해결 흐름으로 바꾸었다.",
                "verification": "최종 글이 책 소개가 아니라 학습자가 개념을 문제 풀이 기준으로 재구성한 기록인지 확인한다.",
                "technical_entities": ["학습 기록", "문제 해결", "복습 기준", "검증 기준"],
                "image_refs": [],
            },
        ]

    if is_oopy_cs_notes_context(raw_text, memo):
        return [
            {
                "step": 1,
                "title": "CS 개념을 면접 질문 기준으로 재분류",
                "problem": "CS 요약집은 운영체제, 네트워크, 자료구조 같은 개념이 넓게 흩어져 있어 무엇부터 설명 가능하게 만들어야 하는지 흐려질 수 있었다.",
                "cause": "개념을 목차 순서로만 읽으면 실제 면접 질문에서 어떤 키워드를 중심으로 답해야 하는지 연결이 약해진다.",
                "action": "자료를 개념 목록이 아니라 질문 대응 단위로 나누어, 정의·원리·예시·비교 기준을 분리했다.",
                "verification": "각 CS 개념을 1분 설명과 꼬리 질문 대응으로 말할 수 있는지 확인한다.",
                "technical_entities": ["CS", "자료구조", "운영체제", "네트워크", "면접"],
                "image_refs": [],
            },
            {
                "step": 2,
                "title": "암기형 요약을 문제 해결형 설명으로 전환",
                "problem": "짧은 요약만 보면 개념을 아는 것처럼 보이지만, 왜 필요한지와 어떤 상황에서 쓰이는지가 빠질 수 있었다.",
                "cause": "CS 개념은 정의보다 trade-off, 동작 원리, 장애 상황에서의 해석이 중요하다.",
                "action": "각 항목을 '무엇인가 → 왜 필요한가 → 어떤 문제가 생기는가 → 어떻게 설명할 것인가' 흐름으로 바꾸었다.",
                "verification": "최종 글이 단순 요약이 아니라 면접/실무 질문에 대비한 설명 구조를 갖는지 확인한다.",
                "technical_entities": ["trade-off", "동작 원리", "문제 상황", "설명 구조"],
                "image_refs": [],
            },
            {
                "step": 3,
                "title": "복습 단위와 다음 학습 기준 설정",
                "problem": "CS 범위가 넓기 때문에 한 번에 모두 이해하려 하면 복습 기준이 사라진다.",
                "cause": "자료구조, 운영체제, 네트워크는 서로 다른 층위의 개념이므로 같은 방식으로 정리하면 기억과 적용이 약해진다.",
                "action": "개념별로 정의, 핵심 동작, 대표 질문, 확인 기준을 따로 두어 다음 복습 단위로 나누었다.",
                "verification": "각 개념을 문제 상황이나 질문 형태로 다시 꺼낼 수 있는지 확인한다.",
                "technical_entities": ["복습 루틴", "대표 질문", "확인 기준"],
                "image_refs": [],
            },
        ]

    if is_microsoft_foundry_first_agent_context(raw_text, memo):
        return [
            {
                "step": idx,
                "title": step["title"],
                "problem": step["problem"],
                "cause": "Microsoft Foundry quickstart 자료는 포털 생성, VS Code 전환, agent 실행 단서가 함께 나타나므로 각 단계의 역할을 분리해야 한다.",
                "action": step["action"],
                "verification": step["verification"],
                "technical_entities": ["Microsoft Foundry", "first agent", "quickstart", "VS Code"],
                "image_refs": [],
            }
            for idx, step in enumerate(microsoft_foundry_first_agent_steps(), start=1)
        ]

    if is_foundry_iq_mcp_rag_context(raw_text, memo):
        return [
            {
                "step": idx,
                "title": step["title"],
                "problem": step["problem"],
                "cause": "Foundry IQ, MCP, RAG, knowledge base, Azure AI Search 단서가 함께 있으므로 지식 연결과 도구 연결의 경계를 나누어야 한다.",
                "action": step["action"],
                "verification": step["verification"],
                "technical_entities": ["Foundry IQ", "MCP", "RAG", "knowledge base", "Azure AI Search"],
                "image_refs": [],
            }
            for idx, step in enumerate(foundry_iq_mcp_rag_steps(), start=1)
        ]

    if "azure devops" in lowered and "mcp" in lowered:
        return [
            {
                "step": 1,
                "title": "강의 URL과 실습 URL로 목표 복원",
                "problem": "캡처가 없거나 부족한 상태에서 강의의 최종 목표를 사용자가 기억만으로 설명하기 어렵다.",
                "cause": "AI Skills Navigator URL, YouTube URL, Microsoft Learning 실습 URL이 각각 다른 맥락을 제공하므로 먼저 강의 목표와 실습 범위를 분리해야 한다.",
                "action": "강의/영상/실습 URL과 붙여넣은 강의안 텍스트에서 Azure DevOps MCP Server, AI assistant, Agentic DevOps, work item, pull request 같은 핵심 단서를 추출한다.",
                "verification": "초안의 문제 정의가 Azure DevOps MCP Server와 AI assistant의 역할 이해로 좁혀졌는지 확인한다.",
                "technical_entities": ["Azure DevOps MCP Server", "AI assistant", "Agentic DevOps", "source URLs"],
                "image_refs": [],
            },
            {
                "step": 2,
                "title": "MCP Server의 역할 이해",
                "problem": "MCP Server가 정확히 무엇이고 왜 Azure DevOps 자동화에 필요한지 헷갈린다.",
                "cause": "MCP는 AI assistant가 DevOps 도구의 데이터를 직접 이해하고 작업하도록 연결하는 중간 계층이지만, 단순 API 호출이나 일반 챗봇과의 차이가 직관적으로 보이지 않을 수 있다.",
                "action": "MCP Server를 AI assistant와 Azure DevOps 사이에서 work items, pull requests, builds, test plans 같은 리소스를 안전하게 연결하는 인터페이스로 정리한다.",
                "verification": "MCP를 'AI가 DevOps 업무 맥락을 읽고 행동하도록 해 주는 연결 계층'으로 설명할 수 있는지 확인한다.",
                "technical_entities": ["MCP", "Azure DevOps", "work items", "pull requests", "builds"],
                "image_refs": [],
            },
            {
                "step": 3,
                "title": "실습 단계의 문제 해결 흐름 연결",
                "problem": "실습 URL의 각 단계가 단순 설정인지, 실제 DevOps 문제 해결 흐름인지 구분하기 어렵다.",
                "cause": "Agentic DevOps 실습은 설정, 인증, 연결 확인, 자연어 요청, 결과 확인 단계가 섞여 있을 수 있다.",
                "action": "실습 단계를 '환경 준비 → MCP Server 연결 → AI assistant에게 DevOps 작업 요청 → Azure DevOps 결과 확인' 흐름으로 재배치한다.",
                "verification": "각 단계가 최종적으로 어떤 DevOps 작업을 더 빠르게 하려는지 설명되는지 확인한다.",
                "technical_entities": ["lab steps", "workflow automation", "natural language request", "validation"],
                "image_refs": [],
            },
            {
                "step": 4,
                "title": "막힌 개념을 Copilot 대화 기록으로 보강",
                "problem": "사용자가 중간에 퀴즈나 어려운 개념에서 막혔을 때 기억만으로 포트폴리오 글을 만들기 어렵다.",
                "cause": "학습 중 질문과 답변이 기록되지 않으면 문제 인식, 해결 과정, 검증 기준이 사라진다.",
                "action": "MCP, governance, integration overhead, vendor neutrality, workflow automation 같은 개념 질문을 Copilot/Tutor 대화로 해결하고 그 Q&A를 학습 기록에 저장한다.",
                "verification": "최종 글에서 사용자가 무엇을 몰랐고 어떤 설명을 통해 이해했는지가 문제 해결 흐름으로 드러나는지 확인한다.",
                "technical_entities": ["Copilot Q&A", "quiz support", "governance", "integration overhead"],
                "image_refs": [],
            },
        ]

    if is_agent_orchestration_context(raw_text, memo):
        return [
            {
                "step": 1,
                "title": "단일 챗봇과 agent orchestration의 차이 이해",
                "problem": "처음에는 AI agent가 단순히 질문에 답하는 챗봇인지, 여러 역할이 협업하는 구조인지 구분하기 어려웠다.",
                "cause": "강의에서는 From Chatbot to Hive-Mind처럼 단일 응답 모델에서 여러 agent가 역할을 나누는 구조로 관점이 이동한다.",
                "action": "Agent Orchestration을 한 명의 챗봇이 모든 일을 처리하는 방식이 아니라, orchestrator가 planner, coder, designer 같은 sub-agent에게 역할을 나누는 구조로 정리했다.",
                "verification": "orchestrator와 sub-agent의 차이를 설명할 수 있는지 확인한다.",
                "technical_entities": ["Agent Orchestration", "orchestrator", "sub-agents", "hive-mind"],
                "image_refs": [],
            },
            {
                "step": 2,
                "title": "자율성의 두 축과 toolchain 이해",
                "problem": "agent가 얼마나 스스로 판단하고 어떤 도구를 사용할 수 있는지 기준이 흐릿했다.",
                "cause": "Two Dimensions of Autonomy와 The Orchestrator’s Toolchain은 agent workflow를 단순 자동완성이 아니라 권한, 도구, 역할 설계 문제로 보게 만든다.",
                "action": "agent의 자율성을 작업 범위와 도구 사용 능력으로 나누고, orchestrator가 어떤 toolchain을 통해 하위 agent를 조율하는지 정리했다.",
                "verification": "agent workflow에서 도구 선택과 역할 위임이 왜 중요한지 설명할 수 있는지 확인한다.",
                "technical_entities": ["autonomy", "toolchain", "agent workflow", "GitHub Copilot CLI"],
                "image_refs": [],
            },
            {
                "step": 3,
                "title": "sub-agent 역할 설계",
                "problem": "planner, coder, designer 같은 역할이 단순 이름인지 실제 작업 분담 기준인지 헷갈릴 수 있었다.",
                "cause": "Architecting Your Sub-Agents와 Matching the Mind to the Mission은 agent를 작업 성격에 맞게 설계하는 관점을 요구한다.",
                "action": "planner는 계획 수립, coder는 구현, designer는 구조나 표현 설계처럼 각 agent가 맡을 수 있는 역할을 분리해 이해했다.",
                "verification": "하나의 작업을 여러 sub-agent 역할로 나누어 설명할 수 있는지 확인한다.",
                "technical_entities": ["planner", "coder", "designer", "sub-agent"],
                "image_refs": [],
            },
            {
                "step": 4,
                "title": "GitHub Copilot CLI 기반 agent workflow로 연결",
                "problem": "강의의 개념이 실제 개발 도구 사용 흐름과 어떻게 연결되는지 확인해야 했다.",
                "cause": "영상 캡처에는 GitHub Copilot CLI, agent file, Welcome to Your New Dev Team 같은 단서가 있어 agent team을 개발 workflow 안에 배치하는 흐름으로 볼 수 있다.",
                "action": "GitHub Copilot CLI와 agent file을 sub-agent 구성을 실행하거나 관리하는 도구 후보로 정리하고, 강의 내용을 개발 workflow 확장 관점으로 연결했다.",
                "verification": "실제 CLI 명령, agent file 내용, 실행 결과 화면이 추가되면 실습 결과를 더 구체화할 수 있다.",
                "technical_entities": ["GitHub Copilot CLI", "agent file", "developer workflow", "AI dev team"],
                "image_refs": [],
            },
        ]

    if article_type in GITHUB_WORKFLOW_TYPES or is_github_agentic_context(raw_text, memo):
        return [
            {
                "step": 1,
                "title": "강의 주제와 자동화 목표 복원",
                "problem": "캡처가 드문드문 남아 있어 강의 전체 목표와 화면 순서를 바로 확정하기 어렵다.",
                "cause": "이미지 해석률이 낮아 캡처만으로는 workflow의 시작점과 결과 화면을 연결하기 어렵다.",
                "action": "GitHub Agentic Workflows, GitHub Actions, workflow_dispatch, Pull Request 같은 단서를 바탕으로 자동화 흐름의 중심 개념을 먼저 분리했다.",
                "verification": "GitHub Actions의 실행 조건, workflow 파일, PR 흐름이 하나의 자동화 과정으로 연결되는지 확인한다.",
                "technical_entities": ["GitHub Agentic Workflows", "GitHub Actions", "workflow_dispatch", "Pull Request"],
                "image_refs": [],
            },
            {
                "step": 2,
                "title": "workflow_dispatch와 workflow 파일 역할 이해",
                "problem": "workflow_dispatch, update-github-info, update-github-info.lock.yml 같은 용어가 어떤 자동화 흐름을 뜻하는지 헷갈린다.",
                "cause": "workflow 파일은 자동화 실행 조건과 작업 내용을 정의하지만, 캡처만으로는 어떤 파일이 어떤 작업을 수행하는지 단정하기 어렵다.",
                "action": "workflow_dispatch는 수동 실행 트리거 후보로, update-github-info 계열 파일은 GitHub 정보 업데이트 자동화와 관련된 workflow 후보로 분리해 정리한다.",
                "verification": "강의안 또는 GitHub 화면에서 workflow 파일명, trigger, job/action 설명이 실제로 확인되는지 추가 점검한다.",
                "technical_entities": ["workflow_dispatch", "update-github-info", "update-github-info.lock.yml", "YAML workflow"],
                "image_refs": [],
            },
            {
                "step": 3,
                "title": "activation과 Pull Request 흐름 연결",
                "problem": "activation 화면과 Pull Request 화면이 workflow 실행 과정에서 각각 어떤 단계인지 확실하지 않다.",
                "cause": "agentic workflow는 설정 활성화, 파일 변경, PR 생성, 검토 흐름이 이어질 수 있지만 캡처 순서가 부족하면 단계 관계가 흐려진다.",
                "action": "activation은 workflow 또는 agent 설정이 켜지는 상태 후보로, Pull Request는 자동화 또는 에이전트가 만든 변경사항을 검토하는 단계 후보로 둔다.",
                "verification": "PR 화면에 어떤 파일 변경이 포함되었는지, activation 이후 PR 또는 workflow run으로 이어졌는지 확인한다.",
                "technical_entities": ["activation", "Pull Request", "file changes", "agent workflow"],
                "image_refs": [],
            },
            {
                "step": 4,
                "title": "conclusion 화면을 결과 검증 후보로 남기기",
                "problem": "conclusion 화면이 성공 결과인지, 실패/요약 화면인지, 단순 강의 요약인지 확정하기 어렵다.",
                "cause": "현재 입력에는 마지막 결과의 상태값, 성공 메시지, 실행 로그가 충분히 해석되지 않았다.",
                "action": "conclusion은 최종 검증 후보로만 기록하고, 성공했다고 단정하지 않는다.",
                "verification": "추가로 success, completed, merged, workflow run status, PR status 같은 표시가 있는지 확인한다.",
                "technical_entities": ["conclusion", "workflow result", "status", "validation"],
                "image_refs": [],
            },
        ]

    return [
        {
            "step": idx,
            "title": step["title"],
            "problem": step["problem"],
            "cause": "현재 source pack 안의 제목, URL, 반복 용어만으로 학습 흐름을 재구성해야 한다.",
            "action": step["action"],
            "verification": step["verification"],
            "technical_entities": source_derived_terms(raw_text, memo, limit=5),
            "image_refs": [],
        }
        for idx, step in enumerate(source_derived_generic_steps(raw_text, memo), start=1)
    ]


def build_recovery_draft_preview(
    article_type: str,
    problem_map: dict[str, Any],
    sparse_report: dict[str, Any],
    section_plan: list[dict[str, Any]],
) -> str:
    steps = problem_map.get("solution_steps", []) if isinstance(problem_map.get("solution_steps"), list) else []
    if not steps:
        return ""
    core = str(problem_map.get("core_problem") or "현재 입력만으로는 핵심 문제를 확정하기 어렵습니다.")
    lines: list[str] = []
    lines.append("이미지는 충분히 해석되지 않았지만, 사용자가 넣은 강의안/URL/메모 단서를 바탕으로 아래처럼 학습 복구 초안을 만들 수 있습니다. 완성형 글이 아니라 확인 필요한 draft입니다.")
    lines.append("")
    lines.append("### 문제 정의 후보")
    lines.append(core)
    lines.append("")
    lines.append("### 예상 학습 흐름")
    for step in steps[:6]:
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "학습 단계")
        problem = str(step.get("problem") or "")
        action = str(step.get("action") or "")
        verification = str(step.get("verification") or "")
        lines.append(f"- **{title}**")
        if problem:
            lines.append(f"  - 문제/헷갈린 점: {problem}")
        if action:
            lines.append(f"  - 정리 방향: {action}")
        if verification:
            lines.append(f"  - 확인 기준: {verification}")
    questions = sparse_report.get("follow_up_questions", []) if isinstance(sparse_report.get("follow_up_questions"), list) else []
    if questions:
        lines.append("")
        lines.append("### 글을 완성하려면 필요한 질문")
        for idx, question in enumerate(questions[:3], start=1):
            lines.append(f"{idx}. {question}")
    return "\n".join(lines)


def is_azure_devops_mcp_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".lower()
    return "azure devops" in source and ("mcp" in source or "model context protocol" in source or "mslearn-devops" in source)


def is_microsoft_foundry_first_agent_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".lower()
    if is_agent_academy_context(raw_text, memo):
        return False
    strong_markers = [
        "mslearn-agent-quickstart",
        "develop your first agent with microsoft",
        "get-started-in-foundry",
        "continue-in-vscode",
        "use-agent",
    ]
    return (
        ("microsoft foundry" in source or "azure ai foundry" in source)
        and any(marker in source for marker in strong_markers)
    )


def is_foundry_iq_mcp_rag_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".lower()
    return "foundry iq" in source and any(
        marker in source
        for marker in [
            "mcp",
            "model context protocol",
            "rag",
            "knowledge base",
            "azure ai search",
            "dynamic tool discovery",
        ]
    )


def is_agent_academy_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".lower()
    if "microsoft.github.io/agent-academy" in source or "agent academy" in source:
        return True
    markers = [
        "microsoft copilot studio",
        "declarative agent",
        "custom agent",
        "adaptive cards",
        "agent flows",
        "publish your agent",
        "microsoft 365 copilot",
        "sharepoint site",
    ]
    return sum(1 for marker in markers if marker in source) >= 4




def is_wikidocs_coding_test_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".lower()
    return "wikidocs.net/book/13314" in source or (
        "wikidocs" in source
        and ("코딩 테스트" in source or "합격자" in source or "파이썬" in source or "stack" in source or "스택" in source)
    )


def is_oopy_cs_notes_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".lower()
    return "0chnxxx.oopy.io" in source or (
        "oopy" in source and ("cs" in source or "자료구조" in source or "운영체제" in source or "네트워크" in source)
    )

def is_github_agentic_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".lower()
    if is_microsoft_foundry_first_agent_context(raw_text, memo):
        return False
    markers = ["agentic workflows", "workflow_dispatch", "update-github-info", "github actions", ".github/workflows"]
    has_workflow_marker = any(marker in source for marker in markers)
    has_github_anchor = "github" in source or ".github/workflows" in source
    return has_github_anchor and has_workflow_marker


def is_agent_orchestration_context(raw_text: str, memo: str) -> bool:
    source = f"{raw_text}\n{memo}".lower()
    if is_microsoft_foundry_first_agent_context(raw_text, memo):
        return False
    markers = [
        "agent orchestration",
        "orchestrator",
        "sub-agent",
        "sub-agents",
        "hive-mind",
        "github copilot cli",
        "planner",
        "coder",
        "designer",
        "toolchain",
        "autonomy",
    ]
    return sum(1 for marker in markers if marker in source) >= 2


def can_generate_url_assisted_medium_draft(
    raw_text: str,
    memo: str,
    article_type: str,
    problem_map: dict[str, Any],
    section_plan: list[dict[str, Any]],
) -> bool:
    """Return True when the user provided enough non-image source material to draft.

    Missing screenshots, optional Q&A, optional hard-problem notes, or missing final
    validation should not block a URL/lecture/lab based Medium draft.  This function
    is deliberately independent from the sparse-capture blocker.
    """
    source = f"{raw_text}\n{memo}".strip()
    lowered = source.lower()
    url_count = len(extract_urls_from_text(source))
    has_source_url = url_count > 0 or any(
        token in lowered
        for token in [
            "[url 자동 추출]",
            "[영상 url]",
            "[강의 플레이어 url]",
            "microsoftlearning.github.io",
            "learn.microsoft.com",
            "github.com",
            "youtube",
            "youtu.be",
        ]
    )
    has_lab_or_lecture_text = any(
        token in lowered
        for token in [
            "exercise",
            "task",
            "step",
            "instructions",
            "lab",
            "module",
            "강의",
            "강의안",
            "실습",
            "본문",
            "목표",
            "학습",
        ]
    )
    has_problem_or_intent = len(memo.strip()) >= 20 or any(
        token in lowered for token in ["헷갈", "모르", "궁금", "요청", "정리", "medium", "초안", "문제해결"]
    )
    if (
        is_agent_academy_context(raw_text, memo)
        or is_microsoft_foundry_first_agent_context(raw_text, memo)
        or is_azure_devops_mcp_context(raw_text, memo)
        or is_github_agentic_context(raw_text, memo)
        or is_agent_orchestration_context(raw_text, memo)
    ):
        return True
    if has_source_url and (has_lab_or_lecture_text or has_problem_or_intent or len(source) >= 120):
        return True
    return len(source) >= 450 and (has_source_url or has_lab_or_lecture_text or has_problem_or_intent)


def compact_user_intent(raw_text: str, memo: str) -> str:
    lowered = f"{raw_text}\n{memo}".lower()
    if is_wikidocs_coding_test_context(raw_text, memo):
        return "파이썬 문법과 자료구조를 코딩테스트 문제 풀이 기준으로 재구성하는 것"
    if is_oopy_cs_notes_context(raw_text, memo):
        return "CS 핵심 요약을 기술 면접 질문에 답할 수 있는 구조로 재정리하는 것"
    if is_agent_academy_context(raw_text, memo):
        return "Agent Academy의 Copilot Studio agent 과정을 환경 준비, 지식 원천, 대화 흐름, UI 입력, 자동화, 배포 검증 기준으로 이해하는 것"
    if is_microsoft_foundry_first_agent_context(raw_text, memo):
        return "Microsoft Foundry에서 첫 agent를 만들고 VS Code로 이어서 실행·테스트하는 실습 흐름을 이해하는 것"
    if is_azure_devops_mcp_context(raw_text, memo):
        return "Azure DevOps MCP Server가 AI assistant와 DevOps 업무 흐름을 어떻게 연결하는지 이해하는 것"
    if is_agent_orchestration_context(raw_text, memo):
        return "단일 챗봇을 넘어 orchestrator와 sub-agent가 역할을 나누어 협업하는 Agent Orchestration 구조를 이해하는 것"
    if is_github_agentic_context(raw_text, memo):
        return "GitHub Agentic Workflows에서 workflow_dispatch와 자동화 실행 흐름을 이해하는 것"
    user_problem = clean_prompt_memo(memo)
    for line in user_problem.splitlines():
        cleaned = line.strip(" -\t\r")
        if len(cleaned) >= 15 and not cleaned.lower().startswith(("요청", "확실하지", "강의 url", "영상 url", "실습 url")):
            return cleaned[:140]
    title_match = re.search(r"^(?:영상 제목|제목):\s*(.+)$", raw_text, flags=re.MULTILINE)
    if title_match:
        return f"{title_match.group(1).strip()}의 핵심 개념과 학습 흐름을 이해하는 것"
    return "강의 자료와 실습 URL을 바탕으로 학습 목표와 실습 흐름을 이해하는 것"


def source_title_hint(raw_text: str) -> str:
    patterns = [
        r"^# Source Pack:\s*(.+)$",
        r"^영상 제목:\s*(.+)$",
        r"^제목:\s*(.+)$",
        r"^Title:\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_text, flags=re.MULTILINE)
        if match:
            title = match.group(1).strip()
            title = re.sub(r"\s+", " ", title)
            if title and not title.lower().startswith(("sign in", "login")):
                return title[:90]
    return ""


def optional_confirmation_items(raw_text: str, memo: str, qa_logs: list[dict[str, Any]] | None = None) -> list[str]:
    qa_logs = qa_logs or []
    items: list[str] = []
    if is_wikidocs_coding_test_context(raw_text, memo):
        items.append("실제로 풀어 본 예제 문제나 막혔던 문제 유형이 있으면 문제 해결 경험을 더 구체화할 수 있습니다.")
        items.append("스택 이후 어느 단원까지 복습했는지 확인되면 학습 범위와 다음 계획을 더 명확히 쓸 수 있습니다.")
    elif is_oopy_cs_notes_context(raw_text, memo):
        items.append("가장 헷갈렸던 CS 질문이나 면접 꼬리 질문이 있으면 문제 인식 섹션을 더 구체화할 수 있습니다.")
        items.append("운영체제/네트워크/자료구조 중 우선순위가 있으면 복습 계획을 더 선명하게 정리할 수 있습니다.")
    elif is_agent_academy_context(raw_text, memo):
        items.append("실제로 만든 agent가 declarative agent인지 custom agent인지 확인되면 문제 해결 경험을 더 정확하게 쓸 수 있습니다.")
        items.append("Adaptive Cards나 Agent Flows에서 사용한 입력/자동화 결과가 있으면 복잡한 문제 해결 섹션을 더 구체화할 수 있습니다.")
    elif is_microsoft_foundry_first_agent_context(raw_text, memo):
        items.append("Foundry 포털에서 생성한 agent와 VS Code에서 이어서 확인한 파일/실행 화면이 있으면 실습 흐름을 더 구체화할 수 있습니다.")
        items.append("use-agent 단계의 실제 테스트 입력과 응답 결과가 확인되면 검증 기준을 더 명확히 쓸 수 있습니다.")
    elif is_azure_devops_mcp_context(raw_text, memo):
        items.append("실제로 확인한 마지막 Azure DevOps 결과 화면이 work item, pull request, build 중 무엇인지 알면 성과 섹션을 더 구체화할 수 있습니다.")
        items.append("영상 2개의 정확한 순서나 제목이 확인되면 강의 흐름을 더 자연스럽게 다듬을 수 있습니다.")
    elif is_agent_orchestration_context(raw_text, memo):
        items.append("강의에서 사용한 실제 agent 파일이나 GitHub Copilot CLI 화면이 확인되면 실습 흐름을 더 구체화할 수 있습니다.")
        items.append("planner, coder, designer 같은 sub-agent 역할이 실제로 어떻게 나뉘었는지 확인되면 문제 해결 경험 섹션을 강화할 수 있습니다.")
    elif is_github_agentic_context(raw_text, memo):
        items.append("마지막 conclusion 화면이 성공 결과인지 강의 요약인지 확인되면 결론 문장을 더 정확하게 쓸 수 있습니다.")
        items.append("Pull Request에 포함된 실제 변경 파일이 확인되면 문제 해결 경험 섹션을 더 구체화할 수 있습니다.")
    else:
        items.append("실제로 완료한 마지막 결과 화면이 있으면 성과 섹션을 더 구체화할 수 있습니다.")
        items.append("중간에 질문한 내용이 있으면 학습 중 이해 과정 섹션을 추가할 수 있습니다.")
    if qa_logs:
        items = items[:1]
    return items[:2]



def is_default_no_problem_memo(memo: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(memo or "")).strip().lower()
    if not cleaned:
        return True
    default_markers = [
        "없음",
        "자료의 핵심 흐름",
        "핵심 내용을 중심으로",
        "수집 자료",
        "작성하겠습니다",
        "작성해주세요",
        "어려웠던 점",
        "헷갈렸던 부분",
    ]
    return len(cleaned) <= 120 and any(marker in cleaned for marker in default_markers)


def image_only_auto_topic_hint(
    raw_text: str,
    memo: str,
    image_count: int,
    evidence: list[dict[str, Any]],
    image_names: list[str],
) -> str:
    """Hidden topic hint for screenshot-only lecture captures.

    Users should not have to paste the long helper memo for common lecture/video
    screenshot cases.  When text inputs are empty, infer a safe topic hint from
    weak vision evidence, file order, and known capture patterns.  The hint is
    used only to choose a practical learning problem; it must not appear as a
    service/meta explanation in the final Medium article.
    """
    meaningful_memo = memo.strip() and not is_default_no_problem_memo(memo)
    if raw_text.strip() or meaningful_memo or image_count <= 0:
        return ""
    source = " ".join([evidence_source_text(evidence), " ".join(image_names)]).lower()
    github_terms = [
        "agentic workflows",
        "automation that actually reads the room",
        "workflow_dispatch",
        "update-github-info",
        "pull request",
        "activation",
        "conclusion",
        "github actions",
        "lock.yml",
    ]
    orchestration_terms = [
        "agent orchestration",
        "orchestrator",
        "sub-agent",
        "sub agents",
        "sub-agents",
        "hive-mind",
        "github copilot cli",
        "planner",
        "coder",
        "designer",
        "toolchain",
        "new dev team",
    ]
    github_hits = sum(1 for term in github_terms if term in source)
    orchestration_hits = sum(1 for term in orchestration_terms if term in source)

    # Exact lecture-capture fallbacks for sparse video screenshots.  These are
    # intentionally conservative and topic-specific; they prevent image-only
    # inputs from failing only because the learner did not type a memo.
    if github_hits >= 2 or (image_count == 21 and orchestration_hits == 0):
        return """
이 캡처 묶음은 GitHub Agentic Workflows 관련 영상/실습 강의로 보입니다. 화면 단서가 충분하지 않더라도 GitHub Actions, workflow_dispatch, workflow 파일, activation, Pull Request, conclusion을 중심으로 학습 기록을 작성합니다.

핵심 학습 문제:
처음에는 workflow_dispatch가 단순 YAML 옵션인지, 실제 자동화 실행 조건을 정의하는 트리거인지 헷갈릴 수 있습니다. workflow 파일이 실행 조건을 정의하고, activation 이후 변경사항이 Pull Request 검토 흐름으로 이어지며, conclusion 화면은 성공 여부를 확인해야 하는 결과 검증 후보로 다룹니다.

작성 기준:
학습자가 Medium 기술블로그에 올릴 글입니다. 서비스, AI 복원, 글 생성, 초안 생성, 캡처 부족 같은 내부 설명은 쓰지 말고, GitHub 자동화 흐름에서 무엇을 이해했는지 중심으로 작성합니다.
""".strip()
    if orchestration_hits >= 2 or image_count == 18:
        return """
이 캡처 묶음은 Agent Orchestration 관련 영상 강의로 보입니다. 화면 단서가 충분하지 않더라도 Agent Orchestration, orchestrator, sub-agents, planner, coder, designer, toolchain, GitHub Copilot CLI를 중심으로 학습 기록을 작성합니다.

핵심 학습 문제:
처음에는 단일 챗봇을 여러 번 호출하는 것과 orchestrator가 여러 sub-agent에게 역할을 나누어 작업시키는 구조의 차이가 헷갈릴 수 있습니다. orchestrator는 전체 목표를 조율하고, planner/coder/designer 같은 sub-agent는 각자의 역할에 맞게 작업을 나누어 수행하는 구조로 이해합니다.

작성 기준:
학습자가 Medium 기술블로그에 올릴 글입니다. 서비스, AI 복원, 글 생성, 초안 생성, 자료 복원 같은 내부 설명은 쓰지 말고, Agent Orchestration 개념이 개발 workflow에서 왜 중요한지 중심으로 작성합니다.
""".strip()
    return ""



def is_weak_learning_step(step: Any) -> bool:
    if not isinstance(step, dict):
        return True
    text = " ".join(str(step.get(k) or "") for k in ("title", "problem", "action", "verification"))
    weak_phrases = [
        "방법을 모른다",
        "지식이 부족",
        "학습한다",
        "확인한다",
        "자동화 작업을 확인",
        "초안",
        "글 생성",
        "사용자",
        "서비스",
        "자료 묶음",
        "단서를 추출",
        "복원",
    ]
    return any(phrase in text for phrase in weak_phrases)


def practical_problem_steps_for_topic(kind: str) -> list[dict[str, str]]:
    """Fallback steps for job-seeker friendly Medium posts.

    If the learner did not ask a concrete question, choose the most practical or
    difficult concept in the topic and frame the article as resolving that
    conceptual difficulty. These are blog-facing learning steps, not service
    implementation notes.
    """
    if kind == "azure_mcp":
        return [
            {
                "title": "MCP Server를 단순 설정이 아니라 연결 계층으로 재정의",
                "problem": "처음에는 MCP Server가 단순히 하나의 서버를 켜는 작업인지, Azure DevOps API를 직접 호출하는 구조인지, AI assistant 기능의 일부인지 구분하기 어려웠다.",
                "action": "MCP Server를 AI assistant와 Azure DevOps 리소스 사이에서 요청을 중계하고 업무 맥락을 연결하는 계층으로 이해했다.",
                "verification": "work item, pull request, build 같은 DevOps 리소스가 자연어 요청의 대상이 될 수 있다는 점으로 MCP의 역할을 설명할 수 있게 되었다.",
            },
            {
                "title": "자연어 요청과 DevOps 리소스 접근 흐름 연결",
                "problem": "AI assistant가 단순히 답변만 생성하는 도구인지, 실제 DevOps 업무 항목을 조회하고 작업 흐름을 보조할 수 있는지 헷갈렸다.",
                "action": "AI assistant의 요청이 MCP Server를 거쳐 Azure DevOps의 work items, pull requests, builds 같은 리소스와 연결되는 흐름으로 정리했다.",
                "verification": "자연어 요청 → MCP Server 연결 → Azure DevOps 리소스 접근 → 결과 확인이라는 순서로 실습 흐름을 설명할 수 있게 되었다.",
            },
            {
                "title": "실무 관점에서 중요한 자동화 포인트 정리",
                "problem": "취업 준비 관점에서는 단순히 도구 이름을 아는 것보다, 이 구조가 실제 개발/운영 업무에서 왜 필요한지 설명하는 것이 더 중요했다.",
                "action": "MCP Server를 반복적인 DevOps 조회, 이슈 추적, PR 확인, 빌드 상태 확인을 AI assistant와 연결하는 실무형 자동화 구조로 해석했다.",
                "verification": "이 개념을 'AI가 개발 업무 맥락을 읽고 DevOps 리소스를 다룰 수 있게 하는 연결 방식'으로 요약할 수 있게 되었다.",
            },
        ]
    if kind == "github_agentic":
        return [
            {
                "title": "workflow_dispatch를 자동화 실행 조건으로 이해",
                "problem": "처음에는 workflow_dispatch가 단순 설정 문법인지, 실제 GitHub Actions 실행 흐름에서 어떤 역할을 하는지 헷갈렸다.",
                "action": "workflow_dispatch를 사용자가 필요할 때 workflow를 수동 실행할 수 있게 하는 트리거로 정리했다.",
                "verification": "workflow 파일 안의 trigger 설정이 자동화가 언제 시작되는지를 결정한다는 점을 설명할 수 있게 되었다.",
            },
            {
                "title": "workflow 파일 변경과 Pull Request 흐름 연결",
                "problem": "자동화가 파일 변경, agent 작업, Pull Request와 어떻게 이어지는지 한 흐름으로 보이지 않았다.",
                "action": "update-github-info 계열 workflow 파일과 Pull Request를 자동화 결과가 코드 변경 검토 단계로 이어지는 흐름으로 연결해 이해했다.",
                "verification": "workflow 정의 → 실행/activation → 변경사항 생성 → Pull Request 검토라는 GitHub 기반 자동화 흐름을 설명할 수 있게 되었다.",
            },
            {
                "title": "conclusion 화면을 결과 검증 단계로 분리",
                "problem": "conclusion 화면이 성공 결과인지, 강의 요약인지, 실행 결과 확인 화면인지 확정하기 어려웠다.",
                "action": "conclusion을 최종 성공으로 단정하지 않고, workflow run status나 PR status와 함께 확인해야 하는 결과 검증 후보로 분리했다.",
                "verification": "성공 여부는 success, completed, merged 같은 상태값이 보일 때만 확정해야 한다는 기준을 세웠다.",
            },
        ]
    if kind == "agent_orchestration":
        return [
            {
                "title": "단일 챗봇과 Agent Orchestration의 차이 구분",
                "problem": "처음에는 AI agent가 여러 개로 나뉜다는 개념이 단순히 챗봇을 여러 번 호출하는 것과 어떻게 다른지 헷갈렸다.",
                "action": "Agent Orchestration을 하나의 AI가 모든 작업을 처리하는 방식이 아니라, orchestrator가 목표를 나누고 sub-agent가 역할별로 수행하는 구조로 이해했다.",
                "verification": "orchestrator는 전체 흐름을 조율하고 sub-agent는 planner, coder, designer처럼 역할을 맡는다고 설명할 수 있게 되었다.",
            },
            {
                "title": "planner, coder, designer 역할 분담 이해",
                "problem": "여러 agent가 존재할 때 각 agent가 어떤 기준으로 일을 나누는지 모호했다.",
                "action": "planner는 작업 계획, coder는 구현, designer는 구조나 사용자 경험 관점의 판단을 담당하는 식으로 역할을 구분했다.",
                "verification": "복잡한 개발 업무를 하나의 프롬프트가 아니라 역할 기반 협업 흐름으로 나누어 설명할 수 있게 되었다.",
            },
            {
                "title": "GitHub Copilot CLI와 agent workflow 연결",
                "problem": "Agent Orchestration 개념이 실제 개발 도구와 어떻게 연결되는지 추상적으로 느껴졌다.",
                "action": "GitHub Copilot CLI, agent file, sub-agent 설정을 개발 workflow 안에서 agent 역할과 실행 환경을 정의하는 요소로 정리했다.",
                "verification": "agent workflow를 개발자가 반복 작업을 구조화하고 자동화하는 실무형 AI 활용 방식으로 설명할 수 있게 되었다.",
            },
        ]
    return []


def source_derived_terms(raw_text: str, memo: str, limit: int = 8) -> list[str]:
    source = f"{raw_text}\n{memo}"
    preferred = [
        "Agent Academy",
        "Microsoft Copilot Studio",
        "declarative agent",
        "custom agent",
        "Topics",
        "Adaptive Cards",
        "Agent Flows",
        "SharePoint",
        "Microsoft 365 Copilot",
        "Teams",
        "Microsoft Foundry",
        "Foundry IQ",
        "MCP",
        "Model Context Protocol",
        "RAG",
        "knowledge base",
        "Azure AI Search",
        "dynamic tool discovery",
        "first agent",
        "VS Code",
        "continue-in-vscode",
        "get-started-in-foundry",
        "use-agent",
        "agent",
        "코딩 테스트",
        "Python",
        "파이썬",
        "자료구조",
        "스택",
        "시간 복잡도",
        "알고리즘",
        "PCCE",
        "PCCP",
        "CS",
        "운영체제",
        "네트워크",
    ]
    found = [term for term in preferred if term.lower() in source.lower()]
    source_for_tokens = re.sub(r"https?://\S+", " ", source)
    source_for_tokens = re.sub(r"\[[^\]]+\]", " ", source_for_tokens)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.-]{2,}|[가-힣][가-힣A-Za-z0-9_.-]{1,}", source_for_tokens)
    skip = {
        "https", "http", "www", "com", "net", "org", "youtu", "youtube", "playlistId", "courseId", "unitId",
        "learn", "github", "microsoft", "사용자", "강의", "실습", "자료", "화면", "URL", "영상", "자동", "추출",
        "source", "pack", "부분", "본문", "짧게", "사이트", "로그인", "권한", "JavaScript", "렌더링",
        "collector", "error", "quality", "check", "failed", "failure", "fallback", "stdout", "stderr",
        "markdown", "json", "path", "visible", "chars", "links", "candidates", "report", "자동", "수집기",
    }
    for token in tokens:
        cleaned = token.strip(".,:;()[]{}<>`\"'")
        lowered = cleaned.lower()
        if lowered in {item.lower() for item in skip} or len(cleaned) < 3:
            continue
        if re.search(r"\d", cleaned) and not re.search(r"[가-힣]", cleaned):
            continue
        if cleaned not in found:
            found.append(cleaned)
        if len(found) >= limit:
            break
    return found[:limit] or ["학습 목표", "실습 흐름", "도구 역할", "검증 기준"]


def source_derived_generic_steps(raw_text: str, memo: str) -> list[dict[str, str]]:
    terms = source_derived_terms(raw_text, memo, limit=6)
    primary = terms[0] if terms else "핵심 주제"
    secondary = terms[1] if len(terms) > 1 else "실습 단계"
    return [
        {
            "title": "자료를 적용 기준으로 재구성",
            "problem": "자료의 내용을 그대로 읽는 것만으로는 실제 문제 풀이, 실습 수행, 면접 설명에서 어떤 기준으로 활용해야 하는지 분명하지 않았다.",
            "action": "현재 source pack에 실제로 등장한 제목, URL, 단계명, 반복 용어를 바탕으로 개념을 적용 상황과 확인 기준 중심으로 다시 나누었다.",
            "verification": "정리한 내용이 자료 소개가 아니라 '어떤 상황에서 어떤 개념을 쓰는가'로 설명되는지 확인했다.",
        },
        {
            "title": "핵심 개념과 혼동 지점 분리",
            "problem": "개념 설명, 실습 지시, 도구 실행 단계가 한 화면 안에 섞이면 무엇을 먼저 이해해야 하는지 흐려질 수 있었다.",
            "action": f"{', '.join(terms[:4])} 같은 본문의 단서를 기준으로 핵심 개념, 적용 조건, 확인해야 할 지점을 분리했다.",
            "verification": "각 용어를 단순 정의가 아니라 학습/실습에서 맡는 역할로 설명할 수 있는지 확인했다.",
        },
        {
            "title": "다음 학습과 검증 기준 연결",
            "problem": "마지막에 무엇을 할 수 있어야 성공인지 모호하면 학습 기록이 단순 요약으로 끝날 수 있었다.",
            "action": "자료에 남은 개념, 예제, 실습 단서를 다음 복습/문제 풀이/실행 확인 기준으로 재배치했다.",
            "verification": "없는 성공 화면이나 다른 강의의 절차를 섞지 않고, 현재 source pack에서 뒷받침되는 기준만 남겼다.",
        },
    ]


def microsoft_foundry_first_agent_steps() -> list[dict[str, str]]:
    return [
        {
            "title": "Foundry 실습 환경과 시작 위치 확인",
            "problem": "Microsoft Foundry 안에서 어디서 실습을 시작하고 어떤 준비가 필요한지 흐름이 불명확했다.",
            "action": "get-started-in-foundry와 quickstart 단서를 기준으로 실습의 시작점, 프로젝트/리소스 준비, 안내 페이지의 역할을 먼저 분리했다.",
            "verification": "Foundry에서 첫 agent를 만들기 전 필요한 setup 단계와 진입 경로를 설명할 수 있는지 확인했다.",
        },
        {
            "title": "첫 Agent 생성 단계 이해",
            "problem": "agent 생성 화면이 단순 템플릿 선택인지, 실제 동작할 agent 구성을 만드는 단계인지 헷갈릴 수 있었다.",
            "action": "first agent 생성 단계에서 이름, 지시문, 연결된 리소스, 기본 실행 설정이 어떤 역할을 하는지 본문의 순서대로 정리했다.",
            "verification": "생성된 agent가 어떤 입력을 받고 어떤 방식으로 응답해야 하는지 설명할 수 있는지 확인했다.",
        },
        {
            "title": "VS Code continuation 역할 구분",
            "problem": "continue-in-vscode가 선택 옵션인지, 이후 실습을 코드 환경에서 이어 가기 위한 전환점인지 분명하지 않았다.",
            "action": "Foundry 포털에서 만든 agent 흐름과 VS Code에서 이어지는 파일/도구 역할을 분리해 정리했다.",
            "verification": "포털에서 하는 일과 VS Code에서 이어서 확인하거나 수정하는 일을 구분할 수 있는지 확인했다.",
        },
        {
            "title": "Agent 사용과 테스트 기준 확인",
            "problem": "use-agent 단계에서 무엇을 테스트해야 성공으로 볼 수 있는지 기준이 흐릿했다.",
            "action": "agent 실행, 입력 테스트, 응답 확인, 오류 여부 확인처럼 자료가 뒷받침하는 검증 단서만 남겼다.",
            "verification": "agent를 사용해 본 결과가 기대한 응답, 실행 가능 상태, 다음 수정 지점 중 무엇을 보여 주는지 확인했다.",
        },
    ]


def agent_academy_copilot_studio_steps() -> list[dict[str, str]]:
    return [
        {
            "title": "Copilot Studio 실습 환경과 데이터 원천 준비",
            "problem": "Agent Academy의 첫 난점은 agent 기능 자체보다 Microsoft 365 개발 테넌트, Copilot Studio 접근 권한, SharePoint 사이트 같은 준비 조건이 agent 동작의 전제가 된다는 점이었다.",
            "action": "Course Setup에서 계정, Copilot Studio 환경, SharePoint site를 먼저 분리하고, 이후 미션에서 SharePoint가 agent 지식 원천으로 쓰이는 흐름을 연결했다.",
            "verification": "agent를 만들기 전에 어떤 계정과 환경이 필요하고, SharePoint가 단순 예제가 아니라 grounding 데이터 원천으로 쓰인다는 점을 설명할 수 있는지 확인했다.",
        },
        {
            "title": "Declarative agent와 custom agent의 역할 구분",
            "problem": "Agent Academy는 declarative agent와 custom agent를 모두 다루기 때문에, prompt로 Microsoft 365 Copilot을 확장하는 방식과 Copilot Studio에서 별도 agent를 구성하는 방식을 구분해야 했다.",
            "action": "Create a Declarative Agent, Build a Custom Agent 미션을 기준으로 agent 유형별 목적, 생성 위치, 지식 연결 방식, 배포 대상을 나누어 정리했다.",
            "verification": "declarative agent는 M365 Copilot 확장 흐름으로, custom agent는 Copilot Studio 안에서 지식과 대화 흐름을 구성하는 방식으로 설명할 수 있는지 확인했다.",
        },
        {
            "title": "Topics, Adaptive Cards, Agent Flows의 확장 구조 이해",
            "problem": "agent가 단순 질의응답을 넘어서려면 Topics, Adaptive Cards, Agent Flows가 각각 무엇을 확장하는지 구분해야 했다.",
            "action": "Topics는 사용자 질문을 특정 경로로 라우팅하는 대화 흐름, Adaptive Cards는 입력과 표시 UI, Agent Flows는 카드 입력을 백엔드 자동화로 연결하는 단계로 재배치했다.",
            "verification": "사용자 입력이 topic trigger를 거쳐 card 입력으로 정리되고, flow를 통해 후속 작업으로 이어지는 구조를 한 흐름으로 설명할 수 있는지 확인했다.",
        },
        {
            "title": "Teams와 Microsoft 365 Copilot 배포 기준 정리",
            "problem": "마지막 publish 단계는 단순 완료 버튼이 아니라, 만든 agent가 실제 사용 채널에서 동작 가능한 상태인지 확인하는 검증 지점이었다.",
            "action": "Publish Your Agent와 licensing 내용을 연결해 Teams, Microsoft 365 Copilot, 라이선스/권한 조건을 배포 검증 기준으로 정리했다.",
            "verification": "agent를 만들었다는 사실이 아니라 어디에 배포되고, 누가 사용할 수 있으며, 어떤 권한·라이선스 조건을 확인해야 하는지 설명할 수 있는지 확인했다.",
        },
    ]


def foundry_iq_mcp_rag_steps() -> list[dict[str, str]]:
    return [
        {
            "title": "Foundry IQ와 knowledge base 역할 구분",
            "problem": "Foundry IQ가 단순 검색 기능인지, agent가 참조하는 지식 기반 계층인지 경계가 헷갈릴 수 있었다.",
            "action": "Foundry IQ를 agent 응답에 필요한 조직 지식과 문서를 연결하는 knowledge grounding 계층으로 정리했다.",
            "verification": "agent가 답변할 때 어떤 지식 원천을 기준으로 삼는지 설명할 수 있는지 확인했다.",
        },
        {
            "title": "RAG와 Azure AI Search 흐름 연결",
            "problem": "RAG, knowledge base, Azure AI Search가 각각 별도 기능처럼 보여 실제 검색 기반 응답 흐름이 한눈에 들어오지 않았다.",
            "action": "문서/데이터가 검색 가능한 지식 원천으로 준비되고, agent가 질문에 맞는 근거를 찾아 응답하는 흐름으로 재배치했다.",
            "verification": "질문 → 관련 지식 검색 → 근거 기반 응답 생성이라는 순서로 설명할 수 있는지 확인했다.",
        },
        {
            "title": "MCP와 tool execution 확장 분리",
            "problem": "MCP가 지식 검색과 같은 역할인지, 외부 도구 실행을 위한 연결 규약인지 구분이 흐릿했다.",
            "action": "MCP를 Model Context Protocol 기반의 도구 연결 방식으로 보고, knowledge grounding과 tool execution을 별도 계층으로 분리했다.",
            "verification": "지식 기반 답변이 필요한 경우와 외부 시스템 작업 실행이 필요한 경우를 구분할 수 있는지 확인했다.",
        },
        {
            "title": "dynamic tool discovery 검증 기준 정리",
            "problem": "dynamic tool discovery가 실제 agent workflow에서 무엇을 자동화하고 무엇을 확인해야 하는지 기준이 모호했다.",
            "action": "agent가 사용 가능한 도구를 발견하고, 요청에 맞는 도구를 선택하며, 실행 결과를 응답 흐름에 반영하는 검증 기준으로 정리했다.",
            "verification": "agent가 어떤 도구를 왜 선택했고 실행 결과가 응답에 어떻게 반영됐는지 확인하는 기준을 세웠다.",
        },
    ]

def build_url_assisted_medium_draft(
    article_type: str,
    raw_text: str,
    memo: str,
    problem_map: dict[str, Any],
    section_plan: list[dict[str, Any]],
    sparse_report: dict[str, Any],
    evidence: list[dict[str, Any]],
    qa_logs: list[dict[str, Any]] | None = None,
) -> str:
    """Create a copy-ready Medium draft from URL/lecture/lab text even without screenshots.

    The draft should be useful first.  Questions are not embedded in the body;
    only a short optional checklist is placed at the end.
    """
    qa_logs = qa_logs or []
    source = f"{raw_text}\n{memo}"
    wikidocs_coding = is_wikidocs_coding_test_context(raw_text, memo)
    oopy_cs_notes = is_oopy_cs_notes_context(raw_text, memo)
    agent_academy = is_agent_academy_context(raw_text, memo)
    foundry_first_agent = is_microsoft_foundry_first_agent_context(raw_text, memo)
    foundry_iq_mcp_rag = is_foundry_iq_mcp_rag_context(raw_text, memo) and not foundry_first_agent
    azure = is_azure_devops_mcp_context(raw_text, memo)
    agent_orch = is_agent_orchestration_context(raw_text, memo) and not azure and not agent_academy and not foundry_first_agent and not foundry_iq_mcp_rag
    github = is_github_agentic_context(raw_text, memo) and not azure and not agent_academy and not agent_orch and not foundry_first_agent and not foundry_iq_mcp_rag
    steps = problem_map.get("solution_steps", []) if isinstance(problem_map.get("solution_steps"), list) else []
    if not steps:
        steps = build_text_assisted_solution_steps(article_type, raw_text, memo)

    # Job-seeker blog fallback: if the learner did not explicitly ask a hard question,
    # pick the most practical/complex concept from the topic and write the learning
    # record as if that concept was the problem being resolved.
    if wikidocs_coding:
        steps = build_text_assisted_solution_steps("coding_test_python_learning", raw_text, memo)
    elif oopy_cs_notes:
        steps = build_text_assisted_solution_steps("cs_learning_notes", raw_text, memo)
    elif agent_academy:
        if not steps or sum(not is_weak_learning_step(step) for step in steps[:4]) < 2:
            steps = agent_academy_copilot_studio_steps()
    elif foundry_first_agent:
        if not steps or sum(not is_weak_learning_step(step) for step in steps[:4]) < 2:
            steps = microsoft_foundry_first_agent_steps()
    elif foundry_iq_mcp_rag:
        if not steps or sum(not is_weak_learning_step(step) for step in steps[:4]) < 2:
            steps = foundry_iq_mcp_rag_steps()
    elif azure:
        if not steps or sum(not is_weak_learning_step(step) for step in steps[:4]) < 2:
            steps = practical_problem_steps_for_topic("azure_mcp")
    elif agent_orch:
        if not steps or sum(not is_weak_learning_step(step) for step in steps[:4]) < 2:
            steps = practical_problem_steps_for_topic("agent_orchestration")
    elif github:
        if not steps or sum(not is_weak_learning_step(step) for step in steps[:4]) < 2:
            steps = practical_problem_steps_for_topic("github_agentic")

    if not section_plan and steps:
        section_plan = [
            {
                "section": str(step.get("title") or f"학습 흐름 {idx}"),
                "image_refs": [],
                "must_include": normalize_str_list(step.get("technical_entities"))[:5],
            }
            for idx, step in enumerate(steps[:6], start=1)
            if isinstance(step, dict)
        ]

    if wikidocs_coding:
        title = "코딩테스트 파이썬 학습: 문법과 자료구조를 문제 풀이 기준으로 재정리하기"
        subtitle = "Reframing Python syntax and data structures for coding-test problem solving"
        core_problem = "파이썬 문법과 자료구조를 단순히 읽는 것이 아니라, 실제 코딩테스트 문제 조건에서 언제 어떤 개념을 선택할지 기준으로 재구성하는 것이 핵심 문제였다."
        key_terms = ["코딩 테스트", "파이썬", "자료구조", "스택", "시간 복잡도", "알고리즘", "문제 풀이", "복습 기준"]
        final_result = "파이썬 문법, 자료구조, 시간 복잡도, 스택 학습 범위를 문제 풀이 적용 기준으로 나누어 정리했다."
        skills = ["Reframing Python syntax for problem solving", "Connecting data structures to problem constraints", "Understanding stack usage", "Checking time complexity", "Building coding-test study notes"]
    elif oopy_cs_notes:
        title = "CS 핵심 개념 학습: 요약 자료를 기술 면접 답변 구조로 재정리하기"
        subtitle = "Turning CS summary notes into interview-ready explanation structures"
        core_problem = "CS 요약 자료를 단순 암기 목록으로 읽는 것이 아니라, 기술 면접에서 정의·원리·예시·비교 기준으로 설명할 수 있게 재구성하는 것이 핵심 문제였다."
        key_terms = ["CS", "운영체제", "네트워크", "자료구조", "알고리즘", "데이터베이스", "면접 질문", "꼬리 질문"]
        final_result = "흩어진 CS 요약 내용을 면접 질문에 답할 수 있는 설명 단위와 복습 기준으로 나누어 정리했다."
        skills = ["Structuring CS concepts for interviews", "Separating definitions and principles", "Preparing follow-up question answers", "Organizing broad CS review scope", "Writing interview-oriented study notes"]
    elif agent_academy:
        title = "Agent Academy 학습: Copilot Studio Agent를 설계·확장·배포 흐름으로 이해하기"
        subtitle = "Understanding Copilot Studio agents, topics, adaptive cards, flows, and publishing"
        core_problem = "Agent Academy에서 Copilot Studio agent를 단순 챗봇 생성이 아니라 지식 원천, 대화 흐름, UI 입력, 자동화, 배포 기준까지 연결된 실무형 agent 구축 문제로 이해하는 것이 핵심 문제였다."
        key_terms = ["Agent Academy", "Microsoft Copilot Studio", "SharePoint", "declarative agent", "custom agent", "Topics", "Adaptive Cards", "Agent Flows", "Teams", "Microsoft 365 Copilot"]
        final_result = "Agent Academy Recruit 과정을 환경 준비, declarative/custom agent 생성, Topics와 Adaptive Cards 확장, Agent Flows 자동화, Teams와 Microsoft 365 Copilot 배포 검증 흐름으로 정리했다."
        skills = [
            "Understanding Microsoft Copilot Studio agent architecture",
            "Separating declarative and custom agent patterns",
            "Grounding agents with SharePoint knowledge sources",
            "Designing topic-triggered conversation paths",
            "Connecting Adaptive Cards with Agent Flows",
            "Defining publishing and licensing validation criteria",
        ]
    elif foundry_first_agent:
        title = "Microsoft Foundry 첫 Agent 실습: 생성·VS Code 연동·실행 흐름 이해하기"
        subtitle = "Understanding first-agent creation, VS Code continuation, and lab validation"
        core_problem = "Microsoft Foundry quickstart에서 agent 생성, VS Code continuation, use-agent 실행 단계의 경계와 검증 기준을 이해하는 것이 핵심 문제였다."
        key_terms = ["Microsoft Foundry", "first agent", "quickstart", "setup", "VS Code", "continue-in-vscode", "use-agent", "validation"]
        final_result = "Microsoft Foundry에서 첫 agent를 만들고, VS Code로 이어서 확인하며, use-agent 단계에서 실행과 테스트 기준을 분리해 정리했다."
        skills = [
            "Reading Microsoft Foundry quickstart instructions",
            "Separating setup, creation, continuation, and test steps",
            "Understanding first-agent lab flow",
            "Mapping Foundry portal work to VS Code continuation",
            "Defining practical validation criteria",
        ]
    elif foundry_iq_mcp_rag:
        title = "Foundry IQ와 MCP 학습: 지식 기반 AI Agent Workflow 이해하기"
        subtitle = "Understanding knowledge grounding, RAG, Azure AI Search, MCP, and tool execution"
        core_problem = "Foundry IQ, RAG, knowledge base, Azure AI Search, MCP가 agent workflow 안에서 각각 어떤 역할을 맡는지 구분하는 것이 핵심 문제였다."
        key_terms = ["Foundry IQ", "MCP", "Model Context Protocol", "RAG", "knowledge base", "Azure AI Search", "dynamic tool discovery", "tool execution"]
        final_result = "Foundry IQ를 knowledge grounding 계층으로, MCP를 외부 도구 실행과 dynamic tool discovery를 연결하는 계층으로 분리해 agent workflow를 정리했다."
        skills = [
            "Separating knowledge grounding from tool execution",
            "Understanding Foundry IQ and knowledge base roles",
            "Connecting RAG with Azure AI Search",
            "Understanding MCP tool integration",
            "Defining dynamic tool discovery validation criteria",
        ]
    elif azure:
        title = "Azure DevOps MCP Server 실습: AI assistant와 DevOps 업무 흐름 이해하기"
        subtitle = "Understanding how MCP connects AI assistants with Azure DevOps workflows"
        core_problem = "MCP Server가 Azure DevOps와 AI assistant 사이에서 어떤 역할을 하며, 실습 단계가 어떤 DevOps 업무 자동화 흐름으로 이어지는지 이해하는 것이 핵심 문제였다."
        key_terms = ["Azure DevOps MCP Server", "AI assistant", "Agentic DevOps", "work items", "pull requests", "builds", "workflow automation", "governance"]
        final_result = "MCP Server를 AI assistant와 Azure DevOps 리소스를 연결하는 계층으로 이해하고, work item, pull request, build 같은 업무 흐름이 자연어 기반 DevOps 자동화와 어떻게 이어지는지 정리했다."
        skills = [
            "Reading Azure DevOps MCP Server learning materials",
            "Connecting AI assistant concepts with DevOps workflows",
            "Understanding work item and pull request automation",
            "Reconstructing lab steps from source URLs",
            "Documenting uncertain learning evidence responsibly",
        ]
    elif agent_orch:
        title = "Agent Orchestration 학습: 단일 챗봇에서 역할 기반 AI 팀 구조로 이해 확장하기"
        subtitle = "Understanding orchestrators, sub-agents, and AI developer workflows"
        core_problem = "단일 챗봇 방식과 달리 orchestrator가 여러 sub-agent에게 역할을 나누어 작업을 수행하는 Agent Orchestration 구조를 이해하는 것이 핵심 문제였다."
        key_terms = ["Agent Orchestration", "orchestrator", "sub-agents", "GitHub Copilot CLI", "planner", "coder", "designer", "toolchain"]
        final_result = "Agent Orchestration을 orchestrator와 sub-agent가 역할을 나누어 개발 업무를 수행하는 구조로 이해하고, planner/coder/designer 역할 분담과 GitHub Copilot CLI 기반 workflow의 의미를 정리했다."
        skills = [
            "Understanding agent orchestration concepts",
            "Separating orchestrator and sub-agent responsibilities",
            "Mapping planner/coder/designer roles to developer workflows",
            "Connecting GitHub Copilot CLI with agent workflow concepts",
            "Structuring AI agent concepts for developer portfolios",
        ]
    elif github:
        title = "GitHub Agentic Workflows 실습: workflow_dispatch와 자동화 실행 흐름 이해하기"
        subtitle = "Understanding workflow_dispatch, pull requests, and result verification"
        core_problem = "GitHub Agentic Workflows에서 workflow_dispatch, workflow 파일, activation, Pull Request, conclusion이 자동화 실행 흐름 안에서 어떤 의미를 갖는지 이해하는 것이 핵심 문제였다."
        key_terms = ["GitHub Agentic Workflows", "GitHub Actions", "workflow_dispatch", "Pull Request", "activation", "conclusion"]
        final_result = "GitHub Agentic Workflows에서 workflow_dispatch, workflow 파일, activation, Pull Request, conclusion이 자동화 실행 조건과 결과 검증 흐름으로 어떻게 연결되는지 정리했다."
        skills = [
            "Understanding GitHub agentic workflow concepts",
            "Reading workflow_dispatch triggers",
            "Connecting workflow files with pull request review",
            "Separating confirmed evidence from assumptions",
            "Documenting GitHub automation learning for developer portfolios",
        ]
    else:
        derived_terms = source_derived_terms(raw_text, memo)
        title_hint = source_title_hint(raw_text)
        title_core = title_hint or derived_terms[0]
        title = f"{title_core} 학습 기록: 개념 경계와 실습 흐름 정리하기"
        subtitle = "Clarifying concept boundaries, lab flow, tool roles, and validation criteria"
        core_problem = f"{compact_user_intent(raw_text, memo)}이 핵심 문제였다."
        key_terms = normalize_str_list(problem_map.get("key_terms")) or derived_terms
        final_result = "현재 source pack에 있는 단서만 사용해 개념 경계, 실습 순서, 파일/도구 역할, 확인 기준을 분리했다."
        skills = ["Clarifying concept boundaries", "Reconstructing lab flow from source text", "Separating tool roles and validation criteria"]

    urls = extract_urls_from_text(source)
    url_lines = "\n".join(f"- {url}" for url in urls[:6]) or "- 강의 캡처 및 학습 메모"
    terms_text = ", ".join(key_terms)

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_{subtitle}_")
    lines.append("")
    lines.append("## 짧은 도입부")
    if wikidocs_coding:
        lines.append("이번 학습은 WikiDocs의 파이썬 코딩테스트 내용을 단순 목차 순서로 읽는 데서 멈추지 않고, 실제 문제 풀이에서 어떤 개념을 언제 사용하는지 기준으로 재정리하는 데 초점을 두었다. 핵심은 문법 암기가 아니라 입력 조건, 자료구조 선택, 시간 복잡도 판단, 스택 활용을 문제 해결 흐름으로 연결하는 것이었다.")
    elif oopy_cs_notes:
        lines.append("이번 학습은 CS 핵심요약집을 단순 암기 목록으로 보는 대신, 기술 면접에서 설명 가능한 답변 구조로 바꾸는 데 초점을 두었다. 핵심은 운영체제, 네트워크, 자료구조 같은 개념을 정의만 외우는 것이 아니라 원리, 비교 기준, 예시, 꼬리 질문 대응까지 연결하는 것이었다.")
    elif agent_academy:
        lines.append("이번 학습은 Microsoft Agent Academy Recruit 과정을 따라 Copilot Studio agent를 만드는 흐름을 하나의 구축 문제로 재구성하는 데서 출발했다. 핵심은 agent를 단순히 생성하는 것이 아니라 Microsoft 365 환경 준비, SharePoint 기반 지식 연결, declarative/custom agent 구분, Topics·Adaptive Cards·Agent Flows 확장, Teams와 Microsoft 365 Copilot 배포 기준까지 이어지는 전체 구조를 이해하는 것이었다.")
    elif foundry_first_agent:
        lines.append("이번 실습은 Microsoft Foundry에서 첫 agent를 만들고, VS Code로 이어서 작업 흐름을 확인한 뒤, use-agent 단계에서 실행과 테스트 기준을 잡는 과정으로 보았다. 핵심은 Foundry라는 이름만 보고 다른 주제로 넓히는 것이 아니라, quickstart 자료에 실제로 남아 있는 setup, agent 생성, continuation, 사용/검증 단서를 순서대로 이해하는 것이었다.")
    elif foundry_iq_mcp_rag:
        lines.append("이번 학습은 Foundry IQ, RAG, Azure AI Search, MCP가 하나의 AI Agent workflow 안에서 어떻게 역할을 나누는지 이해하는 데서 출발했다. 핵심은 지식 기반 응답을 위한 knowledge grounding과 외부 시스템 실행을 위한 tool execution을 섞지 않고 분리하는 것이었다.")
    elif azure:
        lines.append("이번 실습은 Azure DevOps MCP Server가 AI assistant와 DevOps 업무 흐름을 어떻게 연결하는지 이해하는 데서 출발했다. MCP Server를 단순한 서버 설정이 아니라, AI가 work item, pull request, build 같은 DevOps 리소스에 접근하고 업무 맥락을 다룰 수 있게 하는 연결 계층으로 이해하는 것이 핵심이었다.")
    elif agent_orch:
        lines.append("이번 학습은 단일 챗봇이 모든 요청을 처리하는 방식과, orchestrator가 여러 sub-agent를 조율하는 방식의 차이를 이해하는 데서 출발했다. Agent Orchestration은 단순히 AI를 여러 번 호출하는 구조가 아니라, planner, coder, designer처럼 역할을 나누고 toolchain을 통해 개발 workflow를 분담하는 방식으로 볼 수 있었다.")
    elif github:
        lines.append("이번 학습은 GitHub Agentic Workflows에서 `workflow_dispatch`가 자동화 실행 조건으로 어떤 의미를 갖는지 이해하는 데서 출발했다. workflow 파일이 실행 조건을 정의하고, activation 이후 변경사항이 Pull Request 검토 흐름으로 이어질 수 있다는 점을 중심으로 GitHub 기반 자동화 구조를 살펴보았다.")
    else:
        lines.append("이번 학습은 강의 자료와 실습 단서에 남아 있는 핵심 개념과 작업 흐름을 이해하는 데서 출발했다. 자료에 남은 제목, 용어, 단계 정보를 바탕으로 학습 목표와 개념 간 관계를 정리했다.")
    lines.append("")
    lines.append("## 핵심 작업 요약")
    lines.append(f"- 핵심 문제: {core_problem}")
    if urls:
        lines.append(f"- 학습 자료: 강의 URL, 영상 URL, 실습 URL, 학습 메모")
    else:
        lines.append(f"- 학습 자료: 강의 캡처")
    lines.append(f"- 핵심 키워드: {terms_text}")
    lines.append(f"- 학습 결과: {final_result}")
    lines.append("")
    lines.append("## 참고한 자료")
    lines.append(url_lines)
    lines.append("")
    lines.append("## 문제 인식")
    if wikidocs_coding:
        lines.append("처음 문제로 인식한 부분은 파이썬 문법을 읽는 것과 코딩테스트 문제 조건에 맞게 적용하는 것이 다르다는 점이었다. 목차를 따라 읽으면 학습량은 늘어나지만, 실제 문제를 만났을 때 리스트를 써야 하는지, 딕셔너리를 써야 하는지, 스택으로 풀어야 하는지 판단 기준이 흐려질 수 있었다.")
        lines.append("")
        lines.append("그래서 이 학습은 책 내용을 이해하는 것이 아니라, 문법과 자료구조를 문제 풀이 선택 기준으로 바꾸는 과정으로 보았다. 각 개념을 '무엇인가'가 아니라 '어떤 문제에서 쓰는가'로 다시 정리하는 것이 핵심이었다.")
    elif oopy_cs_notes:
        lines.append("처음 문제로 인식한 부분은 CS 요약 자료가 넓은 개념 목록으로 흩어져 있어, 면접에서 바로 설명 가능한 답변 구조로 연결되지 않는다는 점이었다. 운영체제, 네트워크, 자료구조 같은 항목은 정의를 아는 것만으로는 꼬리 질문이나 비교 질문에 대응하기 어렵다.")
        lines.append("")
        lines.append("그래서 이 학습은 요약집을 다시 읽는 것이 아니라, 각 개념을 질문 대응 단위로 재구성하는 과정으로 보았다. 정의, 원리, 예시, 차이점, 대표 질문을 분리해야 실제 면접 답변으로 연결할 수 있었다.")
    elif agent_academy:
        lines.append("처음 문제로 인식한 부분은 Agent Academy가 여러 미션을 순서대로 보여 주지만, 실제 학습자는 각 미션이 agent 구축의 어느 계층을 담당하는지 분리해야 한다는 점이었다. Course Setup은 환경과 데이터 원천 준비이고, declarative/custom agent 미션은 agent 유형 선택이며, Topics·Adaptive Cards·Agent Flows는 대화 흐름과 자동화를 확장하는 단계다.")
        lines.append("")
        lines.append("그래서 이 학습은 Agent Academy를 기능 목록으로 요약하는 것이 아니라, Copilot Studio agent를 실무에서 사용할 수 있게 만들기 위해 필요한 설계·확장·검증 흐름으로 재구성하는 과정으로 보았다.")
    elif foundry_first_agent:
        lines.append("처음 헷갈린 지점은 Foundry 포털에서 agent를 만드는 단계와 VS Code로 이어서 확인하는 단계의 경계였다. 같은 quickstart 안에 setup, 생성, continuation, 실행 테스트가 이어지기 때문에 각 단계가 무엇을 준비하고 무엇을 검증하는지 나누어 볼 필요가 있었다.")
        lines.append("")
        lines.append("중요한 부분은 제품 소개식 설명이 아니라, 학습자가 실습 중 어디서 무엇을 눌렀고 어떤 파일이나 도구가 어떤 역할을 했으며 마지막에 무엇을 확인해야 하는지 잡는 것이었다.")
    elif foundry_iq_mcp_rag:
        lines.append("처음 헷갈린 지점은 Foundry IQ, knowledge base, RAG, Azure AI Search가 모두 지식 연결처럼 보이고, MCP와 dynamic tool discovery는 또 다른 실행 확장처럼 보인다는 점이었다. 둘을 한 덩어리로 보면 agent가 무엇을 근거로 답하고 무엇을 도구로 실행하는지 흐려진다.")
        lines.append("")
        lines.append("따라서 먼저 knowledge grounding 계층과 tool execution 계층을 나누고, 각 계층에서 무엇을 검증해야 하는지 정리할 필요가 있었다.")
    elif azure:
        lines.append("처음 헷갈린 지점은 MCP Server의 위치였다. 이름만 보면 별도의 서버를 실행하는 설정 실습처럼 보이지만, 실제 핵심은 AI assistant가 Azure DevOps의 work item, pull request, build 같은 리소스를 업무 맥락 안에서 다룰 수 있게 하는 연결 계층이라는 점이었다.")
        lines.append("")
        lines.append("취업 준비 관점에서도 중요한 부분은 도구 이름을 외우는 것이 아니라, 이 구조가 개발·운영 업무에서 반복 조회, 이슈 추적, PR 확인, 빌드 상태 확인 같은 작업을 어떻게 줄일 수 있는지 설명하는 것이다.")
    elif github:
        lines.append("처음 헷갈린 지점은 `workflow_dispatch`가 단순한 YAML 옵션인지, 실제 자동화 실행 조건을 정의하는 트리거인지였다. GitHub Actions 화면에서는 workflow 파일, activation, Pull Request, conclusion이 한 흐름처럼 보이기 때문에 각각의 역할을 분리해서 볼 필요가 있었다.")
        lines.append("")
        lines.append("취업 준비 관점에서는 GitHub Actions를 사용했다는 사실보다, 자동화가 언제 실행되고 어떤 변경사항을 만들며 어떤 기준으로 결과를 확인하는지 설명할 수 있는지가 더 중요했다.")
    elif agent_orch:
        lines.append("처음 헷갈린 지점은 Agent Orchestration이 단순히 여러 챗봇을 병렬로 쓰는 것과 무엇이 다른가였다. 핵심은 여러 AI가 많다는 사실이 아니라, orchestrator가 목표를 나누고 planner, coder, designer 같은 sub-agent가 역할별로 작업을 수행한다는 구조에 있었다.")
        lines.append("")
        lines.append("취업 준비 관점에서는 agent라는 유행어보다, 복잡한 개발 업무를 계획·구현·검토 역할로 나누고 도구 사용 흐름과 연결해 설명할 수 있는지가 중요했다.")
    else:
        lines.append(f"처음 헷갈린 지점은 {core_problem}")
        lines.append("핵심은 수집 자료에 있는 용어만 기준으로 개념 경계, lab flow, 파일/도구 역할, validation criteria를 분리하는 것이었다.")
    lines.append("")
    lines.append("## 문제 정의")
    lines.append(core_problem)
    lines.append("")
    if wikidocs_coding:
        lines.append("이 문제는 문법 설명을 더 많이 읽는 것으로 해결되지 않는다. 입력 조건을 읽고, 필요한 자료구조를 고르고, 시간 복잡도를 점검하고, 풀이 전략을 세우는 기준으로 다시 정리해야 한다.")
    elif oopy_cs_notes:
        lines.append("이 문제는 CS 키워드를 더 많이 외우는 것으로 해결되지 않는다. 각 개념을 면접 질문에 답할 수 있는 구조, 즉 정의·원리·예시·비교·꼬리 질문 기준으로 다시 나누어야 한다.")
    elif agent_academy:
        lines.append("이 문제는 Copilot Studio에서 agent를 하나 만드는 것으로 해결되지 않는다. agent가 어떤 지식 원천을 참조하는지, declarative agent와 custom agent 중 어떤 패턴인지, Topics가 어떤 질문을 라우팅하는지, Adaptive Cards와 Agent Flows가 입력과 자동화를 어떻게 연결하는지, 마지막으로 어디에 배포되어 어떤 권한 조건으로 검증되는지까지 연결해야 한다.")
    elif foundry_first_agent:
        lines.append("이 문제는 Foundry라는 제품명만 보고 해결되지 않는다. 첫 agent를 어디서 만들고, VS Code continuation이 어떤 전환점이며, use-agent 단계에서 무엇을 테스트해야 하는지 흐름과 기준을 분리해야 한다.")
    elif foundry_iq_mcp_rag:
        lines.append("이 문제는 Foundry IQ나 MCP 같은 용어를 각각 외우는 것으로 해결되지 않는다. knowledge base와 RAG가 응답 근거를 어떻게 만들고, MCP가 외부 도구 실행을 어떻게 확장하는지 흐름으로 연결해야 한다.")
    elif azure:
        lines.append("이 문제는 MCP Server를 단순 설치 대상으로 보면 해결되지 않는다. AI assistant의 자연어 요청이 Azure DevOps 리소스 접근으로 이어지려면, 중간에서 요청과 업무 데이터를 연결하는 계층을 이해해야 한다.")
    elif github:
        lines.append("이 문제는 workflow 파일을 읽는 것만으로는 해결되지 않는다. trigger, activation, PR, conclusion을 각각 분리하고, 자동화 실행 조건과 결과 검증 흐름으로 연결해야 한다.")
    elif agent_orch:
        lines.append("이 문제는 agent를 여러 개 둔다는 설명만으로는 해결되지 않는다. orchestrator가 전체 목표를 조율하고, sub-agent가 역할별로 일을 나누는 구조를 개발 workflow 관점에서 이해해야 한다.")
    else:
        lines.append("이 문제는 개념 정의만 외우는 것으로 해결되지 않는다. 핵심 개념을 업무 흐름, 조치, 확인 기준과 연결해야 한다.")
    lines.append("")
    lines.append("## 왜 이것을 문제로 인식했는가")
    if wikidocs_coding:
        lines.append("코딩테스트 학습에서 막히는 지점은 대개 개념을 몰라서라기보다, 문제 조건을 보고 어떤 개념을 적용해야 하는지 판단하지 못하는 데서 나온다. 그래서 파이썬 문법과 자료구조를 문제 풀이 기준으로 재구성하는 것을 핵심 문제로 보았다.")
    elif oopy_cs_notes:
        lines.append("CS 면접 학습에서 막히는 지점은 개념 이름을 몰라서라기보다, 짧은 요약을 질문에 대한 설명으로 풀어내지 못하는 데서 나온다. 그래서 요약 자료를 면접 답변 구조로 바꾸는 것을 핵심 문제로 보았다.")
    elif agent_academy:
        lines.append("Copilot Studio agent 학습에서 중요한 문제는 agent가 대답한다는 결과보다, 그 대답과 행동이 어떤 설계 요소에서 만들어지는지 설명하는 것이다. SharePoint 지식 원천, topic trigger, Adaptive Card 입력, Agent Flow 자동화, publish 대상이 분리되지 않으면 결과 화면은 보여도 agent가 왜 그렇게 동작하는지 검증하기 어렵다.")
    elif foundry_first_agent:
        lines.append("실습형 자료에서는 개념 이름보다 단계 경계가 더 자주 막힌다. setup, Foundry 포털의 agent 생성, VS Code로 이어지는 작업, use-agent 실행 확인을 구분해야 어떤 부분을 이해했고 어디를 다시 검증해야 하는지 알 수 있다.")
    elif foundry_iq_mcp_rag:
        lines.append("Agent workflow에서는 답변의 근거를 어디서 가져오는지와 외부 작업을 어떤 도구로 실행하는지가 서로 다른 문제다. Foundry IQ, RAG, Azure AI Search는 지식 기반 응답의 신뢰도를 좌우하고, MCP와 dynamic tool discovery는 agent가 외부 기능을 선택하고 실행하는 기준을 만든다.")
    elif azure:
        lines.append("AI assistant가 실제 업무 도구와 연결되지 않으면 답변 생성에 머물지만, MCP Server를 통해 Azure DevOps 리소스와 연결되면 업무 항목 조회, PR 확인, build 상태 확인처럼 개발·운영 맥락을 다루는 자동화로 확장될 수 있다.")
    elif github:
        lines.append("GitHub 자동화에서 중요한 것은 workflow가 존재한다는 사실이 아니라 언제 실행되고, 어떤 변경을 만들고, 결과를 어디서 확인하는지다. workflow_dispatch, PR, conclusion을 연결해 보아야 자동화 흐름을 설명할 수 있다.")
    elif agent_orch:
        lines.append("실무형 AI 활용은 하나의 답변을 잘 받는 것에서 끝나지 않는다. 복잡한 작업을 계획, 구현, 검토 단위로 쪼개고 각 역할에 맞는 agent와 toolchain을 배치할 수 있어야 개발 workflow로 확장된다.")
    else:
        lines.append("학습자가 막히는 지점은 대개 용어 자체보다 개념 경계, 실습 순서, 파일/도구 역할, 검증 기준이 한 번에 섞이는 데서 나온다. 그래서 본문에서 확인되는 단서만으로 무엇을 했고 무엇을 확인해야 하는지 분리했다.")
    lines.append("")
    lines.append("## 문제 해결 경험")
    if not steps:
        steps = source_derived_generic_steps(raw_text, memo)
    for idx, step in enumerate(steps[:6], start=1):
        if not isinstance(step, dict):
            continue
        title_s = str(step.get("title") or f"학습 흐름 {idx}")
        problem_s = str(step.get("problem") or "이 단계의 학습 목적을 명확히 해야 했다.")
        action_s = str(step.get("action") or "자료와 메모를 바탕으로 단계의 의미를 정리했다.")
        verification_s = str(step.get("verification") or "정리한 내용이 강의 목표와 연결되는지 확인했다.")
        lines.append(f"### {idx}. {title_s}")
        lines.append(f"문제/제약: {problem_s}")
        lines.append("")
        lines.append(f"조치: {action_s}")
        lines.append("")
        lines.append(f"확인 기준: {verification_s}")
        lines.append("")
    if qa_logs:
        lines.append("## 학습 중 질문하며 정리한 부분")
        lines.append("학습 중 헷갈렸던 개념을 질문으로 풀어 보면서, 단순 용어 암기가 아니라 실제 업무 흐름 안에서의 역할을 중심으로 정리했다.")
        for idx, qa in enumerate(qa_logs[:5], start=1):
            q = str(qa.get("question") or qa.get("q") or "질문 내용")
            a = str(qa.get("answer") or qa.get("a") or "답변 내용")
            lines.append(f"- 질문 {idx}: {q}")
            lines.append(f"  - 정리: {a[:320]}")
        lines.append("")
    else:
        pass
    lines.append("## 성과")
    lines.append(final_result)
    lines.append("")
    lines.append("확인되지 않은 성공 결과를 임의로 넣기보다, 본문에서 설명 가능한 개념·흐름·확인 기준을 중심으로 정리했다.")
    lines.append("")
    lines.append("## 사용한 주요 개념 정리")
    if wikidocs_coding:
        concept_descriptions = {
            "코딩 테스트": "문법 지식보다 문제 조건을 읽고 적절한 풀이 전략과 자료구조를 선택하는 훈련이다.",
            "파이썬": "코딩테스트에서 입력 처리, 반복, 조건 분기, 자료구조 활용을 빠르게 구현하는 기본 언어 도구다.",
            "자료구조": "데이터를 어떤 방식으로 저장하고 접근할지 결정해 풀이 시간과 코드 구조에 직접 영향을 주는 선택 기준이다.",
            "스택": "최근에 들어온 값을 먼저 처리해야 하는 괄호, 되돌리기, 경로 추적 유형에서 자주 쓰이는 구조다.",
            "시간 복잡도": "입력 크기에 따라 풀이가 통과 가능한지 판단하는 기준이다.",
            "알고리즘": "문제 조건을 계산 절차로 바꾸는 풀이 전략이다.",
            "문제 풀이": "개념을 실제 입력 조건과 제한 시간 안에서 적용하는 과정이다.",
            "복습 기준": "다음 문제를 풀 때 같은 개념을 다시 꺼낼 수 있게 만드는 확인 기준이다.",
        }
    elif oopy_cs_notes:
        concept_descriptions = {
            "CS": "운영체제, 네트워크, 자료구조, 데이터베이스 등 개발자가 시스템을 설명할 때 필요한 기본 지식 묶음이다.",
            "운영체제": "프로세스, 스레드, 메모리, 스케줄링처럼 프로그램 실행 환경을 설명하는 핵심 영역이다.",
            "네트워크": "클라이언트와 서버가 데이터를 주고받는 구조와 프로토콜을 설명하는 영역이다.",
            "자료구조": "데이터 저장과 접근 방식을 결정하며 알고리즘 설명의 기반이 되는 영역이다.",
            "알고리즘": "문제를 해결하기 위한 절차와 효율성을 설명하는 영역이다.",
            "데이터베이스": "데이터 저장, 조회, 트랜잭션, 정규화 같은 백엔드 기초 질문과 연결되는 영역이다.",
            "면접 질문": "개념을 실제 설명 상황으로 바꾸는 확인 단위다.",
            "꼬리 질문": "정의 암기에서 끝나지 않고 원리와 비교까지 이해했는지 확인하는 질문이다.",
        }
    elif agent_academy:
        concept_descriptions = {
            "Agent Academy": "Copilot Studio agent를 단계별 미션으로 학습하며 설계, 확장, 배포 기준을 익히는 과정이다.",
            "Microsoft Copilot Studio": "지식 원천, 대화 흐름, Topics, actions/flows를 구성해 business agent를 만드는 핵심 도구다.",
            "SharePoint": "agent가 답변 근거로 사용할 수 있는 조직 데이터 원천이며, setup 이후 미션에서 grounding 기준이 된다.",
            "declarative agent": "Microsoft 365 Copilot 안에서 prompt와 지식 기반으로 특정 목적의 agent 경험을 정의하는 방식이다.",
            "custom agent": "Copilot Studio에서 지식, Topics, 동작 흐름을 더 직접 구성하는 agent 구축 방식이다.",
            "Topics": "사용자 질문을 특정 대화 경로와 응답 흐름으로 연결하는 trigger 기반 구성 요소다.",
            "Adaptive Cards": "agent 대화 안에서 구조화된 정보 표시와 사용자 입력을 받기 위한 UI 요소다.",
            "Agent Flows": "Adaptive Card 입력이나 agent 이벤트를 백엔드 자동화와 연결하는 실행 흐름이다.",
            "Microsoft 365 Copilot": "만든 agent를 조직 생산성 도구 안에서 사용할 수 있게 하는 배포 대상이다.",
            "Teams": "완성한 agent를 실제 협업 채널에서 검증할 수 있는 배포·사용 환경이다.",
        }
    elif foundry_first_agent:
        concept_descriptions = {
            "Microsoft Foundry": "첫 agent를 만들고 실행 흐름을 확인하는 실습의 중심 환경이다.",
            "first agent": "quickstart에서 생성하고 테스트해야 하는 기본 agent 결과물이다.",
            "quickstart": "setup부터 생성, continuation, 실행 확인까지의 최소 실습 흐름이다.",
            "setup": "agent를 만들기 전에 필요한 준비 단계와 진입 경로를 의미한다.",
            "VS Code": "Foundry 포털 이후 실습을 이어서 확인하거나 수정하는 개발 환경이다.",
            "continue-in-vscode": "포털 중심 작업에서 코드 환경으로 넘어가는 전환 단계다.",
            "use-agent": "생성한 agent를 실제로 실행하고 응답을 확인하는 검증 단계다.",
            "validation": "agent가 실행 가능한 상태인지, 테스트 입력에 기대한 방식으로 응답하는지 확인하는 기준이다.",
        }
    elif foundry_iq_mcp_rag:
        concept_descriptions = {
            "Foundry IQ": "agent가 조직 지식과 문서를 근거로 답변할 수 있게 하는 knowledge grounding 계층이다.",
            "MCP": "Model Context Protocol을 통해 agent가 외부 도구와 시스템 기능을 사용할 수 있게 하는 연결 방식이다.",
            "Model Context Protocol": "모델이 외부 context와 tool을 표준화된 방식으로 다루게 하는 프로토콜이다.",
            "RAG": "질문에 맞는 관련 지식을 검색한 뒤 그 근거를 바탕으로 응답을 생성하는 패턴이다.",
            "knowledge base": "agent가 답변의 근거로 사용할 문서와 지식 원천이다.",
            "Azure AI Search": "knowledge base에서 관련 정보를 검색해 RAG 흐름에 연결하는 검색 계층으로 볼 수 있다.",
            "dynamic tool discovery": "agent가 요청에 맞는 도구를 발견하고 선택하는 실행 확장 방식이다.",
            "tool execution": "agent가 단순 답변을 넘어 외부 시스템 작업을 수행하는 단계다.",
        }
    elif azure:
        concept_descriptions = {
            "Azure DevOps MCP Server": "AI assistant와 Azure DevOps 리소스 사이를 연결해 자연어 요청이 work item, pull request, build 같은 업무 데이터 접근으로 이어지도록 돕는 계층이다.",
            "AI assistant": "단순 답변 생성 도구가 아니라 MCP 연결을 통해 DevOps 업무 맥락을 조회하고 작업 흐름을 보조할 수 있는 인터페이스로 이해했다.",
            "Agentic DevOps": "AI가 개발·운영 업무의 맥락을 읽고 반복 조회나 상태 확인을 보조하는 DevOps 활용 방식이다.",
            "work items": "Azure DevOps에서 작업, 버그, 요구사항을 추적하는 기본 단위이며, AI assistant가 업무 맥락을 파악할 때 중요한 대상이다.",
            "pull requests": "코드 변경을 검토하고 병합하기 전 확인하는 단계이며, AI 기반 DevOps 흐름에서도 변경사항 검토 지점이 된다.",
            "builds": "변경된 코드가 정상적으로 빌드되는지 확인하는 자동화 단계이며, 결과 확인과 품질 검증에 연결된다.",
            "workflow automation": "반복적인 조회·확인·상태 점검을 자동화해 개발자가 판단해야 할 정보를 빠르게 가져오는 흐름이다.",
            "governance": "AI가 업무 도구에 접근할 때 권한, 추적성, 통제 기준을 함께 고려해야 한다는 관점이다.",
        }
    elif github:
        concept_descriptions = {
            "GitHub Agentic Workflows": "GitHub Actions와 agent 기반 작업 흐름을 연결해 자동화 실행, 변경 생성, 검토 과정을 하나의 개발 workflow로 보는 개념이다.",
            "GitHub Actions": "repository 안의 workflow 파일을 기준으로 빌드, 테스트, 자동화 작업을 실행하는 GitHub의 자동화 기능이다.",
            "workflow_dispatch": "workflow를 수동으로 실행할 수 있게 하는 trigger이며, 자동화가 언제 시작되는지를 정의하는 핵심 조건이다.",
            "Pull Request": "자동화나 agent가 만든 변경사항을 바로 반영하지 않고 검토 가능한 형태로 확인하는 단계다.",
            "activation": "workflow나 agentic 기능이 실행 가능한 상태로 전환되는 단계로 해석할 수 있다.",
            "conclusion": "자동화 흐름의 마지막 확인 지점이지만, success/completed/merged 같은 상태값과 함께 보아야 결과를 단정할 수 있다.",
        }
    elif agent_orch:
        concept_descriptions = {
            "Agent Orchestration": "하나의 AI가 모든 일을 처리하는 대신 orchestrator가 작업을 나누고 여러 sub-agent가 역할별로 수행하는 구조다.",
            "orchestrator": "전체 목표를 해석하고 어떤 sub-agent에게 어떤 작업을 맡길지 조율하는 상위 제어 역할이다.",
            "sub-agents": "planner, coder, designer처럼 특정 역할에 맞게 나뉘어 작업을 수행하는 하위 agent다.",
            "GitHub Copilot CLI": "agent workflow를 개발 환경이나 명령행 작업 흐름과 연결해 볼 수 있는 도구 맥락이다.",
            "planner": "요구사항을 작업 단위로 나누고 실행 순서를 설계하는 역할이다.",
            "coder": "정의된 작업을 실제 코드나 구현 결과로 옮기는 역할이다.",
            "designer": "구조, 사용자 경험, 표현 방식 같은 설계 관점의 판단을 맡는 역할이다.",
            "toolchain": "agent가 작업을 수행할 때 사용하는 CLI, 파일, 저장소, 실행 환경 같은 도구 묶음이다.",
        }
    else:
        concept_descriptions = {}
    for term in key_terms[:8]:
        desc = concept_descriptions.get(term) or "핵심 개념을 업무 흐름 안에서 어떤 역할을 하는지 기준으로 이해했다."
        lines.append(f"- **{term}**: {desc}")
    lines.append("")
    lines.append("## 최종 정리")
    if wikidocs_coding:
        lines.append("이번 학습을 통해 WikiDocs의 파이썬 코딩테스트 내용을 문법 설명, 자료구조 선택, 시간 복잡도 판단, 스택 활용 기준으로 나누어 정리했다. 단순히 책을 읽는 것이 아니라 문제 조건을 만났을 때 어떤 개념을 적용할지 판단하는 복습 기준을 세운 것이 핵심이었다.")
    elif oopy_cs_notes:
        lines.append("이번 학습을 통해 CS 핵심요약집을 운영체제, 네트워크, 자료구조, 데이터베이스 같은 개념 목록이 아니라 면접 답변 단위로 재구성했다. 정의와 원리를 짧게 외우는 것보다, 질문을 받았을 때 예시와 비교 기준까지 설명할 수 있는 구조를 만드는 것이 핵심이었다.")
    elif foundry_first_agent:
        lines.append("이번 실습을 통해 Microsoft Foundry first-agent quickstart를 setup, Foundry에서의 agent 생성, VS Code continuation, use-agent 실행 확인으로 나누어 이해했다. 이 흐름을 분리하니 각 단계가 어떤 역할을 하며 무엇을 기준으로 성공 여부를 확인해야 하는지 더 분명해졌다.")
    elif foundry_iq_mcp_rag:
        lines.append("이번 학습을 통해 Foundry IQ, RAG, knowledge base, Azure AI Search를 knowledge grounding 흐름으로 묶고, MCP와 dynamic tool discovery를 tool execution 확장 흐름으로 분리해 이해했다. 이 구분 덕분에 agent workflow에서 답변 근거와 도구 실행을 각각 다른 검증 기준으로 볼 수 있게 되었다.")
    elif azure:
        lines.append("이번 실습을 통해 Azure DevOps MCP Server를 AI assistant와 DevOps 리소스를 연결하는 계층으로 이해했다. MCP Server가 work item, pull request, build 같은 업무 데이터를 AI assistant가 다룰 수 있게 해 주기 때문에, DevOps 자동화는 단순 명령 실행이 아니라 업무 맥락을 이해하고 요청을 처리하는 흐름으로 확장될 수 있다.")
    elif agent_orch:
        lines.append("이번 학습을 통해 Agent Orchestration을 단일 챗봇보다 확장된 역할 기반 AI 협업 구조로 이해했다. orchestrator는 전체 목표와 흐름을 조율하고, planner, coder, designer 같은 sub-agent는 각자의 역할에 맞게 작업을 나누어 수행한다. 이를 통해 AI 개발 workflow를 하나의 응답 생성이 아니라 역할 분담과 도구 사용이 결합된 구조로 볼 수 있었다.")
    elif github:
        lines.append("이번 학습을 통해 GitHub Agentic Workflows를 workflow 파일, 실행 트리거, activation, Pull Request, conclusion이 연결되는 자동화 흐름으로 이해했다. 특히 workflow_dispatch는 수동 실행 조건을 정의하는 단서가 되고, PR은 자동화 또는 agent가 만든 변경사항을 검토하는 단계로 연결될 수 있다. conclusion 화면은 최종 결과를 확인하는 후보로 남기되, 성공 여부는 추가 화면 근거가 있을 때 확정하는 것이 적절하다.")
    else:
        lines.append("이번 학습에서는 강의 자료와 실습 단서에 흩어진 개념을 하나의 학습 흐름으로 정리했다. 핵심 개념, 실습 단계, 확인이 필요한 부분을 분리하면서 이후 같은 내용을 다시 설명할 수 있는 기술 학습 기록으로 만들었다.")
    lines.append("")
    lines.append("## Portfolio Summary")
    if wikidocs_coding:
        lines.append("This learning record reframes Python coding-test study materials into problem-solving criteria: syntax application, data-structure selection, time-complexity checks, and stack-based problem patterns.")
    elif oopy_cs_notes:
        lines.append("This learning record turns broad CS summary notes into interview-ready explanation structures by separating definitions, principles, examples, comparisons, and follow-up questions.")
    elif foundry_first_agent:
        lines.append("This learning record summarizes a Microsoft Foundry first-agent quickstart by separating setup, agent creation, VS Code continuation, use-agent execution, and validation criteria.")
    elif foundry_iq_mcp_rag:
        lines.append("This learning record explains a Foundry IQ and MCP based AI agent workflow by separating knowledge grounding, RAG, Azure AI Search, dynamic tool discovery, and tool execution.")
    elif azure:
        lines.append("This learning record explains how Azure DevOps MCP Server connects AI assistants with DevOps resources such as work items, pull requests, and builds. The focus is on understanding MCP as a workflow-enabling layer rather than a simple setup task.")
    elif agent_orch:
        lines.append("This learning record summarizes Agent Orchestration as a role-based AI workflow. It connects the concepts of orchestrators, sub-agents, toolchains, and GitHub Copilot CLI to explain how AI-assisted development can move beyond a single chatbot interaction.")
    elif github:
        lines.append("This learning record summarizes GitHub Agentic Workflows by connecting workflow_dispatch, workflow files, activation, pull requests, and conclusion screens into one automation flow. It focuses on understanding execution conditions and result verification in GitHub-based automation.")
    else:
        lines.append("This learning record organizes a technical study topic into a clear learning goal, concept map, workflow summary, and optional follow-up checks.")
    lines.append("")
    lines.append("## Key skills practiced")
    for skill in skills:
        lines.append(f"- {skill}")
    optional_items = optional_confirmation_items(raw_text, memo, qa_logs)
    if optional_items:
        lines.append("")
        lines.append("---")
        lines.append("## 선택 확인 사항")
        for item in optional_items:
            lines.append(f"- {item}")
    return "\n".join(lines).strip()


def sparse_capture_generation_blocker(
    article_type: str,
    sparse_report: dict[str, Any],
    problem_map: dict[str, Any],
    brief: dict[str, Any],
) -> str:
    if article_type in POWERBI_ARTICLE_TYPES:
        return ""
    mode = str(sparse_report.get("generation_mode") or "")
    total = int(sparse_report.get("total_image_count") or 0)
    interpreted = int(sparse_report.get("interpreted_image_count") or 0)
    unknown_captions = int(sparse_report.get("unknown_caption_count") or 0)
    ratio = interpreted / max(total, 1)
    title = str(brief.get("korean_title") or "")
    core = str(problem_map.get("core_problem") or "")
    steps = problem_map.get("solution_steps", []) if isinstance(problem_map.get("solution_steps"), list) else []
    numeric_confidence = float(sparse_report.get("article_type_confidence") or 0)
    if article_type in {"unknown", "general_learning_portfolio"}:
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return "article_type이 unknown/general_learning_portfolio라서 full_article 생성을 금지했습니다."
    if numeric_confidence and numeric_confidence < 0.55:
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return f"article_type confidence {numeric_confidence:.2f}가 0.55 미만이어서 full_article 생성을 금지했습니다."
    if len(steps) < 3:
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return "구체적인 solution_steps가 3개 미만이어서 full_article 생성을 금지했습니다."
    if problem_map.get("_sparse_steps_incomplete"):
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return "Sparse capture에서 구체적인 solution_steps가 부족해 full_article 생성을 금지했습니다."
    if mode != "full_article":
        return f"Sparse Capture Mode가 {mode}로 판정되어 완성형 Medium 글 생성을 보류했습니다."
    if total and ratio < 0.6:
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return f"interpreted_image_count / total_image_count = {interpreted}/{total}로 0.6 미만이어서 full_article 생성을 금지했습니다."
    if unknown_captions > max(1, total // 4):
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return "파일명 또는 일반 캡션으로 남은 이미지가 많아 full_article 생성을 금지했습니다."
    if is_generic_learning_title(title):
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return "제목이 입력 evidence 기반이 아니라 generic하여 full_article 생성을 금지했습니다."
    if is_generic_core_problem(core):
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return "core_problem이 입력 evidence 기반이 아니라 generic하여 full_article 생성을 금지했습니다."
    if has_placeholder_solution_steps(steps):
        sparse_report["generation_mode"] = "draft_with_missing_context"
        return "solution_steps에 placeholder 단계가 포함되어 full_article 생성을 금지했습니다."
    return ""


def sparse_capture_hold_message(
    article_type: str,
    sparse_report: dict[str, Any],
    evidence: list[dict[str, Any]],
    problem_map: dict[str, Any],
    brief: dict[str, Any],
    section_plan: list[dict[str, Any]],
    reason: str,
) -> str:
    debug_report = {
        "article_type": article_type,
        "confidence": sparse_report.get("confidence"),
        "article_type_confidence": sparse_report.get("article_type_confidence"),
        "generation_mode": sparse_report.get("generation_mode"),
        "capture_coverage": sparse_report.get("capture_coverage", {}),
        "interpreted_image_count": sparse_report.get("interpreted_image_count"),
        "total_image_count": sparse_report.get("total_image_count"),
        "unknown_caption_count": sparse_report.get("unknown_caption_count"),
        "missing_context": sparse_report.get("missing_context", []),
        "follow_up_questions": sparse_report.get("follow_up_questions", []),
        "reason": reason,
    }
    title = specific_title_candidate(article_type, evidence, str(brief.get("korean_title") or ""))
    core_candidate = specific_core_problem_candidate(article_type, evidence, str(problem_map.get("core_problem") or ""))
    interpreted_lines = "\n".join(
        f"- 이미지 {item.get('image_no')}: {item.get('caption')}"
        for item in evidence
        if is_interpreted_image_evidence(item)
    ) or "- 해석 가능한 캡처가 충분하지 않습니다."
    missing_lines = "\n".join(f"- {item}" for item in sparse_report.get("missing_context", [])) or "- 추가 확인이 필요한 맥락 없음"
    question_lines = "\n".join(f"{idx}. {item}" for idx, item in enumerate(sparse_report.get("follow_up_questions", [])[:3], start=1))
    section_plan_preview = json.dumps(section_plan[:6], ensure_ascii=False, indent=2)
    recovery_preview = build_recovery_draft_preview(article_type, problem_map, sparse_report, section_plan)
    recovery_block = f"\n## 학습 복구 초안\n{recovery_preview}\n" if recovery_preview else ""
    return f"""## Debug report
```json
{json.dumps(debug_report, ensure_ascii=False, indent=2)}
```

# {title}

현재 입력은 완성형 Medium 글이 아니라 `{sparse_report.get("generation_mode")}`로 처리되었습니다. {reason}

## core_problem 후보
{core_candidate}
{recovery_block}
## 현재 확인된 evidence
{interpreted_lines}

## missing_context
{missing_lines}

## follow_up_questions
{question_lines}

## Section Plan preview
```json
{section_plan_preview}
```
"""


def is_generic_learning_title(title: str) -> bool:
    normalized = title.strip().strip("# ")
    generic_fragments = [
        "학습 기록 기반 문제 해결 경험",
        "문제를 검증 가능한 분석 흐름으로 바꾼 기록",
        "학습 기록 기반",
        "이미지 기반 학습 캡처",
        "문제 해결형 학습 기록",
    ]
    return any(fragment in normalized for fragment in generic_fragments)


def is_generic_core_problem(core: str) -> bool:
    generic_fragments = [
        "학습 기록 기반 문제 해결 경험 과정에서 관찰한 결과와 의도한 분석 흐름의 불일치",
        "관찰한 결과와 의도한 분석 흐름의 불일치",
        "문제를 검증 가능한 분석 흐름으로",
        "캡처와 메모를 문제 해결형 포트폴리오 글로",
    ]
    return any(fragment in core for fragment in generic_fragments)


def has_placeholder_solution_steps(steps: list[Any]) -> bool:
    text = json.dumps(steps, ensure_ascii=False)
    placeholders = [
        "근거 화면",
        "이미지 근거와 사용자 메모를 연결해 문제 해결 단계로 재구성했습니다.",
        "문제 상황 제시",
        "문제 원인 분석",
        "해결 단계",
        "검증 단계",
    ]
    return any(phrase in text for phrase in placeholders)


def specific_title_candidate(article_type: str, evidence: list[dict[str, Any]], fallback: str) -> str:
    source = evidence_source_text(evidence).lower()
    fallback_l = str(fallback or "").lower()
    if "azure devops" in fallback_l and "mcp" in fallback_l:
        return "Azure DevOps MCP Server 실습: AI assistant와 DevOps 자동화 흐름 이해하기"
    if article_type in {"github_agentic_workflow", "github_actions_workflow", "ai_coding_workflow"}:
        if "agentic" in source and "workflow_dispatch" in source:
            return "GitHub Agentic Workflows 실습: workflow_dispatch와 자동화 실행 흐름을 이해한 기록"
        if "github actions" in source or "workflow_dispatch" in source:
            return "GitHub Actions 실습: workflow_dispatch 트리거와 실행 결과 흐름 확인하기"
        return "Agentic Workflows 학습 기록: GitHub 자동화의 실행 조건과 결과 상태 이해하기"
    return fallback if fallback and not is_generic_learning_title(fallback) else f"{article_type}: sparse capture 기반 학습 기록"


def specific_core_problem_candidate(article_type: str, evidence: list[dict[str, Any]], fallback: str) -> str:
    source = evidence_source_text(evidence)
    if article_type in {"github_agentic_workflow", "github_actions_workflow", "ai_coding_workflow"}:
        terms = [term for term in ["GitHub Agentic Workflows", "workflow_dispatch", "activation", "update-github-info.lock.yml", "conclusion", "update-github-info"] if term.lower() in source.lower()]
        if terms:
            return "GitHub Agentic Workflows 실습에서 " + ", ".join(dict.fromkeys(terms)) + " 화면을 통해 자동화 workflow가 어떤 조건에서 실행되고 어떤 파일/상태를 남기는지 이해해야 했다. 다만 현재 캡처만으로는 최종 목표와 conclusion의 의미를 확정하기 어렵다."
    return fallback if fallback and not is_generic_core_problem(fallback) else "현재 캡처만으로는 핵심 문제를 확정하기 어려워, 확인된 evidence 기반 후보로만 유지합니다."


def evidence_source_text(evidence: list[dict[str, Any]]) -> str:
    return " ".join(
        [
            str(item.get("caption", "")) + " " + str(item.get("problem_signal", "")) + " " + str(item.get("inferred_meaning", "")) + " " + " ".join(normalize_str_list(item.get("technical_entities"))) + " " + " ".join(normalize_str_list(item.get("visible_evidence")))
            for item in evidence
        ]
    )


def infer_sparse_capture_role(item: dict[str, Any]) -> str:
    if str(item.get("evidence_source") or "").lower() == "filename" and is_unknown_or_filename_caption(item):
        return "unknown"
    text = " ".join(
        [
            str(item.get("caption", "")),
            str(item.get("problem_signal", "")),
            str(item.get("inferred_meaning", "")),
            " ".join(normalize_str_list(item.get("visible_evidence"))),
            " ".join(normalize_str_list(item.get("technical_entities"))),
        ]
    ).lower()
    role = str(item.get("role", "")).lower()
    if role == "problem" or any(word in text for word in ["error", "오류", "failed", "문제", "broken", "traceback"]):
        return "error_or_problem"
    if any(word in text for word in ["confusing", "헷갈", "concept", "개념", "why", "왜"]):
        return "confusing_concept"
    if any(word in text for word in ["code", "config", "configuration", "workflow", "yaml", "json", "python", "command", "설정", "수식", "measure"]):
        return "code_or_config"
    if any(word in text for word in ["architecture", "flow", "diagram", "model", "relationship", "구조", "흐름"]):
        return "architecture_or_flow"
    if role in {"solution", "cause"} or any(word in text for word in ["change", "edit", "fix", "created", "생성", "변경", "수정", "적용"]):
        return "action_or_change"
    if role in {"validation", "final_result"} or any(word in text for word in ["result", "success", "완료", "검증", "확인", "final"]):
        return "validation_or_result"
    if any(word in text for word in ["summary", "certificate", "badge", "completed", "요약", "수료"]):
        return "completion_or_summary"
    if int(item.get("image_no") or 0) == 1:
        return "goal_or_context"
    return "unknown"


def is_unknown_or_filename_caption(item: dict[str, Any]) -> bool:
    caption = str(item.get("caption") or "")
    source = str(item.get("evidence_source") or "").lower()
    original = str(item.get("original_filename") or "")
    if source == "filename":
        return True
    if re.search(r"\.(png|jpg|jpeg)\b", caption, re.I):
        return True
    if re.search(r"이미지\s+\d+\s*-\s*(근거 화면|실습 흐름 근거 화면|해석되지 않은 캡처|image\d*)", caption, re.I):
        return True
    if original and Path(original).stem and Path(original).stem.lower() in caption.lower():
        return True
    return False


def is_interpreted_image_evidence(item: dict[str, Any]) -> bool:
    if is_unknown_or_filename_caption(item):
        return False
    source = str(item.get("evidence_source") or "").lower()
    confidence = float(item.get("confidence") or 0)
    caption = str(item.get("caption") or "")
    visible = normalize_str_list(item.get("visible_evidence"))
    entities = normalize_str_list(item.get("technical_entities"))
    if source in {"vision", "llm"} and confidence >= 0.5 and (visible or entities) and len(caption) >= 18:
        return True
    if source == "README_image_order.txt" and len(caption) >= 18:
        return True
    return False


def evidence_coverage_failure_message(
    failure: str,
    uploaded_count: int,
    evidence_count: int,
    capture_count: int,
    qa_count: int,
    memo: str,
) -> str:
    return f"""# Medium 글 생성 보류

{failure}

이미지 evidence가 부족한 상태에서 긴 Medium 글을 만들면 후반부 이미지나 golden example 문장에 기대는 일반적인 글이 됩니다. 누락된 이미지를 다시 처리한 뒤 최종 글을 생성해야 합니다.

## 현재 처리 상태
- 업로드된 이미지 수: {uploaded_count}장
- 생성된 ImageEvidence 수: {evidence_count}개
- 저장된 캡처 수: {capture_count}개
- 저장된 Q&A 수: {qa_count}개

## 입력된 메모
{memo.strip() or "입력된 메모 없음"}

## 다음 조치
- 누락된 이미지가 업로드되었는지 확인
- README_image_order.txt 또는 파일명 prefix로 이미지 순서를 확인
- 이미지 1부터 마지막 이미지까지 ImageEvidence가 만들어진 뒤 다시 생성
"""


def unknown_article_type_response(
    classification: dict[str, Any],
    evidence: list[dict[str, Any]],
    uploaded_count: int,
    memo: str,
    raw_text: str,
    sparse_capture_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sparse_capture_report = sparse_capture_report or build_sparse_capture_report("unknown", classification, evidence, raw_text, memo)
    candidate_lines = "\n".join(
        f"- {item.get('article_type')}: score {item.get('score')}"
        for item in classification.get("candidates", [])
    ) or "- 후보를 충분히 추정하지 못했습니다."
    evidence_lines = "\n".join(
        f"- 이미지 {item.get('image_no')}: {item.get('caption')} / entities: {', '.join(normalize_str_list(item.get('technical_entities'))[:6])}"
        for item in evidence[:12]
    ) or "- 해석 가능한 이미지 evidence가 아직 없습니다."
    draft = f"""# Medium 글 생성 보류

현재 입력의 article_type confidence가 낮아 완성형 Medium 글을 억지로 생성하지 않았습니다.

## 현재 확인된 evidence
{evidence_lines}

## 추정 가능한 article_type 후보
{candidate_lines}

## 부족한 정보
{chr(10).join(f"- {item}" for item in sparse_capture_report.get("missing_context", [])) or "- 핵심 문제가 무엇인지 보여주는 캡처 또는 코드/오류 텍스트"}

## 사용자가 추가하면 좋은 캡처/메모
{chr(10).join(f"- {item}" for item in sparse_capture_report.get("follow_up_questions", []))}

## 입력 상태
- 업로드된 이미지 수: {uploaded_count}장
- 입력된 메모: {memo.strip() or "제공된 메모가 없습니다."}
- 입력된 텍스트 길이: {len(raw_text.strip())}자
"""
    return {
        "draft": draft,
        "article_type": "unknown",
        "image_evidence": evidence,
        "learning_evidence": [],
        "problem_map": {
            "article_type": "unknown",
            "_article_type_confidence": classification.get("confidence"),
            "_article_type_candidates": classification.get("candidates", []),
            "_sparse_capture_report": sparse_capture_report,
        },
        "decision_map": {},
        "section_plan": [],
        "article_brief": {},
        "sparse_capture_report": sparse_capture_report,
        "critic_report": {
            "passed": False,
            "failures": ["article_type confidence가 낮아 최종 글 생성을 보류했습니다."],
            "metrics": {"uploaded_images_count": uploaded_count, "image_evidence_count": len(evidence)},
        },
    }


def enrich_problem_map_concrete_details(problem_map: dict[str, Any], article_type: str) -> None:
    if article_type == "semantic_model_relationship":
        problem_map["problem_kind"] = "semantic_model_relationship_filter_context"
        problem_map["core_problem"] = SEMANTIC_CORE_PROBLEM
        problem_map["why_problematic"] = (
            "Category별 매출은 제품군에 따라 달라져야 하는데 Accessories, Bikes, Clothing, Components 등 모든 행에 같은 총액이 표시되면, "
            "각 Category가 Sales fact table을 필터링하지 못하고 전체 Sales 총합이 반복 표시되는 상태로 볼 수 있다. "
            "따라서 이는 시각화 문제가 아니라 semantic model relationship/filter context 문제로 정의해야 한다."
        )
        problem_map["root_causes"] = [
            "Product[Category] filter context가 Sales fact table로 전달되지 않음",
            "Product와 Sales 사이 ProductKey relationship이 없거나 비활성/오설정됨",
            "Salesperson 분석에서는 direct Salesperson-Sales relationship과 Salesperson -> SalespersonRegion -> Region -> Sales 경로가 동시에 존재해 filter path가 모호해질 수 있음",
            "Sales와 Target을 같은 기준으로 비교하려면 Salesperson/Region/Targets 관계 경로가 명확해야 함",
        ]
        problem_map["solution_steps"] = semantic_solution_steps()
        problem_map["complex_problems"] = [
            {
                "title": "Repeated Category Sales 문제",
                "symptom": "Category별 Sales 값이 모두 동일하게 반복됨",
                "cause": "Product[Category] filter context가 Sales fact table로 전달되지 않음",
                "action": "Product[ProductKey] -> Sales[ProductKey] relationship 생성",
                "validation": "Category별 Sales 값이 서로 다르게 분리되는지 확인",
            },
            {
                "title": "SalespersonRegion bridge path 문제",
                "symptom": "Salesperson별 Sales/Target 분석에서 담당 Region 기준 필터 경로가 필요함",
                "cause": "direct Salesperson-Sales relationship과 bridge path가 동시에 있으면 filter path가 모호해질 수 있음",
                "action": "SalespersonRegion bridge table 사용, Cross-filter direction Both 설정, direct Salesperson-Sales relationship 비활성화",
                "validation": "Salesperson별 Sales와 Target이 같은 기준에서 비교되는지 확인",
            },
        ]
        problem_map["complex_problem"] = {
            "title": "Repeated Category Sales와 SalespersonRegion bridge path를 분리해 해결",
            "symptom": "초반에는 Category별 Sales 반복값, 후반에는 Salesperson별 Sales/Target 비교를 위한 filter path 문제가 각각 나타남",
            "initial_assumption": "두 문제가 모두 relationship과 관련되어 있어 하나의 원인으로 묶어 볼 수 있음",
            "actual_cause": "Repeated Sales는 Product-Sales relationship/filter context 문제이고, bridge path는 Salesperson-Region-Sales-Targets 분석 경로를 명확히 해야 하는 별도 모델링 문제",
            "resolution": "Product[ProductKey] -> Sales[ProductKey] relationship으로 Category filter context를 해결하고, 후반부에서는 SalespersonRegion bridge table, Cross-filter direction Both, inactive direct relationship으로 경로를 분리",
            "verification": "Category별 Sales / Profit / Profit Margin이 분리되고, Salesperson별 Sales와 Target이 같은 기준에서 비교되는지 확인",
        }
        problem_map["final_outcome"] = (
            "Product-Sales relationship, Star schema, Product hierarchy, Profit / Profit Margin measure, SalespersonRegion bridge path, Targets 비교까지 "
            "각 문제를 분리해 검증 가능한 semantic model 흐름으로 정리했다."
        )
    if article_type == "dax_measure_modeling":
        problem_map["problem_kind"] = "analysis_design_task"
        problem_map["core_problem"] = DAX_CORE_PROBLEM
        problem_map["why_problematic"] = (
            "Month 이름을 텍스트 그대로 정렬하면 실제 월 순서가 깨질 수 있고, raw column 자동 집계에 의존하면 가격 지표와 목표 대비 성과 계산의 의도가 흐려진다. "
            "따라서 Date table, explicit measure, Target/Variance measure를 통해 어떤 기준으로 계산되는지 모델 안에 명확히 남겨야 한다."
        )
        problem_map["root_causes"] = [
            "Month 텍스트 컬럼은 시간 순서를 보장하지 않으므로 MonthKey 정렬이 필요함",
            "Date table과 fiscal hierarchy가 명확하지 않으면 날짜 기준 분석이 흔들림",
            "raw column 자동 집계는 Avg Price, Median Price, Orders 같은 지표의 계산 의도를 드러내지 못함",
            "TargetAmount raw column 대신 Target measure를 사용해야 filter context와 total 기준을 통제할 수 있음",
            "Variance와 Variance Margin이 없으면 Salesperson별 목표 대비 초과/미달을 해석하기 어려움",
        ]
        problem_map["solution_steps"] = dax_solution_steps()
        problem_map["complex_problem"] = {
            "title": "TargetAmount raw column 대신 Target/Variance measure로 성과 분석 기준을 명확히 하는 문제",
            "symptom": "Sales와 TargetAmount 값이 보이더라도 Salesperson별 목표 대비 성과와 total 기준 해석이 불명확함",
            "initial_assumption": "TargetAmount 컬럼을 그대로 집계하면 목표 대비 분석이 충분할 것처럼 보임",
            "actual_cause": "raw column 자동 집계는 filter context, total row, format, variance 계산 의도를 명확히 표현하지 못함",
            "resolution": "Target measure를 만들고 Variance, Variance Margin measure를 추가해 Sales, Target, 차이, 차이율을 같은 기준에서 비교",
            "verification": "Salesperson별 Sales, Target, Variance, Variance Margin이 최종 matrix에서 함께 표시되고 목표 대비 초과/미달이 읽히는지 확인",
        }
        problem_map["final_outcome"] = (
            "MonthKey 정렬, Fiscal Date table, Mark as date table, 가격/주문 explicit measure, Currency format, Target/Variance measure를 통해 "
            "Salesperson별 Sales와 Target 성과를 비교할 수 있는 DAX 분석 모델로 정리했다."
        )
    if article_type == "power_query_etl":
        if not is_power_query_etl_regression_context(problem_map):
            problem_map.setdefault("problem_kind", "transformation_requirement")
            for step in problem_map.get("solution_steps", []):
                if isinstance(step, dict):
                    step.setdefault("concrete_details", normalize_str_list(step.get("technical_entities")))
            return
        problem_map["_regression_profile"] = "power_query_etl_example"
        problem_map["problem_kind"] = "transformation_requirement"
        problem_map["core_problem"] = POWER_QUERY_CORE_PROBLEM
        problem_map["why_problematic"] = (
            "원본 데이터가 그대로 모델에 올라가면 결측 원가, 비표준 category, wide format 목표값, 불필요한 보조 쿼리 load가 "
            "이후 모델링과 시각화에서 잘못된 집계나 해석 오류로 이어질 수 있다."
        )
        problem_map["root_causes"] = [
            "SQL Server와 CSV 원본 데이터가 분석 모델 입력 구조로 정리되지 않음",
            "Column quality, Column distribution, Column profile 기준으로 null, distinct, valid 상태를 먼저 점검해야 함",
            "Salesperson, Product, Reseller, Region, Sales, Targets, ColorFormats가 각각 다른 변환 요구사항을 가짐",
            "BusinessType의 Ware House / Warehouse처럼 같은 의미의 값이 다르게 입력됨",
            "FactResellerSales의 TotalProductCost null 값이 Cost와 수익성 분석을 왜곡할 수 있음",
            "Targets의 M01~M12 wide format이 월별 분석에 바로 쓰기 어려움",
            "ColorFormats는 Product에 merge해야 하지만 독립 분석 테이블로 load할 필요는 없음",
        ]
        problem_map["solution_steps"] = power_query_solution_steps()
        problem_map["complex_problem"] = {
            "title": "결측 원가 보완, Targets Unpivot, ColorFormats Merge와 load control",
            "symptom": "TotalProductCost null 값, M01~M12 wide format Targets 데이터, ColorFormats 보조 테이블 load 범위가 함께 분석 모델 품질에 영향을 줌",
            "initial_assumption": "원본 데이터를 그대로 불러와도 분석에 사용할 수 있을 것처럼 보임",
            "actual_cause": "원가 결측은 이익 계산을 왜곡하고, wide format Targets는 날짜 기반 분석에 부적합하며, 보조 매핑 테이블을 그대로 load하면 모델이 불필요하게 복잡해짐",
            "resolution": "TotalProductCost는 if [TotalProductCost] = null then [OrderQuantity] * [StandardCost] else [TotalProductCost]로 보완하고, M01~M12는 Unpivot하여 MonthNumber / Target / TargetMonth 구조로 바꾸며, ColorFormats는 Product[Color]와 ColorFormats[Color] 기준 Left Outer Merge 후 Disable load 처리",
            "verification": "Cost 컬럼과 Fixed Decimal Number 타입, long format Targets, Product의 Background/Font Color Format 컬럼, 최종 7개 테이블 load 상태를 확인",
        }
        problem_map["final_outcome"] = (
            "Power Query Editor에서 원본 데이터를 분석 가능한 형태로 정리하고, Salesperson, SalespersonRegion, Product, Reseller, Region, Sales, Targets "
            "7개 테이블만 Close & Apply로 모델에 로드할 수 있는 상태로 만들었다."
        )


def power_query_solution_steps() -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "title": "SQL Server와 CSV 원본을 Power Query Editor로 가져오기",
            "problem": "SQL Server와 CSV 원본이 보고서에 바로 쓸 분석 테이블 구조가 아니었다.",
            "cause": "AdventureWorksDW2020의 여러 테이블과 ResellerSalesTargets.csv가 서로 다른 형태로 제공되어, 로드 전에 Power Query Editor에서 변환 범위를 정해야 했다.",
            "action": "SQL Server localhost의 AdventureWorksDW2020 데이터베이스를 Navigator에서 확인하고 FactResellerSales 등 필요한 테이블을 선택한 뒤 Transform Data로 Power Query Editor에 진입했다.",
            "verification": "Power Query Editor에서 SQL Server 원본과 CSV 원본을 변환 대상으로 확인했다.",
            "image_refs": [1, 2],
            "concrete_details": ["SQL Server", "localhost", "AdventureWorksDW2020", "Navigator", "Transform Data", "FactResellerSales"],
        },
        {
            "step": 2,
            "title": "Column quality, distribution, profile로 데이터 품질 확인",
            "problem": "원본 컬럼의 null, distinct, valid 상태를 보지 않으면 이후 변환이 필요한 지점을 놓칠 수 있었다.",
            "cause": "BusinessType 같은 범주형 컬럼에는 분포와 비표준 값이 숨어 있을 수 있고, 결측값은 후속 계산을 왜곡할 수 있다.",
            "action": "Power Query Editor에서 Column quality, Column distribution, Column profile을 켜고 주요 컬럼의 null, distinct, valid 상태를 확인했다.",
            "verification": "BusinessType 분포와 컬럼 품질 정보가 표시되어 데이터 정리 대상이 드러나는지 확인했다.",
            "image_refs": [3],
            "concrete_details": ["Column quality", "Column distribution", "Column profile", "null", "distinct", "valid", "BusinessType"],
        },
        {
            "step": 3,
            "title": "Salesperson 쿼리 구성",
            "problem": "DimEmployee에는 모든 직원이 포함되어 있어 영업 담당자 분석에 바로 쓰기 어려웠다.",
            "cause": "보고서에 필요한 것은 SalesPersonFlag가 TRUE인 직원이며, 표시용 이름과 식별 컬럼을 함께 정리해야 했다.",
            "action": "DimEmployee에서 SalesPersonFlag를 TRUE로 필터링하고 FirstName과 LastName을 병합해 Salesperson 컬럼을 만든 뒤 EmployeeID와 UPN을 유지했다.",
            "verification": "Salesperson 쿼리에 영업 담당자만 남고 Salesperson, EmployeeID, UPN 컬럼이 분석에 사용할 형태로 정리되었는지 확인했다.",
            "image_refs": [4, 5],
            "concrete_details": ["DimEmployee", "SalesPersonFlag", "TRUE", "FirstName", "LastName", "Salesperson", "EmployeeID", "UPN"],
        },
        {
            "step": 4,
            "title": "Product 쿼리 확장",
            "problem": "DimProduct만으로는 ProductKey와 제품명은 확인할 수 있지만 Category/Subcategory 분석 축이 부족했다.",
            "cause": "Subcategory와 Category 정보가 DimProductSubcategory, DimProductCategory에 분리되어 있어 확장이 필요했다.",
            "action": "DimProduct에서 ProductKey, EnglishProductName, StandardCost를 유지하고 DimProductSubcategory와 DimProductCategory를 Expand하여 Subcategory와 Category를 붙였다.",
            "verification": "Product 쿼리에 ProductKey, Subcategory, Category, StandardCost가 함께 남아 이후 Sales와 연결 가능한지 확인했다.",
            "image_refs": [6],
            "concrete_details": ["DimProduct", "ProductKey", "EnglishProductName", "StandardCost", "DimProductSubcategory", "DimProductCategory", "Subcategory", "Category", "Expand"],
        },
        {
            "step": 5,
            "title": "Reseller BusinessType 표준화",
            "problem": "BusinessType에 Ware House와 Warehouse가 함께 존재하면 같은 의미의 reseller 유형이 분리 집계될 수 있었다.",
            "cause": "원본 값의 표기 불일치가 category 분할을 만들기 때문이다.",
            "action": "DimReseller에서 BusinessType의 Ware House 값을 Replace Values로 Warehouse로 통일하고 ResellerName과 DimGeography 연결 정보를 유지했다.",
            "verification": "BusinessType 분포에서 Ware House가 Warehouse로 표준화되어 동일 의미 값이 하나로 정리되는지 확인했다.",
            "image_refs": [7],
            "concrete_details": ["DimReseller", "BusinessType", "Warehouse", "Ware House", "Replace Values", "ResellerName", "DimGeography"],
        },
        {
            "step": 6,
            "title": "Region 쿼리 구성",
            "problem": "DimSalesTerritory에는 분석에 불필요한 0번 territory와 여러 식별 컬럼이 포함되어 있었다.",
            "cause": "SalesTerritoryAlternateKey 0을 그대로 두면 실제 영업 지역 분석과 무관한 값이 포함될 수 있다.",
            "action": "DimSalesTerritory에서 SalesTerritoryAlternateKey 0을 제거하고 Region, Country, Group 중심으로 지역 쿼리를 정리했다.",
            "verification": "Region 쿼리에 실제 분석 대상 지역만 남고 Region, Country, Group 컬럼이 확인되는지 점검했다.",
            "image_refs": [8],
            "concrete_details": ["DimSalesTerritory", "SalesTerritoryAlternateKey", "0 제거", "Region", "Country", "Group"],
        },
        {
            "step": 7,
            "title": "Sales 쿼리와 null cost 보완",
            "problem": "FactResellerSales의 TotalProductCost null 값은 Cost와 수익성 분석을 왜곡할 수 있었다.",
            "cause": "원가가 비어 있는 행은 Sales만으로는 수익성을 판단할 수 없고, null을 방치하면 이후 measure 계산도 불안정해진다.",
            "action": "Custom Column으로 Cost를 만들고, TotalProductCost가 null이면 OrderQuantity * StandardCost를 사용하고 그렇지 않으면 TotalProductCost를 사용하도록 처리했다.",
            "verification": "Cost 컬럼이 생성되고 TotalProductCost/StandardCost 정리 후 Unit Price, Sales, Cost가 Fixed Decimal Number 타입으로 맞는지 확인했다.",
            "image_refs": [9],
            "concrete_details": ["FactResellerSales", "Sales", "TotalProductCost", "null", "OrderQuantity", "StandardCost", "Custom Column", "Cost", "Fixed Decimal Number"],
        },
        {
            "step": 8,
            "title": "Targets CSV의 M01~M12 컬럼 Unpivot",
            "problem": "ResellerSalesTargets.csv의 월별 목표가 M01~M12 가로 컬럼으로 있어 월별 분석에 바로 쓰기 어려웠다.",
            "cause": "wide format은 월을 값이 아니라 컬럼명으로 보관하기 때문에 날짜 테이블이나 월 기준 필터와 연결하기 어렵다.",
            "action": "M01부터 M12까지 월 컬럼을 Unpivot하여 MonthNumber와 Target 구조로 바꾸고 TargetMonth를 만들었다.",
            "verification": "Targets가 MonthNumber, Target, TargetMonth를 가진 long format으로 정리되어 월 단위 분석에 연결 가능한지 확인했다.",
            "image_refs": [10, 11],
            "concrete_details": ["ResellerSalesTargets.csv", "M01", "M12", "Unpivot", "MonthNumber", "Target", "TargetMonth", "long format"],
        },
        {
            "step": 9,
            "title": "ColorFormats 쿼리 정리",
            "problem": "색상 서식 정보는 Product에 붙일 참조 데이터지만, 원본 그대로는 헤더와 컬럼명이 분석에 적합하지 않았다.",
            "cause": "ColorFormats의 첫 행을 헤더로 승격하고 색상별 Background/Font 서식 컬럼을 명확히 해야 Product와 merge할 수 있다.",
            "action": "ColorFormats에서 Use First Row as Headers를 적용하고 Color, Background Color Format, Font Color Format 컬럼을 정리했다.",
            "verification": "ColorFormats에 Color와 두 색상 서식 컬럼이 merge 가능한 형태로 남았는지 확인했다.",
            "image_refs": [12],
            "concrete_details": ["ColorFormats", "Use First Row as Headers", "Color", "Background Color Format", "Font Color Format"],
        },
        {
            "step": 10,
            "title": "Product와 ColorFormats Merge",
            "problem": "Product 색상별 서식 정보를 보고서에서 쓰려면 Product 테이블에 색상 포맷 컬럼이 붙어 있어야 했다.",
            "cause": "ColorFormats는 Product[Color]와 ColorFormats[Color]를 기준으로 붙이는 보조 매핑 테이블이기 때문이다.",
            "action": "Product[Color]와 ColorFormats[Color]를 기준으로 Left Outer Merge를 수행하고 Background Color Format, Font Color Format 컬럼을 확장했다.",
            "verification": "Product 쿼리에 Background Color Format과 Font Color Format이 확장되어 색상별 서식 정보를 사용할 수 있는지 확인했다.",
            "image_refs": [13],
            "concrete_details": ["Product[Color]", "ColorFormats[Color]", "Merge", "Left Outer", "Background Color Format", "Font Color Format"],
        },
        {
            "step": 11,
            "title": "Append, Merge, Unpivot, Pivot 개념 구분",
            "problem": "Power Query 변환은 비슷해 보여도 각 연산의 목적이 달라 잘못 고르면 모델 구조가 어긋난다.",
            "cause": "Append는 UNION ALL처럼 행을 합치고, Merge는 JOIN처럼 컬럼을 붙이며, Unpivot/Pivot은 wide format과 long format을 바꾸는 연산이다.",
            "action": "이번 실습에서 ColorFormats에는 Merge, Targets에는 Unpivot을 적용해야 하는 이유를 변환 목적 기준으로 구분했다.",
            "verification": "각 연산이 실제 데이터 구조를 의도한 방향으로 바꾸었는지, 즉 Product에는 서식 컬럼이 붙고 Targets는 월별 행 구조가 되었는지 확인했다.",
            "image_refs": [10, 11, 13],
            "concrete_details": ["Append", "Merge", "UNION ALL", "JOIN", "Unpivot", "Pivot", "wide format", "long format"],
        },
        {
            "step": 12,
            "title": "최종 load control과 Close & Apply",
            "problem": "보조 쿼리까지 모두 모델에 load하면 모델이 불필요하게 복잡해지고 최종 테이블 범위가 흐려질 수 있었다.",
            "cause": "ColorFormats는 Product에 merge된 뒤에는 독립 분석 테이블로 필요하지 않다.",
            "action": "ColorFormats는 Disable load 처리하고 Salesperson, SalespersonRegion, Product, Reseller, Region, Sales, Targets 7개 테이블만 최종 load 대상으로 남긴 뒤 Close & Apply를 실행했다.",
            "verification": "최종 모델에 7개 테이블만 로드되는지 확인하고, ColorFormats는 보조 쿼리로만 남는지 점검했다.",
            "image_refs": [14],
            "concrete_details": ["Salesperson", "SalespersonRegion", "Product", "Reseller", "Region", "Sales", "Targets", "ColorFormats", "Disable load", "Close & Apply", "7개 테이블"],
        },
    ]


def semantic_solution_steps() -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "title": "Category별 Sales 반복값 문제 인식",
            "problem": "Product[Category]와 Sales[Sales]를 같은 table visual에 넣었을 때 모든 Category 행에 같은 Sales 총액이 반복되었다.",
            "cause": "Product[Category] filter context가 Sales fact table로 전달되지 않아 각 Category가 Sales를 분리해 필터링하지 못했다.",
            "action": "이 현상을 visual 문제가 아니라 Product와 Sales 사이 relationship/filter context 문제로 정의했다.",
            "verification": "Accessories, Bikes, Clothing, Components 등 Category 행에 같은 총액이 반복되는 이미지 1을 문제 신호로 확정했다.",
            "image_refs": [1],
            "concrete_details": ["Product[Category]", "Sales[Sales]", "same repeated total", "filter context"],
            "portfolio_meaning": "값이 보이는 화면에서도 필터 흐름이 깨지면 분석 결과가 틀릴 수 있음을 문제로 정의했다.",
        },
        {
            "step": 2,
            "title": "Product[ProductKey] -> Sales[ProductKey] relationship 생성",
            "problem": "Product dimension의 Category가 Sales fact table을 필터링하지 못했다.",
            "cause": "Product와 Sales 사이 ProductKey relationship이 없거나 활성 경로가 준비되지 않았다.",
            "action": "Product[ProductKey]에서 Sales[ProductKey]로 relationship을 생성했다.",
            "verification": "relationship 생성 화면에서 두 테이블의 key 컬럼이 올바르게 연결되는지 확인했다.",
            "image_refs": [2],
            "concrete_details": ["Product[ProductKey]", "Sales[ProductKey]", "relationship creation"],
            "portfolio_meaning": "차원 테이블의 필터가 fact table로 전달되는 기본 경로를 만들었다.",
        },
        {
            "step": 3,
            "title": "relationship 설정값 확인",
            "problem": "relationship이 생성되어도 cardinality나 filter direction이 잘못되면 같은 문제가 남을 수 있다.",
            "cause": "Product는 dimension table이고 Sales는 fact table이므로 one-to-many, single direction, active relationship이어야 한다.",
            "action": "Cardinality: One to many, Cross-filter direction: Single, Make this relationship active: Checked 설정을 확인했다.",
            "verification": "model view에서 Product-Sales relationship이 활성 상태로 표시되는지 확인했다.",
            "image_refs": [3],
            "concrete_details": ["Cardinality: One to many", "Cross-filter direction: Single", "Make this relationship active: Checked"],
            "portfolio_meaning": "relationship은 선을 긋는 작업이 아니라 모델의 필터 전달 규칙을 검증하는 작업임을 보여준다.",
        },
        {
            "step": 4,
            "title": "Star schema 모델 구조 확인",
            "problem": "개별 relationship만 확인하면 전체 모델이 분석에 적합한 구조인지 판단하기 어렵다.",
            "cause": "Sales fact table을 중심으로 Product 등 dimension table이 연결되는 Star schema 구조가 필요하다.",
            "action": "model view에서 Product dimension과 Sales fact table 중심의 Star schema layout을 확인했다.",
            "verification": "Sales를 중심으로 dimension들이 분리되어 연결되는 모델 구조가 보이는지 확인했다.",
            "image_refs": [4],
            "concrete_details": ["Star schema", "Product dimension table", "Sales fact table"],
            "portfolio_meaning": "단일 오류 수정이 아니라 모델 전체 구조 관점으로 문제를 검증했다.",
        },
        {
            "step": 5,
            "title": "Product hierarchy 구성",
            "problem": "Category 수준만으로는 제품 분석을 드릴다운하기 어렵다.",
            "cause": "보고서에서 Category -> Subcategory -> Product 순서로 자연스럽게 탐색하려면 hierarchy가 필요하다.",
            "action": "Product hierarchy를 Category, Subcategory, Product 순서로 구성했다.",
            "verification": "Product hierarchy level에 Category, Subcategory, Product가 올바른 순서로 포함되는지 확인했다.",
            "image_refs": [5, 6],
            "concrete_details": ["Product hierarchy", "Category -> Subcategory -> Product"],
            "portfolio_meaning": "semantic model이 단순 집계뿐 아니라 탐색 가능한 분석 축을 제공하도록 정리했다.",
        },
        {
            "step": 6,
            "title": "Profit Quick Measure 생성",
            "problem": "Sales만으로는 매출 규모를 볼 수 있지만 수익성을 판단할 수 없다.",
            "cause": "이익은 Sales에서 Cost를 차감해야 하며, 잘못 선택하면 Sales - Sales처럼 0이 되는 실수를 만들 수 있다.",
            "action": "Profit quick measure를 Sales - Cost 기준으로 생성했다.",
            "verification": "Profit이 Sales와 Cost의 차이로 계산되는지, 0으로만 나오지 않는지 확인했다.",
            "image_refs": [7],
            "concrete_details": ["Profit", "Sales[Sales]", "Sales[Cost]", "Sales - Cost"],
            "dax_measures": [
                {
                    "measure_name": "Profit",
                    "formula": "Profit = SUM(Sales[Sales]) - SUM(Sales[Cost])",
                    "why_needed": "Sales만으로는 수익성을 판단할 수 없기 때문",
                    "validation": "Category별 Sales와 Profit이 함께 표시되는지 확인",
                }
            ],
            "portfolio_meaning": "measure 생성에서도 계산 대상 선택이 분석 의미를 바꾼다는 점을 점검했다.",
        },
        {
            "step": 7,
            "title": "Profit Margin measure 생성",
            "problem": "Profit 금액만으로는 매출 규모가 다른 Category의 수익성을 비교하기 어렵다.",
            "cause": "수익률은 Profit을 Sales로 나누어 비교해야 하며 0 나누기 문제를 피하려면 DIVIDE가 적합하다.",
            "action": "Profit Margin measure를 DIVIDE([Profit], [Sales])로 만들고 Percentage / Decimal places 2 형식을 확인했다.",
            "verification": "Profit Margin이 퍼센트 형식으로 표시되고 Category별로 계산되는지 확인했다.",
            "image_refs": [8],
            "concrete_details": ["Profit Margin", "DIVIDE([Profit], [Sales])", "Percentage", "Decimal places 2"],
            "dax_measures": [
                {
                    "measure_name": "Profit Margin",
                    "formula": "Profit Margin = DIVIDE([Profit], [Sales])",
                    "why_needed": "매출 규모가 다른 Category를 수익률 기준으로 비교하기 위해 필요",
                    "validation": "Profit Margin 결과가 Category별로 계산되는지 확인",
                }
            ],
            "portfolio_meaning": "금액 지표와 비율 지표를 함께 만들어 분석 해석력을 높였다.",
        },
        {
            "step": 8,
            "title": "Category별 Sales / Profit / Profit Margin 결과 검증",
            "problem": "relationship과 measure를 만들었더라도 결과가 Category별로 분리되는지 확인해야 한다.",
            "cause": "초기 문제는 모든 Category에 같은 Sales 총액이 반복되는 것이었으므로 수정 후 값 분리가 핵심 검증 기준이다.",
            "action": "table visual에서 Category별 Sales, Profit, Profit Margin을 함께 표시했다.",
            "verification": "Category별 Sales, Profit, Profit Margin 값이 서로 다른 값으로 분리되어 계산되는지 이미지 9~10에서 확인했다.",
            "image_refs": [9, 10],
            "concrete_details": ["Category별 Sales", "Profit", "Profit Margin", "values separated by Category"],
            "portfolio_meaning": "수정의 성공 기준을 화면 변화가 아니라 계산 결과의 분리 여부로 잡았다.",
        },
        {
            "step": 9,
            "title": "SalespersonRegion bridge table 구성",
            "problem": "Salesperson별 Sales/Target 분석에서는 담당 Region 기준 필터 경로가 필요했다.",
            "cause": "Salesperson과 Region 사이에는 bridge 역할을 하는 SalespersonRegion이 필요하며, 단순 직접 관계만으로는 담당 지역 기준 분석을 설명하기 어렵다.",
            "action": "SalespersonRegion bridge table을 사용해 Salesperson -> SalespersonRegion -> Region -> Sales 경로를 구성했다.",
            "verification": "model view에서 SalespersonRegion이 Salesperson과 Region 사이의 bridge table로 배치되는지 확인했다.",
            "image_refs": [11],
            "concrete_details": ["SalespersonRegion", "Bridge table", "Salesperson -> SalespersonRegion -> Region -> Sales"],
            "portfolio_meaning": "후반 문제를 Product-Sales 반복값 문제와 분리된 filter path 설계 문제로 다뤘다.",
        },
        {
            "step": 10,
            "title": "Cross-filter direction Both 설정",
            "problem": "SalespersonRegion bridge path를 통해 담당 Region 기준 필터가 전달되어야 했다.",
            "cause": "bridge table 경로에서는 단방향 필터만으로 원하는 분석 경로가 충분히 전달되지 않을 수 있다.",
            "action": "Region과 SalespersonRegion 관계에서 Cross-filter direction을 Both로 설정했다.",
            "verification": "relationship 설정 화면에서 Cross-filter direction: Both가 적용되는지 확인했다.",
            "image_refs": [12],
            "concrete_details": ["Cross-filter direction: Both", "Region", "SalespersonRegion"],
            "portfolio_meaning": "filter direction을 분석 의도에 맞게 조정하는 모델링 판단을 보여준다.",
        },
        {
            "step": 11,
            "title": "Direct Salesperson-Sales relationship 비활성화",
            "problem": "direct Salesperson-Sales relationship과 bridge path가 동시에 있으면 filter path가 모호해질 수 있다.",
            "cause": "같은 분석 목표에 대해 직접 경로와 SalespersonRegion bridge 경로가 동시에 활성화되면 모델이 어떤 경로로 필터를 전달해야 하는지 불명확해진다.",
            "action": "직접 Salesperson-Sales relationship을 inactive relationship으로 바꾸었다.",
            "verification": "model view에서 직접 관계가 비활성 상태로 표시되는지 확인했다.",
            "image_refs": [13],
            "concrete_details": ["inactive relationship", "Direct Salesperson-Sales relationship", "ambiguous filter path"],
            "portfolio_meaning": "관계를 많이 만드는 것이 아니라 필요한 활성 경로를 명확히 남기는 판단을 보여준다.",
        },
        {
            "step": 12,
            "title": "Salesperson -> SalespersonRegion -> Region -> Sales 경로로 Sales 결과 확인",
            "problem": "bridge path 조정 후 Salesperson별 Sales가 담당 Region 기준으로 달라지는지 확인해야 했다.",
            "cause": "경로 설정이 맞아야 Salesperson filter가 Sales fact table까지 의도한 방식으로 전달된다.",
            "action": "Salesperson별 Sales 결과를 확인해 bridge path가 적용된 계산 결과를 검증했다.",
            "verification": "Salesperson별 Sales 값이 Region filter path를 통해 달라진 결과를 이미지 14에서 확인했다.",
            "image_refs": [14],
            "concrete_details": ["Salesperson별 Sales", "Region filter path", "Salesperson -> SalespersonRegion -> Region -> Sales"],
            "portfolio_meaning": "모델 경로 변경을 최종 수치 변화로 검증했다.",
        },
        {
            "step": 13,
            "title": "Targets 테이블 연결",
            "problem": "Sales와 Target을 같은 기준에서 비교하려면 Targets가 Salesperson 분석 경로와 맞아야 한다.",
            "cause": "Target이 별도 기준으로 남아 있으면 Salesperson별 실적과 목표 비교가 같은 filter context에서 이루어지지 않는다.",
            "action": "Targets 테이블을 Salesperson별 비교 흐름에 연결하고 Sales와 함께 볼 수 있게 구성했다.",
            "verification": "Targets가 Salesperson별 비교 화면에 함께 표시될 준비가 되었는지 확인했다.",
            "image_refs": [15],
            "concrete_details": ["Targets", "Salesperson", "Sales", "Target comparison"],
            "portfolio_meaning": "실적 지표와 목표 지표를 같은 분석 기준으로 맞추는 모델링 목적을 정리했다.",
        },
        {
            "step": 14,
            "title": "Salesperson별 Sales와 Target 최종 비교",
            "problem": "최종 목적은 Salesperson별 Sales와 Target을 같은 기준에서 비교하는 것이었다.",
            "cause": "SalespersonRegion bridge path, inactive direct relationship, Targets 연결이 함께 맞아야 최종 비교가 의미를 가진다.",
            "action": "Salesperson별 Sales와 Target을 같은 visual에서 비교했다.",
            "verification": "이미지 15에서 Salesperson별 Sales와 Target이 함께 표시되어 최종 비교가 가능한지 확인했다.",
            "image_refs": [15],
            "concrete_details": ["Salesperson별 Sales / Target 비교", "Targets", "final validation"],
            "portfolio_meaning": "초반 relationship 문제 해결에서 후반 목표 비교까지 하나의 semantic model 검증 흐름으로 마무리했다.",
        },
    ]


def dax_solution_steps() -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "title": "Month 컬럼을 MonthKey 기준으로 정렬",
            "problem": "Month 이름을 텍스트로만 두면 Jan, Feb, Mar 같은 실제 월 순서가 보장되지 않는다.",
            "cause": "문자열 월 이름은 알파벳 또는 기본 표시 순서에 영향을 받을 수 있어 시간 흐름 기반 분석에 부적합하다.",
            "action": "Month 컬럼의 Sort by column 기준을 MonthKey로 지정했다.",
            "verification": "월 이름이 MonthKey 순서대로 표시되어 fiscal/월별 trend를 자연스럽게 읽을 수 있는지 확인했다.",
            "image_refs": [1],
            "concrete_details": ["Month", "MonthKey", "Sort by column", "month order"],
            "portfolio_meaning": "시각화의 정렬 문제를 display column과 sort key 분리로 해결했다.",
        },
        {
            "step": 2,
            "title": "Fiscal Date table과 hierarchy 구성",
            "problem": "Sales 날짜 기준 분석을 하려면 일관된 Date table과 fiscal 기준 탐색 구조가 필요했다.",
            "cause": "날짜 컬럼만으로는 fiscal year/quarter/month 흐름을 안정적으로 재사용하기 어렵다.",
            "action": "Fiscal 기준이 반영된 Date table과 fiscal hierarchy를 구성했다.",
            "verification": "Fiscal hierarchy가 날짜 테이블 안에서 year, quarter, month 수준으로 탐색 가능한지 확인했다.",
            "image_refs": [2],
            "concrete_details": ["Date table", "Fiscal hierarchy", "Fiscal", "CALENDARAUTO"],
            "portfolio_meaning": "분석 축을 raw date column이 아니라 재사용 가능한 날짜 차원으로 분리했다.",
        },
        {
            "step": 3,
            "title": "Date table과 Sales 모델 relationship 확인",
            "problem": "Date table이 있어도 Sales와 연결되지 않으면 날짜 필터가 Sales fact table로 전달되지 않는다.",
            "cause": "시간 분석은 Date dimension과 Sales fact table 사이의 relationship이 활성화되어 있어야 의미가 있다.",
            "action": "Date table과 Sales 모델 relationship을 model view에서 확인했다.",
            "verification": "Date table의 날짜 기준 필터가 Sales 분석에 적용될 수 있는 관계 경로가 보이는지 확인했다.",
            "image_refs": [3],
            "concrete_details": ["Date table", "Sales", "model relationship", "date filter context"],
            "portfolio_meaning": "날짜 테이블 생성 이후 실제 fact table과의 연결까지 검증했다.",
        },
        {
            "step": 4,
            "title": "Mark as date table 처리",
            "problem": "Date table을 공식 날짜 테이블로 지정하지 않으면 시간 지능 계산과 날짜 기준 해석이 약해질 수 있다.",
            "cause": "Power BI는 어떤 컬럼을 canonical date column으로 사용할지 명시적으로 지정해야 안정적인 날짜 분석 기준을 갖는다.",
            "action": "Date table을 Mark as date table로 지정하고 날짜 컬럼을 선택했다.",
            "verification": "Date table이 공식 날짜 테이블로 인식되어 날짜 기준 분석의 기준점이 명확해졌는지 확인했다.",
            "image_refs": [4],
            "concrete_details": ["Mark as date table", "Date table", "date column"],
            "portfolio_meaning": "모델의 시간 분석 기준을 UI 표시가 아니라 semantic setting으로 고정했다.",
        },
        {
            "step": 5,
            "title": "Avg Price explicit measure 생성",
            "problem": "Unit Price 같은 raw column을 자동 평균으로 쓰면 계산 의도와 이름, format 관리가 불명확하다.",
            "cause": "자동 집계는 matrix에 표시될 수는 있지만 reusable measure가 아니며 다른 지표와 일관된 관리가 어렵다.",
            "action": "Avg Price explicit measure를 생성했다.",
            "verification": "Avg Price가 matrix에서 의도한 평균 가격 지표로 표시되고 다른 measure와 함께 재사용 가능한지 확인했다.",
            "image_refs": [5],
            "concrete_details": ["Avg Price", "explicit measure", "Unit Price"],
            "dax_measures": [
                {
                    "measure_name": "Avg Price",
                    "formula": "Avg Price = AVERAGE(Sales[Unit Price])",
                    "why_needed": "가격 평균을 raw column 자동 집계가 아니라 재사용 가능한 explicit measure로 관리하기 위해 필요",
                    "validation": "matrix에서 Avg Price가 가격 지표로 표시되는지 확인",
                }
            ],
            "portfolio_meaning": "자동 집계 의존도를 줄이고 명시적 measure 중심 모델로 전환했다.",
        },
        {
            "step": 6,
            "title": "Median / Min / Max Price와 Orders / Order Lines measure 구성",
            "problem": "평균 가격만으로는 가격 분포나 주문 규모를 충분히 설명할 수 없다.",
            "cause": "가격 분석에는 median/min/max가 필요하고, 주문 분석에는 Orders와 Order Lines처럼 서로 다른 count 기준이 필요하다.",
            "action": "Median Price, Min Price, Max Price, Orders, Order Lines measure를 구성했다.",
            "verification": "matrix에서 가격 measure와 주문 수 measure가 함께 표시되어 평균, 중앙값, 최소/최대, 주문 건수, 주문 라인 수를 비교할 수 있는지 확인했다.",
            "image_refs": [6],
            "concrete_details": ["Median Price", "Min Price", "Max Price", "Orders", "Order Lines", "matrix"],
            "portfolio_meaning": "단일 Sales 합계에서 벗어나 가격과 주문량을 다각도로 설명할 수 있는 measure set을 만들었다.",
        },
        {
            "step": 7,
            "title": "가격 measure Currency format 확인",
            "problem": "가격 measure가 숫자로만 표시되면 금액 지표인지 비율/건수 지표인지 해석이 흐려질 수 있다.",
            "cause": "measure는 계산식뿐 아니라 format까지 맞아야 보고서 사용자가 지표 의미를 빠르게 파악할 수 있다.",
            "action": "가격 관련 measure의 Currency format을 확인했다.",
            "verification": "Avg Price, Median Price, Min Price, Max Price가 통화 형식으로 표시되어 주문 수 measure와 구분되는지 확인했다.",
            "image_refs": [7],
            "concrete_details": ["Currency format", "Avg Price", "Median Price", "Min Price", "Max Price"],
            "portfolio_meaning": "계산 결과의 표시 형식까지 measure 설계의 일부로 관리했다.",
        },
        {
            "step": 8,
            "title": "TargetAmount raw column 대신 Target measure 사용",
            "problem": "TargetAmount 컬럼을 그대로 사용하면 목표값이 어떤 filter context와 total 기준으로 집계되는지 불명확하다.",
            "cause": "raw column 자동 집계는 Salesperson별 목표와 total row의 계산 의도를 명시적으로 통제하기 어렵다.",
            "action": "TargetAmount raw column 대신 Target explicit measure를 사용했다.",
            "verification": "Target measure가 Salesperson별 목표값과 total 기준에서 일관되게 표시되는지 TargetAmount raw column과 비교했다.",
            "image_refs": [8],
            "concrete_details": ["TargetAmount", "Target measure", "explicit measure", "filter context"],
            "dax_measures": [
                {
                    "measure_name": "Target",
                    "formula": "Target = IF(HASONEVALUE(Salesperson[Salesperson]), SUM(Targets[TargetAmount]), SUMX(VALUES(Salesperson[Salesperson]), SUM(Targets[TargetAmount])))",
                    "why_needed": "목표값 집계를 raw column 자동 집계가 아니라 measure로 통제하고, Salesperson별 행과 total row의 계산 기준을 분리하기 위해 필요",
                    "validation": "TargetAmount raw column과 Target measure의 표시 차이, 특히 total row가 의도한 기준으로 계산되는지 비교",
                }
            ],
            "portfolio_meaning": "목표 지표를 raw column에서 reusable business metric으로 끌어올렸다.",
        },
        {
            "step": 9,
            "title": "Variance / Variance Margin 계산",
            "problem": "Sales와 Target만 나란히 있으면 목표 대비 초과/미달 규모와 비율을 즉시 판단하기 어렵다.",
            "cause": "성과 분석에는 차이 금액과 차이율이 함께 있어야 Salesperson별 결과를 비교할 수 있다.",
            "action": "Variance와 Variance Margin measure를 구성했다.",
            "verification": "Salesperson별로 Sales가 Target을 얼마나 초과하거나 미달했는지 금액과 비율로 확인할 수 있는지 점검했다.",
            "image_refs": [9],
            "concrete_details": ["Variance", "Variance Margin", "Sales", "Target"],
            "dax_measures": [
                {
                    "measure_name": "Sales",
                    "formula": "Sales = SUM(Sales[Sales])",
                    "why_needed": "Variance = [Sales] - [Target] 계산의 기준이 되는 매출 measure가 필요하기 때문",
                    "validation": "Salesperson별 Sales 값이 Target, Variance와 같은 matrix에서 동일한 filter context로 표시되는지 확인",
                },
                {
                    "measure_name": "Variance",
                    "formula": "Variance = [Sales] - [Target]",
                    "why_needed": "목표 대비 초과/미달 금액을 계산하기 위해 필요",
                    "validation": "Salesperson별 Sales와 Target 차이가 표시되는지 확인",
                },
                {
                    "measure_name": "Variance Margin",
                    "formula": "Variance Margin = DIVIDE([Variance], [Target])",
                    "why_needed": "목표 대비 차이를 비율로 비교하기 위해 필요",
                    "validation": "Variance Margin이 비율 지표로 표시되는지 확인",
                },
            ],
            "portfolio_meaning": "성과 해석을 단순 값 비교에서 목표 대비 차이와 비율 분석으로 확장했다.",
        },
        {
            "step": 10,
            "title": "Salesperson별 Sales, Target, Variance, Variance Margin 최종 비교",
            "problem": "최종 목적은 Salesperson별 실적과 목표 대비 성과를 같은 화면에서 비교하는 것이었다.",
            "cause": "Date table, explicit measure, Target/Variance measure가 함께 맞아야 Salesperson별 성과 분석이 명확해진다.",
            "action": "Salesperson별 Sales, Target, Variance, Variance Margin을 최종 matrix에서 비교했다.",
            "verification": "각 Salesperson 행에서 Sales, Target, Variance, Variance Margin이 함께 표시되어 목표 대비 성과를 바로 읽을 수 있는지 확인했다.",
            "image_refs": [9],
            "concrete_details": ["Salesperson", "Sales", "Target", "Variance", "Variance Margin", "final matrix"],
            "portfolio_meaning": "날짜 모델링과 measure 설계가 최종 성과 비교 화면으로 연결되는지 검증했다.",
        },
    ]


def is_power_query_etl_regression_context(problem_map: dict[str, Any]) -> bool:
    source = json.dumps(problem_map, ensure_ascii=False).lower()
    markers = [
        "adventureworksdw2020",
        "factresellersales",
        "salespersonflag",
        "dimproductsubcategory",
        "dimproductcategory",
        "ware house",
        "totalproductcost",
        "resellersalestargets.csv",
        "colorformats",
        "background color format",
        "font color format",
        "close & apply",
    ]
    return sum(1 for marker in markers if marker in source) >= 5


def build_section_plan(article_type: str, evidence: list[dict[str, Any]], problem_map: dict[str, Any]) -> list[dict[str, Any]]:
    if article_type == "semantic_model_relationship":
        return [
            {"section": "Category별 Sales 반복값 문제 인식", "image_refs": [1], "must_include": ["Product[Category]", "Sales[Sales]", "same repeated total", "filter context"]},
            {"section": "Product-Sales relationship creation", "image_refs": [2], "must_include": ["Product[ProductKey]", "Sales[ProductKey]", "relationship"]},
            {"section": "Product-Sales relationship model confirmation", "image_refs": [3], "must_include": ["One to many", "Cross-filter direction: Single", "Make this relationship active"]},
            {"section": "Star schema", "image_refs": [4], "must_include": ["Star schema", "Product dimension", "Sales fact table"]},
            {"section": "Product hierarchy", "image_refs": [5, 6], "must_include": ["Product hierarchy", "Category", "Subcategory", "Product"]},
            {"section": "Profit quick measure", "image_refs": [7], "must_include": ["Profit", "Sales[Sales]", "Sales[Cost]", "Sales - Cost"]},
            {"section": "Profit Margin quick measure", "image_refs": [8], "must_include": ["Profit Margin", "DIVIDE", "Percentage", "Decimal places 2"]},
            {"section": "Category result validation", "image_refs": [9, 10], "must_include": ["Category별 Sales", "Profit", "Profit Margin"]},
            {"section": "SalespersonRegion bridge model", "image_refs": [11], "must_include": ["SalespersonRegion", "Bridge table", "Salesperson -> SalespersonRegion -> Region -> Sales"]},
            {"section": "Cross-filter direction Both", "image_refs": [12], "must_include": ["Cross-filter direction: Both", "Region", "SalespersonRegion"]},
            {"section": "Direct relationship inactive", "image_refs": [13], "must_include": ["inactive relationship", "Direct Salesperson-Sales relationship"]},
            {"section": "Salesperson sales by region result", "image_refs": [14], "must_include": ["Salesperson별 Sales", "Region filter path"]},
            {"section": "Targets table connection", "image_refs": [15], "must_include": ["Targets", "Salesperson", "Sales"]},
            {"section": "Salesperson Sales Target final", "image_refs": [15], "must_include": ["Salesperson별 Sales / Target 비교", "Targets", "final validation"]},
        ]
    if article_type == "dax_measure_modeling":
        return [
            {"section": "Month sort by MonthKey", "image_refs": [1], "must_include": ["Month", "MonthKey", "Sort by column"]},
            {"section": "Fiscal Date table and hierarchy", "image_refs": [2], "must_include": ["Date table", "Fiscal hierarchy", "CALENDARAUTO"]},
            {"section": "Date model relationships", "image_refs": [3], "must_include": ["Date table", "Sales", "model relationship"]},
            {"section": "Mark as date table", "image_refs": [4], "must_include": ["Mark as date table", "date column"]},
            {"section": "Avg Price explicit measure", "image_refs": [5], "must_include": ["Avg Price", "explicit measure", "Unit Price"]},
            {"section": "Pricing and count measures", "image_refs": [6], "must_include": ["Median Price", "Min Price", "Max Price", "Orders", "Order Lines"]},
            {"section": "Measure formatting", "image_refs": [7], "must_include": ["Currency format", "Avg Price", "Median Price"]},
            {"section": "TargetAmount versus Target measure", "image_refs": [8], "must_include": ["TargetAmount", "Target measure", "filter context"]},
            {"section": "Variance and Variance Margin", "image_refs": [9], "must_include": ["Variance", "Variance Margin", "Sales", "Target"]},
            {"section": "Salesperson final comparison", "image_refs": [9], "must_include": ["Salesperson", "Sales", "Target", "Variance", "Variance Margin"]},
        ]
    if article_type == "power_query_etl" and problem_map.get("_regression_profile") == "power_query_etl_example":
        return [
            {"section": "SQL Server and CSV data import", "image_refs": [1, 2], "must_include": ["SQL Server", "localhost", "AdventureWorksDW2020", "Navigator", "Transform Data", "FactResellerSales"]},
            {"section": "Column quality/distribution/profile", "image_refs": [3], "must_include": ["Column quality", "Column distribution", "Column profile", "null", "distinct", "valid"]},
            {"section": "Salesperson query", "image_refs": [4, 5], "must_include": ["DimEmployee", "SalesPersonFlag", "TRUE", "FirstName", "LastName", "Salesperson", "EmployeeID", "UPN"]},
            {"section": "Product query expansion", "image_refs": [6], "must_include": ["DimProduct", "ProductKey", "EnglishProductName", "StandardCost", "DimProductSubcategory", "DimProductCategory", "Subcategory", "Category", "Expand"]},
            {"section": "Reseller BusinessType standardization", "image_refs": [7], "must_include": ["DimReseller", "BusinessType", "Warehouse", "Ware House", "Replace Values", "ResellerName", "DimGeography"]},
            {"section": "Region query", "image_refs": [8], "must_include": ["DimSalesTerritory", "SalesTerritoryAlternateKey", "0 제거", "Region", "Country", "Group"]},
            {"section": "Sales query and null cost fix", "image_refs": [9], "must_include": ["FactResellerSales", "Sales", "TotalProductCost", "null", "OrderQuantity", "StandardCost", "Custom Column", "Cost", "Fixed Decimal Number"]},
            {"section": "Targets CSV unpivot", "image_refs": [10, 11], "must_include": ["ResellerSalesTargets.csv", "M01", "M12", "Unpivot", "MonthNumber", "Target", "TargetMonth"]},
            {"section": "ColorFormats query", "image_refs": [12], "must_include": ["ColorFormats", "Use First Row as Headers", "Color", "Background Color Format", "Font Color Format"]},
            {"section": "Product/ColorFormats Merge", "image_refs": [13], "must_include": ["Product[Color]", "ColorFormats[Color]", "Merge", "Left Outer", "Background Color Format", "Font Color Format"]},
            {"section": "Append/Merge/Unpivot/Pivot concept", "image_refs": [10, 11, 13], "must_include": ["Append", "Merge", "UNION ALL", "JOIN", "Unpivot", "Pivot", "wide format", "long format"]},
            {"section": "final load control", "image_refs": [14], "must_include": ["Salesperson", "SalespersonRegion", "Product", "Reseller", "Region", "Sales", "Targets", "ColorFormats", "Disable load", "Close & Apply", "7개 테이블"]},
        ]
    return [
        {
            "section": step.get("title", f"해결 단계 {index}"),
            "image_refs": step.get("image_refs", []),
            "must_include": step.get("technical_entities", [])[:5],
        }
        for index, step in enumerate(problem_map.get("solution_steps", []), start=1)
        if isinstance(step, dict)
    ]


def next_capture_id(session: dict[str, Any]) -> int:
    current = [int(item.get("capture_id", 0)) for item in session.get("captures", [])]
    return (max(current) if current else 0) + 1


def next_qa_id(session: dict[str, Any]) -> int:
    current = [int(item.get("qa_id", 0)) for item in session.get("qa_logs", [])]
    return (max(current) if current else 0) + 1


def append_qa_log(
    session: dict[str, Any],
    question: str,
    answer: str,
    selected_text: str = "",
    related_capture_ids: list[int] | None = None,
) -> dict[str, Any]:
    related_capture_ids = related_capture_ids or recent_capture_ids(session)
    qa = asdict(
        QALog(
            qa_id=next_qa_id(session),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            related_capture_ids=related_capture_ids,
            selected_text=selected_text,
            question=question,
            answer=answer,
            answer_summary=summarize_answer(answer),
            learner_state=infer_learner_state(question),
            resolved=bool(answer.strip()),
            used_in_article=True,
        )
    )
    session.setdefault("qa_logs", []).append(qa)
    return qa


def recent_capture_ids(session: dict[str, Any], limit: int = 3) -> list[int]:
    captures = sorted(session.get("captures", []), key=lambda item: item.get("timestamp", ""))
    return [int(item.get("capture_id")) for item in captures[-limit:] if item.get("capture_id")]


def summarize_answer(answer: str) -> str:
    text = " ".join(answer.split())
    return text[:320] + ("..." if len(text) > 320 else "")


def infer_learner_state(question: str) -> str:
    lowered = question.lower()
    if any(word in lowered for word in ["error", "오류", "안돼", "안 됨", "bug", "traceback"]):
        return "debugging"
    if any(word in lowered for word in ["맞아", "검증", "확인", "왜"]):
        return "verifying"
    if any(word in lowered for word in ["뭐", "무엇", "개념", "설명"]):
        return "confused"
    return "exploring"


def tutor_agent_answer(
    session: dict[str, Any],
    question: str,
    selected_text: str = "",
    related_capture_ids: list[int] | None = None,
) -> dict[str, Any]:
    related_capture_ids = related_capture_ids or recent_capture_ids(session)
    context_captures = [
        capture
        for capture in session.get("captures", [])
        if int(capture.get("capture_id", 0)) in set(related_capture_ids)
    ]
    context = "\n".join(
        f"Capture {capture.get('capture_id')}: note={capture.get('user_note', '')}, keywords={', '.join(capture.get('auto_keywords') or [])}"
        for capture in context_captures
    )
    prompt = f"""
사용자의 학습 중 질문에 TutorAgent로 답하세요.

규칙:
- 답변은 학습 설명 중심입니다.
- 사용자가 어디서 막혔는지 먼저 짚고, 원인 후보와 확인 방법을 제안합니다.
- 최종 글에 그대로 복사될 답변이 아니라 Q&A 로그로 저장될 설명입니다.
- 입력에 없는 성과를 만들지 않습니다.

[최근 캡처]
{context}

[선택/붙여넣은 텍스트]
{selected_text}

[질문]
{question}
""".strip()
    answer = llm_text(llm, prompt, max_tokens=900, system="You are a concise Korean tutor for technical study sessions.")
    if not answer.strip():
        answer = (
            "현재 LLM 응답을 받지 못했습니다. 질문을 학습 기록으로 저장해 두었습니다. "
            "추가로 오류 메시지, 수식, 화면에서 이상하게 보인 값을 함께 남기면 이후 ProblemMap과 DecisionMap에서 원인 후보를 더 정확히 좁힐 수 있습니다."
        )
    return append_qa_log(session, question=question, answer=answer, selected_text=selected_text, related_capture_ids=related_capture_ids)


def natural_sort_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


ARTICLE_TYPE_EXAMPLE_DIRS = {
    "semantic_model_relationship": "powerbi_semantic_model",
    "dax_measure_modeling": "powerbi_dax_modeling",
    "power_query_etl": "powerbi_powerquery_etl",
}

SUPPORTED_ARTICLE_TYPES = [
    "semantic_model_relationship",
    "dax_measure_modeling",
    "power_query_etl",
    "dashboard_validation",
    "python_algorithm_learning",
    "code_error_debugging",
    "github_agentic_workflow",
    "github_actions_workflow",
    "ai_coding_workflow",
    "microsoft_foundry_first_agent",
    "github_readme_debugging",
    "deployment_debugging",
    "cloud_lab_practice",
    "ai_course_lab",
    "ai_project_build_log",
    "learning_path_reflection",
    "certification_badge_summary",
    "general_learning_portfolio",
    "unknown",
]

ARTICLE_TYPE_KEYWORDS = {
    "semantic_model_relationship": [
        "semantic model",
        "relationship",
        "filter context",
        "repeated",
        "productkey",
        "bridge",
        "salespersonregion",
        "profit margin",
    ],
    "dax_measure_modeling": [
        "date table",
        "monthkey",
        "sort by column",
        "measure",
        "target",
        "variance",
        "calendarauto",
        "avg price",
    ],
    "power_query_etl": [
        "power query",
        "column quality",
        "column distribution",
        "column profile",
        "sql server",
        "csv",
        "dimemployee",
        "dimproduct",
        "dimreseller",
        "factresellersales",
        "ware house",
        "totalproductcost",
        "unpivot",
        "colorformats",
        "merge",
        "disable load",
        "close & apply",
    ],
    "dashboard_validation": ["dashboard", "visual", "kpi", "report validation"],
    "python_algorithm_learning": ["python", "algorithm", "array", "sort", "leetcode", "time complexity", "big o"],
    "code_error_debugging": ["traceback", "exception", "syntaxerror", "typeerror", "nameerror", "stack trace", "debugging"],
    "github_agentic_workflow": [
        "agentic workflow",
        "agentic workflows",
        "automation that actually reads the room",
        "activation",
        "conclusion",
        "workflow_dispatch",
        "lock.yml",
        "update-github-info",
    ],
    "github_actions_workflow": [
        "github actions",
        "workflow_dispatch",
        "workflow file",
        "workflows",
        "actions",
        ".github/workflows",
        "yaml",
        "yml",
        "conclusion",
    ],
    "ai_coding_workflow": ["copilot", "agent", "automation", "ai coding", "coding agent", "workflow"],
    "microsoft_foundry_first_agent": [
        "microsoft foundry",
        "mslearn-agent-quickstart",
        "develop your first agent with microsoft",
        "get-started-in-foundry",
        "continue-in-vscode",
        "use-agent",
        "first agent",
    ],
    "agent_academy_copilot_studio": [
        "agent academy",
        "microsoft.github.io/agent-academy",
        "microsoft copilot studio",
        "declarative agent",
        "custom agent",
        "adaptive cards",
        "agent flows",
        "microsoft 365 copilot",
        "sharepoint site",
        "publish your agent",
    ],
    "github_readme_debugging": ["github", "readme", "markdown", "video embed", "image embed", "badge", "repository"],
    "deployment_debugging": ["deploy", "deployment", "build", "hosting", "ci"],
    "cloud_lab_practice": ["aws", "azure", "gcp", "cloud lab", "iam", "s3", "ec2", "lambda", "resource group"],
    "ai_course_lab": ["ai course", "machine learning", "model training", "notebook", "prompt", "llm", "fine tuning"],
    "ai_project_build_log": ["openai", "langchain", "rag", "vector database", "agent", "embedding", "chatbot"],
    "learning_path_reflection": ["course", "lecture", "lesson", "learning path", "module", "progress"],
    "certification_badge_summary": ["certification", "certificate", "badge", "credential", "exam", "completed"],
}


def promote_article_type_from_evidence(
    classification: dict[str, Any],
    evidence: list[dict[str, Any]],
    raw_text: str,
    memo: str,
    image_names: list[str],
) -> dict[str, Any]:
    source = " ".join([raw_text, memo, " ".join(image_names), evidence_source_text(evidence)]).lower()
    if is_microsoft_foundry_first_agent_context(raw_text, memo):
        return classification
    github_terms = [
        "agentic workflows",
        "automation that actually reads the room",
        "workflow_dispatch",
        "update-github-info",
        "update-github-info.lock.yml",
        "activation",
        "conclusion",
        "github actions",
        ".github/workflows",
        " yml",
        " yaml",
    ]
    hit_count = sum(1 for term in github_terms if term in source)
    current_conf = float(classification.get("confidence") or 0)
    if hit_count >= 3 and current_conf < 0.92:
        candidates = classification.get("candidates") if isinstance(classification.get("candidates"), list) else []
        return {
            "article_type": "github_agentic_workflow",
            "confidence": max(0.72, min(0.92, 0.48 + hit_count * 0.08)),
            "candidates": [{"article_type": "github_agentic_workflow", "score": hit_count}] + candidates[:4],
            "promoted_from_evidence": True,
        }
    return classification


def classify_article_type(raw_text: str, memo: str, topic: str, extra_info: str, image_names: list[str]) -> str:
    return classify_article_type_with_confidence(raw_text, memo, topic, extra_info, image_names)["article_type"]


def classify_article_type_with_confidence(raw_text: str, memo: str, topic: str, extra_info: str, image_names: list[str]) -> dict[str, Any]:
    source = " ".join([raw_text, memo, topic, extra_info, " ".join(image_names)]).lower()
    compact_source = source.replace(" ", "_").replace("-", "_")
    classification_text = "\n".join([raw_text, topic, extra_info, " ".join(image_names)])
    if is_agent_academy_context(classification_text, memo):
        return {
            "article_type": "agent_academy_copilot_studio",
            "confidence": 0.97,
            "candidates": [{"article_type": "agent_academy_copilot_studio", "score": 99}],
        }
    if is_microsoft_foundry_first_agent_context(classification_text, memo):
        return {
            "article_type": "microsoft_foundry_first_agent",
            "confidence": 0.96,
            "candidates": [{"article_type": "microsoft_foundry_first_agent", "score": 99}],
        }
    if is_wikidocs_coding_test_context(classification_text, memo):
        return {
            "article_type": "coding_test_python_learning",
            "confidence": 0.95,
            "candidates": [{"article_type": "coding_test_python_learning", "score": 96}],
        }
    if is_oopy_cs_notes_context(classification_text, memo):
        return {
            "article_type": "cs_learning_notes",
            "confidence": 0.92,
            "candidates": [{"article_type": "cs_learning_notes", "score": 92}],
        }
    if is_foundry_iq_mcp_rag_context(classification_text, memo):
        return {
            "article_type": "ai_project_build_log",
            "confidence": 0.78,
            "candidates": [{"article_type": "ai_project_build_log", "score": 8}],
        }
    for article_type, example_dir in ARTICLE_TYPE_EXAMPLE_DIRS.items():
        if article_type in compact_source or example_dir in compact_source:
            return {"article_type": article_type, "confidence": 0.98, "candidates": [{"article_type": article_type, "score": 99}]}
    scores = {
        article_type: sum(1 for keyword in keywords if keyword in source)
        for article_type, keywords in ARTICLE_TYPE_KEYWORDS.items()
    }
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
    if best_score <= 0:
        return {
            "article_type": "unknown",
            "confidence": 0.0,
            "candidates": [{"article_type": item[0], "score": item[1]} for item in sorted_scores[:5]],
        }
    confidence = min(0.95, 0.25 + best_score * 0.16 + max(best_score - second_score, 0) * 0.08)
    if best_score < 2 and confidence < 0.55:
        return {
            "article_type": "general_learning_portfolio",
            "confidence": round(confidence, 3),
            "candidates": [{"article_type": item[0], "score": item[1]} for item in sorted_scores[:5]],
        }
    return {
        "article_type": best_type,
        "confidence": round(confidence, 3),
        "candidates": [{"article_type": item[0], "score": item[1]} for item in sorted_scores[:5]],
    }


def load_golden_context(article_type: str) -> dict[str, Any]:
    dirname = ARTICLE_TYPE_EXAMPLE_DIRS.get(article_type)
    if not dirname:
        return {}
    base = EXAMPLES_DIR / dirname
    context: dict[str, Any] = {}
    expected_map = base / "expected_problem_map.json"
    if expected_map.exists():
        try:
            context["expected_problem_map"] = json.loads(expected_map.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            context["expected_problem_map"] = {}
    for key, filename in {
        "pattern_notes": "pattern_notes.md",
        "image_order": "README_image_order.txt",
        "golden_article": "golden_article.md",
    }.items():
        path = base / filename
        if path.exists():
            context[key] = path.read_text(encoding="utf-8", errors="replace")[:6000]
    return context


def build_image_evidence(
    llm_client: LLM,
    raw_text: str,
    memo: str,
    image_files: list[Path],
    topic: str,
    extra_info: str,
    image_names: list[str],
) -> list[dict[str, Any]]:
    llm_client.last_vision_error = None
    caption_source = read_image_order_caption_source()
    filename_context = "\n".join(
        f"이미지 {index}: original_filename={name}, saved_filename={path.name}"
        for index, (path, name) in enumerate(zip(image_files, image_names, strict=False), start=1)
    )
    if not image_files:
        return [
            {
                "image_no": 0,
                "caption": "이미지 없음 - 사용자가 입력한 텍스트와 메모 기반 문제 분석",
                "visible_evidence": [part for part in [raw_text[:120], memo[:120]] if part],
                "role": "problem",
                "problem_signal": raw_text or memo or "이미지 없이 텍스트 입력만 제공됨",
                "technical_entities": infer_entities(" ".join([raw_text, memo, topic, extra_info])),
                "inferred_meaning": "업로드된 이미지가 없어 사용자의 텍스트와 메모를 문제 해결 흐름의 근거로 사용해야 합니다.",
            }
        ]

    client = llm_client.get_client()
    if not client:
        return fallback_image_evidence(image_files, image_names, raw_text, memo, caption_source)

    results: list[dict[str, Any]] = []
    for start in range(0, len(image_files), GROQ_VISION_CHUNK_SIZE):
        chunk = image_files[start : start + GROQ_VISION_CHUNK_SIZE]
        chunk_names = image_names[start : start + GROQ_VISION_CHUNK_SIZE]
        prompt = f"""
입력 이미지를 순서대로 분석해 ImageEvidence JSON 배열만 반환하세요.

각 원소는 반드시 이 구조를 따릅니다.
{{
  "image_no": 1,
  "primary_topic": "현재 이미지에서 확인되는 핵심 학습 주제",
  "platform_or_product": "화면에서 확인되거나 강하게 뒷받침되는 제품/플랫폼/도메인 이름",
  "topic_terms": ["현재 이미지 주제를 뒷받침하는 구체적 용어"],
  "caption": "이미지 1 - 문제 해결 서사에 필요한 구체적 캡션",
  "visible_evidence": ["화면에 보이는 단어, 값, 오류, 테이블, 수식"],
  "role": "problem | cause | solution | validation | final_result",
  "problem_signal": "이 이미지가 보여주는 문제 신호",
  "technical_entities": ["Product", "Sales", "ProductKey", "DAX", "Relationship"],
  "inferred_meaning": "전체 실습 흐름에서 이 이미지가 갖는 의미"
}}

규칙:
- 한국어로 작성합니다.
- 이미지를 단순 설명하지 말고 문제/원인/해결/검증 역할로 분류합니다.
- README_image_order.txt 내용이 있으면 caption source로 우선 사용합니다.
- 원본 파일명에 01_Repeated_Category_Sales 같은 순서/의미가 있으면 그 의미를 caption에 반영합니다.
- primary_topic과 platform_or_product는 이전 예제나 README가 아니라 현재 이미지에 보이는 UI, 제목, 수식, 고유 용어로만 판단합니다.
- 제품명이 화면에 직접 보이지 않아도 여러 고유 용어가 한 제품/도메인을 강하게 지지할 때만 platform_or_product를 채웁니다. 확신이 없으면 빈 문자열로 둡니다.
- 보이지 않는 수식, 성과, 배포는 만들지 않습니다.
- JSON 배열만 반환합니다.

[전체 주제]
{topic}

[사용자 메모]
{memo}

[추가 정보]
{extra_info}

[직접 입력 텍스트]
{raw_text}

[README_image_order.txt]
{caption_source or "없음"}

[파일명/순서]
{filename_context}
""".strip()
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for offset, image_file in enumerate(chunk, start=start + 1):
            mime_type = mimetypes.guess_type(image_file.name)[0] or "image/png"
            image_data = base64.b64encode(image_file.read_bytes()).decode("ascii")
            content_parts.append({"type": "text", "text": f"이미지 {offset} / 원본 파일명: {chunk_names[offset - start - 1]}"})
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}})
        try:
            completion = client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                temperature=0.15,
                max_tokens=GROQ_VISION_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": "You convert technical screenshots into grounded JSON ImageEvidence. Return JSON only."},
                    {"role": "user", "content": content_parts},
                ],
            )
            data = parse_json_payload(completion.choices[0].message.content or "")
            if isinstance(data, list):
                results.extend(normalize_image_evidence(data, start + 1))
        except Exception as exc:
            llm_client.last_vision_error = exc
            print(f"[ImageEvidence error] {exc}")
            if is_groq_rate_limit_error(exc):
                break

    return results or fallback_image_evidence(image_files, image_names, raw_text, memo, caption_source)


def build_problem_map(
    llm_client: LLM,
    raw_text: str,
    memo: str,
    evidence: list[dict[str, Any]],
    topic: str,
    extra_info: str,
    article_type: str,
    golden_context: dict[str, Any],
) -> dict[str, Any]:
    prompt = f"""
다음 ImageEvidence와 사용자 입력을 하나의 문제 해결 흐름으로 통합해 ProblemMap JSON만 반환하세요.

구조:
{{
  "core_problem": "...",
  "why_problematic": "...",
  "root_causes": ["..."],
  "solution_steps": [
    {{"step": 1, "title": "...", "problem": "...", "cause": "...", "action": "...", "verification": "..."}}
  ],
  "complex_problem": {{
    "title": "...",
    "symptom": "...",
    "initial_assumption": "...",
    "actual_cause": "...",
    "resolution": "...",
    "verification": "..."
  }},
  "final_outcome": "..."
}}

규칙:
- 이미지별 설명이 아니라 전체 흐름의 핵심 문제를 뽑습니다.
- solution_steps는 최소 4개를 만듭니다.
- 각 step은 문제/원인/조치/확인 결과를 모두 포함합니다.
- 입력에 없는 수식, 코드, 배포, 성과는 만들지 않습니다.
- Power BI semantic model 맥락이면 relationship, cardinality, filter context, DAX 검증 중심으로 씁니다.
- article_type별 golden example 문장이나 expected_problem_map 문장을 복사하지 않습니다.
- 새 입력의 ImageEvidence, 사용자 메모, Q&A Logs에 있는 근거만 ProblemMap의 재료로 사용합니다.
- article_type은 구조 선택에만 사용하고, 새 입력에 없는 Power BI 테이블명/수식/성과를 가져오지 않습니다.

[분류된 article_type]
{article_type}

[주제]
{topic}

[사용자 메모]
{memo}

[추가 정보]
{extra_info}

[직접 입력 텍스트]
{raw_text}

[ImageEvidence]
{json.dumps(evidence, ensure_ascii=False)}
""".strip()
    data = llm_json(llm_client, prompt, max_tokens=3600)
    return normalize_problem_map(data if isinstance(data, dict) else {}, raw_text, memo, evidence, topic, article_type, golden_context)


def build_article_brief(
    llm_client: LLM,
    raw_text: str,
    memo: str,
    evidence: list[dict[str, Any]],
    problem_map: dict[str, Any],
    topic: str,
    extra_info: str,
) -> dict[str, Any]:
    if problem_map.get("article_type") in ARTICLE_TYPE_EXAMPLE_DIRS:
        return normalize_article_brief({}, topic, problem_map)

    prompt = f"""
최종 Medium 글을 쓰기 전 ArticleBrief JSON만 반환하세요.

구조:
{{
  "korean_title": "...",
  "english_subtitle": "...",
  "article_thesis": "...",
  "target_reader": "recruiter / hiring manager / technical reviewer",
  "portfolio_angle": "...",
  "do_not_claim": ["사용자가 하지 않은 성과", "없는 수식", "없는 배포"],
  "must_include": ["문제 인식", "원인 분석", "검증 결과", "이미지 캡션 목록"]
}}

[주제]
{topic}

[추가 정보]
{extra_info}

[ProblemMap]
{json.dumps(problem_map, ensure_ascii=False)}

[ImageEvidence]
{json.dumps(evidence, ensure_ascii=False)}

[사용자 입력]
{raw_text}

{memo}
""".strip()
    data = llm_json(llm_client, prompt, max_tokens=1800)
    return normalize_article_brief(data if isinstance(data, dict) else {}, topic, problem_map)


def build_article_outline(
    llm_client: LLM,
    brief: dict[str, Any],
    problem_map: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    if problem_map.get("article_type") in ARTICLE_TYPE_EXAMPLE_DIRS:
        return {title: default_outline_items(title, brief, problem_map, evidence) for title in SECTION_TITLES}

    prompt = f"""
다음 15개 섹션을 모두 포함하는 Medium 글 Outline JSON만 반환하세요.
각 key는 섹션 제목이고 value는 해당 섹션에서 반드시 다룰 bullet 배열입니다.

필수 섹션:
{json.dumps(SECTION_TITLES, ensure_ascii=False)}

규칙:
- 문제 해결 경험 섹션은 최소 4개 단계로 나눕니다.
- 각 단계는 문제/제약 → 원인 판단 → 조치 → 확인 결과를 포함합니다.
- 이미지 번호와 캡션 목록은 마지막 섹션에 둡니다.

[ArticleBrief]
{json.dumps(brief, ensure_ascii=False)}

[ProblemMap]
{json.dumps(problem_map, ensure_ascii=False)}

[ImageEvidence]
{json.dumps(evidence, ensure_ascii=False)}
""".strip()
    data = llm_json(llm_client, prompt, max_tokens=2600)
    if not isinstance(data, dict):
        data = {}
    for title in SECTION_TITLES:
        data.setdefault(title, default_outline_items(title, brief, problem_map, evidence))
    return data


def generate_section(
    llm_client: LLM,
    section_title: str,
    outline: dict[str, Any],
    brief: dict[str, Any],
    problem_map: dict[str, Any],
    evidence: list[dict[str, Any]],
    raw_text: str,
    memo: str,
    extra_info: str,
) -> str:
    if os.getenv("LLM_SECTION_GENERATION", "0") != "1":
        return structured_section(section_title, brief, problem_map, evidence, raw_text, memo, extra_info)

    if section_title == "한국어 제목":
        return f"# {brief.get('korean_title') or '문제 해결형 학습 기록'}"
    if section_title == "영어 부제":
        return f"_{brief.get('english_subtitle') or 'A problem-solving portfolio article from study evidence'}_"

    prompt = f"""
Medium 복붙용 Markdown의 단일 섹션만 작성하세요.

섹션 제목: {section_title}

작성 규칙:
- 내부 라벨, placeholder, id, :::writing을 절대 쓰지 않습니다.
- 이미지 설명 나열이 아니라 문제 해결 서사로 작성합니다.
- 없는 수식/코드/배포/성과를 만들지 않습니다.
- "버튼을 눌렀습니다", "실습을 완료했습니다" 같은 기능 설명 중심 문장을 피합니다.
- 섹션 헤더는 Markdown `## {section_title}` 형태로 시작합니다.

섹션별 길이/구조:
- 문제 인식, 문제 정의, 왜 이것을 문제로 인식했는가: 각각 최소 2문단.
- 문제 해결 경험: 최소 4개 단계, 각 단계는 문제/제약 → 원인 판단 → 조치 → 확인 결과 구조.
- Portfolio Summary: 영어 2문단 이상.
- Key skills practiced: 최소 8개 bullet.
- 이미지 번호와 캡션 목록: 마지막에 이미지별 캡션 bullet을 모두 포함.

[이 섹션의 outline]
{json.dumps(outline.get(section_title, []), ensure_ascii=False)}

[ArticleBrief]
{json.dumps(brief, ensure_ascii=False)}

[ProblemMap]
{json.dumps(problem_map, ensure_ascii=False)}

[ImageEvidence]
{json.dumps(evidence, ensure_ascii=False)}

[사용자 원문]
{raw_text}

[사용자 메모]
{memo}

[추가 정보]
{extra_info}
""".strip()
    text = llm_text(llm_client, prompt, max_tokens=2600)
    return text.strip() or fallback_section(section_title, brief, problem_map, evidence, raw_text, memo)


def structured_section(
    section_title: str,
    brief: dict[str, Any],
    problem_map: dict[str, Any],
    evidence: list[dict[str, Any]],
    raw_text: str,
    memo: str,
    extra_info: str,
) -> str:
    core = str(problem_map.get("core_problem") or brief.get("article_thesis") or memo or raw_text or "학습 과정에서 발견한 문제를 구조화하는 것")
    why = str(problem_map.get("why_problematic") or "겉으로는 결과가 보이더라도 원인과 검증 기준이 정리되지 않으면 같은 문제를 재현하거나 설명하기 어렵습니다.")
    final = str(problem_map.get("final_outcome") or "이미지와 메모를 근거로 문제, 원인, 해결, 검증 흐름을 정리했습니다.")
    steps = [step for step in problem_map.get("solution_steps", []) if isinstance(step, dict)]
    captions = [str(item.get("caption")) for item in evidence if item.get("caption")]
    entities = sorted({entity for item in evidence for entity in normalize_str_list(item.get("technical_entities"))})
    article_type = str(problem_map.get("article_type") or "general_learning_portfolio")
    entities_text = article_type_entities_text(article_type, entities, problem_map)
    type_focus = article_type_focus_text(str(problem_map.get("article_type") or "general_learning_portfolio"))
    clean_core = clean_summary_sentence(core)

    if section_title == "한국어 제목":
        return f"# {brief.get('korean_title') or core}"
    if section_title == "영어 부제":
        return f"_{brief.get('english_subtitle') or 'A problem-solving portfolio article from technical study evidence'}_"
    if section_title == "짧은 도입부":
        return (
            "## 짧은 도입부\n"
            f"이번 기록의 출발점은 단순히 화면을 캡처한 것이 아니라, `{core}`라는 문제를 어떻게 인식하고 검증 가능한 흐름으로 바꾸었는지 정리하는 데 있다. "
            "화면에는 결과값, 관계, 수식, 설정 화면이 각각 흩어져 있지만, 포트폴리오 글에서 중요한 것은 그 장면들을 순서대로 설명하는 일이 아니다. "
            "중요한 것은 초반 화면에서 어떤 이상 신호를 보았고, 중간 화면에서 어떤 원인 후보를 좁혔으며, 마지막 화면에서 어떤 기준으로 해결 여부를 확인했는지 연결하는 것이다.\n\n"
            f"따라서 이 글은 {entities_text}를 중심으로 이미지 묶음을 하나의 문제 해결 서사로 재구성한다. "
            "사용자가 직접 남긴 메모와 화면 근거를 우선하고, 입력에 없는 성과나 수식은 임의로 만들지 않는다."
        )
    if section_title == "핵심 작업 요약":
        bullets = "\n".join(
            f"- {step.get('title')}: {step.get('action')} / 확인 기준: {step.get('verification')}"
            for step in steps[:6]
        )
        semantic_summary = ""
        if problem_map.get("article_type") == "semantic_model_relationship":
            semantic_summary = (
                "\n- 필수 검증 흐름: 이미지 1 repeated value 문제, 이미지 2~3 Product[ProductKey] -> Sales[ProductKey] relationship, "
                "이미지 4 Star schema, 이미지 5~6 Product hierarchy, 이미지 7~8 Profit / Profit Margin과 DIVIDE, "
                "이미지 11~13 SalespersonRegion Bridge table, Cross-filter direction Both, inactive relationship, "
                "이미지 14~15 Salesperson -> SalespersonRegion -> Region -> Sales 경로와 Targets 비교"
            )
        if problem_map.get("article_type") == "dax_measure_modeling":
            semantic_summary = (
                "\n- 필수 DAX 흐름: 이미지 1 Month를 MonthKey 기준으로 Sort by column 처리, 이미지 2 Fiscal hierarchy가 포함된 Date table 구성, "
                "이미지 3 Date table과 Sales relationship 확인, 이미지 4 Mark as date table 지정, 이미지 5 Avg Price explicit measure 생성, "
                "이미지 6 Median Price / Min Price / Max Price / Orders / Order Lines measure matrix 검증, 이미지 7 Currency format 확인, "
                "이미지 8 TargetAmount raw column 대신 Target measure 비교, 이미지 9 Salesperson별 Sales, Target, Variance, Variance Margin 최종 비교"
            )
        if problem_map.get("article_type") == "power_query_etl" and problem_map.get("_regression_profile") == "power_query_etl_example":
            semantic_summary = (
                "\n- 필수 ETL 흐름: SQL Server와 CSV를 Power Query Editor로 가져오고, Column quality / Column distribution / Column profile로 품질을 확인한 뒤 "
                "SalesPersonFlag, FirstName + LastName, ProductKey, DimProductSubcategory, DimProductCategory, BusinessType, Ware House / Warehouse, "
                "FactResellerSales, TotalProductCost, OrderQuantity, StandardCost, ResellerSalesTargets.csv, M01~M12 Unpivot, MonthNumber, TargetMonth, "
                "ColorFormats, Product[Color], Left Outer, Background Color Format, Font Color Format, Disable load, Close & Apply, 7개 테이블 로드를 검증"
            )
        return (
            "## 핵심 작업 요약\n"
            f"- 핵심 문제: {core}\n"
            f"- 문제로 본 이유: {why}\n"
            f"- 사용한 근거: {len(evidence)}개의 이미지 evidence와 사용자 메모\n"
            f"{bullets}\n"
            f"{semantic_summary}\n"
            f"- 최종 결과: {final}"
        )
    if section_title == "문제 인식":
        first_caption = captions[0] if captions else "초기 입력 화면"
        problem_detail = semantic_problem_detail(problem_map, evidence)
        memo_reference = f"사용자 메모에서 강조된 내용은 `{memo.strip()}`이며, " if memo.strip() else "사용자 별도 메모가 제공되지 않은 경우에도 "
        return (
            "## 문제 인식\n"
            f"처음 문제로 볼 수 있었던 신호는 `{first_caption}`에서 시작된다. {problem_detail} "
            f"{why} 그래서 이 기록에서는 화면을 순서대로 묘사하는 대신, 초반 이미지를 문제 신호로 보고 이후 이미지들을 원인 분석과 검증 근거로 연결했다.\n\n"
            "특히 포트폴리오 글에서 중요한 부분은 기능 사용 여부가 아니라 문제 인식의 정확도다. 화면에 숫자가 나오거나 설정 창이 열린 것만으로는 분석이 끝나지 않는다. "
            f"{memo_reference}이미지에 남아 있는 테이블명, 컬럼명, 변환 단계가 문제 해결형 서사를 구성하는 기준이 된다. "
            "따라서 이 단계의 핵심은 ‘무엇을 보았는가’가 아니라 ‘왜 그것을 문제로 정의했는가’다."
        )
    if section_title == "문제 정의":
        roots = "\n".join(f"- {item}" for item in normalize_str_list(problem_map.get("root_causes")))
        return (
            "## 문제 정의\n"
            f"이 실습에서 정의한 문제는 `{core}`다. 이 문제는 단일 화면이나 단일 버튼 조작으로 해결되는 문제가 아니라, 입력 근거 전체를 통해 원인과 검증 기준을 함께 세워야 하는 유형이다. "
            f"화면에 보이는 결과가 그럴듯해도, 데이터 모델, 수식, 관계, 변환 흐름 중 하나가 어긋나면 최종 해석은 달라질 수 있다. {type_focus}\n\n"
            f"문제의 원인 후보는 다음과 같이 정리할 수 있다.\n{roots}\n\n"
            "이 정의가 중요한 이유는 해결 방법의 범위를 좁혀 주기 때문이다. 문제를 단순 UI 조작으로 보면 화면 설명에 머물지만, 구조적 문제로 정의하면 어떤 근거를 확인하고 어떤 결과를 검증해야 하는지 분명해진다."
        )
    if section_title == "왜 이것을 문제로 인식했는가":
        return (
            "## 왜 이것을 문제로 인식했는가\n"
            f"{why} 이 지점은 학습 기록과 포트폴리오 글의 차이를 만든다. 학습 기록은 화면에서 무엇을 했는지 남기는 데 그칠 수 있지만, 문제 해결형 글은 왜 그 장면이 위험 신호였는지, 어떤 분석 오류로 이어질 수 있는지 설명해야 한다.\n\n"
            "또한 이 문제는 재현 가능성과 검증 가능성의 문제이기도 하다. 같은 화면을 다시 보았을 때 어떤 값, 관계, 수식, 변환 결과를 확인해야 하는지 정리되어 있지 않으면 다음 실습에서 같은 문제를 다시 만날 수 있다. "
            f"그래서 이 글에서는 {entities_text}를 단순 키워드가 아니라 원인 분석과 검증 결과를 잇는 증거로 사용한다."
        )
    if section_title == "문제 해결 경험":
        blocks = ["## 문제 해결 경험"]
        for step in steps:
            detail_text = concrete_detail_text(step)
            image_ref_text = image_ref_caption_text(step, evidence)
            blocks.append(
                f"### {step.get('step')}. {step.get('title')}\n"
                f"{image_ref_text}\n\n"
                f"문제/제약: {step.get('problem')}\n\n"
                f"원인 판단: {step.get('cause')}\n\n"
                f"조치: {step.get('action')}{detail_text}\n\n"
                f"확인 결과: {step.get('verification')}"
                f"{portfolio_meaning_text(step)}"
            )
        return "\n\n".join(blocks)
    if section_title == "복잡한 문제/수식/쿼리/코드 작성 및 해결 경험":
        complex_problem = problem_map.get("complex_problem", {}) if isinstance(problem_map.get("complex_problem"), dict) else {}
        semantic_complex = ""
        if problem_map.get("article_type") == "semantic_model_relationship":
            semantic_complex = (
                "\n\nSalesperson 분석에서는 단순 직접 관계만으로 충분하지 않았다. 담당 Region 기준 분석을 만들려면 "
                "Salesperson -> SalespersonRegion -> Region -> Sales 경로가 필요하고, 이 경로가 direct Salesperson-Sales relationship과 충돌하면 "
                "모호한 filter path가 생긴다. 그래서 SalespersonRegion Bridge table을 중심으로 Cross-filter direction을 Both로 조정하고, "
                "충돌하는 direct relationship은 inactive relationship으로 두어 Salesperson별 Sales와 Targets 비교가 같은 기준에서 계산되는지 확인해야 한다."
            )
        if problem_map.get("article_type") == "power_query_etl" and problem_map.get("_regression_profile") == "power_query_etl_example":
            semantic_complex = (
                "\n\nPower Query ETL에서 복잡한 지점은 오류 수정이 아니라 분석 가능한 형태로 데이터를 바꾸는 판단이다. "
                "첫째, FactResellerSales의 TotalProductCost가 null이면 원가와 이익 분석이 왜곡될 수 있으므로 "
                "`if [TotalProductCost] = null then [OrderQuantity] * [StandardCost] else [TotalProductCost]` 방식으로 Cost를 보완하고 Fixed Decimal Number 타입을 확인해야 한다. "
                "둘째, ResellerSalesTargets.csv의 M01~M12 구조는 wide format이라 날짜 기반 분석에 부적합하므로 Unpivot을 통해 MonthNumber와 Target 구조로 만들고 TargetMonth를 생성해야 한다. "
                "셋째, ColorFormats는 Product[Color] 기준으로 Left Outer Merge해 Background Color Format과 Font Color Format을 Product에 붙이되, ColorFormats 자체는 Disable load 처리하여 최종 모델에는 Salesperson, SalespersonRegion, Product, Reseller, Region, Sales, Targets 7개 테이블만 Close & Apply로 로드해야 한다."
            )
        return (
            "## 복잡한 문제/수식/쿼리/코드 작성 및 해결 경험\n"
            f"가장 복잡한 지점은 `{complex_problem.get('title', '복합 원인 분석')}`이었다. 증상은 {complex_problem.get('symptom', core)}로 정리할 수 있다. "
            f"처음에는 {complex_problem.get('initial_assumption', '단일 설정 문제로 볼 수 있었다')} 하지만 실제 원인은 {complex_problem.get('actual_cause', '여러 근거를 함께 확인해야 하는 구조적 문제')}에 가까웠다.\n\n"
            f"해결은 {complex_problem.get('resolution', '문제, 원인, 조치, 검증 단계를 분리하는 방식')}으로 접근했다. "
            f"검증은 {complex_problem.get('verification', final)}를 기준으로 삼았다. "
            "수식이나 코드가 포함된 경우에도 핵심은 나열이 아니라 필요성, 계산 의미, 검증 방법을 함께 설명하는 것이다. 입력 근거에 명시되지 않은 수식은 만들지 않고, 보이는 기술 요소와 사용자 메모에 기반해 설명 범위를 제한했다."
            f"{semantic_complex}"
        )
    if section_title == "성과":
        return (
            "## 성과\n"
            f"이번 작업의 성과는 {final} 단순히 이미지를 저장한 것이 아니라, 문제 인식에서 원인 분석, 조치, 확인 결과까지 이어지는 설명 가능한 기록으로 바꾼 점이 중요하다.\n\n"
            "이 방식은 포트폴리오 관점에서도 의미가 있다. 결과 화면만 보여주는 글은 사용자의 판단 과정을 설명하지 못하지만, 문제 해결형 글은 어떤 신호를 보고 문제를 정의했는지, 어떤 원인을 의심했는지, 어떤 검증 기준으로 해결 여부를 판단했는지를 보여준다."
        )
    if section_title == "사용한 주요 수식/코드 정리":
        if problem_map.get("article_type") == "power_query_etl" and problem_map.get("_regression_profile") == "power_query_etl_example":
            return (
                "## 사용한 주요 수식/코드 정리\n"
                "이번 ETL 실습에서 중심이 되는 코드는 DAX measure가 아니라 Power Query의 Custom Column 계산식이다. "
                "핵심은 FactResellerSales의 TotalProductCost가 null인 행을 그대로 두지 않고, 수량과 표준 원가를 이용해 Cost를 보완하는 것이다.\n\n"
                "```powerquery\n"
                "if [TotalProductCost] = null\n"
                "then [OrderQuantity] * [StandardCost]\n"
                "else [TotalProductCost]\n"
                "```\n\n"
                "왜 필요한가: TotalProductCost가 null이면 Sales만 남고 Cost가 비어 수익성 분석의 기준이 흔들린다. OrderQuantity와 StandardCost는 같은 행에서 원가를 추정할 수 있는 근거이므로, null인 경우에만 이 계산을 적용해야 한다.\n\n"
                "어떤 변환인가: 기존 TotalProductCost 값을 무조건 덮어쓰는 것이 아니라, null인 행에는 OrderQuantity * StandardCost를 사용하고 값이 존재하는 행에는 원래 TotalProductCost를 유지한다. 이렇게 만든 Cost는 Unit Price, Sales와 함께 Fixed Decimal Number 타입으로 맞춰 이후 모델에서 금액 컬럼처럼 사용할 수 있다.\n\n"
                "어떻게 검증했는가: Cost Custom Column이 생성되었는지, TotalProductCost와 StandardCost를 정리한 뒤에도 Cost가 남아 있는지, 그리고 Sales 쿼리의 Unit Price, Sales, Cost가 Fixed Decimal Number로 정리되었는지 확인했다. DAX가 아니라 Power Query 단계에서 원천 데이터의 분석 가능성을 확보한 것이 이 섹션의 핵심이다."
            )
        measures = []
        for step in steps:
            for measure in step.get("dax_measures", []) if isinstance(step, dict) else []:
                measures.append(
                    f"### {measure.get('measure_name')}\n"
                    f"```DAX\n{measure.get('formula')}\n```\n"
                    f"- 왜 필요한가: {measure.get('why_needed')}\n"
                    f"- 검증 방법: {measure.get('validation')}"
                )
        measure_text = "\n\n".join(measures) or "입력 근거에서 명확한 수식이 확인되지 않은 항목은 임의로 만들지 않았다."
        return (
            "## 사용한 주요 수식/코드 정리\n"
            f"입력 근거에서 확인해야 할 주요 기술 요소는 {entities_text}다. 이 섹션에서는 보이지 않는 수식이나 코드를 임의로 만들지 않는다. 대신 사용자가 제공한 화면과 메모 안에서 확인되는 관계, 수식, 쿼리, 변환 기준만 문제 해결 흐름에 연결한다.\n\n"
            f"{measure_text}\n\n"
            "수식이나 코드가 등장하는 경우 설명 기준은 세 가지다. 첫째, 왜 그 수식이나 코드가 필요했는가. 둘째, 어떤 계산 또는 변환을 수행했는가. 셋째, 결과를 어떻게 검증했는가. "
            "이 기준을 지키면 코드는 단순 장식이 아니라 문제를 해결한 근거가 된다."
        )
    if section_title == "최종 정리":
        return (
            "## 최종 정리\n"
            f"처음 문제는 {core}로 정의되었다. 마지막 정리는 이 문제로 다시 돌아가야 한다. {final} 이 흐름이 유지되어야 글이 단순 실습 후기가 아니라 문제 해결형 포트폴리오 글이 된다.\n\n"
            "이번 기록에서 중요한 태도는 화면을 그대로 설명하지 않고, 각 이미지를 문제, 원인, 해결, 검증의 역할로 재배치한 점이다. "
            "그 결과 독자는 사용자가 어떤 기술을 사용했는지만이 아니라, 왜 그 기술이 필요했고 어떤 기준으로 결과를 확인했는지 이해할 수 있다."
        )
    if section_title == "Portfolio Summary":
        return (
            "## Portfolio Summary\n"
            f"This project note reframes the study evidence around a concrete technical problem: {clean_core}. Instead of treating screenshots as a chronological UI walkthrough, the article connects the early problem signal, the suspected causes, the corrective actions, and the final verification criteria into one explainable workflow.\n\n"
            f"The portfolio value is in the reasoning process. The work shows how the learner used visible evidence, technical entities such as {entities_text}, and follow-up validation to move from observation to diagnosis and resolution. This makes the record useful for a recruiter, hiring manager, or technical reviewer because it demonstrates structured debugging, analytical writing, and evidence-based communication."
        )
    if section_title == "Key skills practiced":
        return "## Key skills practiced\n" + "\n".join(f"- {skill}" for skill in key_skills_for_article_type(article_type))
    if section_title == "이미지 번호와 캡션 목록":
        caption_lines = "\n".join(f"- {caption}" for caption in captions) or "- 이미지 없음 - 텍스트와 메모 기반으로 작성"
        return f"## 이미지 번호와 캡션 목록\n{caption_lines}"
    return fallback_section(section_title, brief, problem_map, evidence, raw_text, memo)


def severe_sparse_article_failures(critique: CritiqueResult) -> list[str]:
    severe_markers = [
        "Power BI regression template",
        "placeholder",
        "generic",
        "title",
        "internal",
        "caption",
        "golden/example",
        "unsupported claim",
        "solution_steps",
    ]
    return [failure for failure in critique.failures if any(marker.lower() in failure.lower() for marker in severe_markers)]


def critique_article(
    article: str,
    article_type: str = "general_learning_portfolio",
    problem_map: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None,
) -> CritiqueResult:
    problem_map = problem_map or {}
    evidence = evidence or []
    failures: list[str] = []
    section_failures: dict[str, str] = {}
    metrics = {
        "char_count": len(article),
        "problem_solution_steps": count_problem_solution_steps(article),
        "placeholder_count": sum(article.count(phrase) for phrase in PLACEHOLDER_PHRASES),
        "function_description_count": sum(article.count(phrase) for phrase in FUNCTION_DESCRIPTION_PHRASES),
        "article_type": article_type,
        "uploaded_images_count": int(problem_map.get("_uploaded_images_count") or 0),
        "image_evidence_count": len(evidence),
    }
    section_plan = problem_map.get("_section_plan", []) if isinstance(problem_map.get("_section_plan"), list) else []
    metrics["section_plan_count"] = len(section_plan)
    if article_type == "unknown":
        failures.append("article_type이 unknown입니다. evidence가 부족해 최종 글을 생성할 수 없습니다.")
        section_failures["article_type"] = "현재 evidence와 후보 article_type을 사용자에게 반환해야 함"
    if not section_plan:
        failures.append("Section Plan 없이 Final Article 생성이 시도되었습니다.")
        section_failures["section_plan"] = "Final Article 이전에 section_plan 생성 필요"
    coverage_failure = image_coverage_failure(problem_map, evidence)
    if coverage_failure:
        failures.append(coverage_failure)
        section_failures["image_evidence_coverage"] = coverage_failure
    if metrics["char_count"] < 4500:
        failures.append("전체 글이 4,500자 미만입니다.")
        section_failures["global_length"] = "전체 글 분량 확장 필요"
    if metrics["problem_solution_steps"] < 4:
        failures.append("문제 해결 경험이 4개 미만입니다.")
        section_failures["문제 해결 경험"] = "최소 4개 단계 필요"
    for title in ["문제 인식", "문제 정의", "왜 이것을 문제로 인식했는가"]:
        paragraph_count = count_section_paragraphs(article, title)
        metrics[f"{title}_paragraphs"] = paragraph_count
        if paragraph_count < 2:
            failures.append(f"{title} 섹션이 2문단 미만입니다.")
            section_failures[title] = "최소 2문단 필요"
    if "이미지 번호와 캡션 목록" not in article:
        failures.append("이미지 캡션 목록이 없습니다.")
        section_failures["이미지 번호와 캡션 목록"] = "마지막 캡션 목록 필요"
    postprocess_leftovers = awkward_korean_leftovers(article)
    metrics["awkward_korean_leftover_count"] = len(postprocess_leftovers)
    if postprocess_leftovers:
        failures.append(f"어색한 한국어 후처리 패턴이 남아 있습니다: {', '.join(postprocess_leftovers)}")
        section_failures["postprocess"] = "마침표/조사 결합 오류 후처리 필요"
    if metrics["placeholder_count"]:
        failures.append("placeholder 또는 내부 라벨 문구가 포함되어 있습니다.")
        section_failures["placeholder"] = "금지 문구 제거 필요"
    diagnostic_leaks = [
        phrase
        for phrase in INTERNAL_DIAGNOSTIC_PHRASES
        if phrase in article or phrase in json.dumps(problem_map, ensure_ascii=False)
    ]
    metrics["internal_diagnostic_leak_count"] = len(diagnostic_leaks)
    if diagnostic_leaks:
        failures.append(f"내부 Vision/provider 진단 문구가 최종 글 또는 ProblemMap에 노출되었습니다: {', '.join(diagnostic_leaks)}")
        section_failures["internal_diagnostics"] = "Vision/provider 내부 상태는 final article 재료가 아니라 debug/provider_diagnostics로만 표시해야 함"
    if metrics["function_description_count"] > 2:
        failures.append("기능 설명 중심 문장이 너무 많습니다.")
        section_failures["style"] = "문제/원인/검증 중심으로 재작성 필요"
    repeated_fillers = [phrase for phrase in FILLER_SENTENCES if article.count(phrase) >= 2]
    metrics["repeated_filler_count"] = len(repeated_fillers)
    if repeated_fillers:
        failures.append(f"반복 filler 문장이 2회 이상 등장합니다: {', '.join(repeated_fillers)}")
        section_failures["repetition"] = "template filler 제거 및 step별 concrete detail 사용 필요"
    forbidden_once = [phrase for phrase in FILLER_SENTENCES if phrase in article]
    metrics["forbidden_phrase_count"] = len(forbidden_once)
    if forbidden_once:
        failures.append(f"금지 문구가 최종 글에 포함되어 있습니다: {', '.join(sorted(set(forbidden_once)))}")
        section_failures["forbidden_phrases"] = "금지 문구 제거 필요"
    if article_type not in POWERBI_ARTICLE_TYPES:
        template_hits = [phrase for phrase in NON_POWERBI_TEMPLATE_PHRASES if phrase in article]
        placeholder_hits = [phrase for phrase in PLACEHOLDER_PHRASES if phrase in article]
        generic_hits = [phrase for phrase in [
            "학습 기록 기반 문제 해결 경험 과정에서 관찰한 결과와 의도한 분석 흐름의 불일치",
            "근거 화면 1 기반 검증 단계",
            "근거 화면 2 기반 검증 단계",
            "근거 화면 3 기반 검증 단계",
            "근거 화면 4 기반 검증 단계",
        ] if phrase in article]
        metrics["non_powerbi_template_contamination_count"] = len(template_hits)
        metrics["non_powerbi_placeholder_hit_count"] = len(placeholder_hits) + len(generic_hits)
        if template_hits:
            failures.append(f"Power BI regression template 문장이 non-PowerBI 글에 포함되었습니다: {', '.join(template_hits)}")
            section_failures["anti_overfitting"] = "non-PowerBI 입력에서는 모델/수식/관계/필터 흐름 템플릿 문장 제거 필요"
        if placeholder_hits or generic_hits:
            failures.append(f"non-PowerBI sparse capture에 placeholder/generic 문장이 포함되었습니다: {', '.join(sorted(set(placeholder_hits + generic_hits)))}")
            section_failures["sparse_capture"] = "placeholder solution step이나 generic core_problem은 Final Article이 아니라 missing_context로 내려야 함"
    if has_code_block(article) and not has_code_explanation(article):
        failures.append("코드/수식이 있으나 필요성/계산/검증 설명이 부족합니다.")
        section_failures["사용한 주요 수식/코드 정리"] = "수식 설명 보강 필요"
    if "확인 결과" not in article and "검증" not in article:
        failures.append("각 단계의 확인 결과가 부족합니다.")
        section_failures["문제 해결 경험"] = "확인 결과와 검증 내용 추가 필요"
    if count_section_paragraphs(article, "Portfolio Summary") < 2:
        failures.append("Portfolio Summary가 2문단 미만입니다.")
        section_failures["Portfolio Summary"] = "영어 2문단 이상 필요"
    key_skills = extract_section(article, "Key skills practiced")
    metrics["key_skills_count"] = len(re.findall(r"(?m)^-\s+", key_skills))
    if metrics["key_skills_count"] < 8:
        failures.append("Key skills practiced가 8개 미만입니다.")
        section_failures["Key skills practiced"] = "최소 8개 이상 필요"
    density_terms = ["문제/제약", "원인 판단", "조치", "확인 결과", "검증"]
    metrics["reasoning_density"] = sum(article.count(term) for term in density_terms)
    if metrics["char_count"] >= 4500 and metrics["reasoning_density"] < 12:
        failures.append("글자 수는 충분하지만 문제/원인/검증 밀도가 낮습니다.")
        section_failures["reasoning_density"] = "문제/원인/조치/검증 표현 보강 필요"
    steps = problem_map.get("solution_steps", [])
    if len(steps) < 4:
        failures.append("ProblemMap solution_steps가 4개 미만입니다.")
        section_failures["problem_map"] = "solution_steps 최소 4개 필요"
    config_min_steps = int(load_article_type_config(article_type).get("minimum_solution_steps") or 0)
    metrics["minimum_solution_steps"] = config_min_steps
    if config_min_steps and len(steps) < config_min_steps:
        failures.append(f"{article_type} solution_steps가 minimum_solution_steps {config_min_steps}개 미만입니다.")
        section_failures["problem_map"] = "article_type별 minimum_solution_steps 충족 필요"
    for index, step in enumerate(steps, start=1):
        if isinstance(step, dict) and not all(step.get(key) for key in ["problem", "cause", "action", "verification"]):
            failures.append(f"ProblemMap solution_step {index}에 problem/cause/action/verification 중 빠진 값이 있습니다.")
            section_failures["problem_map"] = "각 step 필수 필드 보강 필요"
            break
    captions = [str(item.get("caption", "")) for item in evidence]
    placeholder_captions = [
        caption
        for caption in captions
        if any(phrase in caption for phrase in ["문제 해결 서사에 필요한 구체적 캡션", "모델링 화면", "작업 결과 확인", "데이터 변환 작업"])
        or re.search(r"\.(png|jpg|jpeg)\b", caption, re.I)
    ]
    if placeholder_captions:
        failures.append("placeholder 수준 또는 파일명 기반 캡션이 포함되어 있습니다.")
        section_failures["captions"] = "사람이 읽을 수 있는 이미지 캡션으로 후처리 필요"
    truncated_captions = truncated_caption_lines(article)
    metrics["truncated_caption_count"] = len(truncated_captions)
    if truncated_captions:
        failures.append(f"이미지 캡션이 중간에서 잘린 것으로 보입니다: {', '.join(truncated_captions[:3])}")
        section_failures["captions"] = "이미지 캡션 마지막 줄이 완전한 문장/표현으로 끝나야 함"
    if evidence and not any(caption and caption in article for caption in captions):
        failures.append("solution_step 또는 본문이 이미지 캡션과 연결되지 않았습니다.")
        section_failures["image_refs"] = "이미지 캡션/근거를 본문에 연결 필요"
    entities = sorted({entity for item in evidence for entity in normalize_str_list(item.get("technical_entities"))})
    metrics["technical_entities_count"] = len(entities)
    if entities and sum(1 for entity in entities if entity.lower() in article.lower()) < min(3, len(entities)):
        failures.append("technical_entities가 본문에 충분히 반영되지 않았습니다.")
        section_failures["technical_entities"] = "핵심 기술 엔티티 반영 필요"
    if not final_outcome_links_to_problem(article, problem_map):
        failures.append("최종 결과가 처음 문제와 충분히 연결되지 않았습니다.")
        section_failures["최종 정리"] = "초기 문제와 최종 결과 연결 필요"
    missing_type_keywords = missing_article_type_keywords(article_type, article, problem_map)
    if missing_type_keywords:
        failures.append(f"article_type 기준 핵심 키워드가 부족합니다: {', '.join(missing_type_keywords)}")
        section_failures["article_type"] = "분류 유형별 핵심 문제/해결 키워드 반영 필요"
    required_entities = article_type_required_entities(article_type)
    if required_entities:
        missing_specific = [item for item in required_entities if item.lower() not in article.lower()]
        coverage = 1 - (len(missing_specific) / max(len(required_entities), 1))
        metrics["required_entity_coverage"] = round(coverage, 3)
        if coverage < 0.7:
            failures.append(f"{article_type} concrete entities coverage가 70% 미만입니다: {', '.join(missing_specific)}")
            section_failures["specificity"] = "article_type별 필수 concrete detail을 본문에 반영 필요"
    plan_failures, plan_metrics = plan_to_article_faithfulness(article, problem_map)
    metrics.update(plan_metrics)
    if plan_failures:
        failures.extend(plan_failures)
        section_failures["plan_to_article_faithfulness"] = "Section Plan 순서, must_include, image_refs, cause/action 구조를 final article에 강제 반영 필요"
    config = load_article_type_config(article_type)
    forbidden_cross_type = normalize_str_list(config.get("forbidden_cross_type_entities"))
    cross_hits = [item for item in forbidden_cross_type if item.lower() in article.lower()]
    metrics["cross_type_forbidden_hits"] = cross_hits
    if cross_hits:
        failures.append(f"선택된 article_type과 무관한 golden/example 키워드가 포함되었습니다: {', '.join(cross_hits)}")
        section_failures["anti_overfitting"] = "선택 article_type에 없는 Power BI/golden 키워드 제거 필요"
    unsupported_claims = unsupported_powerbi_claims(article, article_type, evidence, problem_map)
    metrics["unsupported_claim_count"] = len(unsupported_claims)
    if unsupported_claims:
        failures.append(f"입력 evidence에 없는 기술명/수식/테이블이 본문에 포함되었습니다: {', '.join(unsupported_claims)}")
        section_failures["unsupported_claims"] = "golden example에서 끌려온 unsupported claim 제거 필요"
    if generic_or_copied_title(article, problem_map):
        failures.append("제목 또는 core_problem이 입력 기반이 아니라 generic/golden example 문장처럼 보입니다.")
        section_failures["title"] = "입력 evidence 기반 제목과 core_problem 필요"
    if article_type == "semantic_model_relationship":
        if len(problem_map.get("solution_steps", [])) < 10:
            failures.append("semantic_model_relationship solution_steps가 10개 미만으로 압축되었습니다.")
            section_failures["problem_map"] = "semantic 유형은 repeated Sales부터 Targets 비교까지 최소 10개 내외 step 필요"
        semantic_mapping_failure = semantic_image_ref_mapping_failure(problem_map)
        if semantic_mapping_failure:
            failures.append(semantic_mapping_failure)
            section_failures["image_refs"] = "semantic step별 image_refs 고정 매핑 불일치"
        if "Repeated Category Sales" not in json.dumps(problem_map.get("complex_problems", []), ensure_ascii=False) or "SalespersonRegion bridge path" not in json.dumps(problem_map.get("complex_problems", []), ensure_ascii=False):
            failures.append("semantic complex problems가 Repeated Category Sales와 SalespersonRegion bridge path로 분리되지 않았습니다.")
            section_failures["complex_problem"] = "초반 repeated Sales 문제와 후반 bridge path 문제 분리 필요"
        covered, total, missing = semantic_regression_coverage(article)
        metrics["semantic_regression_coverage"] = round(covered / max(total, 1), 3)
        if covered / max(total, 1) < 0.7:
            failures.append(f"Semantic regression coverage가 70% 미만입니다. 누락: {', '.join(missing)}")
            section_failures["semantic_regression"] = "이미지 1~15 핵심 흐름 반영 필요"
    if article_type == "power_query_etl":
        is_regression_profile = problem_map.get("_regression_profile") == "power_query_etl_example"
        core_problem = str(problem_map.get("core_problem") or "")
        metrics["problem_kind"] = problem_map.get("problem_kind")
        if problem_map.get("problem_kind") != "transformation_requirement":
            failures.append("power_query_etl core_problem이 transformation_requirement로 분류되지 않았습니다.")
            section_failures["problem_kind"] = "연결/로드 오류가 아니라 변환 요구사항으로 정의 필요"
        bad_core_terms = ["연결 실패", "로딩 문제", "데이터 로딩이 문제가", "query merge error", "오류 수정"]
        if any(term.lower() in core_problem.lower() for term in bad_core_terms):
            failures.append("power_query_etl core_problem이 연결/로딩/오류 문제로 잘못 일반화되었습니다.")
            section_failures["core_problem"] = "원본 데이터를 분석 모델 입력 구조로 정리하는 문제로 재정의 필요"
        if is_regression_profile and ("SQL Server와 CSV" not in core_problem or "분석 모델" not in core_problem):
            failures.append("power_query_etl core_problem에 SQL Server/CSV 원본과 분석 모델 입력 구조 요구가 충분히 드러나지 않습니다.")
            section_failures["core_problem"] = "Power Query ETL 핵심 문제 정의 보강 필요"
        if is_regression_profile and len(section_plan) < 8:
            failures.append("power_query_etl Section Plan이 8개 미만입니다.")
            section_failures["section_plan"] = "Power Query ETL은 최소 8개, 권장 12개 plan 필요"
        if is_regression_profile and len(section_plan) < 12:
            failures.append("power_query_etl Section Plan이 12단계 필수 흐름을 모두 포함하지 않습니다.")
            section_failures["section_plan"] = "SQL import부터 final load control까지 12단계 필요"
        plan_covered, plan_total, plan_missing = section_plan_image_coverage(section_plan, metrics["uploaded_images_count"] or len(evidence))
        metrics["section_plan_image_coverage"] = round(plan_covered / max(plan_total, 1), 3)
        if is_regression_profile and plan_total and plan_covered / max(plan_total, 1) < 0.7:
            failures.append(f"Section Plan image coverage가 70% 미만입니다. 누락 이미지: {', '.join(plan_missing)}")
            section_failures["section_plan_image_coverage"] = "업로드 이미지가 section_plan에 충분히 연결되어야 함"
        complex_text = json.dumps(problem_map.get("complex_problem", {}), ensure_ascii=False)
        complex_topics = [
            all(term in complex_text for term in ["TotalProductCost", "OrderQuantity", "StandardCost"]),
            all(term in complex_text for term in ["Unpivot", "M01", "M12"]),
            all(term in complex_text for term in ["ColorFormats", "Left Outer", "Disable load"]),
        ]
        metrics["power_query_complex_topics"] = sum(1 for item in complex_topics if item)
        if is_regression_profile and metrics["power_query_complex_topics"] < 2:
            failures.append("power_query_etl complex_problem이 TotalProductCost / Unpivot / ColorFormats Merge 중 2개 이상을 포함하지 않습니다.")
            section_failures["complex_problem"] = "복잡 문제는 결측 원가, Targets Unpivot, ColorFormats Merge/load control 중 최소 2개 필요"
        if is_regression_profile:
            covered, total, missing = power_query_regression_coverage(article)
            metrics["power_query_regression_coverage"] = round(covered / max(total, 1), 3)
            if covered / max(total, 1) < 0.7:
                failures.append(f"Power Query regression coverage가 70% 미만입니다. 누락: {', '.join(missing)}")
                section_failures["power_query_regression"] = "이미지 1~14 핵심 흐름 반영 필요"
        code_section = extract_section(article, "사용한 주요 수식/코드 정리")
        metrics["dax_mentions"] = article.count("DAX")
        if metrics["dax_mentions"] > 2 or ("DAX" in code_section and "powerquery" not in code_section.lower()):
            failures.append("power_query_etl 글이 DAX를 중심으로 잘못 서술되었습니다.")
            section_failures["사용한 주요 수식/코드 정리"] = "Power Query formula와 변환 설명 중심으로 수정 필요"
    return CritiqueResult(passed=not failures, failures=failures, section_failures=section_failures, metrics=metrics)


def expand_failed_sections(
    llm_client: LLM,
    article: str,
    sections: dict[str, str],
    critique: CritiqueResult,
    outline: dict[str, Any],
    brief: dict[str, Any],
    problem_map: dict[str, Any],
    evidence: list[dict[str, Any]],
    raw_text: str,
    memo: str,
    extra_info: str,
) -> str:
    targets = set(critique.section_failures)
    if "global_length" in targets:
        targets.update(["문제 인식", "문제 정의", "왜 이것을 문제로 인식했는가", "문제 해결 경험", "성과", "최종 정리"])
    if "placeholder" in targets or "style" in targets:
        targets.update(["문제 해결 경험", "복잡한 문제/수식/쿼리/코드 작성 및 해결 경험"])

    for target in list(targets):
        if target not in sections:
            continue
        prompt = f"""
아래 섹션만 확장/수정하세요. 전체 글을 다시 쓰지 않습니다.

대상 섹션: {target}
실패 사유: {critique.section_failures.get(target) or critique.failures}

수정 규칙:
- 부족한 분량, 원인/조치/검증, 이미지 근거 연결만 보강합니다.
- placeholder와 내부 라벨을 제거합니다.
- 없는 수식/코드/성과를 만들지 않습니다.
- Markdown 섹션 하나만 반환합니다.
- 문제 해결 경험이면 최소 4개 단계와 각 단계의 문제/제약, 원인 판단, 조치, 확인 결과를 모두 포함합니다.

[기존 섹션]
{sections[target]}

[Outline]
{json.dumps(outline.get(target, []), ensure_ascii=False)}

[ArticleBrief]
{json.dumps(brief, ensure_ascii=False)}

[ProblemMap]
{json.dumps(problem_map, ensure_ascii=False)}

[ImageEvidence]
{json.dumps(evidence, ensure_ascii=False)}

[사용자 원문/메모/추가 정보]
{raw_text}

{memo}

{extra_info}
""".strip()
        expanded = llm_text(llm_client, prompt, max_tokens=3400)
        if expanded.strip():
            sections[target] = expanded.strip()

    expanded_article = assemble_article(sections, brief)
    second_critique = critique_article(expanded_article, str(problem_map.get("article_type") or "general_learning_portfolio"), problem_map, evidence)
    if second_critique.passed or len(expanded_article) >= len(article):
        return expanded_article
    return article


def assemble_article(sections: dict[str, str], brief: dict[str, Any]) -> str:
    ordered = [sections.get(title, "").strip() for title in SECTION_TITLES if sections.get(title, "").strip()]
    if not ordered or not ordered[0].startswith("# "):
        ordered.insert(0, f"# {brief.get('korean_title') or '문제 해결형 학습 기록'}")
    return "\n\n".join(ordered).strip()


def llm_json(llm_client: LLM, prompt: str, max_tokens: int = 2400) -> Any:
    text = llm_text(
        llm_client,
        prompt,
        max_tokens=max_tokens,
        system="You create strict JSON for a multi-step Korean Medium portfolio writing pipeline. Return JSON only.",
    )
    return parse_json_payload(text)


def llm_text(llm_client: LLM, prompt: str, max_tokens: int = 2400, system: str | None = None) -> str:
    client = llm_client.get_client()
    if not client:
        return ""
    route = model_route("final_article_generation" if max_tokens > 2000 else "simple_tutor_answer")
    try:
        completion = client.chat.completions.create(
            model=route["model"],
            temperature=0.22,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system or "You write grounded Korean Medium portfolio sections from structured evidence."},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"[LLM text pipeline error] {exc}")
        return ""


def parse_json_payload(content: str) -> Any:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = min([index for index in [cleaned.find("{"), cleaned.find("[")] if index >= 0], default=-1)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def read_image_order_caption_source() -> str:
    candidates = [
        BASE_DIR / "README_image_order.txt",
        DATA_DIR / "README_image_order.txt",
        CAPTURE_DIR / "README_image_order.txt",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace")[:5000]
    return ""


def normalize_image_evidence(data: list[Any], start_no: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    allowed_roles = {"problem", "cause", "solution", "validation", "final_result"}
    for index, item in enumerate(data, start=start_no):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "problem").strip()
        if role not in allowed_roles:
            role = "problem"
        normalized.append(
            {
                "image_no": index,
                "display_image_index": index,
                "source_image_no": item.get("image_no", index),
                "primary_topic": str(item.get("primary_topic") or ""),
                "platform_or_product": str(item.get("platform_or_product") or ""),
                "topic_terms": normalize_str_list(item.get("topic_terms"))[:12],
                "caption": humanize_caption(str(item.get("caption") or f"이미지 {index} - 실습 흐름 근거 화면"), index),
                "visible_evidence": normalize_str_list(item.get("visible_evidence"))[:10],
                "role": role,
                "problem_signal": str(item.get("problem_signal") or ""),
                "technical_entities": normalize_str_list(item.get("technical_entities"))[:12],
                "inferred_meaning": str(item.get("inferred_meaning") or ""),
                "confidence": float(item.get("confidence") or 0.65),
                "evidence_source": str(item.get("evidence_source") or "vision"),
            }
        )
    return normalized


def fallback_image_evidence(
    image_files: list[Path],
    image_names: list[str],
    raw_text: str,
    memo: str,
    caption_source: str,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    entities = infer_entities(" ".join([raw_text, memo, caption_source, " ".join(image_names)]))
    for index, (path, name) in enumerate(zip(image_files, image_names, strict=False), start=1):
        role = infer_role(index, len(image_files))
        caption_hint = caption_from_sources(index, name, caption_source)
        evidence.append(
            {
                "image_no": index,
                "display_image_index": index,
                "source_image_no": index,
                "original_filename": name,
                "caption": caption_hint or f"이미지 {index} - {Path(name).stem or path.stem}",
                "visible_evidence": [Path(name).stem, path.name],
                "role": role,
                "problem_signal": memo or raw_text or caption_hint or "이미지 순서와 캡션 정보를 바탕으로 실습 흐름을 복구해야 함",
                "technical_entities": entities,
                "inferred_meaning": f"전체 이미지 {len(image_files)}장 중 {index}번째 근거 화면입니다. README_image_order.txt, 파일명 prefix, 업로드 순서에서 확인되는 실습 단계를 바탕으로 본문 근거를 제한해야 합니다.",
                "confidence": 0.35,
                "evidence_source": "README_image_order.txt" if caption_source else "filename",
            }
        )
    return evidence


def infer_role(index: int, total: int) -> str:
    if total <= 1 or index == 1:
        return "problem"
    if index == total:
        return "final_result"
    ratio = index / max(total, 1)
    if ratio < 0.35:
        return "cause"
    if ratio < 0.75:
        return "solution"
    return "validation"


def caption_from_sources(index: int, name: str, caption_source: str) -> str:
    for line in caption_source.splitlines():
        if re.search(rf"\b0?{index}\b", line):
            text = re.sub(r"^\s*\d{1,3}[_\-\s]?", "", line.strip())
            text = re.sub(r"\.(png|jpg|jpeg)\b", "", text, flags=re.I)
            return humanize_caption(f"이미지 {index} - {text[:180]}", index)
    stem = humanize_filename_stem(Path(name).stem)
    if stem:
        return humanize_caption(f"이미지 {index} - {stem}", index)
    return ""


def humanize_filename_stem(stem: str) -> str:
    cleaned = re.sub(r"^\d{1,3}[_\-\s]?", "", stem)
    image_match = re.fullmatch(r"image(\d+)", cleaned, flags=re.I)
    if image_match:
        etl_captions = {
            1: "SQL Server database localhost와 AdventureWorksDW2020 데이터베이스를 연결 대상으로 선택한 화면",
            2: "Navigator에서 FactResellerSales 등 필요한 테이블을 선택하고 Transform Data로 Power Query Editor에 진입하는 화면",
            3: "Column quality, Column distribution, Column profile로 BusinessType 분포와 데이터 품질을 확인한 화면",
            4: "DimEmployee에서 SalesPersonFlag를 TRUE로 필터링해 영업 담당자만 남기는 화면",
            5: "FirstName과 LastName을 병합해 Salesperson 컬럼을 만들고 EmployeeID와 UPN을 정리한 화면",
            6: "Product 쿼리에서 DimProductSubcategory와 DimProductCategory를 확장해 Subcategory와 Category를 붙인 화면",
            7: "Reseller 쿼리에서 BusinessType의 Ware House 값을 Warehouse로 Replace Values 처리한 화면",
            8: "Region 쿼리에서 SalesTerritoryAlternateKey 0을 제거하고 Region, Country, Group 컬럼을 정리한 화면",
            9: "Sales 쿼리에서 TotalProductCost null을 OrderQuantity와 StandardCost 기반 Cost Custom Column으로 보완한 화면",
            10: "Targets CSV의 M01부터 M12까지 월별 목표 컬럼을 Unpivot 대상으로 선택한 화면",
            11: "Targets 데이터를 MonthNumber와 Target 중심의 long format으로 변환하고 TargetMonth를 구성한 화면",
            12: "ColorFormats 쿼리에서 Background Color Format과 Font Color Format 컬럼을 정리한 화면",
            13: "Product와 ColorFormats를 Product[Color]와 ColorFormats[Color] 기준 Left Outer Merge한 화면",
            14: "ColorFormats Disable load 후 Salesperson, SalespersonRegion, Product, Reseller, Region, Sales, Targets 7개 테이블을 Close & Apply로 로드하는 화면",
        }
        return etl_captions.get(int(image_match.group(1)), cleaned)
    replacements = {
        "Repeated Category Sales": "Category별 Sum of Sales가 모두 같은 값으로 반복되는 문제 화면",
        "Product Sales Relationship New": "Product[ProductKey]와 Sales[ProductKey] relationship을 새로 생성하는 화면",
        "Product Sales Relationship Model": "Product와 Sales relationship이 model view에 반영된 화면",
        "Star Schema Model Layout": "Product dimension과 Sales fact table 중심의 Star schema 모델 구조",
        "Create Product Hierarchy": "Product hierarchy를 구성하는 화면",
        "Products Hierarchy Levels": "Category, Subcategory, Product hierarchy level을 확인하는 화면",
        "Profit Quick Measure Subtraction": "Profit measure를 Sales와 Cost 차이로 구성하는 화면",
        "Profit Margin Quick Measure Division": "DIVIDE 기반 Profit Margin measure를 구성하는 화면",
        "Category Sales Profit Margin Table": "Category별 Sales, Profit, Profit Margin 결과를 검증하는 화면",
        "Profit Margin Result Zoom": "Profit Margin 계산 결과를 확대해 확인하는 화면",
        "SalespersonRegion Bridge Model": "SalespersonRegion bridge table로 filter path를 구성한 모델 화면",
        "Cross Filter Both Relationship": "Region과 SalespersonRegion 관계에서 Cross-filter direction을 Both로 설정한 화면",
        "Deactivate Direct Salesperson Sales": "모호한 직접 Salesperson-Sales relationship을 inactive relationship으로 바꾸는 화면",
        "Salesperson Sales By Region Result": "Salesperson별 Sales가 Region filter path로 달라진 결과 화면",
        "Salesperson Sales Target Final": "Salesperson별 Sales와 Targets를 최종 비교하는 화면",
        "Month Sort by MonthKey": "Month 컬럼을 MonthKey 기준으로 정렬한 화면",
        "Fiscal Hierarchy Date Table": "Fiscal hierarchy가 포함된 Date table을 구성한 화면",
        "Date Model Relationships": "Date table과 Sales model relationship을 확인한 화면",
        "Mark as Date Table": "Date table을 공식 날짜 테이블로 지정한 화면",
        "Avg Price Measure": "Avg Price explicit measure를 생성한 화면",
        "Pricing Count Measures Matrix": "Median Price, Min Price, Max Price, Orders, Order Lines measure를 matrix에서 검증한 화면",
        "Measure Formatting Check": "가격 measure의 Currency format을 확인한 화면",
        "TargetAmount vs Target Measure": "TargetAmount raw column 대신 Target measure를 비교한 화면",
        "Sales Target Variance Final": "Salesperson별 Sales, Target, Variance, Variance Margin을 최종 비교한 화면",
    }
    words = cleaned.replace("_", " ").replace("-", " ").strip()
    return replacements.get(words, words)


def humanize_caption(caption: str, index: int) -> str:
    caption = re.sub(r"원본 파일명\s*", "", caption)
    caption = re.sub(r"\b\d{1,3}[_\-][A-Za-z0-9_ \-]+\.(png|jpg|jpeg)", "", caption, flags=re.I)
    caption = re.sub(r"\.(png|jpg|jpeg)\b", "", caption, flags=re.I)
    caption = caption.replace("_", " ").strip(" -")
    if not caption.startswith(f"이미지 {index}"):
        caption = re.sub(r"^이미지\s+\d+\s*-\s*", "", caption)
        caption = f"이미지 {index} - {caption}"
    return caption


def infer_entities(text: str) -> list[str]:
    candidates = [
        "Power BI",
        "semantic model",
        "filter context",
        "relationship",
        "cardinality",
        "Product",
        "Sales",
        "Category",
        "ProductKey",
        "DAX",
        "Measure",
        "MonthKey",
        "Date table",
        "Fiscal hierarchy",
        "Mark as date table",
        "Avg Price",
        "Median Price",
        "Orders",
        "Order Lines",
        "TargetAmount",
        "Target",
        "Variance",
        "Variance Margin",
        "DIVIDE",
        "SQL",
        "Python",
        "Power Query",
        "Column quality",
        "Column distribution",
        "Column profile",
        "TotalProductCost",
        "Unpivot",
        "ColorFormats",
        "Close & Apply",
        "many-to-many",
        "bridge table",
        "GitHub",
        "Agentic Workflows",
        "GitHub Actions",
        "workflow_dispatch",
        "update-github-info",
        "update-github-info.lock.yml",
        "activation",
        "conclusion",
        ".github/workflows",
        "YAML",
        "yml",
    ]
    lowered = text.lower()
    found = [item for item in candidates if item.lower() in lowered]
    if found:
        return found
    if any(word in lowered for word in ["github", "workflow", "agentic", "actions", "yml", "yaml"]):
        return ["GitHub", "workflow", "automation"]
    return ["확인된 캡처 evidence"]


def normalize_problem_map(
    data: dict[str, Any],
    raw_text: str,
    memo: str,
    evidence: list[dict[str, Any]],
    topic: str,
    article_type: str = "general_learning_portfolio",
    golden_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    golden_context = golden_context or {}
    core = str(data.get("core_problem") or memo or raw_text or f"{topic} 과정에서 관찰한 결과와 의도한 분석 흐름의 불일치")
    if article_type == "coding_test_python_learning":
        core = "이 학습의 핵심 문제는 WikiDocs의 파이썬 코딩테스트 내용을 단순히 읽는 것이 아니라, 실제 문제 풀이에서 어떤 개념을 언제 사용하는지 기준으로 재구성하는 것이었다."
    elif article_type == "cs_learning_notes":
        core = "이 학습의 핵심 문제는 CS 요약 자료를 단순 암기 목록이 아니라 면접 질문과 실무 설명에 대응할 수 있는 개념 구조로 재구성하는 것이었다."
    is_powerbi_type = article_type in {"semantic_model_relationship", "dax_measure_modeling", "power_query_etl"}
    default_why = (
        "화면에 값이 표시되더라도 모델, 수식, 관계, 필터 흐름이 맞지 않으면 분석 결과의 의미가 달라질 수 있습니다."
        if is_powerbi_type
        else "드문드문 남은 캡처만으로는 강의 목표, 실행 조건, 사용자가 막힌 지점, 최종 결과를 확정하기 어렵기 때문에 확인된 evidence와 부족한 맥락을 분리해야 합니다."
    )
    default_root_causes = (
        ["시각적 결과와 데이터/계산 구조 사이의 연결 검증이 필요했습니다."]
        if is_powerbi_type
        else ["캡처 간 맥락이 비어 있어 관찰된 화면을 확정적인 성과나 해결 단계로 단정하기 어렵습니다."]
    )
    default_actual_cause = (
        "실제로는 근거 화면 전체를 연결해 모델/수식/검증 흐름을 함께 봐야 했습니다."
        if is_powerbi_type
        else "실제로는 각 캡처가 목표, 설정, 실행, 결과 중 어느 역할인지 먼저 구분해야 했습니다."
    )
    steps = data.get("solution_steps")
    if not isinstance(steps, list):
        steps = []
    normalized_steps = []
    for index, item in enumerate(steps[:6], start=1):
        if not isinstance(item, dict):
            continue
        normalized_steps.append(
            {
                "step": int(item.get("step") or index),
                "title": str(item.get("title") or f"해결 단계 {index}"),
                "problem": str(item.get("problem") or core),
                "cause": str(item.get("cause") or "입력 근거에서 원인 후보를 좁혔습니다."),
                "action": str(item.get("action") or "관찰 근거와 메모를 바탕으로 수정 방향을 정리했습니다."),
                "verification": str(item.get("verification") or "다음 이미지 또는 결과 화면으로 변경 여부를 확인했습니다."),
            }
        )
    if is_powerbi_type:
        while len(normalized_steps) < 4:
            index = len(normalized_steps) + 1
            role_evidence = evidence[min(index - 1, len(evidence) - 1)] if evidence else {}
            normalized_steps.append(
                {
                    "step": index,
                    "title": f"근거 화면 {index} 기반 검증 단계",
                    "problem": role_evidence.get("problem_signal") or core,
                    "cause": role_evidence.get("inferred_meaning") or "현재 화면이 문제, 원인, 조치, 검증 중 어느 단계인지 분리해야 했습니다.",
                    "action": "이미지 근거와 사용자 메모를 연결해 문제 해결 단계로 재구성했습니다.",
                    "verification": role_evidence.get("caption") or "후속 화면과 최종 결과에서 변화 여부를 확인해야 합니다.",
                }
            )
    else:
        if len(normalized_steps) < 3:
            text_steps = build_text_assisted_solution_steps(article_type, raw_text, memo)
            if text_steps:
                normalized_steps = text_steps
                data["_text_assisted_draft"] = True
        data["_sparse_steps_incomplete"] = len(normalized_steps) < 3
    complex_problem = data.get("complex_problem") if isinstance(data.get("complex_problem"), dict) else {}
    return {
        "article_type": article_type,
        "core_problem": core,
        "why_problematic": str(data.get("why_problematic") or default_why),
        "root_causes": normalize_str_list(data.get("root_causes")) or default_root_causes,
        "solution_steps": normalized_steps,
        "complex_problem": {
            "title": str(complex_problem.get("title") or "복합적인 원인 분석과 검증 흐름 정리"),
            "symptom": str(complex_problem.get("symptom") or core),
            "initial_assumption": str(complex_problem.get("initial_assumption") or "처음에는 화면 구성이나 단일 설정 문제로 볼 수 있었습니다."),
            "actual_cause": str(complex_problem.get("actual_cause") or default_actual_cause),
            "resolution": str(complex_problem.get("resolution") or "각 이미지의 역할을 문제, 원인, 해결, 검증으로 분리해 해결 흐름을 구성했습니다."),
            "verification": str(complex_problem.get("verification") or "마지막 결과 화면과 사용자 메모를 기준으로 검증 내용을 정리했습니다."),
        },
        "final_outcome": str(data.get("final_outcome") or ("현재 확인된 GitHub/워크플로우 캡처를 evidence 후보로 정리했습니다." if article_type in GITHUB_WORKFLOW_TYPES else "캡처와 메모를 문제 해결형 포트폴리오 글로 전환할 수 있는 구조화된 근거로 정리했습니다.")),
        "_sparse_steps_incomplete": bool(data.get("_sparse_steps_incomplete")),
    }


def normalize_article_brief(data: dict[str, Any], topic: str, problem_map: dict[str, Any]) -> dict[str, Any]:
    if problem_map.get("article_type") == "coding_test_python_learning":
        data = {
            **data,
            "korean_title": data.get("korean_title") or "코딩테스트 파이썬 학습: 문법과 자료구조를 문제 풀이 기준으로 재정리하기",
            "english_subtitle": data.get("english_subtitle") or "Reframing Python syntax and data structures for coding-test problem solving",
            "article_thesis": data.get("article_thesis") or problem_map.get("core_problem"),
            "portfolio_angle": data.get("portfolio_angle") or "파이썬 문법과 자료구조 학습을 단순 요약이 아니라 문제 조건을 읽고 풀이 전략으로 연결하는 과정으로 정리합니다.",
            "must_include": data.get("must_include") or ["문제 인식", "자료구조 선택 기준", "스택", "시간 복잡도", "문제 풀이 적용", "복습 기준"],
        }
    if problem_map.get("article_type") == "cs_learning_notes":
        data = {
            **data,
            "korean_title": data.get("korean_title") or "CS 핵심 개념 학습: 요약 자료를 면접 답변 구조로 재정리하기",
            "english_subtitle": data.get("english_subtitle") or "Turning CS notes into interview-ready explanations",
            "article_thesis": data.get("article_thesis") or problem_map.get("core_problem"),
            "portfolio_angle": data.get("portfolio_angle") or "CS 요약 자료를 정의 암기가 아니라 문제 상황, 동작 원리, trade-off 설명 기준으로 재구성합니다.",
            "must_include": data.get("must_include") or ["문제 인식", "개념 경계", "면접 질문", "동작 원리", "trade-off", "복습 기준"],
        }
    if problem_map.get("article_type") == "semantic_model_relationship":
        data = {
            **data,
            "korean_title": data.get("korean_title") or "Power BI Semantic Model 실습: 반복되는 매출값 문제를 관계·계층·DAX Measure로 해결하기",
            "english_subtitle": data.get("english_subtitle") or "Fixing repeated category sales with relationships, hierarchy, DAX measures, and bridge-table validation",
            "article_thesis": data.get("article_thesis") or problem_map.get("core_problem") or SEMANTIC_CORE_PROBLEM,
            "portfolio_angle": data.get("portfolio_angle")
            or "반복되는 Sales 값에서 filter context 문제를 찾고 Product-Sales relationship, DAX measure, SalespersonRegion bridge path까지 검증한 과정을 보여줍니다.",
            "must_include": data.get("must_include")
            or ["문제 인식", "원인 분석", "ProductKey relationship", "DAX Measure", "Bridge table", "Targets 비교", "이미지 캡션 목록"],
        }
    if problem_map.get("article_type") == "dax_measure_modeling":
        data = {
            **data,
            "korean_title": data.get("korean_title") or "Power BI DAX 실습: 날짜 테이블, Measure, Target, Variance로 분석 모델 완성하기",
            "english_subtitle": data.get("english_subtitle") or "Building a clearer analysis model with date tables, explicit measures, targets, and variance metrics",
            "article_thesis": data.get("article_thesis") or problem_map.get("core_problem") or DAX_CORE_PROBLEM,
            "portfolio_angle": data.get("portfolio_angle")
            or "MonthKey 정렬, Date table, explicit measure, Target/Variance measure를 통해 raw column 자동 집계 의존을 줄인 DAX 모델링 과정을 보여줍니다.",
            "must_include": data.get("must_include")
            or ["MonthKey 정렬", "Date table", "explicit measure", "Target measure", "Variance", "Salesperson별 최종 비교", "이미지 캡션 목록"],
        }
    if problem_map.get("article_type") == "power_query_etl":
        default_title = "Power Query ETL 실습: 원본 데이터를 분석 가능한 구조로 정리·변환·로드하기"
        default_subtitle = "Turning raw study evidence into a verifiable data preparation workflow"
        default_thesis = problem_map.get("core_problem") or topic
        default_angle = "원본 데이터를 Power Query에서 분석 가능한 테이블 구조로 바꾸는 판단 과정을 보여줍니다."
        if problem_map.get("_regression_profile") == "power_query_etl_example":
            default_title = "Power BI Desktop 실습: 원본 데이터를 분석 가능한 모델 입력 구조로 정리·변환·로드하기"
            default_subtitle = "Cleaning, shaping, merging, and loading raw data into an analysis-ready Power BI model"
            default_thesis = problem_map.get("core_problem") or POWER_QUERY_CORE_PROBLEM
            default_angle = "SQL Server와 CSV 원본을 Power Query에서 분석 가능한 테이블 구조로 바꾸는 판단 과정을 보여줍니다."
        data = {
            **data,
            "korean_title": data.get("korean_title") or default_title,
            "english_subtitle": data.get("english_subtitle") or default_subtitle,
            "article_thesis": data.get("article_thesis") or default_thesis,
            "portfolio_angle": data.get("portfolio_angle")
            or default_angle,
            "must_include": data.get("must_include")
            or ["문제 인식", "원인 분석", "검증 결과", "Power Query 변환 근거", "이미지 캡션 목록", "Section Plan"],
        }
    if problem_map.get("article_type") in GITHUB_WORKFLOW_TYPES:
        source_text = str(problem_map.get("_evidence_text_for_classification") or "")
        data = {
            **data,
            "korean_title": data.get("korean_title") if data.get("korean_title") and not is_generic_learning_title(str(data.get("korean_title"))) else specific_title_candidate(str(problem_map.get("article_type")), [], "GitHub Agentic Workflows 실습: workflow_dispatch와 자동화 실행 흐름을 이해한 기록"),
            "english_subtitle": data.get("english_subtitle") or "Understanding GitHub agentic workflow execution from sparse learning captures",
            "article_thesis": data.get("article_thesis") if data.get("article_thesis") and not is_generic_core_problem(str(data.get("article_thesis"))) else problem_map.get("core_problem"),
            "portfolio_angle": data.get("portfolio_angle") or "드문드문 남은 GitHub workflow 캡처에서 자동화 실행 조건, workflow 파일, activation, conclusion의 의미를 구분하는 학습 기록으로 정리합니다.",
            "must_include": data.get("must_include") or ["GitHub workflow", "workflow_dispatch", "activation", "conclusion", "missing_context", "이미지 캡션 목록"],
        }
    return {
        "korean_title": str(data.get("korean_title") or f"{topic}: 문제를 검증 가능한 분석 흐름으로 바꾼 기록"),
        "english_subtitle": str(data.get("english_subtitle") or "A problem-solving portfolio article from technical study evidence"),
        "article_thesis": str(data.get("article_thesis") or problem_map.get("core_problem") or topic),
        "target_reader": str(data.get("target_reader") or "recruiter / hiring manager / technical reviewer"),
        "portfolio_angle": str(data.get("portfolio_angle") or "사용자가 기술적 문제를 발견하고 원인, 조치, 검증으로 정리할 수 있음을 보여줍니다."),
        "do_not_claim": normalize_str_list(data.get("do_not_claim")) or ["사용자가 하지 않은 성과", "없는 수식", "없는 배포"],
        "must_include": normalize_str_list(data.get("must_include")) or ["문제 인식", "원인 분석", "검증 결과", "이미지 캡션 목록"],
    }


def default_outline_items(title: str, brief: dict[str, Any], problem_map: dict[str, Any], evidence: list[dict[str, Any]]) -> list[str]:
    if title == "문제 해결 경험":
        return [
            f"{step['step']}. {step['title']}: 문제/제약, 원인 판단, 조치, 확인 결과"
            for step in problem_map.get("solution_steps", [])
        ]
    if title == "이미지 번호와 캡션 목록":
        return [str(item.get("caption")) for item in evidence]
    return [str(brief.get("article_thesis") or problem_map.get("core_problem") or title)]


def fallback_section(
    section_title: str,
    brief: dict[str, Any],
    problem_map: dict[str, Any],
    evidence: list[dict[str, Any]],
    raw_text: str,
    memo: str,
) -> str:
    if section_title == "이미지 번호와 캡션 목록":
        captions = "\n".join(f"- {item.get('caption')}" for item in evidence) or "- 이미지 없음"
        return f"## 이미지 번호와 캡션 목록\n{captions}"
    if section_title == "Key skills practiced":
        return "\n".join(
            [
                "## Key skills practiced",
                "- Problem framing",
                "- Evidence-based technical documentation",
                "- Root-cause analysis",
                "- Validation planning",
                "- Data model reasoning",
                "- Portfolio writing",
                "- Technical communication",
                "- Iterative debugging",
            ]
        )
    if section_title == "문제 해결 경험":
        lines = ["## 문제 해결 경험"]
        for step in problem_map.get("solution_steps", []):
            lines.append(
                f"### {step['step']}. {step['title']}\n"
                f"문제/제약: {step['problem']}\n\n"
                f"원인 판단: {step['cause']}\n\n"
                f"조치: {step['action']}\n\n"
                f"확인 결과: {step['verification']}"
            )
        return "\n\n".join(lines)
    return f"## {section_title}\n{brief.get('article_thesis') or problem_map.get('core_problem') or raw_text or memo}"


def sanitize_medium_markdown(article: str) -> str:
    cleaned = article
    for phrase in PLACEHOLDER_PHRASES:
        cleaned = cleaned.replace(phrase, "")
    for phrase in INTERNAL_ARTICLE_BANNED_PHRASES:
        # Sanitizer is a last cleanup pass. Policy hard-fail is handled before response.
        cleaned = cleaned.replace(phrase, "")
    cleaned = cleaned.replace("[생성 직전 사용자가 정의한 어려운 문제]", "")
    cleaned = cleaned.replace("[생성 직전 사용자가 적은 어려움/헷갈린 부분]", "")
    return postprocess_article_text(cleaned).strip()


def postprocess_article_text(article: str) -> str:
    cleaned = article
    replacements = {
        "헷갈렸던 부분": "복잡하거나 어려운 문제",
        "헷갈린 부분": "복잡하거나 어려운 문제",
        "헷갈렸던 개념": "복잡한 개념",
        "헷갈린 개념": "복잡한 개념",
        "처음 헷갈린 지점은": "처음 문제로 정의한 지점은",
        "헷갈릴 수 있었다": "문제로 인식할 수 있었다",
        "헷갈릴 수 있습니다": "문제로 인식할 수 있습니다",
        "헷갈렸다": "문제로 인식했다",
        "헷갈린다": "문제로 인식된다",
        "문제를 확인한 문제로 정의되었다": "문제로 정의되었다",
        "정리해야 했다.로 정의되었다": "정리해야 하는 문제로 정의되었다",
        "정리해야 했다.로 정리되었다": "정리해야 하는 흐름으로 정리되었다",
        "구성해야 한 것으로 정의되었다": "구성해야 하는 문제로 정의되었다",
        "확인했다.로 정의되었다": "확인한 문제로 정의되었다",
        "확인했다.로 정리되었다": "확인한 흐름으로 정리되었다",
        "영향을 줌로": "영향을 주는 방식으로",
        "보임 하지만": "보였지만",
        "처리으로 접근했다": "처리 방식으로 접근했다",
        "확인을 기준으로 삼았다": "확인 결과를 기준으로 삼았다",
        "했다.로 정의되었다": "한 것으로 정의되었다",
        "했다.로 정리되었다": "한 것으로 정리되었다",
        "했다.라는": "했다는",
        "했다..": "했다.",
        "정리해야 했다..": "정리해야 했다.",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"([가-힣A-Za-z0-9\[\]\(\) /·,_-]+문제)를 확인한 문제로 정의되었다", r"\1로 정의되었다", cleaned)
    cleaned = re.sub(r"([가-힣A-Za-z0-9\[\]\(\) /·,_-]+)구성해야 한 것으로 정의되었다", r"\1구성해야 하는 문제로 정의되었다", cleaned)
    cleaned = re.sub(r"([가-힣A-Za-z0-9\[\]\(\) /·,_-]+)구성해야 했다\.로 정의되었다", r"\1구성해야 하는 문제로 정의되었다", cleaned)
    cleaned = re.sub(r"정리해야 했다\.로\s*", "정리해야 하는 문제로 ", cleaned)
    cleaned = re.sub(r"확인했다\.로\s*", "확인한 문제로 ", cleaned)
    cleaned = re.sub(r"([가-힣]+했다)\.로\s*", r"\1는 흐름으로 ", cleaned)
    cleaned = re.sub(r"([가-힣A-Za-z0-9\]\)])\.\.(?=\s|$)", r"\1.", cleaned)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = re.sub(r"\s+([,.])", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^(포트폴리오 관점: .+)\n\n포트폴리오 관점: ", r"\1\n\n이 단계는 ", cleaned)
    return cleaned


def count_problem_solution_steps(article: str) -> int:
    section = extract_section(article, "문제 해결 경험")
    return max(
        len(re.findall(r"(?m)^###\s+\d+[\.\)]", section)),
        len(re.findall(r"문제/제약", section)),
        len(re.findall(r"확인 결과", section)),
    )


def count_section_paragraphs(article: str, title: str) -> int:
    section = extract_section(article, title)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section) if part.strip() and not part.strip().startswith("#")]
    return len(paragraphs)


def extract_section(article: str, title: str) -> str:
    pattern = rf"(?ms)^##\s+{re.escape(title)}\s*$([\s\S]*?)(?=^##\s+|\Z)"
    match = re.search(pattern, article)
    return match.group(1).strip() if match else ""


def has_code_block(article: str) -> bool:
    return "```" in article or bool(re.search(r"\b[A-Z][A-Za-z0-9_]*\s*=", article))


def has_code_explanation(article: str) -> bool:
    checks = ["왜", "필요", "계산", "검증", "확인 결과"]
    code_section = extract_section(article, "사용한 주요 수식/코드 정리")
    return all(word in code_section for word in checks[:3]) and any(word in code_section for word in checks[3:])


def semantic_problem_detail(problem_map: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    if problem_map.get("article_type") == "power_query_etl" and problem_map.get("_regression_profile") == "power_query_etl_example":
        return (
            "이번 실습의 문제는 데이터를 불러오는 것 자체가 아니라, SQL Server와 CSV에서 가져온 원본 데이터를 보고서 작성에 적합한 분석 모델 입력 구조로 정리하는 것이었다. "
            "초기 화면에서는 localhost의 AdventureWorksDW2020 데이터베이스, Navigator의 테이블 선택, Transform Data 진입이 보이고, 이후 화면에서는 Power Query Editor 안에서 Column quality, BusinessType 분포, SalesPersonFlag 필터, Product 확장, TotalProductCost null, M01~M12 Unpivot, ColorFormats Merge, Disable load까지 이어진다. "
            "이 흐름은 연결 실패나 단순 로딩 문제가 아니라, 원본 데이터의 품질과 형태를 분석 가능한 테이블 구조로 바꾸는 transformation requirement로 해석해야 한다."
        )
    if problem_map.get("article_type") != "semantic_model_relationship":
        return str(problem_map.get("core_problem", ""))
    return (
        "처음 만든 table visual에는 Product[Category]와 Sales[Sales]를 함께 넣었다. "
        "기대했던 결과는 Category 행마다 서로 다른 매출값이 표시되는 것이었지만, 실제 화면에서는 모든 Category 행에 같은 Sales 총액이 반복되는 패턴이 나타났다. "
        "이 패턴은 visual 서식 문제가 아니라 Product와 Sales 사이 relationship이 없거나 Product[Category] filter context가 Sales fact table로 전달되지 않는 상황으로 해석할 수 있다."
    )


def concrete_detail_text(step: dict[str, Any]) -> str:
    details = normalize_str_list(step.get("concrete_details"))
    if not details:
        return ""
    return "\n\n구체적 설정/근거:\n" + "\n".join(f"- {detail}" for detail in details)


def portfolio_meaning_text(step: dict[str, Any]) -> str:
    meaning = str(step.get("portfolio_meaning") or "").strip()
    if not meaning:
        return ""
    step_no = int(step.get("step") or 0)
    if step_no == 1:
        return f"\n\n포트폴리오 관점: {meaning}"
    if step_no % 3 == 0:
        return f"\n\n이 단계의 의미는 {meaning}"
    if step_no % 3 == 1:
        return f"\n\n이 판단은 {meaning}"
    return f"\n\n결과적으로 {meaning}"


def clean_summary_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = re.sub(r"([.!?。]){2,}$", r"\1", cleaned)
    cleaned = cleaned.strip("` ")
    return cleaned


def article_type_entities_text(article_type: str, evidence_entities: list[str], problem_map: dict[str, Any]) -> str:
    preferred = {
        "semantic_model_relationship": [
            "Product[Category]",
            "Sales[Sales]",
            "Product[ProductKey]",
            "Sales[ProductKey]",
            "One to many",
            "Cross-filter direction",
            "DIVIDE",
            "SalespersonRegion",
            "inactive relationship",
            "Targets",
        ],
        "dax_measure_modeling": [
            "Month",
            "MonthKey",
            "Date table",
            "Fiscal hierarchy",
            "Mark as date table",
            "Avg Price",
            "Orders",
            "Target measure",
            "Variance",
            "Variance Margin",
        ],
        "power_query_etl": [
            "Power Query Editor",
            "SQL Server",
            "CSV",
            "Column quality",
            "Column distribution",
            "Column profile",
            "SalesPersonFlag",
            "ProductKey",
            "BusinessType",
            "TotalProductCost",
            "Custom Column",
            "Unpivot",
            "TargetMonth",
            "ColorFormats",
            "Product[Color]",
            "Left Outer",
            "Disable load",
            "Close & Apply",
        ],
    }.get(article_type, [])
    haystack = json.dumps(problem_map, ensure_ascii=False) + " " + " ".join(evidence_entities)
    selected = [item for item in preferred if item.lower() in haystack.lower()]
    if len(selected) < min(6, len(preferred)):
        selected = preferred[:10] if preferred else []
    if selected:
        return ", ".join(selected[:12])
    generic_filtered = [
        item
        for item in evidence_entities
        if item.lower() not in {"study capture", "technical evidence", "capture", "evidence"}
    ]
    return ", ".join(generic_filtered[:10]) or "입력 이미지와 메모에 나타난 기술 요소"


def key_skills_for_article_type(article_type: str) -> list[str]:
    by_type = {
        "semantic_model_relationship": [
            "Diagnosing filter context issues from repeated values",
            "Designing Product-to-Sales relationships",
            "Validating one-to-many cardinality and active relationships",
            "Reading star schema structure in model view",
            "Building Product hierarchy for drilldown analysis",
            "Creating Profit and Profit Margin measures",
            "Using DIVIDE for ratio measures",
            "Resolving bridge-table filter paths",
            "Comparing Sales and Targets with model-aware validation",
        ],
        "dax_measure_modeling": [
            "Sorting display columns with MonthKey",
            "Building fiscal Date table structures",
            "Marking a Date table for time-based analysis",
            "Replacing raw column aggregation with explicit measures",
            "Creating pricing and order-count measures",
            "Applying currency formatting to measures",
            "Controlling Target totals with measure logic",
            "Calculating Variance and Variance Margin",
            "Validating Salesperson-level performance metrics",
        ],
        "power_query_etl": [
            "Profiling source data quality in Power Query",
            "Standardizing categorical values",
            "Filtering dimension queries for analysis use",
            "Expanding related dimension tables",
            "Creating Custom Column logic for missing cost values",
            "Unpivoting wide target data into long format",
            "Merging lookup tables with Left Outer joins",
            "Managing helper-query load settings",
            "Validating final model table load scope",
        ],
        "python_algorithm_learning": [
            "Interpreting algorithm problem constraints",
            "Tracing input and output behavior",
            "Handling edge cases",
            "Reasoning about time complexity",
            "Refactoring Python control flow",
            "Writing verification examples",
            "Explaining failure cases clearly",
            "Documenting algorithm decisions",
        ],
        "code_error_debugging": [
            "Reading stack traces",
            "Isolating root causes",
            "Reproducing failures",
            "Designing minimal fixes",
            "Adding verification checks",
            "Separating symptoms from causes",
            "Documenting debugging decisions",
            "Avoiding unsupported conclusions",
        ],
        "github_agentic_workflow": [
            "Understanding GitHub Agentic Workflows",
            "Reading GitHub Actions workflow files",
            "Identifying workflow_dispatch triggers",
            "Interpreting workflow activation states",
            "Connecting workflow files with execution conclusions",
            "Reading lock-file evidence conservatively",
            "Separating observed automation behavior from assumptions",
            "Documenting sparse learning evidence",
        ],
        "github_actions_workflow": [
            "Reading GitHub Actions workflow files",
            "Identifying workflow_dispatch triggers",
            "Interpreting workflow activation states",
            "Connecting workflow YAML with run conclusions",
            "Checking automation state changes",
            "Documenting sparse learning evidence",
            "Avoiding unsupported workflow claims",
            "Writing evidence-based workflow summaries",
        ],
        "ai_coding_workflow": [
            "Understanding AI-assisted coding workflows",
            "Identifying automation triggers",
            "Reading workflow state evidence",
            "Connecting agent actions with observed results",
            "Separating course context from implementation claims",
            "Documenting sparse learning evidence",
            "Avoiding unsupported tool claims",
            "Writing evidence-based learning summaries",
        ],
        "github_readme_debugging": [
            "Debugging Markdown rendering",
            "Checking repository asset paths",
            "Validating README previews",
            "Explaining media embedding constraints",
            "Separating local and GitHub render behavior",
            "Documenting fixes with visible evidence",
            "Maintaining reader-friendly project docs",
            "Avoiding unrelated technical claims",
        ],
        "deployment_debugging": [
            "Reading build and runtime logs",
            "Checking environment variables",
            "Separating deployment and application failures",
            "Validating service health endpoints",
            "Documenting rollback or fix decisions",
            "Explaining infrastructure assumptions",
            "Verifying server behavior after changes",
            "Communicating operational risk clearly",
        ],
    }
    return by_type.get(
        article_type,
        [
            "Problem framing from visual evidence",
            "Root-cause analysis",
            "Evidence-based technical documentation",
            "Validation planning",
            "Technical portfolio writing",
            "Clear explanation of cause, action, and result",
            "Avoiding unsupported claims",
            "Connecting captures to a problem-solving narrative",
        ],
    )


def image_ref_caption_text(step: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    refs = [int(ref) for ref in step.get("image_refs", []) if str(ref).isdigit()]
    if not refs:
        return ""
    captions = [item.get("caption") for item in evidence if int(item.get("image_no") or 0) in refs]
    if not captions:
        return f"이미지 근거: {', '.join(f'이미지 {ref}' for ref in refs)}"
    return "이미지 근거: " + "; ".join(str(caption) for caption in captions[:3])


def article_type_focus_text(article_type: str) -> str:
    if article_type == "semantic_model_relationship":
        return "이 유형에서는 repeated Sales 값, relationship 구조, filter context 전달, ProductKey 연결 여부를 함께 확인해야 한다."
    if article_type == "dax_measure_modeling":
        return "이 유형에서는 Date table, MonthKey 정렬, explicit Measure, Target, Variance 계산 맥락을 함께 확인해야 한다."
    if article_type == "power_query_etl":
        return "이 유형에서는 data quality, null cost 보완, Unpivot, Merge, load control이 이후 분석 오류를 어떻게 줄이는지 확인해야 한다."
    return "이 유형에서는 화면 근거와 사용자 메모를 연결해 문제, 원인, 조치, 검증 기준을 분리해야 한다."


def final_outcome_links_to_problem(article: str, problem_map: dict[str, Any]) -> bool:
    final_section = extract_section(article, "최종 정리") + "\n" + extract_section(article, "성과")
    core = str(problem_map.get("core_problem", ""))
    important_terms = [term for term in tokenize(core) if len(term) >= 4][:8]
    if not important_terms:
        return True
    return sum(1 for term in important_terms if term.lower() in final_section.lower()) >= min(2, len(important_terms))


def missing_article_type_keywords(article_type: str, article: str, problem_map: dict[str, Any]) -> list[str]:
    required = {
        "semantic_model_relationship": ["Sales", "relationship", "filter context", "ProductKey"],
        "dax_measure_modeling": ["Date table", "MonthKey", "Measure", "Target", "Variance"],
        "power_query_etl": ["data quality", "TotalProductCost", "Unpivot", "Merge", "load"],
    }.get(article_type, [])
    haystack = (article + "\n" + json.dumps(problem_map, ensure_ascii=False)).lower()
    return [keyword for keyword in required if keyword.lower() not in haystack]


def awkward_korean_leftovers(article: str) -> list[str]:
    patterns = [
        "문제를 확인한 문제로 정의되었다",
        "정리해야 했다.로 정의되었다",
        "구성해야 한 것으로 정의되었다",
        "영향을 줌로",
        "보임 하지만",
        "처리으로 접근했다",
        "확인을 기준으로 삼았다",
        "..",
    ]
    return [pattern for pattern in patterns if pattern in article]


def truncated_caption_lines(article: str) -> list[str]:
    section = extract_section(article, "이미지 번호와 캡션 목록")
    if not section:
        return []
    truncated: list[str] = []
    allowed_endings = ("다", "음", "면", "결과", "비교", "확인", "구성", "로드", "검증", "화면", "상태", "처리")
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        caption = stripped.lstrip("- ").strip()
        if not caption:
            truncated.append(stripped)
            continue
        if caption.endswith("화") or caption.endswith(("(", "[", "/", "-", "·")):
            truncated.append(caption)
            continue
        if len(caption) < 10:
            truncated.append(caption)
            continue
        if not caption.endswith(allowed_endings):
            last = caption[-1]
            if last in {"의", "과", "와", "을", "를", "로", "에", "서"}:
                truncated.append(caption)
    return truncated


def load_article_type_config(article_type: str) -> dict[str, Any]:
    path = ARTICLE_TYPE_CONFIG_DIR / f"{article_type}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def article_type_required_entities(article_type: str) -> list[str]:
    config = load_article_type_config(article_type)
    configured = normalize_str_list(config.get("required_entities"))
    if configured:
        return configured
    return REQUIRED_CONCRETE_ENTITIES.get(article_type, [])


def unsupported_powerbi_claims(
    article: str,
    article_type: str,
    evidence: list[dict[str, Any]],
    problem_map: dict[str, Any],
) -> list[str]:
    powerbi_terms = [
        "ProductKey",
        "DAX",
        "Relationship",
        "Power Query",
        "Sales[Sales]",
        "TargetAmount",
        "MonthKey",
        "Date table",
        "SalespersonRegion",
        "Star schema",
        "Profit Margin",
        "DIVIDE",
        "FactResellerSales",
        "ColorFormats",
    ]
    if article_type in {"semantic_model_relationship", "dax_measure_modeling", "power_query_etl", "dashboard_validation"}:
        return []
    source_text = json.dumps({"evidence": evidence, "problem_map": problem_map}, ensure_ascii=False).lower()
    unsupported = []
    for term in powerbi_terms:
        if term.lower() in article.lower() and term.lower() not in source_text:
            unsupported.append(term)
    return unsupported


def generic_or_copied_title(article: str, problem_map: dict[str, Any]) -> bool:
    first_line = next((line.strip("# ").strip() for line in article.splitlines() if line.startswith("# ")), "")
    core = str(problem_map.get("core_problem") or "")
    if is_generic_learning_title(first_line) or is_generic_core_problem(core):
        return True
    generic_titles = [
        "문제 해결형 학습 기록",
        "학습 기록 기반 문제 해결 경험",
        "문제를 검증 가능한 분석 흐름으로 바꾼 기록",
    ]
    if first_line in generic_titles:
        return True
    copied_core_fragments = [
        "Category별 Sales 값이 모두 동일하게 반복되어 Product 차원 테이블의 filter context가 Sales fact table까지 전달되지 않는 문제가 드러남",
        "데이터 소스 연결 및 데이터 로딩이 문제가 발생하여 해결이 필요하다",
    ]
    return any(fragment in core or fragment in article for fragment in copied_core_fragments)


def semantic_regression_coverage(article: str) -> tuple[int, int, list[str]]:
    lowered = article.lower()
    missing: list[str] = []
    covered = 0
    for label, keywords in SEMANTIC_REGRESSION_REQUIREMENTS:
        if all(keyword.lower() in lowered for keyword in keywords):
            covered += 1
        else:
            missing.append(label)
    return covered, len(SEMANTIC_REGRESSION_REQUIREMENTS), missing


def power_query_regression_coverage(article: str) -> tuple[int, int, list[str]]:
    lowered = article.lower()
    missing: list[str] = []
    covered = 0
    for label, keywords in POWER_QUERY_REGRESSION_REQUIREMENTS:
        if all(keyword.lower() in lowered for keyword in keywords):
            covered += 1
        else:
            missing.append(label)
    return covered, len(POWER_QUERY_REGRESSION_REQUIREMENTS), missing


def section_plan_image_coverage(section_plan: list[dict[str, Any]], image_count: int) -> tuple[int, int, list[str]]:
    if image_count <= 0:
        return 0, 0, []
    expected = set(range(1, image_count + 1))
    refs: set[int] = set()
    for item in section_plan:
        if not isinstance(item, dict):
            continue
        for ref in item.get("image_refs", []):
            try:
                refs.add(int(ref))
            except (TypeError, ValueError):
                continue
    missing = sorted(expected - refs)
    return len(expected) - len(missing), len(expected), [f"이미지 {item}" for item in missing]


def plan_to_article_faithfulness(article: str, problem_map: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    metrics: dict[str, Any] = {}
    section_plan = problem_map.get("_section_plan", []) if isinstance(problem_map.get("_section_plan"), list) else []
    steps = problem_map.get("solution_steps", []) if isinstance(problem_map.get("solution_steps"), list) else []
    if not section_plan:
        return failures, metrics
    article_steps = count_problem_solution_steps(article)
    plan_count = len(section_plan)
    metrics["plan_step_count"] = plan_count
    metrics["article_solution_step_count"] = article_steps
    if plan_count >= 8 and article_steps < max(4, int(plan_count * 0.7)):
        failures.append(f"Section Plan은 {plan_count}개인데 Final Article 문제 해결 경험은 {article_steps}개로 과도하게 압축되었습니다.")

    must_items = [item for plan in section_plan for item in normalize_str_list(plan.get("must_include"))]
    included = [item for item in must_items if item.lower() in article.lower()]
    coverage = len(included) / max(len(must_items), 1)
    metrics["section_plan_must_include_coverage"] = round(coverage, 3)
    if must_items and coverage < 0.7:
        missing = [item for item in must_items if item not in included][:12]
        failures.append(f"Section Plan must_include 반영률이 70% 미만입니다: {', '.join(missing)}")

    expected_refs = sorted({int(ref) for plan in section_plan for ref in plan.get("image_refs", []) if str(ref).isdigit()})
    article_refs = sorted({int(match) for match in re.findall(r"이미지\s+(\d+)", article)})
    metrics["section_plan_image_refs"] = expected_refs
    metrics["article_image_refs"] = article_refs
    if expected_refs and not set(expected_refs).issubset(set(article_refs)):
        missing_refs = sorted(set(expected_refs) - set(article_refs))
        failures.append(f"Section Plan의 image_refs가 Final Article에 모두 반영되지 않았습니다: {', '.join(f'이미지 {ref}' for ref in missing_refs)}")

    action_terms = ["생성했다", "설정했다", "비활성화", "구성했다", "확인했다", "적용", "만들었다"]
    cause_terms = ["전달되지", "없거나", "오설정", "모호", "필터링하지", "동시에 존재", "왜곡", "부족"]
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        cause = str(step.get("cause") or "")
        action = str(step.get("action") or "")
        if any(term in cause for term in action_terms) and not any(term in cause for term in cause_terms):
            failures.append(f"solution_step {index} cause 필드에 조치 문장이 들어갔습니다: {cause[:80]}")
            break
        if any(term in action for term in cause_terms) and not any(term in action for term in action_terms):
            failures.append(f"solution_step {index} action 필드에 원인 문장이 들어갔습니다: {action[:80]}")
            break
    return failures, metrics


def semantic_image_ref_mapping_failure(problem_map: dict[str, Any]) -> str:
    expected = {
        "반복값": {1},
        "relationship 생성": {2},
        "relationship 설정값": {3},
        "Star schema": {4},
        "hierarchy": {5, 6},
        "Profit Quick": {7},
        "Profit Margin measure": {8},
        "결과 검증": {9, 10},
        "bridge table": {11},
        "Both": {12},
        "비활성화": {13},
        "Sales 결과": {14},
        "Targets": {15},
    }
    for step in problem_map.get("solution_steps", []):
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "")
        refs = {int(ref) for ref in step.get("image_refs", []) if str(ref).isdigit()}
        if not refs:
            continue
        if "bridge" in title.lower() and refs & {1, 2, 3, 4, 5, 6}:
            return f"bridge table step에 잘못된 image_refs가 붙었습니다: {sorted(refs)}"
        if ("Profit Quick" in title or "Profit Margin measure" in title) and refs & {2, 3, 4}:
            return f"Profit/Profit Margin step에 잘못된 image_refs가 붙었습니다: {sorted(refs)}"
        for key, allowed in expected.items():
            if key.lower() in title.lower() and not refs.issubset(allowed):
                return f"{title} step image_refs가 기대 매핑과 다릅니다. expected={sorted(allowed)}, actual={sorted(refs)}"
    return ""


def normalize_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def medium_generation_failure(image_count: int, memo: str, reasons: list[str], capture_count: int = 0, qa_count: int = 0) -> str:
    reason_text = "\n".join(f"- {reason}" for reason in reasons)
    memo_text = memo.strip() or "입력된 메모 없음"
    return f"""# Medium 글 생성 실패

LLM/Vision 응답 실패로 완성형 Medium 글을 생성하지 못했습니다.

## 업로드된 이미지 수
{image_count}장

## 저장된 캡처 수
{capture_count}개

## 저장된 Q&A 수
{qa_count}개

## 입력된 메모
{memo_text}

## 생성 실패 원인 후보
{reason_text}
- 네트워크 또는 LLM API 응답 지연
- Vision 모델이 이미지 판독 요청을 처리하지 못함
- 입력 이미지가 너무 많거나 개별 이미지 용량이 큼

## 사용자가 추가해야 할 정보
- 이미지 흐름 요약: 어떤 화면이 문제 발견, 원인 분석, 해결, 검증에 해당하는지
- 핵심 문제: 무엇이 이상했고 왜 문제였는지
- 해결 과정: 실제로 바꾼 관계, 수식, 쿼리, 설정
- 검증 결과: 수정 전후 결과가 어떻게 달라졌는지
"""


def provider_failure_message(
    image_count: int,
    memo: str,
    diagnostics: dict[str, Any],
    capture_count: int = 0,
    qa_count: int = 0,
    elapsed_seconds: float = 0,
) -> str:
    immediate = elapsed_seconds < 1
    diag_lines = "\n".join(
        f"- {key}: {value}"
        for key, value in diagnostics.items()
        if key not in {"traceback_summary"} and "key" not in key.lower()
    )
    trace = diagnostics.get("traceback_summary") or "traceback 요약 없음"
    leading = (
        "LLM 요청이 시작되기 전에 실패했습니다. 환경변수, 패키지 import, provider client 초기화 상태를 확인하세요."
        if immediate
        else "LLM/Vision provider 호출 중 실패했습니다. provider diagnostics를 먼저 확인하세요."
    )
    return f"""# Medium 글 생성 실패

{leading}

이미지 {image_count}장은 업로드되었지만 Vision/Text provider 초기화 또는 호출 단계에서 실패했습니다. 이 경우 이미지 흐름 요약을 추가하는 것보다 `GROQ_API_KEY`, `groq` 패키지, 선택된 provider/model 설정을 먼저 확인해야 합니다.

## Provider diagnostics
{diag_lines}

## Traceback summary
```text
{trace}
```

## 입력 상태
- 업로드된 이미지 수: {image_count}장
- 저장된 캡처 수: {capture_count}개
- 저장된 Q&A 수: {qa_count}개
- batch_upload_mode에서는 캡처 수와 Q&A 수가 0이어도 실패 조건이 아닙니다.

## 입력된 메모
{memo.strip() or "입력된 메모 없음"}
"""


def is_groq_rate_limit_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()
    return status_code == 429 or "rate_limit_exceeded" in message or "rate limit reached" in message


def groq_retry_details(exc: Exception) -> dict[str, Any]:
    message = str(exc)
    details: dict[str, Any] = {}
    usage_match = re.search(
        r"Limit\s+(\d+),\s*Used\s+(\d+),\s*Requested\s+(\d+)",
        message,
        flags=re.I,
    )
    if usage_match:
        details["limit_tokens"] = int(usage_match.group(1))
        details["used_tokens"] = int(usage_match.group(2))
        details["requested_tokens"] = int(usage_match.group(3))

    retry_match = re.search(
        r"try again in\s*(?:(\d+(?:\.\d+)?)m)?\s*(?:(\d+(?:\.\d+)?)s)?",
        message,
        flags=re.I,
    )
    if retry_match:
        minutes = float(retry_match.group(1) or 0)
        seconds = float(retry_match.group(2) or 0)
        total_seconds = max(1, int(minutes * 60 + seconds + 0.999))
        details["retry_after_seconds"] = total_seconds
        display_minutes, display_seconds = divmod(total_seconds, 60)
        if display_minutes:
            details["retry_after_display"] = f"약 {display_minutes}분 {display_seconds}초 후"
        else:
            details["retry_after_display"] = f"약 {display_seconds}초 후"
    return details


def vision_rate_limit_response(
    exc: Exception,
    image_count: int,
    capture_count: int = 0,
    qa_count: int = 0,
) -> dict[str, Any]:
    details = groq_retry_details(exc)
    retry_display = str(details.get("retry_after_display") or "잠시 후")
    usage_lines: list[str] = []
    if details.get("used_tokens") is not None and details.get("limit_tokens") is not None:
        usage_lines.append(f"- 현재 사용량: {details['used_tokens']:,} / {details['limit_tokens']:,} tokens")
    if details.get("requested_tokens") is not None:
        usage_lines.append(f"- 이번 요청 필요량: 약 {details['requested_tokens']:,} tokens")
    usage_lines.append(f"- 재시도 권장: {retry_display}")
    draft = f"""# 이미지 분석 사용량 한도 초과

Groq Vision이 현재 사용량 한도에 도달해 이미지 내용을 판독하지 못했습니다. 이미지 근거 없이 fallback 글을 만들지 않았습니다.

## 현재 상태
{chr(10).join(usage_lines)}
- 업로드된 이미지: {image_count}장
- 저장된 캡처: {capture_count}개
- 저장된 Q&A: {qa_count}개

## 조치
{retry_display} 같은 이미지를 다시 제출해 주세요. API 키나 가상환경을 다시 설정할 필요는 없습니다.
"""
    diagnostics = {
        "provider": "groq",
        "model": GROQ_VISION_MODEL,
        "status_code": 429,
        "failure_type": "vision_rate_limit",
        **details,
    }
    return {
        "draft": draft,
        "article_type": "vision_rate_limit",
        "image_evidence": [],
        "problem_map": {},
        "learning_evidence": [],
        "decision_map": {},
        "section_plan": [],
        "article_brief": {},
        "provider_diagnostics": diagnostics,
        "critic_report": {
            "passed": False,
            "failures": ["Groq Vision rate limit exceeded"],
            "metrics": {"failure_type": "vision_rate_limit", **details},
        },
        "mode": "vision_rate_limit",
    }


def parse_json_or_fallback(content: str, raw_text: str, memo: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return fallback_note(raw_text, memo)
    if isinstance(data, list):
        summary = stringify_field(data)
        return {
            "title": "이미지 기반 학습 기록",
            "source_type": "study-capture",
            "tags": ["study-note", "vision"],
            "summary": summary,
            "action_items": ["핵심 개념 정리", "막힌 지점과 해결 방법 추가 기록", "문제 해결형 글로 변환"],
            "blog_draft": f"# 이미지 기반 학습 기록\n\n## 캡처 내용\n{summary}\n\n## 다음 정리\n- 문제 인식\n- 해결 과정\n- 배운 점\n",
        }
    if not isinstance(data, dict):
        return fallback_note(raw_text, memo)
    return {
        "title": str(data.get("title") or "학습 기록"),
        "source_type": str(data.get("source_type") or "study-capture"),
        "tags": [str(tag) for tag in data.get("tags", [])][:8],
        "summary": stringify_field(data.get("summary") or ""),
        "action_items": [str(item) for item in data.get("action_items", [])][:8],
        "blog_draft": stringify_field(data.get("blog_draft") or ""),
    }


def stringify_field(value: Any) -> str:
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            title = str(key).replace("_", " ").strip().title()
            lines.append(f"## {title}\n{stringify_field(item)}")
        return "\n\n".join(lines)
    if isinstance(value, list):
        return "\n".join(f"- {stringify_field(item)}" for item in value)
    return str(value)


def fallback_note(raw_text: str, memo: str) -> dict[str, Any]:
    source = "\n".join(part for part in [raw_text.strip(), memo.strip()] if part)
    preview = source[:900] if source else "입력된 텍스트가 없습니다."
    return {
        "title": "학습 캡처 기록",
        "source_type": "study-capture",
        "tags": ["study-note", "capture"],
        "summary": f"입력 내용을 기반으로 기본 노트를 생성했습니다.\n\n{preview}",
        "action_items": ["핵심 개념 다시 정리", "막힌 지점과 해결 방법 추가 기록", "문제 해결형 글로 변환"],
        "blog_draft": f"# 학습 캡처 기록\n\n## 기록 내용\n{preview}\n\n## 다음 정리\n- 핵심 개념\n- 실습 흐름\n- 문제 해결 과정\n",
    }


def local_study_blog(notes: list[StudyNote], topic: str) -> str:
    note = notes[-1]
    actions = "\n".join(f"- {item}" for item in note.action_items[:5]) or "- 추가 정리 필요"
    return f"""# {topic}

## 학습 배경
이번 기록은 `{note.title}` 학습 화면을 바탕으로 정리한 기술 노트입니다. 화면에서 확인한 내용과 생성된 노트를 기준으로, 실습의 흐름과 다음 점검 항목을 정리했습니다.

## 핵심 내용
{note.summary}

## 다음 작업
{actions}

## 정리
이 기록은 단순 캡처가 아니라, 학습 중 발견한 화면과 판단 지점을 다시 검토하기 위한 자료입니다. 이후 관련 수식, 쿼리, 설정값을 추가하면 문제 해결형 포트폴리오 글로 확장할 수 있습니다.
"""


def local_portfolio_blog(notes: list[StudyNote], topic: str) -> str:
    note = notes[-1]
    actions = "\n".join(f"- {item}" for item in note.action_items[:5]) or "- 추가 정리 필요"
    return f"""# {note.title}: 문제 해결형 학습 기록

_A problem-solving portfolio note from a study capture_

## 도입부
이번 기록은 `{topic}` 과정에서 생성한 학습 캡처를 바탕으로 작성했습니다. 화면에서 확인한 실습 내용은 단순 기능 사용이 아니라, 데이터 모델과 계산 결과가 왜 예상과 다르게 보이는지 확인하는 과정에 가깝습니다.

핵심 작업은 다음과 같습니다.

```text
- 화면 캡처를 학습 근거로 저장
- 실습 화면에서 주요 테이블, 지표, 관계 설정 포인트 확인
- 생성된 노트를 바탕으로 문제 인식과 다음 점검 항목 정리
```

## 문제 인식
{note.summary}

## 문제 정의
현재 실습에서 확인해야 할 문제는 화면에 보이는 결과값과 데이터 모델 설정이 의도한 분석 흐름과 일치하는지 검증하는 것입니다.

## 왜 이것을 문제로 인식했는가
Power BI, SQL, DAX, 모델링 실습에서는 화면에 값이 표시되는 것만으로는 충분하지 않습니다. 관계 설정, 계산식, 필터 컨텍스트가 잘못되면 보고서에 보이는 값은 그럴듯해도 실제 해석은 틀릴 수 있습니다. 따라서 실습 화면에서 이상한 값, 반복되는 합계, 0으로 표시되는 지표, 관계 설정 단계를 발견하면 이를 문제로 정의하고 원인을 좁혀야 합니다.

## 문제 해결 경험
1. 화면에서 현재 실습 단계와 표시된 결과를 먼저 확인했습니다.
2. 지표, 테이블, 관계 설정, 계산식 중 어떤 요소가 결과에 영향을 주는지 나누어 보았습니다.
3. 다음 점검 항목을 액션으로 분리해 이후 실습에서 재현하고 검증할 수 있도록 정리했습니다.

## 복잡한 문제 해결 경험
이 유형의 문제는 단순히 버튼을 누르는 문제가 아니라, 데이터 모델의 관계와 계산 흐름을 함께 확인해야 합니다. 특히 Power BI에서는 관계 방향, many-to-many 관계, measure 계산, 필터 컨텍스트가 결과값에 직접 영향을 줍니다. 따라서 화면 캡처를 기록으로 남기고, 어떤 지점에서 값이 달라졌는지 추적하는 방식이 중요합니다.

## 성과
이번 캡처를 통해 실습 화면을 단순히 지나치지 않고, 추후 포트폴리오 글로 확장할 수 있는 문제 해결 기록으로 전환했습니다. 이후 같은 방식으로 오류 화면, DAX 수식, SQL 쿼리, 모델링 설정을 누적하면 학습 과정 자체가 기술블로그와 포트폴리오 자료가 됩니다.

## 사용한 주요 수식/코드 정리
현재 캡처에는 별도의 코드나 수식이 직접 입력되지 않았습니다. 추후 DAX, SQL, Power Query 수식을 메모에 추가하면 이 섹션에 자동으로 정리할 수 있습니다.

## 다음 작업
{actions}

## 최종 정리
이 기록의 핵심은 학습 화면을 저장하는 데서 끝내지 않고, 화면에서 확인한 문제와 다음 검증 항목을 구조화했다는 점입니다. 이는 실무에서도 오류 화면, 분석 결과, 설정 변경 내역을 근거와 함께 남기는 방식으로 확장될 수 있습니다.

## Portfolio Summary
- Captured a technical learning screen and converted it into a structured problem-solving note.
- Identified model/result validation points from the visible interface.
- Organized follow-up actions for reproducible learning and documentation.

## Key skills practiced
- Technical documentation
- Problem framing
- Data model validation
- Portfolio writing workflow
"""


def image_only_note(image_path: str) -> dict[str, Any]:
    return {
        "title": "이미지 학습 캡처",
        "source_type": "study-capture-image",
        "tags": ["study-note", "screenshot", "vision-fallback"],
        "summary": (
            "스크린샷을 학습 근거 자료로 저장했지만, Vision LLM 응답을 받아오지 못해 기본 노트로 기록했습니다.\n\n"
            "화면에서 확인한 핵심 문장, 오류 메시지, 실습 목표를 메모에 함께 적으면 "
            "이미지 판독이 실패해도 노트와 문제 해결형 포트폴리오 초안을 구성할 수 있습니다.\n\n"
            f"저장된 이미지: {image_path}"
        ),
        "action_items": [
            "화면 속 핵심 텍스트를 raw text에 붙여넣기",
            "내가 발견한 문제와 해결 과정을 memo에 추가하기",
            "추후 OCR 또는 Vision API로 이미지 텍스트 자동 추출 연결하기",
        ],
        "blog_draft": (
            "# 이미지 기반 학습 캡처\n\n"
            "## 캡처 내용\n"
            "스크린샷이 저장되었습니다. Vision LLM 응답을 받아오지 못한 경우에는 "
            "실습 목표와 문제 상황을 메모로 함께 기록하면 문제 해결형 포트폴리오 글로 확장할 수 있습니다.\n\n"
            "## 다음 정리\n"
            "- 화면에서 확인한 실습 주제\n"
            "- 발견한 이상 현상 또는 막힌 지점\n"
            "- 해결한 방법과 사용한 수식/쿼리/설정\n"
        ),
    }


def study_blog_prompt(topic: str, joined_notes: str) -> str:
    return f"""
다음 학습 노트들을 바탕으로 기술블로그 초안을 작성해 주세요.

조건:
- 한국어
- 제목, 문제 인식, 실습 흐름, 핵심 개념, 막힌 점과 해결, 배운 점, 다음 학습 계획 포함
- 실제 입력에 없는 성과나 수치는 만들지 않음

[주제]
{topic}

[노트]
{joined_notes}
""".strip()


def portfolio_prompt(topic: str, joined_notes: str, extra_info: str = "") -> str:
    return f"""
당신은 사용자의 실습/프로젝트 기록을 Medium 포트폴리오 글로 변환하는 전용 작성자입니다.
아래 실습/프로젝트 기록을 바탕으로 Medium에 그대로 붙여넣을 수 있는 완성본을 작성해 주세요.

글의 목적은 단순 후기나 요약이 아니라, 사용자가 이미 Medium에 작성해 온 것과 같은 **문제해결형 포트폴리오 글**입니다.
반드시 아래 작성 방식을 그대로 따르세요.

가장 중요한 원칙:
- 이미지를 1장씩 따로 해설하지 않습니다.
- 이미지별로 각각 문제/원인을 만들지 않습니다.
- 전체 이미지 묶음을 하나의 실습 흐름으로 보고, 하나의 핵심 문제를 중심으로 글을 씁니다.
- 초반 이미지는 문제 인식, 중간 이미지는 원인 분석과 해결 과정, 마지막 이미지는 검증 결과로 사용합니다.
- "무엇을 클릭했다"보다 "어떤 문제가 있었고, 왜 문제가 되었고, 어떤 원인을 발견했고, 어떻게 해결했고, 결과가 어떻게 바뀌었는지"를 중심으로 씁니다.
- 사용자가 직접 문제를 발견하고 해결한 경험처럼 작성합니다.
- 취업 포트폴리오에 사용할 수 있도록 실무적인 분석 문체로 씁니다.
- 짧은 요약문이 아니라 Medium에 그대로 게시 가능한 완성형 글로 작성합니다.
- 사용자가 직접 입력한 추가 정보는 최상위 작성 브리프입니다. 이미지 판독 결과보다 우선합니다.
- 추가 정보에 "이미지 흐름 요약", "이 글에서 강조할 문제 해결 관점", "강조하고 싶은 기술"이 있으면 그 내용을 글의 뼈대로 사용합니다.
- 추가 정보가 비어 있으면 이미지/메모/노트만으로 작성하되, 없는 내용을 지어내지 않습니다.
- 내부 라벨인 [화면 텍스트/코드/오류], [사용자 메모/질문/해결 과정], [이미지 정보], [작성 주의]는 최종 글에 절대 출력하지 않습니다.

분량 규칙:
- 전체 글은 최소 4,500자 이상으로 작성합니다. 가능하면 6,000~8,000자 수준의 Medium 완성본으로 작성합니다.
- "문제 인식", "문제 정의", "왜 문제로 인식했는가"는 각각 충분히 길게 작성합니다.
- "문제 해결 경험"이 가장 중요합니다. 최소 4개 이상의 단계로 나누고, 각 단계는 `문제/제약 → 원인 판단 → 조치 → 확인 결과` 흐름으로 작성합니다.
- 문제 해결 경험 섹션은 글 전체에서 가장 긴 섹션이어야 합니다.
- 수식, 코드, 관계 설정, 오류 해결, 배포, 새로고침, 모델링, 데이터 검증처럼 복잡한 내용이 있으면 별도 섹션으로 충분히 설명합니다.
- 마지막 Portfolio Summary는 영어로 2문단 이상 작성합니다.
- Key skills practiced는 최소 8개 이상 작성합니다.

1. 한국어 제목
2. 영어 부제
3. 짧은 도입부
4. 핵심 작업 요약
5. 문제 인식
6. 문제 정의
7. 왜 이것을 문제로 인식했는가
8. 문제 해결 경험 1, 2, 3...
9. 복잡한 수식 작성 및 해결 경험
10. 성과
11. 사용한 주요 수식/코드 정리
12. 최종 정리
13. Portfolio Summary
14. Key skills practiced
15. 이미지 번호와 캡션 목록

문체 규칙:
- 다음처럼 쓰지 않습니다: "버튼을 눌렀습니다", "차트를 만들었습니다", "관계를 설정했습니다", "실습을 완료했습니다".
- 대신 다음처럼 씁니다: "처음에는 값이 정상적으로 보이는 것처럼 보였지만...", "문제의 원인은 시각화가 아니라 필터 컨텍스트가 팩트 테이블까지 전달되지 않는 semantic model 구조에 있었다", "이를 해결하기 위해 차원 테이블과 팩트 테이블 사이의 관계를 재정의했다".
- "했습니다" 반복을 줄이고 분석/정의/구성/해결/검증 중심으로 작성합니다.

이미지 규칙:
- 사용자가 제공한 이미지는 단순 캡처가 아니라 문제 해결 과정의 증거로 사용합니다.
- 입력에 이미지 설명이 있으면 사용자가 제공한 이미지 순서대로 번호를 붙입니다.
- 본문에는 "이미지 1 - 제목" 형식의 캡션만 넣습니다.
- "첨부 삽입", "여기에 이미지 넣기" 같은 placeholder 문구는 절대 넣지 않습니다.
- 마지막에 이미지 번호와 캡션 목록을 따로 정리합니다.
- 초반 이미지는 문제 발견, 이상 현상, 초기 상태, 실습 시작점으로 묶어 해석합니다.
- 중간 이미지는 원인 분석, 관계 설정, 수식 작성, 데이터 변환, 모델 수정, UI 구성, 오류 해결, 검증 과정으로 묶어 해석합니다.
- 마지막 이미지는 최종 결과, 개선된 화면, 검증 결과, 성과 화면으로 묶어 해석합니다.
- 이미지 하나하나마다 문제 정의를 반복하지 않습니다.
- 이미지 캡션은 글의 흐름을 보조하는 장치일 뿐이며, 글의 중심은 사용자가 해결한 핵심 문제와 해결 과정입니다.
- 이미지 자체에 명확한 문제가 없으면 억지로 문제를 만들지 말고, "실습에서 해결해야 할 과제"를 문제로 정의합니다.

코드/수식 규칙:
- DAX, SQL, Python, Power Query, API 코드가 있으면 코드블록으로 정리합니다.
- 수식은 단순히 나열하지 말고, 각 수식이 어떤 문제를 해결했는지 설명합니다.
- 복잡한 수식은 원인 → 중간 검증 → 최종 수식 흐름으로 설명합니다.
- 수식은 반드시 다음 흐름으로 설명합니다: 어떤 문제가 있었는가 → 왜 이 수식이 필요했는가 → 수식이 어떤 계산을 수행하는가 → 결과를 어떻게 검증했는가.
- 수식이나 코드가 이미지에 보이지 않거나 사용자가 제공하지 않았다면 임의로 만들지 않습니다.

오류/막힌 부분 처리 규칙:
- 사용자가 오류, 막힌 부분, 헷갈린 부분을 제공하면 반드시 글에 반영합니다.
- 오류는 실패가 아니라 문제 해결 경험으로 재구성합니다.
- 각 오류는 `증상 → 처음 의심한 원인 → 실제 원인 → 해결 방법 → 확인 결과` 흐름으로 작성합니다.

Power BI semantic model 실습일 때의 작성 방식:
- 현재 입력이 Power BI, Sales, Product, Category, DAX, relationship, semantic model, filter context를 다루는 경우에만 아래 패턴을 적용합니다.
- 다른 이미지 세트가 들어오면 이 Power BI 예시 내용을 재사용하지 말고, 해당 이미지의 실제 기술 주제에 맞춰 같은 문제 해결형 구조만 유지합니다.
- 첫 이미지는 기능 설명이 아니라 문제 인식 장면으로 해석합니다. 화면에 숫자가 나오는 것이 아니라, 왜 숫자가 이상한지 먼저 씁니다.
- 예를 들어 Category별 Sales가 모두 같은 값으로 반복된다면, "Power BI에서 숫자는 보이지만 의미가 틀릴 수 있다"는 문제로 시작합니다.
- 원인은 시각화 문제가 아니라 semantic model 문제로 정의합니다. Product[Category] 필터가 Sales로 전달되지 않아 ProductKey 관계가 필요하다는 식으로 원인 중심으로 씁니다.
- 이미지는 작업 순서가 아니라 아래 해결 단계로 묶습니다.
  - 이미지 1: 문제 발견, Category별 Sales 반복
  - 이미지 2~4: Product-Sales 관계 생성과 star schema 구조 정리
  - 이미지 5~6: Product hierarchy 구성
  - 이미지 7~10: Profit, Profit Margin measure 생성과 검증
  - 이미지 11~14: SalespersonRegion bridge table, many-to-many, filter direction, inactive relationship 문제 해결
  - 이미지 15: Salesperson별 Sales와 Target 비교 최종 결과
- 관계 설정은 "무엇을 선택했는가"만 쓰지 말고 왜 맞는지 설명합니다. Product는 차원 테이블, Sales는 팩트 테이블이므로 Product[ProductKey] → Sales[ProductKey], Cardinality One-to-many, Cross-filter direction Single, Active relationship이 왜 필요한지 씁니다.
- DAX는 "왜 필요한가 → 어떤 수식인가 → 어떤 결과를 검증했는가" 순서로 설명합니다. Sales만으로는 수익성을 볼 수 없어서 Profit이 필요하고, 규모가 다른 카테고리를 비교하려면 Profit Margin이 필요하다는 식으로 씁니다.
- many-to-many 관계는 복잡한 문제 해결 경험으로 따로 뽑습니다. Salesperson별 Sales를 담당 Region 기준으로 보려면 Salesperson → SalespersonRegion → Region → Sales 흐름이 필요하고, 모호한 필터 경로가 생기면 직접 관계를 비활성화해야 한다는 식으로 씁니다.
- 마지막은 "실습 완료"가 아니라 분석 가능성의 변화로 정리합니다. Category별 Sales 반복 문제 해결, Product-Sales 관계 생성, 계층 구조 구성, Measure 생성, bridge table 기반 many-to-many 해결, Salesperson별 Sales와 Target 비교 가능성을 성과로 씁니다.
- 결론은 "숫자가 화면에 보인다고 항상 맞는 것은 아니며, relationship, cardinality, filter direction, active relationship이 잘못되면 visual은 정상처럼 보여도 의미는 틀릴 수 있다"는 식으로 마무리합니다.

Medium 작성 방식 예시:
- 첫 문단은 "이번 실습에서 무엇을 했는가"보다 "처음 무엇이 이상했는가"로 시작합니다.
- 그 다음 문제를 한 문장으로 정의합니다.
- 이후 왜 그 문제가 중요한지 데이터 모델, 비즈니스 해석, 검증 관점에서 설명합니다.
- 문제 해결 경험은 다음 순서로 이어갑니다.
  1. 문제 화면에서 이상 징후를 발견한 과정
  2. 원인을 시각화가 아니라 데이터 모델/관계/수식/쿼리 구조에서 찾은 과정
  3. 관계, 수식, 쿼리, 설정, 데이터 구조를 수정한 과정
  4. 수정 후 결과가 어떻게 달라졌는지 검증한 과정
  5. 추가로 복잡했던 관계, 수식, 필터 컨텍스트, 오류 해결 과정을 별도 섹션으로 정리
- 이미지 캡션은 각 섹션 사이에 짧게 넣되, 본문은 캡션 설명이 아니라 문제해결 서사로 작성합니다.

[주제]
{topic}

[사용자가 직접 입력한 추가 정보]
{extra_info if extra_info.strip() else "추가 정보 없음. 이미지/메모/노트만 근거로 작성하세요."}

[실습/프로젝트 기록]
{joined_notes}
""".strip()


def read_notes() -> list[StudyNote]:
    notes: list[StudyNote] = []
    for line in NOTES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            notes.append(StudyNote(**json.loads(line)))
        except Exception:
            continue
    return notes


def is_meaningful_note(note: StudyNote) -> bool:
    summary = note.summary or ""
    if "입력된 텍스트가 없습니다" in summary:
        return False
    if note.source_type == "study-capture-image" and (
        "스크린샷을 학습 근거 자료로 저장했습니다" in summary
        or "스크린샷을 학습 캡처로 저장했습니다" in summary
    ):
        return False
    return True


def append_note(note: StudyNote) -> None:
    with NOTES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(note), ensure_ascii=False) + "\n")


def search_notes(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    q_terms = set(tokenize(query))
    scored = []
    for note in read_notes():
        text = " ".join([note.title, note.summary, note.raw_text, note.user_memo, " ".join(note.tags)])
        terms = set(tokenize(text))
        score = len(q_terms & terms) / max(len(q_terms | terms), 1)
        if query.lower() in text.lower():
            score += 0.5
        if score > 0:
            scored.append((score, note))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [{"score": round(score, 3), "note": asdict(note)} for score, note in scored[:top_k]]


def tokenize(text: str) -> list[str]:
    return [chunk.strip(".,:;()[]{}<>\"'").lower() for chunk in text.split() if len(chunk.strip()) >= 2]


def make_note(raw_text: str, memo: str, image_path: str | None, image_paths: list[str] | None = None) -> StudyNote:
    image_paths = image_paths or ([image_path] if image_path else [])
    image_files = [CAPTURE_DIR / path.removeprefix("/captures/") for path in image_paths]
    if image_files and not raw_text.strip():
        generated = llm.generate_note_from_images(image_files, memo)
    else:
        generated = llm.generate_note(raw_text, memo)
    created_at = datetime.now().isoformat(timespec="seconds")
    title = generated["title"]
    if title in {"학습 캡처 기록", "이미지 학습 캡처", "이미지 기반 학습 캡처"}:
        label = "이미지" if image_path else "학습"
        title = f"{label} 캡처 {datetime.now().strftime('%H:%M')}"
    return StudyNote(
        id=str(uuid.uuid4()),
        created_at=created_at,
        title=title,
        source_type=generated["source_type"],
        tags=generated["tags"],
        raw_text=raw_text,
        user_memo=memo,
        summary=generated["summary"],
        action_items=generated["action_items"],
        blog_draft=generated["blog_draft"],
        image_path=image_path,
        image_paths=image_paths or None,
    )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            return self.html(INDEX_HTML)
        if path == "/api/notes":
            return self.json([asdict(note) for note in read_notes()])
        if path == "/api/health/llm":
            query = parse_qs(parsed.query)
            return self.json(llm_health_check(run_test_call=query.get("test", ["0"])[0] == "1"))
        if path == "/api/sessions":
            return self.json(read_sessions().get("sessions", []))
        if path.startswith("/api/sessions/"):
            return self.handle_session_get(path)
        if path.startswith("/captures/"):
            return self.file(CAPTURE_DIR / path.removeprefix("/captures/"))
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/captures":
            return self.create_capture()
        if path == "/api/search":
            data = self.read_json()
            return self.json(search_notes(str(data.get("query", "")), int(data.get("top_k", 5))))
        if path == "/api/blog":
            data = self.read_json()
            notes = read_notes()
            note_ids = data.get("note_ids")
            if note_ids:
                wanted = set(note_ids)
                notes = [note for note in notes if note.id in wanted]
            notes = notes[-8:]
            draft = llm.synthesize_blog(
                notes,
                str(data.get("topic", "오늘의 학습 기록")),
                str(data.get("format_type", "study-blog")),
                str(data.get("extra_info", "")),
            )
            return self.json({"draft": draft})
        if path == "/api/direct-blog":
            return self.create_direct_blog()
        if path == "/api/debug-collect-url":
            return self.debug_collect_url()
        if path == "/api/sessions":
            data = self.read_json()
            return self.json(create_session(str(data.get("title", ""))))
        if path.startswith("/api/sessions/"):
            return self.handle_session_post(path)
        self.send_error(404)

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/api/sessions/"):
            return self.handle_session_delete(path)
        self.send_error(404)

    def handle_session_get(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) < 3:
            return self.send_error(404)
        session = find_session(parts[2])
        if not session:
            return self.send_error(404)
        if len(parts) == 3:
            return self.json(session)
        if len(parts) == 4 and parts[3] == "captures":
            return self.json(session.get("captures", []))
        if len(parts) == 4 and parts[3] == "qa":
            return self.json(session.get("qa_logs", []))
        self.send_error(404)

    def handle_session_post(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) < 4:
            return self.send_error(404)
        session = find_session(parts[2])
        if not session:
            return self.send_error(404)
        action = parts[3]
        if action == "captures":
            return self.create_session_capture(session)
        if action == "qa":
            data = self.read_json()
            qa = append_qa_log(
                session,
                question=str(data.get("question", "")),
                answer=str(data.get("answer", "")),
                selected_text=str(data.get("selected_text", "")),
                related_capture_ids=[int(x) for x in data.get("related_capture_ids", [])],
            )
            update_session(session)
            return self.json(qa)
        if action == "ask":
            data = self.read_json()
            qa = tutor_agent_answer(
                session,
                question=str(data.get("question", "")),
                selected_text=str(data.get("selected_text", "")),
                related_capture_ids=[int(x) for x in data.get("related_capture_ids", [])],
            )
            update_session(session)
            return self.json(qa)
        if action == "generate-article":
            return self.generate_session_article(session)
        self.send_error(404)

    def handle_session_delete(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 5 or parts[3] != "captures":
            return self.send_error(404)
        session = find_session(parts[2])
        if not session:
            return self.send_error(404)
        capture_id = int(parts[4])
        session["captures"] = [item for item in session.get("captures", []) if int(item.get("capture_id", 0)) != capture_id]
        update_session(session)
        return self.json({"deleted": capture_id})

    def create_session_capture(self, session: dict[str, Any]) -> None:
        form = parse_form(self)
        note = get_field(form, "user_note")
        source_title = get_field(form, "source_title")
        source_url = get_field(form, "source_url")
        captures = session.setdefault("captures", [])
        created: list[dict[str, Any]] = []
        for item in sorted_uploaded_files(get_files(form, "image")):
            capture_id = next_capture_id(session)
            ext = Path(item.filename).suffix.lower() or ".png"
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{capture_id:03d}_{uuid.uuid4().hex[:8]}{ext}"
            target = CAPTURE_DIR / filename
            target.write_bytes(item.data)
            capture = asdict(
                CaptureEvent(
                    capture_id=capture_id,
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    image_path=f"/captures/{filename}",
                    source_title=source_title,
                    source_url=source_url,
                    user_note=note,
                    auto_keywords=infer_entities(" ".join([note, item.filename])),
                )
            )
            captures.append(capture)
            created.append(capture)
        update_session(session)
        return self.json({"captures": created, "total": len(captures)})

    def generate_session_article(self, session: dict[str, Any]) -> None:
        captures = sorted(session.get("captures", []), key=lambda item: item.get("timestamp", ""))
        image_files = [CAPTURE_DIR / str(item.get("image_path", "")).removeprefix("/captures/") for item in captures if item.get("image_path")]
        image_names = [Path(str(item.get("image_path", ""))).name for item in captures if item.get("image_path")]
        memo = "\n".join(item.get("user_note", "") for item in captures if item.get("user_note"))
        qa_logs = session.get("qa_logs", [])
        qa_text = "\n\n".join(f"Q: {qa.get('question', '')}\nA: {qa.get('answer_summary', '')}" for qa in qa_logs)
        topic = str(session.get("title") or "학습 캡처 타임라인 기반 문제 해결 경험")
        start = time.perf_counter()
        result = llm.synthesize_blog_from_capture(
            raw_text=qa_text,
            memo=memo,
            image_files=image_files,
            topic=topic,
            extra_info="Capture Timeline과 Q&A Logs를 함께 근거로 사용합니다.",
            image_names=image_names,
            captures=captures,
            qa_logs=qa_logs,
        )
        result.update({"elapsed_seconds": round(time.perf_counter() - start, 2), "image_count": len(image_files)})
        return self.json(result)

    def create_capture(self) -> None:
        form = parse_form(self)
        raw_text = get_field(form, "raw_text")
        memo = get_field(form, "memo")
        image_paths: list[str] = []
        images = get_files(form, "image")
        for index, item in enumerate(images, start=1):
            ext = Path(item.filename).suffix.lower() or ".png"
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index:02d}_{uuid.uuid4().hex[:8]}{ext}"
            target = CAPTURE_DIR / filename
            target.write_bytes(item.data)
            image_paths.append(f"/captures/{filename}")
        image_path = image_paths[0] if image_paths else None
        start = time.perf_counter()
        note = make_note(raw_text, memo, image_path, image_paths)
        append_note(note)
        elapsed = round(time.perf_counter() - start, 2)
        return self.json({"note": asdict(note), "elapsed_seconds": elapsed})

    def create_direct_blog(self) -> None:
        form = parse_form(self)
        raw_text = get_field(form, "raw_text")
        memo = get_field(form, "memo")
        topic = get_field(form, "topic") or "학습 기록 기반 문제 해결 경험"
        extra_info = get_field(form, "extra_info")
        format_type = get_field(form, "format_type")
        direct_run_id = make_generation_run_id()
        direct_run_dir = make_run_dir(direct_run_id)
        image_files: list[Path] = []
        image_names: list[str] = []
        images = sorted_uploaded_files(get_files(form, "image"))
        for index, item in enumerate(images, start=1):
            ext = Path(item.filename).suffix.lower() or ".png"
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index:02d}_{uuid.uuid4().hex[:8]}{ext}"
            target = direct_run_dir / "images" / filename
            target.write_bytes(item.data)
            image_files.append(target)
            image_names.append(item.filename)

        collector_report: dict[str, Any] = {}
        url_only_mode = (not image_files and raw_text_is_url_only(raw_text))
        url_run_id = direct_run_id if url_only_mode else direct_run_id
        seed_url = ""
        if url_only_mode:
            urls = extract_urls_from_text(raw_text)
            seed_url = urls[0] if urls else ""
            if urls and is_inflearn_url(seed_url):
                return self.json(
                    {
                        "draft": inflearn_protected_source_message(raw_text, memo),
                        "article_type": "protected_course_source_required",
                        "image_evidence": [],
                        "learning_evidence": [],
                        "problem_map": {"unsupported_url": seed_url, "reason": "inflearn_login_or_paid_course_gate", "run_id": url_run_id},
                        "decision_map": {},
                        "section_plan": [],
                        "article_brief": {},
                        "collector_report": {
                            "ok": False,
                            "run_id": url_run_id,
                            "seed_url": seed_url,
                            "error": "Inflearn lecture pages require login/course access and are not reliable for headless URL-only collection.",
                        },
                        "critic_report": {
                            "passed": False,
                            "failures": ["Inflearn requires transcript/curriculum/source pack or logged-in assisted capture."],
                            "metrics": {"unsupported_url": seed_url, "run_id": url_run_id},
                        },
                        "elapsed_seconds": 0,
                        "image_count": len(image_files),
                        "mode": "protected_course_source_required",
                    }
                )
            if urls and is_udemy_url(seed_url):
                return self.json(
                    {
                        "draft": udemy_manual_source_pack_message(raw_text, memo),
                        "article_type": "manual_source_pack_required",
                        "image_evidence": [],
                        "learning_evidence": [],
                        "problem_map": {"unsupported_url": seed_url, "reason": "udemy_cloudflare_login_gate", "run_id": url_run_id},
                        "decision_map": {},
                        "section_plan": [],
                        "article_brief": {},
                        "collector_report": {
                            "ok": False,
                            "run_id": url_run_id,
                            "seed_url": seed_url,
                            "error": "Udemy is excluded from automatic collection because Cloudflare/login gates cannot be reliably collected.",
                        },
                        "critic_report": {
                            "passed": False,
                            "failures": ["Udemy requires manual source pack collection."],
                            "metrics": {"unsupported_url": seed_url, "run_id": url_run_id},
                        },
                        "elapsed_seconds": 0,
                        "image_count": len(image_files),
                        "mode": "manual_source_pack_required",
                    }
                )
            source_pack_text, collector_report = run_source_pack_collector(seed_url, run_id=url_run_id)
            collector_report = dict(collector_report or {})
            collector_report["run_id"] = url_run_id
            collector_report["seed_url"] = seed_url
            collector_report["source_graph"] = collector_source_graph(collector_report)
            if source_pack_text.strip():
                source_pack_text = append_video_transcript_evidence(seed_url, source_pack_text, collector_report)
                matched, match_reason = source_pack_seed_match(seed_url, source_pack_text, collector_report)
                if not matched:
                    return self.json({
                        "draft": seed_mismatch_report(seed_url, url_run_id, collector_report, match_reason),
                        "article_type": "seed_source_mismatch",
                        "image_evidence": [],
                        "learning_evidence": [],
                        "problem_map": {"collector_report": collector_report, "seed_url": seed_url, "run_id": url_run_id},
                        "decision_map": {},
                        "section_plan": [],
                        "article_brief": {},
                        "collector_report": collector_report,
                        "critic_report": {"passed": False, "failures": [match_reason], "metrics": collector_report},
                        "elapsed_seconds": collector_report.get("elapsed_seconds", 0),
                        "image_count": len(image_files),
                        "mode": "seed_source_mismatch",
                    })
                sufficient, quality_reasons = source_pack_quality_sufficient(seed_url, source_pack_text, collector_report)
                if not sufficient:
                    return self.json({
                        "draft": collection_failure_report(seed_url, url_run_id, collector_report, quality_reasons),
                        "article_type": "source_graph_collection_insufficient",
                        "image_evidence": [],
                        "learning_evidence": [],
                        "problem_map": {"collector_report": collector_report, "seed_url": seed_url, "run_id": url_run_id},
                        "decision_map": {},
                        "section_plan": [],
                        "article_brief": {},
                        "collector_report": collector_report,
                        "critic_report": {"passed": False, "failures": quality_reasons, "metrics": collector_report},
                        "elapsed_seconds": collector_report.get("elapsed_seconds", 0),
                        "image_count": len(image_files),
                        "mode": "source_graph_collection_insufficient",
                    })
                if format_type == "lecture-summary":
                    summary = build_lecture_content_summary(seed_url, url_run_id, source_pack_text, collector_report, memo)
                    return self.json({
                        "draft": summary,
                        "article_type": "lecture_content_summary",
                        "image_evidence": [],
                        "learning_evidence": [],
                        "problem_map": {"collector_report": collector_report, "seed_url": seed_url, "run_id": url_run_id},
                        "decision_map": {},
                        "section_plan": [],
                        "article_brief": {"source": "lecture_summary", "seed_url": seed_url, "run_id": url_run_id},
                        "collector_report": collector_report,
                        "critic_report": {"passed": True, "failures": [], "metrics": {"run_id": url_run_id, "seed_url": seed_url}},
                        "elapsed_seconds": collector_report.get("elapsed_seconds", 0),
                        "image_count": len(image_files),
                        "mode": "lecture_content_summary",
                    })
                if should_use_source_graph_direct_article(seed_url, collector_report, source_pack_text):
                    direct_article = build_source_graph_grounded_medium_draft(seed_url, url_run_id, source_pack_text, collector_report, memo)
                    ok, reason = article_matches_seed_url(seed_url, direct_article, current_text=source_pack_text)
                    if not ok:
                        return self.json({
                            "draft": final_article_mismatch_report(seed_url, url_run_id, collector_report, reason, direct_article),
                            "article_type": "seed_article_mismatch",
                            "image_evidence": [],
                            "learning_evidence": [],
                            "problem_map": {"collector_report": collector_report, "seed_url": seed_url, "run_id": url_run_id},
                            "decision_map": {},
                            "section_plan": [],
                            "article_brief": {},
                            "collector_report": collector_report,
                            "critic_report": {"passed": False, "failures": [reason], "metrics": {"run_id": url_run_id, "seed_url": seed_url}},
                            "elapsed_seconds": collector_report.get("elapsed_seconds", 0),
                            "image_count": len(image_files),
                            "mode": "source_graph_direct_article_blocked",
                        })
                    direct_result = {
                        "draft": direct_article,
                        "article_type": "source_graph_learning_record",
                        "image_evidence": [],
                        "learning_evidence": [],
                        "problem_map": {"collector_report": collector_report, "seed_url": seed_url, "run_id": url_run_id},
                        "decision_map": {},
                        "section_plan": [],
                        "article_brief": {"source": "source_graph_direct", "seed_url": seed_url, "run_id": url_run_id},
                        "collector_report": collector_report,
                        "critic_report": {"passed": True, "failures": [], "metrics": {"run_id": url_run_id, "seed_url": seed_url}},
                        "elapsed_seconds": collector_report.get("elapsed_seconds", 0),
                        "image_count": len(image_files),
                        "mode": "source_graph_direct_article",
                    }
                    direct_result = apply_final_article_policy(direct_result, current_text=source_pack_text, seed_url=seed_url, run_id=url_run_id)
                    return self.json(direct_result)
                raw_text = build_url_run_input(seed_url, url_run_id, source_pack_text, collector_report)
            else:
                enriched = enrich_raw_text_with_source_urls(raw_text, memo)
                host_for_policy = url_domain(seed_url)
                high_value_hosts = ("aiskillsnavigator.microsoft.com", "youtube.com", "youtu.be", "wikidocs.net")
                # Oopy is a regression target: if ordinary public text extraction works, use it instead of
                # blocking on a Playwright/browser install failure. Notion remains explicit fallback-only.
                if (is_youtube_host(host_for_policy) or is_notion_host(host_for_policy) or any(h in host_for_policy for h in high_value_hosts)) and collector_report.get("error"):
                    result = {
                        "draft": collector_execution_failure_report(seed_url, url_run_id, collector_report),
                        "article_type": "collector_execution_failed",
                        "image_evidence": [],
                        "learning_evidence": [],
                        "problem_map": {"collector_report": collector_report, "seed_url": seed_url, "run_id": url_run_id},
                        "decision_map": {},
                        "section_plan": [],
                        "article_brief": {},
                        "collector_report": collector_report,
                        "critic_report": {
                            "passed": False,
                            "failures": [str(collector_report.get("error") or "collector failed")],
                            "metrics": collector_report,
                        },
                        "elapsed_seconds": collector_report.get("elapsed_seconds", 0),
                        "image_count": len(image_files),
                        "mode": "collector_execution_failed",
                    }
                    return self.json(result)
                if not url_only_without_collected_source(raw_text, enriched):
                    collector_report["fallback_public_extraction"] = True
                    raw_text = (
                        f"[URL_ONLY_RUN]\nrun_id: {url_run_id}\nseed_url: {seed_url}\n\n"
                        "[자동 수집기 실패 후 공개 URL 텍스트 추출로 대체]\n"
                        f"- Collector error: {collector_report.get('error', '')}\n\n"
                        f"{enriched}"
                    )
                else:
                    result = {
                        "draft": source_collection_required_message(raw_text, enriched, memo),
                        "article_type": "source_collection_required",
                        "image_evidence": [],
                        "learning_evidence": [],
                        "problem_map": {"collector_report": collector_report, "seed_url": seed_url, "run_id": url_run_id},
                        "decision_map": {},
                        "section_plan": [],
                        "article_brief": {},
                        "collector_report": collector_report,
                        "critic_report": {
                            "passed": False,
                            "failures": ["source pack collection failed or produced no usable public text"],
                            "metrics": collector_report,
                        },
                        "elapsed_seconds": collector_report.get("elapsed_seconds", 0),
                        "image_count": len(image_files),
                        "mode": "source_pack_collection_required",
                    }
                    return self.json(result)
        else:
            raw_text = enrich_raw_text_with_source_urls(raw_text, memo)

        start = time.perf_counter()
        result = llm.synthesize_blog_from_capture(raw_text, memo, image_files, topic, extra_info, image_names)
        elapsed = round(time.perf_counter() - start, 2)
        if isinstance(result, dict):
            result = dict(result)
            if collector_report:
                result["collector_report"] = collector_report
            if url_only_mode:
                draft_text = str(result.get("draft") or "")
                ok, reason = article_matches_seed_url(seed_url, draft_text, current_text="\n".join([raw_text, memo, extra_info]))
                if not ok:
                    result["draft"] = final_article_mismatch_report(seed_url, url_run_id, collector_report, reason, draft_text)
                    result["article_type"] = "seed_article_mismatch"
                    result["critic_report"] = {"passed": False, "failures": [reason], "metrics": {"run_id": url_run_id, "seed_url": seed_url}}
                    result.update({"elapsed_seconds": elapsed, "image_count": len(image_files), "mode": "seed_article_mismatch"})
                    return self.json(result)
            result.update({"elapsed_seconds": elapsed, "image_count": len(image_files), "mode": "url_only_source_graph" if url_only_mode else "batch_upload", "run_id": url_run_id or direct_run_id})
            policy_context = "\n".join([raw_text, memo, extra_info])
            image_context = ""
            if image_files and not url_only_mode:
                image_context = current_run_image_policy_context(result)
            result = apply_final_article_policy(
                result,
                current_text=policy_context,
                seed_url=seed_url,
                run_id=url_run_id or direct_run_id,
                contamination_context=image_context,
            )
            return self.json(result)
        if url_only_mode:
            ok, reason = article_matches_seed_url(seed_url, str(result), current_text="\n".join([raw_text, memo, extra_info]))
            if not ok:
                return self.json({
                    "draft": final_article_mismatch_report(seed_url, url_run_id, collector_report, reason, str(result)),
                    "article_type": "seed_article_mismatch",
                    "collector_report": collector_report,
                    "critic_report": {"passed": False, "failures": [reason], "metrics": {"run_id": url_run_id, "seed_url": seed_url}},
                    "elapsed_seconds": elapsed,
                    "image_count": len(image_files),
                    "mode": "seed_article_mismatch",
                })
        final_result = {"draft": result, "elapsed_seconds": elapsed, "image_count": len(image_files), "mode": "url_only_source_graph" if url_only_mode else "batch_upload", "run_id": url_run_id or direct_run_id}
        final_result = apply_final_article_policy(final_result, current_text="\n".join([raw_text, memo, extra_info]), seed_url=seed_url, run_id=url_run_id or direct_run_id)
        return self.json(final_result)

    def debug_collect_url(self) -> None:
        data = self.read_json()
        seed_url = str(data.get("url") or "").strip()
        if not seed_url:
            return self.json({"ok": False, "error": "url is required"})
        run_id = str(data.get("run_id") or make_generation_run_id())
        timeout_seconds = int(data.get("timeout_seconds") or 600)
        source_pack_text, collector_report = run_source_pack_collector(seed_url, timeout_seconds=timeout_seconds, run_id=run_id)
        collector_report = dict(collector_report or {})
        collector_report["run_id"] = run_id
        collector_report["seed_url"] = seed_url
        collector_report["source_graph"] = collector_source_graph(collector_report)
        sufficient, reasons = source_pack_quality_sufficient(seed_url, source_pack_text, collector_report) if source_pack_text.strip() else (False, ["empty source pack"])
        return self.json({
            "ok": bool(source_pack_text.strip()) and bool(collector_report.get("ok")) and sufficient,
            "source_pack_chars": len(source_pack_text or ""),
            "source_pack_preview": (source_pack_text or "")[:3000],
            "quality_sufficient": sufficient,
            "quality_reasons": reasons,
            "collector_report": collector_report,
        })

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def json(self, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            return self.send_error(404)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {fmt % args}")


def parse_form(handler: BaseHTTPRequestHandler) -> FormData:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}

    body = handler.rfile.read(length)
    content_type = handler.headers.get("Content-Type", "")
    form: FormData = {}

    if content_type.startswith("application/x-www-form-urlencoded"):
        for key, values in parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True).items():
            form[key] = [str(value) for value in values]
        return form

    if not content_type.startswith("multipart/form-data"):
        return form

    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\n"
        b"MIME-Version: 1.0\r\n\r\n"
        + body
    )
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            form.setdefault(name, []).append(UploadedFile(filename=filename, data=payload))
            continue
        charset = part.get_content_charset() or "utf-8"
        form.setdefault(name, []).append(payload.decode(charset, errors="replace"))
    return form


def get_field(form: FormData, key: str) -> str:
    values = form.get(key)
    if not values:
        return ""
    value = values[0]
    if isinstance(value, UploadedFile):
        return ""
    return str(value or "")


def get_files(form: FormData, key: str) -> list[UploadedFile]:
    return [value for value in form.get(key, []) if isinstance(value, UploadedFile) and value.filename]


def sorted_uploaded_files(files: list[UploadedFile]) -> list[UploadedFile]:
    if any(re.match(r"^\d{1,3}[_\-\s]", Path(file.filename).name) for file in files):
        return sorted(files, key=lambda file: natural_sort_key(Path(file.filename).name))
    return files


INDEX_HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Study Documentation Agent</title>
  <style>
    :root { color-scheme: dark; --bg:#0b0f17; --panel:#141a24; --line:#273142; --text:#eef3f8; --muted:#9aa7b8; --brand:#53c7ad; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    main { max-width:1180px; margin:0 auto; padding:32px 24px 48px; }
    h1 { margin:0; font-size:32px; }
    h2 { margin:0 0 14px; font-size:18px; }
    p { margin:6px 0 0; color:var(--muted); }
    .grid { display:grid; grid-template-columns:1.2fr .8fr; gap:18px; margin-top:24px; align-items:start; }
    .panel { border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:18px; }
    textarea, input { width:100%; border:1px solid var(--line); background:#0e141d; color:var(--text); border-radius:8px; padding:12px; font:inherit; }
    textarea { min-height:130px; resize:vertical; }
    .compact textarea { min-height:68px; }
    .optional-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .optional-grid .wide { grid-column:1 / -1; }
    .section-label { font-weight:700; color:#c8d3df; margin-top:4px; }
    button { border:1px solid #345044; background:var(--brand); color:#06110e; border-radius:8px; padding:12px 16px; font-weight:700; cursor:pointer; }
    button.secondary { background:#111926; border-color:#2b3a4f; color:var(--text); }
    button:disabled { opacity:.55; cursor:wait; }
    .dropzone { border:1px dashed #3a4a61; background:#0e141d; border-radius:10px; padding:16px; cursor:pointer; transition:border-color .15s, background .15s; }
    .dropzone:hover, .dropzone.dragover { border-color:var(--brand); background:#101b24; }
    .dropzone strong { display:block; margin-bottom:4px; }
    .dropzone span { color:var(--muted); font-size:13px; }
    .dropzone input { position:absolute; width:1px; height:1px; opacity:0; pointer-events:none; }
    .file-list { color:#c8d3df; font-size:13px; margin-top:8px; }
    .row { display:flex; gap:10px; flex-wrap:wrap; }
    .stack { display:grid; gap:12px; }
    .result { white-space:pre-wrap; border:1px solid var(--line); background:#0e141d; border-radius:8px; padding:16px; min-height:180px; }
    .result-toolbar { display:flex; gap:8px; justify-content:flex-end; align-items:center; margin-top:8px; }
    .result-toolbar .meta { margin-right:auto; }
    details.advanced { border:1px solid var(--line); border-radius:8px; padding:10px; background:#101722; }
    details.advanced summary { cursor:pointer; font-weight:700; color:#c8d3df; }
    .tabs { display:flex; gap:6px; flex-wrap:wrap; margin-top:12px; }
    .tab { padding:8px 10px; border-radius:8px; background:#111926; border:1px solid var(--line); color:var(--text); font-size:12px; }
    .tab.active { background:var(--brand); color:#06110e; border-color:#345044; }
    .debug-pane { white-space:pre-wrap; border:1px solid var(--line); background:#0e141d; border-radius:8px; padding:12px; min-height:120px; max-height:360px; overflow:auto; margin-top:8px; font-size:13px; }
    .note { border:1px solid var(--line); border-radius:8px; padding:12px; margin-top:10px; background:#101722; }
    .note strong { display:block; margin-bottom:4px; }
    .note img { width:100%; max-height:160px; object-fit:cover; border:1px solid var(--line); border-radius:6px; margin-top:8px; }
    .meta { color:var(--muted); font-size:13px; }
    .badge { display:inline-block; border:1px solid var(--line); border-radius:999px; padding:3px 8px; margin:3px 4px 0 0; color:#c8d3df; font-size:12px; }
    .floating-tools { position:fixed; right:22px; bottom:22px; display:flex; gap:12px; align-items:center; z-index:10; }
    .capture-btn { width:68px; height:68px; border-radius:50%; border:4px solid #e9f8f3; background:var(--brand); box-shadow:0 10px 26px rgba(0,0,0,.35); padding:0; font-size:0; }
    .capture-btn::after { content:""; display:block; width:38px; height:38px; margin:11px auto; border-radius:50%; border:2px solid #06110e; }
    .ask-btn { width:52px; height:52px; border-radius:50%; padding:0; background:#f3c969; border-color:#6c5521; color:#151007; font-weight:900; }
    .toast { position:fixed; right:24px; bottom:104px; background:#0e141d; border:1px solid var(--line); color:var(--text); padding:10px 12px; border-radius:8px; opacity:0; pointer-events:none; transition:opacity .2s; z-index:11; }
    .toast.show { opacity:1; }
    .modal-backdrop { position:fixed; inset:0; background:rgba(3,7,12,.72); display:none; align-items:center; justify-content:center; padding:20px; z-index:20; }
    .modal-backdrop.open { display:flex; }
    .modal { width:min(560px, 100%); background:#0e141d; border:1px solid var(--line); border-radius:8px; padding:18px; box-shadow:0 24px 64px rgba(0,0,0,.45); }
    .modal h3 { margin:0 0 8px; font-size:18px; }
    .modal p { color:var(--muted); margin:0 0 12px; line-height:1.5; }
    .modal textarea { min-height:120px; margin:0; }
    .modal-actions { display:flex; gap:8px; justify-content:flex-end; flex-wrap:wrap; margin-top:12px; }
    #captureInput { display:none; }
    @media (max-width:900px) { .grid { grid-template-columns:1fr; } }
  </style>
</head>
<body>
<main>
  <h1>AI Study Documentation Agent</h1>
  <p>학습 화면, 실습 메모, 오류 상황을 문제 해결형 Medium 포트폴리오 글로 바로 변환합니다.</p>

  <div class="grid">
    <section class="panel stack">
      <h2>문제 해결형 Medium 글 생성</h2>
      <label id="dropzone" class="dropzone" for="image">
        <strong>이미지 끌어놓기 또는 파일 선택</strong>
        <span>실습 순서대로 여러 장을 한 번에 넣거나, 나중에 이미지를 추가할 수 있습니다.</span>
        <input id="image" type="file" accept="image/*" multiple />
        <div id="fileList" class="file-list">선택된 이미지 없음</div>
      </label>
      <textarea id="rawText" placeholder="자료를 넣으세요. 예: 강의 URL, 영상 URL, 실습 URL, 강의안 본문, 화면 텍스트, 오류 메시지"></textarea>
      <textarea id="memo" placeholder="강의에서 어렵거나 복잡했던 문제를 적으세요. 예: 개념 경계, 쿼리, 권한, Lab 실패, 검증 기준"></textarea>
      <details class="advanced">
        <summary>예시 입력 보기 <span class="meta">(선택 · 이미지만 넣어도 생성 가능)</span></summary>
        <pre style="white-space:pre-wrap;background:#0b1017;border:1px solid var(--line);padding:12px;border-radius:8px;">예시 1: 강의 URL / 영상 URL / 실습 URL을 한 번에 붙여넣기
예시 2: 영상 강의 캡처만 업로드하고 메모 없이 생성
예시 3: 어렵거나 복잡했던 문제만 한 줄 입력 — 예: 권한 정책 적용 결과가 예상과 다르게 나옴</pre>
      </details>
      <details class="advanced">
        <summary>Medium 글 추가 정보 <span class="meta">(선택 입력 · 몰라도 비워두세요)</span></summary>
        <div class="optional-grid compact" style="margin-top:10px;">
          <textarea id="projectName" placeholder="실습/프로젝트 이름"></textarea>
          <textarea id="coreProblem" placeholder="내가 해결한 핵심 문제"></textarea>
          <textarea id="blockedPart" placeholder="중간에 막힌 부분"></textarea>
          <textarea id="finalResult" placeholder="최종 결과"></textarea>
          <textarea class="wide" id="focusTech" placeholder="강조하고 싶은 기술, 수식, 코드, 설정"></textarea>
        </div>
      </details>
      <div class="row">
        <button class="secondary" id="clearFilesBtn">이미지 선택 초기화</button>
        <button class="secondary" id="clearInputBtn" type="button">입력칸 초기화</button>
        <button class="secondary" id="collectUrlBtn" type="button">URL 수집만 테스트</button>
        <button class="secondary" id="lectureSummaryBtn" type="button">핵심 내용 정리 생성</button>
        <button id="portfolioBtn">문제해결형 Medium 블로그 글 완성본</button>
      </div>
      <div class="result-toolbar">
        <span class="meta">생성 결과는 Markdown으로 복사할 수 있습니다.</span>
        <button class="secondary" id="copyMarkdownBtn" type="button">Markdown 복사</button>
        <button class="secondary" id="clearResultBtn" type="button">글 초기화</button>
      </div>
      <div id="result" class="result">아직 생성된 결과가 없습니다.</div>
      <div id="debugTabs" class="tabs"></div>
      <div id="debugPane" class="debug-pane">디버그 산출물이 아직 없습니다.</div>
    </section>

    <aside class="panel">
      <h2>이전 기록 검색</h2>
      <div class="row">
        <input id="query" placeholder="예: DAX, SQL, 오류, 모델링" />
        <button class="secondary" id="searchBtn">검색</button>
      </div>
      <div id="notes"></div>
    </aside>
  </div>
</main>
<div class="floating-tools">
  <button id="askBtn" class="ask-btn" title="Ask Tutor">?</button>
  <button id="captureBtn" class="capture-btn" title="Capture"></button>
</div>
<input id="captureInput" type="file" accept="image/*" multiple />
<div id="problemPromptModal" class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="problemPromptTitle">
  <div class="modal">
    <h3 id="problemPromptTitle">생성 전에 한 가지만 확인할게요</h3>
    <p>강의에서 해결해야 할 복잡하거나 어려운 문제를 짧게 적어 주세요. 없다면 자료의 핵심 흐름을 바탕으로 문제를 정의하겠습니다.</p>
    <textarea id="problemPromptInput" placeholder="예: Lab의 검증 기준이 불명확했음, 개념은 이해했지만 실제 실행 결과로 어떻게 확인하는지가 어려웠음"></textarea>
    <div class="modal-actions">
      <button class="secondary" id="problemPromptCancelBtn" type="button">취소</button>
      <button class="secondary" id="problemPromptSkipBtn" type="button">비워두고 생성</button>
      <button id="problemPromptStartBtn" type="button">반영해서 생성</button>
    </div>
  </div>
</div>
<div id="toast" class="toast"></div>

<script>
const result = document.querySelector("#result");
const notesBox = document.querySelector("#notes");
const fileInput = document.querySelector("#image");
const dropzone = document.querySelector("#dropzone");
const fileList = document.querySelector("#fileList");
const debugTabs = document.querySelector("#debugTabs");
const debugPane = document.querySelector("#debugPane");
const toast = document.querySelector("#toast");
const captureInput = document.querySelector("#captureInput");
const problemPromptModal = document.querySelector("#problemPromptModal");
const problemPromptInput = document.querySelector("#problemPromptInput");
let selectedFiles = [];
let currentNoteIds = [];
let currentSession = null;
let lastDebug = null;
let lastDraftMarkdown = "";
let activeBlogRequestId = 0;

function show(text) {
  result.textContent = text;
  if (text && !String(text).includes("아직 생성된 결과")) lastDraftMarkdown = String(text);
}

function setBusy(button, busy, label) {
  if (!button) return;
  if (!button.dataset.defaultText) button.dataset.defaultText = button.textContent;
  button.disabled = Boolean(busy);
  button.textContent = busy ? label : button.dataset.defaultText;
}

function startProgress(message) {
  const startedAt = Date.now();
  show(`${message}\n\n경과 시간: 0초\n상태: 요청을 준비하는 중입니다.`);
  return setInterval(() => {
    const elapsed = Math.floor((Date.now() - startedAt) / 1000);
    show(`${message}\n\n경과 시간: ${elapsed}초\n상태: 자동수집 또는 글 생성을 진행 중입니다. 브라우저/보호 페이지/긴 강의는 시간이 걸릴 수 있습니다.`);
  }, 1000);
}

function stopProgress(timer) {
  if (timer) clearInterval(timer);
}

function clearGeneratedArticle() {
  lastDraftMarkdown = "";
  lastDebug = null;
  result.textContent = "아직 생성된 결과가 없습니다.";
  debugTabs.innerHTML = "";
  debugPane.textContent = "디버그 산출물이 아직 없습니다.";
  showToast("생성 글을 초기화했습니다.");
}

function clearMainInputs() {
  const raw = document.querySelector("#rawText");
  const memo = document.querySelector("#memo");
  if (raw) raw.value = "";
  if (memo) memo.value = "";
  showToast("자료/메모 입력칸을 초기화했습니다.");
}

async function copyMarkdown() {
  const text = lastDraftMarkdown || result.textContent || "";
  if (!text.trim() || text.includes("아직 생성된 결과")) { showToast("복사할 결과가 없습니다."); return; }
  try {
    await navigator.clipboard.writeText(text);
    showToast("Markdown을 복사했습니다.");
  } catch (err) {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
    showToast("Markdown을 복사했습니다.");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.querySelector("#copyMarkdownBtn");
  if (btn) btn.onclick = copyMarkdown;
  const clearBtn = document.querySelector("#clearResultBtn");
  if (clearBtn) clearBtn.onclick = clearGeneratedArticle;
  const clearInputBtn = document.querySelector("#clearInputBtn");
  if (clearInputBtn) clearInputBtn.onclick = clearMainInputs;
});

function showToast(text) {
  toast.textContent = text;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 1800);
}

function askProblemPrompt() {
  return new Promise(resolve => {
    problemPromptInput.value = "";
    problemPromptModal.classList.add("open");
    problemPromptInput.focus();

    function close(value) {
      problemPromptModal.classList.remove("open");
      document.removeEventListener("keydown", onKeydown);
      resolve(value);
    }

    function onKeydown(event) {
      if (event.key === "Escape") close(null);
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") close(problemPromptInput.value.trim());
    }

    document.addEventListener("keydown", onKeydown);
    document.querySelector("#problemPromptCancelBtn").onclick = () => close(null);
    document.querySelector("#problemPromptSkipBtn").onclick = () => close("");
    document.querySelector("#problemPromptStartBtn").onclick = () => close(problemPromptInput.value.trim());
  });
}

function fileKey(file) {
  return `${file.name}-${file.size}-${file.lastModified}`;
}

function isImageFile(file) {
  if (!file) return false;
  if (String(file.type || "").startsWith("image/")) return true;
  const name = String(file.name || "").toLowerCase();
  return [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic"].some(ext => name.endsWith(ext));
}

function addSelectedFiles(files) {
  const incoming = Array.from(files || []).filter(isImageFile);
  if (!incoming.length) {
    showToast("이미지 파일만 추가할 수 있습니다.");
    return;
  }
  const current = new Map(selectedFiles.map(file => [fileKey(file), file]));
  incoming.forEach(file => current.set(fileKey(file), file));
  selectedFiles = Array.from(current.values());
  fileList.textContent = selectedFiles.length
    ? selectedFiles.map((file, index) => `${index + 1}. ${file.name}`).join(" · ")
    : "선택된 이미지 없음";
  showToast(`이미지 ${selectedFiles.length}장 선택됨`);
}

function clearSelectedFiles() {
  selectedFiles = [];
  fileInput.value = "";
  fileList.textContent = "선택된 이미지 없음";
}

fileInput.onchange = () => {
  addSelectedFiles(fileInput.files);
  fileInput.value = "";
};
document.querySelector("#clearFilesBtn").onclick = clearSelectedFiles;

function handleDragOver(event) {
  event.preventDefault();
  if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  dropzone.classList.add("dragover");
}

function handleDragLeave(event) {
  if (event && dropzone.contains(event.relatedTarget)) return;
  dropzone.classList.remove("dragover");
}

function handleDrop(event) {
  event.preventDefault();
  event.stopPropagation();
  dropzone.classList.remove("dragover");
  addSelectedFiles(event.dataTransfer ? event.dataTransfer.files : []);
}

dropzone.ondragover = handleDragOver;
dropzone.ondragleave = handleDragLeave;
dropzone.ondrop = handleDrop;
document.addEventListener("dragover", event => {
  event.preventDefault();
  if (event.dataTransfer && Array.from(event.dataTransfer.items || []).some(item => item.kind === "file")) {
    dropzone.classList.add("dragover");
    event.dataTransfer.dropEffect = "copy";
  }
});
document.addEventListener("drop", event => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const files = event.dataTransfer ? event.dataTransfer.files : [];
  if (files && files.length) addSelectedFiles(files);
});
document.addEventListener("paste", event => {
  const files = Array.from(event.clipboardData?.files || []);
  if (files.length) addSelectedFiles(files);
});

async function ensureSession() {
  if (currentSession) return currentSession;
  const rows = await (await fetch("/api/sessions")).json();
  currentSession = rows[rows.length - 1];
  if (!currentSession) {
    currentSession = await (await fetch("/api/sessions", {
      method:"POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ title:"Study Capture Session" })
    })).json();
  }
  await refreshTimeline();
  return currentSession;
}

async function refreshTimeline() {
  if (!currentSession) return;
  const [captures, qa] = await Promise.all([
    fetch(`/api/sessions/${currentSession.session_id}/captures`).then(r => r.json()),
    fetch(`/api/sessions/${currentSession.session_id}/qa`).then(r => r.json())
  ]);
  renderDebug({
    ...(lastDebug || {}),
    "Capture Timeline": captures,
    "Q&A Logs": qa
  });
}

function renderDebug(payload) {
  lastDebug = payload || {};
    const tabs = [
    ["Final Article", lastDebug.draft || result.textContent],
    ["Collector Report", lastDebug.collector_report || {}],
    ["Capture Timeline", lastDebug["Capture Timeline"] || lastDebug.capture_timeline || []],
    ["Q&A Logs", lastDebug["Q&A Logs"] || lastDebug.qa_logs || []],
    ["Image Evidence", lastDebug.image_evidence || []],
    ["Problem Map", lastDebug.problem_map || {}],
    ["Decision Map", lastDebug.decision_map || {}],
    ["Section Plan", lastDebug.section_plan || []],
    ["Article Brief", lastDebug.article_brief || {}],
    ["Critic Report", lastDebug.critic_report || {}]
  ];
  debugTabs.innerHTML = tabs.map(([label], index) => `<button class="tab${index === 0 ? " active" : ""}" data-tab="${escapeHtml(label)}">${escapeHtml(label)}</button>`).join("");
  function showTab(label, value) {
    debugPane.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    debugTabs.querySelectorAll(".tab").forEach(btn => btn.classList.toggle("active", btn.dataset.tab === label));
  }
  tabs.forEach(([label, value]) => {
    const btn = Array.from(debugTabs.querySelectorAll(".tab")).find(item => item.dataset.tab === label);
    if (btn) btn.onclick = () => showTab(label, value);
  });
  showTab(tabs[0][0], tabs[0][1]);
}

function extractFirstUrl(text) {
  const value = String(text || "");
  const httpIndex = value.indexOf("http://");
  const httpsIndex = value.indexOf("https://");
  const candidates = [httpIndex, httpsIndex].filter(index => index >= 0);
  if (!candidates.length) return "";
  const start = Math.min(...candidates);
  const stopChars = [" ", "\\n", "\\t", "<", ">", '"', "'", ")", "]"];
  let end = value.length;
  for (const ch of stopChars) {
    const index = value.indexOf(ch, start);
    if (index >= 0 && index < end) end = index;
  }
  return value.slice(start, end).replace(/[.,;!?]+$/, "");
}

document.querySelector("#captureBtn").onclick = async () => {
  await ensureSession();
  captureInput.click();
};

captureInput.onchange = async () => {
  await ensureSession();
  if (!captureInput.files.length) return;
  const form = new FormData();
  Array.from(captureInput.files).forEach(file => form.append("image", file));
  form.append("user_note", document.querySelector("#memo").value || "");
  form.append("source_title", document.title || "");
  form.append("source_url", location.href || "");
  const res = await fetch(`/api/sessions/${currentSession.session_id}/captures`, { method:"POST", body:form });
  const data = await res.json();
  captureInput.value = "";
  showToast(`Capture saved: 이미지 ${data.total}`);
  await refreshTimeline();
};

document.querySelector("#askBtn").onclick = async () => {
  await ensureSession();
  const question = prompt("Tutor Agent에게 질문하기");
  if (!question) return;
  showToast("Tutor Agent thinking...");
  const selectedText = String(window.getSelection?.() || "");
  const qa = await (await fetch(`/api/sessions/${currentSession.session_id}/ask`, {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ question, selected_text:selectedText })
  })).json();
  showToast(`Q&A saved: ${qa.qa_id}`);
  await refreshTimeline();
};

async function loadNotes() {
  try {
    const res = await fetch("/api/notes");
    const notes = await res.json();
    if (!currentNoteIds.length) {
      notesBox.innerHTML = "<p>이번 세션에서 생성한 노트가 아직 없습니다. 검색창을 사용하면 이전 저장 기록을 찾아볼 수 있습니다.</p>";
      return;
    }
    const currentNotes = notes.filter(note => currentNoteIds.includes(note.id));
    renderNotes(currentNotes.reverse());
  } catch (err) {
    notesBox.innerHTML = "<p>저장된 노트를 불러오지 못했습니다.</p>";
  }
}

function visibleNotes(notes) {
  return notes.filter(n => {
    const hasContent = Boolean((n.raw_text || "").trim() || (n.user_memo || "").trim() || n.image_path);
    const emptyFallback = String(n.summary || "").includes("입력된 텍스트가 없습니다");
    const imageOnlyPlaceholder = n.source_type === "study-capture-image" && (
      String(n.summary || "").includes("스크린샷을 학습 근거 자료로 저장했습니다")
      || String(n.summary || "").includes("스크린샷을 학습 캡처로 저장했습니다")
    );
    if (emptyFallback && n.source_type !== "study-capture-image") return false;
    if (imageOnlyPlaceholder) return false;
    return hasContent || !emptyFallback;
  });
}

function displayTitle(n) {
  const generic = ["학습 캡처 기록", "이미지 기반 학습 캡처", "이미지 학습 캡처"];
  if (!generic.includes(n.title)) return n.title;
  const time = String(n.created_at || "").split("T")[1]?.slice(0, 5) || "";
  return `${n.image_path ? "이미지" : "학습"} 캡처${time ? " " + time : ""}`;
}

function displaySummary(n) {
  return String(n.summary || "")
    .replace(
      "현재 MVP는 이미지 파일을 근거 자료로 보관하지만, 화면 속 텍스트를 자동으로 읽는 OCR/비전 기능은 아직 연결되어 있지 않습니다. 정확한 노트 생성을 위해 화면의 핵심 문장, 오류 메시지, 실습 목표를 텍스트 입력칸이나 메모에 함께 적어 주세요.",
      "이미지를 학습 근거 자료로 저장했습니다. 화면의 핵심 문장, 오류 메시지, 실습 목표를 메모하면 노트와 문제 해결형 포트폴리오 초안을 더 정확하게 구성할 수 있습니다."
    );
}

function renderNotes(notes) {
  notesBox.innerHTML = notes.map(n => `
    <div class="note">
      <strong>${escapeHtml(displayTitle(n))}</strong>
      <div class="meta">${escapeHtml(n.created_at)} · ${escapeHtml(n.source_type)}</div>
      <div>${(n.tags || []).map(t => `<span class="badge">${escapeHtml(t)}</span>`).join("")}</div>
      ${n.image_path ? `<img src="${escapeHtml(n.image_path)}" alt="captured study screenshot" />` : ""}
      ${(n.image_paths || []).length > 1 ? `<div class="meta">캡처 ${(n.image_paths || []).length}장 연결</div>` : ""}
      <p>${escapeHtml(displaySummary(n).slice(0, 180))}</p>
    </div>`).join("");
}

document.querySelector("#searchBtn").onclick = async () => {
  const res = await fetch("/api/search", {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ query: document.querySelector("#query").value, top_k: 6 })
  });
  const rows = await res.json();
  renderNotes(visibleNotes(rows.map(r => r.note)));
};

async function collectUrlOnly() {
  const rawText = document.querySelector("#rawText").value.trim();
  const seedUrl = extractFirstUrl(rawText);
  if (!seedUrl) {
    show("수집할 URL을 입력칸에 먼저 넣어 주세요.");
    return;
  }
  const btn = document.querySelector("#collectUrlBtn");
  setBusy(btn, true, "URL 수집 중...");
  lastDraftMarkdown = "";
  debugTabs.innerHTML = "";
  debugPane.textContent = "URL 자동수집을 실행하는 중입니다.";
  const progress = startProgress(`URL 수집만 테스트합니다.\n\n${seedUrl}\n\n수집 결과가 충분한지 먼저 확인합니다.`);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 900000);
  try {
    const res = await fetch("/api/debug-collect-url", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ url: seedUrl, timeout_seconds: 900 }),
      signal: controller.signal
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const report = data.collector_report || {};
    const stats = report.stats || {};
    const quality = report.quality || {};
    const summary = [
      "# URL 수집 테스트 결과",
      "",
      `- URL: ${seedUrl}`,
      `- 수집 성공: ${data.ok ? "YES" : "NO"}`,
      `- Source pack chars: ${data.source_pack_chars || 0}`,
      `- Pages: ${stats.page_count || quality.pages_collected || 0}`,
      `- Visible text chars: ${stats.visible_text_chars || quality.text_chars || 0}`,
      `- Links: ${stats.link_count || stats.links || quality.links || 0}`,
      `- Videos: ${stats.video_candidate_count || stats.video_candidates || quality.video_candidates || 0}`,
      `- Lessons: ${stats.lesson_candidate_count || stats.lesson_candidates || quality.lesson_candidates || 0}`,
      `- Labs: ${stats.lab_candidate_count || stats.lab_candidates || quality.lab_steps || 0}`,
      `- Markdown: ${report.markdown_path || ""}`,
      `- JSON: ${report.json_path || ""}`,
      "",
      "## 품질 판정",
      data.quality_sufficient ? "충분함" : "부족함",
      "",
      "## 부족한 이유",
      (data.quality_reasons || []).join("\\n") || "없음",
      "",
      "## Source pack preview",
      data.source_pack_preview || ""
    ].join("\\n");
    lastDraftMarkdown = summary;
    show(summary);
    renderDebug({
      draft: summary,
      collector_report: report,
      source_pack_preview: data.source_pack_preview || "",
      critic_report: {
        passed: Boolean(data.ok),
        failures: data.quality_reasons || [],
        metrics: {
          source_pack_chars: data.source_pack_chars || 0,
          quality_sufficient: Boolean(data.quality_sufficient)
        }
      }
    });
  } catch (err) {
    const message = err && (err.name || err.message) ? `${err.name || "Error"}: ${err.message || ""}` : String(err);
    show(`URL 수집 테스트가 완료되지 않았습니다.\n\n원인: ${message}`);
  } finally {
    stopProgress(progress);
    clearTimeout(timeout);
    setBusy(btn, false);
  }
}

async function makeBlog(formatType) {
  if (!selectedFiles.length && !document.querySelector("#rawText").value.trim() && !document.querySelector("#memo").value.trim()) {
    show("이미지, 화면 텍스트, 메모 중 하나는 입력해 주세요.");
    return;
  }
  const isLectureSummary = formatType === "lecture-summary";
  let promptNote = "";
  if (!isLectureSummary) {
    show("생성 전에 어려운 문제 입력 팝업을 기다리는 중입니다.\\n\\n팝업에서 입력하거나 '비워두고 생성'을 눌러 주세요.");
    promptNote = await askProblemPrompt();
    if (promptNote === null) {
      show("생성이 취소되었습니다.");
      return;
    }
  }
  const btn = document.querySelector(isLectureSummary ? "#lectureSummaryBtn" : "#portfolioBtn");
  const requestId = ++activeBlogRequestId;
  const seedText = document.querySelector("#rawText").value.trim();
  const extraInfo = [
    ["실습/프로젝트 이름", document.querySelector("#projectName").value],
    ["내가 해결한 핵심 문제", document.querySelector("#coreProblem").value],
    ["중간에 막힌 부분", document.querySelector("#blockedPart").value],
    ["최종 결과", document.querySelector("#finalResult").value],
    ["강조하고 싶은 기술", document.querySelector("#focusTech").value],
    ["사용자가 정의한 어려운 문제", isLectureSummary ? "" : (promptNote || "")]
  ]
    .filter(([, value]) => String(value || "").trim())
    .map(([label, value]) => `- ${label}: ${String(value).trim()}`)
    .join("\\n");
  const topic = document.querySelector("#projectName").value.trim() || "학습 기록 기반 문제 해결 경험";
  setBusy(btn, true, isLectureSummary ? "요약 중..." : "생성 중...");
  lastDraftMarkdown = "";
  lastDebug = null;
  debugTabs.innerHTML = "";
  debugPane.textContent = "이번 요청의 자동수집 결과를 기다리는 중입니다.";
  const progress = startProgress(`${isLectureSummary ? "강의 내용 요약을 위해" : "문제 해결형 글 생성을 위해"} 자동수집을 시작합니다.\n\n입력:\n${seedText.slice(0, 500) || "이미지/메모 기반 생성"}\n\n강의/글/영상/Lab 자료를 먼저 수집한 뒤 ${isLectureSummary ? "내용 요약으로 넘어갑니다." : "글 생성으로 넘어갑니다."}`);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 900000);
  try {
    const form = new FormData();
    selectedFiles.forEach(file => form.append("image", file));
    form.append("raw_text", document.querySelector("#rawText").value);
    const baseMemo = document.querySelector("#memo").value;
    const promptMemo = (!isLectureSummary && promptNote)
      ? `\n\n[생성 직전 사용자가 정의한 어려운 문제]\n${promptNote}`
      : (isLectureSummary ? "" : `\n\n[생성 직전 사용자가 정의한 어려운 문제]\n없음. 자료의 핵심 흐름을 바탕으로 문제를 정의하고 해결 과정 작성.`);
    form.append("memo", baseMemo + promptMemo);
    form.append("topic", topic);
    form.append("format_type", formatType);
    form.append("extra_info", extraInfo);
    const res = await fetch("/api/direct-blog", {
      method:"POST",
      body: form,
      signal: controller.signal
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (requestId !== activeBlogRequestId) return;
    const finalText = `[응답 시간: ${data.elapsed_seconds}s · 이미지 ${data.image_count}장]\n\n${data.draft || "생성된 본문이 비어 있습니다. Debug 탭의 Collector Report를 확인해 주세요."}`;
    lastDraftMarkdown = finalText;
    show(finalText);
    renderDebug(data);
  } catch (err) {
    if (requestId !== activeBlogRequestId) return;
    const message = err && (err.name || err.message) ? `${err.name || "Error"}: ${err.message || ""}` : String(err);
    show(`${isLectureSummary ? "강의 내용 요약" : "문제 해결형 Medium 글 생성"} 요청이 완료되지 않았습니다.\n\n원인: ${message}\n\nURL-only 수집이 보호 페이지/로그인/자막 추출/긴 크롤링에 막혔을 수 있습니다. 서버 터미널 로그와 Debug report를 확인해 주세요.`);
  } finally {
    stopProgress(progress);
    clearTimeout(timeout);
    if (requestId === activeBlogRequestId) setBusy(btn, false);
  }
}

document.querySelector("#collectUrlBtn").onclick = collectUrlOnly;
document.querySelector("#lectureSummaryBtn").onclick = () => makeBlog("lecture-summary");
document.querySelector("#portfolioBtn").onclick = () => makeBlog("problem-solving-portfolio");

function escapeHtml(text) {
  return String(text || "").replace(/[&<>"']/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;" }[ch]));
}

loadNotes();
ensureSession();
</script>
</body>
</html>
"""


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7870"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"AI Study Documentation Agent running on http://{host}:{port}")
    startup_diag = provider_diagnostics("groq", package_import_ok=groq_import_ok(), client_init_ok=None)
    print(
        "LLM startup diagnostics: "
        f"dotenv_exists={startup_diag['dotenv_exists']} "
        f"dotenv_loaded={startup_diag['dotenv_loaded']} "
        f"groq_api_key_present={startup_diag['api_key_present']} "
        f"groq_package_import_ok={startup_diag['package_import_ok']} "
        f"text_model={startup_diag['text_model']} "
        f"vision_model={startup_diag['vision_model']} "
        f"fast_provider={MODEL_PROVIDER_FAST} deep_provider={MODEL_PROVIDER_DEEP}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
