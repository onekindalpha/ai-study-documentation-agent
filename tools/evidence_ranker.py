from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


PROBLEM_KEYWORDS = {
    "복잡도": 18,
    "시간 복잡도": 20,
    "공간 복잡도": 18,
    "프로세스": 16,
    "스레드": 16,
    "스케줄링": 18,
    "동기화": 18,
    "교착": 18,
    "메모리": 14,
    "가상 메모리": 18,
    "캐시": 14,
    "TCP": 18,
    "UDP": 16,
    "HTTP": 14,
    "REST": 14,
    "트랜잭션": 18,
    "조인": 18,
    "정규화": 14,
    "인덱스": 16,
    "스택": 16,
    "큐": 16,
    "해시": 18,
    "트리": 18,
    "그래프": 18,
    "재귀": 18,
    "정렬": 14,
    "탐색": 14,
    "API": 14,
    "인증": 14,
    "권한": 14,
    "오류": 16,
    "예외": 14,
    "실습": 16,
    "구현": 16,
    "코드": 14,
    "쿼리": 14,
    "모델": 12,
    "데이터": 10,
    "파이프라인": 14,
}

ANTI_KEYWORDS = {
    "Home": -30,
    "Devlog": -30,
    "Backend Engineer": -40,
    "김영찬": -40,
    "profile": -30,
    "portfolio": -20,
}


@dataclass
class EvidenceItem:
    rank: int
    score: float
    title: str
    url: str
    node_type: str
    text_chars: int
    reason: list[str]
    excerpt: str
    meta: dict[str, Any]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def flatten_nodes(nodes: list[dict[str, Any]], parent_title: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for node in nodes:
        item = dict(node)
        if parent_title:
            item.setdefault("meta", {})
            item["meta"]["parent_title"] = parent_title
        out.append(item)

        children = node.get("children") or []
        if isinstance(children, list):
            out.extend(flatten_nodes(children, parent_title=node.get("title") or parent_title))

    return out


def keyword_score(title: str, text: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    haystack = f"{title}\n{text}"

    for kw, weight in PROBLEM_KEYWORDS.items():
        if kw in haystack:
            score += weight
            reasons.append(f"핵심 개념 키워드: {kw}")

    for kw, weight in ANTI_KEYWORDS.items():
        if kw in haystack:
            score += weight
            reasons.append(f"목차 외/프로필성 키워드 감점: {kw}")

    return score, reasons


def code_signal_score(text: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    patterns = [
        r"\bdef\s+\w+\(",
        r"\bclass\s+\w+",
        r"\bimport\s+\w+",
        r"\bSELECT\b|\bJOIN\b|\bWHERE\b",
        r"\bcurl\b",
        r"```",
        r"\bfor\s+.+\bin\b",
        r"\bwhile\s+",
        r"\breturn\b",
    ]

    hits = 0
    for p in patterns:
        if re.search(p, text, re.I):
            hits += 1

    if hits:
        score += min(24, hits * 6)
        reasons.append(f"코드/쿼리/구현 신호 {hits}개")

    return score, reasons


def structure_score(node: dict[str, Any], text: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    node_type = node.get("type") or node.get("node_type") or ""

    if node_type in {"oopy_toc_page", "toc_page"}:
        score += 8
        reasons.append("목차 기반 수집 페이지")

    if "transcript" in node_type:
        score += 6
        reasons.append("강의 transcript 근거")

    if "lab" in node_type or "exercise" in node_type:
        score += 16
        reasons.append("실습/과제 근거")

    text_chars = len(text)

    if text_chars >= 8000:
        score += 18
        reasons.append("본문 근거가 충분히 김")
    elif text_chars >= 3000:
        score += 12
        reasons.append("본문 근거가 중간 이상")
    elif text_chars >= 800:
        score += 6
        reasons.append("본문 근거가 있음")
    else:
        score -= 8
        reasons.append("본문 근거가 짧음")

    if re.search(r"(왜|문제|해결|비교|차이|장단점|주의|핵심|면접|동작|원리)", text):
        score += 10
        reasons.append("문제 인식/비교/원리 설명 신호")

    return score, reasons


def score_node(node: dict[str, Any]) -> tuple[float, list[str]]:
    title = node.get("title") or ""
    text = clean_text(node.get("text") or "")

    score = 0.0
    reasons: list[str] = []

    s, r = keyword_score(title, text)
    score += s
    reasons.extend(r)

    s, r = code_signal_score(text)
    score += s
    reasons.extend(r)

    s, r = structure_score(node, text)
    score += s
    reasons.extend(r)

    if title and title in text[:500]:
        score += 3
        reasons.append("제목과 본문이 일치")

    if not text:
        score -= 30
        reasons.append("본문 없음")

    return score, reasons


def make_excerpt(text: str, max_chars: int = 520) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def build_article_brief(graph: dict[str, Any], items: list[EvidenceItem]) -> str:
    title = graph.get("title") or "Untitled source"
    url_type = graph.get("url_type") or ""
    input_url = graph.get("input_url") or ""
    quality = graph.get("quality") or {}

    focus = items[0] if items else None
    top_titles = [x.title for x in items[:5]]

    lines: list[str] = []

    lines.append("# Article Brief")
    lines.append("")
    lines.append("## Source")
    lines.append(f"- Title: {title}")
    lines.append(f"- URL type: {url_type}")
    lines.append(f"- Input URL: {input_url}")
    lines.append(f"- Pages collected: {quality.get('pages_collected')}")
    lines.append(f"- Text chars: {quality.get('text_chars')}")
    lines.append(f"- Can generate article: {quality.get('can_generate_article')}")
    lines.append("")

    if "toc_coverage" in quality:
        lines.append("## Coverage")
        lines.append(f"- TOC candidates: {quality.get('toc_candidates')}")
        lines.append(f"- TOC collected: {quality.get('toc_collected')}")
        lines.append(f"- TOC missing: {quality.get('toc_missing')}")
        lines.append(f"- TOC coverage: {quality.get('toc_coverage')}")
        lines.append(f"- Extra outside TOC: {quality.get('extra_collected_outside_toc')}")
        lines.append("")

    lines.append("## Writer role guard")
    lines.append("- 글의 주체는 크롤러/AI 시스템 개발자가 아니라 학습자다.")
    lines.append("- 문제는 수집기 구현 문제가 아니라 학습 중 어렵거나 복잡했던 개념, 실습, 비교, 원리 이해 지점이다.")
    lines.append("- 근거 없는 성과, 코드, 수식, 결과는 만들지 않는다.")
    lines.append("- 사용자 메모가 있으면 사용자 메모를 최우선 문제로 삼는다.")
    lines.append("")

    if focus:
        lines.append("## Selected focus")
        lines.append(f"- Focus title: {focus.title}")
        lines.append(f"- Focus URL: {focus.url}")
        lines.append(f"- Score: {focus.score}")
        lines.append(f"- Why selected: {', '.join(focus.reason[:6])}")
        lines.append("")
        lines.append("## Problem framing candidate")
        lines.append(f"이 학습 기록의 중심 문제는 `{focus.title}`를 단순 암기 항목으로 넘기지 않고, 왜 중요한 개념인지, 어떤 조건에서 헷갈리기 쉬운지, 어떻게 구조화해서 이해했는지를 정리하는 것이다.")
        lines.append("")
        lines.append("## Suggested problem-solving stages")
        lines.append("1. 개념을 그대로 외우는 대신, 핵심 조건과 비교 기준을 분리한다.")
        lines.append("2. 비슷한 개념과의 차이를 표나 기준값 중심으로 다시 정리한다.")
        lines.append("3. 면접/실무 질문으로 바뀌었을 때 설명 가능한 형태로 재구성한다.")
        lines.append("4. 필요한 경우 예시, 코드, 흐름도, 이미지 근거를 붙여 검증한다.")
        lines.append("")

    lines.append("## Top evidence sections")
    for item in items:
        lines.append(f"### {item.rank}. {item.title}")
        lines.append(f"- Score: {item.score}")
        lines.append(f"- Type: {item.node_type}")
        lines.append(f"- Text chars: {item.text_chars}")
        lines.append(f"- URL: {item.url}")
        lines.append(f"- Reasons: {', '.join(item.reason[:8])}")
        lines.append("")
        lines.append(item.excerpt)
        lines.append("")

    lines.append("## Medium template mapping")
    lines.append("1. 한국어 제목: 선택된 focus 개념을 문제해결형 제목으로 바꾼다.")
    lines.append("2. 영어 부제: 학습 주제와 문제해결 관점을 짧게 번역한다.")
    lines.append("3. 짧은 도입부: 왜 이 부분을 그냥 넘기기 어려웠는지 쓴다.")
    lines.append("4. 핵심 작업 요약: 수집된 source의 범위와 학습한 개념을 요약한다.")
    lines.append("5. 문제 인식: focus 개념에서 헷갈린 지점을 쓴다.")
    lines.append("6. 문제 정의: 무엇을 구분하거나 설명할 수 있어야 하는지 정의한다.")
    lines.append("7. 왜 이것을 문제로 인식했는가: 면접/실무/구현 관점의 이유를 쓴다.")
    lines.append("8. 문제 해결 경험 1, 2, 3: 기준 분리, 비교, 재구성 순서로 쓴다.")
    lines.append("9. 복잡한 수식/코드/흐름 해결 경험: 근거가 있을 때만 쓴다.")
    lines.append("10. 성과: 설명 가능해진 점, 정리 산출물, 이해 구조를 쓴다.")
    lines.append("11. 사용한 주요 수식/코드 정리: evidence에 있는 경우만 넣는다.")
    lines.append("12. 최종 정리")
    lines.append("13. Portfolio Summary")
    lines.append("14. Key skills practiced")
    lines.append("15. 이미지 번호/캡션 목록")
    lines.append("")

    lines.append("## Candidate focus list")
    for t in top_titles:
        lines.append(f"- {t}")
    lines.append("")

    return "\n".join(lines)


def run(run_dir: Path, top_k: int) -> int:
    graph_path = run_dir / "source_graph.json"
    if not graph_path.exists():
        print("source_graph.json not found:", graph_path)
        return 1

    graph = json.loads(graph_path.read_text())
    nodes = flatten_nodes(graph.get("nodes") or [])

    scored = []
    for node in nodes:
        text = clean_text(node.get("text") or "")
        score, reasons = score_node(node)
        title = node.get("title") or ""
        url = node.get("url") or ""
        node_type = node.get("type") or node.get("node_type") or ""

        if node_type in {"oopy_root"}:
            continue

        scored.append(
            {
                "score": round(score, 2),
                "title": title,
                "url": url,
                "node_type": node_type,
                "text_chars": len(text),
                "reason": reasons,
                "excerpt": make_excerpt(text),
                "meta": node.get("meta") or {},
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)

    items = [
        EvidenceItem(rank=i, **item)
        for i, item in enumerate(scored[:top_k], 1)
    ]

    output = {
        "run_dir": str(run_dir),
        "source_title": graph.get("title"),
        "url_type": graph.get("url_type"),
        "quality": graph.get("quality"),
        "selected_focus": asdict(items[0]) if items else None,
        "ranked_items": [asdict(x) for x in items],
    }

    (run_dir / "evidence_rank.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    (run_dir / "article_brief.md").write_text(build_article_brief(graph, items))

    print("== EVIDENCE RANKER RESULT ==")
    print("run_dir:", run_dir)
    print("source_title:", graph.get("title"))
    print("url_type:", graph.get("url_type"))
    print("items:", len(items))
    print("files:")
    print("-", run_dir / "evidence_rank.json")
    print("-", run_dir / "article_brief.md")
    print()

    for item in items:
        print(f"{item.rank:02d}. {item.title}")
        print("    score:", item.score)
        print("    type:", item.node_type)
        print("    chars:", item.text_chars)
        print("    reasons:", "; ".join(item.reason[:4]))
        print()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    return run(Path(args.run_dir), args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
