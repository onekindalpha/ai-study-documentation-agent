from app.main import (
    build_sparse_capture_report,
    classify_article_type_with_confidence,
    critique_article,
    sparse_capture_generation_blocker,
    specific_core_problem_candidate,
    specific_title_candidate,
)


def assert_type(text, expected):
    result = classify_article_type_with_confidence(text, "", "smoke", "", [])
    assert result["article_type"] in expected, result


def test_python_error_not_powerbi():
    text = "Python traceback TypeError list index out of range stack trace debugging fix verification"
    assert_type(text, {"code_error_debugging", "python_algorithm_learning"})
    article = "# Python 오류 디버깅\n\n## 문제 인식\nPython TypeError가 발생했다.\n\n원인은 인덱스 범위였다.\n\n## 문제 정의\n입력 배열 길이를 벗어난 접근이었다.\n\n재현 가능한 오류였다.\n\n## 왜 이것을 문제로 인식했는가\nstack trace가 같은 라인을 가리켰다.\n\n수정 후 테스트가 필요했다.\n\n## 문제 해결 경험\n### 1. index guard\n문제/제약: 범위를 벗어났다.\n\n원인 판단: 조건문이 없었다.\n\n조치: guard를 추가했다.\n\n확인 결과: 테스트가 통과했다.\n\n## Portfolio Summary\nThis debugging note explains the cause.\n\nIt verifies the fix.\n\n## Key skills practiced\n- Debugging\n- Testing\n- Python\n- Root cause\n- Verification\n- Error reading\n- Refactoring\n- Documentation\n\n## 이미지 번호와 캡션 목록\n- 이미지 없음\n\nProductKey DAX Power Query"
    report = critique_article(article, "code_error_debugging", {"article_type": "code_error_debugging", "_section_plan": [{"section": "debug", "image_refs": [], "must_include": []}]}, [])
    assert not report.passed
    assert "anti_overfitting" in report.section_failures or "unsupported_claims" in report.section_failures


def test_github_readme_not_powerbi():
    text = "GitHub README Markdown video embed image asset path preview repository badge not rendering"
    assert_type(text, {"github_readme_debugging"})


def test_badge_not_error_debugging():
    text = "course learning path module completed certification badge credential progress lecture summary"
    result = classify_article_type_with_confidence(text, "", "smoke", "", [])
    assert result["article_type"] in {"learning_path_reflection", "certification_badge_summary"}, result
    assert result["article_type"] != "code_error_debugging"


def test_deployment_not_relationship():
    text = "deployment build log server hosting environment variable ci failed health check verification"
    result = classify_article_type_with_confidence(text, "", "smoke", "", [])
    assert result["article_type"] == "deployment_debugging", result
    assert result["article_type"] != "semantic_model_relationship"


def test_cloud_lab_not_powerbi():
    text = "AWS cloud lab IAM policy S3 bucket Lambda resource permission hands-on module verification"
    result = classify_article_type_with_confidence(text, "", "smoke", "", [])
    assert result["article_type"] == "cloud_lab_practice", result
    assert result["article_type"] not in {"semantic_model_relationship", "dax_measure_modeling", "power_query_etl"}


def test_ai_project_not_powerbi():
    text = "AI project build log RAG embedding vector database chatbot agent prompt evaluation"
    result = classify_article_type_with_confidence(text, "", "smoke", "", [])
    assert result["article_type"] == "ai_project_build_log", result
    assert result["article_type"] not in {"semantic_model_relationship", "dax_measure_modeling", "power_query_etl"}


def test_unseen_non_powerbi_article_rejects_powerbi_terms():
    article = "# GitHub README 영상 삽입 문제 해결\n\n## 문제 인식\nREADME에서 영상 미리보기가 표시되지 않았다.\n\nasset path와 Markdown 문법을 확인해야 했다.\n\n## 문제 정의\n문제는 repository 문서 렌더링이었다.\n\nGitHub preview 기준으로 검증했다.\n\n## 왜 이것을 문제로 인식했는가\n로컬에서는 보이지만 GitHub에서는 경로가 달라질 수 있다.\n\nREADME는 프로젝트 첫 화면이므로 재현 가능한 문서가 중요했다.\n\n## 문제 해결 경험\n### 1. asset path 확인\n문제/제약: README 미디어가 깨졌다.\n\n원인 판단: 상대 경로가 repository 기준과 맞지 않았다.\n\n조치: assets 폴더 경로를 수정했다.\n\n확인 결과: GitHub preview에서 이미지가 보였다.\n\n## Portfolio Summary\nThis README debugging note explains a documentation rendering issue.\n\nIt validates the fix through repository preview.\n\n## Key skills practiced\n- Markdown debugging\n- GitHub preview validation\n- Asset path reasoning\n- Documentation quality\n- Reproduction\n- Root cause analysis\n- Verification\n- Technical writing\n\n## 이미지 번호와 캡션 목록\n- 이미지 없음\n\nDAX ProductKey Power Query"
    report = critique_article(
        article,
        "github_readme_debugging",
        {"article_type": "github_readme_debugging", "_section_plan": [{"section": "README", "image_refs": [], "must_include": ["GitHub", "README"]}]},
        [],
    )
    assert not report.passed
    assert "anti_overfitting" in report.section_failures or "unsupported_claims" in report.section_failures


def test_unseen_sparse_capture_report():
    text = "GitHub Copilot agent workflow course lab prompt repository pull request review automation"
    classification = classify_article_type_with_confidence(text, "", "Copilot agent workflow lab", "", ["01_course_goal.png", "02_agent_flow.png", "03_pr_result.png"])
    evidence = [
        {
            "image_no": 1,
            "caption": "이미지 1 - Copilot agent workflow lab 목표 화면",
            "role": "problem",
            "problem_signal": "agent workflow course lab goal",
            "technical_entities": ["GitHub", "Copilot", "agent workflow", "repository"],
            "inferred_meaning": "강의 목표와 실습 맥락을 보여주는 화면",
        },
        {
            "image_no": 2,
            "caption": "이미지 2 - repository에서 agent workflow 구성을 확인하는 화면",
            "role": "solution",
            "problem_signal": "workflow configuration",
            "technical_entities": ["GitHub", "workflow", "pull request"],
            "inferred_meaning": "작업 흐름 또는 설정 변경을 보여주는 화면",
        },
        {
            "image_no": 3,
            "caption": "이미지 3 - pull request review 결과를 확인하는 화면",
            "role": "validation",
            "problem_signal": "review result",
            "technical_entities": ["pull request", "review", "automation"],
            "inferred_meaning": "검증 또는 결과 화면",
        },
    ]
    report = build_sparse_capture_report(classification["article_type"], classification, evidence, text, "")
    assert report["article_type"] not in {"semantic_model_relationship", "dax_measure_modeling", "power_query_etl"}, report
    assert report["generation_mode"] in {"full_article", "draft_with_missing_context", "ask_before_generate"}, report
    assert report["capture_coverage"]["has_goal"] is True
    assert report["capture_coverage"]["has_action"] is True
    assert report["capture_coverage"]["has_validation"] is True
    dumped = str(report)
    for forbidden in ["ProductKey", "DAX", "Power Query", "Sales[Sales]", "TargetAmount", "MonthKey", "Date table"]:
        assert forbidden not in dumped


def test_unseen_sparse_github_agentic_workflow_blocks_full_article():
    names = [f"{index:02d}_capture.png" for index in range(1, 22)]
    captions = [
        "Agentic Workflows: Automation That Actually Reads the Room",
        "update-github-info workflow file",
        "activation state",
        "update-github-info.lock.yml",
        "conclusion",
        "workflow_dispatch trigger",
    ] + [f"screen_{index}.png" for index in range(7, 22)]
    evidence = []
    for index, caption in enumerate(captions, start=1):
        interpreted = index <= 6
        evidence.append(
            {
                "image_no": index,
                "caption": f"이미지 {index} - {caption}",
                "visible_evidence": [caption],
                "technical_entities": ["GitHub Actions", "workflow_dispatch"] if interpreted else [],
                "role": "solution" if index in {2, 3, 4} else ("validation" if index in {5, 6} else "problem"),
                "problem_signal": caption,
                "inferred_meaning": caption,
                "confidence": 0.8 if interpreted else 0.35,
                "evidence_source": "vision" if interpreted else "filename",
            }
        )
    classification = classify_article_type_with_confidence(
        "GitHub Agentic Workflows workflow_dispatch activation conclusion update-github-info.lock.yml",
        "",
        "",
        "",
        names,
    )
    report = build_sparse_capture_report(classification["article_type"], classification, evidence, "", "")
    problem_map = {
        "core_problem": "학습 기록 기반 문제 해결 경험 과정에서 관찰한 결과와 의도한 분석 흐름의 불일치",
        "solution_steps": [{"title": "근거 화면 1 기반 검증 단계"}],
        "_section_plan": [{"section": "문제 인식", "image_refs": [1]}],
    }
    brief = {"korean_title": "학습 기록 기반 문제 해결 경험: 문제를 검증 가능한 분석 흐름으로 바꾼 기록"}
    reason = sparse_capture_generation_blocker(classification["article_type"], report, problem_map, brief)
    title = specific_title_candidate(classification["article_type"], evidence, brief["korean_title"])
    core = specific_core_problem_candidate(classification["article_type"], evidence, problem_map["core_problem"])
    assert classification["article_type"] == "github_agentic_workflow", classification
    assert report["generation_mode"] == "draft_with_missing_context", report
    assert report["interpreted_image_count"] == 6, report
    assert report["total_image_count"] == 21, report
    assert report["unknown_caption_count"] == 15, report
    assert report["capture_roles"][6]["role"] == "unknown", report["capture_roles"][6]
    assert reason
    assert "GitHub Agentic Workflows" in title and "workflow_dispatch" in title
    assert "workflow_dispatch" in core and "conclusion" in core
    dumped = str(report) + title + core
    for forbidden in ["ProductKey", "DAX", "Power Query", "Sales[Sales]", "TargetAmount", "MonthKey", "Date table"]:
        assert forbidden not in dumped


if __name__ == "__main__":
    test_python_error_not_powerbi()
    test_github_readme_not_powerbi()
    test_badge_not_error_debugging()
    test_deployment_not_relationship()
    test_cloud_lab_not_powerbi()
    test_ai_project_not_powerbi()
    test_unseen_non_powerbi_article_rejects_powerbi_terms()
    test_unseen_sparse_capture_report()
    test_unseen_sparse_github_agentic_workflow_blocks_full_article()
    print("article type smoke tests passed")
