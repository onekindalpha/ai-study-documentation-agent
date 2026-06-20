import unittest

from app.main import (
    apply_final_article_policy,
    current_run_image_policy_context,
    final_article_policy_failures,
    groq_retry_details,
    vision_rate_limit_response,
)


class ImageUploadFinalPolicyTests(unittest.TestCase):
    def test_grounded_power_bi_vision_evidence_allows_image_article(self) -> None:
        result = {
            "image_evidence": [
                {
                    "caption": "Power BI Desktop에서 DAX measure를 작성한 화면",
                    "visible_evidence": ["Power BI", "DAX measure", "Variance Margin"],
                    "technical_entities": ["Power BI", "DAX measure"],
                    "problem_signal": "측정값 계산 검증",
                    "inferred_meaning": "현재 이미지의 계산 단계",
                    "confidence": 0.9,
                    "evidence_source": "vision",
                }
            ]
        }
        context = current_run_image_policy_context(result)
        article = "# Power BI 실습\nDAX measure와 Variance Margin을 검증했다."

        self.assertEqual(final_article_policy_failures(article, contamination_context=context), [])

    def test_unrelated_topic_is_still_blocked(self) -> None:
        result = {
            "image_evidence": [
                {
                    "caption": "Docker 컨테이너 실행 화면",
                    "visible_evidence": ["docker run", "container"],
                    "technical_entities": ["Docker"],
                    "confidence": 0.9,
                    "evidence_source": "vision",
                }
            ]
        }
        context = current_run_image_policy_context(result)
        article = "# Power BI 실습\nDAX measure를 작성했다."

        failures = final_article_policy_failures(article, contamination_context=context)
        self.assertTrue(any("stale-run topic contamination" in failure for failure in failures))

    def test_filename_fallback_cannot_whitelist_topic(self) -> None:
        result = {
            "image_evidence": [
                {
                    "caption": "Power BI DAX measure",
                    "visible_evidence": ["Power BI"],
                    "technical_entities": ["DAX measure"],
                    "confidence": 0.35,
                    "evidence_source": "filename",
                }
            ]
        }
        self.assertEqual(current_run_image_policy_context(result), "")

    def test_image_evidence_does_not_drive_text_topic_classifier(self) -> None:
        result = {
            "image_evidence": [
                {
                    "primary_topic": "ETL 변환과 모델 계산식 구분",
                    "platform_or_product": "Power BI",
                    "topic_terms": ["Power Query", "DAX measure", "Custom Column"],
                    "caption": "Power Query transformation과 aggregation 확인 화면",
                    "visible_evidence": ["Power Query Editor", "aggregation", "transformation"],
                    "technical_entities": ["Power BI", "Power Query"],
                    "confidence": 0.9,
                    "evidence_source": "vision",
                }
            ]
        }
        context = current_run_image_policy_context(result)
        article = "# Power BI Power Query ETL\nDAX measure가 아니라 Custom Column을 사용했다."

        failures = final_article_policy_failures(article, contamination_context=context)
        self.assertFalse(any("pandas_groupby" in failure for failure in failures))

    def test_vision_topic_contract_allows_parent_product_without_manual_dax_mapping(self) -> None:
        result = {
            "image_evidence": [
                {
                    "primary_topic": "날짜 테이블과 측정값 모델링",
                    "platform_or_product": "Power BI",
                    "topic_terms": ["DAX", "time intelligence", "measure"],
                    "caption": "MonthKey와 Date table을 설정한 화면",
                    "visible_evidence": ["MonthKey", "Date table", "Variance Margin"],
                    "technical_entities": ["CALENDARAUTO", "Avg Price"],
                    "confidence": 0.9,
                    "evidence_source": "vision",
                },
                {
                    "primary_topic": "측정값 기반 목표 대비 분석",
                    "platform_or_product": "Power BI",
                    "topic_terms": ["DAX", "target measure", "variance"],
                    "caption": "Target과 Variance 결과를 확인한 화면",
                    "visible_evidence": ["Target", "Variance", "Variance Margin"],
                    "technical_entities": ["Target measure"],
                    "confidence": 0.9,
                    "evidence_source": "vision",
                }
            ]
        }
        context = current_run_image_policy_context(result)
        article = "# Power BI DAX 실습\nMonthKey와 Date table을 구성했다."

        self.assertEqual(final_article_policy_failures(article, contamination_context=context), [])

    def test_single_unseen_parent_product_claim_is_not_trusted(self) -> None:
        result = {
            "image_evidence": [
                {
                    "primary_topic": "컨테이너 실행",
                    "platform_or_product": "Power BI",
                    "topic_terms": ["container"],
                    "caption": "Docker 컨테이너 실행 화면",
                    "visible_evidence": ["docker run", "container"],
                    "technical_entities": ["Docker"],
                    "confidence": 0.9,
                    "evidence_source": "vision",
                }
            ]
        }
        context = current_run_image_policy_context(result)
        failures = final_article_policy_failures("# Power BI 실습", contamination_context=context)

        self.assertTrue(any("power bi" in failure for failure in failures))

    def test_rate_limit_message_exposes_retry_time_and_skips_article_policy(self) -> None:
        exc = RuntimeError(
            "Error code: 429 rate_limit_exceeded. Limit 500000, Used 498565, "
            "Requested 6982. Please try again in 15m58.5216s."
        )
        details = groq_retry_details(exc)
        self.assertEqual(details["retry_after_seconds"], 959)
        result = vision_rate_limit_response(exc, image_count=18)
        checked = apply_final_article_policy(result, current_text="")

        self.assertEqual(checked["mode"], "vision_rate_limit")
        self.assertIn("약 15분 59초 후", checked["draft"])
        self.assertNotIn("최종 글 검수 실패", checked["draft"])


if __name__ == "__main__":
    unittest.main()
