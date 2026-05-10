import unittest

from chatbot_bc.chat_handler import build_analysis_prompt


class ChatbotContextPromptTest(unittest.TestCase):
    def test_build_analysis_prompt_includes_analysis_context(self):
        prompt = build_analysis_prompt(
            {
                "overall_bias_score": 0.73,
                "opinion_score": 0.68,
                "emotion_score": 0.45,
                "headline_body_gap_score": 0.2,
                "tone_label": "NEGATIVE",
                "summary_text": "부정적 프레임을 중심으로 설명합니다.",
                "perspective_summary": "특정 관점을 강조합니다.",
                "evidence_summary": "감정적 표현이 반복됩니다.",
                "keywords": [{"keyword_text": "탄핵"}],
                "evidences": [{"description": "부정적 표현 반복"}],
            }
        )

        self.assertIn("[분석 대상 영상 데이터]", prompt)
        self.assertIn("전체 편향 점수: 0.73", prompt)
        self.assertIn("주요 키워드: 탄핵", prompt)
        self.assertIn("편향 근거: 부정적 표현 반복", prompt)


if __name__ == "__main__":
    unittest.main()
