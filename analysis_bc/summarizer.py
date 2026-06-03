from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from anthropic import Anthropic
from anthropic.types import TextBlock
from dotenv import load_dotenv

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto

logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_FALLBACK: dict = {
    "summary_text": "",
    "perspective_summary": "",
    "evidence_summary": "",
    "tone_label": "",
}


class BiasSummarizer:
    def __init__(self) -> None:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            logger.warning("BiasSummarizer init: ANTHROPIC_API_KEY is empty; LLM summaries will fall back")
        else:
            logger.debug("BiasSummarizer init: key_len=%d prefix=%s", len(key), key[:12])
        self.client = Anthropic()  # SDK가 ANTHROPIC_API_KEY 환경변수를 직접 읽음
        self.model = (
            os.environ.get("ANTHROPIC_SUMMARY_MODEL")
            or os.environ.get("ANTHROPIC_MODEL")
            or "claude-haiku-4-5-20251001"
        )

    def summarize(
        self,
        fact_sentences: list[ClassifiedSentenceDto],
        opinion_sentences: list[ClassifiedSentenceDto],
        title: str,
        language: str,
        span_labels: list[SpanLabelDto] | None = None,
    ) -> dict:
        text = ""
        try:
            prompt = self._build_prompt(
                fact_sentences=fact_sentences,
                opinion_sentences=opinion_sentences,
                title=title,
            )
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            block = response.content[0]
            if not isinstance(block, TextBlock) and not hasattr(block, "text"):
                raise ValueError(f"Unexpected content block type: {type(block)}")
            text = block.text.strip()
            logger.debug("summarize raw response: %s", text)
            start = text.find("{")
            end = text.rfind("}") + 1
            logger.debug("summarize json slice: %s", text[start:end])
            result: dict = json.loads(text[start:end])
            logger.debug("summarize result: %s", result)
            if not str(result.get("summary_text", "")).strip():
                return dict(_FALLBACK)
            return result
        except json.JSONDecodeError as e:
            logger.error("summarize json parse failed: %s | raw: %s", e, text)
            return dict(_FALLBACK)
        except Exception as e:
            logger.error("summarize failed: %s", type(e).__name__, exc_info=True)
            return dict(_FALLBACK)

    def summarize_score_reason(
        self,
        *,
        overall_bias_score: float,
        opinion_score: float,
        emotion_score: float,
        fact_ratio: float,
        headline_body_gap_score: float | None,
        score_evidence: str,
        opinion_sentences: list[ClassifiedSentenceDto],
        span_labels: list[SpanLabelDto],
        language: str,
    ) -> str:
        fallback = self._build_score_reason_fallback(
            overall_bias_score=overall_bias_score,
            opinion_score=opinion_score,
            emotion_score=emotion_score,
            fact_ratio=fact_ratio,
            headline_body_gap_score=headline_body_gap_score,
            score_evidence=score_evidence,
        )
        text = ""
        try:
            prompt = self._build_score_reason_prompt(
                overall_bias_score=overall_bias_score,
                opinion_score=opinion_score,
                emotion_score=emotion_score,
                fact_ratio=fact_ratio,
                headline_body_gap_score=headline_body_gap_score,
                score_evidence=score_evidence,
                opinion_sentences=opinion_sentences,
                span_labels=span_labels,
                language=language,
            )
            response = self.client.messages.create(
                model=self.model,
                max_tokens=700,
                messages=[{"role": "user", "content": prompt}],
            )
            block = response.content[0]
            if not isinstance(block, TextBlock) and not hasattr(block, "text"):
                raise ValueError(f"Unexpected content block type: {type(block)}")
            text = block.text.strip()
            logger.debug("summarize_score_reason raw response: %s", text)
            start = text.find("{")
            end = text.rfind("}") + 1
            result: dict = json.loads(text[start:end])
            summary = str(result.get("score_reason_summary", "")).strip()
            return summary or fallback
        except json.JSONDecodeError as e:
            logger.error("score reason json parse failed: %s | raw: %s", e, text)
            return fallback
        except Exception as e:
            logger.error("summarize_score_reason failed: %s", type(e).__name__, exc_info=True)
            return fallback

    def build_score_reason_fallback(
        self,
        *,
        overall_bias_score: float,
        opinion_score: float,
        emotion_score: float,
        fact_ratio: float,
        headline_body_gap_score: float | None = None,
        score_evidence: str = "",
    ) -> str:
        return self._build_score_reason_fallback(
            overall_bias_score=overall_bias_score,
            opinion_score=opinion_score,
            emotion_score=emotion_score,
            fact_ratio=fact_ratio,
            headline_body_gap_score=headline_body_gap_score,
            score_evidence=score_evidence,
        )

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        fact_sentences: list[ClassifiedSentenceDto],
        opinion_sentences: list[ClassifiedSentenceDto],
        title: str,
    ) -> str:
        fact_text = "\n".join(f"- {s.sentence_text}" for s in fact_sentences)
        opinion_text = "\n".join(
            f"- {s.sentence_text}" for s in opinion_sentences[:10]
        )

        return f"""
아래는 뉴스 영상의 제목과 문장 분석 결과야.

제목: {title}

[사실 문장]
{fact_text}

[주관적 문장]
{opinion_text}

아래 1가지를 한국어로 간결하게 작성해줘.
뉴스 사건의 핵심 내용만 2~3문장으로 요약해.
다음 내용은 요약에 절대 포함하지 마:
- URL, 링크, 유튜브 채널 주소
- 해시태그
- 구독/좋아요/알림 설정 요청
- 저작권 고지, 무단 전재/재배포 금지 문구
- 기자명, 채널 홍보, 출처 홍보 문구
- 영상 설명란에 들어간 홍보성 문장
JSON 형식으로만 응답해줘.

{{
  "summary_text": "홍보 문구, URL, 해시태그, 저작권 고지를 제외하고 뉴스 핵심 내용만 2~3문장으로 요약"
}}
""".strip()

    def _build_score_reason_prompt(
        self,
        *,
        overall_bias_score: float,
        opinion_score: float,
        emotion_score: float,
        fact_ratio: float,
        headline_body_gap_score: float | None,
        score_evidence: str,
        opinion_sentences: list[ClassifiedSentenceDto],
        span_labels: list[SpanLabelDto],
        language: str,
    ) -> str:
        subjectivity_points = round(overall_bias_score * 100)
        opinion_percent = round(opinion_score * 100)
        emotion_percent = round(emotion_score * 100)
        fact_percent = round(fact_ratio * 100)
        if subjectivity_points <= 20:
            score_band = "주관적 표현이 낮은 구간"
        elif subjectivity_points <= 40:
            score_band = "약간의 해석이 있는 구간"
        elif subjectivity_points <= 60:
            score_band = "의견과 정보가 섞여 있는 중간 구간"
        elif subjectivity_points <= 80:
            score_band = "주관적 표현이 많은 구간"
        else:
            score_band = "감정적이거나 주장형 표현이 강한 구간"

        opinion_text = "\n".join(
            f"- {s.sentence_text} (confidence={s.confidence:.4f})"
            for s in opinion_sentences[:5]
        )
        emotion_text = "\n".join(
            f"- {span.matched_word or '감정 표현'} (score={span.score:.4f})"
            for span in span_labels[:8]
            if span.label_type in (
                SentenceLabelType.EMOTIONALLY_LOADED,
                SentenceLabelType.EMOTIONALLY_LOADED.value,
            )
        )
        return f"""
뉴스 영상의 주관성 점수 근거를 사용자에게 설명해줘.
사용자가 "왜 이 영상이 이 주관성 점수를 받았는지" 쉽게 이해할 수 있도록 자연스럽게 작성해.
반드시 아래 제공된 점수와 근거만 사용하고, 새로운 사실이나 원인을 추론하지 마.
단순히 수치를 나열하지 말고, 이 영상이 사건을 어떤 보도 관점이나 표현 방식으로 전달하는지 먼저 설명해.

[언어]
{language}

[용어 해석]
- overall_bias_score는 사용자에게 "주관성 점수"라고 설명해.
- opinion_score는 "보도자의 해석이나 주장이 들어간 문장 비율"을 의미해.
- emotion_score는 "의견 문장 안에서 감정이 실린 표현의 정도"를 의미해.
- fact_ratio는 "사실을 전달하는 문장 비율"을 의미해.

[점수]
- 주관성 점수: {subjectivity_points}점 / 100점 ({score_band})
- 보도자의 해석이나 주장이 들어간 문장 비율: {opinion_percent}%
- 의견 문장 안에서 감정이 실린 표현 정도: {emotion_percent}%
- 사실을 전달하는 문장 비율: {fact_percent}%

[주관성 점수 구간 기준]
- 0~20점: 주관적 표현이 낮음
- 21~40점: 약간의 해석이 있음
- 41~60점: 의견과 정보가 섞여 있음
- 61~80점: 주관적 표현이 많음
- 81~100점: 감정적이거나 주장형 표현이 강함

[자동 추출 근거]
{score_evidence}

[보도자의 해석이나 주장이 들어간 문장 예시]
{opinion_text}

[감정이 실린 표현 예시]
{emotion_text}

아래 조건을 지켜 JSON 형식으로만 응답해줘.
- score_reason_summary는 한국어 2~3문장
- 사용자가 읽기 쉬운 자연스러운 설명체로 작성
- "전체 편향 점수", "편향 점수"라는 표현은 쓰지 말고 반드시 "주관성 점수"라고 표현
- overall_bias_score, opinion_score, emotion_score, fact_ratio 같은 내부 변수명은 절대 사용하지 않기
- "주관성", "감정성", "사실비중" 같은 딱딱한 분석 용어도 되도록 사용하지 않기
- 산식, 가중치, 제목-본문 괴리 점수는 언급하지 않기
- 숫자는 꼭 필요할 때만 퍼센트로 간단히 표현하고, 문장 전체를 숫자 설명으로만 채우지 않기
- 첫 문장은 반드시 위 주관성 점수 구간 기준과 일치하게 설명
- 41~60점 구간은 "낮은 편"이라고 표현하지 말고, "의견과 정보가 섞여 있는 수준" 또는 "중간 수준"이라고 설명
- 첫 문장에는 점수 구간 설명과 함께 영상의 보도 관점 또는 표현 방식이 드러나야 함
- 이어서 보도자의 해석이나 주장, 감정이 실린 표현, 사실 전달 문장 중 점수에 영향을 준 핵심 이유를 설명
- [보도자의 해석이나 주장이 들어간 문장 예시] 또는 [감정이 실린 표현 예시]가 있으면, 그 표현을 일반화하지 말고 한 가지 이상 자연스럽게 반영
- 피해야 할 문장: "전체 내용의 n%는 사실이고, n%는 의견입니다"처럼 수치만 반복하는 문장

{{
  "score_reason_summary": "사용자 친화적인 주관성 점수 근거 설명"
}}
""".strip()

    def _build_score_reason_fallback(
        self,
        *,
        overall_bias_score: float,
        opinion_score: float,
        emotion_score: float,
        fact_ratio: float,
        headline_body_gap_score: float | None,
        score_evidence: str,
    ) -> str:
        subjectivity_points = round(overall_bias_score * 100)
        if subjectivity_points <= 20:
            level_sentence = "이 영상은 사실을 전달하는 문장이 비교적 많아 주관성 점수가 낮은 편입니다."
        elif subjectivity_points <= 40:
            level_sentence = "이 영상은 일부 문장에서 보도자의 해석이나 주장이 나타나 주관성 점수가 약간 있는 편입니다."
        elif subjectivity_points <= 60:
            level_sentence = "이 영상은 사실 전달과 보도자의 해석이나 주장이 함께 나타나 주관성 점수가 중간 수준입니다."
        elif subjectivity_points <= 80:
            level_sentence = "이 영상은 보도자의 해석이나 주장이 드러나는 문장이 많아 주관성 점수가 높은 편입니다."
        else:
            level_sentence = "이 영상은 감정적이거나 주장형 표현이 강하게 나타나 주관성 점수가 매우 높은 편입니다."

        reason_parts: list[str] = []
        if opinion_score >= 0.45:
            reason_parts.append("보도자의 해석이나 주장이 들어간 문장이 여러 곳에서 확인되어 점수에 크게 반영되었습니다.")
        elif opinion_score >= 0.15:
            reason_parts.append("일부 문장에서 보도자의 해석이나 주장이 섞여 점수에 반영되었습니다.")
        else:
            reason_parts.append("보도자의 해석이나 주장이 들어간 문장은 많지 않았습니다.")

        if fact_ratio >= 0.7:
            reason_parts.append("대부분은 사실 전달 중심이라 점수가 크게 높아지지는 않았습니다.")
        elif fact_ratio >= 0.4:
            reason_parts.append("사실 전달 문장도 함께 있어 점수가 중간 수준으로 조정되었습니다.")
        else:
            reason_parts.append("사실 전달보다 해석이나 주장으로 읽히는 흐름이 더 강하게 나타났습니다.")

        if emotion_score > 0:
            if emotion_score >= 0.2:
                reason_parts.append("의견 문장 안에서 감정이 실린 표현도 함께 감지되어 주관성 점수를 높였습니다.")
            else:
                reason_parts.append("감정이 실린 표현은 일부만 감지되어 점수에는 제한적으로 반영되었습니다.")
        else:
            reason_parts.append("감정이 실린 표현은 거의 감지되지 않아 점수를 크게 높이지 않았습니다.")

        return f"{level_sentence} {' '.join(reason_parts)}"
