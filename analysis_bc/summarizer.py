from __future__ import annotations

import json
import logging

from anthropic import Anthropic
from anthropic.types import TextBlock

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto

logger = logging.getLogger(__name__)

_FALLBACK: dict = {
    "summary_text": "",
    "perspective_summary": "",
    "evidence_summary": "",
    "tone_label": "",
}


class BiasSummarizer:
    def __init__(self) -> None:
        import os
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        logger.debug("BiasSummarizer init: key_len=%d prefix=%s", len(key), key[:12])
        self.client = Anthropic()  # SDK가 ANTHROPIC_API_KEY 환경변수를 직접 읽음
        self.model = "claude-haiku-4-5-20251001"  # TODO: 검증 후 claude-sonnet-4-6 으로 교체

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
JSON 형식으로만 응답해줘.

{{
  "summary_text": "사실 문장 기반 객관적 요약 (3문장 이내)"
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
        gap_text = (
            "정보 없음"
            if headline_body_gap_score is None
            else f"{headline_body_gap_score:.4f}"
        )

        return f"""
뉴스 편향 분석 점수의 산출 근거를 사용자에게 설명해줘.
반드시 아래 제공된 점수와 근거만 사용하고, 새로운 사실이나 원인을 추론하지 마.
제목-본문 괴리 점수는 현재 overall_bias_score 산식에 직접 포함되지 않는 별도 참고 지표라고 설명해.

[언어]
{language}

[산식]
overall_bias_score = 0.4 * opinion_score + 0.3 * emotion_score + 0.3 * (1 - fact_ratio)

[점수]
- overall_bias_score: {overall_bias_score:.4f}
- opinion_score: {opinion_score:.4f}
- emotion_score: {emotion_score:.4f}
- fact_ratio: {fact_ratio:.4f}
- headline_body_gap_score: {gap_text}

[산식 기반 근거]
{score_evidence}

[주요 주관적 문장]
{opinion_text}

[감정 표현 근거]
{emotion_text}

아래 조건을 지켜 JSON 형식으로만 응답해줘.
- score_reason_summary는 한국어 2~4문장
- 수식이 어떤 구성 점수로 0~1 사이의 최종 점수로 합산되는지 쉽게 설명
- 숫자는 제공된 값을 그대로 사용

{{
  "score_reason_summary": "전체 편향 점수 산출 근거 설명"
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
        fact_gap = 1 - fact_ratio
        summary = (
            "전체 편향 점수는 의견성 점수 40%, 감정성 점수 30%, "
            "사실 기반 문장이 부족한 정도 30%를 더해 계산됩니다. "
            f"이번 결과는 opinion_score {opinion_score:.4f}, "
            f"emotion_score {emotion_score:.4f}, "
            f"1 - fact_ratio {fact_gap:.4f}를 반영해 "
            f"overall_bias_score {overall_bias_score:.4f}로 산출되었습니다."
        )
        if score_evidence:
            summary += f" 주요 근거는 {score_evidence}"
        if headline_body_gap_score is not None:
            summary += (
                f" 제목-본문 괴리 점수 {headline_body_gap_score:.4f}는 "
                "최종 산식에는 직접 포함되지 않는 별도 참고 지표입니다."
            )
        return summary
