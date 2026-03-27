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
        span_labels: list[SpanLabelDto],
        title: str,
        language: str,
    ) -> dict:
        try:
            prompt = self._build_prompt(
                fact_sentences=fact_sentences,
                opinion_sentences=opinion_sentences,
                span_labels=span_labels,
                title=title,
            )
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            block = response.content[0]
            if not isinstance(block, TextBlock):
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

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        fact_sentences: list[ClassifiedSentenceDto],
        opinion_sentences: list[ClassifiedSentenceDto],
        span_labels: list[SpanLabelDto],
        title: str,
    ) -> str:
        fact_text = "\n".join(f"- {s.sentence_text}" for s in fact_sentences)
        opinion_text = "\n".join(
            f"- {s.sentence_text}" for s in opinion_sentences[:10]
        )

        emotional_spans = [
            s for s in span_labels
            if s.label_type in (
                SentenceLabelType.EMOTIONALLY_LOADED,
                SentenceLabelType.EMOTIONALLY_LOADED.value,
            )
        ]
        anonymous_spans = [
            s for s in span_labels
            if s.label_type in (
                SentenceLabelType.ANONYMOUS_SOURCE,
                SentenceLabelType.ANONYMOUS_SOURCE.value,
            )
        ]

        return f"""
아래는 뉴스 영상의 제목과 문장 분석 결과야.

제목: {title}

[사실 문장]
{fact_text}

[주관적 문장]
{opinion_text}

[감지된 편향]
- 감정적 표현: {len(emotional_spans)}건
- 익명 출처: {len(anonymous_spans)}건

아래 4가지를 한국어로 간결하게 작성해줘.
JSON 형식으로만 응답해줘.

{{
  "summary_text": "사실 문장 기반 객관적 요약 (3문장 이내)",
  "perspective_summary": "이 영상이 어떤 관점을 취하는지 (2문장 이내)",
  "evidence_summary": "편향의 주요 근거 (2문장 이내)",
  "tone_label": "논조를 한 단어로 (예: 비판적, 중립적, 긍정적, 선동적)"
}}
""".strip()
