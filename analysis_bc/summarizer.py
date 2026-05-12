from __future__ import annotations

import json
import logging

from anthropic import Anthropic
from anthropic.types import TextBlock

from analysis_bc.classifier import ClassifiedSentenceDto

logger = logging.getLogger(__name__)

_FALLBACK: dict = {
    "summary_text": "",
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
    ) -> dict:
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
