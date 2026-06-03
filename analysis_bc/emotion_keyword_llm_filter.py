from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from analysis_bc.config import (
    ANTHROPIC_API_KEY,
    EMOTION_KEYWORD_LLM_FILTER_ENABLED,
    EMOTION_KEYWORD_LLM_FILTER_MODEL,
)
from analysis_bc.enums import BiasKeywordType
from analysis_bc.schemas import BiasAnalysisKeywordDto

logger = logging.getLogger(__name__)


def _is_emotion_keyword(row: BiasAnalysisKeywordDto) -> bool:
    return str(row.keyword_type) == BiasKeywordType.EMOTION.value


class EmotionKeywordLlmFilter:
    def __init__(
        self,
        *,
        client: Any | None = None,
        enabled: bool | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.enabled = EMOTION_KEYWORD_LLM_FILTER_ENABLED if enabled is None else enabled
        self.api_key = (ANTHROPIC_API_KEY if api_key is None else api_key).strip()
        self.model = model or EMOTION_KEYWORD_LLM_FILTER_MODEL
        self.client = client
        self._cache: dict[tuple[str, ...], frozenset[str]] = {}

    def filter_keywords(
        self,
        keywords: Sequence[BiasAnalysisKeywordDto],
    ) -> list[BiasAnalysisKeywordDto]:
        rows = list(keywords)
        emotion_words = [row.keyword_text for row in rows if _is_emotion_keyword(row)]
        keep_words = self.filter_words(emotion_words)
        if keep_words is None:
            return rows

        return [
            row for row in rows
            if not _is_emotion_keyword(row) or row.keyword_text in keep_words
        ]

    def filter_words(self, words: Sequence[str]) -> frozenset[str] | None:
        emotion_words = [word.strip() for word in words if word and word.strip()]
        if not emotion_words:
            return frozenset()
        if not self.enabled or not self.api_key:
            return None
        return self._filter_emotion_words(emotion_words)

    def _filter_emotion_words(self, words: list[str]) -> frozenset[str] | None:
        normalized_words = tuple(dict.fromkeys(word.strip() for word in words if word.strip()))
        if not normalized_words:
            return frozenset()
        if normalized_words in self._cache:
            return self._cache[normalized_words]

        try:
            client = self._client()
            response = client.messages.create(
                model=self.model,
                max_tokens=400,
                temperature=0,
                messages=[{"role": "user", "content": self._build_prompt(normalized_words)}],
            )
            text = self._response_text(response)
            payload = self._parse_json_object(text)
            keep = self._normalize_keep_words(payload, normalized_words)
            self._cache[normalized_words] = keep
            return keep
        except Exception as exc:
            logger.warning("emotion keyword LLM filter failed: %s", exc, exc_info=True)
            return None

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        from anthropic import Anthropic

        self.client = Anthropic(api_key=self.api_key)
        return self.client

    @staticmethod
    def _response_text(response: Any) -> str:
        blocks = getattr(response, "content", []) or []
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
        return ""

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("LLM response does not contain JSON object")
        payload = json.loads(text[start:end])
        if not isinstance(payload, dict):
            raise ValueError("LLM response JSON is not an object")
        return payload

    @staticmethod
    def _normalize_keep_words(
        payload: dict[str, Any],
        candidates: tuple[str, ...],
    ) -> frozenset[str]:
        raw_keep = payload.get("keep")
        if raw_keep is None:
            raw_keep = payload.get("emotion_keywords")
        if not isinstance(raw_keep, list):
            raise ValueError("LLM response must include keep list")

        candidate_set = set(candidates)
        keep = {
            str(item).strip()
            for item in raw_keep
            if str(item).strip() in candidate_set
        }
        return frozenset(keep)

    @staticmethod
    def _build_prompt(words: tuple[str, ...]) -> str:
        payload = json.dumps({"candidates": list(words)}, ensure_ascii=False)
        return f"""
너는 한국어 뉴스 편향 분석 시스템의 최종 감정 키워드 검수기다.
아래 후보 중 진짜 감정 단어만 keep에 남겨라.

KEEP 기준:
- 사람의 감정 상태, 정서 반응, 감정적 평가/태도를 직접 나타내는 명사/기본형 서술어
- 긍정/부정 평가 형용사도 감정적으로 쓰일 수 있으면 KEEP
- 예: 분노, 불안, 공포, 슬픔, 실망, 행복, 기쁨, 짜증, 혐오, 걱정, 놀라다, 재밌다, 아름답다, 무섭다, 슬프다, 끔찍하다

BLOCK 기준:
- 명백히 감정 단어가 아닌 사람/관계/직책/국가/정치/경제/사회 이슈 주제어
- 사건, 정책, 사업, 발언, 전달, 추진, 수사 같은 일반 뉴스 주제어
- 일반 동작/기능 동사, 보도 동사
- 예: 친구, 이야기, 정책, 추진, 대통령, 트럼프, 알리다, 밝히다, 전하다, 부르다, 통하다

감정/평가 표현인지 애매하면 KEEP으로 둔다. 너무 명백히 아닌 것만 BLOCK한다.
반드시 입력 후보 문자열을 그대로 사용하고, 새 단어를 만들지 마라.
JSON만 반환해라.

입력:
{payload}

출력 스키마:
{{
  "keep": ["진짜 감정 단어"],
  "block": ["감정 단어가 아닌 후보"]
}}
""".strip()
