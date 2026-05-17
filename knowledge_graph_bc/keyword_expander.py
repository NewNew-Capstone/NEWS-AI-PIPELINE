from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

_BAD_CHARS = re.compile(r"[\r\n\t]+")
_MULTI_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ExpandedKeywordSet:
    requested_keyword: str
    ko: list[str]
    en: list[str]
    zh: list[str]


class MultilingualKeywordExpander:
    def __init__(self) -> None:
        self.api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()

    def expand(self, keyword_ko: str, max_terms_per_language: int = 4) -> ExpandedKeywordSet:
        base = self._normalize_term(keyword_ko)
        if not base:
            raise ValueError("keyword_ko must not be blank")

        terms_limit = max(1, min(int(max_terms_per_language), 10))
        llm_result = self._expand_with_llm(base, terms_limit)
        if llm_result is not None:
            return llm_result

        return self._expand_with_fallback(base, terms_limit)

    def _expand_with_llm(self, base: str, limit: int) -> ExpandedKeywordSet | None:
        if not self.api_key:
            return None
        try:
            from anthropic import Anthropic
            from anthropic.types import TextBlock
        except Exception as exc:  # pragma: no cover - import failure fallback
            logger.warning("Anthropic import failed, using fallback: %s", exc)
            return None

        prompt = f"""
다음 한국어 검색어를 비교 뉴스 수집용 키워드로 확장하세요.
- 입력: "{base}"
- 출력은 JSON만 반환
- 각 언어(ko/en/zh)는 짧은 검색어 배열
- 각 배열 길이는 최대 {limit}
- 첫 번째 원소는 입력 의미에 가장 가까운 핵심어

JSON 스키마:
{{
  "ko": ["..."],
  "en": ["..."],
  "zh": ["..."]
}}
"""
        try:
            client = Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=400,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            blocks = [block for block in response.content if isinstance(block, TextBlock)]
            text = blocks[0].text if blocks else ""
            payload = self._safe_parse_json(text)
            if not isinstance(payload, dict):
                return None
            ko = self._normalize_terms(payload.get("ko"), limit, include_seed=base)
            en = self._normalize_terms(payload.get("en"), limit)
            zh = self._normalize_terms(payload.get("zh"), limit)
            if not en or not zh:
                return None
            return ExpandedKeywordSet(requested_keyword=base, ko=ko, en=en, zh=zh)
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            logger.warning("LLM keyword expansion failed, fallback enabled: %s", exc)
            return None

    def _expand_with_fallback(self, base: str, limit: int) -> ExpandedKeywordSet:
        en_seed = self._translate(base, source="ko", target="en")
        zh_seed = self._translate(base, source="ko", target="zh-CN")
        ko = self._normalize_terms([base], limit, include_seed=base)
        en = self._normalize_terms([en_seed], limit)
        zh = self._normalize_terms([zh_seed], limit)
        return ExpandedKeywordSet(requested_keyword=base, ko=ko, en=en, zh=zh)

    def _translate(self, text: str, source: str, target: str) -> str:
        try:
            return self._normalize_term(GoogleTranslator(source=source, target=target).translate(text))
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            logger.warning("Fallback translate failed (%s->%s): %s", source, target, exc)
            return text

    def _normalize_terms(
        self,
        raw_terms: Any,
        limit: int,
        *,
        include_seed: str | None = None,
    ) -> list[str]:
        values = list(raw_terms) if isinstance(raw_terms, Iterable) and not isinstance(raw_terms, (str, bytes)) else [raw_terms]
        out: list[str] = []
        seen: set[str] = set()

        if include_seed:
            seed = self._normalize_term(include_seed)
            if seed:
                out.append(seed)
                seen.add(seed.lower())

        for raw in values:
            term = self._normalize_term(raw)
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(term)
            if len(out) >= limit:
                break
        return out[:limit]

    def _normalize_term(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        text = _BAD_CHARS.sub(" ", text)
        text = _MULTI_SPACE.sub(" ", text).strip()
        return text[:80]

    def _safe_parse_json(self, text: str) -> Any:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to recover first JSON object block.
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None
