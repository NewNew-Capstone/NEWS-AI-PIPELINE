from __future__ import annotations

import logging
import re

from langdetect import DetectorFactory, LangDetectException
from langdetect import detect

from analysis_bc.schemas import SentenceInputDto

DetectorFactory.seed = 0  # 결정론적 언어 감지 결과 보장

MIN_CHAR_LENGTH = 5

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
# 한국어 문장 종결어미 기반 분리 (pecab 없이 동작)
_KO_SENTENCE_PATTERN = re.compile(
    r"(?<=[다요죠야어네까군죠봐봐])"   # 종결어미 뒤
    r"(?=[.!?]?)"                        # 마침표 선택
    r"[\s\n]+"                           # 공백 또는 줄바꿈
)


def _split_korean_regex(text: str) -> list[str]:
    """kss 없이 regex 기반으로 한국어 문장을 분리한다."""
    # 1차: 줄바꿈 기준 분리
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences: list[str] = []
    for line in lines:
        # 2차: 종결어미 + 공백 기준 분리
        parts = _KO_SENTENCE_PATTERN.split(line)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def split_into_sentences(text: str, language: str) -> list[SentenceInputDto]:
    """raw text를 문장 단위로 분리하여 SentenceInputDto 리스트로 반환한다."""
    if language == "ko":
        raw_sentences: list[str] = _split_korean_regex(text)
    else:
        raw_sentences = [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]

    return [
        SentenceInputDto(
            content_sentence_id=idx,
            sentence_text=sent,
            sentence_order=idx,
        )
        for idx, sent in enumerate(raw_sentences)
    ]


class SentencePreprocessor:
    def __init__(self, expected_language: str) -> None:
        self.expected_language = expected_language.lower()

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _is_long_enough(self, text: str) -> bool:
        return len(text) >= MIN_CHAR_LENGTH

    def _is_valid_language(self, text: str) -> bool:
        try:
            return detect(text) == self.expected_language
        except LangDetectException:
            return False
    def preprocess(self, sentences: list[SentenceInputDto]) -> list[SentenceInputDto]:
        results: list[SentenceInputDto] = []
        for s in sorted(sentences, key=lambda x: x.sentence_order):
            normalized = self._normalize(s.sentence_text)
            if not self._is_long_enough(normalized):
                logger.debug("filtered (short): id=%d", s.content_sentence_id)
                continue
            if not self._is_valid_language(normalized):
                logger.debug("filtered (lang): id=%d", s.content_sentence_id)
                continue
            results.append(s.model_copy(update={"sentence_text": normalized}))
        return results
