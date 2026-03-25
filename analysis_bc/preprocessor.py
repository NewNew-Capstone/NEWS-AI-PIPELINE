from __future__ import annotations

import logging
import re

from langdetect import DetectorFactory, LangDetectException
from langdetect import detect

from analysis_bc.schemas import SentenceInputDto

DetectorFactory.seed = 0  # 결정론적 언어 감지 결과 보장

MIN_CHAR_LENGTH = 5

logger = logging.getLogger(__name__)


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
#전처리 메서드 
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
