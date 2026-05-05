from __future__ import annotations

import logging
import re

from langdetect import DetectorFactory, LangDetectException
from langdetect import detect

from analysis_bc.schemas import SentenceInputDto

DetectorFactory.seed = 0

MIN_CHAR_LENGTH = 5
KO_LONG_CHUNK_THRESHOLD = 120
KO_MIN_CHUNK_LENGTH = 20
KO_FALLBACK_SPLIT_MIN = 100
KO_FALLBACK_SPLIT_MAX = 120

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_KO_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|(?<=[다요죠네까])\s+")
_KO_TRANSITION_PATTERN = re.compile(
    "|".join(
        re.escape(phrase)
        for phrase in (
            "하지만",
            "그러나",
            "반면",
            "그런데",
            "다만",
            "문제는",
            "결국",
            "사실상",
            "이건",
            "이게",
            "고 밝혔다",
            "고 말했다",
            "이라며",
            "라고",
        )
    )
)


def _clean_chunks(chunks: list[str]) -> list[str]:
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _split_by_transition(text: str) -> list[str]:
    for match in _KO_TRANSITION_PATTERN.finditer(text):
        split_at = match.start()
        left = text[:split_at].strip(" ,")
        right = text[split_at:].strip()
        if len(left) >= KO_MIN_CHUNK_LENGTH and len(right) >= KO_MIN_CHUNK_LENGTH:
            return [left, right]

        comma_at = text.rfind(",", 0, match.start())
        if comma_at != -1:
            left = text[:comma_at].strip(" ,")
            right = text[comma_at + 1:].strip()
            if len(left) >= KO_MIN_CHUNK_LENGTH and len(right) >= KO_MIN_CHUNK_LENGTH:
                return [left, right]

    return [text]


def _split_by_fallback_length(text: str) -> list[str]:
    if len(text) <= KO_LONG_CHUNK_THRESHOLD:
        return [text]

    split_at = -1
    search_start = min(KO_FALLBACK_SPLIT_MAX, len(text) - KO_MIN_CHUNK_LENGTH)
    for index in range(search_start, KO_FALLBACK_SPLIT_MIN - 1, -1):
        if text[index].isspace():
            split_at = index
            break

    if split_at == -1:
        return [text]

    left = text[:split_at].strip()
    right = text[split_at:].strip()
    if len(left) < KO_MIN_CHUNK_LENGTH or len(right) < KO_MIN_CHUNK_LENGTH:
        return [text]
    return [left, right]


def _split_long_korean_chunk(text: str) -> list[str]:
    pending = [text.strip()]
    results: list[str] = []

    while pending:
        chunk = pending.pop(0)
        if len(chunk) <= KO_LONG_CHUNK_THRESHOLD:
            results.append(chunk)
            continue

        split = _split_by_transition(chunk)
        if len(split) == 1:
            split = _split_by_fallback_length(chunk)

        if len(split) == 1:
            results.append(chunk)
        else:
            pending = split + pending

    return results


def _split_korean_regex(text: str) -> list[str]:
    """Split Korean raw text into classifier-friendly chunks without kss."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences: list[str] = []
    for line in lines:
        primary_chunks = _clean_chunks(_KO_SENTENCE_SPLIT_PATTERN.split(line))
        for chunk in primary_chunks:
            sentences.extend(_split_long_korean_chunk(chunk))
    return sentences


def split_into_sentences(text: str, language: str) -> list[SentenceInputDto]:
    """Split raw text into SentenceInputDto objects."""
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
