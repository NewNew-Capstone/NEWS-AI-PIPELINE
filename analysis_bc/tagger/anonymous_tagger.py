from __future__ import annotations

import logging

from redis import Redis

from analysis_bc.config import (
    REDIS_ANONYMOUS_KEY,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_SPECULATIVE_KEY,
)
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SentenceInputDto, SpanLabelDto

logger = logging.getLogger(__name__)

_DEFAULT_ANONYMOUS_PATTERNS: list[str] = [
    "관계자에 따르면",
    "관계자는",
    "전문가들은",
    "일각에서는",
    "업계에서는",
    "소식통에 의하면",
    "복수의 관계자",
    "익명을 요구한",
]

_DEFAULT_SPECULATIVE_PATTERNS: list[str] = [
    "것으로 알려졌다",
    "것으로 전해졌다",
    "전망이다",
    "것으로 예상된다",
    "카더라",
    "것으로 보인다",
]


class AnonymousTagger:
    def __init__(self) -> None:
        self.redis: Redis = Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )
        self._init_patterns()

    def _init_patterns(self) -> None:
        if not self.redis.exists(REDIS_ANONYMOUS_KEY):
            self.redis.rpush(REDIS_ANONYMOUS_KEY, *_DEFAULT_ANONYMOUS_PATTERNS)
        if not self.redis.exists(REDIS_SPECULATIVE_KEY):
            self.redis.rpush(REDIS_SPECULATIVE_KEY, *_DEFAULT_SPECULATIVE_PATTERNS)

    def tag(self, sentences: list[SentenceInputDto]) -> list[SpanLabelDto]:
        anonymous_patterns: list[str] = self.redis.lrange(REDIS_ANONYMOUS_KEY, 0, -1)
        speculative_patterns: list[str] = self.redis.lrange(REDIS_SPECULATIVE_KEY, 0, -1)

        results: list[SpanLabelDto] = []
        for s in sentences:
            for pattern in anonymous_patterns:
                idx = s.sentence_text.find(pattern)
                if idx != -1:
                    results.append(
                        SpanLabelDto(
                            content_sentence_id=s.content_sentence_id,
                            start_offset=idx,
                            end_offset=idx + len(pattern),
                            label_type=SentenceLabelType.ANONYMOUS_SOURCE,
                            score=1.0,
                            matched_word=pattern,
                        )
                    )
                    logger.debug(
                        "anonymous tag: id=%d pattern=%s", s.content_sentence_id, pattern
                    )
            for pattern in speculative_patterns:
                idx = s.sentence_text.find(pattern)
                if idx != -1:
                    results.append(
                        SpanLabelDto(
                            content_sentence_id=s.content_sentence_id,
                            start_offset=idx,
                            end_offset=idx + len(pattern),
                            label_type=SentenceLabelType.SPECULATIVE,
                            score=1.0,
                            matched_word=pattern,
                        )
                    )
                    logger.debug(
                        "speculative tag: id=%d pattern=%s", s.content_sentence_id, pattern
                    )
        return results
