from __future__ import annotations

import logging

from kiwipiepy import Kiwi
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.config import (
    ANONYMOUS_SIMILARITY_THRESHOLD,
    EMOTION_SIMILARITY_THRESHOLD,
    QDRANT_ANONYMOUS_COLLECTION,
    QDRANT_EMOTION_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto

logger = logging.getLogger(__name__)

_ANON_WINDOW_SIZES = (2, 3, 4)

_VALID_POS: frozenset[str] = frozenset({
    "NNG",  # 일반 명사
    "NNP",  # 고유 명사
    "VV",   # 동사
    "VA",   # 형용사
    "MAG",  # 일반 부사
})


class SpanTagger:
    """OPINION 문장의 span에 단일 태그를 붙여 반환한다.

    - anonymous 검색: 토큰 2~4개 슬라이딩 윈도우 → anonymous_patterns 컬렉션
    - emotion 검색  : 형태소 토큰 단위 → emotion_words 컬렉션
    - 두 span이 겹치면 score가 더 높은 쪽 하나만 유지한다.
    """

    def __init__(self) -> None:
        self.kiwi = Kiwi()
        self.model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def tag(self, sentences: list[ClassifiedSentenceDto]) -> list[SpanLabelDto]:
        results: list[SpanLabelDto] = []
        for sentence in sentences:
            tokens = self.kiwi.tokenize(sentence.sentence_text)
            anon_span = self._search_anonymous(sentence, tokens)
            emotion_spans = self._search_emotion(sentence, tokens)
            results.extend(self._merge(anon_span, emotion_spans))
        return results

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _search_anonymous(
        self,
        sentence: ClassifiedSentenceDto,
        tokens: list,
    ) -> SpanLabelDto | None:
        """토큰 2~4개 슬라이딩 윈도우로 anonymous_patterns 컬렉션을 검색한다.

        모든 윈도우 후보 중 score가 가장 높은 것을 반환한다.
        """
        best: tuple[float, int, int, dict] | None = None  # (score, start, end, payload)

        for window_size in _ANON_WINDOW_SIZES:
            for i in range(len(tokens) - window_size + 1):
                window = tokens[i : i + window_size]
                span_start = window[0].start
                span_end = window[-1].start + window[-1].len
                candidate = sentence.sentence_text[span_start:span_end]

                embedding = self.model.encode(candidate).tolist()
                _result = self.qdrant.query_points(
                    collection_name=QDRANT_ANONYMOUS_COLLECTION,
                    query=embedding,
                    limit=1,
                    score_threshold=ANONYMOUS_SIMILARITY_THRESHOLD,
                )
                hits = _result.points
                if hits and (best is None or hits[0].score > best[0]):
                    best = (hits[0].score, span_start, span_end, hits[0].payload or {})

        if best is None:
            return None

        score, start, end, payload = best
        label_type = SentenceLabelType.ANONYMOUS_SOURCE
        matched = payload.get("phrase")
        logger.debug(
            "anonymous tag: id=%d offset=%d~%d label=%s score=%.4f",
            sentence.content_sentence_id,
            start,
            end,
            label_type,
            score,
        )
        return SpanLabelDto(
            content_sentence_id=sentence.content_sentence_id,
            start_offset=start,
            end_offset=end,
            label_type=label_type,
            score=score,
            matched_word=matched,
        )

    def _search_emotion(
        self,
        sentence: ClassifiedSentenceDto,
        tokens: list,
    ) -> list[SpanLabelDto]:
        """형태소 토큰 단위로 emotion_words 컬렉션을 검색한다."""
        spans: list[SpanLabelDto] = []
        for token in tokens:
            if token.tag not in _VALID_POS:
                continue
            if len(token.form) < 2:
                continue
            embedding = self.model.encode(token.form).tolist()
            _result = self.qdrant.query_points(
                collection_name=QDRANT_EMOTION_COLLECTION,
                query=embedding,
                limit=1,
                score_threshold=EMOTION_SIMILARITY_THRESHOLD,
            )
            hits = _result.points
            if not hits:
                continue
            payload = hits[0].payload or {}
            logger.debug(
                "emotion tag: id=%d token=%s score=%.4f",
                sentence.content_sentence_id,
                token.form,
                hits[0].score,
            )
            spans.append(
                SpanLabelDto(
                    content_sentence_id=sentence.content_sentence_id,
                    start_offset=token.start,
                    end_offset=token.start + token.len,
                    label_type=SentenceLabelType.EMOTIONALLY_LOADED,
                    score=hits[0].score,
                    matched_word=payload.get("word_root"),
                )
            )
        return spans

    def _merge(
        self,
        anon_span: SpanLabelDto | None,
        emotion_spans: list[SpanLabelDto],
    ) -> list[SpanLabelDto]:
        """anon span과 겹치는 emotion span은 score 비교 후 높은 쪽만 유지한다."""
        if anon_span is None:
            return emotion_spans

        final: list[SpanLabelDto] = []
        winner = anon_span

        for e_span in emotion_spans:
            overlaps = not (
                e_span.end_offset <= anon_span.start_offset
                or e_span.start_offset >= anon_span.end_offset
            )
            if overlaps:
                if e_span.score > winner.score:
                    winner = e_span
            else:
                final.append(e_span)

        final.append(winner)
        return sorted(final, key=lambda s: s.start_offset)
