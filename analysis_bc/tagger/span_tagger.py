from __future__ import annotations

import logging
import time

import fasttext
import numpy as np
from kiwipiepy import Kiwi
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest
from sentence_transformers import SentenceTransformer

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.config import (
    ANONYMOUS_SIMILARITY_THRESHOLD,
    EMOTION_SIMILARITY_THRESHOLD,
    FASTTEXT_MODEL_PATH,
    QDRANT_ANONYMOUS_COLLECTION,
    QDRANT_EMOTION_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto

logger = logging.getLogger(__name__)

_ANON_WINDOW_SIZES = (2, 3, 4)
_HEALTH_CHECK_TTL = 30.0  # 초: 이 시간 동안 재확인 없이 캐시된 결과 사용

_VALID_POS: frozenset[str] = frozenset({
    "NNG",  # 일반 명사
    "NNP",  # 고유 명사
    "VV",   # 동사
    "VA",   # 형용사
    "MAG",  # 일반 부사
    "XR",   # 어근 (심각하다 → 심각, 결렬하다 → 결렬 등 한자어 어근)
})


class SpanTagger:
    """OPINION 문장의 span에 단일 태그를 붙여 반환한다.

    - anonymous 검색: 토큰 2~4개 슬라이딩 윈도우 → anonymous_patterns 컬렉션 (ko-sroberta)
    - emotion 검색  : 형태소 토큰 단위 → emotion_words 컬렉션 (FastText)
    - 두 span이 겹치면 score가 더 높은 쪽 하나만 유지한다.
    """

    def __init__(self) -> None:
        self.kiwi = Kiwi()
        self.st_model = SentenceTransformer("jhgan/ko-sroberta-multitask")  # anonymous용
        self.ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)            # emotion용
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3.0)
        self._qdrant_healthy: bool = False
        self._health_checked_at: float = 0.0

    def _is_qdrant_healthy(self) -> bool:
        now = time.monotonic()
        if now - self._health_checked_at < _HEALTH_CHECK_TTL:
            return self._qdrant_healthy

        try:
            self.qdrant.get_collections()
            self._qdrant_healthy = True
            print("[SpanTagger] Qdrant health check passed")
        except Exception:
            self._qdrant_healthy = False
            print("[SpanTagger] Qdrant health check failed — skipping SpanTagger")
            logger.warning("Qdrant health check failed, skipping SpanTagger")

        self._health_checked_at = now
        return self._qdrant_healthy

    def tag(self, sentences: list[ClassifiedSentenceDto]) -> list[SpanLabelDto]:
        print(f"[SpanTagger] tag() 시작 — opinion 문장 수: {len(sentences)}")
        if not self._is_qdrant_healthy():
            return []
        results: list[SpanLabelDto] = []
        for i, sentence in enumerate(sentences):
            tokens = self.kiwi.tokenize(sentence.sentence_text)
            anon_span = self._search_anonymous(sentence, tokens)
            emotion_spans = self._search_emotion(sentence, tokens)
            merged = self._merge(anon_span, emotion_spans)

            if anon_span or emotion_spans:
                anon_info = f"ANONYMOUS(\"{anon_span.matched_word}\" {anon_span.score:.2f})" if anon_span else "ANONYMOUS 없음"
                emotion_info = ", ".join(
                    f"EMOTION(\"{s.matched_word}\" {s.score:.2f})" for s in emotion_spans
                ) if emotion_spans else "EMOTION 없음"
                preview = sentence.sentence_text[:30].replace("\n", " ")
                print(f"[SpanTagger] 문장 {i + 1}/{len(sentences)} id={sentence.content_sentence_id} \"{preview}...\" → {anon_info} / {emotion_info}")

            results.extend(merged)
        print(f"[SpanTagger] tag() 완료 — 총 라벨 수: {len(results)}")
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

        # 후보 텍스트와 offset을 먼저 수집
        candidates: list[tuple[str, int, int]] = []  # (text, span_start, span_end)
        for window_size in _ANON_WINDOW_SIZES:
            for i in range(len(tokens) - window_size + 1):
                window = tokens[i : i + window_size]
                span_start = window[0].start
                span_end = window[-1].start + window[-1].len
                candidate = sentence.sentence_text[span_start:span_end]
                candidates.append((candidate, span_start, span_end))

        if not candidates:
            return None

        # 배치 encode (1회 호출) — anonymous는 ko-sroberta 사용
        texts = [c[0] for c in candidates]
        embeddings = self.st_model.encode(texts)

        # 배치 쿼리 (1회 HTTP 요청)
        batch_results = self.qdrant.query_batch_points(
            collection_name=QDRANT_ANONYMOUS_COLLECTION,
            requests=[
                QueryRequest(
                    query=embedding.tolist(),
                    limit=1,
                    score_threshold=ANONYMOUS_SIMILARITY_THRESHOLD,
                    with_payload=True,
                )
                for embedding in embeddings
            ],
        )

        for (_, span_start, span_end), result in zip(candidates, batch_results):
            hits = result.points
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
        """형태소 토큰을 FastText로 embed 후 emotion_words 컬렉션에서 검색한다.

        ko-sroberta 대신 FastText를 사용하는 이유:
        FastText는 단어/형태소 레벨 임베딩에 최적화된 모델로, 단일 형태소 간
        의미 거리를 정확하게 표현한다. ko-sroberta는 문장 쌍 학습 모델이라
        단일 형태소 embed 시 노이즈가 심해 false positive가 발생한다.
        """
        spans: list[SpanLabelDto] = []

        valid_tokens = [
            t for t in tokens
            if t.tag in _VALID_POS and len(t.form) >= 2
        ]
        if not valid_tokens:
            return spans

        # FastText embed + 단위 벡터 정규화 (init_qdrant와 동일하게)
        # token.form(어간) 대신 원문 표면형 사용 → init_qdrant의 word 필드와 형태 일치
        embeddings = []
        for t in valid_tokens:
            surface = sentence.sentence_text[t.start:t.start + t.len]
            vec = self.ft_model.get_word_vector(surface)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)

        # 배치 쿼리 (1회 HTTP 요청)
        batch_results = self.qdrant.query_batch_points(
            collection_name=QDRANT_EMOTION_COLLECTION,
            requests=[
                QueryRequest(
                    query=vec.tolist(),
                    limit=1,
                    score_threshold=EMOTION_SIMILARITY_THRESHOLD,
                    with_payload=True,
                )
                for vec in embeddings
            ],
        )

        for token, result in zip(valid_tokens, batch_results):
            hits = result.points
            if not hits:
                continue
            surface = sentence.sentence_text[token.start:token.start + token.len]
            logger.debug(
                "emotion tag: id=%d surface=%s score=%.4f",
                sentence.content_sentence_id,
                surface,
                hits[0].score,
            )
            spans.append(
                SpanLabelDto(
                    content_sentence_id=sentence.content_sentence_id,
                    start_offset=token.start,
                    end_offset=token.start + token.len,
                    label_type=SentenceLabelType.EMOTIONALLY_LOADED,
                    score=hits[0].score,
                    matched_word=surface,
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
