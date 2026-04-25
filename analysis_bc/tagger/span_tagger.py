from __future__ import annotations

import logging
import time

import fasttext
import numpy as np
from kiwipiepy import Kiwi
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.config import (
    EMOTION_SIMILARITY_THRESHOLD,
    FASTTEXT_MODEL_PATH,
    QDRANT_EMOTION_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto

logger = logging.getLogger(__name__)

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
    """OPINION 문장의 span에 감정 태그를 붙여 반환한다.

    - emotion 검색: 형태소 토큰 단위 → emotion_words 컬렉션 (FastText)
    """

    def __init__(self) -> None:
        self.kiwi = Kiwi()
        self.ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
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
        for sentence in sentences:
            tokens = self.kiwi.tokenize(sentence.sentence_text)
            emotion_spans = self._search_emotion(sentence, tokens)
            results.extend(emotion_spans)
        print(f"[SpanTagger] tag() 완료 — 총 라벨 수: {len(results)}")
        return results

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

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
