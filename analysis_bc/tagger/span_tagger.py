from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal

import fasttext
import httpx
import numpy as np
from kiwipiepy import Kiwi
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.config import (
    EMOTION_SCORE_MARGIN,
    EMOTION_SIMILARITY_THRESHOLD,
    EMOTION_TOP_K,
    FASTTEXT_MODEL_PATH,
    QDRANT_EMOTION_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)
from analysis_bc.enums import SentenceLabelType
from analysis_bc.score_utils import normalize_score
from analysis_bc.schemas import SpanLabelDto

_FASTTEXT_SERVER_URL: str = os.getenv("FASTTEXT_SERVER_URL", "").strip().rstrip("/")

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


RejectReason = Literal[
    "invalid_pos",
    "too_short",
    "no_hit",
    "low_margin",
    "below_threshold",
]


@dataclass
class TokenTrace:
    form: str
    surface: str
    tag: str
    start: int
    length: int
    accepted: bool = False
    reject_reason: RejectReason | None = None
    qdrant_scores: list[float] = field(default_factory=list)
    created_span: SpanLabelDto | None = None


@dataclass
class SentenceTrace:
    content_sentence_id: int
    sentence_text: str
    tokens: list[TokenTrace] = field(default_factory=list)


@dataclass
class TagTrace:
    qdrant_healthy: bool
    qdrant_health_reason: str
    sentences: list[SentenceTrace] = field(default_factory=list)


class SpanTagger:
    """OPINION 문장의 span에 감정 태그를 붙여 반환한다.

    - emotion 검색: 형태소 토큰 단위 → emotion_words 컬렉션 (FastText)
    """

    def __init__(self) -> None:
        self.kiwi = Kiwi()
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=3.0)
        self._qdrant_healthy: bool = False
        self._health_checked_at: float = 0.0

        if _FASTTEXT_SERVER_URL:
            self.ft_model = None
            self._ft_server_url = _FASTTEXT_SERVER_URL
        else:
            self.ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
            self._ft_server_url = ""
        self.last_debug_trace: TagTrace | None = None

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

    def tag(
        self,
        sentences: list[ClassifiedSentenceDto],
        debug: bool = False,
    ) -> list[SpanLabelDto]:
        print(f"[SpanTagger] tag() 시작 — opinion 문장 수: {len(sentences)}")
        self.last_debug_trace = None

        is_healthy = self._is_qdrant_healthy()
        if debug:
            self.last_debug_trace = TagTrace(
                qdrant_healthy=is_healthy,
                qdrant_health_reason="ok" if is_healthy else "health_check_failed",
            )

        if not is_healthy:
            return []
        results: list[SpanLabelDto] = []
        for sentence in sentences:
            tokens = self.kiwi.tokenize(sentence.sentence_text)
            sentence_trace = None
            if debug and self.last_debug_trace is not None:
                sentence_trace = SentenceTrace(
                    content_sentence_id=sentence.content_sentence_id,
                    sentence_text=sentence.sentence_text,
                )
            emotion_spans = self._search_emotion(
                sentence=sentence,
                tokens=tokens,
                debug=debug,
                sentence_trace=sentence_trace,
            )
            if debug and self.last_debug_trace is not None and sentence_trace is not None:
                self.last_debug_trace.sentences.append(sentence_trace)
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
        debug: bool = False,
        sentence_trace: SentenceTrace | None = None,
    ) -> list[SpanLabelDto]:
        """형태소 토큰을 FastText로 embed 후 emotion_words 컬렉션에서 검색한다.

        ko-sroberta 대신 FastText를 사용하는 이유:
        FastText는 단어/형태소 레벨 임베딩에 최적화된 모델로, 단일 형태소 간
        의미 거리를 정확하게 표현한다. ko-sroberta는 문장 쌍 학습 모델이라
        단일 형태소 embed 시 노이즈가 심해 false positive가 발생한다.
        """
        spans: list[SpanLabelDto] = []

        valid_tokens = []
        token_traces: list[TokenTrace] = []
        for t in tokens:
            surface = sentence.sentence_text[t.start:t.start + t.len]
            trace = TokenTrace(
                form=t.form,
                surface=surface,
                tag=t.tag,
                start=t.start,
                length=t.len,
            )
            if t.tag not in _VALID_POS:
                trace.reject_reason = "invalid_pos"
            elif len(t.form) < 2:
                trace.reject_reason = "too_short"
            else:
                valid_tokens.append(t)
            token_traces.append(trace)

        if debug and sentence_trace is not None:
            sentence_trace.tokens.extend(token_traces)

        if not valid_tokens:
            return spans

        # FastText embed + 단위 벡터 정규화 (init_qdrant와 동일하게)
        # token.form(어간) 대신 원문 표면형 사용 → init_qdrant의 word 필드와 형태 일치
        surfaces = [
            sentence.sentence_text[t.start:t.start + t.len]
            for t in valid_tokens
        ]

        if self.ft_model is not None:
            raw_vecs = [self.ft_model.get_word_vector(s) for s in surfaces]
        else:
            resp = httpx.post(
                f"{self._ft_server_url}/embed",
                json={"words": surfaces},
                timeout=10.0,
            )
            resp.raise_for_status()
            raw_vecs = [np.array(v) for v in resp.json()["vectors"]]

        embeddings = []
        for vec in raw_vecs:
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
                    limit=EMOTION_TOP_K,
                    score_threshold=None if debug else EMOTION_SIMILARITY_THRESHOLD,
                    with_payload=True,
                )
                for vec in embeddings
            ],
        )

        valid_index = 0
        for token, result in zip(valid_tokens, batch_results):
            hits = result.points
            trace = None
            if debug and sentence_trace is not None:
                while valid_index < len(sentence_trace.tokens):
                    candidate = sentence_trace.tokens[valid_index]
                    valid_index += 1
                    if candidate.reject_reason is None:
                        trace = candidate
                        break

            if debug and trace is not None:
                trace.qdrant_scores = [float(h.score) for h in hits]

            if not hits:
                if debug and trace is not None:
                    trace.reject_reason = "no_hit"
                continue

            if debug:
                if hits[0].score < EMOTION_SIMILARITY_THRESHOLD:
                    if trace is not None:
                        trace.reject_reason = "below_threshold"
                    continue

            if len(hits) >= 2:
                margin = hits[0].score - hits[1].score
                if margin < EMOTION_SCORE_MARGIN:
                    if debug and trace is not None:
                        trace.reject_reason = "low_margin"
                    continue
            surface = sentence.sentence_text[token.start:token.start + token.len]
            logger.debug(
                "emotion tag: id=%d surface=%s score=%.4f",
                sentence.content_sentence_id,
                surface,
                hits[0].score,
            )
            span = SpanLabelDto(
                content_sentence_id=sentence.content_sentence_id,
                start_offset=token.start,
                end_offset=token.start + token.len,
                label_type=SentenceLabelType.EMOTIONALLY_LOADED,
                score=normalize_score(hits[0].score),
                matched_word=surface,
            )
            spans.append(span)
            if debug and trace is not None:
                trace.accepted = True
                trace.created_span = span
        return spans
