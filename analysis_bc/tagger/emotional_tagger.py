from __future__ import annotations

import logging

from kiwipiepy import Kiwi
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.config import (
    EMOTION_SIMILARITY_THRESHOLD,
    QDRANT_EMOTION_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
)
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto

logger = logging.getLogger(__name__)


class EmotionalTagger:
    def __init__(self) -> None:
        self.kiwi = Kiwi()
        self.model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def tag(self, sentences: list[ClassifiedSentenceDto]) -> list[SpanLabelDto]:
        results: list[SpanLabelDto] = []
        for sentence in sentences:
            tokens = self.kiwi.tokenize(sentence.sentence_text)
            for token in tokens:
                embedding = self.model.encode(token.form).tolist()
                _result = self.qdrant.query_points(
                    collection_name=QDRANT_EMOTION_COLLECTION,
                    query=embedding,
                    limit=1,
                    score_threshold=EMOTION_SIMILARITY_THRESHOLD,
                )
                hits = _result.points
                if hits:
                    results.append(
                        SpanLabelDto(
                            content_sentence_id=sentence.content_sentence_id,
                            start_offset=token.start,
                            end_offset=token.start + token.len,
                            label_type=SentenceLabelType.EMOTIONALLY_LOADED,
                            score=hits[0].score,
                            matched_word=hits[0].payload.get("word_root"),
                        )
                    )
                    logger.debug(
                        "emotion tag: id=%d word=%s score=%.4f",
                        sentence.content_sentence_id,
                        token.form,
                        hits[0].score,
                    )
        return results
