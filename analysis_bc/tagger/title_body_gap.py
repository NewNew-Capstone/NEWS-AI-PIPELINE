from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer, util

from analysis_bc.schemas import SentenceInputDto

logger = logging.getLogger(__name__)


class TitleBodyGapCalculator:
    def __init__(self) -> None:
        self.model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    def calculate(self, title: str, sentences: list[SentenceInputDto]) -> float:
        if not sentences:
            return 0.0
        body_text = " ".join(s.sentence_text for s in sentences)
        title_emb = self.model.encode(title)
        body_emb = self.model.encode(body_text)
        similarity = float(util.cos_sim(title_emb, body_emb)[0][0])
        gap = round(1.0 - similarity, 4)
        logger.debug("title_body_gap: %.4f", gap)
        return gap
