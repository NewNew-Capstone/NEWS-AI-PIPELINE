from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer, util

from analysis_bc.schemas import SentenceInputDto

logger = logging.getLogger(__name__)

_POSITION_DECAY = 0.3
_LEAD_N = 3
_TAIL_N = 3


@dataclass
class GapResult:
    gap_score: float   # 전체 본문 gap (위치 가중 평균)
    gap_std:   float   # 문장별 유사도 표준편차
    gap_lead:  float   # 제목 vs 첫 3문장 gap
    gap_tail:  float   # 제목 vs 마지막 3문장 gap
    gap_label: str = "trustworthy"   # buried_lede / clickbait / trustworthy / neutral


def _classify_gap_label(gap_score: float, gap_lead: float, gap_tail: float) -> str:
    diff = gap_tail - gap_lead
    if diff > 0.4:
        return "buried_lede"
    if gap_lead > 0.4:
        return "clickbait"
    if gap_score < 0.3:
        return "trustworthy"
    return "neutral"


class TitleBodyGapCalculator:
    def __init__(self) -> None:
        self.model = SentenceTransformer("jhgan/ko-sroberta-multitask")

    def _gap_segment(self, title_emb: np.ndarray, seg_embs: np.ndarray) -> float:
        if len(seg_embs) == 0:
            return 0.0
        mean_emb = np.mean(seg_embs, axis=0).astype(np.float32)
        sim = float(util.cos_sim(title_emb, mean_emb)[0][0])
        return round(1.0 - sim, 4)

    def calculate(self, title: str, sentences: list[SentenceInputDto]) -> GapResult:
        if not sentences:
            return GapResult(gap_score=0.0, gap_std=0.0, gap_lead=0.0, gap_tail=0.0, gap_label="trustworthy")

        texts = [s.sentence_text for s in sentences]

        # 문장별 배치 인코딩 — 512토큰 제한이 문장 단위로 적용되므로 절단 없음
        sentence_embs: np.ndarray = self.model.encode(texts, convert_to_numpy=True)
        title_emb: np.ndarray = self.model.encode(title, convert_to_numpy=True)

        # 위치 기반 지수 감쇠 가중치 (리드 문장 비중↑)
        weights = np.exp(-np.arange(len(sentence_embs)) * _POSITION_DECAY)
        weights /= weights.sum()
        body_emb = np.average(sentence_embs, axis=0, weights=weights).astype(np.float32)

        similarity = float(util.cos_sim(title_emb, body_emb)[0][0])
        gap_score = round(1.0 - similarity, 4)

        # 문장별 유사도 표준편차 — 들쭉날쭉 vs 균일 패턴 구분
        per_sim = np.array([
            float(util.cos_sim(title_emb, emb.astype(np.float32))[0][0])
            for emb in sentence_embs
        ])
        gap_std = round(float(np.std(per_sim)), 4)

        # 리드 / 결말 gap — buried lede 탐지
        gap_lead = self._gap_segment(title_emb, sentence_embs[:_LEAD_N])
        gap_tail = self._gap_segment(title_emb, sentence_embs[-_TAIL_N:])

        gap_label = _classify_gap_label(gap_score, gap_lead, gap_tail)

        logger.debug(
            "title_body_gap: score=%.4f std=%.4f lead=%.4f tail=%.4f label=%s",
            gap_score, gap_std, gap_lead, gap_tail, gap_label,
        )
        return GapResult(
            gap_score=gap_score, gap_std=gap_std,
            gap_lead=gap_lead, gap_tail=gap_tail,
            gap_label=gap_label,
        )
