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

# 임계값 근거:
#   Springer 2023 (SBERT 낚시 탐지): https://link.springer.com/article/10.1007/s13278-023-01162-0
#   MDPI 2025 (패러프레이즈 탐지 최적 임계값): https://www.mdpi.com/2073-431X/14/9/385
#   KLUE-STS (한국어 STS 패러프레이즈 경계 3.0/5.0): https://openreview.net/forum?id=q-8h8-LZiUm
_GAP_TRUSTWORTHY = 0.30  # cos_sim > 0.70 — 패러프레이즈 최적 임계값 0.671~0.771 (MDPI 2025) 상단
_GAP_CLICKBAIT   = 0.40  # cos_sim < 0.60 — KLUE-STS 패러프레이즈 경계 ≈0.60, 낚시 탐지 하한 0.589 (Springer 2023)
_GAP_BURIED_DIFF = 0.40  # tail-lead 차이 임계값 — _GAP_CLICKBAIT 과 크기 통일 (구조적 heuristic)
_STD_HIGH        = 0.20  # 문장 간 유사도 표준편차 — 이 이상이면 문장 간 불균일, trustworthy 조건 제외


@dataclass
class GapResult:
    gap_score: float   # 전체 본문 gap (위치 가중 평균)
    gap_std:   float   # 문장별 유사도 표준편차
    gap_lead:  float   # 제목 vs 첫 3문장 gap
    gap_tail:  float   # 제목 vs 마지막 3문장 gap
    gap_label: str = "unknown"   # trustworthy / neutral / clickbait / buried_lede / unknown


def _classify_gap_label(gap_score: float, gap_lead: float, gap_tail: float, gap_std: float) -> str:
    diff = gap_tail - gap_lead
    if gap_lead > _GAP_CLICKBAIT:
        return "clickbait"
    if diff > _GAP_BURIED_DIFF:
        return "buried_lede"
    if gap_score < _GAP_TRUSTWORTHY and gap_std <= _STD_HIGH:
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
            return GapResult(gap_score=0.0, gap_std=0.0, gap_lead=0.0, gap_tail=0.0, gap_label="unknown")

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
        # 기사가 짧을 때 lead/tail 슬라이스가 겹치지 않도록 tail_start를 lead 끝 이후로 보장
        n = len(sentence_embs)
        lead_end   = min(_LEAD_N, n)
        tail_start = max(lead_end, n - _TAIL_N)

        gap_lead = self._gap_segment(title_emb, sentence_embs[:lead_end])
        gap_tail = (
            self._gap_segment(title_emb, sentence_embs[tail_start:])
            if tail_start < n
            else gap_lead
        )

        gap_label = _classify_gap_label(gap_score, gap_lead, gap_tail, gap_std)

        logger.debug(
            "title_body_gap: score=%.4f std=%.4f lead=%.4f tail=%.4f label=%s",
            gap_score, gap_std, gap_lead, gap_tail, gap_label,
        )
        return GapResult(
            gap_score=gap_score, gap_std=gap_std,
            gap_lead=gap_lead, gap_tail=gap_tail,
            gap_label=gap_label,
        )
