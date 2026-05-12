from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from analysis_bc.schemas import SentenceInputDto
from analysis_bc.tagger.title_body_gap import GapResult, TitleBodyGapCalculator

_DIM = 768

# 코사인 유사도 제어용 단위 벡터
_SIMILAR = np.array([1.0] + [0.0] * (_DIM - 1), dtype=np.float32)          # title과 sim=1.0
_ORTHOG  = np.array([0.0, 1.0] + [0.0] * (_DIM - 2), dtype=np.float32)     # title과 sim=0.0
_HALF    = np.array([0.7071, 0.7071] + [0.0] * (_DIM - 2), dtype=np.float32)  # title과 sim≈0.707


def _s(cid: int, text: str = "문장") -> SentenceInputDto:
    return SentenceInputDto(
        content_sentence_id=cid,
        sentence_text=text,
        sentence_order=cid,
    )


class TestTitleBodyGapCalculator:
    def setup_method(self) -> None:
        with patch("analysis_bc.tagger.title_body_gap.SentenceTransformer") as mock_st_cls:
            self.mock_model = MagicMock()
            mock_st_cls.return_value = self.mock_model
            self.calc = TitleBodyGapCalculator()

    def _set_encode(self, title_emb: np.ndarray, sentence_embs: np.ndarray) -> None:
        """title(str) 호출 → title_emb, batch(list) 호출 → sentence_embs 반환."""
        def fake_encode(inp: object, **kw: object) -> np.ndarray:
            if isinstance(inp, str):
                return title_emb
            return sentence_embs
        self.mock_model.encode.side_effect = fake_encode

    # ── 케이스 1: 빈 입력 ──────────────────────────────────────────────────

    def test_empty_sentences_returns_zero(self) -> None:
        result = self.calc.calculate("제목", [])
        assert result == GapResult(gap_score=0.0, gap_std=0.0, gap_lead=0.0, gap_tail=0.0, gap_label="unknown")

    # ── 케이스 2: 1문장 — lead == tail == gap_score, std == 0 ──────────────

    def test_single_sentence_lead_tail_equal(self) -> None:
        self._set_encode(_SIMILAR, np.stack([_HALF]))
        result = self.calc.calculate("제목", [_s(0)])
        assert result.gap_lead == result.gap_tail
        assert result.gap_lead == result.gap_score
        assert result.gap_std == 0.0

    # ── 케이스 3: gap_score = 1 - cos_sim 공식 검증 ───────────────────────

    def test_gap_score_equals_one_minus_similarity(self) -> None:
        # cos_sim(SIMILAR, HALF) ≈ 0.7071 → gap ≈ 0.2929
        self._set_encode(_SIMILAR, np.stack([_HALF]))
        result = self.calc.calculate("제목", [_s(0)])
        assert result.gap_score == pytest.approx(1.0 - 0.7071, abs=0.01)

    # ── 케이스 4: 위치 가중치 — 앞 문장이 뒤 문장보다 더 영향 ──────────────

    def test_position_decay_weights_lead_more(self) -> None:
        # 문장 0: title과 동일(sim=1.0), 문장 1: 완전 직교(sim=0.0)
        # 단순 평균이면 gap ≈ 0.29, 위치 가중치로 앞 문장 비중↑ → gap < 0.25
        self._set_encode(_SIMILAR, np.stack([_SIMILAR, _ORTHOG]))
        result = self.calc.calculate("제목", [_s(0), _s(1)])
        assert result.gap_score < 0.25

    # ── 케이스 5: buried_lede — gap_tail >> gap_lead ──────────────────────

    def test_buried_lede_high_tail_low_lead(self) -> None:
        # 앞 3문장: SIMILAR (title과 유사), 뒤 3문장: ORTHOG (title과 무관)
        embs = np.stack([_SIMILAR] * 3 + [_ORTHOG] * 3)
        self._set_encode(_SIMILAR, embs)
        result = self.calc.calculate("제목", [_s(i) for i in range(6)])
        assert result.gap_lead < 0.1
        assert result.gap_tail > 0.9
        assert (result.gap_tail - result.gap_lead) > 0.4

    # ── 케이스 6: clickbait — 모든 문장이 title과 무관 ────────────────────

    def test_clickbait_high_lead_gap(self) -> None:
        embs = np.stack([_ORTHOG] * 5)
        self._set_encode(_SIMILAR, embs)
        result = self.calc.calculate("제목", [_s(i) for i in range(5)])
        assert result.gap_lead > 0.9
        assert result.gap_score > 0.9

    # ── 케이스 7: gap_std == 0 (동일 유사도) ──────────────────────────────

    def test_gap_std_zero_for_single_sentence(self) -> None:
        self._set_encode(_SIMILAR, np.stack([_SIMILAR]))
        result = self.calc.calculate("제목", [_s(0)])
        assert result.gap_std == 0.0

    # ── 케이스 8: gap_std > 0 (유사도 분산 있음) ──────────────────────────

    def test_gap_std_nonzero_for_mixed_similarity(self) -> None:
        # per_sim = [1.0, 0.0] → std = 0.5
        embs = np.stack([_SIMILAR, _ORTHOG])
        self._set_encode(_SIMILAR, embs)
        result = self.calc.calculate("제목", [_s(0), _s(1)])
        assert result.gap_std > 0.0

    # ── 케이스 9~12: gap_label 분류 검증 ─────────────────────────────────

    def test_label_trustworthy(self) -> None:
        # gap_score ≈ 0, gap_std = 0 (단일 SIMILAR 문장) → trustworthy
        self._set_encode(_SIMILAR, np.stack([_SIMILAR]))
        result = self.calc.calculate("제목", [_s(0)])
        assert result.gap_label == "trustworthy"

    def test_label_trustworthy_high_std_becomes_neutral(self) -> None:
        # gap_score 낮아도 gap_std > _STD_HIGH(0.20) 이면 trustworthy 제외 → neutral
        # per_sim = [1.0, 0.0] → std = 0.5 > 0.20
        embs = np.stack([_SIMILAR, _ORTHOG])
        self._set_encode(_SIMILAR, embs)
        result = self.calc.calculate("제목", [_s(i) for i in range(2)])
        assert result.gap_label == "neutral"

    def test_label_clickbait(self) -> None:
        # gap_lead > 0.4, diff ≤ 0.4 → clickbait
        # 5문장 모두 ORTHOG: gap_lead ≈ 1.0, gap_tail ≈ 1.0, diff ≈ 0
        embs = np.stack([_ORTHOG] * 5)
        self._set_encode(_SIMILAR, embs)
        result = self.calc.calculate("제목", [_s(i) for i in range(5)])
        assert result.gap_label == "clickbait"

    def test_label_clickbait_priority_over_buried(self) -> None:
        # gap_lead > 0.4 이면서 diff > 0.4 인 경우 → clickbait (우선순위 수정 검증)
        # 앞 3문장 ORTHOG(gap_lead≈1.0), 뒤 3문장 SIMILAR(gap_tail≈0.0) → diff < 0, gap_lead > 0.4
        embs = np.stack([_ORTHOG] * 3 + [_SIMILAR] * 3)
        self._set_encode(_SIMILAR, embs)
        result = self.calc.calculate("제목", [_s(i) for i in range(6)])
        assert result.gap_label == "clickbait"

    def test_label_buried_lede(self) -> None:
        # 앞 3문장 SIMILAR(gap_lead≈0), 뒤 3문장 ORTHOG(gap_tail≈1.0) → diff > 0.4 → buried_lede
        embs = np.stack([_SIMILAR] * 3 + [_ORTHOG] * 3)
        self._set_encode(_SIMILAR, embs)
        result = self.calc.calculate("제목", [_s(i) for i in range(6)])
        assert result.gap_label == "buried_lede"

    def test_label_neutral(self) -> None:
        # cos_sim(SIMILAR, _mod) = 0.65 → gap = 0.35
        # 모든 문장이 _mod → gap_lead=0.35 ≤ 0.4, diff=0 ≤ 0.4, gap_score=0.35 ≥ 0.3 → neutral
        _mod = np.array([0.65, 0.7599] + [0.0] * (_DIM - 2), dtype=np.float32)
        embs = np.stack([_mod, _mod, _mod])
        self._set_encode(_SIMILAR, embs)
        result = self.calc.calculate("제목", [_s(i) for i in range(3)])
        assert result.gap_label == "neutral"

    # ── 케이스 13~14: 짧은 기사 lead/tail 비중복 ─────────────────────────

    def test_short_article_3sentences_no_overlap(self) -> None:
        # 3문장 이하: lead=[0:3], tail 없음 → gap_lead == gap_tail
        embs = np.stack([_SIMILAR, _HALF, _ORTHOG])
        self._set_encode(_SIMILAR, embs)
        result = self.calc.calculate("제목", [_s(i) for i in range(3)])
        assert result.gap_lead == result.gap_tail

    def test_short_article_5sentences_no_overlap(self) -> None:
        # 5문장: lead=[0:3], tail=[3:5] — 중복 없음
        # lead(SIMILAR*3) gap ≈ 0, tail(ORTHOG*2) gap ≈ 1.0 → diff > 0.4 → buried_lede
        embs = np.stack([_SIMILAR] * 3 + [_ORTHOG] * 2)
        self._set_encode(_SIMILAR, embs)
        result = self.calc.calculate("제목", [_s(i) for i in range(5)])
        assert result.gap_lead < 0.1
        assert result.gap_tail > 0.9
        assert result.gap_label == "buried_lede"
