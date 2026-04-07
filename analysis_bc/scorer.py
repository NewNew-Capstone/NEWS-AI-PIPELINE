from __future__ import annotations

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto

_W_OPINION   = 0.6
_W_EMOTION   = 0.3
_W_ANONYMOUS = 0.1

_EMPTY_RESULT: dict = {
    "subjectivity_score": 0.0,
    "score_evidence": "",
    "bias_type_scores": {
        "OPINION": 0.0,
        "EMOTIONAL": 0.0,
        "ANONYMOUS": 0.0,
        "SPECULATIVE": 0.0,
    },
    "opinion_score": 0.0,
    "emotion_score": 0.0,
    "anonymous_source_score": 0.0,
    "overall_bias_score": 0.0,
}


class BiasScorer:
    """수식 기반 편향 점수 계산기 (LLM 로직 없음)."""

    def calculate(
        self,
        classified: list[ClassifiedSentenceDto],
        span_labels: list[SpanLabelDto],
        headline_body_gap: float,
    ) -> dict:
        total = len(classified)
        if total == 0:
            return dict(_EMPTY_RESULT)

        # ① opinion 문장 추출
        opinion_sentences = [s for s in classified if s.label == "opinion_like"]
        opinion_count = len(opinion_sentences)

        # ② raw_score
        raw_score = sum(
            s.confidence * self._get_position_weight(s.sentence_order, total)
            for s in opinion_sentences
        ) / total * 100

        # ③ gap_weight → subjectivity_score
        gap_weight = 1.0 + headline_body_gap * 0.3
        subjectivity_score = round(min(raw_score * gap_weight, 100.0), 2)

        # ④ 문장 기준 span 카운트 (감정/익명/추측 span이 1개 이상 있는 고유 문장 수)
        emotional_sentence_count = len({
            s.content_sentence_id for s in span_labels
            if s.label_type in (
                SentenceLabelType.EMOTIONALLY_LOADED,
                SentenceLabelType.EMOTIONALLY_LOADED.value,
            )
        })
        anonymous_sentence_count = len({
            s.content_sentence_id for s in span_labels
            if s.label_type in (
                SentenceLabelType.ANONYMOUS_SOURCE,
                SentenceLabelType.ANONYMOUS_SOURCE.value,
            )
        })
        speculative_sentence_count = len({
            s.content_sentence_id for s in span_labels
            if s.label_type in (
                SentenceLabelType.SPECULATIVE,
                SentenceLabelType.SPECULATIVE.value,
            )
        })

        # ⑤ bias_type_scores (모두 문장 기준 → 0~1 보장)
        bias_type_scores = {
            "OPINION":     round(opinion_count              / total, 4),
            "EMOTIONAL":   round(emotional_sentence_count   / total, 4),
            "ANONYMOUS":   round(anonymous_sentence_count   / total, 4),
            "SPECULATIVE": round(speculative_sentence_count / total, 4),
        }

        # ⑥ score_evidence
        score_evidence = self._build_evidence(
            classified=classified,
            opinion_sentences=opinion_sentences,
            opinion_count=opinion_count,
            emotional_sentence_count=emotional_sentence_count,
            anonymous_sentence_count=anonymous_sentence_count,
            speculative_sentence_count=speculative_sentence_count,
            total=total,
            headline_body_gap=headline_body_gap,
        )

        # 문장 기준이므로 클리핑 불필요 (이미 0~1)
        overall_bias_score = round(
            _W_OPINION   * (subjectivity_score / 100)
            + _W_EMOTION   * bias_type_scores["EMOTIONAL"]
            + _W_ANONYMOUS * bias_type_scores["ANONYMOUS"],
            4,
        )

        return {
            "subjectivity_score":     subjectivity_score,
            "score_evidence":         score_evidence,
            "bias_type_scores":       bias_type_scores,
            "opinion_score":          bias_type_scores["OPINION"],
            "emotion_score":          bias_type_scores["EMOTIONAL"],
            "anonymous_source_score": bias_type_scores["ANONYMOUS"],
            "overall_bias_score":     overall_bias_score,
        }

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _get_position_weight(self, order: int, total: int) -> float:
        ratio = order / total
        if ratio <= 0.33:
            return 1.3
        elif ratio <= 0.66:
            return 1.0
        else:
            return 0.8

    def _build_evidence(
        self,
        classified: list[ClassifiedSentenceDto],
        opinion_sentences: list[ClassifiedSentenceDto],
        opinion_count: int,
        emotional_sentence_count: int,
        anonymous_sentence_count: int,
        speculative_sentence_count: int,
        total: int,
        headline_body_gap: float,
    ) -> str:
        evidence: list[str] = []

        # 빈도
        evidence.append(
            f"전체 문장 중 {round(opinion_count / total * 100)}%가 주관적 문장입니다."
        )

        # 위치 — 앞 33% 구간에 OPINION이 절반 이상
        front_opinion_count = sum(
            1 for s in opinion_sentences
            if s.sentence_order / total <= 0.33
        )
        if opinion_count > 0 and front_opinion_count / opinion_count >= 0.5:
            evidence.append("주관적 표현이 도입부에 집중되어 있습니다.")

        # span 근거 (문장 수 기준)
        if emotional_sentence_count > 0:
            evidence.append(f"감정적 표현이 {emotional_sentence_count}개 문장에서 감지되었습니다.")
        if anonymous_sentence_count > 0:
            evidence.append(f"익명 출처 표현이 {anonymous_sentence_count}개 문장에서 감지되었습니다.")
        if speculative_sentence_count > 0:
            evidence.append(f"추측성 표현이 {speculative_sentence_count}개 문장에서 감지되었습니다.")

        # 제목-본문 갭
        if headline_body_gap >= 0.7:
            evidence.append(
                f"제목과 본문 내용의 차이가 큽니다. (갭 점수: {headline_body_gap:.2f})"
            )
        elif headline_body_gap >= 0.4:
            evidence.append(
                f"제목과 본문 사이에 다소 차이가 있습니다. (갭 점수: {headline_body_gap:.2f})"
            )

        return " ".join(evidence)
