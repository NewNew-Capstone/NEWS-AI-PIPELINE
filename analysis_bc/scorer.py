from __future__ import annotations

from dataclasses import dataclass

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto


@dataclass
class ScorerWeights:
    w_opinion: float = 0.4
    w_emotion: float = 0.3
    w_fact:    float = 0.3
    gap_evidence_high_threshold: float = 0.7
    gap_evidence_mid_threshold:  float = 0.4

    def __post_init__(self) -> None:
        if abs(self.w_opinion + self.w_emotion + self.w_fact - 1.0) > 1e-6:
            raise ValueError(
                f"w_opinion + w_emotion + w_fact must equal 1.0, "
                f"got {self.w_opinion + self.w_emotion + self.w_fact}"
            )


_EMPTY_RESULT: dict = {
    "score_evidence": "",
    "bias_type_scores": {"OPINION": 0.0, "EMOTIONAL": 0.0, "FACT": 0.0},
    "opinion_score": 0.0,
    "emotion_score": 0.0,
    "fact_ratio": 0.0,
    "overall_bias_score": 0.0,
}


class BiasScorer:
    """수식 기반 편향 점수 계산기 (LLM 로직 없음)."""

    def __init__(self, weights: ScorerWeights = ScorerWeights()) -> None:
        self.weights = weights

    def calculate(
        self,
        classified: list[ClassifiedSentenceDto],
        span_labels: list[SpanLabelDto],
        headline_body_gap: float,
    ) -> dict:
        total = len(classified)
        if total == 0:
            return dict(_EMPTY_RESULT)

        # ① opinion / fact 문장 추출
        opinion_sentences = [s for s in classified if s.label == "opinion_like"]
        fact_count = len([s for s in classified if s.label == "fact_like"])

        opinion_score = len(opinion_sentences) / total
        fact_ratio    = fact_count / total

        # ② 감정 span이 1개 이상 있는 고유 문장 수
        emotional_sentence_count = len({
            s.content_sentence_id for s in span_labels
            if s.label_type in (
                SentenceLabelType.EMOTIONALLY_LOADED,
                SentenceLabelType.EMOTIONALLY_LOADED.value,
            )
        })
        emotion_score = emotional_sentence_count / total

        # ③ overall_bias_score (Vargas 2023 / Garimella 2025 / Media Bias Detector 2024)
        overall_bias_score = round(
            min(
                self.weights.w_opinion * opinion_score
                + self.weights.w_emotion * emotion_score
                + self.weights.w_fact   * (1 - fact_ratio),
                1.0,
            ),
            4,
        )

        score_evidence = self._build_evidence(
            classified=classified,
            opinion_sentences=opinion_sentences,
            emotional_sentence_count=emotional_sentence_count,
            total=total,
            fact_ratio=fact_ratio,
            headline_body_gap=headline_body_gap,
        )

        return {
            "score_evidence": score_evidence,
            "bias_type_scores": {
                "OPINION":   round(opinion_score, 4),
                "EMOTIONAL": round(emotion_score, 4),
                "FACT":      round(fact_ratio,    4),
            },
            "opinion_score":      round(opinion_score, 4),
            "emotion_score":      round(emotion_score, 4),
            "fact_ratio":         round(fact_ratio,    4),
            "overall_bias_score": overall_bias_score,
        }

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _build_evidence(
        self,
        classified: list[ClassifiedSentenceDto],
        opinion_sentences: list[ClassifiedSentenceDto],
        emotional_sentence_count: int,
        total: int,
        fact_ratio: float,
        headline_body_gap: float,
    ) -> str:
        opinion_count = len(opinion_sentences)
        evidence: list[str] = []

        evidence.append(
            f"전체 문장 중 {round(opinion_count / total * 100)}%가 주관적 문장입니다."
        )

        if emotional_sentence_count > 0:
            evidence.append(f"감정적 표현이 {emotional_sentence_count}개 문장에서 감지되었습니다.")

        if fact_ratio < 0.3:
            evidence.append("사실 기반 문장 비율이 낮습니다.")

        if headline_body_gap >= self.weights.gap_evidence_high_threshold:
            evidence.append(
                f"제목과 본문 내용의 차이가 큽니다. (갭 점수: {headline_body_gap:.2f})"
            )
        elif headline_body_gap >= self.weights.gap_evidence_mid_threshold:
            evidence.append(
                f"제목과 본문 사이에 다소 차이가 있습니다. (갭 점수: {headline_body_gap:.2f})"
            )

        return " ".join(evidence)
