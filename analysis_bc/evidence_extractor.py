from __future__ import annotations

import logging

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import EvidenceType, SentenceLabelType
from analysis_bc.schemas import BiasEvidenceDto, SentenceInputDto, SpanLabelDto

logger = logging.getLogger(__name__)

_LABEL_TO_EVIDENCE: dict[str, EvidenceType] = {
    SentenceLabelType.EMOTIONALLY_LOADED.value: EvidenceType.EMOTION,
    SentenceLabelType.OPINION_LIKE.value: EvidenceType.OPINION,
}

_EVIDENCE_TITLE: dict[EvidenceType, str] = {
    EvidenceType.EMOTION: "감정적 표현",
    EvidenceType.OPINION: "주관적 의견",
}


class EvidenceExtractor:
    def extract(
        self,
        sentences: list[SentenceInputDto],
        classified: list[ClassifiedSentenceDto],
        span_labels: list[SpanLabelDto],
    ) -> list[BiasEvidenceDto]:
        sentence_map = {s.content_sentence_id: s.sentence_text for s in sentences}
        results: list[BiasEvidenceDto] = []

        for span in span_labels:
            label_value = (
                span.label_type.value
                if isinstance(span.label_type, SentenceLabelType)
                else span.label_type
            )
            evidence_type = _LABEL_TO_EVIDENCE.get(label_value)
            if evidence_type is None:
                continue

            sentence_text = sentence_map.get(span.content_sentence_id, "")
            source_text = span.matched_word or sentence_text[span.start_offset:span.end_offset]

            results.append(
                BiasEvidenceDto(
                    evidence_type=evidence_type,
                    title=_EVIDENCE_TITLE[evidence_type],
                    description=sentence_text,
                    source_text=source_text,
                    confidence_score=span.score,
                )
            )

        # opinion_like classified 문장도 OPINION evidence로 추가
        for sent in classified:
            if sent.label != "opinion_like":
                continue
            sentence_text = sentence_map.get(sent.content_sentence_id, sent.sentence_text)
            results.append(
                BiasEvidenceDto(
                    evidence_type=EvidenceType.OPINION,
                    title=_EVIDENCE_TITLE[EvidenceType.OPINION],
                    description=sentence_text,
                    source_text=sentence_text,
                    confidence_score=sent.confidence,
                )
            )

        return results
