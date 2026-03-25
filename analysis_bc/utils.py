from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto

LABEL_PRIORITY: dict[str, int] = {
    SentenceLabelType.ANONYMOUS_SOURCE:   1,
    SentenceLabelType.SPECULATIVE:        2,
    SentenceLabelType.EMOTIONALLY_LOADED: 3,
}


def resolve_overlapping_spans(spans: list[SpanLabelDto]) -> list[SpanLabelDto]:
    """우선순위 기반으로 중복 offset span을 제거한다.

    동일 offset 구간에 복수 태그가 존재하면 LABEL_PRIORITY가 낮은(우선순위 높은) 것만 남긴다.
    결과는 start_offset 오름차순으로 반환한다.
    """
    sorted_spans = sorted(spans, key=lambda s: LABEL_PRIORITY.get(s.label_type, 99))
    result: list[SpanLabelDto] = []
    for span in sorted_spans:
        overlaps = any(
            not (span.end_offset <= r.start_offset or span.start_offset >= r.end_offset)
            for r in result
        )
        if not overlaps:
            result.append(span)
    return sorted(result, key=lambda s: s.start_offset)
