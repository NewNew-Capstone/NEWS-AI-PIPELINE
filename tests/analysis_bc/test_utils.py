from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SpanLabelDto
from analysis_bc.utils import resolve_overlapping_spans


def _make_span(start: int, end: int, label: SentenceLabelType) -> SpanLabelDto:
    return SpanLabelDto(
        content_sentence_id=1,
        start_offset=start,
        end_offset=end,
        label_type=label,
        score=1.0,
    )


def test_overlapping_keeps_higher_priority() -> None:
    """겹치는 경우 우선순위 높은 것(SPECULATIVE)만 남긴다."""
    spans = [
        _make_span(0, 13, SentenceLabelType.SPECULATIVE),
        _make_span(0, 3, SentenceLabelType.EMOTIONALLY_LOADED),
    ]

    result = resolve_overlapping_spans(spans)

    assert len(result) == 1
    assert result[0].label_type == SentenceLabelType.SPECULATIVE
    assert result[0].start_offset == 0
    assert result[0].end_offset == 13


def test_non_overlapping_keeps_both() -> None:
    """겹치지 않는 경우 둘 다 start_offset 오름차순으로 반환한다."""
    spans = [
        _make_span(0, 3, SentenceLabelType.EMOTIONALLY_LOADED),
        _make_span(4, 12, SentenceLabelType.ANONYMOUS_SOURCE),
    ]

    result = resolve_overlapping_spans(spans)

    assert len(result) == 2
    assert result[0].start_offset == 0
    assert result[0].label_type == SentenceLabelType.EMOTIONALLY_LOADED
    assert result[1].start_offset == 4
    assert result[1].label_type == SentenceLabelType.ANONYMOUS_SOURCE


def test_empty_input_returns_empty() -> None:
    """빈 입력 → 빈 리스트."""
    assert resolve_overlapping_spans([]) == []


def test_three_spans_two_overlap() -> None:
    """3개 중 2개가 겹치는 경우: 우선순위 높은 것 + 안 겹치는 것 반환."""
    spans = [
        _make_span(0, 13, SentenceLabelType.SPECULATIVE),
        _make_span(0, 3, SentenceLabelType.EMOTIONALLY_LOADED),
        _make_span(14, 20, SentenceLabelType.ANONYMOUS_SOURCE),
    ]

    result = resolve_overlapping_spans(spans)

    assert len(result) == 2
    assert result[0].start_offset == 0
    assert result[0].label_type == SentenceLabelType.SPECULATIVE
    assert result[1].start_offset == 14
    assert result[1].label_type == SentenceLabelType.ANONYMOUS_SOURCE
