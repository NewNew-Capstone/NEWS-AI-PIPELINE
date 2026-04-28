from analysis_bc.enums import BiasKeywordType, EvidenceType, SentenceLabelType, TargetType
from analysis_bc.schemas import (
    AnalyzeRequestDto,
    BiasAnalysisKeywordDto,
    BiasAnalysisResultDto,
    BiasEvidenceDto,
    SentenceBiasLabelDto,
    SentenceInputDto,
)


def _make_sentence_input(order: int = 0) -> SentenceInputDto:
    return SentenceInputDto(
        content_sentence_id=order + 1,
        sentence_text=f"문장 {order}",
        sentence_order=order,
    )


def test_sentence_input_dto_required_fields() -> None:
    s = _make_sentence_input()
    assert s.content_sentence_id == 1
    assert s.sentence_text == "문장 0"
    assert s.sentence_order == 0
    assert s.start_time_ms is None
    assert s.end_time_ms is None


def test_sentence_input_dto_optional_time() -> None:
    s = SentenceInputDto(
        content_sentence_id=1,
        sentence_text="문장",
        sentence_order=0,
        start_time_ms=1000,
        end_time_ms=2000,
    )
    assert s.start_time_ms == 1000
    assert s.end_time_ms == 2000


def test_analyze_request_dto_valid() -> None:
    req = AnalyzeRequestDto(
        target_id=42,
        target_type=TargetType.YOUTUBE_VIDEO,
        transcript_id=7,
        country="KR",
        language="KO",
        sentences=[_make_sentence_input()],
    )
    assert req.target_id == 42
    assert req.target_type == TargetType.YOUTUBE_VIDEO
    assert len(req.sentences) == 1


def test_sentence_bias_label_dto_optional_defaults() -> None:
    label = SentenceBiasLabelDto(
        content_sentence_id=1,
        label_type=SentenceLabelType.OPINION_LIKE,
        score=0.8,
    )
    assert label.highlight_color is None
    assert label.evidence_keyword is None


def test_bias_analysis_keyword_dto_valid() -> None:
    kw = BiasAnalysisKeywordDto(
        keyword_text="감정적",
        keyword_type=BiasKeywordType.EMOTION,
        score=0.7,
    )
    assert kw.keyword_text == "감정적"
    assert kw.keyword_type == BiasKeywordType.EMOTION


def test_bias_evidence_dto_valid() -> None:
    ev = BiasEvidenceDto(
        evidence_type=EvidenceType.OPINION,
        title="제목",
        description="설명",
        source_text="원문",
        confidence_score=0.9,
    )
    assert ev.evidence_type == EvidenceType.OPINION
    assert ev.confidence_score == 0.9


def test_bias_analysis_result_dto_full() -> None:
    result = BiasAnalysisResultDto(
        target_id=42,
        transcript_id=7,
        overall_bias_score=0.5,
        opinion_score=0.6,
        emotion_score=0.4,
        summary_text="요약",
        perspective_summary="관점 요약",
        evidence_summary="근거 요약",
        tone_label="중립",
        keywords=[
            BiasAnalysisKeywordDto(keyword_text="kw", keyword_type=BiasKeywordType.FRAME, score=0.5)
        ],
        sentence_labels=[
            SentenceBiasLabelDto(content_sentence_id=1, label_type=SentenceLabelType.FACT_LIKE, score=0.9)
        ],
        evidences=[
            BiasEvidenceDto(
                evidence_type=EvidenceType.SPECULATION,
                title="t",
                description="d",
                source_text="s",
                confidence_score=0.8,
            )
        ],
    )
    assert result.target_id == 42
    assert result.transcript_id == 7
    assert result.headline_body_gap_score is None
    assert result.neutrality_score is None
    assert len(result.keywords) == 1
    assert len(result.sentence_labels) == 1
    assert len(result.evidences) == 1
