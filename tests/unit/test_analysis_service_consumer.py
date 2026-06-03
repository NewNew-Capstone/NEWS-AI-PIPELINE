from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import EvidenceType, SentenceLabelType, TargetType
from analysis_bc.evidence_extractor import EvidenceExtractor
from analysis_bc.keyword_extractor import KeywordExtractor
from analysis_bc.schemas import AnalyzeRequestDto, SentenceInputDto, SpanLabelDto
from analysis_bc.scorer import BiasScorer
from analysis_bc.service import AnalysisService
from analysis_bc.tagger.title_body_gap import GapResult


def _make_request(num_sentences: int = 2) -> AnalyzeRequestDto:
    return AnalyzeRequestDto(
        target_id=1,
        title="테스트 뉴스 제목",
        target_type=TargetType.YOUTUBE_VIDEO,
        transcript_id=10,
        country="KR",
        language="KO",
        sentences=[
            SentenceInputDto(
                content_sentence_id=i + 1,
                sentence_text=f"문장 {i}",
                sentence_order=i,
            )
            for i in range(num_sentences)
        ],
    )


def _patch_taggers():
    """EmotionalTagger, TitleBodyGapCalculator mock."""
    return (
        patch("analysis_bc.tagger.emotional_tagger.Kiwi"),
        patch("analysis_bc.tagger.emotional_tagger.SentenceTransformer"),
        patch("analysis_bc.tagger.emotional_tagger.QdrantClient"),
        patch("analysis_bc.tagger.title_body_gap.SentenceTransformer"),
    )


def test_analyze_content_returns_target_id() -> None:
    request = _make_request(num_sentences=3)

    patches = _patch_taggers()
    with patches[0], patches[1], patches[2], patches[3]:
        result = AnalysisService().analyze(request)

    assert result.target_id == 1
    assert result.transcript_id == 10


def test_analyze_content_stub_returns_empty_lists() -> None:
    request = _make_request(num_sentences=2)

    patches = _patch_taggers()
    with patches[0], patches[1], patches[2], patches[3]:
        result = AnalysisService().analyze(request)

    assert result.sentence_labels == []
    assert result.keywords == []
    assert result.focus_keywords == []
    assert result.evidences == []


def test_prepare_sentence_inputs_sorts_by_order() -> None:
    sentences = [
        SentenceInputDto(content_sentence_id=3, sentence_text="정렬테스트문장셋", sentence_order=2),
        SentenceInputDto(content_sentence_id=1, sentence_text="정렬테스트문장원", sentence_order=0),
        SentenceInputDto(content_sentence_id=2, sentence_text="정렬테스트문장투", sentence_order=1),
    ]

    patches = _patch_taggers()
    with patches[0], patches[1], patches[2], patches[3]:
        with patch("analysis_bc.preprocessor.detect", return_value="ko"):
            result = AnalysisService().prepare_sentences(sentences, "ko")

    assert [s.sentence_order for s in result] == [0, 1, 2]
    assert result[0].content_sentence_id == 1


def test_analyze_filters_llm_rejected_emotion_spans_before_scoring() -> None:
    request = AnalyzeRequestDto(
        target_id=1,
        title="테스트 뉴스 제목",
        target_type=TargetType.YOUTUBE_VIDEO,
        transcript_id=10,
        country="KR",
        language="ko",
        sentences=[
            SentenceInputDto(
                content_sentence_id=1,
                sentence_text="정책",
                sentence_order=0,
            ),
            SentenceInputDto(
                content_sentence_id=2,
                sentence_text="분노",
                sentence_order=1,
            ),
        ],
    )
    classified = [
        ClassifiedSentenceDto(
            content_sentence_id=1,
            sentence_text="정책",
            sentence_order=0,
            start_time_ms=None,
            end_time_ms=None,
            label="opinion_like",
            confidence=0.9,
        ),
        ClassifiedSentenceDto(
            content_sentence_id=2,
            sentence_text="분노",
            sentence_order=1,
            start_time_ms=None,
            end_time_ms=None,
            label="opinion_like",
            confidence=0.9,
        ),
    ]
    raw_spans = [
        SpanLabelDto(
            content_sentence_id=1,
            start_offset=0,
            end_offset=2,
            label_type=SentenceLabelType.EMOTIONALLY_LOADED,
            score=1.0,
            matched_word="정책",
        ),
        SpanLabelDto(
            content_sentence_id=2,
            start_offset=0,
            end_offset=2,
            label_type=SentenceLabelType.EMOTIONALLY_LOADED,
            score=1.0,
            matched_word="분노",
        ),
    ]

    service = object.__new__(AnalysisService)
    service.prepare_sentences = lambda sentences, _language: sentences
    service.classifier = SimpleNamespace(classify=lambda _sentences: classified)
    service.span_tagger = SimpleNamespace(tag=lambda _opinion_sentences: raw_spans)
    service.title_body_gap_calculator = SimpleNamespace(
        calculate=lambda **_kwargs: GapResult(
            gap_score=0.0,
            gap_std=0.0,
            gap_lead=0.0,
            gap_tail=0.0,
            gap_label="neutral",
        )
    )
    service.scorer = BiasScorer()
    service.summarizer = SimpleNamespace(
        build_score_reason_fallback=lambda **_kwargs: "fallback"
    )
    service.keyword_extractor = KeywordExtractor()
    service.emotion_keyword_filter = SimpleNamespace(
        filter_words=lambda _words: frozenset({"분노"})
    )
    service.evidence_extractor = EvidenceExtractor()

    result = service.analyze(request)

    assert result.emotion_score == 0.5
    assert [span.matched_word for span in result.sentence_labels] == ["분노"]
    assert [kw.keyword_text for kw in result.emotion_keywords] == ["분노"]
    emotion_evidences = [
        evidence for evidence in result.evidences
        if evidence.evidence_type == EvidenceType.EMOTION
    ]
    assert [evidence.source_text for evidence in emotion_evidences] == ["분노"]
