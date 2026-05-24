from unittest.mock import MagicMock, patch

from analysis_bc.enums import TargetType
from analysis_bc.schemas import AnalyzeRequestDto, SentenceInputDto
from analysis_bc.service import AnalysisService


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
