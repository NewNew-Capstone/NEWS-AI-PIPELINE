from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import SentenceLabelType
from analysis_bc.tagger.emotional_tagger import EmotionalTagger


@pytest.fixture
def mock_tagger() -> EmotionalTagger:
    """실제 모델/Qdrant 연결 없이 EmotionalTagger 인스턴스 반환."""
    with (
        patch("analysis_bc.tagger.emotional_tagger.Kiwi") as mock_kiwi_cls,
        patch(
            "analysis_bc.tagger.emotional_tagger.SentenceTransformer"
        ) as mock_sbert_cls,
        patch(
            "analysis_bc.tagger.emotional_tagger.QdrantClient"
        ) as mock_qdrant_cls,
    ):
        mock_kiwi_cls.return_value = MagicMock()
        mock_sbert_cls.return_value = MagicMock()
        mock_qdrant_cls.return_value = MagicMock()
        tagger = EmotionalTagger()
    return tagger


def _make_sentence(
    cid: int, text: str, order: int = 0, label: str = "opinion_like"
) -> ClassifiedSentenceDto:
    return ClassifiedSentenceDto(
        content_sentence_id=cid,
        sentence_text=text,
        sentence_order=order,
        start_time_ms=None,
        end_time_ms=None,
        label=label,
        confidence=0.9,
    )


def _make_token(form: str, start: int, length: int) -> MagicMock:
    token = MagicMock()
    token.form = form
    token.start = start
    token.len = length
    return token


def test_tag_returns_emotionally_loaded_span(mock_tagger: EmotionalTagger) -> None:
    """끔찍한 위치에 EMOTIONALLY_LOADED 태깅 및 offset 검증."""
    text = "끔찍한 사건이 발생했다"
    sentence = _make_sentence(1, text)

    # KIWI: "끔찍한" 토큰 (start=0, len=3)
    mock_token = _make_token("끔찍한", 0, 3)
    mock_tagger.kiwi.tokenize.return_value = [mock_token]

    # 임베딩
    import numpy as np
    mock_tagger.model.encode.return_value = np.zeros(768)

    # Qdrant 검색 결과: 유사도 0.92로 매칭
    hit = MagicMock()
    hit.score = 0.92
    hit.payload = {"word_root": "끔찍", "word": "끔찍한", "polarity": "-2"}
    mock_tagger.qdrant.search.return_value = [hit]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    span = result[0]
    assert span.content_sentence_id == 1
    assert span.label_type == SentenceLabelType.EMOTIONALLY_LOADED
    assert span.start_offset == 0
    assert span.end_offset == 3
    assert span.score == pytest.approx(0.92)
    assert span.matched_word == "끔찍"


def test_tag_no_match_returns_empty(mock_tagger: EmotionalTagger) -> None:
    """Qdrant 검색 결과 없음 → 빈 리스트 반환."""
    sentence = _make_sentence(2, "평범한 문장입니다")

    mock_token = _make_token("평범한", 0, 3)
    mock_tagger.kiwi.tokenize.return_value = [mock_token]

    import numpy as np
    mock_tagger.model.encode.return_value = np.zeros(768)
    mock_tagger.qdrant.search.return_value = []

    result = mock_tagger.tag([sentence])

    assert result == []


def test_tag_empty_sentences(mock_tagger: EmotionalTagger) -> None:
    """빈 입력 → 빈 리스트 반환."""
    assert mock_tagger.tag([]) == []


def test_tag_multiple_tokens_one_match(mock_tagger: EmotionalTagger) -> None:
    """여러 토큰 중 1개만 매칭 → span 1개 반환."""
    sentence = _make_sentence(3, "끔찍한 사건이 발생했다")

    tokens = [
        _make_token("끔찍한", 0, 3),
        _make_token("사건", 4, 2),
        _make_token("이", 6, 1),
    ]
    mock_tagger.kiwi.tokenize.return_value = tokens

    import numpy as np
    mock_tagger.model.encode.return_value = np.zeros(768)

    hit = MagicMock()
    hit.score = 0.91
    hit.payload = {"word_root": "끔찍"}
    # 첫 토큰만 매칭, 나머지는 빈 결과
    mock_tagger.qdrant.search.side_effect = [[hit], [], []]

    result = mock_tagger.tag([sentence])

    assert len(result) == 1
    assert result[0].start_offset == 0
    assert result[0].end_offset == 3
