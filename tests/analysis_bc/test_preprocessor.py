from __future__ import annotations

from unittest.mock import patch

import pytest
from langdetect import LangDetectException

from analysis_bc.preprocessor import MIN_CHAR_LENGTH, SentencePreprocessor
from analysis_bc.schemas import SentenceInputDto


def _s(
    cid: int,
    text: str,
    order: int,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
) -> SentenceInputDto:
    return SentenceInputDto(
        content_sentence_id=cid,
        sentence_text=text,
        sentence_order=order,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )


@pytest.fixture
def preprocessor() -> SentencePreprocessor:
    return SentencePreprocessor(expected_language="ko")


# ── normalize ────────────────────────────────────────────────────────────────


def test_normalize_strips_whitespace(preprocessor: SentencePreprocessor) -> None:
    assert preprocessor._normalize("  안녕하세요  ") == "안녕하세요"


def test_normalize_multispace_collapsed(preprocessor: SentencePreprocessor) -> None:
    assert preprocessor._normalize("a  b   c") == "a b c"


# ── _is_long_enough ───────────────────────────────────────────────────────────


def test_is_long_enough_boundary(preprocessor: SentencePreprocessor) -> None:
    assert preprocessor._is_long_enough("1234") is False   # len=4, MIN=5
    assert preprocessor._is_long_enough("12345") is True   # len=5, MIN=5


def test_min_char_length_constant() -> None:
    assert MIN_CHAR_LENGTH == 5


# ── _is_valid_language ────────────────────────────────────────────────────────


def test_is_valid_language_match(preprocessor: SentencePreprocessor) -> None:
    with patch("analysis_bc.preprocessor.detect", return_value="ko"):
        assert preprocessor._is_valid_language("한국어 문장") is True


def test_is_valid_language_mismatch(preprocessor: SentencePreprocessor) -> None:
    with patch("analysis_bc.preprocessor.detect", return_value="en"):
        assert preprocessor._is_valid_language("English sentence") is False


def test_is_valid_language_detect_fails(preprocessor: SentencePreprocessor) -> None:
    with patch(
        "analysis_bc.preprocessor.detect",
        side_effect=LangDetectException(0, "no features"),
    ):
        assert preprocessor._is_valid_language("???") is False


# ── preprocess: 정렬 ──────────────────────────────────────────────────────────


def test_preprocess_sorts_by_order(preprocessor: SentencePreprocessor) -> None:
    sentences = [
        _s(3, "정렬테스트문장셋", 2),
        _s(1, "정렬테스트문장원", 0),
        _s(2, "정렬테스트문장투", 1),
    ]
    with patch("analysis_bc.preprocessor.detect", return_value="ko"):
        result = preprocessor.preprocess(sentences)

    assert [s.sentence_order for s in result] == [0, 1, 2]
    assert result[0].content_sentence_id == 1


# ── preprocess: 짧은 문장 필터 ────────────────────────────────────────────────


def test_preprocess_removes_short_sentences(preprocessor: SentencePreprocessor) -> None:
    sentences = [
        _s(1, "짧음", 0),            # len=2 → 제거
        _s(2, "한국어긴문장입니다", 1),  # len=9 → 통과
    ]
    with patch("analysis_bc.preprocessor.detect", return_value="ko"):
        result = preprocessor.preprocess(sentences)

    assert len(result) == 1
    assert result[0].content_sentence_id == 2


# ── preprocess: 언어 필터 ─────────────────────────────────────────────────────


def test_preprocess_removes_wrong_language(preprocessor: SentencePreprocessor) -> None:
    sentences = [
        _s(1, "한국어문장이에요", 0),       # ko → 통과
        _s(2, "English sentence here", 1),  # en → 제거
    ]
    with patch("analysis_bc.preprocessor.detect", side_effect=["ko", "en"]):
        result = preprocessor.preprocess(sentences)

    assert len(result) == 1
    assert result[0].content_sentence_id == 1


def test_preprocess_removes_detect_failure(preprocessor: SentencePreprocessor) -> None:
    sentences = [
        _s(1, "한국어문장입니다", 0),
        _s(2, "감지실패케이스예요", 1),
        _s(3, "또다른한국어문장", 2),
    ]
    side_effects: list = ["ko", LangDetectException(0, ""), "ko"]
    with patch("analysis_bc.preprocessor.detect", side_effect=side_effects):
        result = preprocessor.preprocess(sentences)

    assert len(result) == 2
    assert result[0].content_sentence_id == 1
    assert result[1].content_sentence_id == 3


# ── preprocess: 정규화 반영 ───────────────────────────────────────────────────


def test_preprocess_updates_sentence_text(preprocessor: SentencePreprocessor) -> None:
    sentence = _s(1, "  공백  포함  문장  ", 0)
    with patch("analysis_bc.preprocessor.detect", return_value="ko"):
        result = preprocessor.preprocess([sentence])

    assert result[0].sentence_text == "공백 포함 문장"
    assert sentence.sentence_text == "  공백  포함  문장  "  # 원본 불변


# ── preprocess: 메타데이터 보존 ───────────────────────────────────────────────


def test_preprocess_preserves_metadata(preprocessor: SentencePreprocessor) -> None:
    sentence = _s(42, "메타데이터보존테스트", 0, start_time_ms=1000, end_time_ms=2000)
    with patch("analysis_bc.preprocessor.detect", return_value="ko"):
        result = preprocessor.preprocess([sentence])

    dto = result[0]
    assert dto.content_sentence_id == 42
    assert dto.start_time_ms == 1000
    assert dto.end_time_ms == 2000


# ── preprocess: 경계 케이스 ───────────────────────────────────────────────────


def test_preprocess_empty_input(preprocessor: SentencePreprocessor) -> None:
    assert preprocessor.preprocess([]) == []


def test_preprocess_all_filtered_out(preprocessor: SentencePreprocessor) -> None:
    sentences = [_s(1, "짧", 0), _s(2, "음", 1)]
    result = preprocessor.preprocess(sentences)
    assert result == []


# ── preprocess: 통합 ─────────────────────────────────────────────────────────


def test_preprocess_integration_mixed(preprocessor: SentencePreprocessor) -> None:
    """정렬 + 정규화 + 짧은 문장 필터 + 언어 필터 통합 검증."""
    sentences = [
        _s(3, "이건  긴  문장입니다", 2),         # 정규화 후 ko → 통과
        _s(1, "짧음", 0),                         # len=2 → 제거 (detect 미호출)
        _s(4, "English sentence here", 3),        # en → 제거
        _s(2, "다른  긴  한국어  문장", 1),         # 정규화 후 ko → 통과
    ]
    # detect 호출 순서: order=1("다른..."), order=2("이건..."), order=3("English...")
    with patch("analysis_bc.preprocessor.detect", side_effect=["ko", "ko", "en"]):
        result = preprocessor.preprocess(sentences)

    assert len(result) == 2
    assert result[0].sentence_order == 1
    assert result[0].sentence_text == "다른 긴 한국어 문장"
    assert result[1].sentence_order == 2
    assert result[1].sentence_text == "이건 긴 문장입니다"
