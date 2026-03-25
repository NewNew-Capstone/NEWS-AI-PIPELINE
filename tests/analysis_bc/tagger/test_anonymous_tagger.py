from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from analysis_bc.enums import SentenceLabelType
from analysis_bc.schemas import SentenceInputDto
from analysis_bc.tagger.anonymous_tagger import (
    AnonymousTagger,
    _DEFAULT_ANONYMOUS_PATTERNS,
    _DEFAULT_SPECULATIVE_PATTERNS,
)


def _make_sentence(cid: int, text: str, order: int = 0) -> SentenceInputDto:
    return SentenceInputDto(
        content_sentence_id=cid,
        sentence_text=text,
        sentence_order=order,
    )


@pytest.fixture
def mock_redis() -> MagicMock:
    redis = MagicMock()
    redis.exists.return_value = True  # 기본: 패턴이 이미 존재
    redis.lrange.side_effect = lambda key, *_: (
        _DEFAULT_ANONYMOUS_PATTERNS
        if "anonymous" in key
        else _DEFAULT_SPECULATIVE_PATTERNS
    )
    return redis


@pytest.fixture
def mock_tagger(mock_redis: MagicMock) -> AnonymousTagger:
    """실제 Redis 연결 없이 AnonymousTagger 인스턴스 반환."""
    with patch("analysis_bc.tagger.anonymous_tagger.Redis", return_value=mock_redis):
        tagger = AnonymousTagger()
    return tagger


# ── 익명출처 패턴 ─────────────────────────────────────────────────────────────


def test_tag_anonymous_source_pattern(mock_tagger: AnonymousTagger) -> None:
    """'관계자에 따르면' → ANONYMOUS_SOURCE 태깅."""
    sentence = _make_sentence(1, "관계자에 따르면 이번 사태는 심각하다")

    result = mock_tagger.tag([sentence])

    assert len(result) >= 1
    span = next(s for s in result if s.label_type == SentenceLabelType.ANONYMOUS_SOURCE)
    assert span.content_sentence_id == 1
    assert span.start_offset == 0
    assert span.end_offset == len("관계자에 따르면")
    assert span.matched_word == "관계자에 따르면"
    assert span.score == pytest.approx(1.0)


def test_tag_no_pattern_returns_empty(mock_tagger: AnonymousTagger) -> None:
    """패턴 없는 문장 → 빈 리스트 반환."""
    sentence = _make_sentence(2, "서울 아파트 가격이 10억을 넘었다")

    result = mock_tagger.tag([sentence])

    assert result == []


# ── 추측어미 패턴 ─────────────────────────────────────────────────────────────


def test_tag_speculative_pattern(mock_tagger: AnonymousTagger) -> None:
    """'것으로 알려졌다' → SPECULATIVE 태깅."""
    text = "이번 사건은 내부 갈등 때문인 것으로 알려졌다"
    sentence = _make_sentence(3, text)

    result = mock_tagger.tag([sentence])

    speculative = [s for s in result if s.label_type == SentenceLabelType.SPECULATIVE]
    assert len(speculative) >= 1
    assert speculative[0].matched_word == "것으로 알려졌다"


# ── 빈 입력 ───────────────────────────────────────────────────────────────────


def test_tag_empty_sentences(mock_tagger: AnonymousTagger) -> None:
    assert mock_tagger.tag([]) == []


# ── Redis 초기화 ──────────────────────────────────────────────────────────────


def test_init_patterns_when_redis_empty() -> None:
    """Redis에 키가 없으면 기본 패턴을 rpush로 초기화한다."""
    mock_redis = MagicMock()
    mock_redis.exists.return_value = False  # 키 없음

    with patch("analysis_bc.tagger.anonymous_tagger.Redis", return_value=mock_redis):
        AnonymousTagger()

    # rpush가 두 번(anonymous, speculative) 호출됐는지 확인
    assert mock_redis.rpush.call_count == 2
    calls = mock_redis.rpush.call_args_list
    keys = [call.args[0] for call in calls]
    assert any("anonymous" in k for k in keys)
    assert any("speculative" in k for k in keys)


def test_init_patterns_skips_when_redis_has_data() -> None:
    """Redis에 키가 이미 있으면 rpush를 호출하지 않는다."""
    mock_redis = MagicMock()
    mock_redis.exists.return_value = True  # 키 이미 존재

    with patch("analysis_bc.tagger.anonymous_tagger.Redis", return_value=mock_redis):
        AnonymousTagger()

    mock_redis.rpush.assert_not_called()


# ── 복수 문장 ─────────────────────────────────────────────────────────────────


def test_tag_multiple_sentences(mock_tagger: AnonymousTagger) -> None:
    """여러 문장 중 일부만 패턴 매칭."""
    sentences = [
        _make_sentence(1, "관계자에 따르면 이번 사태는 심각하다"),
        _make_sentence(2, "사실에 기반한 보도입니다"),
        _make_sentence(3, "전망이다 라는 표현이 있다"),
    ]

    result = mock_tagger.tag(sentences)

    cids = {s.content_sentence_id for s in result}
    assert 1 in cids   # 익명출처 패턴
    assert 2 not in cids  # 패턴 없음
    assert 3 in cids   # 추측어미 패턴
