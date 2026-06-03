from __future__ import annotations

from types import SimpleNamespace

from analysis_bc.emotion_keyword_llm_filter import EmotionKeywordLlmFilter
from analysis_bc.enums import BiasKeywordType
from analysis_bc.schemas import BiasAnalysisKeywordDto


def _keyword(text: str, keyword_type: BiasKeywordType = BiasKeywordType.EMOTION) -> BiasAnalysisKeywordDto:
    return BiasAnalysisKeywordDto(keyword_text=text, keyword_type=keyword_type, score=1.0)


class _FakeMessages:
    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(text=self.text or "{}")])


class _FakeClient:
    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self.messages = _FakeMessages(text=text, error=error)


def test_llm_filter_keeps_only_true_emotion_keywords() -> None:
    client = _FakeClient(
        text='{"keep":["재밌다","놀라다","분노"],"block":["친구","이야기","추진"]}'
    )
    keyword_filter = EmotionKeywordLlmFilter(client=client, api_key="test-key", enabled=True)
    rows = [
        _keyword("친구"),
        _keyword("이야기"),
        _keyword("추진"),
        _keyword("재밌다"),
        _keyword("놀라다"),
        _keyword("분노"),
        _keyword("정책", BiasKeywordType.TOPIC),
    ]

    result = keyword_filter.filter_keywords(rows)

    assert [row.keyword_text for row in result] == ["재밌다", "놀라다", "분노", "정책"]
    assert client.messages.calls == 1


def test_llm_filter_words_returns_keep_set() -> None:
    client = _FakeClient(text='{"keep":["아름답다","분노"],"block":["정책"]}')
    keyword_filter = EmotionKeywordLlmFilter(client=client, api_key="test-key", enabled=True)

    result = keyword_filter.filter_words(["정책", "아름답다", "분노"])

    assert result == frozenset({"아름답다", "분노"})
    assert client.messages.calls == 1


def test_llm_filter_falls_back_when_disabled_or_missing_key() -> None:
    client = _FakeClient(text='{"keep":[],"block":["분노"]}')
    rows = [_keyword("분노")]

    assert EmotionKeywordLlmFilter(client=client, api_key="", enabled=True).filter_keywords(rows) == rows
    assert EmotionKeywordLlmFilter(client=client, api_key="test-key", enabled=False).filter_keywords(rows) == rows
    assert client.messages.calls == 0


def test_llm_filter_falls_back_on_llm_failure() -> None:
    client = _FakeClient(error=RuntimeError("boom"))
    keyword_filter = EmotionKeywordLlmFilter(client=client, api_key="test-key", enabled=True)
    rows = [_keyword("친구"), _keyword("분노")]

    result = keyword_filter.filter_keywords(rows)

    assert result == rows
    assert client.messages.calls == 1


def test_llm_filter_caches_same_candidate_set() -> None:
    client = _FakeClient(text='{"keep":["분노"],"block":["친구"]}')
    keyword_filter = EmotionKeywordLlmFilter(client=client, api_key="test-key", enabled=True)
    rows = [_keyword("친구"), _keyword("분노")]

    first = keyword_filter.filter_keywords(rows)
    second = keyword_filter.filter_keywords(rows)

    assert [row.keyword_text for row in first] == ["분노"]
    assert [row.keyword_text for row in second] == ["분노"]
    assert client.messages.calls == 1
