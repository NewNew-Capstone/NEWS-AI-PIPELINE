from __future__ import annotations

from knowledge_graph_bc.keyword_expander import MultilingualKeywordExpander


def test_expand_uses_fallback_when_no_api_key(monkeypatch) -> None:
    expander = MultilingualKeywordExpander()
    monkeypatch.setattr(expander, "api_key", "")
    monkeypatch.setattr(expander, "_translate", lambda text, source, target: f"{target}:{text}")

    result = expander.expand("트럼프 대만", max_terms_per_language=3)

    assert result.requested_keyword == "트럼프 대만"
    assert result.ko[0] == "트럼프 대만"
    assert result.en[0].startswith("en:")
    assert result.zh[0].startswith("zh-CN:")


def test_expand_normalizes_and_limits_terms(monkeypatch) -> None:
    expander = MultilingualKeywordExpander()
    monkeypatch.setattr(
        expander,
        "_expand_with_llm",
        lambda base, limit: None,
    )
    monkeypatch.setattr(expander, "_translate", lambda text, source, target: "  test  term  ")

    result = expander.expand("  트럼프   대만  ", max_terms_per_language=1)

    assert result.requested_keyword == "트럼프 대만"
    assert result.ko == ["트럼프 대만"]
    assert result.en == ["test term"]
    assert result.zh == ["test term"]

