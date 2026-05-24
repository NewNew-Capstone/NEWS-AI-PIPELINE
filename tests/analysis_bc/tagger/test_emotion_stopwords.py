from __future__ import annotations

from analysis_bc.tagger.emotion_stopwords import (
    EMOTION_STOPWORDS,
    is_blocked_emotion_stopword,
)


def test_emotion_stopwords_include_review_decisions() -> None:
    assert "함께" in EMOTION_STOPWORDS
    assert "대한민국" in EMOTION_STOPWORDS
    assert "대통령" in EMOTION_STOPWORDS
    assert "트럼프" in EMOTION_STOPWORDS


def test_is_blocked_emotion_stopword_matches_surface_and_form() -> None:
    assert is_blocked_emotion_stopword("함께")
    assert is_blocked_emotion_stopword("1부리")
    assert is_blocked_emotion_stopword("잘못된", "잘못")
    assert is_blocked_emotion_stopword("트럼프", "트럼프")


def test_is_blocked_emotion_stopword_keeps_clear_emotion_words() -> None:
    assert not is_blocked_emotion_stopword("분노")
    assert not is_blocked_emotion_stopword("불안")
