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
    assert "지우" in EMOTION_STOPWORDS
    assert "흐르다" in EMOTION_STOPWORDS
    assert "궁금하다" in EMOTION_STOPWORDS
    assert "정한" in EMOTION_STOPWORDS
    assert "똑같" in EMOTION_STOPWORDS
    assert "그렇다" in EMOTION_STOPWORDS
    assert "알리다" in EMOTION_STOPWORDS
    assert "어렵다" in EMOTION_STOPWORDS
    assert "죄송하다" in EMOTION_STOPWORDS
    assert "미치다" in EMOTION_STOPWORDS
    assert "위대하다" in EMOTION_STOPWORDS
    assert "내세우다" in EMOTION_STOPWORDS
    assert "꺼내다" in EMOTION_STOPWORDS
    assert "물어보다" in EMOTION_STOPWORDS
    assert "친구" in EMOTION_STOPWORDS
    assert "이야기" in EMOTION_STOPWORDS
    assert "추진하다" in EMOTION_STOPWORDS


def test_is_blocked_emotion_stopword_matches_surface_and_form() -> None:
    assert is_blocked_emotion_stopword("함께")
    assert is_blocked_emotion_stopword("1부리")
    assert is_blocked_emotion_stopword("잘못된", "잘못")
    assert is_blocked_emotion_stopword("트럼프", "트럼프")
    assert is_blocked_emotion_stopword("지우", "지우")
    assert is_blocked_emotion_stopword("궁금하다", "궁금하다")
    assert is_blocked_emotion_stopword("정한", "정한")
    assert is_blocked_emotion_stopword("똑같", "똑같")
    assert is_blocked_emotion_stopword("부르", "부르")
    assert is_blocked_emotion_stopword("죄송", "죄송")
    assert is_blocked_emotion_stopword("미치", "미치")
    assert is_blocked_emotion_stopword("위대", "위대")
    assert is_blocked_emotion_stopword("내세우", "내세우")
    assert is_blocked_emotion_stopword("꺼내", "꺼내")
    assert is_blocked_emotion_stopword("물어보", "물어보")
    assert is_blocked_emotion_stopword("친구", "친구")
    assert is_blocked_emotion_stopword("이야기", "이야기")
    assert is_blocked_emotion_stopword("추진", "추진")


def test_is_blocked_emotion_stopword_keeps_clear_emotion_words() -> None:
    assert not is_blocked_emotion_stopword("분노")
    assert not is_blocked_emotion_stopword("불안")
