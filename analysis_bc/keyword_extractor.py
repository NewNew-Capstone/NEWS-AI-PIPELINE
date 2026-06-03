from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from kiwipiepy import Kiwi

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import BiasKeywordType, SentenceLabelType
from analysis_bc.score_utils import clamp_score, normalize_score
from analysis_bc.schemas import (
    BiasAnalysisKeywordDto,
    FocusKeywordDto,
    SentenceInputDto,
    SpanLabelDto,
)
from analysis_bc.tagger.emotion_stopwords import is_blocked_emotion_stopword

logger = logging.getLogger(__name__)

_TOP_N = 5

_FRAME_POS: frozenset[str] = frozenset({"NNG", "NNP", "VV"})
_TOPIC_POS: frozenset[str] = frozenset({"NNG", "NNP"})
_STOPWORDS: frozenset[str] = frozenset(
    {
        "관련",
        "뉴스",
        "영상",
        "오늘",
        "이번",
        "대한",
        "있는",
        "없는",
        "것",
        "수",
        "때",
        "중",
        "등",
        "및",
        "그리고",
        "하지만",
        "그러나",
        "지난",
        "최근",
        "현재",
        "기자",
        "앵커",
        "보도",
    }
)
_FOCUS_NOUN_POS: frozenset[str] = frozenset({"NNG", "NNP"})
_FOCUS_TOP_N = 5
_FOCUS_MIN_OCCURRENCE = 2
_FOCUS_MIN_SENTENCE = 2
_KEYWORD_NOUN_QUOTA = 2
_FOCUS_STOPWORDS: frozenset[str] = _STOPWORDS | frozenset(
    {
        "kind",
        "captions",
        "language",
        "gt",
        "lt",
        "대해",
        "함께",
        "정도",
        "사람",
        "문제",
        "저희",
        "우리",
        "여러분",
        "그것",
        "이것",
        "저것",
        "있는",
        "없는",
        "같은",
        "많은",
        "이런",
        "저런",
        "그런",
        "하나",
        "다시",
        "계속",
        "정말",
        "매우",
        "너무",
        "그리고",
        "하지만",
        "그러나",
        "그런데",
        "때문",
        "통해",
        "경우",
        "상황",
        "무엇",
        "어떻게",
        "있다",
        "없다",
        "하다",
        "되다",
        "보다",
        "말하다",
    }
)


def _is_predicate_tag(tag: str) -> bool:
    return tag == "VV" or tag.startswith("VA")


def _is_derivational_verb_tag(tag: str) -> bool:
    return tag == "XSV"


def _is_verb_tag(tag: str) -> bool:
    return tag.startswith("VV")


def _predicate_keyword_text(form: str) -> str:
    return form if form.endswith("다") else f"{form}다"


def _is_predicate_keyword_text(keyword: str) -> bool:
    return keyword.endswith("다")


@dataclass
class _FocusStats:
    occurrence_count: int = 0
    sentence_ids: set[int] | None = None
    surfaces: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.sentence_ids is None:
            self.sentence_ids = set()
        if self.surfaces is None:
            self.surfaces = Counter()

    @property
    def sentence_count(self) -> int:
        return len(self.sentence_ids or set())


class KeywordExtractor:
    def __init__(self) -> None:
        self.kiwi = Kiwi()

    def extract(
        self,
        sentences: list[SentenceInputDto],
        classified: list[ClassifiedSentenceDto],
        span_labels: list[SpanLabelDto],
    ) -> list[BiasAnalysisKeywordDto]:
        emotion = self._extract_emotion(span_labels)
        fact_sentences = [s for s in classified if s.label == "fact_like"]
        opinion_sentences = [s for s in classified if s.label == "opinion_like"]
        frame = self._extract_morpheme(opinion_sentences, _FRAME_POS, BiasKeywordType.FRAME)
        topic = self._extract_morpheme(fact_sentences, _TOPIC_POS, BiasKeywordType.TOPIC)
        return self._filter(emotion) + self._filter(frame) + self._filter(topic)

    def extract_focus_keywords(
        self,
        classified: list[ClassifiedSentenceDto],
    ) -> list[FocusKeywordDto]:
        target_sentences = classified
        total_count = len(target_sentences)
        if total_count == 0:
            return []

        stats_by_form: dict[str, _FocusStats] = {}
        for sentence in target_sentences:
            seen_forms: set[str] = set()
            for token in self.kiwi.tokenize(sentence.sentence_text):
                form = str(token.form).strip()
                surface = sentence.sentence_text[token.start:token.start + token.len].strip()
                keyword = self._focus_keyword_key(form, surface, str(token.tag))
                if keyword is None:
                    continue

                stats = stats_by_form.setdefault(keyword, _FocusStats())
                stats.occurrence_count += 1
                stats.surfaces[keyword] += 1
                if surface and len(surface) >= 2 and surface != keyword:
                    stats.surfaces[surface] += 1
                seen_forms.add(keyword)

            for keyword in seen_forms:
                stats_by_form[keyword].sentence_ids.add(sentence.content_sentence_id)

        repeated = [
            (keyword, stats)
            for keyword, stats in stats_by_form.items()
            if (
                stats.occurrence_count >= _FOCUS_MIN_OCCURRENCE
                and stats.sentence_count >= _FOCUS_MIN_SENTENCE
            )
        ]
        if not repeated:
            return []

        max_occurrence = max(stats.occurrence_count for _keyword, stats in repeated)
        rows: list[FocusKeywordDto] = []
        for keyword, stats in repeated:
            sentence_ratio = stats.sentence_count / total_count
            occurrence_ratio = stats.occurrence_count / max_occurrence if max_occurrence else 0.0
            score = normalize_score(0.8 * sentence_ratio + 0.2 * occurrence_ratio)
            rows.append(
                FocusKeywordDto(
                    keyword_text=keyword,
                    score=score,
                    occurrence_count=stats.occurrence_count,
                    sentence_count=stats.sentence_count,
                )
            )

        sorted_rows = sorted(
            rows,
            key=lambda row: (
                -row.sentence_count,
                -row.occurrence_count,
                -row.score,
                row.keyword_text,
            ),
        )
        return self._ensure_noun_quota(
            sorted_rows,
            limit=_FOCUS_TOP_N,
            noun_quota=_KEYWORD_NOUN_QUOTA,
        )

    def _extract_emotion(
        self,
        span_labels: list[SpanLabelDto],
    ) -> list[BiasAnalysisKeywordDto]:
        weights: dict[str, float] = {}
        for span in span_labels:
            if span.label_type not in (
                SentenceLabelType.EMOTIONALLY_LOADED,
                SentenceLabelType.EMOTIONALLY_LOADED.value,
            ):
                continue
            word = span.matched_word
            if not word or len(word) < 2:
                continue
            keyword = self._emotion_keyword_key(word)
            if not keyword or len(keyword) < 2:
                continue
            if self._is_stopword(keyword) or is_blocked_emotion_stopword(keyword):
                continue
            weights[keyword] = weights.get(keyword, 0.0) + clamp_score(span.score)
        return self._to_relative_keywords(weights, BiasKeywordType.EMOTION)

    def _emotion_keyword_key(self, matched_word: str) -> str:
        word = matched_word.strip()
        if not word:
            return ""

        try:
            tokens = self.kiwi.tokenize(word)
        except Exception:
            logger.warning("emotion keyword tokenization failed: %s", word, exc_info=True)
            return word

        if not tokens:
            return word

        first = tokens[0]
        form = str(first.form).strip()
        tag = str(first.tag)
        if form and _is_predicate_tag(tag):
            return _predicate_keyword_text(form)
        if form and _is_derivational_verb_tag(tag):
            return _predicate_keyword_text(form)
        if form and tag == "XR" and len(tokens) >= 2:
            second = tokens[1]
            if second.tag == "XSA" and second.form == "하":
                return _predicate_keyword_text(f"{form}하")
        if form and tag == "NNG" and len(tokens) >= 2:
            second = tokens[1]
            if second.tag == "XSV" and second.form == "하":
                return _predicate_keyword_text(f"{form}하")
        return word

    def normalize_emotion_keyword(self, matched_word: str | None) -> str:
        if not matched_word:
            return ""
        return self._emotion_keyword_key(matched_word)

    def extract_emotion_filter_candidates(
        self,
        span_labels: list[SpanLabelDto],
    ) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for span in span_labels:
            if span.label_type not in (
                SentenceLabelType.EMOTIONALLY_LOADED,
                SentenceLabelType.EMOTIONALLY_LOADED.value,
            ):
                continue
            keyword = self.normalize_emotion_keyword(span.matched_word)
            if not keyword or len(keyword) < 2:
                continue
            if self._is_stopword(keyword) or is_blocked_emotion_stopword(keyword):
                continue
            if keyword in seen:
                continue
            seen.add(keyword)
            candidates.append(keyword)
        return candidates

    def _extract_morpheme(
        self,
        sentences: list[ClassifiedSentenceDto],
        valid_pos: frozenset[str],
        keyword_type: BiasKeywordType,
    ) -> list[BiasAnalysisKeywordDto]:
        weights: dict[str, float] = {}
        for sentence in sentences:
            sentence_weight = clamp_score(sentence.confidence)
            for token in self.kiwi.tokenize(sentence.sentence_text):
                keyword = self._morpheme_keyword_key(
                    str(token.form).strip(),
                    str(token.tag),
                    valid_pos,
                )
                if keyword is None:
                    continue
                if self._is_stopword(keyword):
                    continue
                weights[keyword] = weights.get(keyword, 0.0) + sentence_weight
        return self._to_relative_keywords(weights, keyword_type)

    def _morpheme_keyword_key(
        self,
        form: str,
        tag: str,
        valid_pos: frozenset[str],
    ) -> str | None:
        if tag in _FOCUS_NOUN_POS and tag in valid_pos:
            keyword = form
        elif _is_verb_tag(tag) and tag in valid_pos:
            keyword = _predicate_keyword_text(form)
        else:
            return None

        if len(keyword) < 2:
            return None
        return keyword

    def _focus_keyword_key(self, form: str, surface: str, tag: str) -> str | None:
        if tag in _FOCUS_NOUN_POS:
            keyword = form.strip()
        elif _is_verb_tag(tag):
            keyword = _predicate_keyword_text(form.strip())
        else:
            return None

        if len(keyword) < 2:
            return None
        if keyword.lower() in _FOCUS_STOPWORDS:
            return None
        if surface.strip().lower() in _FOCUS_STOPWORDS:
            return None
        return keyword

    def _focus_display_text(self, keyword: str, stats: _FocusStats) -> str:
        if not stats.surfaces:
            return keyword
        return sorted(
            stats.surfaces.items(),
            key=lambda item: (-item[1], item[0] != keyword, item[0]),
        )[0][0]

    def _to_relative_keywords(
        self,
        weights: dict[str, float],
        keyword_type: BiasKeywordType,
    ) -> list[BiasAnalysisKeywordDto]:
        total = sum(weights.values())
        if total <= 0:
            return []
        return [
            BiasAnalysisKeywordDto(
                keyword_text=word,
                keyword_type=keyword_type,
                score=normalize_score(weight / total),
            )
            for word, weight in weights.items()
        ]

    def _filter(
        self,
        keywords: list[BiasAnalysisKeywordDto],
    ) -> list[BiasAnalysisKeywordDto]:
        if not keywords:
            return []
        sorted_list = sorted(keywords, key=lambda k: k.score, reverse=True)
        filtered = [
            keyword
            for keyword in sorted_list
            if not self._is_stopword(keyword.keyword_text)
        ]
        return self._ensure_noun_quota(
            filtered,
            limit=_TOP_N,
            noun_quota=_KEYWORD_NOUN_QUOTA,
        )

    def _ensure_noun_quota(
        self,
        rows: list,
        *,
        limit: int,
        noun_quota: int,
    ) -> list:
        if len(rows) <= limit:
            return rows[:limit]

        selected = rows[:limit]
        selected_keys = {row.keyword_text for row in selected}
        noun_count = sum(
            1 for row in selected
            if not _is_predicate_keyword_text(row.keyword_text)
        )
        if noun_count >= noun_quota:
            return selected

        noun_candidates = [
            row for row in rows
            if (
                row.keyword_text not in selected_keys
                and not _is_predicate_keyword_text(row.keyword_text)
            )
        ]
        if not noun_candidates:
            return selected

        out = selected[:]
        replacement_count = min(noun_quota - noun_count, len(noun_candidates))
        replace_index = len(out) - 1
        for noun in noun_candidates[:replacement_count]:
            while (
                replace_index >= 0
                and not _is_predicate_keyword_text(out[replace_index].keyword_text)
            ):
                replace_index -= 1
            if replace_index < 0:
                break
            out[replace_index] = noun
            replace_index -= 1
        return out

    def _is_stopword(self, keyword: str) -> bool:
        normalized = keyword.strip().lower()
        return normalized in _STOPWORDS
