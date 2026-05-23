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

logger = logging.getLogger(__name__)

_TOP_N = 5

_FRAME_POS: frozenset[str] = frozenset({"NNG", "NNP", "VV", "VA"})
_TOPIC_POS: frozenset[str] = frozenset({"NNG", "NNP"})
_FOCUS_POS: frozenset[str] = frozenset({"NNG", "NNP", "VV", "VA", "XR"})
_FOCUS_TOP_N = 5
_FOCUS_MIN_OCCURRENCE = 2
_FOCUS_MIN_SENTENCE = 2
_FOCUS_STOPWORDS: frozenset[str] = frozenset(
    {
        "kind",
        "captions",
        "language",
        "gt",
        "lt",
        "뉴스",
        "영상",
        "오늘",
        "이번",
        "관련",
        "대해",
        "대한",
        "함께",
        "현재",
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
                if token.tag not in _FOCUS_POS:
                    continue

                form = str(token.form).strip()
                surface = sentence.sentence_text[token.start:token.start + token.len].strip()
                keyword = self._focus_keyword_key(form, surface)
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
                    keyword_text=self._focus_display_text(keyword, stats),
                    score=score,
                    occurrence_count=stats.occurrence_count,
                    sentence_count=stats.sentence_count,
                )
            )

        return sorted(
            rows,
            key=lambda row: (
                -row.sentence_count,
                -row.occurrence_count,
                -row.score,
                row.keyword_text,
            ),
        )[:_FOCUS_TOP_N]

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
            weights[word] = weights.get(word, 0.0) + clamp_score(span.score)
        return self._to_relative_keywords(weights, BiasKeywordType.EMOTION)

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
                if token.tag not in valid_pos:
                    continue
                if len(token.form) < 2:
                    continue
                weights[token.form] = weights.get(token.form, 0.0) + sentence_weight
        return self._to_relative_keywords(weights, keyword_type)

    def _focus_keyword_key(self, form: str, surface: str) -> str | None:
        candidates = [form, surface]
        for candidate in candidates:
            keyword = candidate.strip()
            if len(keyword) < 2:
                continue
            if keyword.lower() in _FOCUS_STOPWORDS:
                continue
            return keyword
        return None

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
        return sorted_list[:_TOP_N]
