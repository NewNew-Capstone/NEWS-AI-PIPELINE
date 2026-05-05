from __future__ import annotations

import logging

from kiwipiepy import Kiwi

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import BiasKeywordType, SentenceLabelType
from analysis_bc.score_utils import clamp_score, normalize_score
from analysis_bc.schemas import BiasAnalysisKeywordDto, SentenceInputDto, SpanLabelDto

logger = logging.getLogger(__name__)

_TOP_N = 5

_FRAME_POS: frozenset[str] = frozenset({"NNG", "NNP", "VV", "VA"})
_TOPIC_POS: frozenset[str] = frozenset({"NNG", "NNP"})


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
