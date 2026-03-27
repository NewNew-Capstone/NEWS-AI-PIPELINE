from __future__ import annotations

import logging

from kiwipiepy import Kiwi

from analysis_bc.classifier import ClassifiedSentenceDto
from analysis_bc.enums import BiasKeywordType, SentenceLabelType
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

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _extract_emotion(
        self,
        span_labels: list[SpanLabelDto],
    ) -> list[BiasAnalysisKeywordDto]:
        results: list[BiasAnalysisKeywordDto] = []
        for span in span_labels:
            if span.label_type not in (
                SentenceLabelType.EMOTIONALLY_LOADED,
                SentenceLabelType.EMOTIONALLY_LOADED.value,
            ):
                continue
            word = span.matched_word
            if not word:
                continue
            results.append(
                BiasAnalysisKeywordDto(
                    keyword_text=word,
                    keyword_type=BiasKeywordType.EMOTION,
                    score=span.score,
                )
            )
        return results

    def _extract_morpheme(
        self,
        sentences: list[ClassifiedSentenceDto],
        valid_pos: frozenset[str],
        keyword_type: BiasKeywordType,
    ) -> list[BiasAnalysisKeywordDto]:
        results: list[BiasAnalysisKeywordDto] = []
        for sentence in sentences:
            tokens = self.kiwi.tokenize(sentence.sentence_text)
            for token in tokens:
                if token.tag not in valid_pos:
                    continue
                if len(token.form) < 2:
                    continue
                results.append(
                    BiasAnalysisKeywordDto(
                        keyword_text=token.form,
                        keyword_type=keyword_type,
                        score=sentence.confidence,
                    )
                )
        return results

    def _filter(
        self,
        keywords: list[BiasAnalysisKeywordDto],
    ) -> list[BiasAnalysisKeywordDto]:
        if not keywords:
            return []
        # 2글자 미만 제거
        keywords = [k for k in keywords if len(k.keyword_text) >= 2]
        # 중복 제거 (같은 text+type 조합 → score 높은 것 유지)
        deduped: dict[tuple[str, str], BiasAnalysisKeywordDto] = {}
        for k in keywords:
            key = (k.keyword_text, str(k.keyword_type))
            if key not in deduped or k.score > deduped[key].score:
                deduped[key] = k
        # 타입별 top-N
        sorted_list = sorted(deduped.values(), key=lambda k: k.score, reverse=True)
        return sorted_list[:_TOP_N]
