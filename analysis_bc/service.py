import logging

from analysis_bc.classifier import FactOpinionClassifier
from analysis_bc.preprocessor import SentencePreprocessor
from analysis_bc.schemas import (
    AnalyzeRequestDto,
    BiasAnalysisKeywordDto,
    BiasAnalysisResultDto,
    BiasEvidenceDto,
    SentenceInputDto,
    SpanLabelDto,
)
from analysis_bc.tagger.anonymous_tagger import AnonymousTagger
from analysis_bc.tagger.emotional_tagger import EmotionalTagger
from analysis_bc.tagger.title_body_gap import TitleBodyGapCalculator
from analysis_bc.utils import resolve_overlapping_spans

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self) -> None:
        self.classifier = FactOpinionClassifier(
            model_path="analysis_bc/models/best_model"
        )
        self.emotional_tagger = EmotionalTagger()
        self.anonymous_tagger = AnonymousTagger()
        self.title_body_gap_calculator = TitleBodyGapCalculator()

    def analyze(self, request: AnalyzeRequestDto) -> BiasAnalysisResultDto:
        # 전처리
        sentences = self.prepare_sentences(request.sentences, request.language)

        # classifier 호출 → fact / opinion 분리
        classified = self.classifier.classify(sentences)
        fact_sentences = [s for s in classified if s.label == "fact_like"]
        opinion_sentences = [s for s in classified if s.label == "opinion_like"]

        logger.debug(
            "classify: fact=%d, opinion=%d",
            len(fact_sentences),
            len(opinion_sentences),
        )

        # 감정성 태깅 (opinion_sentences 대상)
        emotion_spans = self.emotional_tagger.tag(opinion_sentences)

        # 익명성 태깅 (전체 문장 대상)
        anonymous_spans = self.anonymous_tagger.tag(sentences)

        # span 합치기 (중복 offset 우선순위 기반 제거)
        all_spans = emotion_spans + anonymous_spans
        sentence_labels = resolve_overlapping_spans(all_spans)

        # 제목-본문 갭
        title_body_gap = self.title_body_gap_calculator.calculate(
            title=request.title,
            sentences=sentences,
        )

        # TODO(B): scorer 호출 → overall_bias_score 등 점수 계산
        overall_bias_score = 0.0
        opinion_score = 0.0
        emotion_score = 0.0
        anonymous_source_score = 0.0

        # TODO(B): keyword_extractor 호출 → keywords 생성
        keywords: list[BiasAnalysisKeywordDto] = []

        # TODO(B): evidence_extractor 호출 → evidences 생성
        evidences: list[BiasEvidenceDto] = []

        logger.debug("analyze: target_id=%d, sentences=%d", request.target_id, len(sentences))

        return BiasAnalysisResultDto(
            target_id=request.target_id,
            overall_bias_score=overall_bias_score,
            opinion_score=opinion_score,
            emotion_score=emotion_score,
            anonymous_source_score=anonymous_source_score,
            headline_body_gap_score=title_body_gap,
            summary_text="",
            perspective_summary="",
            evidence_summary="",
            tone_label="",
            keywords=keywords,
            sentence_labels=sentence_labels,
            evidences=evidences,
        )

    def prepare_sentences(
        self, sentences: list[SentenceInputDto], language: str
    ) -> list[SentenceInputDto]:
        return SentencePreprocessor(expected_language=language).preprocess(sentences)
