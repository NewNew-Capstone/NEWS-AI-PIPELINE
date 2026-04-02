import logging

from analysis_bc.classifier import FactOpinionClassifier
from analysis_bc.evidence_extractor import EvidenceExtractor
from analysis_bc.keyword_extractor import KeywordExtractor
from analysis_bc.preprocessor import SentencePreprocessor
from analysis_bc.scorer import BiasScorer
from analysis_bc.summarizer import BiasSummarizer
from analysis_bc.schemas import (
    AnalyzeRequestDto,
    BiasAnalysisResultDto,
    SentenceInputDto,
)
from analysis_bc.tagger.span_tagger import SpanTagger
from analysis_bc.tagger.title_body_gap import TitleBodyGapCalculator

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self) -> None:
        self.classifier = FactOpinionClassifier(
            model_path="analysis_bc/models/best_model"
        )
        self.span_tagger = SpanTagger()
        self.title_body_gap_calculator = TitleBodyGapCalculator()
        self.scorer = BiasScorer()
        self.summarizer = BiasSummarizer()
        self.keyword_extractor = KeywordExtractor()
        self.evidence_extractor = EvidenceExtractor()

    def analyze(self, request: AnalyzeRequestDto) -> BiasAnalysisResultDto:
        # 전처리
        sentences = self.prepare_sentences(request.sentences, request.language)
        print(f"[서비스] 전처리 후 문장 수: {len(sentences)} / 입력: {len(request.sentences)}")
        if not sentences:
            print("[서비스] ⚠ 전처리 후 문장이 0개 — 언어 감지 필터에서 모두 제거됨")

        # classifier 호출 → fact / opinion 분리
        classified = self.classifier.classify(sentences)
        fact_sentences = [s for s in classified if s.label == "fact_like"]
        opinion_sentences = [s for s in classified if s.label == "opinion_like"]

        print(f"[서비스] 분류 결과 — fact: {len(fact_sentences)}, opinion: {len(opinion_sentences)}")

        logger.debug(
            "classify: fact=%d, opinion=%d",
            len(fact_sentences),
            len(opinion_sentences),
        )

        # span 태깅 (opinion_sentences 대상 — emotion vs anonymous 유사도 비교 후 단일 태그)
        sentence_labels = self.span_tagger.tag(opinion_sentences)

        # 제목-본문 갭
        title_body_gap = self.title_body_gap_calculator.calculate(
            title=request.title,
            sentences=sentences,
        )

        scores = self.scorer.calculate(
            classified=classified,
            span_labels=sentence_labels,
            headline_body_gap=title_body_gap,
        )

        # 4단계: FACT 문장 상위 필터링
        FACT_TOP_N = 10
        top_facts = sorted(
            fact_sentences,
            key=lambda s: s.confidence,
            reverse=True,
        )[:FACT_TOP_N]

        # 5단계: 요약 생성 (Claude API)
        summary = self.summarizer.summarize(
            fact_sentences=top_facts,
            opinion_sentences=opinion_sentences,
            span_labels=sentence_labels,
            title=request.title,
            language=request.language,
        )

        keywords = self.keyword_extractor.extract(
            sentences=sentences,
            classified=classified,
            span_labels=sentence_labels,
        )

        evidences = self.evidence_extractor.extract(
            sentences=sentences,
            classified=classified,
            span_labels=sentence_labels,
        )

        logger.debug("analyze: target_id=%d, sentences=%d", request.target_id, len(sentences))

        return BiasAnalysisResultDto(
            target_id=request.target_id,
            overall_bias_score=scores["overall_bias_score"],
            opinion_score=scores["opinion_score"],
            emotion_score=scores["emotion_score"],
            anonymous_source_score=scores["anonymous_source_score"],
            headline_body_gap_score=title_body_gap,
            subjectivity_score=scores["subjectivity_score"],
            score_evidence=scores["score_evidence"],
            bias_type_scores=scores["bias_type_scores"],
            summary_text=summary["summary_text"],
            perspective_summary=summary["perspective_summary"],
            evidence_summary=summary["evidence_summary"],
            tone_label=summary["tone_label"],
            keywords=keywords,
            sentence_labels=sentence_labels,
            evidences=evidences,
        )

    def prepare_sentences(
        self, sentences: list[SentenceInputDto], language: str
    ) -> list[SentenceInputDto]:
        return SentencePreprocessor(expected_language=language).preprocess(sentences)
