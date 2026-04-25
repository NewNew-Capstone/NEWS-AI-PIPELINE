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
        print("[서비스] SpanTagger 시작")
        sentence_labels = self.span_tagger.tag(opinion_sentences)
        print(f"[서비스] SpanTagger 완료 — 라벨 수: {len(sentence_labels)}")

        # 제목-본문 갭
        print("[서비스] TitleBodyGap 계산 시작")
        title_body_gap = self.title_body_gap_calculator.calculate(
            title=request.title,
            sentences=sentences,
        )
        print(f"[서비스] TitleBodyGap 완료 — gap: {title_body_gap:.4f}")

        print("[서비스] Scorer 시작")
        scores = self.scorer.calculate(
            classified=classified,
            span_labels=sentence_labels,
            headline_body_gap=title_body_gap,
        )
        print(f"[서비스] Scorer 완료 — overall: {scores['overall_bias_score']:.4f}")

        # FACT 문장 상위 필터링
        FACT_TOP_N = 10
        top_facts = sorted(
            fact_sentences,
            key=lambda s: s.confidence,
            reverse=True,
        )[:FACT_TOP_N]
        print(f"[서비스] FACT 상위 필터링 완료 — {len(top_facts)}개")

        # 요약 생성 (Claude API)
        print("[서비스] Summarizer 시작")
        summary = self.summarizer.summarize(
            fact_sentences=top_facts,
            opinion_sentences=opinion_sentences,
            span_labels=sentence_labels,
            title=request.title,
            language=request.language,
        )
        print("[서비스] Summarizer 완료")

        print("[서비스] KeywordExtractor 시작")
        keywords = self.keyword_extractor.extract(
            sentences=sentences,
            classified=classified,
            span_labels=sentence_labels,
        )
        print(f"[서비스] KeywordExtractor 완료 — 키워드 수: {len(keywords)}")

        print("[서비스] EvidenceExtractor 시작")
        evidences = self.evidence_extractor.extract(
            sentences=sentences,
            classified=classified,
            span_labels=sentence_labels,
        )
        print(f"[서비스] EvidenceExtractor 완료 — 근거 수: {len(evidences)}")

        logger.debug(
            "analyze: target_id=%d, transcript_id=%s, sentences=%d",
            request.target_id,
            request.transcript_id,
            len(sentences),
        )

        print(
            f"[서비스] 최종 응답\n"
            f"  - overall_bias_score  : {scores['overall_bias_score']:.4f}\n"
            f"  - opinion_score       : {scores['opinion_score']:.4f}\n"
            f"  - emotion_score       : {scores['emotion_score']:.4f}\n"
            f"  - headline_body_gap   : {title_body_gap:.4f}\n"
            f"  - subjectivity_score  : {scores['subjectivity_score']:.4f}\n"
            f"  - tone_label          : {summary['tone_label']}\n"
            f"  - keywords            : {len(keywords)}개\n"
            f"  - sentence_labels     : {len(sentence_labels)}개\n"
            f"  - evidences           : {len(evidences)}개\n"
            f"  - summary_text        : {summary['summary_text'][:50]}...\n"
            f"  - perspective_summary : {summary['perspective_summary'][:50]}...\n"
            f"  - evidence_summary    : {summary['evidence_summary'][:50]}..."
        )

        return BiasAnalysisResultDto(
            target_id=request.target_id,
            transcript_id=request.transcript_id,
            overall_bias_score=scores["overall_bias_score"],
            opinion_score=scores["opinion_score"],
            emotion_score=scores["emotion_score"],
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
