import logging

from analysis_bc.schemas import (
    AnalyzeRequestDto,
    BiasAnalysisKeywordDto,
    BiasAnalysisResultDto,
    BiasEvidenceDto,
    SentenceBiasLabelDto,
    SentenceInputDto,
)

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self) -> None:
        pass

    def analyze(self, request: AnalyzeRequestDto) -> BiasAnalysisResultDto:
        sentences = self.prepare_sentences(request.sentences)

        # TODO(B): classifier 호출 → sentence_labels 생성
        sentence_labels: list[SentenceBiasLabelDto] = []

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
            summary_text="",
            perspective_summary="",
            evidence_summary="",
            tone_label="",
            keywords=keywords,
            sentence_labels=sentence_labels,
            evidences=evidences,
        )

    def prepare_sentences(self, sentences: list[SentenceInputDto]) -> list[SentenceInputDto]:
        # TODO(B): preprocessor 연결 예정
        return sorted(sentences, key=lambda s: s.sentence_order)
