from pydantic import BaseModel, ConfigDict

from analysis_bc.enums import BiasKeywordType, EvidenceType, SentenceLabelType, TargetType


# 입력

class SentenceInputDto(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    content_sentence_id: int
    sentence_text: str
    sentence_order: int
    start_time_ms: int | None = None
    end_time_ms: int | None = None


class AnalyzeRequestDto(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    target_id: int
    title: str
    target_type: TargetType
    transcript_id: int
    country: str
    language: str
    sentences: list[SentenceInputDto]


# 출력

class SpanLabelDto(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    content_sentence_id: int
    start_offset: int
    end_offset: int
    label_type: SentenceLabelType
    score: float
    matched_word: str | None = None


class SentenceBiasLabelDto(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    content_sentence_id: int
    label_type: SentenceLabelType
    score: float
    highlight_color: str | None = None
    evidence_keyword: str | None = None


class BiasAnalysisKeywordDto(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    keyword_text: str
    keyword_type: BiasKeywordType
    score: float


class BiasEvidenceDto(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    evidence_type: EvidenceType
    title: str
    description: str
    source_text: str
    confidence_score: float


class BiasAnalysisResultDto(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    target_id: int
    overall_bias_score: float
    opinion_score: float
    emotion_score: float
    anonymous_source_score: float
    headline_body_gap_score: float | None = None
    neutrality_score: float | None = None
    summary_text: str
    perspective_summary: str
    evidence_summary: str
    tone_label: str
    keywords: list[BiasAnalysisKeywordDto]
    sentence_labels: list[SpanLabelDto]
    evidences: list[BiasEvidenceDto]
