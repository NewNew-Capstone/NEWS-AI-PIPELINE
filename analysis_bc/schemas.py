from pydantic import BaseModel, ConfigDict, Field

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
    ## optional로 변경
    target_type: TargetType | None = None
    transcript_id: int | None = None
    country: str | None = None
    language: str
    sentences: list[SentenceInputDto]


class AnalyzeRawTextRequestDto(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    target_id: int
    title: str
    language: str
    raw_text: str
    target_type: TargetType | None = None
    transcript_id: int | None = None
    country: str | None = None
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


class SentenceResultDto(BaseModel):
    content_sentence_id: int
    sentence_text: str
    sentence_order: int


class BiasAnalysisResultDto(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    target_id: int
    transcript_id: int | None = None
    overall_bias_score: float
    opinion_score: float
    emotion_score: float
    anonymous_source_score: float = 0.0
    headline_body_gap_score: float | None = None
    neutrality_score: float | None = None
    summary_text: str
    perspective_summary: str
    evidence_summary: str
    tone_label: str
    subjectivity_score: float = 0.0
    score_evidence: str = ""
    bias_type_scores: dict[str, float] = Field(default_factory=dict)
    keywords: list[BiasAnalysisKeywordDto]
    sentence_labels: list[SpanLabelDto]
    evidences: list[BiasEvidenceDto]


class RawAnalysisResultDto(BiasAnalysisResultDto):
    sentences: list[SentenceResultDto]
