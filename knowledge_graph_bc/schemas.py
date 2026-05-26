from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class VideoSummary(BaseModel):
    video_id: str
    target_id: int | None = None
    title: str = ""
    description: str | None = None
    thumbnail_url: str | None = None
    channel_name: str | None = None
    published_at: str | None = None
    view_count: float = 0.0
    country_code: str | None = None
    language: str | None = None
    analysis_status: str | None = None
    cluster_types: list[str] = Field(default_factory=list)


class VideoSection(BaseModel):
    country_code: str
    language: str
    label: str
    videos: list[VideoSummary] = Field(default_factory=list)


class ComparisonHomeResponse(BaseModel):
    applied_cluster_type: str
    issue_keywords: list[str] = Field(default_factory=list)
    sections: list[VideoSection] = Field(default_factory=list)


class SearchVideosResponse(BaseModel):
    applied_cluster_type: str
    keyword: str
    sections: list[VideoSection] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    video_id: str
    target_id: int | None = None
    title: str = ""
    thumbnail_url: str | None = None
    channel_name: str | None = None
    country_code: str | None = None
    language: str | None = None
    node_type: str
    analysis_status: str | None = None
    cluster_types: list[str] = Field(default_factory=list)


class GraphEdge(BaseModel):
    source: str
    target: str
    relation_type: str
    weight: float
    similarity_score: float | None = None
    opinion_distance: float | None = None
    keywords: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class CountryPerspective(BaseModel):
    country_code: str
    language: str
    summary: str
    top_keywords: list[str] = Field(default_factory=list)


class ComparisonGraphResponse(BaseModel):
    applied_cluster_type: str
    source_video: VideoSummary
    core_keywords: list[str] = Field(default_factory=list)
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    country_perspectives: list[CountryPerspective] = Field(default_factory=list)


class MultilingualKeywordExpandRequest(BaseModel):
    keyword_ko: str = Field(min_length=1, max_length=200)
    max_terms_per_language: int = Field(default=4, ge=1, le=10)


class ExpandedKeywords(BaseModel):
    ko: list[str] = Field(default_factory=list)
    en: list[str] = Field(default_factory=list)
    zh: list[str] = Field(default_factory=list)


class MultilingualKeywordExpandResponse(BaseModel):
    requested_keyword: str
    expanded_keywords: ExpandedKeywords


class RealtimeIngestCandidate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    video_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = ""
    country_code: str = Field(min_length=2, max_length=8)
    language: str = Field(min_length=2, max_length=16)
    channel_id: str | None = None
    channel_name: str | None = None
    published_at: str | None = None
    thumbnail_url: str | None = None
    view_count: float = Field(default=0.0, ge=0)

    @field_validator("country_code")
    @classmethod
    def normalize_country_code(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: str) -> str:
        return value.strip().lower()


class RealtimeIngestRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    keyword: str = Field(min_length=1, max_length=200)
    max_per_country: int = Field(default=5, ge=1, le=20)
    candidates: list[RealtimeIngestCandidate] = Field(min_length=1, max_length=120)


class RealtimeIngestJobResponse(BaseModel):
    request_id: str
    keyword: str
    status: str
    queued_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class RealtimeIngestResponse(BaseModel):
    request_id: str
    queued_count: int
    skipped_existing_count: int
    current_results: SearchVideosResponse


class RealtimeClickedVideoIngestRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    keyword: str = Field(min_length=1, max_length=200)
    selected_video: RealtimeIngestCandidate
    related_candidates: list[RealtimeIngestCandidate] = Field(default_factory=list, max_length=120)
    max_per_country: int = Field(default=5, ge=1, le=20)


class RealtimeClickedVideoIngestResponse(BaseModel):
    request_id: str
    selected_video_id: str
    queued_count: int
    skipped_existing_count: int
    current_graph: ComparisonGraphResponse | None = None


class ClickedVideo(BaseModel):
    video_id: str = Field(min_length=1, max_length=100)
    title: str = ""
    description: str | None = None
    country_code: str | None = None
    language: str | None = None
    channel_id: str | None = None
    channel_name: str | None = None
    published_at: str | None = None
    thumbnail_url: str | None = None
    view_count: float | None = None


class ClickedVideoCompareRequest(BaseModel):
    keyword: str = Field(min_length=1, max_length=200)
    max_per_country: int = Field(default=3, ge=1, le=20)
    selected_video: ClickedVideo
    related_candidates: list[dict[str, Any]] = Field(default_factory=list)


class ClickedVideoCompareResponse(BaseModel):
    request_id: str
    selected_video_id: str
    queued_count: int = 0
    skipped_existing_count: int = 0
    current_graph: ComparisonGraphResponse | None = None
