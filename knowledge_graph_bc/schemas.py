from __future__ import annotations

from pydantic import BaseModel, Field


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
