from __future__ import annotations

from unittest.mock import MagicMock, patch

import knowledge_graph_bc.router as router_mod
from knowledge_graph_bc.schemas import (
    ComparisonGraphResponse,
    ComparisonHomeResponse,
    GraphNode,
    MultilingualKeywordExpandRequest,
    SearchVideosResponse,
    VideoSection,
    VideoSummary,
)


def test_comparison_home_endpoint_returns_service_response() -> None:
    response = ComparisonHomeResponse(
        applied_cluster_type="CURATION_MANUAL",
        issue_keywords=["반도체"],
        sections=[VideoSection(country_code="KR", language="ko", label="한국", videos=[])],
    )
    service = MagicMock()
    service.get_comparison_home.return_value = response

    with patch("knowledge_graph_bc.router.KnowledgeGraphComparisonService", return_value=service):
        result = router_mod.comparison_home(limit=5)

    assert result == response
    service.get_comparison_home.assert_called_once_with(limit=5)


def test_search_videos_endpoint_returns_service_response() -> None:
    response = SearchVideosResponse(applied_cluster_type="ALL_AVAILABLE", keyword="반도체", sections=[])
    service = MagicMock()
    service.search_videos.return_value = response

    with patch("knowledge_graph_bc.router.KnowledgeGraphComparisonService", return_value=service):
        result = router_mod.search_videos(keyword="반도체", limit=5, scope="all")

    assert result == response
    service.search_videos.assert_called_once_with(keyword="반도체", limit=5, scope="all")


def test_comparison_graph_endpoint_returns_service_response() -> None:
    source = VideoSummary(video_id="kr001", country_code="KR", language="ko")
    response = ComparisonGraphResponse(
        applied_cluster_type="CURATION_MANUAL",
        source_video=source,
        core_keywords=["반도체"],
        nodes=[GraphNode(id="video:kr001", video_id="kr001", node_type="source")],
        edges=[],
        country_perspectives=[],
    )
    service = MagicMock()
    service.get_comparison_graph.return_value = response

    with patch("knowledge_graph_bc.router.KnowledgeGraphComparisonService", return_value=service):
        result = router_mod.comparison_graph(video_id="kr001", limit_per_country=5, scope="all")

    assert result == response
    service.get_comparison_graph.assert_called_once_with(
        video_id="kr001",
        limit_per_country=5,
        scope="all",
    )


def test_expand_multilingual_keywords_endpoint_returns_expanded_terms() -> None:
    payload = MultilingualKeywordExpandRequest(keyword_ko="트럼프 대만", max_terms_per_language=3)
    expander = MagicMock()
    expander.expand.return_value.requested_keyword = "트럼프 대만"
    expander.expand.return_value.ko = ["트럼프 대만", "미중 정상회담"]
    expander.expand.return_value.en = ["trump taiwan", "us china summit"]
    expander.expand.return_value.zh = ["特朗普 台湾", "中美 峰会"]

    with patch("knowledge_graph_bc.router.MultilingualKeywordExpander", return_value=expander):
        result = router_mod.expand_multilingual_keywords(payload)

    assert result.requested_keyword == "트럼프 대만"
    assert result.expanded_keywords.en[0] == "trump taiwan"
    assert result.expanded_keywords.zh[0] == "特朗普 台湾"
    expander.expand.assert_called_once_with("트럼프 대만", max_terms_per_language=3)
